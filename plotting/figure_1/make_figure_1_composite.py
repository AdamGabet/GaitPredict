"""
Figure 1 composite, assembled as a single vector SVG.

Layout comes from assets/figure_1_layout.svg - the draw.io export with the
embedded mxfile source and draw.io's dark-mode CSS variables stripped, and with
the four rasters that have vector sources replaced by build-time marker slots
(data-vector-slot). Everything else in that file is kept byte-for-byte - the
panel frame, the three reconstruction outlines, the anatomy graphic, the
extracted pose grids and their inset clip-paths - so the composite matches the
draw.io figure. The participant photographs and pose-extraction arrows are
removed at build time, and the pose grids are enlarged into the freed space.

The four slots are filled with true vector content:

  TILES_MAIN       biomarker tiles, re-rendered from gait_biomarker_infographic
                   with the Medical Conditions / Medications row appended. In the
                   draw.io file those ten rows came from two overlapping PNG
                   exports of different vintages, each clipped to a different
                   band; one ten-row render replaces both.
  TILES_EXTRA_ROW  dropped, absorbed into the ten-row render above
  LOSS_PLOT        training-loss curve from loss_plot.py
  MAE_SCHEMATIC    assets/mae_schematic.svg, converted from page 1 of the
                   GaitMAE / downstream slide PDF with its text kept as text

Every piece of text in the output is a real SVG <text> element: matplotlib writes
with svg.fonttype='none', the schematic keeps the text from its PDF, and the four
panel headings - which draw.io emits as foreignObject HTML with a rasterised
<image> fallback - are re-emitted as <text> anchored on the ink positions
measured off those fallbacks.

Each vector source is placed by registering its ink bounding box onto the ink
bounding box the raster it replaces occupied (SLOT_TARGETS), so nothing shifts.

Building needs only Python (cairosvg measures ink boxes and writes the PDF/PNG
copies; Inkscape is used instead when it happens to be installed).

Output: output/figure_1_composite.svg + .pdf + .png
"""
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, 'assets')
LAYOUT = os.path.join(ASSETS, 'figure_1_layout.svg')
SCHEMATIC = os.path.join(ASSETS, 'mae_schematic.svg')
OUTPUT_DIR = os.path.join(HERE, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUT_STEM = os.path.join(OUTPUT_DIR, 'figure_1_composite')

INKSCAPE = '/Applications/Inkscape.app/Contents/MacOS/inkscape'

# The tiles and the loss curve were rasterised from matplotlib's default DejaVu
# Sans, which is not a system font, so keeping their labels as editable <text>
# means naming a font that renders: Helvetica. Glyph shapes then differ slightly
# from the draw.io figure while every position stays put. Set this to True to
# convert those labels to outlines instead - glyph-for-glyph identical to the
# original, no longer selectable or editable.
TEXT_AS_OUTLINES = False

# Ink bounding box, in composite user units, that each replaced raster covered in
# the draw.io export. Measured from the embedded PNGs and their inset clips.
SLOT_TARGETS = {
    'TILES_MAIN':    (116.000,  48.000, 232.000, 336.000),
    'LOSS_PLOT':     (467.508, 463.448, 349.243, 188.344),
    'MAE_SCHEMATIC': (125.094, 469.996, 333.991, 176.207),
}

# Geometry of the two photographs removed from the draw.io export.
PHOTO_TOP = (449.9, 21.7, 113.33, 189.24)
PHOTO_BOTTOM = (434.8333333333334, 203.5523073684211,
                150.66666666666666, 221.85263157894738)

# The four path elements of the two original horizontal arrows (shaft + head
# each), matched on their exact `d` prefix so a layout edit cannot silently
# match the wrong path.
ARROW_PATH_PREFIXES = (
    'M 563.23 116.27', 'M 591.78 116.31',
    'M 562.9 316.27', 'M 601.92 316.32',
)

# Original and revised geometry for the anatomy graphic and both three-pose
# grids. The pose rasters are kept separate because their opaque white margins
# otherwise cover the head at the top of the lower sequence.
TOP_PANEL_IMAGE_LAYOUTS = (
    ((336.16, 117.7, 110.35, 200.0), (360.0, 105.0, 123.59, 224.0)),
    ((592.9, 17.57, 210.3, 197.49), (537.0, 25.0, 210.3, 197.49)),
    ((603.04, 213.82, 200.16, 205.0), (542.0, 222.5, 200.16, 205.0)),
)


def remove_photos_and_arrows(doc):
    """Remove the two participant photographs and pose-extraction arrows."""
    spans = []
    for m in re.finditer(r'<image\b[^>]*?>', doc):
        t = m.group(0)
        if 'data-vector-slot' in t:
            continue
        vals = [re.search(r'\b%s="([-\d.]+)"' % a, t)
                for a in ('x', 'y', 'width', 'height')]
        if not all(vals):
            continue
        xywh = tuple(float(v.group(1)) for v in vals)
        for ref in (PHOTO_TOP, PHOTO_BOTTOM):
            if all(abs(a - b) < 0.01 for a, b in zip(xywh, ref)):
                spans.append(m.span())
    if len(spans) != 2:
        sys.exit("expected the 2 participant photographs in the layout, "
                 "matched %d" % len(spans))

    arrows = []
    for m in re.finditer(r'<path\b[^>]*?/>', doc):
        d = re.search(r'\bd="([^"]+)"', m.group(0))
        if d and any(d.group(1).startswith(p) for p in ARROW_PATH_PREFIXES):
            arrows.append(m.span())
    if len(arrows) != len(ARROW_PATH_PREFIXES):
        sys.exit("expected %d pose-arrow paths, matched %d"
                 % (len(ARROW_PATH_PREFIXES), len(arrows)))

    edits = ([(s, e, '') for s, e in spans] +
             [(s, e, '') for s, e in arrows])
    for s, e, rep in sorted(edits, key=lambda t: -t[0]):
        doc = doc[:s] + rep + doc[e:]
    return doc


def reposition_top_panel_images(doc):
    """Enlarge the anatomy graphic and place the pose grids without overlap."""
    matched = 0
    for original, target in TOP_PANEL_IMAGE_LAYOUTS:
        found = []
        for m in re.finditer(r'<image\b[^>]*?>', doc):
            tag = m.group(0)
            vals = [re.search(r'\b%s="([-\d.]+)"' % a, tag)
                    for a in ('x', 'y', 'width', 'height')]
            if not all(vals):
                continue
            xywh = tuple(float(v.group(1)) for v in vals)
            if all(abs(a - b) < 0.01 for a, b in zip(xywh, original)):
                found.append(m)
        if len(found) != 1:
            sys.exit("expected one top-panel image at %s, matched %d"
                     % (original, len(found)))

        m = found[0]
        tag = m.group(0)
        for attr, value in zip(('x', 'y', 'width', 'height'), target):
            tag = re.sub(r'\b%s="[-\d.]+"' % attr,
                         '%s="%.2f"' % (attr, value), tag, count=1)
        doc = doc[:m.start()] + tag + doc[m.end():]
        matched += 1
    if matched != len(TOP_PANEL_IMAGE_LAYOUTS):
        sys.exit("not all top-panel images were repositioned")
    return doc


# The tenth tile row. Grey in the figure because neither key is in SYSTEM_COLOR_MAP.
EXTRA_TILE_ROW = [[
    ('Medical Conditions', 'medical_conditions', ['Diabetes', 'Hypertension']),
    ('Medications', 'medications', ['Statins', 'ACE inhibitors']),
]]

# Panel headings, in draw.io's document order. x is the left edge of the ink (or
# its centre for the two-line label), baselines are absolute; both were measured
# off the rasterised fallbacks, which is steadier than re-deriving them from
# draw.io's flexbox centring. Originally Helvetica 14/12 px, matching the
# foreignObject CSS; both dropped to 11 px so the panel letters print at 6.7 pt
# instead of 8.5 pt, inside Nature's 5-7 pt window at the 180 mm width.
HEADINGS = [
    dict(lines=['a  Multimodal Phenotyping and Data Acquisition'],
         x=117.66, baselines=[29.41], size=11, bold=True, anchor='start'),
    dict(lines=['b MAE Model and Downstream Predictions'],
         x=122.34, baselines=[449.41], size=11, bold=True, anchor='start'),
    dict(lines=['c Model Training and Reconstruction'],
         x=517.38, baselines=[449.41], size=11, bold=True, anchor='start'),
    dict(lines=['Pose Estimation'],
         x=641.00, baselines=[29.41], size=11, bold=True, anchor='middle'),
]


def svg_geometry(path):
    """(inner_markup, root_tag, [x, y, w, h]) for an SVG file."""
    s = open(path, encoding='utf-8').read()
    s = re.sub(r'<\?xml[^>]*\?>', '', s)
    s = re.sub(r'<!DOCTYPE[^>]*>', '', s)
    m = re.search(r'<svg\b[^>]*>', s)
    root = m.group(0)
    inner = s[m.end():s.rindex('</svg>')]
    vb = re.search(r'viewBox="([-\d.eE\s]+)"', root)
    if vb:
        box = [float(v) for v in vb.group(1).split()]
    else:
        num = lambda a: float(re.search(rf'{a}="([\d.]+)', root).group(1))
        box = [0.0, 0.0, num('width'), num('height')]
    return inner, root, box


def ink_box(path, box, px_width=1600):
    """Ink bounding box of an SVG, in its own user units.

    Rendered with cairosvg rather than resvg: these SVGs reference fonts by name
    (svg.fonttype='none'), and resvg ships no font database, so on a machine
    where it cannot resolve the family it renders the graphics and silently drops
    every glyph. The ink box then excludes all text, and the clip rect built from
    it shears off any label that overhangs the artwork - axis tick labels, the
    right-hand ends of the schematic's body-system list. cairosvg resolves system
    fonts, so the box covers the text too.
    """
    import numpy as np
    import cairosvg
    from PIL import Image
    png = cairosvg.svg2png(url=path, output_width=px_width, background_color='#ffffff')
    a = np.asarray(Image.open(io.BytesIO(png)).convert('RGB')).astype(int)
    nz = a.sum(axis=2) < 735
    cols = np.where(nz.any(axis=0))[0]
    rows = np.where(nz.any(axis=1))[0]
    H, W = nz.shape
    x0, y0, w, h = box
    return (x0 + cols.min() / W * w,
            y0 + rows.min() / H * h,
            (cols.max() + 1 - cols.min()) / W * w,
            (rows.max() + 1 - rows.min()) / H * h)


def nested(path, target, clip_id):
    """Nest an SVG so its ink box lands exactly on `target` (x, y, w, h)."""
    inner, root, box = svg_geometry(path)
    ix, iy, iw, ih = ink_box(path, box)
    tx, ty, tw, th = target
    tag = root
    for attr in ('width', 'height', 'x', 'y', 'viewBox', 'preserveAspectRatio'):
        tag = re.sub(rf'\s{attr}="[^"]*"', '', tag)
    tag = tag[:-1] + (f' x="{tx:.4f}" y="{ty:.4f}" width="{tw:.4f}" height="{th:.4f}"'
                      f' viewBox="{ix:.4f} {iy:.4f} {iw:.4f} {ih:.4f}"'
                      f' preserveAspectRatio="none">')
    # Clip to the ink box: a nested viewport does not clip foreignObject content,
    # and a PDF-derived SVG can carry a page-sized background rect.
    clip = (f'<defs><clipPath id="{clip_id}"><rect x="{ix:.4f}" y="{iy:.4f}" '
            f'width="{iw:.4f}" height="{ih:.4f}"/></clipPath></defs>')
    return f'{tag}{clip}<g clip-path="url(#{clip_id})">{inner}</g></svg>'


def schematic_without_loss_symbol(build_dir):
    """The MAE schematic with the raster L_MPJPE label removed.

    The symbol sits directly above the "Reconstruction Loss" caption, which says
    the same thing in words, and it is the one embedded raster left inside an
    otherwise vector panel. It is a single <g> wrapping one base64 PNG, so it can
    be dropped wholesale. Removing it leaves the schematic's ink bounding box
    unchanged (verified: 88.8, 20.4, 859.8, 452.4 either way), so nested() still
    lands the panel on exactly the same target rect.
    """
    s = open(SCHEMATIC, encoding='utf-8').read()
    j = s.find('id="image53"')
    if j < 0:
        sys.exit("the L_MPJPE label (image53) is no longer in %s -- if the "
                 "schematic was re-exported, re-identify it before building"
                 % os.path.basename(SCHEMATIC))
    start = s.rfind('<g', 0, j)
    end = s.find('</g>', j) + len('</g>')
    block = s[start:end]
    if block.count('<image') != 1 or block.count('<g') != 1:
        sys.exit("the group holding the L_MPJPE label now wraps more than that "
                 "one image; removing it would take other artwork with it")
    out = os.path.join(build_dir, 'mae_schematic_no_loss_symbol.svg')
    with open(out, 'w', encoding='utf-8') as fh:
        fh.write(s[:start] + s[end:])
    return out


def build_vector_sources(build_dir):
    """Render the tiles and the loss curve as vector; return the three sources.

    Both figure_1 scripts save `save_path` and `save_path.replace('.png', '.pdf')`,
    so handing them an .svg path writes SVG (twice, harmlessly). Going through
    matplotlib's own SVG writer keeps the text as text.
    """
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(HERE, '..'))
    import matplotlib.pyplot as plt
    import gait_biomarker_infographic as info
    import loss_plot
    from publication_colors import SYSTEM_COLOR_MAP

    # The draw.io figure's tenth row is a mid grey, because the older export it
    # was cropped from resolved these two unmapped keys to a 0.5 grey rather than
    # to today's 0.8 fallback. Register them for this build only so the row keeps
    # the shade it has in the figure; drop these two lines for the lighter one.
    SYSTEM_COLOR_MAP.setdefault('medical_conditions', (0.5, 0.5, 0.5, 1.0))
    SYSTEM_COLOR_MAP.setdefault('medications', (0.5, 0.5, 0.5, 1.0))

    # Real <text> rather than outlines, in a font that actually resolves: with
    # svg.fonttype='none' the SVG only names a family, and matplotlib's default
    # DejaVu Sans is not a system font on macOS, so the labels would silently
    # fall back to whatever the renderer picks and shift against the tile boxes.
    if TEXT_AS_OUTLINES:
        plt.rcParams['svg.fonttype'] = 'path'
    else:
        plt.rcParams['svg.fonttype'] = 'none'
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

    tiles_svg = os.path.join(build_dir, 'tiles.svg')
    info.create_gait_infographic(save_path=tiles_svg, extra_systems=EXTRA_TILE_ROW)

    loss_svg = os.path.join(build_dir, 'loss.svg')
    loss_plot.SAVE_PATH = loss_svg             # leaves output/plot_loss.* alone
    loss_plot.main()

    return {'TILES_MAIN': tiles_svg, 'LOSS_PLOT': loss_svg,
            'MAE_SCHEMATIC': schematic_without_loss_symbol(build_dir)}


def heading_markup():
    out = []
    for h in HEADINGS:
        weight = ' font-weight="bold"' if h['bold'] else ''
        for line, baseline in zip(h['lines'], h['baselines']):
            out.append(
                f'<text xml:space="preserve" x="{h["x"]:.2f}" y="{baseline:.2f}" '
                f'font-family="Helvetica, Arial, sans-serif" font-size="{h["size"]}"'
                f'{weight} fill="#000000" text-anchor="{h["anchor"]}">{line}</text>')
    return ''.join(out)


def main():
    missing = [p for p in (LAYOUT, SCHEMATIC) if not os.path.exists(p)]
    if missing:
        sys.exit("Missing Figure 1 assets:\n  " + "\n  ".join(missing) +
                 "\n\nBoth are hand-prepared inputs (see assets/README.md) and have to be"
                 "\ncopied in before this composite can be built.")

    with tempfile.TemporaryDirectory() as build:
        sources = build_vector_sources(build)
        doc = open(LAYOUT, encoding='utf-8').read()

        # Drop draw.io's four <switch> label blocks (foreignObject + raster
        # fallback) and re-emit them as text.
        doc, n = re.subn(r'<switch>.*?</switch>', '', doc, flags=re.S)
        if n != len(HEADINGS):
            sys.exit(f"expected {len(HEADINGS)} label switches in the layout, found {n}")
        doc = doc.replace('</svg>', heading_markup() + '</svg>')
        doc = remove_photos_and_arrows(doc)
        doc = reposition_top_panel_images(doc)

        for slot, target in SLOT_TARGETS.items():
            pat = re.compile(r'<image\b[^>]*?data-vector-slot="' + slot + r'"[^>]*?/?>')
            if not pat.search(doc):
                sys.exit(f"slot {slot} not found in {LAYOUT}")
            doc = pat.sub(lambda _m, s=slot, t=target: nested(sources[s], t, 'clip_' + s.lower()),
                          doc, count=1)
        doc = re.sub(r'<image\b[^>]*?data-vector-slot="TILES_EXTRA_ROW"[^>]*?/?>', '', doc)

        out_svg = OUT_STEM + '.svg'
        with open(out_svg, 'w', encoding='utf-8') as fh:
            fh.write(doc)
        print(f"Saved: {out_svg}  ({len(doc) // 1024} KB)")

        exe = INKSCAPE if os.path.exists(INKSCAPE) else shutil.which('inkscape')
        if exe:
            for ext, extra in (('pdf', []), ('png', ['--export-width=3000'])):
                subprocess.run([exe, '--export-type=' + ext,
                                '--export-filename=' + OUT_STEM + '.' + ext,
                                '--export-background=white', '--export-background-opacity=1',
                                *extra, out_svg], check=True, capture_output=True)
                print(f"Saved: {OUT_STEM}.{ext}")
            return

        # No Inkscape: cairosvg produces the same two copies. This matters beyond
        # convenience - the SVG only *names* its fonts (svg.fonttype='none'), so a
        # reader without Helvetica/Arial installed drops or substitutes the text.
        # The PDF embeds them, and PDF is what Nature accepts (SVG is not on the
        # accepted vector list: AI, EPS, PDF).
        try:
            import cairosvg
        except ImportError:
            print("Neither Inkscape nor cairosvg found - SVG written, no PDF/PNG copies.\n"
                  "  pip install cairosvg    (PDF is the format Nature accepts)")
            return
        cairosvg.svg2pdf(url=out_svg, write_to=OUT_STEM + '.pdf')
        print(f"Saved: {OUT_STEM}.pdf")
        cairosvg.svg2png(url=out_svg, write_to=OUT_STEM + '.png', output_width=3000)
        print(f"Saved: {OUT_STEM}.png")


if __name__ == '__main__':
    main()
