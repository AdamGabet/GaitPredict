"""2x2 grid for Supplementary Figure 4: asbv bars on top, figure-4-style boxplot below.

Columns = Male | Female. Rows = asbv (a, b) / figure-4-style legs-vs-full (c, d).

The panel PDFs are placed as vector form XObjects, so every axis label, tick and
legend entry stays real, selectable text. The previous version rendered each panel
to a PNG and composed the grid on a PIL raster canvas, which flattened the whole
figure to pixels -- the resulting PDF held a single image and no fonts at all. It
printed acceptably (620 dpi at 180 mm) but was the only figure in the article
without editable text, which Nature's artwork guide asks for.

Panels are placed at their NATIVE size (never rescaled), so effective font sizes
are identical across panels, and each is centred in its grid cell. Placing them at
native size also reproduces the old raster geometry: the panels were rendered at
300 dpi and saved into a 450 dpi canvas, i.e. shown at two-thirds size, and the
same ratio falls out of a 14.6 in page holding 7.2 in panels.

Run make_grid_panels.py first so the panel PDFs are current.
Output written to output/grid_asbv_figure4.{png,pdf,svg}.
"""
import os

import pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))
PANEL_DIR = os.path.join(HERE, 'individual_plots', 'grid_panels')
OUTPUT_DIR = os.path.join(HERE, 'output')

PANELS = [
    ('a', 'asbv_male.pdf'),
    ('b', 'asbv_female.pdf'),
    ('c', 'figure4_male.pdf'),
    ('d', 'figure4_female.pdf'),
]

OUT_STEM = os.path.join(OUTPUT_DIR, 'grid_asbv_figure4')

PT = 72.0
# The old PIL compositor used 48 / 2 / 12 px at 300 dpi; these are the same gaps
# converted to points, so the grid keeps its former spacing.
GAP_X = 11.5         # gap between columns (points)
GAP_Y = 0.5          # gap between rows (points)
OUTER = 3.0          # outer margin (points)
LETTER_FS = 12       # panel-letter size; 69 px at 300 dpi in the raster version
LETTER_DX = 2.0      # letter inset from the cell's left edge (points)
LETTER_DY = 2.0      # letter drop from the cell's top edge (points)
PNG_DPI = 300

LETTER_FONT = 'hebo'


def main():
    missing = [fn for _, fn in PANELS if not os.path.exists(os.path.join(PANEL_DIR, fn))]
    if missing:
        raise SystemExit(
            "Missing panel PDFs: " + ", ".join(missing) +
            "\nRun plotting/supp_figure_4/individual_plots/make_grid_panels.py first.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    docs = [(letter, pymupdf.open(os.path.join(PANEL_DIR, fn))) for letter, fn in PANELS]
    rects = [d[0].rect for _, d in docs]
    w = [r.width for r in rects]
    h = [r.height for r in rects]

    col_w = [max(w[0], w[2]), max(w[1], w[3])]
    row_h = [max(h[0], h[1]), max(h[2], h[3])]

    page_w = OUTER * 2 + col_w[0] + GAP_X + col_w[1]
    page_h = OUTER * 2 + row_h[0] + GAP_Y + row_h[1]

    out = pymupdf.open()
    page = out.new_page(width=page_w, height=page_h)
    page.draw_rect(page.rect, color=None, fill=(1, 1, 1))

    cell_x = [OUTER, OUTER + col_w[0] + GAP_X]
    cell_y = [OUTER, OUTER + row_h[0] + GAP_Y]
    cells = [(cell_x[0], cell_y[0], col_w[0]),
             (cell_x[1], cell_y[0], col_w[1]),
             (cell_x[0], cell_y[1], col_w[0]),
             (cell_x[1], cell_y[1], col_w[1])]

    font = pymupdf.Font(LETTER_FONT)
    writer = pymupdf.TextWriter(page.rect)

    for (letter, doc), (x, y, cw), pw, ph in zip(docs, cells, w, h):
        # Centre the panel horizontally in its cell, at native size.
        px = x + (cw - pw) / 2.0
        page.show_pdf_page(pymupdf.Rect(px, y, px + pw, y + ph), doc, 0)
        writer.append((x + LETTER_DX, y + LETTER_DY + font.ascender * LETTER_FS),
                      letter, font=font, fontsize=LETTER_FS)

    writer.write_text(page)

    out.save(OUT_STEM + '.pdf', garbage=4, deflate=True)
    page.get_pixmap(dpi=PNG_DPI).save(OUT_STEM + '.png')
    with open(OUT_STEM + '.svg', 'w') as fh:
        fh.write(page.get_svg_image(text_as_path=False))
    out.close()
    for _, d in docs:
        d.close()
    print(f"Saved: {OUT_STEM}.png / .pdf / .svg  "
          f"(figure {page_w / PT:.1f} x {page_h / PT:.1f} in)")


if __name__ == '__main__':
    main()
