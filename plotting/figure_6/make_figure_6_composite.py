"""
Assemble the Figure 6 composite (panels a-c) from the individually-rendered
review1 panels in this directory:

  a = figure6a/body_system_heatmap_top10_both.png      (Both-sexes body-system x body-group heatmap)
  b = figure6b/grouped_by_body_part_male_pearson.png   (Male: 4 body groups + top labels)
  c = figure6b/grouped_by_body_part_female_pearson.png (Female: 4 body groups + top labels)

Layout: panel a fills the left column and is exactly as tall as panels b and c
combined; b (top) and c (bottom) stack on the right. Each panel is autocropped
to its non-white bounding box first, so surrounding whitespace is removed, and
placed at its native aspect ratio (no distortion).

Run plot_body_system_heatmap.py and plot_grouped_by_body_part.py --metric pearson
first so the component PNGs are current, then run this.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

HERE = os.path.dirname(os.path.abspath(__file__))
PANELS = os.path.join(HERE, 'individual_plots')

HEATMAP = os.path.join(PANELS, 'figure6a', 'body_system_heatmap_top10_both.png')
MALE = os.path.join(PANELS, 'figure6b', 'grouped_by_body_part_male_pearson.png')
FEMALE = os.path.join(PANELS, 'figure6b', 'grouped_by_body_part_female_pearson.png')

OUT_STEM = os.path.join(HERE, 'output', 'figure_6_composite')

FIGW = 22.0          # total figure width (inches)
HGAP = 0.45          # horizontal gap between left (a) and right (b/c) columns (inches)
VGAP = 0.10          # vertical gap between panels b and c (inches)
PAD = 8              # px of white padding kept around each autocropped panel
LETTER_FS = 34


def autocrop(img, thresh=0.985, pad=PAD):
    """Trim uniform near-white borders. img is HxWx{3,4} float in [0,1]."""
    rgb = img[..., :3]
    nonwhite = np.any(rgb < thresh, axis=2)
    if not nonwhite.any():
        return img
    rows = np.where(nonwhite.any(axis=1))[0]
    cols = np.where(nonwhite.any(axis=0))[0]
    r0, r1 = rows[0], rows[-1] + 1
    c0, c1 = cols[0], cols[-1] + 1
    r0 = max(0, r0 - pad); c0 = max(0, c0 - pad)
    r1 = min(img.shape[0], r1 + pad); c1 = min(img.shape[1], c1 + pad)
    return img[r0:r1, c0:c1]


def ar(img):
    """Aspect ratio width / height."""
    return img.shape[1] / img.shape[0]


def main():
    heat = autocrop(mpimg.imread(HEATMAP))
    male = autocrop(mpimg.imread(MALE))
    female = autocrop(mpimg.imread(FEMALE))

    ar_h, ar_m, ar_f = ar(heat), ar(male), ar(female)

    # Solve geometry so panel a height == (b height + VGAP + c height), both
    # columns at native AR, total width == FIGW.
    #   right_h = right_w/ar_m + VGAP + right_w/ar_f
    #   a_w     = right_h * ar_h          (a is as tall as b+c)
    #   a_w + HGAP + right_w == FIGW
    inv = 1.0 / ar_m + 1.0 / ar_f
    right_w = (FIGW - HGAP - ar_h * VGAP) / (1.0 + ar_h * inv)
    male_h = right_w / ar_m
    female_h = right_w / ar_f
    right_h = male_h + VGAP + female_h
    a_w = right_h * ar_h
    figh = right_h

    fig = plt.figure(figsize=(FIGW, figh))

    def add(x_in, y_in, w_in, h_in):
        axx = fig.add_axes([x_in / FIGW, y_in / figh, w_in / FIGW, h_in / figh])
        axx.axis('off')
        return axx

    # --- Panel a: left column, full height ---
    ax_a = add(0.0, 0.0, a_w, figh)
    ax_a.imshow(heat)
    ax_a.text(-0.04, 1.0, 'a', transform=ax_a.transAxes,
              ha='right', va='top', fontsize=LETTER_FS, fontweight='bold')

    # --- Right column: b (top) over c (bottom) ---
    right_x = a_w + HGAP
    b_y = figh - male_h
    c_y = 0.0

    ax_b = add(right_x, b_y, right_w, male_h)
    ax_b.imshow(male)
    ax_b.text(0.0, 1.0, 'b', transform=ax_b.transAxes,
              ha='right', va='top', fontsize=LETTER_FS, fontweight='bold')

    ax_c = add(right_x, c_y, right_w, female_h)
    ax_c.imshow(female)
    ax_c.text(0.0, 1.0, 'c', transform=ax_c.transAxes,
              ha='right', va='top', fontsize=LETTER_FS, fontweight='bold')

    for ext in ('png', 'pdf'):
        out = f'{OUT_STEM}.{ext}'
        fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'Saved: {out}')
    plt.close(fig)


if __name__ == '__main__':
    main()
