"""
Assemble the Figure 6 composite (panels a-c) from the individually-rendered
panels in individual_plots/:

  a = figure6a/body_system_heatmap_top10_both.pdf      (Both-sexes body-system x body-group heatmap)
  b = figure6b/grouped_by_body_part_male_pearson.pdf   (Male: 4 body groups + top labels)
  c = figure6b/grouped_by_body_part_female_pearson.pdf (Female: 4 body groups + top labels)

The panel PDFs are placed as vector form XObjects, so every axis label, tick and
legend entry in the composite stays real, selectable text. The previous version
pasted rendered PNGs, which flattened all of that to pixels.

Layout: panel a fills the left column and is exactly as tall as panels b and c
combined; b (top) and c (bottom) stack on the right. Panels keep their native
aspect ratios (no distortion); the panel PDFs are already tightly cropped by
bbox_inches='tight', so no autocropping is needed.

Run plot_body_system_heatmap.py and plot_grouped_by_body_part.py --metric pearson
first so the component PDFs are current, then run this.
Output written to output/figure_6_composite.{png,pdf,svg}.
"""
import os
import pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))
PANELS = os.path.join(HERE, 'individual_plots')
OUTPUT_DIR = os.path.join(HERE, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

HEATMAP = os.path.join(PANELS, 'figure6a', 'body_system_heatmap_top10_both.pdf')
MALE = os.path.join(PANELS, 'figure6b', 'grouped_by_body_part_male_pearson.pdf')
FEMALE = os.path.join(PANELS, 'figure6b', 'grouped_by_body_part_female_pearson.pdf')

OUT_STEM = os.path.join(OUTPUT_DIR, 'figure_6_composite')

PT = 72.0            # points per inch
FIGW = 22.0 * PT     # total figure width
HGAP = 0.45 * PT     # gap between left (a) and right (b/c) columns
VGAP = 0.10 * PT     # gap between panels b and c
LETTER_FS = 34       # panel-letter size (points)
PNG_DPI = 200

# Panel letters use the PDF base-14 Helvetica-Bold, matching Nature's
# sans-serif requirement. base-14 needs no embedding, which also avoids the
# viewer-rejection seen when an OpenType/CFF face was embedded here.
LETTER_FONT = 'hebo'


def ar(page):
    """Aspect ratio width / height."""
    return page.rect.width / page.rect.height


def main():
    heat, male, female = (pymupdf.open(p) for p in (HEATMAP, MALE, FEMALE))
    ar_h, ar_m, ar_f = ar(heat[0]), ar(male[0]), ar(female[0])

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

    # Margins hold the panel letters that overhang up and to the left.
    ml = 0.02 * a_w
    mt = 0.03 * right_h

    page_w = ml + FIGW
    page_h = mt + right_h

    out = pymupdf.open()
    page = out.new_page(width=page_w, height=page_h)
    page.draw_rect(page.rect, color=None, fill=(1, 1, 1))

    def place(doc, x, y, w, h):
        page.show_pdf_page(pymupdf.Rect(x, y, x + w, y + h), doc, 0)

    # Panel a: left column, full height
    place(heat, ml, mt, a_w, right_h)

    # Right column: b (top) over c (bottom)
    right_x = ml + a_w + HGAP
    place(male, right_x, mt, right_w, male_h)
    place(female, right_x, mt + male_h + VGAP, right_w, female_h)

    # --- Panel letters ---
    font = pymupdf.Font(LETTER_FONT)
    writer = pymupdf.TextWriter(page.rect)

    def letter(s, x, y_top):
        """x = left edge, y_top = top of the glyph box (matplotlib va='top')."""
        writer.append((x, y_top + font.ascender * LETTER_FS), s,
                      font=font, fontsize=LETTER_FS)

    letter('a', ml - 0.018 * a_w, mt - 0.025 * right_h)
    letter('b', right_x - 0.018 * right_w, mt - 0.025 * right_h)
    letter('c', right_x - 0.018 * right_w, mt + male_h + VGAP - 0.025 * right_h)
    writer.write_text(page)

    out.save(OUT_STEM + '.pdf', garbage=4, deflate=True)
    page.get_pixmap(dpi=PNG_DPI).save(OUT_STEM + '.png')
    with open(OUT_STEM + '.svg', 'w') as fh:
        fh.write(page.get_svg_image(text_as_path=False))
    out.close()
    for d in (heat, male, female):
        d.close()
    print(f"Saved: {OUT_STEM}.png / .pdf / .svg  "
          f"(figure {page_w / PT:.1f} x {page_h / PT:.1f} in)")


if __name__ == '__main__':
    main()
