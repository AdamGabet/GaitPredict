"""
Assemble the Extended Data Figure 6 grid (panels a-d) from the individually
rendered panels in individual_plots/panels/:

  a = auc_sensitivity_significant_top10_male.pdf
  b = scenario_sensitivity_top10_male.pdf
  c = auc_sensitivity_significant_top10_female.pdf
  d = scenario_sensitivity_top10_female.pdf

The panel PDFs are placed as vector form XObjects, so every axis label, tick and
legend entry stays real, selectable text. The previous version composed the grid
as a PIL raster canvas, which flattened the whole figure to pixels.

Panels are placed at their NATIVE size (never rescaled), so effective font sizes
are identical across panels, and each is centred in its grid cell. Column widths
and row heights are the max over the panels they contain.

Run make_clinical_scenario_panels.py first so the panel PDFs are current.
Output written to output/clinical_scenario_medical_conditions_grid_ABCD.{png,pdf,svg}.
"""
import os
import pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))
PANEL_DIR = os.path.join(HERE, 'individual_plots', 'panels')
OUTPUT_DIR = os.path.join(HERE, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

PANELS = [
    ('a', 'auc_sensitivity_significant_top10_male.pdf'),
    ('b', 'scenario_sensitivity_top10_male.pdf'),
    ('c', 'auc_sensitivity_significant_top10_female.pdf'),
    ('d', 'scenario_sensitivity_top10_female.pdf'),
]

OUT_STEM = os.path.join(OUTPUT_DIR, 'clinical_scenario_medical_conditions_grid_ABCD')

PT = 72.0
GAP_X = 8.0          # gap between columns (points)
GAP_Y = 2.0          # gap between rows (points)
OUTER = 4.0          # outer margin (points)
LETTER_FS = 11       # larger panel letters, matching the source-panel artwork
PNG_DPI = 300

LETTER_FONT = 'hebo'


def main():
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
        heading = next(
            span["bbox"]
            for block in doc[0].get_text("dict")["blocks"]
            for line in block.get("lines", [])
            for span in line["spans"]
            if span["text"].strip() == "MEDICAL CONDITIONS"
        )
        legend_baseline = next(
            span["bbox"][3]
            for block in doc[0].get_text("dict")["blocks"]
            for line in block.get("lines", [])
            for span in line["spans"]
            if span["text"].strip() == "Age, BMI, Height"
        )
        writer.append(
            (px + heading[0], y + legend_baseline),
            letter,
            font=font,
            fontsize=LETTER_FS,
        )

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
