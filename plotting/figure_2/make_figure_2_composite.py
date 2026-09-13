"""
Assemble the Figure 2 composite (panels a-h) from the individually-rendered panels
in individual_plots/:

  a = gender_roc_curves_long_seq.pdf        (Sex classification ROC)
  b = umap_by_activity.pdf + _legend.pdf     (UMAP, no-Room-A embeddings)
  c-h = grid_2x3_long_seq_with_movement_data.pdf  (Age/BMI/VAT x Male/Female bars)

The panel PDFs are placed as vector form XObjects, so every axis label, tick and
legend entry in the composite stays real, selectable text (the UMAP scatter is the
one deliberate raster - it is rasterized at the panel level). Panels keep their
native aspect ratios (no distortion). Run the three component scripts in
individual_plots/ first so the PDFs are current, then run this.
Output written to output/figure_2_composite.{png,pdf,svg}.
"""
import os
import pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))
PANELS = os.path.join(HERE, 'individual_plots')
OUTPUT_DIR = os.path.join(HERE, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

ROC = os.path.join(PANELS, 'gender_roc_curves_long_seq.pdf')
UMAP = os.path.join(PANELS, 'umap_by_activity.pdf')
UMAP_LEGEND = os.path.join(PANELS, 'umap_by_activity_legend.pdf')
GRID = os.path.join(PANELS, 'grid_2x3_long_seq_with_movement_data.pdf')

OUT_STEM = os.path.join(OUTPUT_DIR, 'figure_2_composite')

PT = 72.0            # points per inch
FIGW = 20.0 * PT     # figure width
HGAP = 0.5 * PT      # horizontal gap between the two top panels
VGAP = 0.3 * PT      # vertical gap between top row and the bar grid
LETTER_FS = 30       # panel-letter size (points)
PNG_DPI = 200

# Panel letters use the PDF base-14 Helvetica-Bold, matching Nature's
# sans-serif requirement. base-14 needs no embedding, which also avoids the
# viewer-rejection seen when an OpenType/CFF face was embedded here.
LETTER_FONT = 'hebo'


def ar(page):
    """Aspect ratio width / height."""
    return page.rect.width / page.rect.height


def main():
    roc, umap, legend, grid = (pymupdf.open(p) for p in (ROC, UMAP, UMAP_LEGEND, GRID))

    # --- Top-row layout: two panels share a common height, fill FIGW ---
    wa, wb = ar(roc[0]), ar(umap[0])
    top_h = (FIGW - HGAP) / (wa + wb)
    roc_w = wa * top_h
    umap_w = wb * top_h

    # --- Bottom grid spans full width ---
    grid_h = FIGW / ar(grid[0])

    # Margins hold the panel letters that overhang up and to the left (as the old
    # bbox_inches='tight' save used to do).
    ml = 0.02 * roc_w
    mt = 0.04 * top_h

    page_w = ml + FIGW
    page_h = mt + top_h + VGAP + grid_h

    out = pymupdf.open()
    page = out.new_page(width=page_w, height=page_h)
    page.draw_rect(page.rect, color=None, fill=(1, 1, 1))

    def place(doc, x, y, w, h):
        page.show_pdf_page(pymupdf.Rect(x, y, x + w, y + h), doc, 0)

    grid_y = mt + top_h + VGAP

    # Panel a (ROC, top-left)
    place(roc, ml, mt, roc_w, top_h)

    # Panel b (UMAP, top-right) + inset legend (lower-left of the panel)
    umap_x = ml + roc_w + HGAP
    place(umap, umap_x, mt, umap_w, top_h)
    leg_w = 0.36 * umap_w
    leg_h = leg_w / ar(legend[0])
    place(legend, umap_x + 0.043 * umap_w,
          mt + top_h - 0.047 * top_h - leg_h, leg_w, leg_h)

    # Panels c-h (bar grid, full width below)
    place(grid, ml, grid_y, FIGW, grid_h)

    # --- Panel letters ---
    font = pymupdf.Font(LETTER_FONT)
    writer = pymupdf.TextWriter(page.rect)

    def letter(s, x, y_top):
        """x = left edge, y_top = top of the glyph box (matplotlib va='top')."""
        writer.append((x, y_top + font.ascender * LETTER_FS), s,
                      font=font, fontsize=LETTER_FS)

    letter('a', ml - 0.02 * roc_w, mt - 0.04 * top_h)
    letter('b', umap_x - 0.02 * umap_w, mt - 0.04 * top_h)
    # 3 rows x 2 cols inside the grid panel; letters at each cell's top-left
    cell_x = [0.005, 0.505]
    cell_y = [1.0, 0.665, 0.33]
    grid_letters = [['c', 'd'], ['e', 'f'], ['g', 'h']]
    for r, row in enumerate(grid_letters):
        for c, s in enumerate(row):
            letter(s, ml + cell_x[c] * FIGW, grid_y + (1 - cell_y[r]) * grid_h)
    writer.write_text(page)

    out.save(OUT_STEM + '.pdf', garbage=4, deflate=True)
    page.get_pixmap(dpi=PNG_DPI).save(OUT_STEM + '.png')
    # text_as_path=False keeps <text> elements (font is referenced, not embedded,
    # so the opening app needs the serif face installed).
    with open(OUT_STEM + '.svg', 'w') as fh:
        fh.write(page.get_svg_image(text_as_path=False))
    out.close()
    for d in (roc, umap, legend, grid):
        d.close()
    print(f"Saved: {OUT_STEM}.png / .pdf / .svg  "
          f"(figure {page_w / PT:.0f} x {page_h / PT:.1f} in)")


if __name__ == '__main__':
    main()
