"""
Assemble the Figure 2 composite (panels a-h) from the individually-rendered panels
in individual_plots/:

  a = gender_roc_curves_long_seq.png        (Sex classification ROC)
  b = umap_by_activity.png + _legend.png     (UMAP, no-Room-A embeddings)
  c-h = grid_2x3_long_seq_with_movement_data.png  (Age/BMI/VAT x Male/Female bars)

Panels are placed at their native aspect ratios (no distortion). Run the three
component scripts in individual_plots/ first so the PNGs are current, then run this.
Output written to output/figure_2_composite.{png,pdf}.
"""
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

HERE = os.path.dirname(os.path.abspath(__file__))
PANELS = os.path.join(HERE, 'individual_plots')
OUTPUT_DIR = os.path.join(HERE, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

ROC = os.path.join(PANELS, 'gender_roc_curves_long_seq.png')
UMAP = os.path.join(PANELS, 'umap_by_activity.png')
UMAP_LEGEND = os.path.join(PANELS, 'umap_by_activity_legend.png')
GRID = os.path.join(PANELS, 'grid_2x3_long_seq_with_movement_data.png')

OUT_STEM = os.path.join(OUTPUT_DIR, 'figure_2_composite')

FIGW = 20.0          # figure width (inches)
HGAP = 0.5           # horizontal gap between the two top panels (inches)
VGAP = 0.3           # vertical gap between top row and the bar grid (inches)
LETTER_FS = 30


def ar(img):
    """Aspect ratio width / height."""
    return img.shape[1] / img.shape[0]


def main():
    roc = mpimg.imread(ROC)
    umap = mpimg.imread(UMAP)
    legend = mpimg.imread(UMAP_LEGEND)
    grid = mpimg.imread(GRID)

    # --- Top-row layout: two panels share a common height, fill FIGW ---
    wa, wb = ar(roc), ar(umap)
    top_h = (FIGW - HGAP) / (wa + wb)     # inches
    roc_w = wa * top_h
    umap_w = wb * top_h

    # --- Bottom grid spans full width ---
    grid_h = FIGW / ar(grid)

    figh = top_h + VGAP + grid_h
    fig = plt.figure(figsize=(FIGW, figh))

    def add(x_in, y_in, w_in, h_in):
        axx = fig.add_axes([x_in / FIGW, y_in / figh, w_in / FIGW, h_in / figh])
        axx.axis('off')
        return axx

    top_y = grid_h + VGAP

    # Panel a (ROC, top-left)
    ax_a = add(0.0, top_y, roc_w, top_h)
    ax_a.imshow(roc)

    # Panel b (UMAP, top-right) + inset legend (lower-left of the panel)
    ax_b = add(roc_w + HGAP, top_y, umap_w, top_h)
    ax_b.imshow(umap)
    leg_w = 0.36
    leg_h = leg_w * (umap_w / top_h) / ar(legend)   # keep legend aspect
    ax_leg = ax_b.inset_axes([0.043, 0.047, leg_w, leg_h])
    ax_leg.imshow(legend)
    ax_leg.axis('off')

    # Panels c-h (bar grid, full width below)
    ax_grid = add(0.0, 0.0, FIGW, grid_h)
    ax_grid.imshow(grid)

    # --- Panel letters ---
    def letter(ax, s, x=0.0, y=1.0):
        ax.text(x, y, s, transform=ax.transAxes, fontsize=LETTER_FS,
                fontweight='bold', va='top', ha='left')

    letter(ax_a, 'a', x=-0.02, y=1.04)
    letter(ax_b, 'b', x=-0.02, y=1.04)
    # 3 rows x 2 cols inside the grid image; letters at each cell's top-left
    cell_x = [0.005, 0.505]
    cell_y = [1.0, 0.665, 0.33]
    grid_letters = [['c', 'd'], ['e', 'f'], ['g', 'h']]
    for r, row in enumerate(grid_letters):
        for c, s in enumerate(row):
            ax_grid.text(cell_x[c], cell_y[r], s, transform=ax_grid.transAxes,
                         fontsize=LETTER_FS, fontweight='bold', va='top', ha='left')

    fig.savefig(OUT_STEM + '.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig.savefig(OUT_STEM + '.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved: {OUT_STEM}.png / .pdf  (figure {FIGW:.0f} x {figh:.1f} in)")


if __name__ == '__main__':
    main()
