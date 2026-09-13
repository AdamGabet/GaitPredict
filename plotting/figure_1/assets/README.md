# Figure 1 assets

`make_figure_1_composite.py` needs two hand-prepared files that are not generated
by any script in this repo. Copy both in before building.

## figure_1_layout.svg

The draw.io export of the Figure 1 layout, with three edits applied:

1. the embedded `mxfile` source stripped,
2. draw.io's dark-mode CSS variables stripped,
3. the four rasters that have vector sources replaced by marker slots, i.e. the
   `<image>` element carrying `data-vector-slot="NAME"` for each of
   `TILES_MAIN`, `TILES_EXTRA_ROW`, `LOSS_PLOT`, `MAE_SCHEMATIC`.

Everything else is kept byte-for-byte, so the composite matches the draw.io
figure exactly. The build also expects exactly four `<switch>` label blocks
(draw.io's foreignObject headings), which it removes and re-emits as real
`<text>`.

## mae_schematic.svg

Page 1 of the GaitMAE / downstream slide PDF converted to SVG with its text kept
as text (not outlined).

## Build

    python plotting/figure_1/make_figure_1_composite.py

Needs `resvg-py` (installed) to measure ink bounding boxes. Inkscape is optional
and only used to export the PDF/PNG copies alongside the SVG.
