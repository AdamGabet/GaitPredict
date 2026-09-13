# Extended Data Figure 7 — bundled frame

`skeleton_schematic_frame.csv` is the single skeleton frame drawn in Extended
Data Figure 7, plus the frame either side of it.

The figure is a schematic of the 26 retained Azure Kinect K4ABT joints, drawn
from one real A-pose frame so the limb proportions and joint spacing are
anatomically honest rather than hand-drawn. The frame was chosen by
`plot_skeleton_schematic.py --scan` over 300 randomly sampled front-camera
A-pose recordings (seed 0), scoring for neutral stance: minimal trunk lean,
level shoulders/hips/ankles, squareness to the camera, limb symmetry, and no
inter-frame motion.

## De-identification

The full recording is Human Phenotype Project cohort data and is not
redistributable. What is bundled here is only the three-frame window, with

- the participant code, camera and visit date (which are carried in the source
  recording's **filename**) not reproduced — the file is named for the figure,
  and `parse_identifier` reports the participant as `de-identified`;
- `timestamp_usec` dropped;
- `FrameNumber` renumbered `0, 1, 2`.

What remains is 32 joint positions, confidences and orientation quaternions for
three consecutive frames of one person standing still. The coordinates are
unmodified, so the published figure redraws exactly; the projected 2-D values
that are actually inked are also published in
`source_data/Extended_Data_Figure_7/`.

Frames 0 and 2 are present **only** so the central-difference velocity term in
`pose_quality` matches the value computed on the full recording; frame 1 is the
one drawn.

## Rebuild

```bash
python plotting/extended_figure_7/plot_skeleton_schematic.py
```

With access to the lab share, set `GAITPREDICT_SCHEMATIC_RECORDING` to a
recording's basename (or pass `--file` / `--scan`) to redraw from a full
recording instead.
