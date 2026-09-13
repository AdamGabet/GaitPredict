"""Figure 1 schematic of the 26-joint skeleton the encoder consumes.

Everything anatomical in this figure comes from the GaitPredict repo and from a
real recording; nothing is drawn from synthetic or hand-placed coordinates.

  Joint set     deep_learning/preprocessing/joints_file.py
                `k4abt_joints` (the 32-joint Azure Kinect Body Tracking layout)
                minus `noise_joints` (the six low-confidence hand joints), which
                is exactly what preprocessing.py does for joint_format="noise26"
                and what normalizing_time_and_space.py assumes downstream.
  Connectivity  `noise_bones` from the same module -- 25 parent/child pairs
                already expressed in the 26-joint index space.
  Node positions  one frame of one participant's A-pose recording, projected to
                the coronal (frontal) plane by dropping the camera depth axis.

Projection
----------
Azure Kinect depth-camera coordinates are +X to the camera's right, +Y down and
+Z away from the camera (depth). A participant stands facing the camera, so the
camera's +X is the participant's anatomical LEFT. Plotting camera X unchanged on
the horizontal axis therefore puts anatomical left on the viewer's right -- the
standard anterior-view convention -- with no mirroring applied. The vertical
axis is -Y so that up on the page is up on the participant. Depth (Z) is dropped;
that is what makes this the coronal projection.

One consequence is worth knowing: K4ABT places HEAD inside the skull and NOSE in
front of it, so the two differ almost entirely in depth and land within ~3 mm of
each other in this projection. NOSE carries a small upward display offset (see
DISPLAY_OFFSETS_MM) so the pair reads as two nodes; nothing else is moved.

Why the A-pose activity: it is the neutral standing calibration pose, and the
arms held away from the trunk keep every limb joint separable, which a
Romberg/arms-at-sides stance would not.

Output
------
Vector PDF and EPS with live, editable text (fonttype 42 -- no outlining, no
flattening), plus a 300 dpi PNG for quick viewing. No transparency is used
anywhere, because the EPS backend silently rasterises artists that have alpha.

Usage
-----
    conda run -n NewtonModels python plotting/extended_figure_7/plot_skeleton_schematic.py
    ... --scan            re-run the neutral-frame search instead of using the
                          stored selection (prints the ranked candidates)
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Repo / data locations
# --------------------------------------------------------------------------

GAITPREDICT_REPO = os.environ.get(
    "GAITPREDICT_REPO", "/home/adamgab/PycharmProjects/GaitPredict"
)
APOSE_DIR = (
    "/net/mraid20/ifs/wisdom/segal_lab/jafar/Adam/skeleton_data/"
    "skeleton_full_data_sept/all_together/apose"
)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "figures")
STEM = "skeleton_schematic"

# The frame drawn in the published figure, selected by --scan over 300 randomly
# sampled front-camera A-pose recordings (seed 0). See `scan_for_neutral_frame`
# for the scoring.
#
# The recording it came from is Human Phenotype Project cohort data and is not
# redistributable, so the repo instead bundles a de-identified three-frame
# window around the selected frame (data/README.md). That is enough to redraw
# the figure exactly: the two neighbouring frames are present only so the
# central-difference velocity term in `pose_quality` matches the value computed
# on the full recording.
BUNDLED_RECORDING = os.path.join(HERE, "data", "skeleton_schematic_frame.csv")
BUNDLED_FRAME = 1  # the published frame sits in the middle of the window

# With access to the lab share, set GAITPREDICT_SCHEMATIC_RECORDING to the
# recording's basename to redraw from the full recording instead, or pass
# --file / --scan to pick a neutral frame from any recordings you do have.
SELECTED_BASENAME = os.environ.get("GAITPREDICT_SCHEMATIC_RECORDING", "")
if SELECTED_BASENAME:
    SELECTED_FILE = os.path.join(APOSE_DIR, SELECTED_BASENAME)
    SELECTED_FRAME = 190
else:
    SELECTED_FILE = BUNDLED_RECORDING
    SELECTED_FRAME = BUNDLED_FRAME

# --------------------------------------------------------------------------
# Style -- flat monochrome greyscale, Nature artwork rules
# --------------------------------------------------------------------------

BONE_COLOR = "#7A7F85"
NODE_COLOR = "#111111"
LEADER_COLOR = "#BFC3C7"
TEXT_COLOR = "#111111"

BONE_LW = 1.1
LEADER_LW = 0.5
NODE_SIZE = 16.0  # points^2
LABEL_PT = 6.0

FIG_WIDTH_MM = 180.0  # hard cap from the brief
FIG_MAX_HEIGHT_MM = 185.0  # leaves room for a caption on a Nature page
PAD_MM = 2.0

# --------------------------------------------------------------------------
# Display offsets -- the ONLY departure from the projected coordinates
# --------------------------------------------------------------------------
# K4ABT puts HEAD inside the skull and NOSE in front of it, so the two differ
# almost entirely in depth and land 2.9 mm apart in the coronal plane -- about a
# fifth of a node diameter at print size, which renders as a single blob. NOSE is
# lifted into the gap between HEAD (295 mm) and the lower eye joint (328 mm) so
# that the two nodes and the HEAD-NOSE bone are legible while NOSE still sits
# below both eyes. Nothing else is moved.
DISPLAY_OFFSETS_MM = {"NOSE": (0.0, 23.0)}

LEGEND_STRING = (
    "Coronal (frontal-plane) projection of the 26 retained Azure Kinect K4ABT "
    "joints; anatomical left is shown on the viewer's right."
)

MM_PER_INCH = 25.4


def mm2in(x: float) -> float:
    return x / MM_PER_INCH


def _renderer(fig):
    """The Agg renderer; fig.canvas.draw() must have run at least once."""
    return fig.canvas.get_renderer()  # type: ignore[attr-defined]


# --------------------------------------------------------------------------
# Joint set and connectivity, loaded straight from the model repo
# --------------------------------------------------------------------------


@dataclass
class JointSpec:
    names: list          # 26 K4ABT names, in model order
    orig_indices: list   # index of each in the original 32-joint layout
    bones: list          # 25 (parent, child) pairs in 26-joint index space
    dropped: list        # the 6 names removed as noise joints
    source: str


_JOINTS_MODULE_CACHE = {}


def _load_joints_module(repo: str = GAITPREDICT_REPO):
    """Import joints_file.py by path, once per repo.

    joints_file.py has no imports of its own, so it loads standalone without
    needing the GaitPredict package installed or on sys.path.
    """
    if repo in _JOINTS_MODULE_CACHE:
        return _JOINTS_MODULE_CACHE[repo]
    path = os.path.join(repo, "deep_learning", "preprocessing", "joints_file.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"joints_file.py not found at {path}. Set GAITPREDICT_REPO to the "
            "GaitPredict checkout."
        )
    spec = importlib.util.spec_from_file_location("gp_joints_file", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build an import spec for {path}")
    jf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(jf)
    _JOINTS_MODULE_CACHE[repo] = (jf, path)
    return jf, path


def _name_by_index(jf) -> dict:
    joints = jf.k4abt_joints
    return {v: k for k, v in vars(joints).items() if isinstance(v, int) and k != "len"}


def load_joint_spec(repo: str = GAITPREDICT_REPO) -> JointSpec:
    """Derive the 26-joint set and connectivity the model actually uses."""
    jf, path = _load_joints_module(repo)
    joints = jf.k4abt_joints
    by_index = _name_by_index(jf)

    # Exactly the expression used in preprocessing.py (joint_format="noise26")
    # and in normalizing_time_and_space.py.
    keep = [j for j in range(joints.len) if j not in jf.noise_joints]
    bones = [(int(a), int(b)) for a, b in jf.noise_bones]

    n = len(keep)
    if n != 26:
        raise AssertionError(f"expected 26 retained joints, derived {n}")
    if len(bones) != n - 1:
        raise AssertionError(f"expected {n - 1} bones for a tree over {n} joints, got {len(bones)}")
    for a, b in bones:
        if not (0 <= a < n and 0 <= b < n):
            raise AssertionError(f"bone ({a},{b}) is outside the 26-joint index space")

    return JointSpec(
        names=[by_index[j] for j in keep],
        orig_indices=keep,
        bones=bones,
        dropped=[by_index[j] for j in jf.noise_joints],
        source=path,
    )


# --------------------------------------------------------------------------
# Recording I/O and neutral-frame selection
# --------------------------------------------------------------------------

def _full32_names(repo: str = GAITPREDICT_REPO) -> list:
    """The 32 K4ABT joint names in their original order, as the CSVs name them."""
    jf, _ = _load_joints_module(repo)
    by_index = _name_by_index(jf)
    return [by_index[j] for j in range(jf.k4abt_joints.len)]


def read_recording(path: str):
    """Return (xyz [T,32,3] mm, confidence [T,32], dataframe)."""
    df = pd.read_csv(path)
    names = _full32_names()
    xyz = np.stack(
        [np.asarray(df[[f"{j}_{names[j]}_{a}" for a in "xyz"]].values, dtype=float)
         for j in range(32)], axis=1
    )
    conf = np.stack(
        [np.asarray(df[f"{j}_{names[j]}_c"].values, dtype=float) for j in range(32)], axis=1
    )
    return xyz, conf, df


def pose_quality(xyz: np.ndarray, conf: np.ndarray, keep: list, names32: list) -> dict:
    """Per-frame neutral-standing quality terms. Lower is better."""
    idx = {n: i for i, n in enumerate(names32)}
    K = xyz[:, keep, :]

    ok = (conf[:, keep] >= 2).all(axis=1)  # every retained joint tracked

    vel = np.full(len(K), np.inf)
    if len(K) > 2:
        vel[1:-1] = np.linalg.norm((K[2:] - K[:-2]) / 2.0, axis=2).mean(axis=1)

    sl, sr = idx["SHOULDER_LEFT"], idx["SHOULDER_RIGHT"]
    hl, hr = idx["HIP_LEFT"], idx["HIP_RIGHT"]

    sh_tilt = np.abs(np.degrees(np.arctan2(xyz[:, sl, 1] - xyz[:, sr, 1],
                                           xyz[:, sl, 0] - xyz[:, sr, 0])))
    hp_tilt = np.abs(np.degrees(np.arctan2(xyz[:, hl, 1] - xyz[:, hr, 1],
                                           xyz[:, hl, 0] - xyz[:, hr, 0])))
    # squareness to the camera: shoulder and hip lines should carry no depth
    frontal = (np.abs(xyz[:, sl, 2] - xyz[:, sr, 2])
               + np.abs(xyz[:, hl, 2] - xyz[:, hr, 2]))
    trunk = np.abs(np.degrees(np.arctan2(
        xyz[:, idx["NECK"], 0] - xyz[:, idx["PELVIS"], 0],
        xyz[:, idx["PELVIS"], 1] - xyz[:, idx["NECK"], 1])))

    def seg(a, b):
        return np.linalg.norm(xyz[:, idx[a]] - xyz[:, idx[b]], axis=1)

    sym = (np.abs(seg("HIP_LEFT", "KNEE_LEFT") - seg("HIP_RIGHT", "KNEE_RIGHT"))
           + np.abs(seg("KNEE_LEFT", "ANKLE_LEFT") - seg("KNEE_RIGHT", "ANKLE_RIGHT"))
           + np.abs(seg("SHOULDER_LEFT", "ELBOW_LEFT") - seg("SHOULDER_RIGHT", "ELBOW_RIGHT"))
           + np.abs(seg("ELBOW_LEFT", "WRIST_LEFT") - seg("ELBOW_RIGHT", "WRIST_RIGHT")))
    ankle_level = np.abs(xyz[:, idx["ANKLE_LEFT"], 1] - xyz[:, idx["ANKLE_RIGHT"], 1])

    score = (vel / 2.0 + sh_tilt + hp_tilt + frontal / 10.0
             + trunk + sym / 10.0 + ankle_level / 10.0)
    score = np.where(ok, score, np.inf)
    if len(score) > 10:
        score[:5] = np.inf   # tracker warm-up
        score[-5:] = np.inf  # drop-out at the tail
    return dict(score=score, vel=vel, shoulder_tilt_deg=sh_tilt, hip_tilt_deg=hp_tilt,
                frontal_mm=frontal, trunk_lean_deg=trunk, limb_asymmetry_mm=sym,
                ankle_level_mm=ankle_level, all_joints_tracked=ok)


def scan_for_neutral_frame(spec: JointSpec, n_files: int = 300, seed: int = 0, top: int = 10):
    """Rank frames across a random sample of front-camera A-pose recordings."""
    names32 = _full32_names()
    files = sorted(glob.glob(os.path.join(APOSE_DIR, "*_front__*.csv")))
    if not files:
        raise FileNotFoundError(f"no front-camera A-pose recordings under {APOSE_DIR}")
    rng = np.random.RandomState(seed)
    pick = sorted(rng.choice(len(files), size=min(n_files, len(files)), replace=False))

    rows = []
    for i in pick:
        f = files[i]
        try:
            xyz, conf, _ = read_recording(f)
        except Exception:
            continue
        if len(xyz) < 30:
            continue
        q = pose_quality(xyz, conf, spec.orig_indices, names32)
        t = int(np.argmin(q["score"]))
        if np.isfinite(q["score"][t]):
            rows.append((float(q["score"][t]), f, t))
    rows.sort()
    print(f"\nscanned {len(pick)} recordings, {len(rows)} usable\n")
    for s, f, t in rows[:top]:
        print(f"  {os.path.basename(f):46s} frame {t:4d}  score {s:7.2f}")
    return rows[:top]


def parse_identifier(path: str) -> dict:
    """<cohort>_<code>_<camera>__<date>.csv -> its parts."""
    base = os.path.basename(path)
    if os.path.abspath(path) == os.path.abspath(BUNDLED_RECORDING):
        return dict(participant="de-identified", camera="front", date="",
                    activity="apose", file=base)
    stem = base[:-4] if base.endswith(".csv") else base
    left, _, date = stem.partition("__")
    parts = left.split("_")
    return dict(
        participant="_".join(parts[:2]),  # cohort prefix + participant code
        camera=parts[2] if len(parts) > 2 else "",
        date=date,
        activity=os.path.basename(os.path.dirname(path)),
        file=path,
    )


# --------------------------------------------------------------------------
# Label layout
# --------------------------------------------------------------------------
# Anchors are hand-tuned against the stored frame, in its projected plot
# coordinates (mm; +x = anatomical left = viewer's right, +y = up). Side joints
# are labelled outward, midline joints are placed in the nearest clear space,
# and the head cluster -- far too tight for inline text -- is fanned out on
# leader lines.
#
# `pose_to_layout_frame` maps any pose onto the frame these anchors were tuned
# in (pelvis at the origin, scaled to the reference stature), so overriding
# --file/--frame moves the labels with the skeleton instead of detaching them.
#
#   name: (text_x, text_y, horizontal alignment, draw a leader line)

LABEL_LAYOUT = {
    # midline / spine
    "PELVIS":         (110, -200, "left", True),
    "SPINE_NAVEL":    (30, -103, "left", False),
    "SPINE_CHEST":    (30, 22, "left", False),
    "NECK":           (30, 230, "left", False),
    # head cluster -- all on leaders
    "NOSE":           (-45, 455, "right", True),
    "HEAD":           (45, 455, "left", True),
    "EYE_LEFT":       (165, 340, "left", True),
    "EAR_LEFT":       (165, 405, "left", True),
    "EYE_RIGHT":      (-165, 340, "right", True),
    "EAR_RIGHT":      (-165, 405, "right", True),
    # anatomical left (viewer's right)
    "CLAVICLE_LEFT":  (250, 252, "left", True),
    "SHOULDER_LEFT":  (190, 163, "left", False),
    "ELBOW_LEFT":     (338, -4, "left", False),
    "WRIST_LEFT":     (463, -143, "left", False),
    "HIP_LEFT":       (114, -254, "left", False),
    "KNEE_LEFT":      (100, -601, "left", False),
    "ANKLE_LEFT":     (92, -912, "left", False),
    "FOOT_LEFT":      (94, -1056, "left", False),
    # anatomical right (viewer's left)
    "CLAVICLE_RIGHT": (-250, 252, "right", True),
    "SHOULDER_RIGHT": (-168, 161, "right", False),
    "ELBOW_RIGHT":    (-334, -5, "right", False),
    "WRIST_RIGHT":    (-473, -142, "right", False),
    "HIP_RIGHT":      (-96, -254, "right", False),
    "KNEE_RIGHT":     (-104, -599, "right", False),
    "ANKLE_RIGHT":    (-106, -914, "right", False),
    "FOOT_RIGHT":     (-140, -1044, "right", False),
}


# Pelvis position and vertical extent of the stored frame, in projected plot
# coordinates, measured from the recording itself (verified at runtime below).
REF_PELVIS_PLOT = (4.9, -253.7)
REF_STATURE_MM = 1407.8


def pose_to_layout_frame(pts_plot: np.ndarray, pelvis_row: int):
    """Put a projected pose into the frame LABEL_LAYOUT was tuned in.

    Pelvis to the origin, then scaled so the pose spans REF_STATURE_MM
    vertically. For the stored frame this is the identity to within rounding.
    """
    centred = pts_plot - pts_plot[pelvis_row]
    stature = float(centred[:, 1].max() - centred[:, 1].min())
    scale = REF_STATURE_MM / stature
    return centred * scale, scale


def apply_display_offsets(pts_layout: np.ndarray, names: list) -> np.ndarray:
    """Apply DISPLAY_OFFSETS_MM. Returns a copy; the input is left untouched."""
    out = pts_layout.copy()
    for name, (dx, dy) in DISPLAY_OFFSETS_MM.items():
        if name not in names:
            raise KeyError(f"display offset names unknown joint {name!r}")
        out[names.index(name)] += (dx, dy)
    return out


def layout_anchor(name: str):
    """Anchor from LABEL_LAYOUT, moved into the pelvis-centred layout frame."""
    tx, ty, ha, leader = LABEL_LAYOUT[name]
    return tx - REF_PELVIS_PLOT[0], ty - REF_PELVIS_PLOT[1], ha, leader


def normalize_eps_origin(path: str):
    """Move the EPS artwork onto a 0,0 origin.

    matplotlib 3.7 has no ps.papersize='figure', so it centres the figure on a
    standard sheet and emits a bounding box offset by that margin, with a single
    matching `<x0> <y0> translate` as the first operator of the page. Both are
    rewritten here so the file is a tight page starting at the origin, which is
    what layout software expects. Nothing else is touched, and the translate has
    to match the bounding box or the file is left alone.
    """
    with open(path) as fh:
        lines = fh.read().split("\n")

    hires = next(i for i, ln in enumerate(lines) if ln.startswith("%%HiResBoundingBox:"))
    x0, y0, x1, y1 = (float(v) for v in lines[hires].split(":")[1].split())
    if (x0, y0) == (0.0, 0.0):
        return

    tr = next(i for i, ln in enumerate(lines)
              if i > hires and ln.endswith(" translate") and len(ln.split()) == 3)
    tx, ty = (float(v) for v in lines[tr].split()[:2])
    if abs(tx - x0) > 1e-3 or abs(ty - y0) > 1e-3:
        raise AssertionError(
            f"{os.path.basename(path)}: page translate ({tx}, {ty}) does not match "
            f"the bounding-box origin ({x0}, {y0}); not rewriting"
        )

    w, h = x1 - x0, y1 - y0
    bbox = next(i for i, ln in enumerate(lines) if ln.startswith("%%BoundingBox:"))
    lines[bbox] = f"%%BoundingBox: 0 0 {int(np.ceil(w))} {int(np.ceil(h))}"
    lines[hires] = f"%%HiResBoundingBox: 0.000000 0.000000 {w:.6f} {h:.6f}"
    lines[tr] = "0 0 translate"

    with open(path, "w") as fh:
        fh.write("\n".join(lines))


def apply_style():
    """Nature artwork rules. Set here as well as in the repo matplotlibrc so the
    script is correct when run from any working directory."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Nimbus Sans", "DejaVu Sans"],
        "pdf.fonttype": 42,   # TrueType: text stays text, not outlines
        "ps.fonttype": 42,
        "ps.papersize": "auto",
        "svg.fonttype": "none",
        "pdf.compression": 6,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
    })


def draw(points_mm: np.ndarray, spec: JointSpec):
    """Draw the schematic and size the canvas to the rendered content.

    Text has a fixed point size while the data limits are in millimetres, so the
    content extent and the figure size depend on each other. A few draw/measure
    passes settle it.
    """
    fig = plt.figure(figsize=(mm2in(FIG_WIDTH_MM), mm2in(FIG_WIDTH_MM)))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="box")

    # bones first, so nodes sit on top of the line ends
    for a, b in spec.bones:
        ax.plot(points_mm[[a, b], 0], points_mm[[a, b], 1],
                color=BONE_COLOR, lw=BONE_LW, solid_capstyle="round",
                zorder=2, clip_on=False)

    ax.scatter(points_mm[:, 0], points_mm[:, 1], s=NODE_SIZE, c=NODE_COLOR,
               linewidths=0, edgecolors="none", zorder=4, clip_on=False)

    texts, leaders = [], []
    for i, name in enumerate(spec.names):
        if name not in LABEL_LAYOUT:
            raise KeyError(f"no label anchor defined for joint {name}")
        tx, ty, ha, leader = layout_anchor(name)
        if leader:
            ln, = ax.plot([points_mm[i, 0], tx], [points_mm[i, 1], ty],
                          color=LEADER_COLOR, lw=LEADER_LW, zorder=1,
                          solid_capstyle="round", clip_on=False)
            leaders.append(ln)
        pad = 6.0 if ha == "left" else -6.0   # small gap between node and text
        texts.append(ax.text(tx + (pad if not leader else 0), ty, name,
                             ha=ha, va="center", fontsize=LABEL_PT,
                             color=TEXT_COLOR, zorder=5, clip_on=False))

    # ---- size the canvas to the content -------------------------------------
    width_mm, height_mm = FIG_WIDTH_MM, FIG_WIDTH_MM
    for _ in range(8):
        fig.canvas.draw()
        inv = ax.transData.inverted()
        xs = [points_mm[:, 0].min(), points_mm[:, 0].max()]
        ys = [points_mm[:, 1].min(), points_mm[:, 1].max()]
        for artist in texts + leaders:
            bb = artist.get_window_extent(_renderer(fig))
            (x0, y0), (x1, y1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
            xs += [x0, x1]
            ys += [y0, y1]
        dx_raw = max(xs) - min(xs)
        dy_raw = max(ys) - min(ys)

        pad_x = PAD_MM * dx_raw / width_mm
        pad_y = PAD_MM * dy_raw / height_mm
        xlim = (min(xs) - pad_x, max(xs) + pad_x)
        ylim = (min(ys) - pad_y, max(ys) + pad_y)
        dx, dy = xlim[1] - xlim[0], ylim[1] - ylim[0]

        # honour both the width cap and the page-height cap
        width_mm = min(FIG_WIDTH_MM, FIG_MAX_HEIGHT_MM * dx / dy)
        height_mm = width_mm * dy / dx

        fig.set_size_inches(mm2in(width_mm), mm2in(height_mm))
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    return fig, width_mm, height_mm


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", action="store_true",
                    help="re-run the neutral-frame search and print candidates")
    ap.add_argument("--file", default=SELECTED_FILE, help="recording CSV to draw from")
    ap.add_argument("--frame", type=int, default=SELECTED_FRAME, help="frame index")
    ap.add_argument("--outdir", default=OUT_DIR)
    args = ap.parse_args()

    apply_style()
    spec = load_joint_spec()
    names32 = _full32_names()

    print(f"joint set / connectivity source: {spec.source}")
    print(f"  retained joints : {len(spec.names)}")
    print(f"  bones           : {len(spec.bones)}")
    print(f"  dropped as noise: {', '.join(spec.dropped)}")

    if args.scan:
        top = scan_for_neutral_frame(spec)
        if top:
            _, args.file, args.frame = top[0]
            print(f"\nusing top candidate: {os.path.basename(args.file)} frame {args.frame}")

    if not args.file or not os.path.exists(args.file):
        raise SystemExit(
            f"recording not found: {args.file or '<none>'}\n"
            "Expected the bundled de-identified frame window at "
            f"{BUNDLED_RECORDING}.\n"
            "With access to the lab share you can instead set "
            "GAITPREDICT_SCHEMATIC_RECORDING to a\nrecording's basename, pass "
            "--file <recording.csv>, or run with --scan."
        )

    ident = parse_identifier(args.file)
    xyz, conf, df = read_recording(args.file)
    t = args.frame
    if not 0 <= t < len(xyz):
        raise IndexError(f"frame {t} outside recording of {len(xyz)} frames")

    q = pose_quality(xyz, conf, spec.orig_indices, names32)
    if not q["all_joints_tracked"][t]:
        raise ValueError(f"frame {t} has untracked joints among the 26 retained")

    frame_number = int(df["FrameNumber"].iloc[t]) if "FrameNumber" in df else t
    timestamp_usec = float(df["timestamp_usec"].iloc[t]) if "timestamp_usec" in df else float("nan")

    print("\nframe used")
    print(f"  participant   : {ident['participant']}")
    print(f"  visit date    : {ident['date']}")
    print(f"  camera        : {ident['camera']}")
    print(f"  activity      : {ident['activity']}")
    print(f"  frame index   : {t}  (FrameNumber {frame_number}, "
          f"timestamp {timestamp_usec:.0f} us)")
    print(f"  source file   : {ident['file']}")
    print("  pose quality  : "
          f"trunk lean {q['trunk_lean_deg'][t]:.2f} deg, "
          f"shoulder tilt {q['shoulder_tilt_deg'][t]:.2f} deg, "
          f"hip tilt {q['hip_tilt_deg'][t]:.2f} deg, "
          f"inter-frame motion {q['vel'][t]:.2f} mm/frame, "
          "all 26 joints at max tracker confidence")

    # ---- coronal projection -------------------------------------------------
    # keep the 26 model joints, drop depth, flip Y so up is up. Camera +X is the
    # participant's anatomical left, so no mirroring is applied.
    kept = xyz[t, spec.orig_indices, :]
    pts_plot = np.column_stack([kept[:, 0], -kept[:, 1]])
    pts_true, layout_scale = pose_to_layout_frame(pts_plot, spec.names.index("PELVIS"))
    if abs(layout_scale - 1.0) > 1e-3:
        print(f"  note: pose scaled by {layout_scale:.3f} onto the label-layout frame")

    pts = apply_display_offsets(pts_true, spec.names)
    sep_before = float(np.linalg.norm(pts_true[spec.names.index("HEAD")]
                                      - pts_true[spec.names.index("NOSE")]))
    sep_after = float(np.linalg.norm(pts[spec.names.index("HEAD")]
                                     - pts[spec.names.index("NOSE")]))
    print(f"\ndisplay offset: {DISPLAY_OFFSETS_MM} "
          f"(HEAD-NOSE separation {sep_before:.1f} -> {sep_after:.1f} mm)")

    # the offset must not push NOSE into one of its neighbours
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    i, j = np.unravel_index(int(np.argmin(d)), d.shape)
    print(f"closest node pair after offsets: {spec.names[i]} / {spec.names[j]} "
          f"at {d[i, j]:.1f} mm")
    for near in np.argsort(d[spec.names.index("NOSE")])[:3]:
        print(f"  NOSE -> {spec.names[near]:14s} {d[spec.names.index('NOSE'), near]:6.1f} mm")

    fig, w_mm, h_mm = draw(pts, spec)

    os.makedirs(args.outdir, exist_ok=True)
    paths = {}
    for ext in ("pdf", "eps", "png"):
        p = os.path.join(args.outdir, f"{STEM}.{ext}")
        fig.savefig(p, format=ext, dpi=300 if ext == "png" else None,
                    facecolor="white", transparent=False)
        if ext == "eps":
            normalize_eps_origin(p)
        paths[ext] = p
    plt.close(fig)

    provenance = dict(
        figure="Figure 1 -- 26-joint K4ABT skeleton schematic",
        joint_source=spec.source,
        joints=spec.names,
        original_k4abt_indices=spec.orig_indices,
        bones=spec.bones,
        dropped_noise_joints=spec.dropped,
        projection="coronal (frontal); camera X kept, camera Y negated, depth Z dropped",
        orientation=LEGEND_STRING,
        display_offsets_mm={k: list(v) for k, v in DISPLAY_OFFSETS_MM.items()},
        head_nose_separation_mm=dict(projected=round(sep_before, 2),
                                     as_drawn=round(sep_after, 2)),
        # projected positions before any display offset, so the figure can be
        # checked against the recording
        projected_xy_mm={n: [round(float(x), 2), round(float(y), 2)]
                         for n, (x, y) in zip(spec.names, pts_true)},
        **{f"record_{k}": v for k, v in ident.items()},
        frame_index=t,
        frame_number=frame_number,
        timestamp_usec=timestamp_usec,
        pose_quality={k: float(q[k][t]) for k in
                      ("trunk_lean_deg", "shoulder_tilt_deg", "hip_tilt_deg",
                       "frontal_mm", "limb_asymmetry_mm", "ankle_level_mm", "vel")},
        figure_size_mm=[round(w_mm, 2), round(h_mm, 2)],
        outputs=paths,
    )
    prov_path = os.path.join(args.outdir, f"{STEM}_provenance.json")
    with open(prov_path, "w") as fh:
        json.dump(provenance, fh, indent=2)

    print(f"\nfigure size: {w_mm:.1f} x {h_mm:.1f} mm  (cap {FIG_WIDTH_MM:.0f} mm wide)")
    print("written:")
    for ext in ("pdf", "eps", "png"):
        print(f"  {paths[ext]}")
    print(f"  {prov_path}")
    print(f"\nlegend string:\n  {LEGEND_STRING}")
    return LEGEND_STRING


if __name__ == "__main__":
    main()
