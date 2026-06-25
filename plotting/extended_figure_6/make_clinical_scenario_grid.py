from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
FINAL_DIR = HERE / "individual_plots" / "panels"   # input panels
OUT_DIR = HERE / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = OUT_DIR / "clinical_scenario_medical_conditions_grid_ABCD.png"
OUT_PDF = OUT_DIR / "clinical_scenario_medical_conditions_grid_ABCD.pdf"

PANELS = [
    ("a", "auc_sensitivity_significant_top10_male.png"),
    ("b", "scenario_sensitivity_top10_male.png"),
    ("c", "auc_sensitivity_significant_top10_female.png"),
    ("d", "scenario_sensitivity_top10_female.png"),
]


def trim_white(path: Path, border: int = 30) -> Image.Image:
    img = Image.open(path).convert("RGB")
    bg = Image.new("RGB", img.size, "white")
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    if bbox is None:
        return img
    left = max(0, bbox[0] - border)
    top = max(0, bbox[1] - border)
    right = min(img.size[0], bbox[2] + border)
    bottom = min(img.size[1], bbox[3] + border)
    return img.crop((left, top, right, bottom))


def load_font(size: int) -> ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/google-droid/DroidSans-Bold.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def pad_to_height(img: Image.Image, height: int) -> Image.Image:
    if img.height == height:
        return img
    out = Image.new("RGB", (img.width, height), "white")
    out.paste(img, (0, 0))
    return out


def trim_grid_vertical(img: Image.Image, border: int = 10) -> Image.Image:
    bg = Image.new("RGB", img.size, "white")
    bbox = ImageChops.difference(img, bg).getbbox()
    if bbox is None:
        return img
    top = max(0, bbox[1] - border)
    bottom = min(img.size[1], bbox[3] + border)
    return img.crop((0, top, img.size[0], bottom))


def main() -> None:
    # Do not resize panels: this preserves identical effective font sizes.
    # Do not trim either: independent trimming shifts axes/title baselines between panels.
    images = [(letter, Image.open(FINAL_DIR / filename).convert("RGB")) for letter, filename in PANELS]
    top_h = max(images[0][1].height, images[1][1].height)
    bottom_h = max(images[2][1].height, images[3][1].height)
    images = [
        (images[0][0], pad_to_height(images[0][1], top_h)),
        (images[1][0], pad_to_height(images[1][1], top_h)),
        (images[2][0], pad_to_height(images[2][1], bottom_h)),
        (images[3][0], pad_to_height(images[3][1], bottom_h)),
    ]
    label_font = load_font(69)

    gap_x = 48
    gap_y = 2
    outer = 12
    label_h = 0

    col_widths = [
        max(images[0][1].width, images[2][1].width),
        max(images[1][1].width, images[3][1].width),
    ]
    row_heights = [
        label_h + max(images[0][1].height, images[1][1].height),
        label_h + max(images[2][1].height, images[3][1].height),
    ]

    canvas_w = outer * 2 + col_widths[0] + gap_x + col_widths[1]
    canvas_h = outer * 2 + row_heights[0] + gap_y + row_heights[1]
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    positions = [
        (outer, outer, col_widths[0], row_heights[0]),
        (outer + col_widths[0] + gap_x, outer, col_widths[1], row_heights[0]),
        (outer, outer + row_heights[0] + gap_y, col_widths[0], row_heights[1]),
        (outer + col_widths[0] + gap_x, outer + row_heights[0] + gap_y, col_widths[1], row_heights[1]),
    ]

    for (letter, img), (x, y, cell_w, _cell_h) in zip(images, positions):
        paste_x = x + (cell_w - img.width) // 2
        paste_y = y + label_h
        canvas.paste(img, (paste_x, paste_y))
        draw.text((x + 10, y + 100), letter, font=label_font, fill="black")

    canvas = trim_grid_vertical(canvas, border=10)
    canvas.save(OUT_PNG, dpi=(450, 450))
    canvas.save(OUT_PDF, resolution=450)
    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
