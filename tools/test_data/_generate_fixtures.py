#!/usr/bin/env python3
"""
Generate the three test fixtures from synthetic content. Deterministic and
offline — does not download anything. Useful when the committed PNGs are
missing or have been regenerated locally.

Produces three 1024x1024 RGBA images:
  - source.png:               a procedural mountain-ish scene
  - canny.png:                canny edges of source.png
  - canny_of_different.png:   canny of an unrelated procedural scene
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


HERE = Path(__file__).resolve().parent
SIZE = 1024


def make_landscape(seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    img = Image.new("RGBA", (SIZE, SIZE), (135, 180, 220, 255))  # sky
    draw = ImageDraw.Draw(img)

    # Three mountain layers
    for layer, color in enumerate([(80, 110, 140), (60, 90, 130), (40, 60, 100)]):
        y_base = 480 + layer * 70
        amp = 180 - layer * 40
        xs = np.linspace(0, SIZE, 64)
        ys = y_base + rng.normal(0, amp * 0.3, xs.shape).astype(np.int32)
        ys = np.clip(ys, 100, SIZE - 100)
        pts = list(zip(xs.tolist(), ys.tolist()))
        pts = [(0, SIZE)] + pts + [(SIZE, SIZE)]
        draw.polygon(pts, fill=color + (255,))

    # Foreground "ground"
    draw.rectangle([0, int(SIZE * 0.85), SIZE, SIZE], fill=(50, 90, 60, 255))
    return img


def make_geometric(seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    img = Image.new("RGBA", (SIZE, SIZE), (240, 240, 240, 255))
    draw = ImageDraw.Draw(img)
    for _ in range(40):
        x0, y0 = rng.integers(0, SIZE - 200, size=2)
        w, h = rng.integers(80, 400, size=2)
        c = tuple(rng.integers(40, 220, size=3).tolist()) + (255,)
        if rng.random() < 0.5:
            draw.ellipse([x0, y0, x0 + int(w), y0 + int(h)], fill=c)
        else:
            draw.rectangle([x0, y0, x0 + int(w), y0 + int(h)], fill=c)
    return img


def to_canny(img: Image.Image) -> Image.Image:
    # Pillow has no canny; approximate with edges + threshold.
    gray = img.convert("L").filter(ImageFilter.GaussianBlur(radius=1.5))
    edges = gray.filter(ImageFilter.FIND_EDGES)
    arr = np.asarray(edges, dtype=np.uint8)
    binary = (arr > 30).astype(np.uint8) * 255
    out = Image.new("RGBA", img.size, (0, 0, 0, 255))
    out_arr = np.asarray(out).copy()
    out_arr[..., 0] = binary
    out_arr[..., 1] = binary
    out_arr[..., 2] = binary
    return Image.fromarray(out_arr, mode="RGBA")


def main() -> None:
    src = make_landscape(seed=1)
    src.save(HERE / "source.png")
    print(f"wrote {HERE / 'source.png'}")

    canny_src = to_canny(src)
    canny_src.save(HERE / "canny.png")
    print(f"wrote {HERE / 'canny.png'}")

    other = make_geometric(seed=2)
    canny_other = to_canny(other)
    canny_other.save(HERE / "canny_of_different.png")
    print(f"wrote {HERE / 'canny_of_different.png'}")


if __name__ == "__main__":
    main()
