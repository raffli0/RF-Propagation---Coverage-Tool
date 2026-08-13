"""Colormap lookup tables, value-to-RGBA mapping, and colorbar rendering."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import COLORMAPS


def colormap_lut(name):
    """Build a (256, 3) uint8 RGB lookup table from a colormap name."""
    stops = np.asarray(COLORMAPS[name], dtype=float)
    pos = np.linspace(0.0, 1.0, len(stops))
    lut = np.empty((256, 3))
    for c in range(3):
        lut[:, c] = np.interp(np.linspace(0.0, 1.0, 256), pos, stops[:, c])
    return lut.astype(np.uint8)


def colorize(loss, cmap_name, vmin, vmax):
    """Map a value array to an RGBA uint8 image via the colormap LUT."""
    lut = colormap_lut(cmap_name)
    span = vmax - vmin
    if span <= 1e-9:
        norm = np.zeros_like(loss, dtype=float)
    else:
        norm = np.clip((loss - vmin) / span, 0.0, 1.0)
    idx = (norm * 255).astype(np.uint8)
    rgba = np.zeros(loss.shape + (4,), dtype=np.uint8)
    rgba[..., :3] = lut[idx]
    rgba[..., 3] = 255
    return rgba


def nice_limits(vmin, vmax):
    """Round to a clean 10 dB range for the colorbar scale."""
    lo = float(np.floor(vmin / 10.0) * 10.0)
    hi = float(np.ceil(vmax / 10.0) * 10.0)
    if hi - lo < 10.0:
        hi = lo + 10.0
    return lo, hi


def colorbar_image(cmap_name, vmin, vmax, label="Path Loss [dB]"):
    """Render a gradient colorbar PNG (horizontal) with labels, geemap-style."""
    width, height = 430, 64
    img = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=12)
    lut = colormap_lut(cmap_name)
    x0, y0, x1, y1 = 18, 12, width - 18, 26
    for i in range(x1 - x0):
        color = tuple(int(v) for v in lut[int(i * 255 / max(x1 - x0 - 1, 1))])
        draw.line([(x0 + i, y0), (x0 + i, y1)], fill=color + (255,))
    draw.text((x0, 28), f"{vmin:g}", font=font, fill=(0, 0, 0, 255))
    max_text = f"{vmax:g}"
    draw.text((x1 - draw.textlength(max_text, font=font), 28), max_text, font=font, fill=(0, 0, 0, 255))
    title_w = draw.textlength(label, font=font)
    draw.text(((width - title_w) / 2, 42), label, font=font, fill=(0, 0, 0, 255))
    return img
