"""
Dark Channel Prior Image Dehazing (Python + OpenCV)

Based on: "Single Image Haze Removal Using Dark Channel Prior" (He et al., 2009)
Replaces MATLAB soft matting with OpenCV guided filter for much faster processing.

Usage:
    python dehaze.py <input_image> [-o output_image] [--omega 0.95] [--t0 0.1] [--radius 7]
"""

import argparse
import sys

import cv2
import numpy as np


def get_dark_channel(img, radius=7):
    """Compute dark channel: local minimum across RGB channels."""
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * radius + 1, 2 * radius + 1))
    return cv2.erode(min_channel, kernel)


def get_airlight(img, dark, top_percent=0.001):
    """Estimate atmospheric light from brightest pixels in dark channel."""
    h, w = dark.shape
    num_pixels = max(1, int(h * w * top_percent))
    flat_dark = dark.ravel()
    indices = np.argpartition(flat_dark, -num_pixels)[-num_pixels:]
    brightest = img.reshape(-1, 3)[indices]
    return brightest.max(axis=0)


def get_transmission(img, airlight, omega=0.95, radius=7):
    """Estimate raw transmission map."""
    normalized = img / airlight[np.newaxis, np.newaxis, :]
    return 1.0 - omega * get_dark_channel(normalized, radius)


def refine_transmission(img, transmission, radius=60, eps=1e-3):
    """Refine transmission with guided filter (replaces soft matting)."""
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    t = transmission.astype(np.float32)
    return cv2.ximgproc.guidedFilter(gray, t, radius, eps)


def recover(img, transmission, airlight, t0=0.1):
    """Recover haze-free image from atmospheric scattering model."""
    t = np.maximum(transmission, t0)[:, :, np.newaxis]
    return np.clip((img - airlight) / t + airlight, 0, 1)


def dehaze(img, omega=0.95, t0=0.1, radius=7):
    """
    Full dehazing pipeline.

    Args:
        img: BGR image (uint8)
        omega: haze removal strength (0-1), default 0.95
        t0: minimum transmission threshold, default 0.1
        radius: dark channel window radius, default 7

    Returns:
        Dehazed image (uint8)
    """
    img_float = img.astype(np.float64) / 255.0
    dark = get_dark_channel(img_float, radius)
    airlight = get_airlight(img_float, dark)
    transmission = get_transmission(img_float, airlight, omega, radius)
    transmission = refine_transmission(img_float, transmission)
    result = recover(img_float, transmission, airlight, t0)
    return (result * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Dark Channel Prior Image Dehazing")
    parser.add_argument("input", help="Input hazy image path")
    parser.add_argument("-o", "--output", help="Output image path (default: <input>_dehazed.<ext>)")
    parser.add_argument("--omega", type=float, default=0.95, help="Haze removal strength (default: 0.95)")
    parser.add_argument("--t0", type=float, default=0.1, help="Min transmission threshold (default: 0.1)")
    parser.add_argument("--radius", type=int, default=7, help="Dark channel window radius (default: 7)")
    args = parser.parse_args()

    img = cv2.imread(args.input)
    if img is None:
        print(f"Error: cannot read image '{args.input}'", file=sys.stderr)
        sys.exit(1)

    result = dehaze(img, omega=args.omega, t0=args.t0, radius=args.radius)

    if args.output:
        out_path = args.output
    else:
        name, ext = args.input.rsplit(".", 1)
        out_path = f"{name}_dehazed.{ext}"

    cv2.imwrite(out_path, result)
    print(f"Saved dehazed image to: {out_path}")


if __name__ == "__main__":
    main()
