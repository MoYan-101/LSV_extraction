#!/usr/bin/env python3
"""Extract colored line-chart curves into CSV.

This is a pragmatic digitizer for plots where each curve can be separated by
color. It uses a calibrated plot rectangle plus axis ranges to convert pixels
into data values.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np


def parse_hex_color(value):
    value = value.strip()
    if value.startswith("#"):
        value = value[1:]
    if len(value) != 6:
        raise ValueError("hex color must look like #RRGGBB")
    return [int(value[i : i + 2], 16) for i in (0, 2, 4)]


def load_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError("Could not read image: {}".format(path))
    return image


def ensure_parent(path):
    parent = Path(path).parent
    if str(parent):
        parent.mkdir(parents=True, exist_ok=True)


def auto_detect_plot_area(image):
    """Best-effort plot rectangle detection using long straight border lines."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 50, 150)

    height, width = gray.shape[:2]
    min_line_len = int(min(width, height) * 0.25)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=min_line_len,
        maxLineGap=8,
    )
    if lines is None:
        return None

    horizontals = []
    verticals = []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = [int(v) for v in line]
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dy <= 4 and dx >= min_line_len:
            horizontals.append((min(x1, x2), int(round((y1 + y2) / 2)), max(x1, x2), dx))
        elif dx <= 4 and dy >= min_line_len:
            verticals.append((int(round((x1 + x2) / 2)), min(y1, y2), max(y1, y2), dy))

    if len(horizontals) < 2 or len(verticals) < 2:
        return None

    top = min(horizontals, key=lambda item: item[1])[1]
    bottom = max(horizontals, key=lambda item: item[1])[1]
    left = min(verticals, key=lambda item: item[0])[0]
    right = max(verticals, key=lambda item: item[0])[0]

    if right - left < width * 0.2 or bottom - top < height * 0.2:
        return None
    return {"left": left, "right": right, "top": top, "bottom": bottom}


def default_plot_area(image):
    height, width = image.shape[:2]
    return {
        "left": int(width * 0.12),
        "right": int(width * 0.92),
        "top": int(height * 0.08),
        "bottom": int(height * 0.88),
    }


def write_config_template(args):
    image = load_image(args.image)
    plot_area = auto_detect_plot_area(image) or default_plot_area(image)

    series = []
    for raw in args.series or []:
        parts = raw.split(":")
        if len(parts) not in (2, 3):
            raise ValueError("--series must be NAME:#RRGGBB or NAME:#RRGGBB:TOLERANCE")
        name = parts[0]
        color = parse_hex_color(parts[1])
        tolerance = float(parts[2]) if len(parts) == 3 else 35.0
        item = {
            "name": name,
            "rgb": color,
            "tolerance": tolerance,
            "min_area": 20,
            "min_x_span": 20,
        }
        if max(color) < 90:
            item["keep_largest_components"] = 1
        series.append(item)

    if not series:
        series = [
            {
                "name": "blue_curve",
                "rgb": [35, 45, 220],
                "tolerance": 35,
                "min_area": 20,
                "min_x_span": 20,
            },
            {
                "name": "red_curve",
                "rgb": [220, 35, 55],
                "tolerance": 35,
                "min_area": 20,
                "min_x_span": 20,
            },
            {
                "name": "black_curve",
                "rgb": [30, 30, 30],
                "tolerance": 28,
                "min_area": 20,
                "min_x_span": 20,
                "keep_largest_components": 1,
            },
        ]

    config = {
        "plot_area": plot_area,
        "axes": {
            "x": {
                "min": args.x_min,
                "max": args.x_max,
                "scale": "linear",
                "pixel_min": plot_area["left"],
                "pixel_max": plot_area["right"],
            },
            "y": {
                "min": args.y_min,
                "max": args.y_max,
                "scale": "linear",
                "pixel_min": plot_area["bottom"],
                "pixel_max": plot_area["top"],
            },
        },
        "border_margin_px": 2,
        "ignore_regions": [],
        "series": series,
        "output": {
            "x_step": args.x_step,
            "max_interpolate_gap_px": 8,
        },
    }
    ensure_parent(args.out)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    print("Wrote config template:", args.out)
    print("Detected image size: {}x{}".format(image.shape[1], image.shape[0]))
    print("Initial plot_area:", plot_area)


def normalize_plot_area(plot_area, image):
    height, width = image.shape[:2]
    left = int(round(plot_area["left"]))
    right = int(round(plot_area["right"]))
    top = int(round(plot_area["top"]))
    bottom = int(round(plot_area["bottom"]))
    left = max(0, min(left, width - 2))
    right = max(left + 1, min(right, width - 1))
    top = max(0, min(top, height - 2))
    bottom = max(top + 1, min(bottom, height - 1))
    return {"left": left, "right": right, "top": top, "bottom": bottom}


def group_tick_candidates(indices, scores, origin, max_cluster_width):
    if len(indices) == 0:
        return []

    groups = []
    start = int(indices[0])
    prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx == prev + 1:
            prev = idx
        else:
            groups.append((start, prev))
            start = idx
            prev = idx
    groups.append((start, prev))

    ticks = []
    for start, end in groups:
        width = end - start + 1
        if width > max_cluster_width:
            continue
        cluster_indices = np.arange(start, end + 1)
        cluster_scores = scores[cluster_indices].astype(np.float64)
        if np.sum(cluster_scores) > 0:
            center = float(np.average(cluster_indices, weights=cluster_scores))
        else:
            center = float((start + end) / 2.0)
        ticks.append(
            {
                "pixel": int(round(origin + center)),
                "strength": int(np.max(cluster_scores)) if len(cluster_scores) else 0,
                "width_px": int(width),
            }
        )
    return ticks


def merge_tick_candidates(candidate_lists, tolerance=2):
    merged = []
    candidates = []
    for candidate_list in candidate_lists:
        candidates.extend(candidate_list or [])
    if not candidates:
        return []

    candidates = sorted(candidates, key=lambda item: item["pixel"])
    group = [candidates[0]]
    for candidate in candidates[1:]:
        if abs(int(candidate["pixel"]) - int(group[-1]["pixel"])) <= int(tolerance):
            group.append(candidate)
        else:
            merged.append(merge_tick_candidate_group(group))
            group = [candidate]
    merged.append(merge_tick_candidate_group(group))
    return merged


def merge_tick_candidate_group(group):
    strengths = np.array([max(1, int(item.get("strength", 1))) for item in group], dtype=np.float64)
    pixels = np.array([float(item["pixel"]) for item in group], dtype=np.float64)
    pixel = int(round(np.average(pixels, weights=strengths)))
    return {
        "pixel": pixel,
        "strength": int(np.max(strengths)),
        "width_px": int(max(item.get("width_px", 1) for item in group)),
    }


def structural_tick_candidates(dark_band, origin, axis_name, min_tick_len, max_cluster_width):
    if dark_band.size == 0:
        return []

    band = dark_band.astype(np.uint8) * 255
    tick_len = max(2, int(min_tick_len))
    if axis_name == "x":
        kernel = np.ones((tick_len, 1), np.uint8)
        opened = cv2.morphologyEx(band, cv2.MORPH_OPEN, kernel)
        scores = np.sum(opened > 0, axis=0)
    else:
        kernel = np.ones((1, tick_len), np.uint8)
        opened = cv2.morphologyEx(band, cv2.MORPH_OPEN, kernel)
        scores = np.sum(opened > 0, axis=1)

    indices = np.where(scores >= max(1, tick_len - 1))[0]
    return group_tick_candidates(indices, scores, origin, int(max_cluster_width))


def auto_detect_ticks(image, plot_area, search_px=18, dark_threshold=170, min_tick_len=5, max_cluster_width=14):
    """Detect short axis tick lines near the left and bottom axes."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark = gray <= int(dark_threshold)
    height, width = gray.shape[:2]

    left = plot_area["left"]
    right = plot_area["right"]
    top = plot_area["top"]
    bottom = plot_area["bottom"]

    x_y1 = max(0, bottom - int(search_px))
    x_y2 = min(height, bottom + int(search_px) + 1)
    x_x1 = max(0, left)
    x_x2 = min(width, right + 1)
    x_band = dark[x_y1:x_y2, x_x1:x_x2]
    x_scores = np.sum(x_band, axis=0)
    x_baseline = np.median(x_scores)
    x_indices = np.where(x_scores >= x_baseline + int(min_tick_len))[0]
    x_projection_ticks = group_tick_candidates(x_indices, x_scores, x_x1, int(max_cluster_width))
    x_structural_ticks = structural_tick_candidates(
        x_band,
        x_x1,
        "x",
        int(min_tick_len),
        int(max_cluster_width),
    )
    x_ticks = merge_tick_candidates([x_projection_ticks, x_structural_ticks])

    y_x1 = max(0, left - int(search_px))
    y_x2 = min(width, left + int(search_px) + 1)
    y_y1 = max(0, top)
    y_y2 = min(height, bottom + 1)
    y_band = dark[y_y1:y_y2, y_x1:y_x2]
    y_scores = np.sum(y_band, axis=1)
    y_baseline = np.median(y_scores)
    y_indices = np.where(y_scores >= y_baseline + int(min_tick_len))[0]
    y_projection_ticks = group_tick_candidates(y_indices, y_scores, y_y1, int(max_cluster_width))
    y_structural_ticks = structural_tick_candidates(
        y_band,
        y_y1,
        "y",
        int(min_tick_len),
        int(max_cluster_width),
    )
    y_ticks = merge_tick_candidates([y_projection_ticks, y_structural_ticks])

    return {
        "x": sorted(x_ticks, key=lambda item: item["pixel"]),
        "y": sorted(y_ticks, key=lambda item: item["pixel"], reverse=True),
    }


def merge_calibration_template(existing_calibration, detected_ticks, tolerance=3):
    existing = existing_calibration or []
    merged = []
    for tick in detected_ticks:
        pixel = tick["pixel"]
        best = None
        best_dist = None
        for item in existing:
            if "pixel" not in item:
                continue
            dist = abs(float(item["pixel"]) - float(pixel))
            if best_dist is None or dist < best_dist:
                best = item
                best_dist = dist
        value = None
        if best is not None and best_dist is not None and best_dist <= tolerance:
            value = best.get("value")
        merged.append({"pixel": pixel, "value": value})
    return merged


def draw_tick_preview(image, plot_area, ticks, out_path):
    preview = image.copy()
    cv2.rectangle(
        preview,
        (plot_area["left"], plot_area["top"]),
        (plot_area["right"], plot_area["bottom"]),
        (0, 180, 255),
        2,
    )
    for idx, tick in enumerate(ticks.get("x", []), start=1):
        x = int(tick["pixel"])
        y = int(plot_area["bottom"])
        cv2.line(preview, (x, y - 12), (x, y + 12), (0, 180, 0), 2)
        cv2.putText(preview, str(idx), (x + 3, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 120, 0), 1)
    for idx, tick in enumerate(ticks.get("y", []), start=1):
        x = int(plot_area["left"])
        y = int(tick["pixel"])
        cv2.line(preview, (x - 12, y), (x + 12, y), (0, 120, 255), 2)
        cv2.putText(preview, str(idx), (max(0, x - 35), y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 90, 220), 1)
    ensure_parent(out_path)
    cv2.imwrite(str(out_path), preview)


def detect_ticks(args):
    image = load_image(args.image)
    if args.config:
        with open(args.config, "r", encoding="utf-8") as handle:
            config = json.load(handle)
    else:
        config = {
            "plot_area": auto_detect_plot_area(image) or default_plot_area(image),
            "axes": {"x": {"scale": "linear"}, "y": {"scale": "linear"}},
        }

    plot_area = normalize_plot_area(config.get("plot_area") or auto_detect_plot_area(image), image)
    config["plot_area"] = plot_area
    config.setdefault("axes", {})
    config["axes"].setdefault("x", {"scale": "linear"})
    config["axes"].setdefault("y", {"scale": "linear"})

    ticks = auto_detect_ticks(
        image,
        plot_area,
        search_px=args.search_px,
        dark_threshold=args.dark_threshold,
        min_tick_len=args.min_tick_len,
        max_cluster_width=args.max_cluster_width,
    )
    config["detected_ticks"] = ticks
    config["axes"]["x"]["calibration"] = merge_calibration_template(
        config["axes"]["x"].get("calibration"), ticks["x"]
    )
    config["axes"]["y"]["calibration"] = merge_calibration_template(
        config["axes"]["y"].get("calibration"), ticks["y"]
    )

    ensure_parent(args.out)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    print("Detected x ticks:", [item["pixel"] for item in ticks["x"]])
    print("Detected y ticks:", [item["pixel"] for item in ticks["y"]])
    print("Wrote tick config:", args.out)

    if args.preview:
        draw_tick_preview(image, plot_area, ticks, args.preview)
        print("Wrote tick preview:", args.preview)


def apply_ignore_regions(mask, plot_area, regions):
    if not regions:
        return
    left = plot_area["left"]
    top = plot_area["top"]
    for region in regions:
        x1 = int(round(region["left"])) - left
        x2 = int(round(region["right"])) - left
        y1 = int(round(region["top"])) - top
        y2 = int(round(region["bottom"])) - top
        x1 = max(0, min(x1, mask.shape[1]))
        x2 = max(0, min(x2, mask.shape[1]))
        y1 = max(0, min(y1, mask.shape[0]))
        y2 = max(0, min(y2, mask.shape[0]))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 0


def make_color_mask(roi_bgr, series_cfg):
    if "hsv_lower" in series_cfg and "hsv_upper" in series_cfg:
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        lower = np.array(series_cfg["hsv_lower"], dtype=np.uint8)
        upper = np.array(series_cfg["hsv_upper"], dtype=np.uint8)
        if lower[0] <= upper[0]:
            mask = cv2.inRange(hsv, lower, upper)
        else:
            lower_a = np.array([0, lower[1], lower[2]], dtype=np.uint8)
            upper_a = upper
            lower_b = lower
            upper_b = np.array([179, upper[1], upper[2]], dtype=np.uint8)
            mask = cv2.bitwise_or(cv2.inRange(hsv, lower_a, upper_a), cv2.inRange(hsv, lower_b, upper_b))
        return mask

    rgb = series_cfg.get("rgb")
    if rgb is None and "hex" in series_cfg:
        rgb = parse_hex_color(series_cfg["hex"])
    if rgb is None:
        raise ValueError("Each series needs rgb, hex, or hsv_lower/hsv_upper")

    tolerance = float(series_cfg.get("tolerance", 35))
    target_rgb = np.array([[rgb]], dtype=np.uint8)
    target_bgr = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2BGR)
    target_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0, :]
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    dist = np.linalg.norm(lab - target_lab, axis=2)
    mask = (dist <= tolerance).astype(np.uint8) * 255

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    sat_min = series_cfg.get("saturation_min")
    value_min = series_cfg.get("value_min")
    value_max = series_cfg.get("value_max")
    if sat_min is not None:
        mask[hsv[:, :, 1] < int(sat_min)] = 0
    if value_min is not None:
        mask[hsv[:, :, 2] < int(value_min)] = 0
    if value_max is not None:
        mask[hsv[:, :, 2] > int(value_max)] = 0
    return mask


def postprocess_mask(mask, series_cfg):
    close_iter = int(series_cfg.get("close_iterations", 1))
    dilate_iter = int(series_cfg.get("dilate_iterations", 0))
    if close_iter > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    if dilate_iter > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=dilate_iter)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    min_area = int(series_cfg.get("min_area", 20))
    min_x_span = int(series_cfg.get("min_x_span", 10))
    min_y_span = int(series_cfg.get("min_y_span", 0))
    components = []

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < min_area or w < min_x_span or h < min_y_span:
            continue
        components.append((label, int(area), int(w), int(h)))

    if not components:
        return out

    keep_largest = series_cfg.get("keep_largest_components")
    if keep_largest is not None:
        components = sorted(components, key=lambda item: item[1], reverse=True)[: int(keep_largest)]

    for label, _, _, _ in components:
        out[labels == label] = 255
    return out


def axis_endpoints(axis_name, axis_cfg, plot_area):
    """Return two pixel/value anchors for one axis.

    pixel_min/pixel_max are image pixel coordinates, not cropped ROI
    coordinates. For y axes, pixel_min is usually the lower tick, e.g. 0, and
    pixel_max is usually the upper tick, e.g. 24.
    """
    if "pixel_min" in axis_cfg and "pixel_max" in axis_cfg:
        return (
            float(axis_cfg["pixel_min"]),
            float(axis_cfg["min"]),
            float(axis_cfg["pixel_max"]),
            float(axis_cfg["max"]),
        )

    if axis_name == "x":
        return (
            float(plot_area["left"]),
            float(axis_cfg["min"]),
            float(plot_area["right"]),
            float(axis_cfg["max"]),
        )
    return (
        float(plot_area["bottom"]),
        float(axis_cfg["min"]),
        float(plot_area["top"]),
        float(axis_cfg["max"]),
    )


def pixel_to_value(pixel, axis_name, axis_cfg, plot_area):
    scale = axis_cfg.get("scale", "linear")
    if "calibration" in axis_cfg:
        calibration = [
            item
            for item in axis_cfg["calibration"]
            if item.get("value") not in (None, "") and item.get("pixel") not in (None, "")
        ]
        if len(calibration) >= 2:
            pixels = np.array([float(item["pixel"]) for item in calibration], dtype=np.float64)
            values = np.array([float(item["value"]) for item in calibration], dtype=np.float64)
            if scale == "linear":
                slope, intercept = np.polyfit(pixels, values, 1)
                return float(slope * pixel + intercept)
            if scale == "log10":
                if np.any(values <= 0):
                    raise ValueError("log10 axes require positive calibration values")
                slope, intercept = np.polyfit(pixels, np.log10(values), 1)
                return float(10 ** (slope * pixel + intercept))
            raise ValueError("Unsupported axis scale: {}".format(scale))

        if "min" not in axis_cfg or "max" not in axis_cfg:
            raise ValueError(
                "{} axis needs at least two calibration ticks with numeric values".format(axis_name)
            )

    pixel_min, value_min, pixel_max, value_max = axis_endpoints(axis_name, axis_cfg, plot_area)
    frac = (pixel - pixel_min) / float(pixel_max - pixel_min)
    if scale == "linear":
        return value_min + frac * (value_max - value_min)
    if scale == "log10":
        if value_min <= 0 or value_max <= 0:
            raise ValueError("log10 axes require positive calibration values")
        return 10 ** (math.log10(value_min) + frac * (math.log10(value_max) - math.log10(value_min)))
    raise ValueError("Unsupported axis scale: {}".format(scale))


def extract_points_from_mask(mask, plot_area, axes_cfg, series_cfg):
    min_pixels = int(series_cfg.get("min_pixels_per_column", 1))
    reducer = series_cfg.get("column_reducer", "median")

    raw = []
    for local_x in range(mask.shape[1]):
        ys = np.where(mask[:, local_x] > 0)[0]
        if len(ys) < min_pixels:
            continue
        if reducer == "mean":
            local_y = float(np.mean(ys))
        elif reducer == "top":
            local_y = float(np.min(ys))
        elif reducer == "bottom":
            local_y = float(np.max(ys))
        else:
            local_y = float(np.median(ys))

        pixel_x = plot_area["left"] + local_x
        pixel_y = plot_area["top"] + local_y
        x_val = pixel_to_value(pixel_x, "x", axes_cfg["x"], plot_area)
        y_val = pixel_to_value(pixel_y, "y", axes_cfg["y"], plot_area)
        raw.append(
            {
                "pixel_x": float(pixel_x),
                "pixel_y": float(pixel_y),
                "x": float(x_val),
                "y": float(y_val),
            }
        )
    return raw


def resample_points(points, x_step):
    if not points or x_step is None:
        return points
    x_step = float(x_step)
    if x_step <= 0:
        return points

    xs = np.array([p["x"] for p in points], dtype=np.float64)
    ys = np.array([p["y"] for p in points], dtype=np.float64)
    pxs = np.array([p["pixel_x"] for p in points], dtype=np.float64)
    pys = np.array([p["pixel_y"] for p in points], dtype=np.float64)

    order = np.argsort(xs)
    xs, ys, pxs, pys = xs[order], ys[order], pxs[order], pys[order]
    unique_xs, unique_indices = np.unique(xs, return_index=True)
    xs = unique_xs
    ys = ys[unique_indices]
    pxs = pxs[unique_indices]
    pys = pys[unique_indices]
    if len(xs) < 2:
        return points

    qxs = np.arange(xs[0], xs[-1] + x_step * 0.5, x_step)
    qys = np.interp(qxs, xs, ys)
    qpxs = np.interp(qxs, xs, pxs)
    qpys = np.interp(qxs, xs, pys)
    return [
        {"x": float(x), "y": float(y), "pixel_x": float(px), "pixel_y": float(py)}
        for x, y, px, py in zip(qxs, qys, qpxs, qpys)
    ]


def write_csv(path, rows):
    ensure_parent(path)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["series", "x", "y", "pixel_x", "pixel_y"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def draw_preview(image, plot_area, rows, series_cfgs, out_path, point_radius=3):
    preview = image.copy()
    cv2.rectangle(
        preview,
        (plot_area["left"], plot_area["top"]),
        (plot_area["right"], plot_area["bottom"]),
        (0, 180, 255),
        2,
    )
    color_map = {}
    for series in series_cfgs:
        rgb = series.get("rgb")
        if rgb is None and "hex" in series:
            rgb = parse_hex_color(series["hex"])
        if rgb is None:
            rgb = [0, 255, 0]
        color_map[series["name"]] = (int(rgb[2]), int(rgb[1]), int(rgb[0]))

    for row in rows:
        color = color_map.get(row["series"], (0, 255, 0))
        cv2.circle(
            preview,
            (int(round(row["pixel_x"])), int(round(row["pixel_y"]))),
            int(point_radius),
            color,
            -1,
        )

    ensure_parent(out_path)
    cv2.imwrite(str(out_path), preview)


def extract(args):
    with open(args.config, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    image = load_image(args.image)
    plot_area = normalize_plot_area(config.get("plot_area") or auto_detect_plot_area(image), image)
    axes_cfg = config["axes"]

    left = plot_area["left"]
    right = plot_area["right"]
    top = plot_area["top"]
    bottom = plot_area["bottom"]
    margin = int(config.get("border_margin_px", 2))
    roi = image[top + margin : bottom - margin, left + margin : right - margin]
    roi_plot_area = {
        "left": left + margin,
        "right": right - margin,
        "top": top + margin,
        "bottom": bottom - margin,
    }

    rows = []
    debug_dir = Path(args.debug_dir) if args.debug_dir else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    for series_cfg in config["series"]:
        name = series_cfg["name"]
        mask = make_color_mask(roi, series_cfg)
        ignore_regions = list(config.get("ignore_regions", [])) + list(series_cfg.get("ignore_regions", []))
        apply_ignore_regions(mask, roi_plot_area, ignore_regions)
        mask = postprocess_mask(mask, series_cfg)

        points = extract_points_from_mask(mask, roi_plot_area, axes_cfg, series_cfg)
        x_step = series_cfg.get("x_step", config.get("output", {}).get("x_step"))
        points = resample_points(points, x_step)

        for point in points:
            rows.append(
                {
                    "series": name,
                    "x": "{:.10g}".format(point["x"]),
                    "y": "{:.10g}".format(point["y"]),
                    "pixel_x": "{:.3f}".format(point["pixel_x"]),
                    "pixel_y": "{:.3f}".format(point["pixel_y"]),
                }
            )

        if debug_dir:
            safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)
            cv2.imwrite(str(debug_dir / "{}_mask.png".format(safe_name)), mask)
        print("{}: {} points".format(name, len(points)))

    write_csv(args.out, rows)
    print("Wrote CSV:", args.out)

    if args.preview:
        preview_rows = []
        for row in rows:
            preview_rows.append(
                {
                    "series": row["series"],
                    "pixel_x": float(row["pixel_x"]),
                    "pixel_y": float(row["pixel_y"]),
                }
            )
        draw_preview(image, plot_area, preview_rows, config["series"], args.preview)
        print("Wrote preview:", args.preview)


def build_parser():
    parser = argparse.ArgumentParser(description="Digitize colored line charts into CSV.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create a starting JSON config for an image.")
    init_parser.add_argument("image", type=Path)
    init_parser.add_argument("--out", default="line_digitizer_config.json")
    init_parser.add_argument("--x-min", type=float, required=True)
    init_parser.add_argument("--x-max", type=float, required=True)
    init_parser.add_argument("--y-min", type=float, required=True)
    init_parser.add_argument("--y-max", type=float, required=True)
    init_parser.add_argument("--x-step", type=float, default=None, help="Optional resampling step in x-axis units.")
    init_parser.add_argument(
        "--series",
        action="append",
        help="Series definition as NAME:#RRGGBB[:LAB_TOLERANCE]. Can be repeated.",
    )
    init_parser.set_defaults(func=write_config_template)

    ticks_parser = subparsers.add_parser("detect-ticks", help="Detect axis tick-line pixel positions.")
    ticks_parser.add_argument("image", type=Path)
    ticks_parser.add_argument("--config", default=None, help="Existing config to update.")
    ticks_parser.add_argument("--out", default="line_digitizer_ticks_config.json")
    ticks_parser.add_argument("--preview", default=None, help="Optional image showing detected ticks.")
    ticks_parser.add_argument(
        "--search-px",
        type=int,
        default=18,
        help="Pixel band around bottom/left axis to search for tick lines.",
    )
    ticks_parser.add_argument(
        "--dark-threshold",
        type=int,
        default=170,
        help="Grayscale threshold for dark axis/tick pixels.",
    )
    ticks_parser.add_argument(
        "--min-tick-len",
        type=int,
        default=5,
        help="Minimum dark-pixel run length for a tick candidate.",
    )
    ticks_parser.add_argument(
        "--max-cluster-width",
        type=int,
        default=14,
        help="Maximum tick candidate thickness in pixels.",
    )
    ticks_parser.set_defaults(func=detect_ticks)

    extract_parser = subparsers.add_parser("extract", help="Extract line data with a JSON config.")
    extract_parser.add_argument("image", type=Path)
    extract_parser.add_argument("--config", required=True)
    extract_parser.add_argument("--out", default="line_data.csv")
    extract_parser.add_argument("--preview", default=None)
    extract_parser.add_argument("--debug-dir", default=None)
    extract_parser.set_defaults(func=extract)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
