#!/usr/bin/env python3
"""Streamlit batch UI for colored line-chart digitization."""

import csv
import copy
import html
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

import line_digitizer as ld

try:
    import pytesseract
except Exception:
    pytesseract = None


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def list_images(folder):
    root = Path(folder).expanduser()
    if not root.exists():
        return []
    return sorted([path for path in root.iterdir() if path.suffix.lower() in IMAGE_EXTS])


def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def hex_to_rgb(value):
    return ld.parse_hex_color(value)


def output_paths(output_root, image_path):
    root = Path(output_root).expanduser()
    stem = Path(image_path).stem
    return {
        "config": root / "configs" / f"{stem}.json",
        "csv": root / "csv" / f"{stem}.csv",
        "preview": root / "previews" / f"{stem}_overlay.png",
        "ticks": root / "previews" / f"{stem}_ticks.png",
        "debug": root / "debug_masks" / stem,
    }


def default_series():
    return [
        {
            "selected": True,
            "name": "blue",
            "hex": "#232ddc",
            "tolerance": 38,
            "saturation_min": 40,
            "min_area": 20,
            "min_x_span": 20,
            "keep_largest_components": "",
        },
        {
            "selected": True,
            "name": "red",
            "hex": "#dc2337",
            "tolerance": 42,
            "saturation_min": 40,
            "min_area": 20,
            "min_x_span": 20,
            "keep_largest_components": "",
        },
        {
            "selected": True,
            "name": "black",
            "hex": "#1e1e1e",
            "tolerance": 30,
            "saturation_min": "",
            "min_area": 20,
            "min_x_span": 20,
            "keep_largest_components": 1,
        },
    ]


def series_to_frame(series):
    rows = []
    for item in series:
        rgb = item.get("rgb")
        if rgb is None and "hex" in item:
            hex_value = item["hex"]
        elif rgb is None:
            hex_value = "#000000"
        else:
            hex_value = rgb_to_hex(rgb)
        rows.append(
            {
                "selected": item.get("selected", True),
                "name": item.get("name", "series"),
                "hex": hex_value,
                "tolerance": item.get("tolerance", 35),
                "saturation_min": item.get("saturation_min", ""),
                "min_area": item.get("min_area", 20),
                "min_x_span": item.get("min_x_span", 20),
                "keep_largest_components": item.get("keep_largest_components", ""),
            }
        )
    return pd.DataFrame(rows)


def is_truthy_selected(value):
    return value not in (False, "False", "false", 0, "0", "")


def frame_to_series(frame, selected_only=False):
    series = []
    for _, row in frame.fillna("").iterrows():
        selected = is_truthy_selected(row.get("selected", True))
        if selected_only and not selected:
            continue
        name = str(row.get("name", "")).strip()
        hex_value = str(row.get("hex", "")).strip()
        if not name or not hex_value:
            continue
        item = {
            "selected": selected,
            "name": name,
            "rgb": hex_to_rgb(hex_value),
            "tolerance": float(row.get("tolerance") or 35),
            "min_area": int(float(row.get("min_area") or 20)),
            "min_x_span": int(float(row.get("min_x_span") or 20)),
        }
        sat = row.get("saturation_min")
        if sat not in ("", None):
            item["saturation_min"] = int(float(sat))
        keep = row.get("keep_largest_components")
        if keep not in ("", None):
            item["keep_largest_components"] = int(float(keep))
        series.append(item)
    return series


def axis_values(vmin, vmax, count):
    count = int(count)
    if count <= 1:
        return [float(vmin)]
    return [float(v) for v in np.linspace(float(vmin), float(vmax), count)]


def select_ticks_by_count(ticks, count, plot_area, axis_name):
    count = int(count)
    if count <= 0 or not ticks:
        return []

    if axis_name == "x":
        ordered = sorted(ticks, key=lambda item: item["pixel"])
        expected_span = float(plot_area["right"] - plot_area["left"])
    else:
        ordered = sorted(ticks, key=lambda item: item["pixel"], reverse=True)
        expected_span = float(plot_area["bottom"] - plot_area["top"])

    if len(ordered) <= count:
        return ordered[:count]

    best = None
    max_strength = max(1.0, max(float(item.get("strength", 1)) for item in ordered))
    max_pairs = len(ordered) * (len(ordered) - 1) // 2
    if max_pairs > 2500:
        step = max(1, len(ordered) // 60)
        endpoint_indices = list(range(0, len(ordered), step))
        if endpoint_indices[-1] != len(ordered) - 1:
            endpoint_indices.append(len(ordered) - 1)
    else:
        endpoint_indices = list(range(len(ordered)))

    for start_idx in endpoint_indices:
        for end_idx in endpoint_indices:
            if end_idx <= start_idx:
                continue
            start_pixel = float(ordered[start_idx]["pixel"])
            end_pixel = float(ordered[end_idx]["pixel"])
            span = abs(end_pixel - start_pixel)
            if span < expected_span * 0.35:
                continue

            targets = np.linspace(start_pixel, end_pixel, count)
            selected = []
            selected_indices = []
            used = set()
            residuals = []
            for target in targets:
                candidates = []
                for idx, tick in enumerate(ordered):
                    if idx in used:
                        continue
                    dist = abs(float(tick["pixel"]) - float(target))
                    candidates.append((dist, idx, tick))
                if not candidates:
                    break
                dist, idx, tick = min(candidates, key=lambda item: item[0])
                selected.append(tick)
                selected_indices.append(idx)
                residuals.append(dist)
                used.add(idx)

            if len(selected) != count or selected_indices != sorted(selected_indices):
                continue

            mean_residual = float(np.mean(residuals))
            span_penalty = 0.02 * max(0.0, expected_span - span)
            strength = np.mean([float(item.get("strength", 1)) for item in selected]) / max_strength
            strength_reward = 2.0 * strength
            score = mean_residual + span_penalty - strength_reward
            candidate = (score, mean_residual, -span, selected)
            if best is None or candidate[:3] < best[:3]:
                best = candidate

    if best is not None:
        return best[3]

    targets = np.linspace(float(ordered[0]["pixel"]), float(ordered[-1]["pixel"]), count)
    selected = []
    used = set()
    for target in targets:
        candidates = []
        for idx, tick in enumerate(ordered):
            if idx in used:
                continue
            candidates.append((abs(float(tick["pixel"]) - float(target)), idx, tick))
        if not candidates:
            break
        _, idx, tick = min(candidates, key=lambda item: item[0])
        selected.append(tick)
        used.add(idx)
    return selected


def ticks_to_calibration(ticks, values):
    return [
        {"pixel": int(tick["pixel"]), "value": float(value)}
        for tick, value in zip(ticks, values)
    ]


def manual_pixels_to_calibration(axis_name, pixels, values):
    if axis_name == "x":
        ordered = sorted(int(round(pixel)) for pixel in pixels)
    else:
        ordered = sorted((int(round(pixel)) for pixel in pixels), reverse=True)
    return [
        {"pixel": int(pixel), "value": float(value)}
        for pixel, value in zip(ordered, values[: len(ordered)])
    ]


def click_to_image_pixel(click, image):
    if not click:
        return None
    display_width = float(click.get("width") or image.shape[1])
    display_height = float(click.get("height") or image.shape[0])
    scale_x = image.shape[1] / display_width
    scale_y = image.shape[0] / display_height
    return {
        "x": int(round(float(click["x"]) * scale_x)),
        "y": int(round(float(click["y"]) * scale_y)),
        "unix_time": click.get("unix_time"),
    }


def detect_curve_colors(image, plot_area, max_colors=8):
    margin = 6
    roi = image[
        plot_area["top"] + margin : plot_area["bottom"] - margin,
        plot_area["left"] + margin : plot_area["right"] - margin,
    ]
    if roi.size == 0:
        return []

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    color_mask = (sat > 45) & (val > 40) & (val < 245)
    pixels = rgb[color_mask]
    candidates = []

    if len(pixels) > 0:
        sample = pixels
        if len(sample) > 20000:
            rng = np.random.default_rng(7)
            sample = sample[rng.choice(len(sample), size=20000, replace=False)]
        k = max(1, min(int(max_colors), len(sample), 8))
        data = np.float32(sample)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        labels = labels.flatten()
        for idx, center in enumerate(centers):
            count = int(np.sum(labels == idx))
            if count < 30:
                continue
            center = np.clip(center, 0, 255).astype(int).tolist()
            candidates.append((count, center))

    dark_mask = (sat < 80) & (val < 110)
    dark_pixels = rgb[dark_mask]
    if len(dark_pixels) > 80:
        center = np.median(dark_pixels, axis=0).astype(int).tolist()
        candidates.append((int(len(dark_pixels)), center))

    deduped = []
    for count, center in sorted(candidates, key=lambda item: item[0], reverse=True):
        if all(np.linalg.norm(np.array(center) - np.array(prev)) > 35 for _, prev in deduped):
            deduped.append((count, center))
        if len(deduped) >= int(max_colors):
            break

    series = []
    for idx, (count, center) in enumerate(deduped, start=1):
        is_dark = max(center) < 120 and (max(center) - min(center) < 45)
        item = {
            "selected": True,
            "name": f"color_{idx}",
            "rgb": center,
            "tolerance": 30 if is_dark else 38,
            "min_area": 20,
            "min_x_span": 20,
        }
        if not is_dark:
            item["saturation_min"] = 40
        else:
            item["keep_largest_components"] = 1
        series.append(item)
    return series


def calibration_to_frame(axis_cfg):
    calibration = axis_cfg.get("calibration", [])
    return pd.DataFrame(
        [
            {
                "value": "" if item.get("value") is None else item.get("value"),
                "pixel": item.get("pixel", ""),
            }
            for item in calibration
        ],
        columns=["value", "pixel"],
    )


def frame_to_calibration(frame):
    calibration = []
    for _, row in frame.fillna("").iterrows():
        pixel = row.get("pixel")
        if pixel in ("", None):
            continue
        value = row.get("value")
        calibration.append(
            {
                "pixel": int(round(float(pixel))),
                "value": None if value in ("", None) else float(value),
            }
        )
    return calibration


def make_default_config(image_path, x_min, x_max, y_min, y_max, x_step, series_frame):
    image = ld.load_image(image_path)
    plot_area, plot_detection = detect_plot_area_candidates(image, x_tick_count=6, y_tick_count=7)
    return {
        "plot_area": plot_area,
        "plot_area_detection": plot_detection,
        "plot_area_verification": {"verified": False},
        "axes": {
            "x": {
                "min": float(x_min),
                "max": float(x_max),
                "scale": "linear",
                "pixel_min": plot_area["left"],
                "pixel_max": plot_area["right"],
            },
            "y": {
                "min": float(y_min),
                "max": float(y_max),
                "scale": "linear",
                "pixel_min": plot_area["bottom"],
                "pixel_max": plot_area["top"],
            },
        },
        "border_margin_px": 2,
        "ignore_regions": [],
        "series": frame_to_series(series_frame),
        "output": {"x_step": None if not x_step else float(x_step), "max_interpolate_gap_px": 8},
    }


def load_or_create_config(image_path, paths, x_min, x_max, y_min, y_max, x_step, series_frame):
    if paths["config"].exists():
        with open(paths["config"], "r", encoding="utf-8") as handle:
            return json.load(handle)
    return make_default_config(image_path, x_min, x_max, y_min, y_max, x_step, series_frame)


def save_config(config, path):
    ld.ensure_parent(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


def unique_ints(values):
    out = []
    for value in values:
        value = int(round(float(value)))
        if value not in out:
            out.append(value)
    return out


def tick_target_error(ticks, count, plot_area, axis_name):
    count = int(count)
    selected = select_ticks_by_count(ticks, count, plot_area, axis_name)
    if len(selected) < count:
        return 10000.0 + 1000.0 * (count - len(selected))
    if axis_name == "x":
        targets = np.linspace(plot_area["left"], plot_area["right"], count)
    else:
        targets = np.linspace(plot_area["bottom"], plot_area["top"], count)
    pixels = np.array([float(item["pixel"]) for item in selected], dtype=np.float64)
    return float(np.mean(np.abs(pixels - targets)))


def detect_ticks_best(image, plot_area, params, x_tick_count, y_tick_count):
    base_search = int(params["search_px"])
    base_dark = int(params["dark_threshold"])
    base_min_len = int(params["min_tick_len"])
    base_cluster = int(params["max_cluster_width"])
    search_values = unique_ints([base_search, base_search + 8, max(4, base_search - 6), 24])
    dark_values = unique_ints([base_dark, min(245, base_dark + 30), max(30, base_dark - 30), 140, 210])
    min_len_values = unique_ints([base_min_len, max(1, base_min_len - 2), base_min_len + 3, 3])
    cluster_values = unique_ints([base_cluster, max(base_cluster + 6, 18), 28])

    best = None
    for search_px in search_values:
        for dark_threshold in dark_values:
            for min_tick_len in min_len_values:
                for max_cluster_width in cluster_values:
                    ticks = ld.auto_detect_ticks(
                        image,
                        plot_area,
                        search_px=search_px,
                        dark_threshold=dark_threshold,
                        min_tick_len=min_tick_len,
                        max_cluster_width=max_cluster_width,
                    )
                    x_error = tick_target_error(ticks.get("x", []), x_tick_count, plot_area, "x")
                    y_error = tick_target_error(ticks.get("y", []), y_tick_count, plot_area, "y")
                    count_penalty = 0.03 * (
                        abs(len(ticks.get("x", [])) - int(x_tick_count))
                        + abs(len(ticks.get("y", [])) - int(y_tick_count))
                    )
                    score = x_error + y_error + count_penalty
                    candidate = (
                        score,
                        ticks,
                        {
                            "search_px": search_px,
                            "dark_threshold": dark_threshold,
                            "min_tick_len": min_tick_len,
                            "max_cluster_width": max_cluster_width,
                        },
                    )
                    if best is None or candidate[0] < best[0]:
                        best = candidate
                    if score <= 2.0:
                        return candidate
    return best


def detect_ticks_for_config(image_path, config, paths, params, x_tick_count=None, y_tick_count=None):
    image = ld.load_image(image_path)
    plot_area = ld.normalize_plot_area(config["plot_area"], image)
    best = None
    if x_tick_count and y_tick_count:
        best = detect_ticks_best(image, plot_area, params, x_tick_count, y_tick_count)
    if best is None:
        ticks = ld.auto_detect_ticks(
            image,
            plot_area,
            search_px=params["search_px"],
            dark_threshold=params["dark_threshold"],
            min_tick_len=params["min_tick_len"],
            max_cluster_width=params["max_cluster_width"],
        )
        used_params = dict(params)
        score = None
    else:
        score, ticks, used_params = best
    config["plot_area"] = plot_area
    config["detected_ticks"] = ticks
    config["tick_detection"] = {
        "params": used_params,
        "score": score,
        "x_candidates": len(ticks.get("x", [])),
        "y_candidates": len(ticks.get("y", [])),
    }
    config.setdefault("axes", {})
    config["axes"].setdefault("x", {"scale": "linear"})
    config["axes"].setdefault("y", {"scale": "linear"})
    config["axes"]["x"]["calibration"] = ld.merge_calibration_template(
        config["axes"]["x"].get("calibration"), ticks["x"]
    )
    config["axes"]["y"]["calibration"] = ld.merge_calibration_template(
        config["axes"]["y"].get("calibration"), ticks["y"]
    )
    ld.draw_tick_preview(image, plot_area, ticks, paths["ticks"])
    return config


def group_label_rows(rows, tolerance=8):
    if not rows:
        return []
    rows = sorted(rows, key=lambda item: item["pixel"])
    groups = [[rows[0]]]
    for row in rows[1:]:
        if abs(float(row["pixel"]) - float(groups[-1][-1]["pixel"])) <= float(tolerance):
            groups[-1].append(row)
        else:
            groups.append([row])

    merged = []
    for group in groups:
        weights = np.array([max(1.0, float(item.get("area", 1))) for item in group], dtype=np.float64)
        pixels = np.array([float(item["pixel"]) for item in group], dtype=np.float64)
        merged.append(
            {
                "pixel": int(round(np.average(pixels, weights=weights))),
                "strength": int(np.sum(weights)),
                "width_px": int(max(item.get("width_px", 1) for item in group)),
                "source": "label",
            }
        )
    return merged


def detect_y_label_rows(image, plot_area, expected_count, search_px=72, dark_threshold=180):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    left = int(plot_area["left"])
    top = int(plot_area["top"])
    bottom = int(plot_area["bottom"])
    x1 = max(0, left - int(search_px))
    x2 = max(x1 + 1, min(width, left - 3))
    y1 = max(0, top - 12)
    y2 = min(height, bottom + 18)
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return []

    dark = (roi <= int(dark_threshold)).astype(np.uint8) * 255
    kernel = np.ones((2, 3), np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=1)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark, connectivity=8)

    components = []
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < 8:
            continue
        if h < 5 or h > 34:
            continue
        if w < 3 or w > 42:
            continue
        # Avoid the vertical y-axis title by requiring the component to be close to the axis.
        if x + w < roi.shape[1] * 0.25:
            continue
        cy = float(centroids[label][1]) + y1
        components.append({"pixel": cy, "area": int(area), "width_px": int(w), "height_px": int(h)})

    rows = group_label_rows(components, tolerance=9)
    if len(rows) <= int(expected_count):
        return sorted(rows, key=lambda item: item["pixel"], reverse=True)

    # Select rows that best form the expected evenly spaced y-label sequence.
    best = None
    ordered = sorted(rows, key=lambda item: item["pixel"], reverse=True)
    for start_idx in range(len(ordered)):
        for end_idx in range(start_idx + 1, len(ordered)):
            start_pixel = float(ordered[start_idx]["pixel"])
            end_pixel = float(ordered[end_idx]["pixel"])
            span = abs(end_pixel - start_pixel)
            if span < (bottom - top) * 0.35:
                continue
            targets = np.linspace(start_pixel, end_pixel, int(expected_count))
            selected = []
            used = set()
            residuals = []
            for target in targets:
                candidates = []
                for idx, row in enumerate(ordered):
                    if idx in used:
                        continue
                    candidates.append((abs(float(row["pixel"]) - target), idx, row))
                if not candidates:
                    break
                dist, idx, row = min(candidates, key=lambda item: item[0])
                selected.append(row)
                used.add(idx)
                residuals.append(dist)
            if len(selected) != int(expected_count):
                continue
            score = float(np.mean(residuals)) - 0.002 * sum(float(item.get("strength", 1)) for item in selected)
            candidate = (score, selected)
            if best is None or candidate[0] < best[0]:
                best = candidate
    if best is not None:
        return sorted(best[1], key=lambda item: item["pixel"], reverse=True)
    return ordered[: int(expected_count)]


def tesseract_command():
    candidates = [shutil.which("tesseract")]
    current_env_binary = Path(sys.executable).with_name("tesseract")
    candidates.append(str(current_env_binary))
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


def axis_ocr_status():
    if pytesseract is None:
        return {
            "available": False,
            "message": "Python package pytesseract is not installed. Run: python -m pip install pytesseract",
        }
    command = tesseract_command()
    if command is None:
        return {
            "available": False,
            "message": (
                "Tesseract OCR is not installed. Run either: sudo apt install tesseract-ocr, "
                "or: conda install -p ./.venv -c conda-forge tesseract -y"
            ),
        }
    pytesseract.pytesseract.tesseract_cmd = command
    return {"available": True, "message": "", "command": command}


def axis_ocr_roi(image, plot_area, axis_name):
    height, width = image.shape[:2]
    left = int(plot_area["left"])
    right = int(plot_area["right"])
    top = int(plot_area["top"])
    bottom = int(plot_area["bottom"])
    if axis_name == "x":
        return {
            "x1": max(0, left - 20),
            "x2": min(width, right + 20),
            "y1": min(height - 1, bottom + 2),
            # Keep this tight so the x-axis title is not merged into the tick labels.
            "y2": min(height, bottom + 52),
        }
    return {
        "x1": max(0, left - 92),
        "x2": max(1, min(width, left - 2)),
        "y1": max(0, top - 24),
        "y2": min(height, bottom + 24),
    }


def preprocess_axis_ocr_roi(image, roi, scale=3):
    crop = image[roi["y1"] : roi["y2"], roi["x1"] : roi["x2"]]
    if crop.size == 0:
        return None, scale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float(np.mean(binary)) < 127.0:
        binary = cv2.bitwise_not(binary)
    binary = cv2.copyMakeBorder(binary, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
    return binary, scale


def parse_ocr_number(text):
    text = str(text or "").replace("−", "-").replace("–", "-").strip()
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if not matches:
        return None
    try:
        return float(matches[0])
    except ValueError:
        return None


def longest_increasing_axis_sequence(candidates, axis_name):
    if axis_name == "x":
        ordered = sorted(candidates, key=lambda item: item["center_x"])
    else:
        ordered = sorted(candidates, key=lambda item: item["center_y"], reverse=True)
    if not ordered:
        return []

    n = len(ordered)
    lengths = [1] * n
    prev = [-1] * n
    for idx in range(n):
        for prior in range(idx):
            if ordered[idx]["value"] > ordered[prior]["value"]:
                if lengths[prior] + 1 > lengths[idx]:
                    lengths[idx] = lengths[prior] + 1
                    prev[idx] = prior
    best_idx = max(range(n), key=lambda idx: (lengths[idx], ordered[idx]["confidence"]))
    keep = []
    while best_idx != -1:
        keep.append(ordered[best_idx])
        best_idx = prev[best_idx]
    return list(reversed(keep))


def spacing_residual_px(items, axis_name):
    if len(items) < 3:
        return 0.0
    pixels = np.array(
        [float(item["center_x"] if axis_name == "x" else item["center_y"]) for item in items],
        dtype=np.float64,
    )
    expected = np.linspace(float(pixels[0]), float(pixels[-1]), len(pixels))
    return float(np.max(np.abs(pixels - expected)))


def value_spacing_residual(items):
    if len(items) < 3:
        return 0.0, 0.0
    values = np.array([float(item["value"]) for item in items], dtype=np.float64)
    expected = np.linspace(float(values[0]), float(values[-1]), len(values))
    residual = float(np.max(np.abs(values - expected)))
    step = abs(float(values[-1] - values[0])) / max(1, len(values) - 1)
    ratio = residual / max(step, 1e-9)
    return residual, ratio


def infer_axis_range_from_ocr(candidates, axis_name):
    warnings = []
    selected = longest_increasing_axis_sequence(candidates, axis_name)
    if len(selected) < len(candidates):
        warnings.append(f"filtered {len(candidates) - len(selected)} non-monotonic OCR number(s)")
    if len(selected) < 2:
        warnings.append("need at least 2 tick labels")
        return {
            "status": "fail",
            "count": len(selected),
            "min": None,
            "max": None,
            "confidence": 0.0,
            "max_spacing_residual_px": None,
            "numbers": selected,
            "warnings": warnings,
        }

    residual = spacing_residual_px(selected, axis_name)
    if residual > 18.0:
        warnings.append(f"tick label spacing residual {residual:.1f}px > 18px")
    value_residual, value_residual_ratio = value_spacing_residual(selected)
    if value_residual_ratio > 0.35:
        warnings.append(f"tick value spacing residual {value_residual:g} is too large")
    confidences = [float(item.get("confidence", 0.0)) for item in selected]
    confidence = float(np.mean(confidences)) if confidences else 0.0
    if confidence < 45.0:
        warnings.append(f"low OCR confidence {confidence:.0f}")
    if value_residual_ratio > 1.0:
        status = "fail"
    else:
        status = "pass" if not warnings else "review"
    return {
        "status": status,
        "count": len(selected),
        "min": float(selected[0]["value"]),
        "max": float(selected[-1]["value"]),
        "confidence": round(confidence, 1),
        "max_spacing_residual_px": round(residual, 2),
        "value_spacing_residual": round(value_residual, 6),
        "numbers": selected,
        "warnings": warnings,
    }


def read_axis_numbers_with_tesseract(image, plot_area, axis_name):
    roi = axis_ocr_roi(image, plot_area, axis_name)
    read = read_numbers_from_roi(image, roi, psm=6)
    numbers = []
    for number in read["numbers"]:
        if axis_name == "y":
            # Ignore far-left rotated axis-title artifacts such as the "2" in cm2.
            if float(number["left"]) + float(number["width"]) < float(plot_area["left"]) - 68.0:
                continue
        numbers.append(number)
    read["numbers"] = numbers
    return read


def read_numbers_from_roi(image, roi, psm=6, scale=3):
    processed, scale = preprocess_axis_ocr_roi(image, roi, scale=scale)
    if processed is None:
        return {"raw_text": "", "numbers": [], "roi": roi, "warnings": ["empty OCR ROI"]}

    config = f"--psm {int(psm)} -c tessedit_char_whitelist=0123456789.-+"
    data = pytesseract.image_to_data(processed, config=config, output_type=pytesseract.Output.DICT)
    numbers = []
    raw_text = []
    for idx, text in enumerate(data.get("text", [])):
        if not str(text).strip():
            continue
        raw_text.append(str(text))
        value = parse_ocr_number(text)
        if value is None:
            continue
        try:
            confidence = float(data.get("conf", ["0"])[idx])
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < 0:
            confidence = 0.0
        left = (float(data["left"][idx]) - 12.0) / scale + roi["x1"]
        top = (float(data["top"][idx]) - 12.0) / scale + roi["y1"]
        width = float(data["width"][idx]) / scale
        height = float(data["height"][idx]) / scale
        center_x = left + width / 2.0
        center_y = top + height / 2.0

        numbers.append(
            {
                "text": str(text),
                "value": float(value),
                "confidence": round(confidence, 1),
                "left": round(left, 1),
                "top": round(top, 1),
                "width": round(width, 1),
                "height": round(height, 1),
                "center_x": round(center_x, 1),
                "center_y": round(center_y, 1),
            }
        )
    return {"raw_text": " ".join(raw_text), "numbers": numbers, "roi": roi, "warnings": []}


def auto_read_axis_ranges(image, plot_area):
    status = axis_ocr_status()
    result = {
        "available": bool(status["available"]),
        "message": status["message"],
        "x": {"status": "fail", "count": 0, "warnings": []},
        "y": {"status": "fail", "count": 0, "warnings": []},
        "warnings": [],
    }
    if not status["available"]:
        result["warnings"].append(status["message"])
        return result

    for axis_name in ("x", "y"):
        try:
            read = read_axis_numbers_with_tesseract(image, plot_area, axis_name)
            inferred = infer_axis_range_from_ocr(read["numbers"], axis_name)
            inferred["raw_text"] = read["raw_text"]
            inferred["roi"] = read["roi"]
            inferred["all_numbers"] = read["numbers"]
            inferred["warnings"] = list(read.get("warnings", [])) + list(inferred.get("warnings", []))
            result[axis_name] = inferred
        except Exception as exc:
            result[axis_name] = {
                "status": "fail",
                "count": 0,
                "min": None,
                "max": None,
                "confidence": 0.0,
                "raw_text": "",
                "numbers": [],
                "warnings": [str(exc)],
            }
    result["warnings"] = [
        f"{axis}: {warning}"
        for axis in ("x", "y")
        for warning in result.get(axis, {}).get("warnings", [])
    ]
    return result


def axis_ocr_suggestion_frame(axis_ocr):
    rows = []
    for axis_name in ("x", "y"):
        axis = axis_ocr.get(axis_name, {}) if axis_ocr else {}
        rows.append(
            {
                "axis": axis_name,
                "status": axis.get("status", "fail"),
                "min": axis.get("min"),
                "max": axis.get("max"),
                "tick_count": axis.get("count", 0),
                "confidence": axis.get("confidence", 0),
                "value_residual": axis.get("value_spacing_residual"),
                "warnings": "; ".join(axis.get("warnings", [])),
            }
        )
    return pd.DataFrame(rows)


def finite_plot_area(plot_area):
    try:
        return {
            "left": int(round(float(plot_area["left"]))),
            "right": int(round(float(plot_area["right"]))),
            "top": int(round(float(plot_area["top"]))),
            "bottom": int(round(float(plot_area["bottom"]))),
        }
    except (KeyError, TypeError, ValueError):
        return None


def plot_area_signature(plot_area):
    plot_area = finite_plot_area(plot_area)
    if not plot_area:
        return None
    return (plot_area["left"], plot_area["right"], plot_area["top"], plot_area["bottom"])


def add_plot_area_candidate(candidates, image, plot_area, source, details=None):
    plot_area = finite_plot_area(plot_area)
    if not plot_area:
        return
    plot_area = ld.normalize_plot_area(plot_area, image)
    height, width = image.shape[:2]
    if plot_area["right"] - plot_area["left"] < width * 0.18:
        return
    if plot_area["bottom"] - plot_area["top"] < height * 0.18:
        return
    signature = plot_area_signature(plot_area)
    if signature in {plot_area_signature(item["plot_area"]) for item in candidates}:
        return
    candidates.append({"plot_area": plot_area, "source": source, "details": details or {}})


def filtered_ocr_numbers(numbers, axis_name):
    filtered = []
    for item in numbers:
        width = float(item.get("width", 0))
        height = float(item.get("height", 0))
        value = item.get("value")
        if value is None:
            continue
        if height < 5 or height > 46:
            continue
        if width < 2 or width > (95 if axis_name == "y" else 90):
            continue
        filtered.append(item)
    return filtered


def group_numbers_by_row(numbers, tolerance=12):
    if not numbers:
        return []
    ordered = sorted(numbers, key=lambda item: float(item["center_y"]))
    groups = [[ordered[0]]]
    for item in ordered[1:]:
        if abs(float(item["center_y"]) - float(groups[-1][-1]["center_y"])) <= tolerance:
            groups[-1].append(item)
        else:
            groups.append([item])
    return groups


def label_sequence_score(items, axis_name, expected_count=None):
    if len(items) < 2:
        return 100000.0
    pixel_residual = spacing_residual_px(items, axis_name)
    _, value_ratio = value_spacing_residual(items)
    mean_conf = np.mean([max(0.0, float(item.get("confidence", 0.0))) for item in items])
    count_penalty = 0.0
    if expected_count:
        count_penalty = 30.0 * abs(len(items) - int(expected_count))
    return count_penalty + pixel_residual + value_ratio * 35.0 - mean_conf * 0.04 - len(items) * 3.0


def best_x_label_sequence(numbers, expected_count=None):
    best = None
    for group in group_numbers_by_row(filtered_ocr_numbers(numbers, "x"), tolerance=11):
        if len(group) < 2:
            continue
        selected = longest_increasing_axis_sequence(group, "x")
        if len(selected) < 2:
            continue
        selected = sorted(selected, key=lambda item: float(item["center_x"]))
        score = label_sequence_score(selected, "x", expected_count)
        candidate = (score, selected)
        if best is None or candidate[0] < best[0]:
            best = candidate
    return [] if best is None else best[1]


def best_y_label_sequence(numbers, expected_count=None):
    selected = longest_increasing_axis_sequence(filtered_ocr_numbers(numbers, "y"), "y")
    if len(selected) < 2:
        return []
    if expected_count and len(selected) > int(expected_count):
        # Keep the best evenly spaced subsequence with the requested length.
        ordered = sorted(selected, key=lambda item: float(item["center_y"]), reverse=True)
        best = None
        count = int(expected_count)
        for start in range(0, len(ordered) - count + 1):
            subset = ordered[start : start + count]
            score = label_sequence_score(subset, "y", expected_count)
            candidate = (score, subset)
            if best is None or candidate[0] < best[0]:
                best = candidate
        selected = best[1] if best else ordered[:count]
    return sorted(selected, key=lambda item: float(item["center_y"]), reverse=True)


def global_tick_label_rois(image):
    height, width = image.shape[:2]
    return {
        "x": {
            "x1": 0,
            "x2": width,
            "y1": int(height * 0.55),
            "y2": min(height, int(height * 0.93)),
        },
        "y": {
            "x1": 0,
            "x2": min(width, max(90, int(width * 0.16))),
            "y1": 0,
            "y2": min(height, int(height * 0.88)),
        },
    }


def detect_global_tick_labels(image, x_tick_count=None, y_tick_count=None):
    status = axis_ocr_status()
    result = {
        "available": bool(status.get("available")),
        "x": {"status": "fail", "numbers": [], "all_numbers": [], "warnings": []},
        "y": {"status": "fail", "numbers": [], "all_numbers": [], "warnings": []},
        "warnings": [],
    }
    if not status.get("available"):
        result["warnings"].append(status.get("message", "OCR is not available."))
        return result

    rois = global_tick_label_rois(image)
    try:
        x_read = read_numbers_from_roi(image, rois["x"], psm=6)
        x_numbers = best_x_label_sequence(x_read["numbers"], x_tick_count)
        x_inferred = infer_axis_range_from_ocr(x_numbers, "x")
        x_inferred["all_numbers"] = x_read["numbers"]
        x_inferred["roi"] = rois["x"]
        result["x"] = x_inferred
    except Exception as exc:
        result["x"]["warnings"].append(str(exc))

    try:
        y_read = read_numbers_from_roi(image, rois["y"], psm=6)
        y_numbers = best_y_label_sequence(y_read["numbers"], y_tick_count)
        y_inferred = infer_axis_range_from_ocr(y_numbers, "y")
        y_inferred["all_numbers"] = y_read["numbers"]
        y_inferred["roi"] = rois["y"]
        result["y"] = y_inferred
    except Exception as exc:
        result["y"]["warnings"].append(str(exc))

    for axis_name, expected_count in (("x", x_tick_count), ("y", y_tick_count)):
        axis = result[axis_name]
        if expected_count and axis.get("count", 0) != int(expected_count):
            axis.setdefault("warnings", []).append(
                f"{axis_name} label count {axis.get('count', 0)} != expected {int(expected_count)}"
            )
            if axis.get("status") == "pass":
                axis["status"] = "review"
    result["warnings"] = [
        f"{axis}: {warning}"
        for axis in ("x", "y")
        for warning in result.get(axis, {}).get("warnings", [])
    ]
    return result


def line_strengths(gray, axis_name, dark_threshold=185):
    dark = (gray <= int(dark_threshold)).astype(np.uint8)
    if axis_name == "horizontal":
        return np.sum(dark, axis=1).astype(np.float64)
    return np.sum(dark, axis=0).astype(np.float64)


def smooth_scores(scores, window=9):
    if len(scores) == 0:
        return scores
    kernel = np.ones(int(window), dtype=np.float64) / float(window)
    return np.convolve(scores, kernel, mode="same")


def best_line_in_range(gray, axis_name, start, end, span_start=None, span_end=None):
    height, width = gray.shape[:2]
    start = max(0, int(round(start)))
    end = min((height if axis_name == "horizontal" else width) - 1, int(round(end)))
    if end < start:
        return None
    if axis_name == "horizontal":
        x1 = max(0, int(round(span_start if span_start is not None else 0)))
        x2 = min(width, int(round(span_end if span_end is not None else width)))
        region = gray[start : end + 1, x1:x2]
        if region.size == 0:
            return None
        scores = smooth_scores(line_strengths(region, "horizontal"))
        return int(start + int(np.argmax(scores)))
    y1 = max(0, int(round(span_start if span_start is not None else 0)))
    y2 = min(height, int(round(span_end if span_end is not None else height)))
    region = gray[y1:y2, start : end + 1]
    if region.size == 0:
        return None
    scores = smooth_scores(line_strengths(region, "vertical"))
    return int(start + int(np.argmax(scores)))


def projection_plot_area_candidate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    height, width = gray.shape[:2]
    row_scores = smooth_scores(line_strengths(gray, "horizontal"))
    col_scores = smooth_scores(line_strengths(gray, "vertical"))
    row_threshold = np.percentile(row_scores, 92)
    col_threshold = np.percentile(col_scores, 92)
    rows = [idx for idx, score in enumerate(row_scores) if score >= row_threshold]
    cols = [idx for idx, score in enumerate(col_scores) if score >= col_threshold]
    if len(rows) < 2 or len(cols) < 2:
        return None

    top_candidates = [row for row in rows if row <= height * 0.35]
    bottom_candidates = [row for row in rows if row >= height * 0.45]
    left_candidates = [col for col in cols if width * 0.04 <= col <= width * 0.35]
    right_candidates = [col for col in cols if width * 0.45 <= col <= width * 0.98]
    if not top_candidates or not bottom_candidates or not left_candidates or not right_candidates:
        return None
    top = max(top_candidates, key=lambda row: row_scores[row])
    bottom = max(bottom_candidates, key=lambda row: row_scores[row])
    left = max(left_candidates, key=lambda col: col_scores[col])
    right = max(right_candidates, key=lambda col: col_scores[col])
    if bottom <= top or right <= left:
        return None
    return {"left": left, "right": right, "top": top, "bottom": bottom}


def contour_plot_area_candidate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = gray.shape[:2]
    best = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < width * 0.25 or h < height * 0.25:
            continue
        if x < 2 or y < 2 or x + w > width - 2 or y + h > height - 2:
            continue
        area_ratio = (w * h) / float(width * height)
        if area_ratio < 0.15 or area_ratio > 0.9:
            continue
        score = w + h
        if best is None or score > best[0]:
            best = (score, {"left": x, "right": x + w, "top": y, "bottom": y + h})
    return None if best is None else best[1]


def label_first_plot_area_candidate(image, labels):
    x_numbers = labels.get("x", {}).get("numbers", [])
    y_numbers = labels.get("y", {}).get("numbers", [])
    if len(x_numbers) < 2 or len(y_numbers) < 2:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    x_numbers = sorted(x_numbers, key=lambda item: float(item["center_x"]))
    y_numbers = sorted(y_numbers, key=lambda item: float(item["center_y"]), reverse=True)

    left_est = float(x_numbers[0]["center_x"])
    right_est = float(x_numbers[-1]["center_x"])
    x_label_top = min(float(item["top"]) for item in x_numbers)
    bottom_est = x_label_top - 15.0
    top_label_center = float(y_numbers[-1]["center_y"])
    bottom_label_center = float(y_numbers[0]["center_y"])

    x_span_left = max(0, left_est - 5)
    x_span_right = min(width, right_est + 5)
    bottom = best_line_in_range(gray, "horizontal", x_label_top - 42, x_label_top - 3, x_span_left, x_span_right)
    top = best_line_in_range(gray, "horizontal", top_label_center - 35, top_label_center + 8, x_span_left, x_span_right)
    left = best_line_in_range(gray, "vertical", left_est - 25, left_est + 25, top or 0, bottom or height)
    right = best_line_in_range(gray, "vertical", right_est - 25, right_est + 25, top or 0, bottom or height)

    candidate = {
        "left": int(round(left if left is not None else left_est)),
        "right": int(round(right if right is not None else right_est)),
        "top": int(round(top if top is not None else top_label_center)),
        "bottom": int(round(bottom if bottom is not None else max(bottom_est, bottom_label_center))),
    }
    return candidate


def border_strength(image, plot_area):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark = gray <= 185
    left, right, top, bottom = plot_area["left"], plot_area["right"], plot_area["top"], plot_area["bottom"]
    h_band = max(1, right - left + 1)
    v_band = max(1, bottom - top + 1)
    scores = [
        float(np.mean(dark[max(0, top - 1) : min(gray.shape[0], top + 2), left : right + 1])),
        float(np.mean(dark[max(0, bottom - 1) : min(gray.shape[0], bottom + 2), left : right + 1])),
        float(np.mean(dark[top : bottom + 1, max(0, left - 1) : min(gray.shape[1], left + 2)])),
        float(np.mean(dark[top : bottom + 1, max(0, right - 1) : min(gray.shape[1], right + 2)])),
    ]
    return 100.0 * float(np.mean([score for score in scores if not np.isnan(score)]))


def label_alignment_score(plot_area, labels, x_tick_count=None, y_tick_count=None):
    score = 0.0
    warnings = []
    x_numbers = labels.get("x", {}).get("numbers", [])
    y_numbers = labels.get("y", {}).get("numbers", [])
    if x_numbers:
        if x_tick_count and len(x_numbers) == int(x_tick_count):
            score += 12.0
        else:
            warnings.append(f"x label count {len(x_numbers)}")
        x_numbers = sorted(x_numbers, key=lambda item: float(item["center_x"]))
        x_edge_error = abs(float(x_numbers[0]["center_x"]) - plot_area["left"]) + abs(
            float(x_numbers[-1]["center_x"]) - plot_area["right"]
        )
        score += max(0.0, 16.0 - x_edge_error * 0.35)
        label_gap = min(float(item["top"]) for item in x_numbers) - plot_area["bottom"]
        if 2 <= label_gap <= 45:
            score += 12.0
        else:
            warnings.append(f"x labels gap {label_gap:.1f}px")
    else:
        warnings.append("no x OCR labels")

    if y_numbers:
        if y_tick_count and len(y_numbers) == int(y_tick_count):
            score += 12.0
        else:
            warnings.append(f"y label count {len(y_numbers)}")
        y_numbers = sorted(y_numbers, key=lambda item: float(item["center_y"]), reverse=True)
        y_edge_error = abs(float(y_numbers[0]["center_y"]) - plot_area["bottom"]) + abs(
            float(y_numbers[-1]["center_y"]) - plot_area["top"]
        )
        score += max(0.0, 14.0 - y_edge_error * 0.25)
        rightmost_label = max(float(item["left"]) + float(item["width"]) for item in y_numbers)
        label_gap = plot_area["left"] - rightmost_label
        if -8 <= label_gap <= 45:
            score += 12.0
        else:
            warnings.append(f"y labels gap {label_gap:.1f}px")
    else:
        warnings.append("no y OCR labels")
    return score, warnings


def tick_consistency_score(image, plot_area, tick_params, x_tick_count=None, y_tick_count=None):
    if not (x_tick_count and y_tick_count):
        return 0.0, []
    params = tick_params or {"search_px": 18, "dark_threshold": 170, "min_tick_len": 5, "max_cluster_width": 14}
    ticks = ld.auto_detect_ticks(
        image,
        plot_area,
        search_px=params.get("search_px", 18),
        dark_threshold=params.get("dark_threshold", 170),
        min_tick_len=params.get("min_tick_len", 5),
        max_cluster_width=params.get("max_cluster_width", 14),
    )
    x_error = tick_target_error(ticks.get("x", []), int(x_tick_count), plot_area, "x")
    y_error = tick_target_error(ticks.get("y", []), int(y_tick_count), plot_area, "y")
    score = max(0.0, 22.0 - min(22.0, (x_error + y_error) * 0.9))
    warnings = []
    if len(ticks.get("x", [])) < int(x_tick_count):
        warnings.append(f"x tick candidates {len(ticks.get('x', []))}/{int(x_tick_count)}")
    if len(ticks.get("y", [])) < int(y_tick_count):
        warnings.append(f"y tick candidates {len(ticks.get('y', []))}/{int(y_tick_count)}")
    return score, warnings


def score_plot_area_candidate(image, candidate, labels, tick_params=None, x_tick_count=None, y_tick_count=None):
    plot_area = candidate["plot_area"]
    height, width = image.shape[:2]
    warnings = []
    score = 20.0
    area_width = plot_area["right"] - plot_area["left"]
    area_height = plot_area["bottom"] - plot_area["top"]
    width_ratio = area_width / max(1.0, float(width))
    height_ratio = area_height / max(1.0, float(height))
    if 0.45 <= width_ratio <= 0.94:
        score += 10.0
    else:
        warnings.append(f"width ratio {width_ratio:.2f}")
    if 0.45 <= height_ratio <= 0.92:
        score += 10.0
    else:
        warnings.append(f"height ratio {height_ratio:.2f}")
    border_score = border_strength(image, plot_area)
    score += min(22.0, border_score * 0.32)
    if border_score < 20.0:
        warnings.append(f"weak border {border_score:.1f}")
    label_score, label_warnings = label_alignment_score(plot_area, labels, x_tick_count, y_tick_count)
    score += label_score
    warnings.extend(label_warnings)
    tick_score, tick_warnings = tick_consistency_score(image, plot_area, tick_params, x_tick_count, y_tick_count)
    score += tick_score
    warnings.extend(tick_warnings)
    if candidate["source"] == "tick-label":
        score += 8.0
    score = round(max(0.0, min(100.0, score)), 1)
    status = "pass" if score >= 80.0 else "review" if score >= 55.0 else "fail"
    return {
        "plot_area": plot_area,
        "source": candidate["source"],
        "score": score,
        "status": status,
        "warnings": warnings,
        "details": candidate.get("details", {}),
    }


def detect_plot_area_candidates(image, x_tick_count=None, y_tick_count=None, tick_params=None):
    candidates = []
    labels = detect_global_tick_labels(image, x_tick_count, y_tick_count)
    add_plot_area_candidate(candidates, image, ld.auto_detect_plot_area(image), "hough")
    add_plot_area_candidate(candidates, image, projection_plot_area_candidate(image), "projection")
    add_plot_area_candidate(candidates, image, contour_plot_area_candidate(image), "contour")
    add_plot_area_candidate(candidates, image, label_first_plot_area_candidate(image, labels), "tick-label")
    if not candidates:
        add_plot_area_candidate(candidates, image, ld.default_plot_area(image), "default")

    scored = [
        score_plot_area_candidate(image, candidate, labels, tick_params, x_tick_count, y_tick_count)
        for candidate in candidates
    ]
    scored = sorted(scored, key=lambda item: float(item["score"]), reverse=True)
    best = scored[0]
    detection = {
        "status": best["status"],
        "score": best["score"],
        "source": best["source"],
        "warnings": best["warnings"],
        "candidates": scored,
        "x_labels": labels.get("x", {}),
        "y_labels": labels.get("y", {}),
        "ocr_available": labels.get("available", False),
    }
    return best["plot_area"], detection


def apply_y_label_assisted_calibration(image, config, y_tick_count):
    label_rows = detect_y_label_rows(image, config["plot_area"], int(y_tick_count))
    config["y_label_rows"] = label_rows
    if len(label_rows) == int(y_tick_count):
        y_values = axis_values(config["axes"]["y"]["min"], config["axes"]["y"]["max"], int(y_tick_count))
        config["axes"]["y"]["calibration"] = ticks_to_calibration(label_rows, y_values)
        config.setdefault("tick_detection", {})["y_label_assisted"] = True
        config["tick_detection"]["y_label_rows"] = len(label_rows)
    else:
        config.setdefault("tick_detection", {})["y_label_assisted"] = False
        config["tick_detection"]["y_label_rows"] = len(label_rows)
    return config


def assign_evenly_spaced_tick_values(config, x_tick_count, y_tick_count):
    plot_area = config["plot_area"]
    clear_calibration_verification(config)
    x_values = axis_values(config["axes"]["x"]["min"], config["axes"]["x"]["max"], x_tick_count)
    y_values = axis_values(config["axes"]["y"]["min"], config["axes"]["y"]["max"], y_tick_count)
    x_pixels = np.linspace(plot_area["left"], plot_area["right"], len(x_values))
    y_pixels = np.linspace(plot_area["bottom"], plot_area["top"], len(y_values))
    config["axes"]["x"]["calibration"] = [
        {"pixel": int(round(pixel)), "value": float(value)}
        for pixel, value in zip(x_pixels, x_values)
    ]
    config["axes"]["y"]["calibration"] = [
        {"pixel": int(round(pixel)), "value": float(value)}
        for pixel, value in zip(y_pixels, y_values)
    ]
    config["detected_ticks"] = calibration_ticks_for_preview(config)
    return config


def assign_tick_values_from_axis_settings(config, x_tick_count, y_tick_count):
    plot_area = config["plot_area"]
    clear_calibration_verification(config)
    detected = config.get("detected_ticks", {"x": [], "y": []})
    x_values = axis_values(config["axes"]["x"]["min"], config["axes"]["x"]["max"], x_tick_count)
    y_values = axis_values(config["axes"]["y"]["min"], config["axes"]["y"]["max"], y_tick_count)
    x_ticks = select_ticks_by_count(detected.get("x", []), len(x_values), plot_area, "x")
    y_ticks = select_ticks_by_count(detected.get("y", []), len(y_values), plot_area, "y")
    config["axes"]["x"]["calibration"] = ticks_to_calibration(x_ticks, x_values[: len(x_ticks)])
    config["axes"]["y"]["calibration"] = ticks_to_calibration(y_ticks, y_values[: len(y_ticks)])
    return config


def refresh_calibration_qc(config, x_tick_count, y_tick_count):
    qc = compute_calibration_qc(config, x_tick_count, y_tick_count)
    qc["y_label_rows"] = config.get("y_label_rows", [])
    qc["y_label_assisted"] = bool(config.get("tick_detection", {}).get("y_label_assisted"))
    if qc["y_label_assisted"]:
        qc.setdefault("warnings", []).append("y: calibrated from detected tick label rows")
    verification = config.get("calibration_verification", {})
    plot_verification = config.get("plot_area_verification", {})
    qc["auto_status"] = qc.get("status", "fail")
    qc["verified"] = bool(verification.get("verified"))
    qc["verified_at"] = verification.get("verified_at")
    qc["verified_note"] = verification.get("note", "")
    qc["plot_area_verified"] = bool(plot_verification.get("verified"))
    qc["plot_area_verified_at"] = plot_verification.get("verified_at")
    qc["plot_area_status"] = config.get("plot_area_detection", {}).get("status", "")
    qc["plot_area_score"] = config.get("plot_area_detection", {}).get("score", "")
    qc["usable"] = bool(qc["verified"] and qc["plot_area_verified"])
    if qc["usable"]:
        qc["status"] = "verified"
    elif qc["auto_status"] == "fail":
        qc["status"] = "fail"
    else:
        qc["status"] = "needs_verification"
    config["calibration_qc"] = qc
    return qc


def clear_calibration_verification(config):
    config["calibration_verification"] = {"verified": False}
    if "calibration_qc" in config:
        config["calibration_qc"]["verified"] = False
        config["calibration_qc"]["usable"] = False
        config["calibration_qc"]["status"] = "needs_verification"
    return config


def clear_plot_area_verification(config):
    config["plot_area_verification"] = {"verified": False}
    clear_calibration_verification(config)
    return config


def is_plot_area_verified(config):
    return bool(config.get("plot_area_verification", {}).get("verified"))


def confirm_plot_area(config, method="user"):
    config["plot_area_verification"] = {
        "verified": True,
        "verified_at": datetime.now().isoformat(timespec="seconds"),
        "method": method,
    }
    return config


def calibrations_equal(left, right):
    return json.dumps(left or [], sort_keys=True) == json.dumps(right or [], sort_keys=True)


def draw_config_tick_preview(image, config, paths):
    selected_ticks = calibration_ticks_for_preview(config)
    detected_ticks = config.get("detected_ticks", {"x": [], "y": []})
    qc = config.get("calibration_qc")
    if qc:
        draw_qc_tick_preview(image, config["plot_area"], detected_ticks, selected_ticks, qc, paths["ticks"])
    else:
        ld.draw_tick_preview(image, config["plot_area"], selected_ticks, paths["ticks"])


def auto_calibrate_config(
    image_path,
    config,
    paths,
    tick_params,
    x_tick_count,
    y_tick_count,
    use_y_label_assist=True,
):
    image = ld.load_image(image_path)
    config["plot_area"] = ld.normalize_plot_area(config["plot_area"], image)
    clear_calibration_verification(config)
    config = detect_ticks_for_config(
        image_path,
        config,
        paths,
        tick_params,
        int(x_tick_count),
        int(y_tick_count),
    )
    config = assign_tick_values_from_axis_settings(config, int(x_tick_count), int(y_tick_count))
    if use_y_label_assist:
        config = apply_y_label_assisted_calibration(image, config, int(y_tick_count))
    else:
        config["y_label_rows"] = []
        config.setdefault("tick_detection", {})["y_label_assisted"] = False
        config["tick_detection"]["y_label_rows"] = 0
    refresh_calibration_qc(config, int(x_tick_count), int(y_tick_count))
    draw_config_tick_preview(image, config, paths)
    return config


def qc_summary_row(image_path, config, paths, point_count=None, error=None):
    qc = config.get("calibration_qc", {})
    plot_detection = config.get("plot_area_detection", {})
    plot_verification = config.get("plot_area_verification", {})
    x_qc = qc.get("x", {})
    y_qc = qc.get("y", {})
    max_residuals = [
        value
        for value in (x_qc.get("max_residual_px"), y_qc.get("max_residual_px"))
        if value is not None
    ]
    status = qc.get("status", "fail")
    if error:
        status = "fail"
    return {
        "image": Path(image_path).name,
        "status": status,
        "auto_status": qc.get("auto_status", qc.get("status", "fail")),
        "verified": bool(qc.get("verified", False)),
        "usable": bool(qc.get("usable", False)) and not bool(error),
        "verified_at": qc.get("verified_at", ""),
        "plot_area_verified": bool(plot_verification.get("verified", qc.get("plot_area_verified", False))),
        "plot_area_status": plot_detection.get("status", qc.get("plot_area_status", "")),
        "plot_area_score": plot_detection.get("score", qc.get("plot_area_score", "")),
        "plot_area_warnings": summarize_qc_warnings(plot_detection, limit=4),
        "y_label_assisted": bool(qc.get("y_label_assisted", False)),
        "score": qc.get("score", 0),
        "x_tick_count": x_qc.get("used_count", 0),
        "y_tick_count": y_qc.get("used_count", 0),
        "max_residual_px": "" if not max_residuals else round(max(max_residuals), 3),
        "warnings": error or summarize_qc_warnings(qc, limit=6),
        "points": "" if point_count is None else int(point_count),
        "csv_path": str(paths["csv"]),
        "overlay_path": str(paths["preview"]),
        "tick_overlay_path": str(paths["ticks"]),
        "config_path": str(paths["config"]),
    }


def write_batch_qc(output_root, rows):
    path = Path(output_root).expanduser() / "batch_qc.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "image",
        "status",
        "auto_status",
        "verified",
        "usable",
        "verified_at",
        "plot_area_verified",
        "plot_area_status",
        "plot_area_score",
        "plot_area_warnings",
        "y_label_assisted",
        "score",
        "x_tick_count",
        "y_tick_count",
        "max_residual_px",
        "warnings",
        "points",
        "csv_path",
        "overlay_path",
        "tick_overlay_path",
        "config_path",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    return path


def read_batch_qc(output_root):
    path = Path(output_root).expanduser() / "batch_qc.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def filter_images_by_qc(images, qc_frame, mode):
    if qc_frame.empty or mode == "All":
        return images
    rows_by_image = {
        str(row["image"]): row
        for _, row in qc_frame.fillna("").iterrows()
    }
    if mode == "Verified":
        return [path for path in images if str(rows_by_image.get(path.name, {}).get("usable", "")).lower() in ("true", "1")]
    if mode == "Auto failed":
        return [path for path in images if str(rows_by_image.get(path.name, {}).get("auto_status", "")).lower() == "fail"]
    if mode == "Plot area needs review":
        return [
            path
            for path in images
            if str(rows_by_image.get(path.name, {}).get("plot_area_verified", "")).lower() not in ("true", "1")
            or str(rows_by_image.get(path.name, {}).get("plot_area_status", "")).lower() in ("fail", "review")
        ]
    if mode == "Needs verification":
        return [
            path
            for path in images
            if str(rows_by_image.get(path.name, {}).get("usable", "")).lower() not in ("true", "1")
        ]
    return images


def ensure_detected_series_for_new_config(image, config, is_new_config):
    selected = [series for series in config.get("series", []) if is_truthy_selected(series.get("selected", True))]
    if is_new_config or not selected:
        detected = detect_curve_colors(image, config["plot_area"])
        if detected:
            config["series"] = detected
    return config


def process_image_auto(
    image_path,
    output_dir,
    x_min,
    x_max,
    y_min,
    y_max,
    x_step,
    x_tick_count,
    y_tick_count,
    tick_params,
    base_series_frame,
    use_y_label_assist=True,
):
    paths = output_paths(output_dir, image_path)
    is_new_config = not paths["config"].exists()
    config = load_or_create_config(image_path, paths, x_min, x_max, y_min, y_max, x_step, base_series_frame)
    image = ld.load_image(image_path)
    config["plot_area"] = ld.normalize_plot_area(config.get("plot_area") or ld.default_plot_area(image), image)
    if is_new_config or not config.get("plot_area_detection"):
        config["plot_area"], config["plot_area_detection"] = detect_plot_area_candidates(
            image,
            int(x_tick_count),
            int(y_tick_count),
            tick_params,
        )
        config["plot_area"] = ld.normalize_plot_area(config["plot_area"], image)
        config["plot_area_verification"] = {"verified": False}
    config.setdefault("plot_area_verification", {"verified": False})
    config = sync_axis_settings(config, x_min, x_max, y_min, y_max, x_step)
    config = ensure_detected_series_for_new_config(image, config, is_new_config)
    config = auto_calibrate_config(
        image_path,
        config,
        paths,
        tick_params,
        x_tick_count,
        y_tick_count,
        use_y_label_assist=use_y_label_assist,
    )
    save_config(config, paths["config"])
    rows = extract_to_outputs(image_path, config, paths)
    save_config(config, paths["config"])
    return qc_summary_row(image_path, config, paths, point_count=len(rows))


def calibration_ticks_for_preview(config):
    ticks = {"x": [], "y": []}
    for axis_name in ("x", "y"):
        axis_cfg = config.get("axes", {}).get(axis_name, {})
        for item in axis_cfg.get("calibration", []):
            pixel = item.get("pixel")
            if pixel in ("", None):
                continue
            ticks[axis_name].append({"pixel": int(round(float(pixel))), "strength": 1, "width_px": 1})
    ticks["x"] = sorted(ticks["x"], key=lambda item: item["pixel"])
    ticks["y"] = sorted(ticks["y"], key=lambda item: item["pixel"], reverse=True)
    return ticks


def ordered_calibration(axis_name, axis_cfg):
    calibration = [
        item
        for item in axis_cfg.get("calibration", [])
        if item.get("pixel") not in ("", None) and item.get("value") not in ("", None)
    ]
    reverse = axis_name == "y"
    return sorted(calibration, key=lambda item: float(item["pixel"]), reverse=reverse)


def nearest_tick_strength(pixel, candidates, tolerance=4):
    if not candidates:
        return None
    best = min(candidates, key=lambda item: abs(float(item["pixel"]) - float(pixel)))
    if abs(float(best["pixel"]) - float(pixel)) <= float(tolerance):
        return float(best.get("strength", 1))
    return None


def axis_calibration_quality(axis_name, axis_cfg, plot_area, detected_ticks, expected_count):
    expected_count = int(expected_count)
    calibration = ordered_calibration(axis_name, axis_cfg)
    pixels = np.array([float(item["pixel"]) for item in calibration], dtype=np.float64)
    values = np.array([float(item["value"]) for item in calibration], dtype=np.float64)
    candidates = detected_ticks.get(axis_name, []) if detected_ticks else []
    warnings = []
    score = 100.0

    result = {
        "expected_count": expected_count,
        "used_count": int(len(calibration)),
        "used_pixels": [int(round(pixel)) for pixel in pixels.tolist()],
        "max_residual_px": None,
        "max_spacing_residual_px": None,
        "coverage": None,
        "candidate_match_ratio": None,
        "mean_strength_ratio": None,
        "score": 0.0,
        "status": "fail",
        "warnings": warnings,
    }

    if len(calibration) < 2:
        warnings.append("not enough calibration ticks")
        return result

    if len(calibration) != expected_count:
        warnings.append(f"tick count {len(calibration)} != expected {expected_count}")
        score -= 35.0 + 18.0 * abs(len(calibration) - expected_count)

    if len(calibration) >= 3:
        expected_pixels = np.linspace(float(pixels[0]), float(pixels[-1]), len(pixels))
        spacing_residuals = np.abs(pixels - expected_pixels)
        max_spacing_residual = float(np.max(spacing_residuals))
    else:
        max_spacing_residual = 0.0
    result["max_spacing_residual_px"] = max_spacing_residual

    if axis_cfg.get("scale", "linear") == "log10":
        values_for_fit = np.log10(values)
    else:
        values_for_fit = values
    slope, intercept = np.polyfit(pixels, values_for_fit, 1)
    predicted = slope * pixels + intercept
    residuals = values_for_fit - predicted
    max_residual_px = float(np.max(np.abs(residuals / slope))) if slope != 0 else None
    result["max_residual_px"] = max_residual_px

    residual_for_score = max_spacing_residual
    if max_residual_px is not None:
        residual_for_score = max(residual_for_score, max_residual_px)
    if residual_for_score > 3.0:
        warnings.append(f"tick residual {residual_for_score:.1f}px > 3px")
    score -= min(28.0, residual_for_score * 4.0)

    if axis_name == "x":
        expected_span = max(1.0, float(plot_area["right"] - plot_area["left"]))
    else:
        expected_span = max(1.0, float(plot_area["bottom"] - plot_area["top"]))
    coverage = abs(float(pixels[-1] - pixels[0])) / expected_span
    result["coverage"] = float(coverage)
    if coverage < 0.8:
        warnings.append(f"tick span covers only {coverage:.0%} of plot area")
        score -= 18.0
    elif coverage < 0.9:
        warnings.append(f"tick span covers {coverage:.0%} of plot area")
        score -= 7.0

    if candidates:
        max_strength = max(1.0, max(float(item.get("strength", 1)) for item in candidates))
        matched_strengths = []
        for pixel in pixels:
            strength = nearest_tick_strength(pixel, candidates)
            if strength is not None:
                matched_strengths.append(strength)
        match_ratio = len(matched_strengths) / max(1, len(pixels))
        result["candidate_match_ratio"] = float(match_ratio)
        if matched_strengths:
            mean_strength_ratio = float(np.mean(matched_strengths) / max_strength)
            result["mean_strength_ratio"] = mean_strength_ratio
        else:
            mean_strength_ratio = 0.0
            result["mean_strength_ratio"] = 0.0
        if match_ratio < 0.75:
            warnings.append(f"only {match_ratio:.0%} of used ticks match detected candidates")
            score -= 10.0
        if mean_strength_ratio < 0.35:
            warnings.append("used ticks are weak candidates")
            score -= 6.0

    score = max(0.0, min(100.0, score))
    result["score"] = round(score, 1)
    if len(calibration) < expected_count:
        result["status"] = "fail"
    elif score >= 80.0:
        result["status"] = "pass"
    elif score >= 55.0:
        result["status"] = "review"
    else:
        result["status"] = "fail"
    return result


def compute_calibration_qc(config, x_tick_count, y_tick_count):
    axes = config.get("axes", {})
    plot_area = config.get("plot_area", {})
    detected = config.get("detected_ticks", {"x": [], "y": []})
    x_qc = axis_calibration_quality("x", axes.get("x", {}), plot_area, detected, x_tick_count)
    y_qc = axis_calibration_quality("y", axes.get("y", {}), plot_area, detected, y_tick_count)
    score = round(min(float(x_qc["score"]), float(y_qc["score"])), 1)
    warnings = []
    for axis_name, axis_qc in (("x", x_qc), ("y", y_qc)):
        for warning in axis_qc.get("warnings", []):
            warnings.append(f"{axis_name}: {warning}")
    if x_qc["status"] == "fail" or y_qc["status"] == "fail":
        status = "fail"
    elif score >= 80.0:
        status = "pass"
    elif score >= 55.0:
        status = "review"
    else:
        status = "fail"
    return {
        "status": status,
        "score": score,
        "warnings": warnings,
        "x": x_qc,
        "y": y_qc,
    }


def summarize_qc_warnings(qc, limit=3):
    warnings = qc.get("warnings", []) if qc else []
    if not warnings:
        return ""
    shown = warnings[:limit]
    suffix = "" if len(warnings) <= limit else f" (+{len(warnings) - limit} more)"
    return "; ".join(shown) + suffix


def build_qc_tick_preview_image(image, plot_area, detected_ticks, selected_ticks, qc):
    preview = image.copy()
    cv2.rectangle(
        preview,
        (plot_area["left"], plot_area["top"]),
        (plot_area["right"], plot_area["bottom"]),
        (0, 180, 255),
        2,
    )

    selected_x = {int(item["pixel"]) for item in selected_ticks.get("x", [])}
    selected_y = {int(item["pixel"]) for item in selected_ticks.get("y", [])}
    axis_status = {
        "x": qc.get("x", {}).get("status", "fail") if qc else "fail",
        "y": qc.get("y", {}).get("status", "fail") if qc else "fail",
    }

    for tick in detected_ticks.get("x", []):
        x = int(tick["pixel"])
        if x in selected_x:
            continue
        cv2.line(preview, (x, plot_area["bottom"] - 7), (x, plot_area["bottom"] + 7), (170, 170, 170), 1)
    for tick in detected_ticks.get("y", []):
        y = int(tick["pixel"])
        if y in selected_y:
            continue
        cv2.line(preview, (plot_area["left"] - 7, y), (plot_area["left"] + 7, y), (170, 170, 170), 1)

    for row in qc.get("y_label_rows", []) if qc else []:
        y = int(row["pixel"])
        cv2.line(preview, (max(0, plot_area["left"] - 55), y), (max(0, plot_area["left"] - 28), y), (255, 180, 0), 2)

    for idx, tick in enumerate(selected_ticks.get("x", []), start=1):
        x = int(tick["pixel"])
        color = (0, 170, 0) if axis_status["x"] != "fail" else (0, 0, 220)
        cv2.line(preview, (x, plot_area["bottom"] - 14), (x, plot_area["bottom"] + 14), color, 2)
        cv2.putText(preview, str(idx), (x + 3, plot_area["bottom"] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    for idx, tick in enumerate(selected_ticks.get("y", []), start=1):
        y = int(tick["pixel"])
        color = (0, 170, 0) if axis_status["y"] != "fail" else (0, 0, 220)
        cv2.line(preview, (plot_area["left"] - 14, y), (plot_area["left"] + 14, y), color, 2)
        cv2.putText(preview, str(idx), (max(0, plot_area["left"] - 35), y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    if qc:
        status = str(qc.get("status", "unknown")).upper()
        score = qc.get("score", "")
        label = f"{status} score={score}"
        warning_text = summarize_qc_warnings(qc, limit=2)
        if qc.get("status") == "verified":
            bg_color = (0, 130, 0)
        elif qc.get("status") == "needs_verification":
            bg_color = (0, 160, 220)
        else:
            bg_color = (0, 0, 220)
        box_w = min(360, image.shape[1] - 16)
        box_x1 = max(8, image.shape[1] - box_w - 8)
        box_x2 = image.shape[1] - 8
        cv2.rectangle(preview, (box_x1, 8), (box_x2, 58), bg_color, -1)
        cv2.putText(preview, label, (box_x1 + 8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        if warning_text:
            cv2.putText(preview, warning_text[:52], (box_x1 + 8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)

    return preview


def draw_qc_tick_preview(image, plot_area, detected_ticks, selected_ticks, qc, out_path):
    preview = build_qc_tick_preview_image(image, plot_area, detected_ticks, selected_ticks, qc)
    ld.ensure_parent(out_path)
    cv2.imwrite(str(out_path), preview)


def default_tick_count(axis_cfg, fallback):
    stored_count = axis_cfg.get("major_tick_count")
    if stored_count not in ("", None):
        try:
            stored_count = int(stored_count)
        except (TypeError, ValueError):
            stored_count = 0
        if stored_count >= 2:
            return stored_count
    count = len(axis_cfg.get("calibration", []) or [])
    if count >= 2:
        return count
    return fallback


def render_color_swatches(series_frame):
    pieces = []
    for _, row in series_frame.fillna("").iterrows():
        hex_value = str(row.get("hex", "")).strip()
        name = str(row.get("name", "")).strip() or "series"
        if not hex_value.startswith("#") or len(hex_value) != 7:
            continue
        opacity = "1" if is_truthy_selected(row.get("selected", True)) else "0.35"
        pieces.append(
            "<span style='display:inline-flex;align-items:center;margin:0 10px 8px 0;opacity:{};'>"
            "<span style='width:18px;height:18px;background:{};border:1px solid #666;display:inline-block;margin-right:6px;'></span>"
            "<span>{}</span></span>".format(opacity, hex_value, html.escape(name))
        )
    if pieces:
        st.markdown("".join(pieces), unsafe_allow_html=True)


def sync_axis_settings(config, x_min, x_max, y_min, y_max, x_step):
    config.setdefault("axes", {})
    config["axes"].setdefault("x", {"scale": "linear"})
    config["axes"].setdefault("y", {"scale": "linear"})
    config["axes"]["x"]["scale"] = "linear"
    config["axes"]["y"]["scale"] = "linear"
    config["axes"]["x"]["min"] = float(x_min)
    config["axes"]["x"]["max"] = float(x_max)
    config["axes"]["y"]["min"] = float(y_min)
    config["axes"]["y"]["max"] = float(y_max)
    config["axes"]["x"]["pixel_min"] = config["plot_area"]["left"]
    config["axes"]["x"]["pixel_max"] = config["plot_area"]["right"]
    config["axes"]["y"]["pixel_min"] = config["plot_area"]["bottom"]
    config["axes"]["y"]["pixel_max"] = config["plot_area"]["top"]
    config.setdefault("output", {})["x_step"] = float(x_step) if x_step else None
    return config


def apply_axis_range_suggestion(config, axis_name, suggestion):
    config.setdefault("axes", {})
    config["axes"].setdefault(axis_name, {"scale": "linear"})
    config["axes"][axis_name]["min"] = float(suggestion["min"])
    config["axes"][axis_name]["max"] = float(suggestion["max"])
    config["axes"][axis_name]["major_tick_count"] = int(suggestion["count"])
    clear_calibration_verification(config)
    return config


def render_manual_tick_picker(
    image,
    config,
    paths,
    axis_key,
    editor_version_key,
    overlay_mode_key,
    status_key,
    x_tick_count,
    y_tick_count,
):
    with st.expander("Advanced: click exact ticks"):
        mode = st.radio(
            "Axis",
            ["x axis", "y axis"],
            horizontal=True,
            key=f"manual_axis_{axis_key}",
        )
        axis_name = "x" if mode.startswith("x") else "y"
        expected_count = int(x_tick_count if axis_name == "x" else y_tick_count)
        pick_key = f"manual_{axis_name}_pixels_{axis_key}"
        click_time_key = f"manual_{axis_name}_click_time_{axis_key}"
        st.session_state.setdefault(pick_key, [])
        st.session_state.setdefault(click_time_key, None)

        click = streamlit_image_coordinates(
            bgr_to_rgb(image),
            use_column_width="always",
            key=f"manual_tick_picker_{axis_key}_{axis_name}_{st.session_state[editor_version_key]}",
        )
        image_pixel = click_to_image_pixel(click, image)
        if image_pixel and image_pixel.get("unix_time") != st.session_state[click_time_key]:
            st.session_state[click_time_key] = image_pixel.get("unix_time")
            if axis_name == "x":
                st.session_state[pick_key].append(int(image_pixel["x"]))
            else:
                st.session_state[pick_key].append(int(image_pixel["y"]))

        picked = st.session_state[pick_key]
        st.write(f"Picked {len(picked)} / {expected_count} {axis_name} ticks.")
        if picked:
            ordered = sorted(picked) if axis_name == "x" else sorted(picked, reverse=True)
            st.dataframe(
                pd.DataFrame(
                    {
                        "order": list(range(1, len(ordered) + 1)),
                        "pixel_ref": ordered,
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        manual_cols = st.columns(4)
        if manual_cols[0].button("Undo last pick", use_container_width=True, disabled=not picked):
            st.session_state[pick_key] = picked[:-1]
            st.rerun()
        if manual_cols[1].button("Clear current axis", use_container_width=True, disabled=not picked):
            st.session_state[pick_key] = []
            st.rerun()
        if manual_cols[2].button("Clear both axes", use_container_width=True):
            st.session_state[f"manual_x_pixels_{axis_key}"] = []
            st.session_state[f"manual_y_pixels_{axis_key}"] = []
            st.rerun()
        if manual_cols[3].button(f"Apply clicked {axis_name}", use_container_width=True):
            if len(picked) != expected_count:
                st.warning(f"{axis_name} needs exactly {expected_count} clicked major ticks.")
            else:
                values = axis_values(
                    config["axes"][axis_name]["min"],
                    config["axes"][axis_name]["max"],
                    expected_count,
                )
                clear_calibration_verification(config)
                config["axes"][axis_name]["calibration"] = manual_pixels_to_calibration(axis_name, picked, values)
                refresh_calibration_qc(config, int(x_tick_count), int(y_tick_count))
                draw_config_tick_preview(image, config, paths)
                save_config(config, paths["config"])
                st.session_state[editor_version_key] += 1
                request_overlay_mode(overlay_mode_key, "Ticks")
                st.session_state[status_key] = f"Applied manually clicked {axis_name} tick references."
                st.rerun()
    return config


def fill_values(frame, first_value, step_value):
    out = frame.copy()
    if "value" not in out.columns:
        out["value"] = ""
    for idx in range(len(out)):
        out.at[idx, "value"] = float(first_value) + float(step_value) * idx
    return out


def calibration_qc(axis_name, axis_cfg):
    calibration = [
        item
        for item in axis_cfg.get("calibration", [])
        if item.get("pixel") not in (None, "") and item.get("value") not in (None, "")
    ]
    if len(calibration) < 2:
        return {"axis": axis_name, "ticks": len(calibration), "max_value_residual": None, "max_pixel_residual": None}

    pixels = np.array([float(item["pixel"]) for item in calibration], dtype=np.float64)
    values = np.array([float(item["value"]) for item in calibration], dtype=np.float64)
    if axis_cfg.get("scale", "linear") == "log10":
        values_for_fit = np.log10(values)
    else:
        values_for_fit = values
    slope, intercept = np.polyfit(pixels, values_for_fit, 1)
    predicted = slope * pixels + intercept
    residuals = values_for_fit - predicted
    max_value_residual = float(np.max(np.abs(residuals)))
    max_pixel_residual = float(np.max(np.abs(residuals / slope))) if slope != 0 else None
    return {
        "axis": axis_name,
        "ticks": len(calibration),
        "max_value_residual": max_value_residual,
        "max_pixel_residual": max_pixel_residual,
    }


def extract_to_outputs(image_path, config, paths):
    image = ld.load_image(image_path)
    plot_area = ld.normalize_plot_area(config["plot_area"], image)
    axes_cfg = config["axes"]
    margin = int(config.get("border_margin_px", 2))
    left = plot_area["left"]
    right = plot_area["right"]
    top = plot_area["top"]
    bottom = plot_area["bottom"]
    roi = image[top + margin : bottom - margin, left + margin : right - margin]
    roi_plot_area = {
        "left": left + margin,
        "right": right - margin,
        "top": top + margin,
        "bottom": bottom - margin,
    }

    rows = []
    preview_rows = []
    paths["debug"].mkdir(parents=True, exist_ok=True)

    for series_cfg in config["series"]:
        if not is_truthy_selected(series_cfg.get("selected", True)):
            continue
        name = series_cfg["name"]
        mask = ld.make_color_mask(roi, series_cfg)
        ignore_regions = list(config.get("ignore_regions", [])) + list(series_cfg.get("ignore_regions", []))
        ld.apply_ignore_regions(mask, roi_plot_area, ignore_regions)
        mask = ld.postprocess_mask(mask, series_cfg)
        points = ld.extract_points_from_mask(mask, roi_plot_area, axes_cfg, series_cfg)
        x_step = series_cfg.get("x_step", config.get("output", {}).get("x_step"))
        points = ld.resample_points(points, x_step)

        for point in points:
            row = {
                "series": name,
                "x": float(point["x"]),
                "y": float(point["y"]),
                "pixel_x": float(point["pixel_x"]),
                "pixel_y": float(point["pixel_y"]),
            }
            rows.append(row)
            preview_rows.append(row)

        safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)
        cv2.imwrite(str(paths["debug"] / f"{safe_name}_mask.png"), mask)

    ld.ensure_parent(paths["csv"])
    with open(paths["csv"], "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["series", "x", "y", "pixel_x", "pixel_y"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "series": row["series"],
                    "x": "{:.10g}".format(row["x"]),
                    "y": "{:.10g}".format(row["y"]),
                    "pixel_x": "{:.3f}".format(row["pixel_x"]),
                    "pixel_y": "{:.3f}".format(row["pixel_y"]),
                }
            )

    ld.draw_preview(image, plot_area, preview_rows, config["series"], paths["preview"], point_radius=6)
    return rows


def render_image(path, caption):
    if path and Path(path).exists():
        st.image(str(path), caption=caption, use_container_width=True)


def build_plot_area_preview_image(image, plot_area, detection=None, pending_plot_area=None):
    preview = image.copy()
    detection = detection or {}
    for candidate in detection.get("candidates", [])[1:4]:
        candidate_area = candidate.get("plot_area")
        if not candidate_area:
            continue
        cv2.rectangle(
            preview,
            (int(candidate_area["left"]), int(candidate_area["top"])),
            (int(candidate_area["right"]), int(candidate_area["bottom"])),
            (175, 175, 175),
            1,
        )
    cv2.rectangle(
        preview,
        (plot_area["left"], plot_area["top"]),
        (plot_area["right"], plot_area["bottom"]),
        (0, 180, 255),
        2,
    )
    if pending_plot_area:
        cv2.rectangle(
            preview,
            (pending_plot_area["left"], pending_plot_area["top"]),
            (pending_plot_area["right"], pending_plot_area["bottom"]),
            (255, 80, 0),
            2,
        )
    for item in detection.get("x_labels", {}).get("numbers", []):
        cv2.circle(preview, (int(round(item["center_x"])), int(round(item["center_y"]))), 5, (255, 90, 0), -1)
    for item in detection.get("y_labels", {}).get("numbers", []):
        cv2.circle(preview, (int(round(item["center_x"])), int(round(item["center_y"]))), 5, (0, 170, 0), -1)
    if detection:
        label = f"PLOT {str(detection.get('status', 'unknown')).upper()} score={detection.get('score', '')} {detection.get('source', '')}"
        cv2.rectangle(preview, (8, 8), (min(image.shape[1] - 8, 430), 42), (0, 130, 180), -1)
        cv2.putText(preview, label[:58], (18, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
    return preview


def build_config_tick_preview_image(image, config):
    selected_ticks = calibration_ticks_for_preview(config)
    detected_ticks = config.get("detected_ticks", {"x": [], "y": []})
    qc = config.get("calibration_qc", {})
    return build_qc_tick_preview_image(image, config["plot_area"], detected_ticks, selected_ticks, qc)


def request_overlay_mode(overlay_mode_key, mode):
    st.session_state[f"{overlay_mode_key}_request"] = mode


def drag_to_image_delta(event, image):
    if not event or not all(key in event for key in ("x1", "y1", "x2", "y2")):
        return None
    display_width = float(event.get("width") or image.shape[1])
    display_height = float(event.get("height") or image.shape[0])
    scale_x = image.shape[1] / display_width
    scale_y = image.shape[0] / display_height
    return {
        "dx": int(round((float(event["x2"]) - float(event["x1"])) * scale_x)),
        "dy": int(round((float(event["y2"]) - float(event["y1"])) * scale_y)),
        "identity": (
            event.get("x1"),
            event.get("y1"),
            event.get("x2"),
            event.get("y2"),
            event.get("width"),
            event.get("height"),
            event.get("unix_time"),
        ),
    }


def shift_axis_calibration(config, axis_name, delta_px, clear_verification_state=True):
    if clear_verification_state:
        clear_calibration_verification(config)
    axis_cfg = config.get("axes", {}).get(axis_name, {})
    for item in axis_cfg.get("calibration", []):
        if item.get("pixel") in ("", None):
            continue
        item["pixel"] = int(round(float(item["pixel"]) + float(delta_px)))
    return config


def render_overlay_viewer(
    image,
    config,
    paths,
    overlay_mode_key,
    axis_key,
    editor_version_key,
    pending_shift=None,
    pending_plot_area=None,
):
    modes = ["Plot area", "Ticks", "Points"]
    request_key = f"{overlay_mode_key}_request"
    if st.session_state.get(request_key) in modes:
        st.session_state[overlay_mode_key] = st.session_state.pop(request_key)
    if st.session_state.get(overlay_mode_key) not in modes:
        st.session_state[overlay_mode_key] = "Plot area"

    mode = st.segmented_control(
        "Preview",
        modes,
        default=st.session_state[overlay_mode_key],
        key=overlay_mode_key,
        label_visibility="collapsed",
    )
    mode = mode or st.session_state[overlay_mode_key]
    drag_event = None

    if mode == "Plot area":
        preview = build_plot_area_preview_image(
            image,
            config["plot_area"],
            config.get("plot_area_detection"),
            pending_plot_area=pending_plot_area,
        )
        drag_event = streamlit_image_coordinates(
            bgr_to_rgb(preview),
            use_column_width="always",
            key=f"plot_area_viewer_{axis_key}_{st.session_state[editor_version_key]}_{plot_area_signature(pending_plot_area) if pending_plot_area else 'none'}",
        )
    elif mode == "Points":
        if paths["preview"].exists():
            render_image(paths["preview"], "Extracted point overlay")
        else:
            st.image(bgr_to_rgb(image), caption="Original", use_container_width=True)
    else:
        preview_config = config
        if pending_shift and int(pending_shift.get("delta", 0)):
            preview_config = copy.deepcopy(config)
            shift_axis_calibration(
                preview_config,
                pending_shift.get("axis", "y"),
                int(pending_shift.get("delta", 0)),
                clear_verification_state=False,
            )
        preview = build_config_tick_preview_image(image, preview_config)
        drag_event = streamlit_image_coordinates(
            bgr_to_rgb(preview),
            use_column_width="always",
            key=(
                f"tick_drag_{axis_key}_{st.session_state[editor_version_key]}_"
                f"{pending_shift.get('axis', 'y') if pending_shift else 'none'}_"
                f"{pending_shift.get('delta', 0) if pending_shift else 0}"
            ),
            click_and_drag=True,
        )
    return mode, drag_event


def render_tick_adjuster(
    image,
    config,
    paths,
    axis_key,
    editor_version_key,
    overlay_mode_key,
    pending_shift_key,
    movement_history_key,
    drag_event,
    status_key,
    x_tick_count,
    y_tick_count,
):
    st.subheader("Adjust ticks")
    axis_choice = st.radio(
        "Move axis",
        ["y axis", "x axis"],
        horizontal=True,
        key=f"shift_axis_{axis_key}",
    )
    axis_name = "x" if axis_choice.startswith("x") else "y"
    pending = st.session_state.get(pending_shift_key, {"axis": axis_name, "delta": 0})
    if pending.get("axis") != axis_name:
        pending = {"axis": axis_name, "delta": 0}
        st.session_state[pending_shift_key] = pending

    drag_delta = drag_to_image_delta(drag_event, image)
    drag_seen_key = f"tick_drag_seen_{axis_key}"
    if drag_delta and drag_delta["identity"] != st.session_state.get(drag_seen_key):
        delta = drag_delta["dx"] if axis_name == "x" else drag_delta["dy"]
        st.session_state[drag_seen_key] = drag_delta["identity"]
        if delta:
            st.session_state[pending_shift_key] = {"axis": axis_name, "delta": int(delta)}
            request_overlay_mode(overlay_mode_key, "Ticks")
            st.rerun()

    pending = st.session_state.get(pending_shift_key, {"axis": axis_name, "delta": 0})
    delta_px = int(pending.get("delta", 0)) if pending.get("axis") == axis_name else 0
    st.metric("Pending movement", f"{delta_px:+d} px")

    move_cols = st.columns(3)
    has_ticks = bool(config.get("axes", {}).get(axis_name, {}).get("calibration"))
    if move_cols[0].button("Apply movement", use_container_width=True, disabled=delta_px == 0 or not has_ticks):
        history = st.session_state.setdefault(movement_history_key, [])
        history.append({"axis": axis_name, "delta": delta_px})
        shift_axis_calibration(config, axis_name, delta_px)
        refresh_calibration_qc(config, int(x_tick_count), int(y_tick_count))
        draw_config_tick_preview(image, config, paths)
        save_config(config, paths["config"])
        st.session_state[pending_shift_key] = {"axis": axis_name, "delta": 0}
        st.session_state[editor_version_key] += 1
        request_overlay_mode(overlay_mode_key, "Ticks")
        st.session_state[status_key] = f"Moved {axis_name} ticks by {delta_px:+d} px."
        st.rerun()
    if move_cols[1].button("Reset movement", use_container_width=True, disabled=delta_px == 0):
        st.session_state[pending_shift_key] = {"axis": axis_name, "delta": 0}
        request_overlay_mode(overlay_mode_key, "Ticks")
        st.rerun()
    history = st.session_state.setdefault(movement_history_key, [])
    if move_cols[2].button("Undo last movement", use_container_width=True, disabled=not history):
        last_move = history.pop()
        shift_axis_calibration(config, last_move["axis"], -int(last_move["delta"]))
        refresh_calibration_qc(config, int(x_tick_count), int(y_tick_count))
        draw_config_tick_preview(image, config, paths)
        save_config(config, paths["config"])
        st.session_state[editor_version_key] += 1
        request_overlay_mode(overlay_mode_key, "Ticks")
        st.session_state[status_key] = (
            f"Undid {last_move['axis']} tick movement {int(last_move['delta']):+d} px."
        )
        st.rerun()
    return config


def pending_plot_area_from_corners(corners, image):
    if len(corners) < 2:
        return None
    x1, y1 = int(corners[0]["x"]), int(corners[0]["y"])
    x2, y2 = int(corners[1]["x"]), int(corners[1]["y"])
    return ld.normalize_plot_area(
        {
            "left": min(x1, x2),
            "right": max(x1, x2),
            "top": min(y1, y2),
            "bottom": max(y1, y2),
        },
        image,
    )


def render_plot_area_controls(
    image,
    config,
    paths,
    axis_key,
    editor_version_key,
    overlay_mode_key,
    status_key,
    plot_click_event,
    x_min,
    x_max,
    y_min,
    y_max,
    csv_x_interval,
    x_tick_count,
    y_tick_count,
    tick_params,
):
    st.subheader("Plot area")
    detection = config.get("plot_area_detection", {})
    verify = config.get("plot_area_verification", {})
    status_text = str(detection.get("status", "unknown")).upper()
    verify_text = "verified" if verify.get("verified") else "not verified"
    st.caption(
        f"Auto plot area: {status_text} score={detection.get('score', '')} source={detection.get('source', '')}; {verify_text}"
    )

    controls = st.columns([1.25, 1.0, 1.0])
    if controls[0].button("Re-detect plot area", use_container_width=True):
        config["plot_area"], config["plot_area_detection"] = detect_plot_area_candidates(
            image,
            int(x_tick_count),
            int(y_tick_count),
            tick_params,
        )
        config["plot_area"] = ld.normalize_plot_area(config["plot_area"], image)
        clear_plot_area_verification(config)
        config = sync_axis_settings(config, x_min, x_max, y_min, y_max, csv_x_interval)
        refresh_calibration_qc(config, int(x_tick_count), int(y_tick_count))
        draw_config_tick_preview(image, config, paths)
        save_config(config, paths["config"])
        st.session_state[editor_version_key] += 1
        request_overlay_mode(overlay_mode_key, "Plot area")
        st.session_state[status_key] = (
            f"Plot area updated from {config['plot_area_detection'].get('source', 'auto')} "
            f"score={config['plot_area_detection'].get('score', '')}."
        )
        st.rerun()

    if controls[1].button("Confirm plot area", use_container_width=True):
        confirm_plot_area(config, method="user-confirmed-overlay")
        refresh_calibration_qc(config, int(x_tick_count), int(y_tick_count))
        save_config(config, paths["config"])
        st.session_state[editor_version_key] += 1
        request_overlay_mode(overlay_mode_key, "Plot area")
        st.session_state[status_key] = "Plot area confirmed."
        st.rerun()

    if controls[2].button("Clear plot confirmation", use_container_width=True, disabled=not verify.get("verified")):
        clear_plot_area_verification(config)
        refresh_calibration_qc(config, int(x_tick_count), int(y_tick_count))
        save_config(config, paths["config"])
        st.session_state[editor_version_key] += 1
        request_overlay_mode(overlay_mode_key, "Plot area")
        st.session_state[status_key] = "Plot area confirmation cleared."
        st.rerun()

    redraw_key = f"plot_redraw_enabled_{axis_key}"
    corners_key = f"plot_corner_picks_{axis_key}"
    click_time_key = f"plot_corner_click_time_{axis_key}"
    st.session_state.setdefault(corners_key, [])
    st.session_state.setdefault(click_time_key, None)
    redraw_enabled = st.checkbox("Redraw by two corners", value=False, key=redraw_key)

    if redraw_enabled:
        image_pixel = click_to_image_pixel(plot_click_event, image)
        if image_pixel and image_pixel.get("unix_time") != st.session_state[click_time_key]:
            st.session_state[click_time_key] = image_pixel.get("unix_time")
            corners = st.session_state[corners_key]
            if len(corners) >= 2:
                corners = []
            corners.append({"x": int(image_pixel["x"]), "y": int(image_pixel["y"])})
            st.session_state[corners_key] = corners
            request_overlay_mode(overlay_mode_key, "Plot area")
            st.rerun()

    corners = st.session_state.get(corners_key, [])
    if redraw_enabled:
        st.write(f"Picked {len(corners)} / 2 corners.")
        if corners:
            st.dataframe(
                pd.DataFrame(
                    {
                        "corner": ["first", "second"][: len(corners)],
                        "x": [item["x"] for item in corners],
                        "y": [item["y"] for item in corners],
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        pending = pending_plot_area_from_corners(corners, image)
        action_cols = st.columns(3)
        if action_cols[0].button("Apply plot area", use_container_width=True, disabled=pending is None):
            config["plot_area"] = pending
            config["plot_area_detection"] = {
                "status": "review",
                "score": "",
                "source": "manual-two-corners",
                "warnings": ["manual plot area needs confirmation"],
                "candidates": [],
                "x_labels": config.get("plot_area_detection", {}).get("x_labels", {}),
                "y_labels": config.get("plot_area_detection", {}).get("y_labels", {}),
            }
            clear_plot_area_verification(config)
            config = sync_axis_settings(config, x_min, x_max, y_min, y_max, csv_x_interval)
            refresh_calibration_qc(config, int(x_tick_count), int(y_tick_count))
            draw_config_tick_preview(image, config, paths)
            save_config(config, paths["config"])
            st.session_state[corners_key] = []
            st.session_state[editor_version_key] += 1
            request_overlay_mode(overlay_mode_key, "Plot area")
            st.session_state[status_key] = "Applied manually redrawn plot area. Confirm it before extracting CSV."
            st.rerun()
        if action_cols[1].button("Undo corner", use_container_width=True, disabled=not corners):
            st.session_state[corners_key] = corners[:-1]
            request_overlay_mode(overlay_mode_key, "Plot area")
            st.rerun()
        if action_cols[2].button("Clear corners", use_container_width=True, disabled=not corners):
            st.session_state[corners_key] = []
            request_overlay_mode(overlay_mode_key, "Plot area")
            st.rerun()

    if detection:
        with st.expander("Plot area QC details"):
            rows = []
            for item in detection.get("candidates", []):
                pa = item.get("plot_area", {})
                rows.append(
                    {
                        "source": item.get("source"),
                        "status": item.get("status"),
                        "score": item.get("score"),
                        "left": pa.get("left"),
                        "right": pa.get("right"),
                        "top": pa.get("top"),
                        "bottom": pa.get("bottom"),
                        "warnings": "; ".join(item.get("warnings", [])[:4]),
                    }
                )
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            label_rows = []
            for axis_name in ("x", "y"):
                labels = detection.get(f"{axis_name}_labels", {})
                label_rows.append(
                    {
                        "axis": axis_name,
                        "status": labels.get("status"),
                        "count": labels.get("count", 0),
                        "min": labels.get("min"),
                        "max": labels.get("max"),
                        "warnings": "; ".join(labels.get("warnings", [])[:4]),
                    }
                )
            st.dataframe(pd.DataFrame(label_rows), use_container_width=True, hide_index=True)
    return config


def confirm_ticks_and_preview(image_path, image, config, paths, x_tick_count, y_tick_count):
    clear_calibration_verification(config)
    refresh_calibration_qc(config, int(x_tick_count), int(y_tick_count))
    save_config(config, paths["config"])
    rows = extract_to_outputs(image_path, config, paths)
    config["calibration_verification"] = {
        "verified": True,
        "verified_at": datetime.now().isoformat(timespec="seconds"),
        "note": "User confirmed ticks and generated CSV preview",
    }
    refresh_calibration_qc(config, int(x_tick_count), int(y_tick_count))
    draw_config_tick_preview(image, config, paths)
    save_config(config, paths["config"])
    return rows


def main():
    st.set_page_config(page_title="Batch Line Digitizer", layout="wide")
    st.title("Batch Line Digitizer")

    with st.sidebar:
        image_dir = st.text_input(
            "Image folder",
            value="/media/herryao/81ca6f19-78c8-470d-b5a1-5f35b4678058/work_dir/Document/Yan/Data_extraction/LSVs/Original",
        )
        output_dir = st.text_input(
            "Output folder",
            value="/media/herryao/81ca6f19-78c8-470d-b5a1-5f35b4678058/work_dir/Document/Yan/Data_extraction/LSVs/Outputs",
        )
        all_images = list_images(image_dir)
        qc_frame = read_batch_qc(output_dir)
        qc_filter = st.selectbox(
            "QC filter",
            ["All", "Plot area needs review", "Needs verification", "Verified", "Auto failed"],
            index=0,
        )
        if not qc_frame.empty:
            usable_source = qc_frame["usable"] if "usable" in qc_frame else pd.Series([False] * len(qc_frame))
            auto_status_source = qc_frame["auto_status"] if "auto_status" in qc_frame else qc_frame.get("status", pd.Series([""] * len(qc_frame)))
            plot_verified_source = qc_frame["plot_area_verified"] if "plot_area_verified" in qc_frame else pd.Series([False] * len(qc_frame))
            usable = usable_source.fillna(False).astype(str).str.lower().isin(("true", "1"))
            auto_failed = auto_status_source.fillna("").astype(str).str.lower().eq("fail")
            plot_verified = plot_verified_source.fillna(False).astype(str).str.lower().isin(("true", "1"))
            st.caption(
                f"QC: verified {int(usable.sum())}, plot area unverified {int((~plot_verified).sum())}, "
                f"needs verification {int((~usable).sum())}, auto failed {int(auto_failed.sum())}"
            )
        images = filter_images_by_qc(all_images, qc_frame, qc_filter)
        if not images:
            st.warning("No images found for this folder/filter.")
            return

        image_names = [path.name for path in images]
        pending_image_select = st.session_state.pop("pending_image_select", None)
        if pending_image_select in image_names:
            st.session_state["image_select"] = pending_image_select
        if "image_select" in st.session_state and st.session_state.image_select not in image_names:
            del st.session_state["image_select"]
        if "image_index" not in st.session_state:
            st.session_state.image_index = 0
        st.session_state.image_index = min(st.session_state.image_index, len(images) - 1)

        selected_name = st.selectbox(
            "Image",
            image_names,
            index=st.session_state.image_index,
            key="image_select",
        )
        st.session_state.image_index = image_names.index(selected_name)
        image_path = images[st.session_state.image_index]

        col_prev, col_next = st.columns(2)
        if col_prev.button("Previous", disabled=st.session_state.image_index == 0):
            st.session_state.image_index -= 1
            st.session_state.pending_image_select = image_names[st.session_state.image_index]
            st.rerun()
        if col_next.button("Next", disabled=st.session_state.image_index >= len(images) - 1):
            st.session_state.image_index += 1
            st.session_state.pending_image_select = image_names[st.session_state.image_index]
            st.rerun()

    paths = output_paths(output_dir, image_path)
    base_series_frame = pd.DataFrame(default_series())
    config = load_or_create_config(image_path, paths, 1.0, 2.0, 0.0, 24.0, 0.0, base_series_frame)

    image = ld.load_image(image_path)
    config["plot_area"] = ld.normalize_plot_area(config["plot_area"], image)
    config.setdefault("axes", {})
    config["axes"].setdefault("x", {"scale": "linear", "min": 1.0, "max": 2.0})
    config["axes"].setdefault("y", {"scale": "linear", "min": 0.0, "max": 24.0})
    config.setdefault("output", {})
    config.setdefault("series", frame_to_series(base_series_frame))

    with st.sidebar:
        st.divider()
        st.caption("Axis values")
        x_axis = config["axes"]["x"]
        y_axis = config["axes"]["y"]
        axis_key = image_path.name
        axis_ocr_key = f"axis_ocr_{axis_key}"
        ocr_status = axis_ocr_status()
        if not ocr_status.get("available"):
            st.caption(ocr_status.get("message", "OCR is not available."))
        if st.button("Auto read axis ranges", use_container_width=True):
            axis_ocr = auto_read_axis_ranges(image, config["plot_area"])
            config["axis_ocr"] = axis_ocr
            st.session_state[axis_ocr_key] = axis_ocr
            save_config(config, paths["config"])
            if not axis_ocr.get("available"):
                st.warning(axis_ocr.get("message", "OCR is not available."))

        axis_ocr = st.session_state.get(axis_ocr_key) or config.get("axis_ocr")
        if axis_ocr:
            with st.expander("Suggested axis ranges", expanded=True):
                st.dataframe(axis_ocr_suggestion_frame(axis_ocr), use_container_width=True, hide_index=True)
                if axis_ocr.get("warnings"):
                    st.caption("; ".join(axis_ocr.get("warnings", [])[:4]))
                suggestion_cols = st.columns(3)
                x_suggestion = axis_ocr.get("x", {})
                y_suggestion = axis_ocr.get("y", {})
                x_ok = (
                    x_suggestion.get("status") != "fail"
                    and x_suggestion.get("min") is not None
                    and x_suggestion.get("max") is not None
                )
                y_ok = (
                    y_suggestion.get("status") != "fail"
                    and y_suggestion.get("min") is not None
                    and y_suggestion.get("max") is not None
                )
                if suggestion_cols[0].button("Apply x", use_container_width=True, disabled=not x_ok):
                    st.session_state[f"x_min_{axis_key}"] = float(x_suggestion["min"])
                    st.session_state[f"x_max_{axis_key}"] = float(x_suggestion["max"])
                    st.session_state[f"x_tick_count_{axis_key}"] = int(x_suggestion["count"])
                    config = apply_axis_range_suggestion(config, "x", x_suggestion)
                    save_config(config, paths["config"])
                    st.rerun()
                if suggestion_cols[1].button("Apply y", use_container_width=True, disabled=not y_ok):
                    st.session_state[f"y_min_{axis_key}"] = float(y_suggestion["min"])
                    st.session_state[f"y_max_{axis_key}"] = float(y_suggestion["max"])
                    st.session_state[f"y_tick_count_{axis_key}"] = int(y_suggestion["count"])
                    config = apply_axis_range_suggestion(config, "y", y_suggestion)
                    save_config(config, paths["config"])
                    st.rerun()
                if suggestion_cols[2].button("Apply all", use_container_width=True, disabled=not (x_ok and y_ok)):
                    st.session_state[f"x_min_{axis_key}"] = float(x_suggestion["min"])
                    st.session_state[f"x_max_{axis_key}"] = float(x_suggestion["max"])
                    st.session_state[f"x_tick_count_{axis_key}"] = int(x_suggestion["count"])
                    st.session_state[f"y_min_{axis_key}"] = float(y_suggestion["min"])
                    st.session_state[f"y_max_{axis_key}"] = float(y_suggestion["max"])
                    st.session_state[f"y_tick_count_{axis_key}"] = int(y_suggestion["count"])
                    config = apply_axis_range_suggestion(config, "x", x_suggestion)
                    config = apply_axis_range_suggestion(config, "y", y_suggestion)
                    save_config(config, paths["config"])
                    st.rerun()
        x_min = st.number_input(
            "x min",
            value=float(x_axis.get("min", 1.0)),
            format="%.6f",
            key=f"x_min_{axis_key}",
        )
        x_max = st.number_input(
            "x max",
            value=float(x_axis.get("max", 2.0)),
            format="%.6f",
            key=f"x_max_{axis_key}",
        )
        x_tick_count = st.number_input(
            "x major tick count",
            value=int(default_tick_count(x_axis, 6)),
            min_value=2,
            step=1,
            key=f"x_tick_count_{axis_key}",
            help="输入横轴主刻度数量，比如 1.0, 1.2, 1.4, 1.6, 1.8, 2.0 就是 6。",
        )
        y_min = st.number_input(
            "y min",
            value=float(y_axis.get("min", 0.0)),
            format="%.6f",
            key=f"y_min_{axis_key}",
        )
        y_max = st.number_input(
            "y max",
            value=float(y_axis.get("max", 24.0)),
            format="%.6f",
            key=f"y_max_{axis_key}",
        )
        y_tick_count = st.number_input(
            "y major tick count",
            value=int(default_tick_count(y_axis, 7)),
            min_value=2,
            step=1,
            key=f"y_tick_count_{axis_key}",
            help="输入纵轴主刻度数量，比如 0, 4, 8, 12, 16, 20, 24 就是 7。",
        )
        config["axes"]["x"]["major_tick_count"] = int(x_tick_count)
        config["axes"]["y"]["major_tick_count"] = int(y_tick_count)
        csv_x_interval = st.number_input(
            "CSV x interval",
            value=float(config.get("output", {}).get("x_step") or 0.0),
            min_value=0.0,
            format="%.6f",
            key=f"csv_x_interval_{axis_key}",
            help="可选。0 表示按图像像素列输出原始点；填 0.002 这类数值会把曲线插值成固定 x 间隔。",
        )

        with st.expander("Advanced tick detection"):
            use_y_label_assist = st.checkbox(
                "Use y tick labels to assist y-axis",
                value=True,
                help=(
                    "默认开启。系统会在 y 轴左侧找数字标签的行中心；只有数量和 y major tick count "
                    "完全一致时才用它们校准 y 轴。overlay 中短的青色横线就是检测到的 y label 行。"
                ),
            )
            tick_params = {
                "search_px": st.number_input("search px", value=18, min_value=2, step=1),
                "dark_threshold": st.number_input("dark threshold", value=170, min_value=1, max_value=255, step=1),
                "min_tick_len": st.number_input("min tick length", value=5, min_value=1, step=1),
                "max_cluster_width": st.number_input("max cluster width", value=14, min_value=2, step=1),
            }

    config = sync_axis_settings(config, x_min, x_max, y_min, y_max, csv_x_interval)

    editor_version_key = f"editor_version_{image_path.name}"
    overlay_mode_key = f"overlay_mode_{image_path.name}"
    pending_shift_key = f"pending_tick_shift_{image_path.name}"
    movement_history_key = f"tick_movement_history_{image_path.name}"
    plot_corners_key = f"plot_corner_picks_{axis_key}"
    status_key = f"status_{image_path.name}"
    last_extract_key = f"last_extract_{image_path.name}"
    st.session_state.setdefault(editor_version_key, 0)
    st.session_state.setdefault(pending_shift_key, {"axis": "y", "delta": 0})
    st.session_state.setdefault(movement_history_key, [])
    editor_version = st.session_state[editor_version_key]

    st.subheader(image_path.name)
    st.caption(f"Image {st.session_state.image_index + 1} / {len(images)}")
    pending_plot_area = pending_plot_area_from_corners(st.session_state.get(plot_corners_key, []), image)
    canvas_cols = st.columns([0.125, 0.75, 0.125])
    with canvas_cols[1]:
        overlay_mode, viewer_event = render_overlay_viewer(
            image,
            config,
            paths,
            overlay_mode_key,
            axis_key,
            editor_version_key,
            pending_shift=st.session_state.get(pending_shift_key),
            pending_plot_area=pending_plot_area,
        )

    with st.container():
        st.subheader("Workflow")
        if st.session_state.get(status_key):
            st.info(st.session_state[status_key])

        main_cols = st.columns([1.25, 1.15, 1.35, 0.7])
        if main_cols[0].button("Auto process folder", type="primary", use_container_width=True):
            progress = st.progress(0.0)
            rows = []
            for idx, batch_image in enumerate(all_images, start=1):
                batch_paths = output_paths(output_dir, batch_image)
                try:
                    row = process_image_auto(
                        batch_image,
                        output_dir,
                        x_min,
                        x_max,
                        y_min,
                        y_max,
                        csv_x_interval,
                        int(x_tick_count),
                        int(y_tick_count),
                        tick_params,
                        base_series_frame,
                        use_y_label_assist=use_y_label_assist,
                    )
                except Exception as exc:
                    row = qc_summary_row(batch_image, {"calibration_qc": {"status": "fail", "score": 0}}, batch_paths, error=str(exc))
                rows.append(row)
                progress.progress(idx / max(1, len(all_images)))
            qc_path = write_batch_qc(output_dir, rows)
            st.session_state[status_key] = f"Auto processed {len(rows)} images. QC: {qc_path}"
            st.rerun()

        if main_cols[1].button("Auto detect ticks", use_container_width=True):
            config = sync_axis_settings(config, x_min, x_max, y_min, y_max, csv_x_interval)
            config = auto_calibrate_config(
                image_path,
                config,
                paths,
                tick_params,
                int(x_tick_count),
                int(y_tick_count),
                use_y_label_assist=use_y_label_assist,
            )
            save_config(config, paths["config"])
            st.session_state[editor_version_key] += 1
            st.session_state[pending_shift_key] = {"axis": "y", "delta": 0}
            request_overlay_mode(overlay_mode_key, "Ticks")
            detection_meta = config.get("tick_detection", {})
            qc = config.get("calibration_qc", {})
            st.session_state[status_key] = (
                f"Ticks used for calibration: x {len(config['axes']['x'].get('calibration', []))}/"
                f"{int(x_tick_count)}, y {len(config['axes']['y'].get('calibration', []))}/{int(y_tick_count)}. "
                f"QC {qc.get('status', 'unknown')} {qc.get('score', 0)}. "
                f"Candidates: x {detection_meta.get('x_candidates', 0)}, y {detection_meta.get('y_candidates', 0)}. "
                f"Y label assist: {'used' if detection_meta.get('y_label_assisted') else 'not used'}."
            )
            st.rerun()

        if main_cols[2].button("Confirm ticks + preview CSV", use_container_width=True):
            if not is_plot_area_verified(config):
                st.error("Confirm plot area first. The CSV will not be marked usable until the plot area is reviewed.")
            else:
                try:
                    rows = confirm_ticks_and_preview(image_path, image, config, paths, int(x_tick_count), int(y_tick_count))
                except Exception as exc:
                    clear_calibration_verification(config)
                    refresh_calibration_qc(config, int(x_tick_count), int(y_tick_count))
                    draw_config_tick_preview(image, config, paths)
                    save_config(config, paths["config"])
                    st.error(f"Extraction failed: {exc}")
                else:
                    st.session_state[last_extract_key] = len(rows)
                    st.session_state[editor_version_key] += 1
                    request_overlay_mode(overlay_mode_key, "Points")
                    st.session_state[status_key] = f"Verified plot area and ticks, extracted {len(rows)} points."
                    st.rerun()

        if main_cols[3].button("Next", use_container_width=True, disabled=st.session_state.image_index >= len(images) - 1):
            st.session_state.image_index += 1
            st.session_state.pending_image_select = image_names[st.session_state.image_index]
            st.rerun()

        if overlay_mode == "Plot area":
            config = render_plot_area_controls(
                image,
                config,
                paths,
                axis_key,
                editor_version_key,
                overlay_mode_key,
                status_key,
                viewer_event,
                x_min,
                x_max,
                y_min,
                y_max,
                csv_x_interval,
                int(x_tick_count),
                int(y_tick_count),
                tick_params,
            )

        if overlay_mode == "Ticks":
            config = render_tick_adjuster(
                image,
                config,
                paths,
                axis_key,
                editor_version_key,
                overlay_mode_key,
                pending_shift_key,
                movement_history_key,
                viewer_event,
                status_key,
                int(x_tick_count),
                int(y_tick_count),
            )
            config = render_manual_tick_picker(
                image,
                config,
                paths,
                axis_key,
                editor_version_key,
                overlay_mode_key,
                status_key,
                int(x_tick_count),
                int(y_tick_count),
            )

        if overlay_mode == "Points" and paths["csv"].exists():
            with open(paths["csv"], "rb") as handle:
                st.download_button("Download current CSV", handle, file_name=paths["csv"].name, mime="text/csv")

        if overlay_mode in ("Plot area", "Ticks"):
            with st.expander("Advanced: pixel crop"):
                pa = dict(config["plot_area"])
                old_pa = dict(pa)
                pa_cols = st.columns(4)
                pa["left"] = int(
                    pa_cols[0].number_input(
                        "crop left px", value=int(pa["left"]), step=1, key=f"pa_left_{axis_key}_{editor_version}"
                    )
                )
                pa["right"] = int(
                    pa_cols[1].number_input(
                        "crop right px", value=int(pa["right"]), step=1, key=f"pa_right_{axis_key}_{editor_version}"
                    )
                )
                pa["top"] = int(
                    pa_cols[2].number_input(
                        "crop top px", value=int(pa["top"]), step=1, key=f"pa_top_{axis_key}_{editor_version}"
                    )
                )
                pa["bottom"] = int(
                    pa_cols[3].number_input(
                        "crop bottom px", value=int(pa["bottom"]), step=1, key=f"pa_bottom_{axis_key}_{editor_version}"
                    )
                )
                new_plot_area = ld.normalize_plot_area(pa, image)
                if new_plot_area != old_pa:
                    config["plot_area"] = new_plot_area
                    config["plot_area_detection"] = {
                        "status": "review",
                        "score": "",
                        "source": "manual-pixel-crop",
                        "warnings": ["manual plot area needs confirmation"],
                        "candidates": [],
                    }
                    clear_plot_area_verification(config)
                    config = sync_axis_settings(config, x_min, x_max, y_min, y_max, csv_x_interval)
                    refresh_calibration_qc(config, int(x_tick_count), int(y_tick_count))
                    draw_config_tick_preview(image, config, paths)
                    save_config(config, paths["config"])
                    st.session_state[editor_version_key] += 1
                    request_overlay_mode(overlay_mode_key, "Plot area")
                    st.session_state[status_key] = "Pixel crop updated."
                    st.rerun()

        with st.expander("Advanced: edit tick table"):
            x_frame = calibration_to_frame(config["axes"].setdefault("x", {"scale": "linear"}))
            y_frame = calibration_to_frame(config["axes"].setdefault("y", {"scale": "linear"}))
            manual_pixel_edit = st.checkbox(
                "Edit pixel refs directly",
                value=False,
                key=f"manual_pixel_edit_{axis_key}",
            )
            pixel_disabled = False if manual_pixel_edit else ["pixel"]
            edit_cols = st.columns(2)
            with edit_cols[0]:
                st.caption("x ticks, left to right")
                x_frame = st.data_editor(
                    x_frame,
                    num_rows="dynamic",
                    use_container_width=True,
                    disabled=pixel_disabled,
                    column_config={
                        "value": st.column_config.NumberColumn("x value", format="%.6f"),
                        "pixel": st.column_config.NumberColumn("auto pixel ref", format="%d"),
                    },
                    key=f"x_ticks_{image_path.name}_{editor_version}",
                )
            with edit_cols[1]:
                st.caption("y ticks, bottom to top")
                y_frame = st.data_editor(
                    y_frame,
                    num_rows="dynamic",
                    use_container_width=True,
                    disabled=pixel_disabled,
                    column_config={
                        "value": st.column_config.NumberColumn("y value", format="%.6f"),
                        "pixel": st.column_config.NumberColumn("auto pixel ref", format="%d"),
                    },
                    key=f"y_ticks_{image_path.name}_{editor_version}",
                )

            new_x_calibration = frame_to_calibration(x_frame)
            new_y_calibration = frame_to_calibration(y_frame)
            if (
                not calibrations_equal(new_x_calibration, config["axes"]["x"].get("calibration"))
                or not calibrations_equal(new_y_calibration, config["axes"]["y"].get("calibration"))
            ):
                clear_calibration_verification(config)
                config["axes"]["x"]["calibration"] = new_x_calibration
                config["axes"]["y"]["calibration"] = new_y_calibration
                config = sync_axis_settings(config, x_min, x_max, y_min, y_max, csv_x_interval)
                refresh_calibration_qc(config, int(x_tick_count), int(y_tick_count))
                draw_config_tick_preview(image, config, paths)
                save_config(config, paths["config"])
                st.session_state[editor_version_key] += 1
                request_overlay_mode(overlay_mode_key, "Ticks")
                st.session_state[status_key] = "Tick table updated."
                st.rerun()

        with st.expander("Advanced: curve colors"):
            color_cols = st.columns(2)
            if color_cols[0].button("Detect colors", use_container_width=True):
                detected_series = detect_curve_colors(image, config["plot_area"])
                if detected_series:
                    config["series"] = detected_series
                    save_config(config, paths["config"])
                    st.session_state[editor_version_key] += 1
                    st.session_state[status_key] = f"Detected {len(detected_series)} color candidates."
                    st.rerun()
                st.warning("No curve colors detected inside the plot area.")
            color_cols[1].write("Select the colors to extract.")

            series_frame = series_to_frame(config.get("series", frame_to_series(base_series_frame)))
            series_frame = st.data_editor(
                series_frame,
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "selected": st.column_config.CheckboxColumn("Use"),
                    "name": st.column_config.TextColumn("series name"),
                    "hex": st.column_config.TextColumn("color"),
                    "tolerance": st.column_config.NumberColumn("tol"),
                    "saturation_min": st.column_config.NumberColumn("sat min"),
                    "min_area": st.column_config.NumberColumn("min area"),
                    "min_x_span": st.column_config.NumberColumn("min x span"),
                    "keep_largest_components": st.column_config.NumberColumn("keep largest"),
                },
                key=f"series_{image_path.name}_{editor_version}",
            )
            render_color_swatches(series_frame)
            config["series"] = frame_to_series(series_frame)

        with st.expander("Advanced: fallback actions"):
            fallback_cols = st.columns(3)
            if fallback_cols[0].button("Use evenly spaced tick refs", use_container_width=True):
                config = sync_axis_settings(config, x_min, x_max, y_min, y_max, csv_x_interval)
                config = assign_evenly_spaced_tick_values(config, int(x_tick_count), int(y_tick_count))
                refresh_calibration_qc(config, int(x_tick_count), int(y_tick_count))
                draw_config_tick_preview(image, config, paths)
                save_config(config, paths["config"])
                st.session_state[editor_version_key] += 1
                request_overlay_mode(overlay_mode_key, "Ticks")
                st.session_state[status_key] = "Used evenly spaced tick references."
                st.rerun()
            if fallback_cols[1].button("Reset config", use_container_width=True):
                config = make_default_config(image_path, x_min, x_max, y_min, y_max, csv_x_interval, base_series_frame)
                save_config(config, paths["config"])
                st.session_state[editor_version_key] += 1
                st.session_state[pending_shift_key] = {"axis": "y", "delta": 0}
                request_overlay_mode(overlay_mode_key, "Plot area")
                st.session_state[status_key] = "Config reset for this image."
                st.rerun()
            batch_qc_path = Path(output_dir).expanduser() / "batch_qc.csv"
            if batch_qc_path.exists():
                with open(batch_qc_path, "rb") as handle:
                    fallback_cols[2].download_button(
                        "Download batch_qc.csv",
                        handle,
                        file_name="batch_qc.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

        with st.expander("Advanced: ignore regions"):
            ignore_text = st.text_area(
                "Regions",
                value=json.dumps(config.get("ignore_regions", []), indent=2),
                height=120,
            )
            try:
                config["ignore_regions"] = json.loads(ignore_text)
            except json.JSONDecodeError:
                st.error("Ignore regions must be valid JSON.")

        st.subheader("QC")
        qc = config.get("calibration_qc") or refresh_calibration_qc(config, int(x_tick_count), int(y_tick_count))
        metric_cols = st.columns(4)
        metric_cols[0].metric("Status", str(qc.get("status", "unknown")).upper())
        metric_cols[1].metric("Auto status", str(qc.get("auto_status", qc.get("status", "unknown"))).upper())
        metric_cols[2].metric("Auto score", qc.get("score", 0))
        metric_cols[3].metric("Plot score", config.get("plot_area_detection", {}).get("score", ""))
        if st.session_state.get(last_extract_key) is not None:
            st.success(f"Last extraction: {st.session_state[last_extract_key]} points.")
        elif not is_plot_area_verified(config):
            st.warning("Not usable yet: confirm plot area first, then confirm ticks.")
        elif not qc.get("usable"):
            st.warning("Not usable yet: confirm ticks to generate the CSV preview.")
        else:
            st.success(f"Calibration verified at {qc.get('verified_at', '')}.")

        qc_rows = []
        for axis_name in ("x", "y"):
            axis_qc = qc.get(axis_name, {})
            qc_rows.append(
                {
                    "axis": axis_name,
                    "status": axis_qc.get("status"),
                    "score": axis_qc.get("score"),
                    "ticks": axis_qc.get("used_count"),
                    "expected": axis_qc.get("expected_count"),
                    "max_residual_px": axis_qc.get("max_residual_px"),
                    "max_spacing_residual_px": axis_qc.get("max_spacing_residual_px"),
                    "coverage": axis_qc.get("coverage"),
                    "warnings": "; ".join(axis_qc.get("warnings", [])),
                }
            )
        with st.expander("QC details"):
            st.dataframe(pd.DataFrame(qc_rows), use_container_width=True)
            st.write(f"Config: `{paths['config']}`")


if __name__ == "__main__":
    main()
