# =====================================================
# Project: Generative Abstract Poster — Part B (3D-like, Streamlit)
# Author: HUANG SHIXIAN
# Course: Arts and Advanced Big Data
# =====================================================

import streamlit as st
import io, os, random, math, colorsys
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Helpers: numeric + color utils
# ------------------------------
def clamp01(x):
    return max(0.0, min(1.0, float(x)))

def hsv_to_rgb_tuple(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(clamp01(h), clamp01(s), clamp01(v))
    return (r, g, b)

def lerp(a, b, t):
    return a + (b - a) * t

def lerp_color(c1, c2, t):
    return (lerp(c1[0], c2[0], t), lerp(c1[1], c2[1], t), lerp(c1[2], c2[2], t))

# ------------------------------
# Palettes
# ------------------------------
def vivid_palette(k=6):
    hues = np.linspace(0, 1, k, endpoint=False)
    np.random.shuffle(hues)
    return [hsv_to_rgb_tuple(h, np.random.uniform(0.8, 1.0), np.random.uniform(0.65, 0.95)) for h in hues]

def mono_palette(k=6, hue=0.58):
    return [hsv_to_rgb_tuple(hue, np.random.uniform(0.25, 0.85), np.random.uniform(0.55, 0.98)) for _ in range(k)]

# ------------------------------
# Geometry: wobbly blob
# ------------------------------
def blob(center=(0.5, 0.5), r=0.3, points=220, wobble=0.15):
    angles = np.linspace(0, 2 * math.pi, points)
    radii  = r * (1 + wobble * (np.random.rand(points) - 0.5))
    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)
    return x, y

# -------------------------------------------------------------
# Painter: "soft" shadow (multi-offset passes) under a blob
# -------------------------------------------------------------
def draw_soft_shadow(ax, x, y, base_alpha=0.22, blur_passes=5, dx=0.015, dy=-0.02):
    for i in range(blur_passes):
        t = (i + 1) / blur_passes
        falloff = (1.0 - t) ** 1.5
        offset_x = x + dx * (i + 1)
        offset_y = y + dy * (i + 1)
        ax.fill(offset_x, offset_y, color=(0, 0, 0), alpha=base_alpha * falloff, edgecolor=(0, 0, 0, 0))

# -------------------------------------------------------------
# Painter: radial-ish gradient by layering scaled blobs
# -------------------------------------------------------------
def draw_gradient_blob(ax, center, r, wobble, inner_color, outer_color, rings=7):
    for i in range(rings):
        t = i / max(1, rings - 1)             # 0 (outer) -> 1 (inner)
        rr = r * (1.0 - 0.18 * t)
        x, y = blob(center=center, r=rr, wobble=wobble * (0.7 + 0.6 * (1 - t)))
        color = lerp_color(outer_color, inner_color, t)
        alpha = lerp(0.45, 0.75, t)
        ax.fill(x, y, color=color, alpha=alpha, edgecolor=(0, 0, 0, 0))

# -------------------------------------------------------------
# Flat baseline (no 3D cues)
# -------------------------------------------------------------
def render_flat(seed=42, n_layers=10, wobble=0.15, palette_fn=vivid_palette, figsize=(6, 8)):
    random.seed(seed); np.random.seed(seed)
    fig = plt.figure(figsize=figsize)
    ax = plt.gca(); ax.axis('off'); ax.set_facecolor((0.98, 0.98, 0.97))
    palette = palette_fn(6)

    for _ in range(n_layers):
        cx, cy = random.random(), random.random()
        rr = random.uniform(0.15, 0.45)
        x, y = blob((cx, cy), rr, wobble=wobble)
        color = random.choice(palette)
        alpha = random.uniform(0.3, 0.65)
        ax.fill(x, y, color=color, alpha=alpha, edgecolor=(0, 0, 0, 0))

    ax.text(0.05, 0.95, "Part B – Flat Baseline", fontsize=14, weight='bold', transform=ax.transAxes)
    plt.tight_layout()
    return fig

# -------------------------------------------------------------
# 3D-like rendering (shadow + warm/cool + gradient)
# -------------------------------------------------------------
def render_3d_like(seed=42, n_layers=10, wobble=0.17, palette_fn=vivid_palette, figsize=(6, 8),
                   add_shadow=True, add_gradient=True, warm_cool=True, bg=(0.95, 0.95, 0.94)):
    random.seed(seed); np.random.seed(seed)
    fig = plt.figure(figsize=figsize)
    ax = plt.gca(); ax.axis('off'); ax.set_facecolor(bg)
    base_palette = palette_fn(6)

    def warm_cool_shift(rgb, depth_t):
        r, g, b = rgb
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        if warm_cool:
            h = (h + 0.10 * depth_t) % 1.0   # shift hue slightly by depth
            v = clamp01(v - 0.12 * depth_t)  # darker with depth
        return colorsys.hsv_to_rgb(h, s, v)

    # draw back-to-front: deep layers first
    for i in range(n_layers):
        depth_t = i / max(1, n_layers - 1)
        cx, cy = random.random(), random.random()
        rr = random.uniform(0.15, 0.45)
        x, y = blob((cx, cy), rr, wobble=wobble)

        if add_shadow:
            draw_soft_shadow(ax, x, y, base_alpha=0.22, blur_passes=5, dx=0.015, dy=-0.02)

        base_color = random.choice(base_palette)
        outer = warm_cool_shift(base_color, depth_t)
        inner = lerp_color(outer, (1.0, 1.0, 1.0), 0.18)

        if add_gradient:
            draw_gradient_blob(ax, (cx, cy), rr, wobble, inner_color=inner, outer_color=outer, rings=7)
        else:
            ax.fill(x, y, color=outer, alpha=np.random.uniform(0.35, 0.7), edgecolor=(0, 0, 0, 0))

    ax.text(0.05, 0.95, "Part B – 3D-like Poster", fontsize=14, weight='bold', transform=ax.transAxes)
    plt.tight_layout()
    return fig

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="3D-like Generative Poster — Part B", layout="wide")
st.title("🧩 Generative Abstract Poster — Part B (3D-like)")

with st.sidebar:
    st.header("Controls")
    mode = st.radio("Mode", ["Flat Baseline", "3D-like"], horizontal=False)
    seed = st.number_input("Random Seed", value=42, step=1)
    n_layers = st.slider("Layers", 4, 24, 10, step=1)
    wobble = st.slider("Wobble", 0.05, 0.40, 0.17, step=0.01)
    palette_mode = st.selectbox("Palette", ["Vivid", "Mono Blue"])
    add_shadow = st.checkbox("Add Shadow (3D cue)", value=True)
    add_gradient = st.checkbox("Add Gradient (3D cue)", value=True)
    warm_cool = st.checkbox("Warm–Cool + Value Shift by Depth", value=True)

palette_fn = vivid_palette if st.session_state.get("palette_mode", None) != "Mono Blue" else vivid_palette
# Ensure palette selector works:
if palette_mode == "Vivid":
    palette_fn = vivid_palette
else:
    palette_fn = lambda k: mono_palette(k, hue=0.58)

col1, col2 = st.columns([2, 1])

if st.button("Generate Poster", type="primary"):
    if mode == "Flat Baseline":
        fig = render_flat(seed=seed, n_layers=n_layers, wobble=wobble, palette_fn=palette_fn)
    else:
        fig = render_3d_like(seed=seed, n_layers=n_layers, wobble=wobble, palette_fn=palette_fn,
                             add_shadow=add_shadow, add_gradient=add_gradient, warm_cool=warm_cool)

    with col1:
        st.pyplot(fig)

        # Download PNG
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=220, bbox_inches="tight")
        st.download_button("Download PNG", data=buf.getvalue(),
                           file_name=f"partB_{mode.replace(' ','_').lower()}.png",
                           mime="image/png")

    with col2:
        st.markdown("### Notes")
        st.write("- **Flat Baseline**: no depth cues; simple layered blobs.")
        st.write("- **3D-like**: adds shadow + gradient + warm–cool value shift by depth.")
        st.write("- Same **seed** → reproducible output.")
