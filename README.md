# 🧩 Generative Abstract Poster — Part B (3D-like, Streamlit)

This app brings your **Part B (3D-like Generative Poster)** into a web interface using **Streamlit**.  
It compares a **Flat Baseline** vs a **3D-like** version using depth cues:
- Soft **drop shadows** (multi-pass)
- **Warm–cool** hue shift + **value gradient** by depth
- Layering + transparency

---

## 🎯 Goals (as per course workflow)
- Add **at least two depth cues** (shadow, gradient/warm–cool colors)
- Compare **Flat vs 3D-like** results
- Save screenshots for submission

---

## 🧠 How it works
- Each layer is a wobbly “blob” generated with sinusoidal noise.
- 3D-like mode:
  - Renders **soft shadows** by stacking offset copies
  - Applies a **warm→cool / value** shift based on depth
  - Optionally draws a **radial-ish gradient** by layering scaled blobs
- **Same seed → same output** (reproducible design)

---

## 🛠 Installation & Run
```bash
git clone https://github.com/yourusername/generative-poster-partB.git
cd generative-poster-partB
pip install -r requirements.txt
streamlit run app.py
