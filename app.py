import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import time
from pathlib import Path

st.set_page_config(
    page_title="WASTE DETECTION",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Rajdhani:wght@400;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #0d0d0d;
    color: #e8e8e8;
}
[data-testid="stSidebar"] {
    background: #111111;
    border-right: 1px solid #2a2a2a;
}
h1, h2, h3 { font-family: 'Rajdhani', sans-serif; }
p, div, span, label { font-family: 'Space Mono', monospace; font-size: 0.82rem; }

.hero-wrap {
    text-align: center;
    padding: 1.2rem 0 0.5rem;
}

.hero-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 3.2rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    background: linear-gradient(110deg, #c084fc, #a855f7, #ec4899, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
    display: inline-block;
}

.hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #555;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0.4rem;
}

@keyframes floatBin {
    0%   { transform: translateY(0px); }
    50%  { transform: translateY(-8px); }
    100% { transform: translateY(0px); }
}

.bin-float {
    display: inline-block;
    animation: floatBin 3s ease-in-out infinite;
}

.metric-card {
    background: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 1.1rem 1.2rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #a855f7; }
.metric-val {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #a855f7;
}
.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}

.bar-wrap {
    background: #1a1a1a;
    border-radius: 4px;
    height: 10px;
    width: 100%;
    margin: 4px 0 10px;
}
.bar-fill {
    height: 10px;
    border-radius: 4px;
    transition: width 0.6s ease;
}

.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #ec4899);
    color: #fff;
    font-family: 'Rajdhani', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.08em;
    border: none;
    border-radius: 6px;
    padding: 0.5rem 1.5rem;
    cursor: pointer;
}
.stButton > button:hover { opacity: 0.85; }

[data-testid="stFileUploader"] {
    border: 2px dashed #2a2a2a;
    border-radius: 8px;
    background: #111;
}

.section-divider {
    border: none;
    border-top: 1px solid #2a2a2a;
    margin: 1.5rem 0;
}

/* ── Highlighted keywords ── */
.kw {
    color: #ffffff;
    font-weight: 700;
    background: rgba(168, 85, 247, 0.45);
    border: 1.5px solid rgba(192, 132, 252, 0.85);
    border-radius: 4px;
    padding: 2px 7px;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.04em;
    box-shadow: 0 0 8px rgba(168, 85, 247, 0.5), inset 0 0 6px rgba(168, 85, 247, 0.15);
}
</style>
""", unsafe_allow_html=True)


CLASS_NAMES = ["Cardboard", "Paper", "Plastic", "Metal", "Glass", "Wood", "Cloth", "Rubber", "Waste"]

RECYCLABILITY = {
    "Cardboard": 0.92,
    "Paper":     0.88,
    "Plastic":   0.60,
    "Metal":     0.95,
    "Glass":     0.90,
    "Wood":      0.45,
    "Cloth":     0.35,
    "Rubber":    0.25,
    "Waste":     0.05,
}

RECYCLE_INFO = {
    "Cardboard": ("♻️ Highly Recyclable",    "#a855f7", "Flatten & drop in paper bin. Remove tape first."),
    "Paper":     ("♻️ Highly Recyclable",    "#a855f7", "Dry paper only. Shred confidential docs before recycling."),
    "Plastic":   ("⚠️ Moderately Recyclable","#ffcc00", "Check resin code (1–7). Rinse before recycling."),
    "Metal":     ("♻️ Highly Recyclable",    "#a855f7", "Aluminium & steel — crush cans to save space."),
    "Glass":     ("♻️ Highly Recyclable",    "#a855f7", "Sort by colour. Remove lids. Do NOT mix with ceramics."),
    "Wood":      ("🔄 Partially Recyclable", "#ff9500", "Clean wood → composting. Treated/painted → general waste."),
    "Cloth":     ("🔄 Partially Recyclable", "#ff9500", "Donate wearable items. Scraps → textile recycling banks."),
    "Rubber":    ("⚠️ Low Recyclability",    "#ff4d4d", "Tyres → tyre recycling drop-off. Small rubber → landfill."),
    "Waste":     ("🗑️ Non-Recyclable",       "#ff2222", "General waste bin. Reduce at source."),
}

CLASS_SIDEBAR_COLOR = {
    "Cardboard": "#c084fc",
    "Paper":     "#d4d4d4",
    "Plastic":   "#e879f9",
    "Metal":     "#c0c0c0",
    "Glass":     "#b39ddb",
    "Wood":      "#ff3366",
    "Cloth":     "#a855f7",
    "Rubber":    "#ff073a",
    "Waste":     "#e0e0e0",
}

CLASS_EMOJI = {
    "Cardboard": "📦",
    "Paper":     "📄",
    "Plastic":   "🧴",
    "Metal":     "🔩",
    "Glass":     "🍶",
    "Wood":      "🧱",
    "Cloth":     "👕",
    "Rubber":    "🛢",
    "Waste":     "🗑️",
}

CLASS_COLORS_BGR = {
    "Cardboard": (50, 205, 50),
    "Paper":     (0, 230, 118),
    "Plastic":   (255, 193, 7),
    "Metal":     (0, 229, 255),
    "Glass":     (100, 181, 246),
    "Wood":      (255, 152, 0),
    "Cloth":     (186, 104, 200),
    "Rubber":    (255, 82, 82),
    "Waste":     (239, 83, 80),
}

SVG_DUSTBIN = """
<svg xmlns="http://www.w3.org/2000/svg" width="54" height="64" viewBox="0 0 54 64">
  <defs>
    <linearGradient id="binGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%"   stop-color="#c084fc"/>
      <stop offset="50%"  stop-color="#a855f7"/>
      <stop offset="100%" stop-color="#ec4899"/>
    </linearGradient>
  </defs>
  <rect x="20" y="2" width="14" height="5" rx="2.5" fill="url(#binGrad)"/>
  <rect x="6" y="9" width="42" height="8" rx="3" fill="url(#binGrad)"/>
  <path d="M10 19 L12 58 Q12 61 15 61 L39 61 Q42 61 42 58 L44 19 Z" fill="url(#binGrad)" opacity="0.92"/>
  <line x1="21" y1="23" x2="20" y2="57" stroke="#0d0d0d" stroke-width="2.5" stroke-linecap="round" opacity="0.35"/>
  <line x1="27" y1="23" x2="27" y2="57" stroke="#0d0d0d" stroke-width="2.5" stroke-linecap="round" opacity="0.35"/>
  <line x1="33" y1="23" x2="34" y2="57" stroke="#0d0d0d" stroke-width="2.5" stroke-linecap="round" opacity="0.35"/>
</svg>
"""


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    from ultralytics import YOLO
    return YOLO(model_path)


def run_detection(model, image_np: np.ndarray, conf_thresh: float, iou_thresh: float):
    results = model.predict(
        source=image_np,
        conf=conf_thresh,
        iou=iou_thresh,
        verbose=False,
    )
    return results[0]


def draw_boxes(image_np: np.ndarray, result) -> np.ndarray:
    img = image_np.copy()
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return img
    for box in boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        label  = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
        color  = CLASS_COLORS_BGR.get(label, (200, 200, 200))
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{label}  {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, text, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def aggregate_detections(result):
    counts = {}
    boxes = result.boxes
    if boxes is None:
        return counts
    for box in boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        label  = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
        counts.setdefault(label, []).append(conf)
    return counts


def overall_recyclability(det_counts: dict) -> float:
    total = sum(len(v) for v in det_counts.values())
    if total == 0:
        return 0.0
    score = sum(RECYCLABILITY.get(k, 0) * len(v) for k, v in det_counts.items())
    return score / total


def check_low(model, img_bgr, cur_conf, iou):
    if cur_conf <= 0.10: return False, 0, {}
    r = model.predict(source=img_bgr, conf=0.10, iou=iou, verbose=False)[0]
    c = aggregate_detections(r)
    return len(c) > 0, sum(len(v) for v in c.values()), c


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='margin-bottom:1.5rem'>
      <div style='font-family:Rajdhani,sans-serif;font-size:1.8rem;font-weight:700;
           background:linear-gradient(110deg,#c084fc,#a855f7,#d4d4d4,#c0c0c0,#b39ddb);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
           line-height:1.25'>
         AI BASED SMART WASTE<br>SEGREGATION SYSTEM
      </div>
      <div style='font-family:Space Mono,monospace;font-size:0.65rem;color:#6b7280;
           letter-spacing:0.15em;text-transform:uppercase;margin-top:6px'>
        ─ YOLOv8 Waste Detector ─
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Settings")

    model_path = st.text_input(
        "Model path (.pt)",
        value="weights/best.pt",
        help="Path to your trained YOLOv8 weights file"
    )

    conf_thresh = st.slider("Confidence threshold", 0.10, 0.95, 0.35, 0.05)
    iou_thresh  = st.slider("IoU (NMS) threshold",  0.10, 0.95, 0.45, 0.05)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("**Waste Categories**")
    for cls in RECYCLABILITY.keys():
        color = CLASS_SIDEBAR_COLOR.get(cls, "#aaa")
        emoji = CLASS_EMOJI.get(cls, "•")
        is_neon = color in ("#ff3366", "#ff073a")
        shadow = f"text-shadow: 0 0 8px {color}aa;" if is_neon else ""
        st.markdown(f"""
        <div style='margin-bottom:6px;display:flex;align-items:center;gap:7px'>
          <span style='font-size:0.85rem'>{emoji}</span>
          <span style='color:{color};font-family:Space Mono;font-size:0.72rem;font-weight:700;{shadow}'>{cls}</span>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────

# Top tagline — purple/silver/red gradient, bold and visible
st.markdown("""
<div style='text-align:center;padding-top:1.2rem;font-family:Space Mono,monospace;
     font-size:0.72rem;letter-spacing:0.2em;text-transform:uppercase;font-weight:700;
     background:linear-gradient(110deg,#a855f7,#d4d4d4,#ff073a);
     -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;'>
  Final Year Project &nbsp;·&nbsp; YOLOv8m &nbsp;·&nbsp; AI Powered
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style='text-align:center;padding:0.3rem 0 0;'>
  <span style='font-family:Rajdhani,sans-serif;font-size:3.2rem;font-weight:700;
       letter-spacing:0.08em;line-height:1.15;
       background:linear-gradient(110deg,#c084fc,#a855f7,#ec4899,#38bdf8);
       -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;'>
    AI BASED SMART WASTE<br>SEGREGATION SYSTEM
  </span>
  <span class='bin-float' style='display:inline-block;margin-left:14px;vertical-align:middle;position:relative;top:-6px;'>
    {SVG_DUSTBIN}
  </span>
</div>
""", unsafe_allow_html=True)

# Subtitle — bright gradient, clearly visible
st.markdown("""
<div style='text-align:center;padding-bottom:0.8rem;font-family:Space Mono,monospace;
     font-size:0.72rem;letter-spacing:0.2em;text-transform:uppercase;margin-top:0.5rem;font-weight:700;
     background:linear-gradient(110deg,#a855f7,#d4d4d4,#ff073a);
     -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;'>
  Mixed Waste Detection &amp; Recyclability Analysis · YOLOv8m
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MODEL LOAD
# ─────────────────────────────────────────────
model = None
if Path(model_path).exists():
    with st.spinner("Loading YOLOv8 model…"):
        try:
            model = load_model(model_path)
            st.markdown(f"""
            <div style='background:#0d1b2a;border:1px solid #1d4ed8;border-radius:6px;
                 padding:0.6rem 1rem;font-family:Space Mono;font-size:0.78rem;color:#60a5fa;'>
              🔵 Model loaded — <code style='color:#93c5fd'>{model_path}</code>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Model load failed: {e}")
else:
    st.warning(f"⚠️ Model file not found at `{model_path}`. Update path in sidebar.")

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  INPUT
# ─────────────────────────────────────────────
tab_upload, tab_cam = st.tabs(["📁  Upload Image", "📷  Camera"])

image_pil = None
source_label = ""

with tab_upload:
    uploaded = st.file_uploader(
        "Drop a road / street waste image here",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
    )
    if uploaded:
        image_pil = Image.open(uploaded).convert("RGB")
        source_label = uploaded.name

with tab_cam:
    cam_img = st.camera_input("Take the picture of garbage.")
    if cam_img:
        image_pil = Image.open(cam_img).convert("RGB")
        source_label = "camera_capture"

# ─────────────────────────────────────────────
#  DETECTION
# ─────────────────────────────────────────────
if image_pil is not None:
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    img_np  = np.array(image_pil)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    col_img, col_result = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown("**Original Image**")
        st.image(image_pil, use_container_width=True)
        h, w = img_np.shape[:2]
        st.caption(f"Resolution: {w}×{h} px  |  Source: {source_label}")

    if model is None:
        with col_result:
            st.info("Load a valid model (update path in sidebar) to run detection.")
    else:
        with st.spinner("🔍 Scanning for waste…"):
            t0 = time.time()
            result        = run_detection(model, img_bgr, conf_thresh, iou_thresh)
            elapsed       = time.time() - t0
            annotated_bgr = draw_boxes(img_bgr, result)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        with col_result:
            st.markdown("**Detection Result**")
            st.image(annotated_rgb, use_container_width=True)
            st.caption(f"Inference time: {elapsed*1000:.0f} ms")

        det_counts = aggregate_detections(result)
        total_dets = sum(len(v) for v in det_counts.values())
        overall_rc = overall_recyclability(det_counts)

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        k1, k2, k3, k4 = st.columns(4)
        kpi_style = lambda val, lbl: f"""
        <div class='metric-card'>
          <div class='metric-val'>{val}</div>
          <div class='metric-label'>{lbl}</div>
        </div>"""

        rc_color = "#a855f7" if overall_rc >= 0.7 else "#ffcc00" if overall_rc >= 0.4 else "#ff4d4d"
        k1.markdown(kpi_style(total_dets, "Items Detected"), unsafe_allow_html=True)
        k2.markdown(kpi_style(len(det_counts), "Unique Classes"), unsafe_allow_html=True)
        k3.markdown(f"""
        <div class='metric-card'>
          <div class='metric-val' style='color:{rc_color}'>{overall_rc*100:.0f}%</div>
          <div class='metric-label'>Recyclability Score</div>
        </div>""", unsafe_allow_html=True)
        k4.markdown(kpi_style(f"{elapsed*1000:.0f}ms", "Inference Time"), unsafe_allow_html=True)

        if det_counts:
            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
            st.markdown("### 🗂️ Per-Class Breakdown")

            for cls_name, confs in sorted(det_counts.items(), key=lambda x: -len(x[1])):
                rc_ratio  = RECYCLABILITY.get(cls_name, 0)
                rc_label, rc_hex, tip = RECYCLE_INFO.get(cls_name, ("Unknown","#888",""))
                avg_conf  = sum(confs) / len(confs)
                count     = len(confs)
                pct_bar   = int(rc_ratio * 100)

                with st.expander(f"{rc_label}  ·  **{cls_name}**  ({count} detected)", expanded=True):
                    ca, cb = st.columns([2, 1])
                    with ca:
                        st.markdown(f"""
                        <div style='margin-bottom:6px;font-family:Space Mono;font-size:0.78rem;color:#aaa'>
                          Recyclability Rate
                        </div>
                        <div class='bar-wrap'>
                          <div class='bar-fill' style='width:{pct_bar}%;background:{rc_hex}'></div>
                        </div>
                        <span style='color:{rc_hex};font-family:Rajdhani;font-size:1.5rem;font-weight:700'>{pct_bar}%</span>
                        """, unsafe_allow_html=True)
                        st.markdown(f"<div style='color:#888;font-size:0.75rem;margin-top:6px'>💡 {tip}</div>",
                                    unsafe_allow_html=True)
                    with cb:
                        st.markdown(f"""
                        <div style='text-align:center;padding:0.8rem;background:#161616;border-radius:6px;border:1px solid #2a2a2a'>
                          <div style='font-family:Rajdhani;font-size:2rem;font-weight:700;color:{rc_hex}'>{count}</div>
                          <div style='font-family:Space Mono;font-size:0.65rem;color:#555'>ITEMS</div>
                          <br>
                          <div style='font-family:Rajdhani;font-size:1.4rem;font-weight:600;color:#aaa'>{avg_conf*100:.1f}%</div>
                          <div style='font-family:Space Mono;font-size:0.65rem;color:#555'>AVG CONF</div>
                        </div>
                        """, unsafe_allow_html=True)

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        if total_dets == 0:
            found, low_n, low_c = check_low(model, img_bgr, conf_thresh, iou_thresh)
            if found:
                st.markdown(f"""
                <div style='background:#111;border:1px solid #2a2a2a;border-radius:8px;padding:1.2rem'>
                  <div style='font-family:Rajdhani;font-size:1.1rem;font-weight:700;color:#a855f7;margin-bottom:0.5rem'>
                    ⚡ Waste Detected — Lower the Threshold
                  </div>
                  <div style='font-family:Space Mono;font-size:0.72rem;color:#666;line-height:1.9'>
                    <span class='kw'>{low_n} item(s)</span> found at minimum sensitivity but filtered by
                    your current threshold of <span class='kw'>{conf_thresh}</span>.<br><br>
                    → Drag <span class='kw'>Confidence threshold</span> down to
                    <span class='kw'>0.10</span> in the sidebar and re-scan.
                  </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background:#111;border:1px solid #2a2a2a;border-radius:8px;padding:1.2rem'>
                  <div style='font-family:Rajdhani;font-size:1.1rem;font-weight:700;color:#a855f7;margin-bottom:0.5rem'>
                    🔍 No Waste Detected
                  </div>
                  <div style='font-family:Space Mono;font-size:0.72rem;color:#666;line-height:1.9'>
                    Nothing found in this image.<br>
                    Try a clearer photo, better lighting, or an image with visible waste objects.
                  </div>
                </div>""", unsafe_allow_html=True)
        else:
            rc_emoji = "🟢" if overall_rc >= 0.7 else "🟡" if overall_rc >= 0.4 else "🔴"
            recyclable_items = sum(
                len(v) for k, v in det_counts.items() if RECYCLABILITY.get(k, 0) >= 0.6
            )
            non_recyclable = total_dets - recyclable_items

            st.markdown(f"""
            <div style='background:#111;border:1px solid #2a2a2a;border-radius:8px;padding:1.2rem'>
              <div style='font-family:Rajdhani;font-size:1.4rem;font-weight:700;margin-bottom:0.5rem'>
                {rc_emoji} Waste Summary Report
              </div>
              <div style='font-family:Space Mono;font-size:0.78rem;color:#aaa;line-height:1.8'>
                Total items detected: <span class='kw'>{total_dets}</span><br>
                Recyclable items (≥60% rate): <span class='kw' style='color:#c084fc;border-color:rgba(192,132,252,0.35);background:rgba(192,132,252,0.12)'>{recyclable_items}</span><br>
                Non-recyclable / difficult: <span class='kw' style='color:#ff6b6b;border-color:rgba(255,107,107,0.35);background:rgba(255,107,107,0.12)'>{non_recyclable}</span><br>
                Overall recyclability score: <span class='kw' style='color:{rc_color};border-color:{rc_color}55;background:{rc_color}18'>{overall_rc*100:.1f}%</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        buf = io.BytesIO()
        Image.fromarray(annotated_rgb).save(buf, format="PNG")
        st.download_button(
            label="⬇️ Download Annotated Image",
            data=buf.getvalue(),
            file_name="waste_scan_result.png",
            mime="image/png",
        )

else:
    st.markdown("""
    <div style='text-align:center;padding:4rem 2rem;background:#111;border:1px dashed #2a2a2a;border-radius:12px'>
      <div style='font-size:3rem;margin-bottom:0.8rem'>♻️</div>
      <div style='font-family:Rajdhani;font-size:1.4rem;letter-spacing:0.1em;font-weight:700;
           background:linear-gradient(110deg,#a855f7,#d4d4d4,#ff073a);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;'>
        Upload an image or use the camera to start scanning
      </div>
      <div style='font-family:Space Mono;font-size:0.72rem;color:#888;margin-top:0.5rem'>
        Supports: JPG · PNG · BMP · WEBP
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;font-family:Space Mono;font-size:0.68rem;color:#888;padding:0.5rem'>
  AI BASED SMART WASTE SEGREGATION SYSTEM · YOLOv8m · 9 Classes · Built with Streamlit
</div>
""", unsafe_allow_html=True)
