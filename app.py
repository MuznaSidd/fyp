import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import time
from pathlib import Path

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Waste Segregation",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CSS  – dark industrial aesthetic
# ─────────────────────────────────────────────
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

.hero-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 3.2rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    background: linear-gradient(90deg, #39ff14, #00e5ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #555;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

/* Metric cards */
.metric-card {
    background: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 1.1rem 1.2rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #39ff14; }
.metric-val {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #39ff14;
}
.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}

/* Detection pill */
.det-pill {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 4px;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    margin: 3px;
    letter-spacing: 0.05em;
}

/* Recyclability bar */
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
    background: linear-gradient(135deg, #39ff14, #00e5ff);
    color: #000;
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
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  CONSTANTS — 9 classes from data.yaml
# ─────────────────────────────────────────────
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
    "Cardboard": ("♻️ Highly Recyclable", "#39ff14",  "Flatten & drop in paper bin. Remove tape first."),
    "Paper":     ("♻️ Highly Recyclable", "#39ff14",  "Dry paper only. Shred confidential docs before recycling."),
    "Plastic":   ("⚠️ Moderately Recyclable","#ffcc00","Check resin code (1–7). Rinse before recycling."),
    "Metal":     ("♻️ Highly Recyclable", "#39ff14",  "Aluminium & steel — crush cans to save space."),
    "Glass":     ("♻️ Highly Recyclable", "#39ff14",  "Sort by colour. Remove lids. Do NOT mix with ceramics."),
    "Wood":      ("🔄 Partially Recyclable","#ff9500", "Clean wood → composting. Treated/painted → general waste."),
    "Cloth":     ("🔄 Partially Recyclable","#ff9500", "Donate wearable items. Scraps → textile recycling banks."),
    "Rubber":    ("⚠️ Low Recyclability",   "#ff4d4d", "Tyres → tyre recycling drop-off. Small rubber → landfill."),
    "Waste":     ("🗑️ Non-Recyclable",      "#ff2222", "General waste bin. Reduce at source."),
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


# ─────────────────────────────────────────────
#  MODEL LOADER
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    from ultralytics import YOLO
    return YOLO(model_path)


# ─────────────────────────────────────────────
#  DETECTION FUNCTION
# ─────────────────────────────────────────────
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

        # Box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Label background
        text = f"{label}  {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, text, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def aggregate_detections(result):
    """Returns dict: class_name -> list of confidences"""
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


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='margin-bottom:1.5rem'>
      <div class='hero-title' style='font-size:1.8rem'>♻️ AI BASED SMART WASTE<br>SEGREGATION SYSTEM</div>
      <div class='hero-sub' style='margin-top:4px'>YOLOv8 Waste Detector</div>
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
    st.markdown("**9 Waste Classes**")
    for cls, ratio in RECYCLABILITY.items():
        color = RECYCLE_INFO[cls][1]
        pct   = int(ratio * 100)
        st.markdown(f"""
        <div style='margin-bottom:6px'>
          <span style='color:{color};font-family:Space Mono;font-size:0.72rem'>{cls}</span>
          <div class='bar-wrap'><div class='bar-fill' style='width:{pct}%;background:{color}'></div></div>
        </div>
        """, unsafe_allow_html=True)

    st.caption(f"Recyclability scores per class")


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style='padding:1rem 0 0.5rem'>
  <div class='hero-title'>AI BASED SMART WASTE SEGREGATION</div>
  <div class='hero-sub'>Mixed Waste Detection & Recyclability Analysis · YOLOv8m</div>
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
            st.success(f"✅ Model loaded — `{model_path}`")
        except Exception as e:
            st.error(f"Model load failed: {e}")
else:
    st.warning(f"⚠️ Model file not found at `{model_path}`. Update path in sidebar.")

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  INPUT  — Upload or Camera
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
    cam_img = st.camera_input("Take a photo of the waste")
    if cam_img:
        image_pil = Image.open(cam_img).convert("RGB")
        source_label = "camera_capture"

# ─────────────────────────────────────────────
#  DETECTION
# ─────────────────────────────────────────────
if image_pil is not None:
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    img_np = np.array(image_pil)
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
            result = run_detection(model, img_bgr, conf_thresh, iou_thresh)
            elapsed = time.time() - t0
            annotated_bgr = draw_boxes(img_bgr, result)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        with col_result:
            st.markdown("**Detection Result**")
            st.image(annotated_rgb, use_container_width=True)
            st.caption(f"Inference time: {elapsed*1000:.0f} ms")

        # ── Aggregate ──────────────────────────────
        det_counts = aggregate_detections(result)
        total_dets = sum(len(v) for v in det_counts.values())
        overall_rc = overall_recyclability(det_counts)

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        # ── KPI row ───────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        kpi_style = lambda val, lbl: f"""
        <div class='metric-card'>
          <div class='metric-val'>{val}</div>
          <div class='metric-label'>{lbl}</div>
        </div>"""

        rc_color = "#39ff14" if overall_rc >= 0.7 else "#ffcc00" if overall_rc >= 0.4 else "#ff4d4d"
        k1.markdown(kpi_style(total_dets, "Items Detected"), unsafe_allow_html=True)
        k2.markdown(kpi_style(len(det_counts), "Unique Classes"), unsafe_allow_html=True)
        k3.markdown(f"""
        <div class='metric-card'>
          <div class='metric-val' style='color:{rc_color}'>{overall_rc*100:.0f}%</div>
          <div class='metric-label'>Recyclability Score</div>
        </div>""", unsafe_allow_html=True)
        k4.markdown(kpi_style(f"{elapsed*1000:.0f}ms", "Inference Time"), unsafe_allow_html=True)

        # ── Per-class breakdown ────────────────────
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

        # ── Summary card ──────────────────────────
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        if total_dets == 0:
            st.info("No waste detected. Try lowering the confidence threshold.")
        else:
            rc_emoji = "🟢" if overall_rc >= 0.7 else "🟡" if overall_rc >= 0.4 else "🔴"
            recyclable_items = sum(
                len(v) for k, v in det_counts.items() if RECYCLABILITY.get(k, 0) >= 0.6
            )
            non_recyclable   = total_dets - recyclable_items

            st.markdown(f"""
            <div style='background:#111;border:1px solid #2a2a2a;border-radius:8px;padding:1.2rem'>
              <div style='font-family:Rajdhani;font-size:1.4rem;font-weight:700;margin-bottom:0.5rem'>
                {rc_emoji} Waste Summary Report
              </div>
              <div style='font-family:Space Mono;font-size:0.78rem;color:#aaa;line-height:1.8'>
                Total items detected: <b style='color:#e8e8e8'>{total_dets}</b><br>
                Recyclable items (≥60% rate): <b style='color:#39ff14'>{recyclable_items}</b><br>
                Non-recyclable / difficult: <b style='color:#ff4d4d'>{non_recyclable}</b><br>
                Overall recyclability score: <b style='color:{rc_color}'>{overall_rc*100:.1f}%</b>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Download annotated image ───────────────
        st.markdown("<br>", unsafe_allow_html=True)
        buf = io.BytesIO()
        Image.fromarray(annotated_rgb).save(buf, format="PNG")
        st.download_button(
            label="⬇️ Download Annotated Image",
            data=buf.getvalue(),
            file_name="kachra_scan_result.png",
            mime="image/png",
        )

else:
    # Placeholder
    st.markdown("""
    <div style='text-align:center;padding:4rem 2rem;background:#111;border:1px dashed #2a2a2a;border-radius:12px'>
      <div style='font-size:3rem;margin-bottom:0.8rem'>🗑️</div>
      <div style='font-family:Rajdhani;font-size:1.4rem;color:#555;letter-spacing:0.1em'>
        Upload an image or use the camera to start scanning
      </div>
      <div style='font-family:Space Mono;font-size:0.72rem;color:#333;margin-top:0.5rem'>
        Supports: JPG · PNG · BMP · WEBP
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;font-family:Space Mono;font-size:0.68rem;color:#333;padding:0.5rem'>
  KACHRA SCANNER · YOLOv8m Mixed Waste · 9 Classes · Built with Streamlit
</div>
""", unsafe_allow_html=True)
