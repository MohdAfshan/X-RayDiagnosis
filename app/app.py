from __future__ import annotations

import base64
from io import BytesIO
import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import streamlit.components.v1 as components
import torch
from PIL import Image
from torchvision import transforms

from src.gradcam import GradCAM, overlay_heatmap
from src.model import build_model
from src.utils import get_device, set_seed

CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "best_model.pth"
EXAMPLES_ROOT = PROJECT_ROOT / "data" / "raw" / "chest_xray" / "test"

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")


@st.cache_resource(show_spinner=False)
def load_artifacts() -> Dict[str, object]:
    set_seed(42)
    device = get_device()
    model = build_model(
        "densenet121",
        num_classes=2,
        pretrained=False,
        dropout=0.5,
        freeze_backbone=False,
    )

    checkpoint = torch.load(str(CHECKPOINT_PATH), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    return {
        "device": device,
        "model": model,
        "gradcam": GradCAM(model, target_layer_name="features.denseblock4"),
        "best_auc": float(checkpoint.get("val_auc", 0.0)),
        "trained_epoch": int(checkpoint.get("epoch", 0)),
    }


INFERENCE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def predict(pil_image: Optional[Image.Image], artifacts: Dict[str, object]) -> Dict[str, object]:
    if pil_image is None:
        return {
            "label": "Awaiting scan",
            "confidence": 0.0,
            "pneumonia_prob": 0.0,
            "normal_prob": 0.0,
            "heatmap_raw": None,
            "gradcam": None,
            "original": None,
            "report": "Please upload a chest X-ray image.",
        }

    image = pil_image.convert("RGB")
    input_tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(artifacts["device"])

    with torch.inference_mode():
        logits = artifacts["model"](input_tensor)
        probs = torch.softmax(logits, dim=1)[0]

    pred_class = int(probs.argmax().item())
    confidence = float(probs[pred_class].item())
    pneumonia_prob = float(probs[1].item())
    normal_prob = float(probs[0].item())

    heatmap_raw = artifacts["gradcam"].generate(input_tensor, class_idx=pred_class)
    gradcam_pil = overlay_heatmap(image, heatmap_raw, alpha=0.5)

    label = "PNEUMONIA" if pred_class == 1 else "NORMAL"
    report = (
        f"Prediction: {label}\n"
        f"Confidence: {confidence * 100:.1f}%\n"
        f"PNEUMONIA Probability: {pneumonia_prob * 100:.1f}%\n"
        f"NORMAL Probability: {normal_prob * 100:.1f}%\n"
        f"Model: DenseNet121 | AUC: {artifacts['best_auc']:.4f}\n"
        "Research use only. Not for clinical diagnosis."
    )

    return {
        "label": label,
        "confidence": confidence,
        "pneumonia_prob": pneumonia_prob,
        "normal_prob": normal_prob,
        "heatmap_raw": heatmap_raw,
        "gradcam": gradcam_pil,
        "original": image,
        "report": report,
    }


def build_example_paths() -> List[Path]:
    if not EXAMPLES_ROOT.exists():
        return []

    supported_suffixes = {".png", ".jpg", ".jpeg"}
    examples: List[Path] = []

    for class_name in ("NORMAL", "PNEUMONIA"):
        class_dir = EXAMPLES_ROOT / class_name
        if not class_dir.exists():
            continue

        class_examples = [
            path
            for path in sorted(class_dir.iterdir())
            if path.is_file() and path.suffix.lower() in supported_suffixes
        ]
        examples.extend(class_examples[:3])

    if examples:
        return examples

    return [
        path
        for path in sorted(EXAMPLES_ROOT.rglob("*"))
        if path.is_file()
        and path.suffix.lower() in supported_suffixes
        and "__MACOSX" not in path.parts
    ][:6]


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def encode_image_base64(pil_image: Image.Image) -> str:
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def render_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800;900&display=swap');

        :root {
            --bg0: #020617;
            --bg1: #020617;
            --bg2: #0a192f;
            --glass: rgba(17, 24, 39, 0.58);
            --stroke: rgba(160, 206, 255, 0.22);
            --text: #edf4ff;
            --muted: #94a3b8;
            --blue: #4facfe;
            --cyan: #00f2fe;
            --green: #26dea2;
            --red: #ff6a7f;
        }

        .stApp {
            background:
                radial-gradient(1000px 700px at 10% 8%, rgba(79,172,254,0.18), transparent 58%),
                radial-gradient(900px 650px at 92% 0%, rgba(0,242,254,0.14), transparent 52%),
                linear-gradient(180deg, var(--bg0) 0%, var(--bg1) 50%, var(--bg2) 100%);
            color: var(--text);
            animation: bgShift 18s ease-in-out infinite alternate;
        }

        [data-testid="stHeader"] { background: transparent !important; }

        .hero-container {
            padding: 14px 4px 28px;
            margin-bottom: 6px;
            animation: fadeIn 0.8s ease-out;
        }

        .hero-title {
            margin: 16px 0 0;
            font-size: clamp(42px, 5.5vw, 68px);
            line-height: 1.0;
            letter-spacing: -2px;
            font-weight: 900;
            color: #e6edf3;
            font-family: 'Syne', sans-serif;
        }

        .gradient-text {
            display: block;
            margin-top: -2px;
            font-size: clamp(42px, 5.5vw, 68px);
            line-height: 1.0;
            letter-spacing: -2px;
            font-weight: 900;
            background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            color: transparent;
            font-family: 'Syne', sans-serif;
        }

        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 7px 14px;
            border-radius: 999px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.10);
            box-shadow: 0 0 0 1px rgba(79,172,254,0.08) inset, 0 10px 30px rgba(0,0,0,0.18);
            color: #dbeafe;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }

        .hero-badge-dot {
            width: 7px;
            height: 7px;
            border-radius: 999px;
            background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
            box-shadow: 0 0 14px rgba(0,242,254,0.45);
        }

        .hero-subtitle {
            max-width: 500px;
            margin-top: 18px;
            color: var(--muted);
            font-size: 15px;
            line-height: 1.6;
            letter-spacing: 0.01em;
        }

        .hero-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 22px;
        }

        .chip {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 14px;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.12);
            background: rgba(255,255,255,0.05);
            box-shadow: 0 0 0 1px rgba(79,172,254,0.10) inset;
            color: #e2e8f0;
            font-size: 12px;
            line-height: 1;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
        }

        .hero-stats {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 14px;
            margin-top: 28px;
        }

        .stat-card {
            background: linear-gradient(180deg, rgba(17, 24, 39, 0.72), rgba(10, 16, 28, 0.58));
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 20px;
            padding: 18px 16px;
            box-shadow: 0 12px 30px rgba(0,0,0,0.18), 0 0 0 1px rgba(255,255,255,0.03) inset;
            transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
        }

        .stat-card:hover {
            transform: translateY(-4px);
            border-color: rgba(79,172,254,0.26);
            box-shadow: 0 18px 38px rgba(0,0,0,0.24), 0 0 0 1px rgba(79,172,254,0.10) inset;
        }

        .stat-value {
            font-size: 28px;
            line-height: 1;
            font-weight: 800;
            letter-spacing: -0.04em;
            color: #f8fbff;
        }

        .stat-label {
            margin-top: 8px;
            color: #94a3b8;
            font-size: 12px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .status-ready {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            margin-top: 14px;
            padding: 8px 13px;
            border-radius: 999px;
            border: 1px solid rgba(53,233,255,0.24);
            background: rgba(16, 28, 44, 0.55);
            color: #dff8ee;
            font-size: 12px;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #36f2ad;
            box-shadow: 0 0 10px rgba(54,242,173,0.8);
            animation: pulse 1.6s infinite ease-in-out;
        }

        .glass-card {
            background: linear-gradient(180deg, rgba(20, 31, 48, 0.72), rgba(13, 22, 34, 0.68));
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--stroke);
            border-radius: 20px;
            padding: 16px;
            box-shadow: 0 10px 28px rgba(0,0,0,0.28), 0 0 0 1px rgba(255,255,255,0.03) inset;
        }

        .hero-sphere {
            width: 300px;
            height: 300px;
            border-radius: 50%;
            margin: 0 auto 28px auto;
            position: relative;
            background: radial-gradient(circle at 38% 38%, #00d4ff 0%, #0a6a7a 35%, #020c18 75%, transparent 100%);
            box-shadow: 0 0 80px rgba(0,212,255,0.25), 0 0 160px rgba(0,212,255,0.10);
            overflow: hidden;
            animation: sphereRotate 6s linear infinite;
        }

        .hero-sphere::before {
            content: '';
            position: absolute;
            inset: 0;
            border-radius: 50%;
            background: conic-gradient(
                from 0deg,
                transparent 0deg,
                rgba(0,212,255,0.35) 60deg,
                rgba(255,255,255,0.18) 90deg,
                transparent 140deg,
                rgba(0,212,255,0.10) 220deg,
                transparent 360deg
            );
            animation: sphereRotate 4s linear infinite;
        }

        .hero-sphere::after {
            content: '';
            position: absolute;
            inset: 0;
            border-radius: 50%;
            background: conic-gradient(
                from 180deg,
                transparent 0deg,
                rgba(0,212,255,0.12) 80deg,
                rgba(79,172,254,0.20) 120deg,
                transparent 200deg
            );
            animation: sphereRotate 8s linear infinite reverse;
        }

        .fade-in {
            animation: fadeIn 0.55s ease-out;
        }

        .hover-lift {
            transition: all 0.3s ease;
        }

        .hover-lift:hover {
            transform: translateY(-4px);
            box-shadow: 0 14px 36px rgba(53,233,255,0.14), 0 0 0 1px rgba(53,233,255,0.20) inset;
        }

        div.stButton > button {
            border-radius: 999px;
            border: 1px solid rgba(53,233,255,0.45);
            transition: all 0.3s ease;
        }

        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #3e79ff 0%, #36dfff 52%, #28d8a0 100%);
            color: #ffffff;
            font-weight: 800;
            box-shadow: 0 10px 30px rgba(62,121,255,0.45), 0 0 22px rgba(53,233,255,0.25);
        }

        div.stButton > button[kind="primary"]:hover {
            transform: scale(1.03);
            box-shadow: 0 14px 34px rgba(62,121,255,0.5), 0 0 30px rgba(53,233,255,0.34);
        }

        div.stButton > button:active {
            transform: scale(0.98);
        }

        [data-testid="stFileUploaderDropzone"] {
            border: 2.5px dashed rgba(77,163,255,0.75) !important;
            background: linear-gradient(180deg, rgba(77,163,255,0.16), rgba(53,233,255,0.08));
            border-radius: 18px !important;
            transition: all 0.3s ease;
        }

        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: rgba(53,233,255,0.95) !important;
            background: linear-gradient(180deg, rgba(77,163,255,0.22), rgba(53,233,255,0.13));
            box-shadow: 0 0 24px rgba(53,233,255,0.22);
        }

        [data-testid="stFileUploaderDropzone"] * {
            color: #ffffff !important;
        }

        .preview-img {
            border-radius: 16px;
            overflow: hidden;
            transition: all 0.3s ease;
        }

        .preview-img:hover {
            transform: scale(1.015);
        }

        .example-tile {
            border: 1.6px solid rgba(160, 206, 255, 0.28);
            border-radius: 16px;
            padding: 8px;
            background: rgba(20, 34, 52, 0.6);
            transition: all 0.3s ease;
        }

        .example-tile.selected {
            border-color: rgba(53,233,255,0.92);
            box-shadow: 0 0 0 1px rgba(53,233,255,0.45) inset, 0 8px 24px rgba(53,233,255,0.20);
        }

        .example-caption {
            color: #d5e5fb;
            font-size: 12px;
            margin-top: 6px;
            text-align: center;
        }

        .diag-title {
            margin: 0;
            font-size: clamp(28px, 4vw, 40px);
            font-weight: 800;
            letter-spacing: -0.03em;
        }

        .diag-title.normal { color: #30e2aa; }
        .diag-title.pneumonia { color: #ff7386; }

        .meter-wrap {
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 8px 0 16px;
        }

        .meter-ring {
            --p: 0;
            --accent: #35e9ff;
            width: 148px;
            height: 148px;
            border-radius: 50%;
            display: grid;
            place-items: center;
            background: conic-gradient(var(--accent) calc(var(--p) * 1%), rgba(255,255,255,0.10) 0);
            position: relative;
            box-shadow: 0 0 24px color-mix(in srgb, var(--accent) 35%, transparent);
            transition: all 0.3s ease;
        }

        .meter-ring::before {
            content: "";
            position: absolute;
            inset: 12px;
            border-radius: 50%;
            background: rgba(8, 14, 24, 0.95);
            border: 1px solid rgba(255,255,255,0.08);
        }

        .meter-value {
            position: relative;
            z-index: 1;
            font-size: 26px;
            font-weight: 800;
        }

        .bar-wrap { margin-top: 8px; }

        .bar-row { margin-bottom: 12px; }

        .bar-meta {
            display: flex;
            justify-content: space-between;
            color: #d9e6f8;
            font-size: 12px;
            margin-bottom: 6px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .bar-track {
            height: 10px;
            border-radius: 999px;
            background: rgba(255,255,255,0.10);
            overflow: hidden;
        }

        .bar-fill {
            height: 100%;
            border-radius: inherit;
            transition: all 0.7s ease;
            animation: fillGrow 0.7s ease;
        }

        .bar-fill.pneumonia {
            background: linear-gradient(90deg, #ff8d60 0%, #ff5f7e 100%);
        }

        .bar-fill.normal {
            background: linear-gradient(90deg, #29b4ff 0%, #25df9f 100%);
        }

        .report-box {
            max-height: 220px;
            overflow: auto;
            border-radius: 14px;
            padding: 12px;
            border: 1px solid rgba(160,206,255,0.22);
            background: rgba(10, 16, 26, 0.75);
            color: #d6e4f7;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 12px;
            line-height: 1.65;
            white-space: pre-wrap;
        }

        .muted {
            color: var(--muted);
            font-size: 14px;
        }

        .section-title {
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 10px;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(14px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .slide-up {
            animation: slideUp 0.75s ease-out;
        }

        @keyframes pulse {
            0%, 100% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.1); opacity: 0.8; }
        }

        @keyframes fillGrow {
            from { width: 0%; }
            to { width: 100%; }
        }

        @keyframes sphereRotate {
            from { transform: rotate(0deg); }
            to   { transform: rotate(360deg); }
        }

        @keyframes bgShift {
            0% { filter: saturate(100%); }
            100% { filter: saturate(115%); }
        }

        @media (max-width: 900px) {
            .hero-title { font-size: clamp(32px, 7vw, 48px); }
            .hero-stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .section-title { font-size: 18px; }
        }

        @media (max-width: 640px) {
            .hero-container { padding: 8px 2px 22px; }
            .hero-stats { grid-template-columns: 1fr; }
            .hero-title { font-size: clamp(28px, 9vw, 40px); }
            .hero-subtitle { max-width: 100%; }
        }

        * { cursor: none !important; }

        #cursor-dot {
            position: fixed;
            top: 0; left: 0;
            width: 8px; height: 8px;
            border-radius: 50%;
            background: #00f2fe;
            pointer-events: none;
            z-index: 999999;
            transform: translate(-50%, -50%);
            transition: transform 0.08s ease;
            box-shadow: 0 0 10px rgba(0,242,254,0.9), 0 0 20px rgba(0,242,254,0.5);
        }

        #cursor-ring {
            position: fixed;
            top: 0; left: 0;
            width: 36px; height: 36px;
            border-radius: 50%;
            border: 1.5px solid rgba(0,242,254,0.6);
            pointer-events: none;
            z-index: 999998;
            transform: translate(-50%, -50%);
            transition: transform 0.12s ease, width 0.2s ease, height 0.2s ease, 
                        border-color 0.2s ease;
            box-shadow: 0 0 12px rgba(0,242,254,0.15);
        }

        #cursor-ring.clicking {
            width: 28px; height: 28px;
            border-color: rgba(0,242,254,1);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def ensure_state() -> None:
    if "current_image" not in st.session_state:
        st.session_state.current_image = None
    if "analysis" not in st.session_state:
        st.session_state.analysis = None
    if "image_source_key" not in st.session_state:
        st.session_state.image_source_key = None
    if "selected_example" not in st.session_state:
        st.session_state.selected_example = None
    if "show_heatmap" not in st.session_state:
        st.session_state.show_heatmap = True
    if "heatmap_alpha" not in st.session_state:
        st.session_state.heatmap_alpha = 0.5


def render_hero(artifacts: Dict[str, object]) -> None:
    stat_cards = f"""
        <div class="hero-stats slide-up">
            <div class="stat-card">
                <div class="stat-value">{artifacts['best_auc']:.4f}</div>
                <div class="stat-label">AUC</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{artifacts['trained_epoch']}</div>
                <div class="stat-label">Epochs</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" style="font-size:18px; letter-spacing:-0.02em; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">DenseNet121</div>
                <div class="stat-label">Model</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">224 px</div>
                <div class="stat-label">Resolution</div>
            </div>
        </div>
    """



    left_col, right_col = st.columns([1.1, 0.9])

    with left_col:
        st.markdown(
            f"""
            <div class="hero-container fade-in">
                <div class="hero-badge slide-up"><span class="hero-badge-dot"></span>Medical AI Research Lab</div>
                <h1 class="hero-title slide-up">
                    Chest X-Ray
                    <span class="gradient-text">AI Diagnosis</span>
                </h1>
                <div class="hero-subtitle slide-up">
                    Premium chest X-ray analysis powered by DenseNet121 and Grad-CAM explainability.
                    Designed as a research-first interface for fast, high-confidence interpretation.
                </div>
                <div class="hero-chips slide-up">
                    <span class="chip">DenseNet121</span>
                    <span class="chip">224 px input</span>
                    <span class="chip">Grad-CAM explainability</span>
                    <span class="chip">Research use only</span>
                </div>
                {stat_cards}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right_col:
        st.markdown(
            """
            <div style="display:flex; flex-direction:column; align-items:stretch; justify-content:flex-start; padding-top: 18px;">
                <div class="hero-sphere"></div>
                <div style="background: rgba(17,24,39,0.55); border: 1px solid rgba(160,206,255,0.18); border-radius: 16px; padding: 20px 22px; backdrop-filter: blur(12px); margin-bottom: 14px;">
                    <div style="font-size: 11px; letter-spacing: 0.18em; text-transform: uppercase; color: #4facfe;">INTELLIGENCE LAYER</div>
                    <div style="font-size: 14px; color: #94a3b8; line-height: 1.6; margin-top: 8px;">Premium diagnostic motion with glass surfaces, subtle medical grids, and smooth result transitions.</div>
                </div>
                <div style="background: rgba(17,24,39,0.55); border: 1px solid rgba(160,206,255,0.18); border-radius: 16px; padding: 20px 22px; backdrop-filter: blur(12px); margin-bottom: 14px;">
                    <div style="font-size: 11px; letter-spacing: 0.18em; text-transform: uppercase; color: #4facfe;">WORKFLOW</div>
                    <div style="font-size: 14px; color: #94a3b8; line-height: 1.6; margin-top: 8px;">Upload a scan, run analysis, inspect diagnosis confidence, review probability bars, and evaluate the explanation heatmap.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        components.html(
                """
                <style>
                    body { margin: 0; overflow: hidden; background: transparent; }
                </style>
                <script>
                    const styleEl = document.createElement('style');
                    styleEl.textContent = `
                        * { cursor: none !important; }
                        #cur-dot {
                            position: fixed;
                            width: 7px; height: 7px;
                            border-radius: 50%;
                            background: #00f2fe;
                            pointer-events: none;
                            z-index: 2147483647;
                            transform: translate(-50%, -50%);
                            box-shadow: 0 0 8px rgba(0,242,254,0.9), 0 0 18px rgba(0,242,254,0.5);
                            transition: width 0.15s ease, height 0.15s ease;
                        }
                        #cur-ring {
                            position: fixed;
                            width: 32px; height: 32px;
                            border-radius: 50%;
                            border: 1.5px solid rgba(0,242,254,0.55);
                            pointer-events: none;
                            z-index: 2147483646;
                            transform: translate(-50%, -50%);
                            box-shadow: 0 0 10px rgba(0,242,254,0.12);
                            transition: width 0.18s ease, height 0.18s ease, 
                                                    border-color 0.18s ease;
                        }
                    `;
                    const target = window.parent.document;
                    target.head.appendChild(styleEl);

                    const dot = target.createElement('div');
                    dot.id = 'cur-dot';
                    const ring = target.createElement('div');
                    ring.id = 'cur-ring';
                    target.body.appendChild(dot);
                    target.body.appendChild(ring);

                    let mx = 0, my = 0, rx = 0, ry = 0;

                    target.addEventListener('mousemove', e => {
                        mx = e.clientX;
                        my = e.clientY;
                        dot.style.left = mx + 'px';
                        dot.style.top  = my + 'px';
                    });

                    function loop() {
                        rx += (mx - rx) * 0.13;
                        ry += (my - ry) * 0.13;
                        ring.style.left = rx + 'px';
                        ring.style.top  = ry + 'px';
                        requestAnimationFrame(loop);
                    }
                    loop();

                    target.addEventListener('mousedown', () => {
                        dot.style.width  = '10px';
                        dot.style.height = '10px';
                        ring.style.width  = '24px';
                        ring.style.height = '24px';
                        ring.style.borderColor = 'rgba(0,242,254,0.95)';
                    });
                    target.addEventListener('mouseup', () => {
                        dot.style.width  = '7px';
                        dot.style.height = '7px';
                        ring.style.width  = '32px';
                        ring.style.height = '32px';
                        ring.style.borderColor = 'rgba(0,242,254,0.55)';
                    });
                </script>
                """,
                height=0,
        )


def render_example_grid(examples: List[Path]) -> None:
    st.markdown('<div class="section-title">Example Images</div>', unsafe_allow_html=True)

    if not examples:
        st.warning("No example images found in dataset.")
        return

    cols = st.columns(min(3, len(examples)))
    for idx, example_path in enumerate(examples):
        with cols[idx % len(cols)]:
            preview = Image.open(example_path).convert("RGB")
            b64 = encode_image_base64(preview)
            is_selected = st.session_state.selected_example == str(example_path)
            selected_class = "selected" if is_selected else ""

            st.markdown(
                f"""
                <div class="example-tile {selected_class}">
                    <img src="data:image/jpeg;base64,{b64}" style="width:100%; border-radius:12px; display:block;" />
                    <div class="example-caption">{example_path.parent.name}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button("Use Example", key=f"use_example_{idx}"):
                st.session_state.current_image = preview
                st.session_state.analysis = None
                st.session_state.image_source_key = f"example:{example_path}"
                st.session_state.selected_example = str(example_path)
                st.rerun()

def render_report_with_copy(report_text: str) -> None:
    st.markdown(f"<div class='report-box'>{report_text}</div>", unsafe_allow_html=True)
    components.html(
        f"""
        <div style='margin-top:8px;'>
          <button id='copyBtn' style='padding:8px 12px;border-radius:10px;border:1px solid rgba(53,233,255,.45);background:linear-gradient(135deg,#2f70ff,#35dfff);color:#fff;font-weight:700;cursor:pointer;'>Copy Report</button>
        </div>
        <script>
          const text = {json.dumps(report_text)};
          const btn = document.getElementById('copyBtn');
          btn.addEventListener('click', async () => {{
            try {{
              await navigator.clipboard.writeText(text);
              const oldText = btn.textContent;
              btn.textContent = 'Copied';
              setTimeout(() => btn.textContent = oldText, 1400);
            }} catch (e) {{
              btn.textContent = 'Copy failed';
            }}
          }});
        </script>
        """,
        height=48,
    )


def main() -> None:
    st.set_page_config(page_title="Chest X-Ray Diagnosis", page_icon="🩻", layout="wide")
    render_theme()
    ensure_state()

    artifacts = load_artifacts()
    examples = build_example_paths()

    render_hero(artifacts)

    left_col, right_col = st.columns([1.04, 0.96], gap="large")

    with left_col:
        st.markdown('<div class="section-title">Upload Image</div>', unsafe_allow_html=True)
        uploader = st.file_uploader(
            "Drop or choose a chest X-ray image",
            type=["png", "jpg", "jpeg"],
            help="PNG, JPG, JPEG",
        )

        if uploader is not None:
            file_bytes = uploader.getvalue()
            upload_key = f"upload:{hash(file_bytes)}"
            if st.session_state.image_source_key != upload_key:
                st.session_state.current_image = load_image_from_bytes(file_bytes)
                st.session_state.analysis = None
                st.session_state.image_source_key = upload_key
                st.session_state.selected_example = None

        st.markdown('<div class="section-title">Image Preview</div>', unsafe_allow_html=True)

        if st.session_state.current_image is not None:
            st.markdown('<div class="preview-img">', unsafe_allow_html=True)
            st.image(st.session_state.current_image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="muted">Awaiting scan. Upload an image or choose an example below.</div>', unsafe_allow_html=True)

        render_example_grid(examples)

        analyze_clicked = st.button(
            "Analyze",
            type="primary",
            disabled=st.session_state.current_image is None,
            use_container_width=True,
        )

        if analyze_clicked and st.session_state.current_image is not None:
            progress_slot = st.empty()
            with st.spinner("Analyzing scan..."):
                pb = progress_slot.progress(0, text="Preparing model...")
                for pct, msg in [(18, "Preparing model..."), (48, "Running inference..."), (78, "Generating heatmap..."), (100, "Finalizing...")]:
                    pb.progress(pct, text=msg)
                    time.sleep(0.12)
                st.session_state.analysis = predict(st.session_state.current_image, artifacts)
            progress_slot.empty()

    with right_col:
        st.markdown('<div class="section-title">Diagnosis</div>', unsafe_allow_html=True)

        if st.session_state.analysis is None:
            st.markdown('<p class="diag-title">Awaiting scan</p>', unsafe_allow_html=True)
            pct = 0.0
            accent = "#35e9ff"
            label_class = ""
            pneumonia_prob = 0.0
            normal_prob = 0.0
        else:
            result = st.session_state.analysis
            label = str(result["label"])
            confidence = float(result["confidence"])
            pneumonia_prob = float(result["pneumonia_prob"])
            normal_prob = float(result["normal_prob"])
            pct = confidence * 100
            is_pneumonia = label == "PNEUMONIA"
            accent = "#ff6a7f" if is_pneumonia else "#26dea2"
            label_class = "pneumonia" if is_pneumonia else "normal"
            st.markdown(f'<p class="diag-title {label_class}">{label}</p>', unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="meter-wrap">
                <div class="meter-ring" style="--p:{pct:.2f}; --accent:{accent};">
                    <div class="meter-value">{pct:.1f}%</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="bar-wrap">
                <div class="bar-row">
                    <div class="bar-meta"><span>Pneumonia</span><span>{pneumonia_prob * 100:.1f}%</span></div>
                    <div class="bar-track"><div class="bar-fill pneumonia" style="width:{pneumonia_prob * 100:.1f}%;"></div></div>
                </div>
                <div class="bar-row">
                    <div class="bar-meta"><span>Normal</span><span>{normal_prob * 100:.1f}%</span></div>
                    <div class="bar-track"><div class="bar-fill normal" style="width:{normal_prob * 100:.1f}%;"></div></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-title">Grad-CAM Visualization</div>', unsafe_allow_html=True)

        toggle_col, slider_col = st.columns([0.38, 0.62], gap="small")
        with toggle_col:
            st.session_state.show_heatmap = st.toggle(
                "Show Heatmap",
                value=st.session_state.show_heatmap,
            )
        with slider_col:
            st.session_state.heatmap_alpha = st.slider(
                "Heatmap Intensity",
                min_value=0.1,
                max_value=1.0,
                value=float(st.session_state.heatmap_alpha),
                step=0.05,
                disabled=not st.session_state.show_heatmap,
            )

        if st.session_state.analysis is None:
            st.info("Heatmap will appear after analysis.")
        else:
            result = st.session_state.analysis
            if st.session_state.show_heatmap:
                blended = overlay_heatmap(
                    result["original"],
                    result["heatmap_raw"],
                    alpha=float(st.session_state.heatmap_alpha),
                )
                st.image(blended, use_container_width=True)
            else:
                st.image(result["original"], use_container_width=True)

        st.markdown('<div class="section-title">Report</div>', unsafe_allow_html=True)
        if st.session_state.analysis is None:
            render_report_with_copy("Awaiting analysis...")
        else:
            render_report_with_copy(str(st.session_state.analysis["report"]))

    st.caption("Research and educational use only. This is not a clinical diagnostic device.")


if __name__ == "__main__":
    main()
