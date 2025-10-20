# streamlit_food_calories.py
# 日本語 UI・PIL で描画・predict.py 連携（Colab/トンネル関連は完全削除）
# 固定4クラス: bread / jelly / riceball / instant noodle

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# ---- Streamlit ページ設定 ----
st.set_page_config(page_title="Food Calories (YOLO11)", layout="wide")

# ---- Ultralytics / あなたの predict.py を利用 ----
from predict import predict as run_predict
from ultralytics import YOLO  # 構成維持のため（実推論は run_predict を使用）

# ---------------- アプリ基本設定 ---------------- #
TARGET_CLASSES = ["bread", "jelly", "riceball", "instant noodle"]

# 同一ディレクトリの best.pt を固定で使用（表示のみ）
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
DEFAULT_WEIGHTS = os.path.join(HERE, "best.pt")

@st.cache_resource(show_spinner=False)
def load_model(weights_path: str):
    # 構成維持のために置いてあります（実推論は run_predict を使用）
    try:
        if os.path.exists(weights_path):
            return YOLO(weights_path)
    except Exception:
        pass
    return None

# ---------- ヘッダー（日本語） ----------
st.title("🍽️ 画像内総カロリー推定 — YOLO11（固定4クラス）")
st.caption(
    "画像をアップロードしてください。対象クラスは固定：bread / jelly / riceball / instant noodle。"
    "フロントで各クラスの1個あたりカロリーを設定し、検出数×単価で総カロリーを算出します。"
)

with st.sidebar:
    st.header("モデルと推論")
    st.text_input("モデル重みのパス（固定）", value=DEFAULT_WEIGHTS, disabled=True,
                  help="このファイルと同じフォルダの best.pt を使用します。")
    conf = st.slider("信頼度 (conf)", 0.0, 1.0, 0.25, 0.01)

# 構成維持のためロード（無くても落ちないように None 許容）
_ = load_model(DEFAULT_WEIGHTS)

# ---------- カロリー設定（固定4行・フロント編集可） ----------
PRESET_KEY = "__fixed_calorie_preset__"
if PRESET_KEY not in st.session_state:
    st.session_state[PRESET_KEY] = pd.DataFrame([
        {"class_name": "bread",          "kcal_per_item": 200.0},
        {"class_name": "jelly",          "kcal_per_item": 100.0},
        {"class_name": "riceball",       "kcal_per_item": 180.0},
        {"class_name": "instant noodle", "kcal_per_item": 380.0},
    ])

with st.expander("カロリー設定（行固定・フロントで編集可）", expanded=True):
    preset_df: pd.DataFrame = st.data_editor(
        st.session_state[PRESET_KEY],
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "class_name": st.column_config.TextColumn("クラス名"),
            "kcal_per_item": st.column_config.NumberColumn("1個あたりのカロリー (kcal)", min_value=0.0, step=10.0),
        },
        key="editor_fixed",
    )
    st.session_state[PRESET_KEY] = preset_df

# ---------- PIL でバウンディングボックス & ラベル描画 ----------
def draw_detections_pil(base_img: Image.Image, det_df: pd.DataFrame, kcal_map: dict) -> Image.Image:
    img = base_img.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for _, r in det_df.iterrows():
        x1, y1, x2, y2 = map(float, (r["x1"], r["y1"], r["x2"], r["y2"]))
        name = str(r["class_name"])
        # 枠線
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        # ラベル（+kcal）
        kcal = int(kcal_map.get(name, 0))
        label = f"+{kcal} kcal"
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = (7 * len(label), 12)
        bx, by = int(x1), int(max(0, y1 - th - 4))
        draw.rectangle([bx, by, bx + tw + 6, by + th + 4], fill=(255, 255, 255))
        draw.text((bx + 3, by + 2), label, fill=(0, 0, 0), font=font)
    return img

# ---------- 画像アップロード ----------
up = st.file_uploader("画像をアップロード", type=["jpg", "jpeg", "png", "bmp", "webp"], accept_multiple_files=False)
img_col, table_col = st.columns([1.3, 0.7], gap="large")

# ---------- 推論 & 表示 ----------
if up is not None:
    # PILで読込（OpenCV不要）
    try:
        pil_img = Image.open(up).convert("RGB")
    except Exception:
        st.error("画像を解析できませんでした。別の画像でお試しください。")
        st.stop()

    # ★ 同一フォルダの predict.py の predict() を直接呼ぶ（PIL画像OK）
    result = run_predict(pil_img, conf=conf, imgsz=640)

    # 検出ボックス抽出
    det_rows = []
    if hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0:
        xyxy = result.boxes.xyxy.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        names = getattr(result, "names", None)  # クラス名は result.names を参照
        for i, (xy, ci, cf) in enumerate(zip(xyxy, clss, confs)):
            x1, y1, x2, y2 = map(float, xy)
            name = names.get(int(ci), str(ci)) if isinstance(names, dict) else str(ci)
            det_rows.append({
                "id": i,
                "class_id": int(ci),
                "class_name": name,
                "conf": float(cf),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })
    det_df = pd.DataFrame(det_rows)

    # 固定4クラスに限定
    if not det_df.empty:
        det_df = det_df[det_df["class_name"].isin(TARGET_CLASSES)].reset_index(drop=True)

    if det_df.empty:
        with img_col:
            st.info("指定の4クラスは検出されませんでした。bread / jelly / riceball / instant noodle を含む画像をご使用ください。")
            st.image(pil_img, use_column_width=True)
    else:
        # クラス別サマリー & 総カロリー
        counts = det_df.groupby("class_name").size().reset_index(name="count")
        preset_slim = preset_df[["class_name", "kcal_per_item"]].copy()
        merged = counts.merge(preset_slim, on="class_name", how="inner")
        merged["subtotal_kcal"] = merged["count"] * merged["kcal_per_item"]
        total_kcal = float(merged["subtotal_kcal"].sum())

        # 画像に +kcal を描画
        kcal_map = {r["class_name"]: float(r["kcal_per_item"]) for _, r in preset_slim.iterrows()}
        vis_img = draw_detections_pil(pil_img, det_df, kcal_map)

        with img_col:
            st.image(vis_img, use_column_width=True)
            st.metric("画像の総カロリー (kcal)", f"{int(total_kcal)}")

        with table_col:
            st.subheader("クラス別サマリー")
            st.dataframe(merged.sort_values("subtotal_kcal", ascending=False).reset_index(drop=True), use_container_width=True)

            st.subheader("検出ボックス一覧 (xyxy)")
            det_view = det_df[["class_name", "conf", "x1", "y1",_]()]()
