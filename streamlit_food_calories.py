# streamlit_food_calories.py
# ---------------------------------
# Minimal Streamlit page for food-calorie demo (fixed 4 classes):
# Classes: bread / jelly / riceball / instant noodle
# - Upload one image
# - YOLO detection (Ultralytics)
# - Front-end table for per-class calories (kcal per item)
# - Compute & display total calories (count × per-class kcal)
#
# Run:
#   pip install -U ultralytics streamlit opencv-python pillow pandas
#   streamlit run streamlit_food_calories.py

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

TARGET_CLASSES = ["bread", "jelly", "riceball", "instant noodle"]

st.set_page_config(page_title="Food Calories (YOLO11)", layout="wide")

# ---------- Model ----------
@st.cache_resource(show_spinner=False)
def load_model(weights_path: str):
    return YOLO(weights_path)

# ---------- UI ----------
st.title("🍽️ 食物总卡路里估算 — YOLO11（固定四类）")
st.caption("上传一张图片；固定类别：bread / jelly / riceball / instant noodle。前端设置每类每份卡路里，按检测\"数量×单份卡路里\"累计总量。")

with st.sidebar:
    st.header("模型与推理")
    weights = st.text_input("模型权重路径", value="yolo11n.pt", help="建议换成你的自训权重，例如 runs/detect/train/weights/best.pt")
    conf = st.slider("置信度 (conf)", 0.0, 1.0, 0.25, 0.01)

model = load_model(weights)

# ---------- Preset calories (front-end editable, fixed rows) ----------
PRESET_KEY = "__fixed_calorie_preset__"
if PRESET_KEY not in st.session_state:
    st.session_state[PRESET_KEY] = pd.DataFrame([
        {"class_name": "bread", "kcal_per_item": 200.0},
        {"class_name": "jelly", "kcal_per_item": 100.0},
        {"class_name": "riceball", "kcal_per_item": 180.0},
        {"class_name": "instant noodle", "kcal_per_item": 380.0},
    ])

with st.expander("预设卡路里（可在前端修改，行数固定）", expanded=True):
    preset_df: pd.DataFrame = st.data_editor(
        st.session_state[PRESET_KEY],
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "class_name": st.column_config.TextColumn("类别名"),
            "kcal_per_item": st.column_config.NumberColumn("每份卡路里 (kcal)", min_value=0.0, step=10.0),
        },
        key="editor_fixed",
    )
    st.session_state[PRESET_KEY] = preset_df

# ---------- Image Upload ----------
up = st.file_uploader("上传图片", type=["jpg", "jpeg", "png", "bmp", "webp"], accept_multiple_files=False)

img_col, table_col = st.columns([1.3, 0.7], gap="large")

# ---------- Inference & Display ----------
if up is not None:
    data = np.frombuffer(up.read(), np.uint8)
    img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.error("无法解析图片文件。请重试或更换图片。")
    else:
        results = model.predict(img_bgr, conf=conf, imgsz=640, verbose=False)
        result = results[0]

        # Collect detections
        det_rows = []
        if result.boxes is not None and hasattr(result.boxes, "xyxy") and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            for i, (xy, ci, cf) in enumerate(zip(xyxy, clss, confs)):
                x1, y1, x2, y2 = map(float, xy)
                name = model.names.get(int(ci), str(ci)) if hasattr(model, "names") else str(ci)
                det_rows.append({
                    "id": i,
                    "class_id": int(ci),
                    "class_name": name,
                    "conf": float(cf),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                })
        det_df = pd.DataFrame(det_rows)

        # Filter to TARGET_CLASSES only
        if not det_df.empty:
            det_df = det_df[det_df["class_name"].isin(TARGET_CLASSES)].reset_index(drop=True)

        # If nothing detected (in our 4 targets)
        if det_df.empty:
            with img_col:
                st.info("未检测到指定的四类目标。请使用包含 bread/jelly/riceball/instant noodle 的图片，或换用你的训练权重。")
                vis_bgr = result.plot()
                vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
                st.image(vis_rgb, channels="RGB", use_column_width=True)
        else:
            # Aggregate counts
            counts = det_df.groupby("class_name").size().reset_index(name="count")
            preset_slim = preset_df[["class_name", "kcal_per_item"]].copy()
            merged = counts.merge(preset_slim, on="class_name", how="inner")
            merged["subtotal_kcal"] = merged["count"] * merged["kcal_per_item"]
            total_kcal = float(merged["subtotal_kcal"].sum())

            # Overlay per-detection calories
            vis_bgr = result.plot()  # default labels
            kcal_map = {r["class_name"]: float(r["kcal_per_item"]) for _, r in preset_slim.iterrows()}
            for _, r in det_df.iterrows():
                name = r["class_name"]
                if name not in kcal_map:
                    continue
                x1, y1 = int(r["x1"]), int(r["y1"]) - 6
                label = f"+{int(kcal_map[name])} kcal"
                cv2.putText(vis_bgr, label, (x1, max(12, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)

            with img_col:
                st.image(vis_rgb, channels="RGB", use_column_width=True)
                st.metric("图片总卡路里 (kcal)", f"{int(total_kcal)}")

            with table_col:
                st.subheader("按类别汇总")
                st.dataframe(merged.sort_values("subtotal_kcal", ascending=False).reset_index(drop=True), use_container_width=True)
else:
    st.info("请在上方上传一张图片。支持 jpg/jpeg/png/bmp/webp。")
