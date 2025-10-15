
# streamlit_food_calories.py (Colab 一键公网隧道版)
# ---------------------------------
# Minimal Streamlit page for food-calorie demo (fixed 4 classes):
# Classes: bread / jelly / riceball / instant noodle
# - Upload one image
# - YOLO detection (Ultralytics)
# - Front-end table for per-class calories (kcal per item)
# - Compute & display total calories (count × per-class kcal)
#
# Colab 一键运行：
#   1) 可选安装（若环境未装）:
#        !pip -q install -U ultralytics streamlit opencv-python pillow pandas
#   2) 直接运行：
#        !streamlit run streamlit_food_calories.py
#   本脚本会在首次运行时自动下载并启动 cloudflared 隧道，
#   并在 Colab 的输出里打印公网地址，同时在应用侧边栏显示可点击链接。

import os
import re
import cv2
import time
import shutil
import threading
import subprocess
import urllib.request
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

# ---------------- Cloudflared 隧道（自动下载 & 启动） ---------------- #
PORT = int(os.environ.get("STREAMLIT_SERVER_PORT", os.environ.get("PORT", "8501")))

@st.cache_resource(show_spinner=False)
def _ensure_cloudflared(bin_hint: str = "/usr/local/bin/cloudflared") -> str:
    """确保 cloudflared 可用，不在则下载到 /usr/local/bin 或当前目录。返回可执行路径。"""
    candidates = [bin_hint, "/usr/bin/cloudflared", "./cloudflared"]
    for p in candidates:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    dest = bin_hint
    try:
        urllib.request.urlretrieve(url, dest)
        os.chmod(dest, 0o755)
        return dest
    except Exception:
        # 无权限写入 /usr/local/bin 时，退回到当前目录
        alt = "./cloudflared"
        urllib.request.urlretrieve(url, alt)
        os.chmod(alt, 0o755)
        return os.path.abspath(alt)

@st.cache_resource(show_spinner=False)
def _start_cloudflared(port: int) -> str:
    """启动 cloudflared 隧道，返回公网 URL（可能为空字符串，表示仍在获取）。"""
    # 尽量清理旧进程（忽略错误）
    try:
        subprocess.run(["pkill", "-f", "cloudflared"], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

    bin_path = _ensure_cloudflared()
    url_pat = re.compile(r"https://[-a-z0-9]+\.trycloudflare\.com")
    url_holder = {"url": ""}

    def _reader():
        # --no-autoupdate 避免自动升级卡住；stdout 合并方便解析
        proc = subprocess.Popen(
            [bin_path, "tunnel", "--url", f"http://localhost:{port}", "--no-autoupdate"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        # 把 PID 放到 session 里，方便后续清理
        st.session_state["__cf_pid__"] = proc.pid
        for line in proc.stdout:  # 持续读取，避免管道阻塞
            m = url_pat.search(line)
            if m and not url_holder["url"]:
                url_holder["url"] = m.group(0)
                # 打印到控制台（Colab 单元格里可见）
                print("🌍 Public URL:", url_holder["url"], flush=True)

    threading.Thread(target=_reader, daemon=True).start()

    # 等待最多 ~20s 获取 URL（UI 侧会继续显示占位提示）
    for _ in range(80):
        if url_holder["url"]:
            break
        time.sleep(0.25)
    return url_holder["url"]

PUBLIC_URL = _start_cloudflared(PORT)

# ---------------- App 基本设置 ---------------- #
TARGET_CLASSES = ["bread", "jelly", "riceball", "instant noodle"]

st.set_page_config(page_title="Food Calories (YOLO11)", layout="wide")

# ---------- Model ----------
@st.cache_resource(show_spinner=False)
def load_model(weights_path: str):
    return YOLO(weights_path)

# ---------- 顶部标题 ----------
st.title("🍽️ 食物总卡路里估算 — YOLO11（固定四类）")
st.caption(
    "上传一张图片；固定类别：bread / jelly / riceball / instant noodle。前端设置每类每份卡路里，按检测\"数量×单份卡路里\"累计总量。"
)

with st.sidebar:
    st.header("公网访问")
    if PUBLIC_URL:
        st.success("已创建 Cloudflare 隧道")
        st.markdown(f"**公网地址：** [{PUBLIC_URL}]({PUBLIC_URL})")
        st.code(PUBLIC_URL)
    else:
        st.info("正在申请公网地址（Cloudflare 隧道）… 若长时间无响应，可重启或重新运行脚本。")

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
