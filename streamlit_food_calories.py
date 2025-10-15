# streamlit_food_calories.py
# Cross-platform Cloudflared・日本語 UI・predict.py 連携・公開URLを実行結果に表示
# -------------------------------------------------------------------
# 固定4クラス: bread / jelly / riceball / instant noodle
# - 画像アップロード
# - 同一フォルダの predict.py の predict() を直接呼び出し
# - 1個あたりカロリーをフロントで設定 → 総カロリー算出
# - 検出ボックス (xyxy) 表示 + JSON（ダウンロードなし）
# - Cloudflared を自動DL & 起動（Win/macOS/Linux/Colab 対応）
# - 起動時に 公開URL を コンソールとページ本文 に即表示＆public_url.txt に保存

import os
import re
import cv2
import time
import json
import platform
import tarfile
import tempfile
import shutil
import threading
import subprocess
import urllib.request
import numpy as np
import pandas as pd
import streamlit as st

# ---- Streamlit ページ設定（最初に呼ぶのが推奨）----
st.set_page_config(page_title="Food Calories (YOLO11)", layout="wide")

# ★ 同一フォルダの predict.py を使用（あなたの実装を呼び出します）
from predict import predict as run_predict
from ultralytics import YOLO  # 構成維持のため残しています（推論は run_predict を使用）

# ===== Cloudflared（クロスプラットフォーム）=====
PORT = int(os.environ.get("STREAMLIT_SERVER_PORT", os.environ.get("PORT", "8501")))
URL_FILE = os.path.abspath("./public_url.txt")  # 公開URLの保存先

@st.cache_resource(show_spinner=False)
def _ensure_cloudflared() -> str:
    """
    Cross-platform downloader for cloudflared.
    Returns absolute path to the executable (cloudflared or cloudflared.exe).
    """
    candidates = [
        "/usr/local/bin/cloudflared",
        "/usr/bin/cloudflared",
        os.path.abspath("./cloudflared"),
        os.path.abspath("./cloudflared.exe"),
    ]
    for p in candidates:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p

    sys = platform.system().lower()
    arch = platform.machine().lower()

    # Windows
    if sys == "windows":
        asset = "cloudflared-windows-amd64.exe" if ("64" in arch or "x86_64" in arch or "amd64" in arch) else "cloudflared-windows-arm64.exe"
        url = f"https://github.com/cloudflare/cloudflared/releases/latest/download/{asset}"
        dest = os.path.abspath("./cloudflared.exe")
        urllib.request.urlretrieve(url, dest)
        os.chmod(dest, 0o755)
        return dest

    # macOS
    if sys == "darwin":
        tar_asset = "cloudflared-darwin-amd64.tgz" if ("x86_64" in arch or "amd64" in arch) else "cloudflared-darwin-arm64.tgz"
        url = f"https://github.com/cloudflare/cloudflared/releases/latest/download/{tar_asset}"
        with tempfile.TemporaryDirectory() as td:
            tgz = os.path.join(td, "cloudflared.tgz")
            urllib.request.urlretrieve(url, tgz)
            with tarfile.open(tgz, "r:gz") as tf:
                tf.extractall(td)
            extracted = os.path.join(td, "cloudflared")
            dest = os.path.abspath("./cloudflared")
            shutil.move(extracted, dest)
            os.chmod(dest, 0o755)
            return dest

    # Linux
    asset = "cloudflared-linux-amd64" if ("x86_64" in arch or "amd64" in arch) else "cloudflared-linux-arm64"
    url = f"https://github.com/cloudflare/cloudflared/releases/latest/download/{asset}"
    dest = os.path.abspath("./cloudflared")
    urllib.request.urlretrieve(url, dest)
    os.chmod(dest, 0o755)
    return dest


@st.cache_resource(show_spinner=False)
def _start_cloudflared(port: int) -> str:
    """
    Start cloudflared tunnel to http://localhost:{port} and return public URL.
    """
    # 旧プロセスを可能な範囲で終了
    try:
        if platform.system().lower() == "windows":
            subprocess.run(["taskkill", "/F", "/IM", "cloudflared.exe", "/T"], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.run(["pkill", "-f", "cloudflared"], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

    bin_path = _ensure_cloudflared()
    url_pat = re.compile(r"https://[-a-z0-9]+\.trycloudflare\.com")
    url_holder = {"url": ""}

    def _reader():
        proc = subprocess.Popen(
            [bin_path, "tunnel", "--url", f"http://localhost:{port}", "--no-autoupdate"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        # 読み取りループ
        for line in proc.stdout or []:
            m = url_pat.search(line)
            if m and not url_holder["url"]:
                url_holder["url"] = m.group(0)
                # ① コンソール（Colabセル出力）に即出力
                print(url_holder["url"], flush=True)
                # ② ファイルにも保存（バックアップ）
                try:
                    with open(URL_FILE, "w", encoding="utf-8") as f:
                        f.write(url_holder["url"])
                except Exception:
                    pass
                break

    threading.Thread(target=_reader, daemon=True).start()

    # 最大 ~20 秒待機
    for _ in range(80):
        if url_holder["url"]:
            break
        time.sleep(0.25)
    return url_holder["url"]

PUBLIC_URL = _start_cloudflared(PORT)
# 起動直後にも再出力（保険）
if PUBLIC_URL:
    print(PUBLIC_URL, flush=True)

# ---------------- アプリ基本設定 ---------------- #
TARGET_CLASSES = ["bread", "jelly", "riceball", "instant noodle"]

# ★ 同一ディレクトリの best.pt を固定で使用（表示のみ、入力不可）
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

# 本文にも公開URLを即表示（存在する場合）
if PUBLIC_URL:
    st.success(f"公開URL：{PUBLIC_URL}")
    st.code(PUBLIC_URL)

st.caption(
    "画像をアップロードしてください。対象クラスは固定：bread / jelly / riceball / instant noodle。"
    "フロントで各クラスの1個あたりカロリーを設定し、検出数×単価で総カロリーを算出します。"
)

with st.sidebar:
    st.header("外部アクセス")
    if PUBLIC_URL:
        st.success("Cloudflare トンネルを作成しました")
        st.markdown(f"**公開URL：** [{PUBLIC_URL}]({PUBLIC_URL})")
        st.code(PUBLIC_URL)
    else:
        st.info("公開URLを取得しています（Cloudflare トンネル）… 反応がない場合は再実行してください。")

    st.header("モデルと推論")
    st.text_input("モデル重みのパス（固定）", value=DEFAULT_WEIGHTS, disabled=True,
                  help="このファイルと同じフォルダの best.pt を使用します。")
    conf = st.slider("信頼度 (conf)", 0.0, 1.0, 0.25, 0.01)

# 構成維持のためロード（無くても落ちないように None 許容）
model = load_model(DEFAULT_WEIGHTS)

# ---------- カロリー設定（固定4行・フロント編集可） ----------
PRESET_KEY = "__fixed_calorie_preset__"
if PRESET_KEY not in st.session_state:
    st.session_state[PRESET_KEY] = pd.DataFrame([
        {"class_name": "bread", "kcal_per_item": 200.0},
        {"class_name": "jelly", "kcal_per_item": 100.0},
        {"class_name": "riceball", "kcal_per_item": 180.0},
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

# ---------- 画像アップロード ----------
up = st.file_uploader("画像をアップロード", type=["jpg", "jpeg", "png", "bmp", "webp"], accept_multiple_files=False)

img_col, table_col = st.columns([1.3, 0.7], gap="large")

# ---------- 推論 & 表示 ----------
if up is not None:
    data = np.frombuffer(up.read(), np.uint8)
    img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.error("画像を解析できませんでした。別の画像でお試しください。")
    else:
        # ★ 同一フォルダの predict.py の predict() を直接呼ぶ
        result = run_predict(img_bgr, conf=conf, imgsz=640)

        # 検出ボックス抽出
        det_rows = []
        if hasattr(result, "boxes") and result.boxes is not None and hasattr(result.boxes, "xyxy") and len(result.boxes) > 0:
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
            det_df = det_df[det_df["class_name"]].isin(TARGET_CLASSES)
            det_df = det_df[det_df["class_name"].isin(TARGET_CLASSES)].reset_index(drop=True)

        if det_df.empty:
            with img_col:
                st.info("指定の4クラスは検出されませんでした。bread / jelly / riceball / instant noodle を含む画像をご使用ください。")
                # 可視化（predict の返り値が Ultralytics Results を想定）
                vis_bgr = result.plot() if hasattr(result, "plot") else img_bgr.copy()
                vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
                st.image(vis_rgb, channels="RGB", use_column_width=True)
        else:
            # クラス別サマリー & 総カロリー
            counts = det_df.groupby("class_name").size().reset_index(name="count")
            preset_slim = preset_df[["class_name", "kcal_per_item"]].copy()
            merged = counts.merge(preset_slim, on="class_name", how="inner")
            merged["subtotal_kcal"] = merged["count"] * merged["kcal_per_item"]
            total_kcal = float(merged["subtotal_kcal"].sum())

            # 画像に +kcal を描画
            vis_bgr = result.plot() if hasattr(result, "plot") else img_bgr.copy()
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
                st.metric("画像の総カロリー (kcal)", f"{int(total_kcal)}")

            with table_col:
                st.subheader("クラス別サマリー")
                st.dataframe(merged.sort_values("subtotal_kcal", ascending=False).reset_index(drop=True), use_container_width=True)

                st.subheader("検出ボックス一覧 (xyxy)")
                det_view = det_df[["class_name", "conf", "x1", "y1", "x2", "y2"]].round(2)
                st.dataframe(det_view, use_container_width=True)
                with st.expander("JSON（コピー可）", expanded=False):
                    st.code(json.dumps(det_view.to_dict(orient='records'), ensure_ascii=False, indent=2), language="json")
else:
    st.info("上部で画像をアップロードしてください。対応形式: jpg/jpeg/png/bmp/webp。")