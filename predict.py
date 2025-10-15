# predict.py
import os
import argparse
from typing import Any
from ultralytics import YOLO

# --------- Frontend-callable API (for Streamlit) --------- #
_MODEL = None

def _default_weights() -> str:
    # best.pt placed next to this file
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "best.pt")

def load_model(weights: str | None = None) -> YOLO:
    global _MODEL
    if _MODEL is None:
        _MODEL = YOLO(weights or _default_weights())
    return _MODEL

def predict(img_bgr, conf: float = 0.35, imgsz: int = 640) -> Any:
    """
    前端（Streamlit）直接调用的函数。
    入参：OpenCV の BGR 画像（np.ndarray）、閾値 conf、入力サイズ imgsz
    返値：Ultralytics の Results オブジェクト（.boxes / .plot() 等が使えます）
    """
    model = load_model()
    results = model.predict(img_bgr, conf=conf, imgsz=imgsz, verbose=False)
    res = results[0]
    # names を補完（可視化やクラス名出力に使用）
    if not hasattr(res, "names"):
        res.names = model.names
    return res

# --------- CLI remains compatible --------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="path to best.pt")
    ap.add_argument("--source",  required=True, help="image/video/folder")
    ap.add_argument("--imgsz",   type=int, default=512)
    ap.add_argument("--conf",    type=float, default=0.35)
    ap.add_argument("--save",    action="store_true")
    args = ap.parse_args()

    model = YOLO(args.weights)
    model.predict(source=args.source, imgsz=args.imgsz, conf=args.conf, save=args.save)

if __name__ == "__main__":
    main()
