import argparse
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--format", default="onnx", choices=["onnx", "torchscript", "openvino", "engine"])
    ap.add_argument("--imgsz", type=int, default=512)
    args = ap.parse_args()

    model = YOLO(args.weights)
    model.export(format=args.format, imgsz=args.imgsz)

if __name__ == "__main__":
    main()
