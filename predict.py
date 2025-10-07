import argparse
from ultralytics import YOLO

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
