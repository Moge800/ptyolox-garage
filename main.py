"""PTYOLOX Garage entry point.

GUI 起動:
    uv run main.py

ヘッドレス使用例:
    from ptyolox_garage import YOLOX

    model = YOLOX("l")
    model.train(data="data.yaml", epochs=[100, 200, 300], device="cuda:0", batch=16)

    model = YOLOX("yolox_l.pt")
    results = model.predict("image.jpg", conf=0.3)
    model.export(format="onnx")
"""


def main() -> None:
    from ptyolox_garage.gui.app import main as _gui_main

    _gui_main()


if __name__ == "__main__":
    main()
