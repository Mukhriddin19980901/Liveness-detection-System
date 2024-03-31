from ultralytics import YOLO

model  = YOLO('yolov8n.pt')

def main():
    model.train(data="Datasets/SplitDatas/data.yaml",epochs=300)

if __name__ == '__main__':
    main()