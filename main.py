from ultralytics import YOLO
# if the env is not on windows, it is not necessary to add if __name__ ...
# Load a model
model = YOLO('/workspaces/SmokingDetector/ultralytics/models/v5/yolov5.yaml')  # build a new model from scratch
#model = YOLO('yolov5m.pt')  # load a pretrained model (recommended for training)

# Use the model
model.train(**('cfg':'/ultralytics/yolo/cfg/default.yaml'))  # train the model

