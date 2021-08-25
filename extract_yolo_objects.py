import torch
import os

yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# level of confidence
yolo_model.conf = 0.05

# only objects from class 0 [person]
yolo_model.classes = [0]

paintings_path = './data/paintings/'
paintings = os.listdir(paintings_path)

for painting in iter(paintings):
    painting_name = painting.split(".")[0]
    reproductions_path = f'./data/reproductions/{painting_name}/'
    reproductions = [reproductions_path + reproduction for reproduction in os.listdir(reproductions_path)]

    yolo_model(paintings_path + painting).crop(f'./objects/paintings/{painting_name}')
    yolo_model(reproductions).crop(f'./objects/reproductions/{painting_name}')
