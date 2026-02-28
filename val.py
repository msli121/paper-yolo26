# -*- coding: utf-8 -*-
# @Time       : 2026/2/28 15:43
# @File       : val.py.py
# @Description:
from ultralytics import YOLO
import warnings

warnings.filterwarnings('ignore')
# 模型配置文件
model_yaml_path = r"./runs/26train/exp/weights/best.pt"
# 数据集配置文件
data_yaml_path = r'./datasets/data.yaml'

if __name__ == '__main__':
    model = YOLO(model_yaml_path)
    model.val(data=data_yaml_path,
              split='val',
              imgsz=640,
              batch=8,
              project='runs/val',
              name='exp',
              save_json=True
              )