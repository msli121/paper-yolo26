# @Time       : 2026/2/28 14:52
# @File       : predict.py
# @Description:
import warnings

warnings.filterwarnings("ignore")
from ultralytics import YOLO

if __name__ == "__main__":
    # 1. 路径配置
    IMG_STACK = r"D:\PycharmProjects\robotic-grasping\yolo\data\1007_RGB.png"  # 堆叠场景
    IMG_SMALL = r"D:\PycharmProjects\robotic-grasping\yolo\data\1836_RGB.png"  # 小目标场景

    # model_path = r'D:\PycharmProjects\ultralytics\runs\detect\runs\26train\paper_yolo26s_ARF_MCA2\weights\best.pt'
    model_path = r"D:\PycharmProjects\ultralytics\runs\detect\runs\26train\paper_yolo26s\weights\best.pt"
    model = YOLO(model_path)  # 使用自己训练好的权重文件
    model.predict(
        source=IMG_SMALL,  # 预测的数据源，可以是图片、文件夹、视频路径等
        imgsz=640,  # 图片的输入尺寸
        project="runs/detect",  # 预测结果保存的项目目录
        name="exp",  # 子目录名（最终保存路径为 runs/detect/exp）
        save=True,  # 是否保存预测后的图片
        # visualize=True,              # 是否可视化中间特征图（默认注释掉，即不启用）
        # show_conf=False,             # 是否显示置信度（关闭显示）
        # show_labels=False,           # 是否显示标签（关闭显示）
    )
