# @Time       : 2026/2/28 14:42
# @File       : train.py.py
# @Description:
# 官网训练参数 https://docs.ultralytics.com/zh/modes/train/#train-settings
# 工业级验证 https://juejin.cn/post/7596874392823611434
import warnings

warnings.filterwarnings("ignore")
from ultralytics import YOLO


def train_yolov8s():
    # 模型配置文件
    model_yaml_path = r"D:\PycharmProjects\paper-yolo26\ultralytics\cfg\models\add26\yolov8s.yaml"  # 创建一个 YOLO 模型实例，加载 yolo26n.yaml 配置文件（Nano 版本）
    # 预训练模型
    pretrained_model_path = r"D:\PycharmProjects\paper-yolo26\pretrained_models\yolov8s.pt"
    # 数据集配置文件
    data_yaml_path = r"D:\PycharmProjects\paper-yolo26\datasets\paper\dataset.yaml"  # 指定数据集的配置文件，这里使用的是 data.yaml 配置文件
    model = YOLO(model_yaml_path)
    model.load(pretrained_model_path)  # 加载预训练权重，如果模型改动很大就不加载预训练权重（自选）
    # 训练模型
    model.train(
        data=data_yaml_path,  # 指定数据集的配置文件
        imgsz=640,  # 输入图像的尺寸，所有输入图像将被缩放到 640x640 像素
        epochs=150,  # 小样本必调：训练轮数，200足够，配合早停
        batch=16,  # 小样本必调：批次大小，2/4/8/16，显存够选4，不够选2
        device=0,  # GPU编号，CPU写cpu
        workers=4,
        save_period=10,  # 保存模型的周期，每训练 10 个周期保存一次模型
        save=True,  # 保存最优权重
        patience=50,
        project="runs/26train",  # 保存训练结果文件夹
        name="yolov8s",
    )


def train_yolo11s():
    # 模型配置文件
    model_yaml_path = r"D:\PycharmProjects\paper-yolo26\ultralytics\cfg\models\add26\yolo11s.yaml"  # 创建一个 YOLO 模型实例，加载 yolo26n.yaml 配置文件（Nano 版本）
    # 预训练模型
    pretrained_model_path = r"D:\PycharmProjects\paper-yolo26\pretrained_models\yolo11s.pt"
    # 数据集配置文件
    data_yaml_path = r"D:\PycharmProjects\paper-yolo26\datasets\paper\dataset.yaml"  # 指定数据集的配置文件，这里使用的是 data.yaml 配置文件
    model = YOLO(model_yaml_path)
    model.load(pretrained_model_path)  # 加载预训练权重，如果模型改动很大就不加载预训练权重（自选）
    # 训练模型
    model.train(
        data=data_yaml_path,  # 指定数据集的配置文件
        imgsz=640,  # 输入图像的尺寸，所有输入图像将被缩放到 640x640 像素
        epochs=150,  # 小样本必调：训练轮数，200足够，配合早停
        batch=16,  # 小样本必调：批次大小，2/4/8/16，显存够选4，不够选2
        device=0,  # GPU编号，CPU写cpu
        workers=4,
        save_period=10,  # 保存模型的周期，每训练 10 个周期保存一次模型
        save=True,  # 保存最优权重
        patience=50,
        project="runs/26train",  # 保存训练结果文件夹
        name="yolo11s",
    )


def train_yolo26s():
    # 模型配置文件
    model_yaml_path = r"D:\PycharmProjects\paper-yolo26\ultralytics\cfg\models\add26\yolo26s.yaml"  # 创建一个 YOLO 模型实例，加载 yolo26n.yaml 配置文件（Nano 版本）
    # 数据集配置文件
    data_yaml_path = r"D:\PycharmProjects\paper-yolo26\datasets\paper\dataset.yaml"  # 指定数据集的配置文件，这里使用的是 data.yaml 配置文件
    # 预训练模型
    pretrained_model_path = r"D:\PycharmProjects\paper-yolo26\pretrained_models\yolo26s.pt"
    model = YOLO(model_yaml_path)
    model.load(pretrained_model_path)  # 加载预训练权重，如果模型改动很大就不加载预训练权重（自选）
    # 训练模型
    model.train(
        **common_args,
        data=data_yaml_path,  # 指定数据集的配置文件
        name="yolo26s",
    )


def train_yolo26s_ARF():
    # 模型配置文件
    model_yaml_path = r"ultralytics/cfg/models/add26/yolo26s_C3k2_ARF.yaml"  # 创建一个 YOLO 模型实例，加载 yolo26n.yaml 配置文件（Nano 版本）
    # 数据集配置文件
    data_yaml_path = r"D:\PycharmProjects\paper-yolo26\datasets\paper\dataset.yaml"  # 指定数据集的配置文件，这里使用的是 data.yaml 配置文件
    # 预训练模型
    pretrained_model_path = r"D:\PycharmProjects\paper-yolo26\pretrained_models\yolo26s.pt"
    model = YOLO(model_yaml_path)
    model.load(pretrained_model_path)  # 加载预训练权重，如果模型改动很大就不加载预训练权重（自选）
    # 训练模型
    model.train(
        **common_args,
        data=data_yaml_path,  # 指定数据集的配置文件
        name="yolo26s_ARF",
    )


def train_yolo26s_MCA():
    # 模型配置文件
    model_yaml_path = r"ultralytics/cfg/models/add26/yolo26s_C3k2_MCA.yaml"  # 创建一个 YOLO 模型实例，加载 yolo26n.yaml 配置文件（Nano 版本）
    # 数据集配置文件
    data_yaml_path = r"D:\PycharmProjects\paper-yolo26\datasets\paper\dataset.yaml"  # 指定数据集的配置文件，这里使用的是 data.yaml 配置文件
    # 预训练模型
    pretrained_model_path = r"D:\PycharmProjects\paper-yolo26\pretrained_models\yolo26s.pt"
    model = YOLO(model_yaml_path)
    model.load(pretrained_model_path)  # 加载预训练权重，如果模型改动很大就不加载预训练权重（自选）
    # 训练模型
    model.train(
        data=data_yaml_path,  # 指定数据集的配置文件
        imgsz=640,  # 输入图像的尺寸，所有输入图像将被缩放到 640x640 像素
        epochs=150,  # 小样本必调：训练轮数，200足够，配合早停
        batch=16,  # 小样本必调：批次大小，2/4/8/16，显存够选4，不够选2
        device=0,  # GPU编号，CPU写cpu
        workers=4,
        save_period=10,  # 保存模型的周期，每训练 10 个周期保存一次模型
        save=True,  # 保存最优权重
        patience=50,
        project="runs/26train",  # 保存训练结果文件夹
        name="yolo26s_ARF_MCA",
    )


def train_yolo26s_STAL():
    # 模型配置文件
    model_yaml_path = r"D:\PycharmProjects\paper-yolo26\ultralytics\cfg\models\add26\yolo26s.yaml"  # 创建一个 YOLO 模型实例，加载 yolo26n.yaml 配置文件（Nano 版本）
    # 数据集配置文件
    data_yaml_path = r"D:\PycharmProjects\paper-yolo26\datasets\paper\dataset.yaml"  # 指定数据集的配置文件，这里使用的是 data.yaml 配置文件
    # 预训练模型
    pretrained_model_path = r"D:\PycharmProjects\paper-yolo26\pretrained_models\yolo26s.pt"
    model = YOLO(model_yaml_path)
    model.load(pretrained_model_path)  # 加载预训练权重，如果模型改动很大就不加载预训练权重（自选）
    # 训练模型
    model.train(
        data=data_yaml_path,  # 指定数据集的配置文件
        imgsz=640,  # 输入图像的尺寸，所有输入图像将被缩放到 640x640 像素
        epochs=150,  # 小样本必调：训练轮数，200足够，配合早停
        batch=16,  # 小样本必调：批次大小，2/4/8/16，显存够选4，不够选2
        device=0,  # GPU编号，CPU写cpu
        workers=4,
        save_period=10,  # 保存模型的周期，每训练 10 个周期保存一次模型
        save=True,  # 保存最优权重
        patience=50,
        project="runs/26train",  # 保存训练结果文件夹
        name="yolo26s_STAL",
    )


def train_yolo26s_ARF_MCA():
    # 模型配置文件
    model_yaml_path = r"ultralytics/cfg/models/add26/yolo26s_ARF_MCA.yaml"  # 创建一个 YOLO 模型实例，加载 yolo26n.yaml 配置文件（Nano 版本）
    # 数据集配置文件
    data_yaml_path = r"D:\PycharmProjects\paper-yolo26\datasets\paper\dataset.yaml"  # 指定数据集的配置文件，这里使用的是 data.yaml 配置文件
    # 预训练模型
    pretrained_model_path = r"D:\PycharmProjects\paper-yolo26\pretrained_models\yolo26s.pt"
    model = YOLO(model_yaml_path)
    model.load(pretrained_model_path)  # 加载预训练权重，如果模型改动很大就不加载预训练权重（自选）
    # 训练模型
    model.train(
        **common_args,
        data=data_yaml_path,  # 指定数据集的配置文件
        name="yolo26s_ARF_MCA",
    )


if __name__ == "__main__":
    common_args = dict(
        imgsz=640,  # 输入图像的尺寸，所有输入图像将被缩放到 640x640 像素
        epochs=150,  # 小样本必调：训练轮数，200足够，配合早停
        batch=16,  # 小样本必调：批次大小，2/4/8/16，显存够选4，不够选2
        device=0,  # GPU编号，CPU写cpu
        workers=4,
        save_period=10,  # 保存模型的周期，每训练 10 个周期保存一次模型
        save=True,  # 保存最优权重
        patience=50,
        project="runs/common_train_args",  # 保存训练结果文件夹
    )
    # 训练yolov8s
    # train_yolov8s()

    # 训练yolov11s
    # train_yolov11s()

    # 训练yolo26s模型
    train_yolo26s()

    # 训练yolo26s_ARF模型
    # train_yolo26s_ARF()

    # 训练yolo26s_MCA模型
    # train_yolo26s_MCA()

    # 训练yolo26s_ARF_MCA模型
    # train_yolo26s_ARF_MCA()
