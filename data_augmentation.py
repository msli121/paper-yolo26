# @Time       : 2026/2/28 15:32
# @File       : data_augmentation.py
# @Description:
# random_flip_horizon：几何增强，水平翻转目标，提高对镜像图像的识别能力
# random_flip_vertical：几何增强，垂直翻转目标，适应遥感图像等特殊场景
# center_crop：裁剪增强，提取图像中央区域，提升模型对中心目标的关注度
# random_bright：光照增强，模拟不同亮度环境
# random_contrast：光照增强，提高或降低图像对比度
# random_saturation：颜色增强，模拟色彩变化，增强模型对多样图像风格的泛化能力
# add_gaussian_noise：噪声增强，添加高斯噪声，增强对模糊图像的鲁棒性
# add_salt_noise：噪声增强，添加白点噪声，模拟摄像头干扰
# add_pepper_noise：噪声增强，添加黑点噪声，增强抗干扰能力

import torch
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import random

import numpy as np
from torchvision import transforms

random.seed(0)


class DataAugmentationOnDetection:
    def __init__(self):
        super().__init__()

    def resize_keep_ratio(self, image, boxes, target_size):
        old_size = image.size[0:2]
        ratio = min(float(target_size) / (old_size[i]) for i in range(len(old_size)))
        new_size = tuple([int(i * ratio) for i in old_size])
        return image.resize(new_size, Image.BILINEAR), boxes

    def resizeDown_keep_ratio(self, image, boxes, target_size):
        old_size = image.size[0:2]
        ratio = min(float(target_size) / (old_size[i]) for i in range(len(old_size)))
        ratio = min(ratio, 1)
        new_size = tuple([int(i * ratio) for i in old_size])

        return image.resize(new_size, Image.BILINEAR), boxes

    def resize(self, img, boxes, size):
        return img.resize((size, size), Image.BILINEAR), boxes

    def random_flip_horizon(self, img, boxes, h_rate=1):
        if np.random.random() < h_rate:
            transform = transforms.RandomHorizontalFlip(p=1)
            img = transform(img)
            if len(boxes) > 0:
                x = 1 - boxes[:, 1]
                boxes[:, 1] = x
        return img, boxes

    def random_flip_vertical(self, img, boxes, v_rate=1):
        if np.random.random() < v_rate:
            transform = transforms.RandomVerticalFlip(p=1)
            img = transform(img)
            if len(boxes) > 0:
                y = 1 - boxes[:, 2]
                boxes[:, 2] = y
        return img, boxes

    def center_crop(self, img, boxes, target_size=None):
        w, h = img.size
        size = min(w, h)
        if len(boxes) > 0:
            label = boxes[:, 0].reshape([-1, 1])
            x_, y_, w_, h_ = boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
            x1 = (w * x_ - 0.5 * w * w_).reshape([-1, 1])
            y1 = (h * y_ - 0.5 * h * h_).reshape([-1, 1])
            x2 = (w * x_ + 0.5 * w * w_).reshape([-1, 1])
            y2 = (h * y_ + 0.5 * h * h_).reshape([-1, 1])
            boxes_xyxy = torch.cat([x1, y1, x2, y2], dim=1)
            if w > h:
                boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]] - (w - h) / 2
            else:
                boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]] - (h - w) / 2
            in_boundary = [i for i in range(boxes_xyxy.shape[0])]
            for i in range(boxes_xyxy.shape[0]):
                if (boxes_xyxy[i, 0] < 0 and boxes_xyxy[i, 2] < 0) or (
                    boxes_xyxy[i, 0] > size and boxes_xyxy[i, 2] > size
                ):
                    in_boundary.remove(i)
                elif (boxes_xyxy[i, 1] < 0 and boxes_xyxy[i, 3] < 0) or (
                    boxes_xyxy[i, 1] > size and boxes_xyxy[i, 3] > size
                ):
                    in_boundary.append(i)
            boxes_xyxy = boxes_xyxy[in_boundary]
            boxes = boxes_xyxy.clamp(min=0, max=size).reshape([-1, 4])
            label = label[in_boundary]
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            xc = ((x1 + x2) / (2 * size)).reshape([-1, 1])
            yc = ((y1 + y2) / (2 * size)).reshape([-1, 1])
            wc = ((x2 - x1) / size).reshape([-1, 1])
            hc = ((y2 - y1) / size).reshape([-1, 1])
            boxes = torch.cat([xc, yc, wc, hc], dim=1)
        transform = transforms.CenterCrop(size)
        img = transform(img)
        if target_size:
            img = img.resize((target_size, target_size), Image.BILINEAR)
        if len(boxes) > 0:
            return img, torch.cat([label.reshape([-1, 1]), boxes], dim=1)
        else:
            return img, boxes

    def random_bright(self, img, u=120, p=1):

        if np.random.random() < p:
            alpha = np.random.uniform(-u, u) / 255
            img += alpha
            img = img.clamp(min=0.0, max=1.0)
        return img

    def random_contrast(self, img, lower=0.5, upper=1.5, p=1):
        if np.random.random() < p:
            alpha = np.random.uniform(lower, upper)
            img *= alpha
            img = img.clamp(min=0, max=1.0)
        return img

    def random_saturation(self, img, lower=0.5, upper=1.5, p=1):
        if np.random.random() < p:
            alpha = np.random.uniform(lower, upper)
            img[1] = img[1] * alpha
            img[1] = img[1].clamp(min=0, max=1.0)
        return img

    def add_gasuss_noise(self, img, mean=0, std=0.1):
        noise = torch.normal(mean, std, img.shape)
        img += noise
        img = img.clamp(min=0, max=1.0)
        return img

    def add_salt_noise(self, img):
        noise = torch.rand(img.shape)
        alpha = np.random.random() / 5 + 0.7
        img[noise[:, :, :] > alpha] = 1.0
        return img

    def add_pepper_noise(self, img):
        noise = torch.rand(img.shape)
        alpha = np.random.random() / 5 + 0.7
        img[noise[:, :, :] > alpha] = 0
        return img


def get_image_list(image_path):
    files_list = []
    for root, sub_dirs, files in os.walk(image_path):
        for special_file in files:
            special_file = special_file[0 : len(special_file)]
            files_list.append(special_file)
    return files_list


def get_label_file(label_path, image_name):
    fname = os.path.join(label_path, image_name[0 : len(image_name) - 4] + ".txt")
    data2 = []
    if not os.path.exists(fname):
        return data2
    if os.path.getsize(fname) == 0:
        return data2
    else:
        with open(fname, encoding="utf-8") as infile:
            for line in infile:
                data_line = line.strip("\n").split()
                data2.append([float(i) for i in data_line])
    return data2


def save_Yolo(img, boxes, save_path, prefix, image_name):
    if not os.path.exists(save_path) or not os.path.exists(os.path.join(save_path, "images")):
        os.makedirs(os.path.join(save_path, "images"))
        os.makedirs(os.path.join(save_path, "labels"))
    try:
        img.save(os.path.join(save_path, "images", prefix + image_name))
        with open(
            os.path.join(save_path, "labels", prefix + image_name[0 : len(image_name) - 4] + ".txt"),
            "w",
            encoding="utf-8",
        ) as f:
            if len(boxes) > 0:
                for data in boxes:
                    str_in = ""
                    for i, a in enumerate(data):
                        if i == 0:
                            str_in += str(int(a))
                        else:
                            str_in += " " + str(float(a))
                    f.write(str_in + "\n")
    except:
        print("ERROR: ", image_name, " is bad.")


def runAugumentation(image_path, label_path, save_path):
    image_list = get_image_list(image_path)
    for image_name in image_list:
        print("dealing: " + image_name)
        img = Image.open(os.path.join(image_path, image_name))
        boxes = get_label_file(label_path, image_name)
        boxes = torch.tensor(boxes)
        DAD = DataAugmentationOnDetection()

        t_img, t_boxes = DAD.random_flip_horizon(img, boxes.clone())
        save_Yolo(t_img, t_boxes, save_path, prefix="fh_", image_name=image_name)

        t_img, t_boxes = DAD.random_flip_vertical(img, boxes.clone())
        save_Yolo(t_img, t_boxes, save_path, prefix="fv_", image_name=image_name)

        t_img, t_boxes = DAD.center_crop(img, boxes.clone(), 1024)
        save_Yolo(t_img, t_boxes, save_path, prefix="cc_", image_name=image_name)

        to_tensor = transforms.ToTensor()
        to_image = transforms.ToPILImage()
        img = to_tensor(img)

        t_img, t_boxes = DAD.random_bright(img.clone()), boxes
        save_Yolo(to_image(t_img), boxes, save_path, prefix="rb_", image_name=image_name)

        t_img, t_boxes = DAD.random_contrast(img.clone()), boxes
        save_Yolo(to_image(t_img), boxes, save_path, prefix="rc_", image_name=image_name)

        t_img, t_boxes = DAD.random_saturation(img.clone()), boxes
        save_Yolo(to_image(t_img), boxes, save_path, prefix="rs_", image_name=image_name)

        t_img, t_boxes = DAD.add_gasuss_noise(img.clone()), boxes
        save_Yolo(to_image(t_img), boxes, save_path, prefix="gn_", image_name=image_name)

        t_img, t_boxes = DAD.add_salt_noise(img.clone()), boxes
        save_Yolo(to_image(t_img), boxes, save_path, prefix="sn_", image_name=image_name)

        t_img, t_boxes = DAD.add_pepper_noise(img.clone()), boxes
        save_Yolo(to_image(t_img), boxes, save_path, prefix="pn_", image_name=image_name)

        print("end: " + image_name)


if __name__ == "__main__":
    image_path = r"E:\CSDN_yolo\YOLO26_AIguai\datasets\data_name\images\train"
    label_path = r"E:\CSDN_yolo\YOLO26_AIguai\datasets\data_name\labels\train"
    save_path = r"E:\CSDN_yolo\YOLO26_AIguai\datasets\data_name\results"
    runAugumentation(image_path, label_path, save_path)
