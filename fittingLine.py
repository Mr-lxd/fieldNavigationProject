import os
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
from scipy.odr import ODR, Model, Data
import matplotlib.pyplot as plt


# 读取 YOLOv8 输出文件
def read_yolo_txt(file_path, image_width, image_height):
    bounding_boxes = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            class_id, center_x, center_y, width, height = map(float, line.strip().split())
            # 将相对坐标转换为绝对像素坐标
            center_x_abs = center_x * image_width
            center_y_abs = center_y * image_height
            width_abs = width * image_width
            height_abs = height * image_height
            x_min = int(center_x_abs - width_abs / 2)
            x_max = int(center_x_abs + width_abs / 2)
            y_min = int(center_y_abs - height_abs / 2)
            y_max = int(center_y_abs + height_abs / 2)
            bounding_boxes.append([x_min, y_min, x_max, y_max, center_x_abs, center_y_abs])
    return bounding_boxes


# 聚类检测框的中心点（调整权重）
def cluster_points_with_weights(points, x_weight=1.5, y_weight=0.5, eps=700, min_samples=2):
    # 对点集进行加权变换
    weighted_points = np.copy(points)
    weighted_points[:, 0] *= x_weight  # 放大 x 方向
    weighted_points[:, 1] *= y_weight  # 缩小 y 方向

    # 使用加权后的点集进行 DBSCAN 聚类
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(weighted_points)
    labels = clustering.labels_
    return labels


# 定义正交拟合模型
def linear_model(beta, x):
    return beta[0] * x + beta[1]  # y = slope * x + intercept


def orthogonal_regression(points):
    x = points[:, 0]
    y = points[:, 1]
    data = Data(x, y)
    model = Model(linear_model)
    odr = ODR(data, model, beta0=[0, 0])  # 初始参数 [slope, intercept]
    output = odr.run()
    return output.beta  # 返回拟合参数 [slope, intercept]


# 主函数，处理检测框
def process_labels_and_images_with_clustering(label_dir, image_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):
            label_path = os.path.join(label_dir, label_file)
            image_file = label_file.replace('.txt', '.jpg')
            image_path = os.path.join(image_dir, image_file)
            if not os.path.exists(image_path):
                print(f"Image not found for label: {label_file}")
                continue

            # 读取图像
            image = cv2.imread(image_path)
            image_height, image_width = image.shape[:2]

            # 读取检测框中心点
            bounding_boxes = read_yolo_txt(label_path, image_width, image_height)
            middle_start = image_width * 0.4
            middle_end = image_width * 0.65
            filtered_boxes = [
                box for box in bounding_boxes if middle_start <= box[4] <= middle_end
            ]
            filtered_points = [(box[4], box[5]) for box in filtered_boxes]

            # 聚类分析
            if len(filtered_points) > 1:
                points_array = np.array(filtered_points)

                # 使用调整权重后的聚类方法
                labels = cluster_points_with_weights(points_array, x_weight=1.5, y_weight=0.5, eps=700, min_samples=2)
                unique_labels = set(labels)
                unique_labels.discard(-1)  # 去除噪声点

                # 选择主要作物行（点最多的簇）
                main_cluster = max(unique_labels, key=list(labels).count)
                main_points = points_array[labels == main_cluster]

                # 绘制主要簇的检测框
                for box, label in zip(filtered_boxes, labels):
                    if label == main_cluster:
                        x_min, y_min, x_max, y_max, _, _ = box
                        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 10)  # 蓝色框

                # 正交拟合
                if len(main_points) > 1:
                    slope, intercept = orthogonal_regression(main_points)

                    # 绘制拟合直线
                    x_line = np.linspace(0, image_width, num=1000)
                    y_line = slope * x_line + intercept
                    for i in range(len(x_line) - 1):
                        x1, y1 = int(x_line[i]), int(y_line[i])
                        x2, y2 = int(x_line[i + 1]), int(y_line[i + 1])
                        if 0 <= y1 < image_height and 0 <= y2 < image_height:
                            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 20)  # 红色实线

                # 保存结果
                output_path = os.path.join(output_dir, image_file)
                cv2.imwrite(output_path, image)
                print(f"Processed and saved: {output_path}")

                # 显示处理结果
            #     plt.figure(figsize=(10, 8))
            #     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            #     plt.title(f"Processed: {image_file}")
            #     plt.axis('off')
            #     plt.show()
            # else:
            #     print(f"Not enough points to cluster for: {label_file}")


# 输入文件夹路径
label_dir = "D:/DL_Project/ultralytics-main/runs/detect/predict_test_ricetest/labels"
image_dir = "D:/DL_Project/data_navigation/test/rice-test"
output_dir = "D:/navigationData/deeplearning_line"

# 运行
process_labels_and_images_with_clustering(label_dir, image_dir, output_dir)
