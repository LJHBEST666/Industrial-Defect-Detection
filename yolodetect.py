import os
import random
import cv2
import numpy as np

def draw_yolo_bbox(image, yolo_bbox, class_id, class_names, color=None):
    """
    在图像上绘制YOLO格式的边界框
    
    参数:
        image: 输入图像
        yolo_bbox: [x_center, y_center, width, height] 归一化坐标
        class_id: 类别ID
        class_names: 类别名称列表
        color: 可选，指定边框颜色
    """
    height, width = image.shape[:2]
    
    # 将归一化坐标转换为绝对坐标
    x_center, y_center, w, h = yolo_bbox
    x_center *= width
    y_center *= height
    w *= width
    h *= height
    
    # 计算左上和右下坐标
    xmin = int(x_center - w/2)
    ymin = int(y_center - h/2)
    xmax = int(x_center + w/2)
    ymax = int(y_center + h/2)
    
    # 确保坐标在图像范围内
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(width-1, xmax)
    ymax = min(height-1, ymax)
    
    # 随机颜色或使用指定颜色
    if color is None:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    # 绘制矩形和标签
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
    
    # 准备标签文本
    label = f"{class_names[class_id]}({class_id})"
    
    # 计算文本背景大小
    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    
    # 绘制文本背景
    cv2.rectangle(image, (xmin, ymin - text_height - 4), (xmin + text_width, ymin), color, -1)
    
    # 绘制文本
    cv2.putText(image, label, (xmin, ymin - 4), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    return image

def visualize_yolo_labels(images_dir, labels_dir, class_names, num_samples=5):
    """
    可视化YOLO标签
    
    参数:
        images_dir: 图片目录
        labels_dir: 标签目录
        class_names: 类别名称列表
        num_samples: 要显示的样本数量
    """
    # 获取所有图片文件
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # 随机选择样本
    selected_files = random.sample(image_files, min(num_samples, len(image_files)))
    
    for img_file in selected_files:
        # 构建对应标签文件路径
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)
        
        # 读取图像
        img_path = os.path.join(images_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
            print(f"无法读取图像: {img_path}")
            continue
        
        # 读取标签
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            # 绘制每个边界框
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    yolo_bbox = list(map(float, parts[1:5]))
                    image = draw_yolo_bbox(image, yolo_bbox, class_id, class_names)
        
        # 显示结果
        cv2.imshow(f"Visualization: {img_file}", image)
        
        # 调整窗口大小
        height, width = image.shape[:2]
        max_height = 800
        if height > max_height:
            scale = max_height / height
            cv2.resizeWindow(f"Visualization: {img_file}", int(width * scale), max_height)
        
        # 等待按键
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 按ESC退出
        if key == 27:
            break

if __name__ == "__main__":
    # 配置参数
    IMAGES_DIR = r"NEU-DET\IMAGES"  # 图片目录
    LABELS_DIR = r"labels"          # YOLO标签目录
    
    # NEU-DET数据集类别名称 (与之前的CLASS_MAPPING顺序一致)
    CLASS_NAMES = [
        "crazing",      # 0
        "inclusion",    # 1
        "patches",      # 2
        "pitted_surface",  # 3
        "rolled-in_scale",  # 4
        "scratches"     # 5
    ]
    
    # 执行可视化检查
    visualize_yolo_labels(
        images_dir=IMAGES_DIR,
        labels_dir=LABELS_DIR,
        class_names=CLASS_NAMES,
        num_samples=5
    )