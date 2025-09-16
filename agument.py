import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm
import shutil
import yaml
import matplotlib.pyplot as plt

class YOLOAugmentor:
    def __init__(self, dataset_root, output_root, img_size=320):
        """
        :param dataset_root: 原始数据集根目录（包含train/val/test子目录）
        :param output_root: 增强数据输出根目录
        :param img_size: 图像目标尺寸
        """
        self.dataset_root = dataset_root
        self.output_root = output_root
        self.img_size = img_size
        
        # 构建增强管道
        self.transform = self._build_augmentation_pipeline()
        
        # 创建输出目录结构
        self._create_dirs()

    def _build_augmentation_pipeline(self):
        """构建增强管道"""
        return A.Compose([
        # 空间变换
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),

        # 随机裁剪和调整大小
        A.RandomResizedCrop(
            size=(self.img_size, self.img_size),  # 关键修复
            scale=(0.6, 1.0),
            ratio=(0.9, 1.1),
            interpolation=cv2.INTER_LINEAR,
            p=0.5
        ),

        # 像素级变换
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10)
        ], p=0.5),

        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),

        # 确保图像最终尺寸一致
        A.Resize(height=self.img_size, width=self.img_size)
        ], bbox_params=A.BboxParams(
        format='yolo',
        min_visibility=0.3,
        label_fields=['class_labels']
    ))
    def _create_dirs(self):
        """创建输出目录结构"""
        os.makedirs(os.path.join(self.output_root, 'train', 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.output_root, 'train', 'labels'), exist_ok=True)
        os.makedirs(os.path.join(self.output_root, 'val'), exist_ok=True)
        os.makedirs(os.path.join(self.output_root, 'test'), exist_ok=True)

    def _load_yolo_annotations(self, label_path):
        """加载YOLO格式标签"""
        with open(label_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        bboxes = []
        class_ids = []
        for line in lines:
            parts = line.split()
            if len(parts) == 5:  # class_id, x_center, y_center, width, height
                class_ids.append(int(parts[0]))
                bboxes.append([float(x) for x in parts[1:5]])
        
        return np.array(bboxes), np.array(class_ids)

    def _save_yolo_annotations(self, label_path, bboxes, class_ids):
        """保存YOLO格式标签"""
        with open(label_path, 'w') as f:
            for cls_id, bbox in zip(class_ids, bboxes):
                line = f"{cls_id} " + " ".join([f"{x:.6f}" for x in bbox]) + "\n"
                f.write(line)

    def _visualize_augmentation(self, image, bboxes, augmented):
        """可视化增强效果"""
        plt.figure(figsize=(15, 5))
        
        # 原始图像
        plt.subplot(1, 2, 1)
        img_orig = image.copy()
        h, w = img_orig.shape[:2]
        for bbox in bboxes:
            x, y, bw, bh = bbox
            x1 = int((x - bw/2) * w)
            y1 = int((y - bh/2) * h)
            x2 = int((x + bw/2) * w)
            y2 = int((y + bh/2) * h)
            cv2.rectangle(img_orig, (x1, y1), (x2, y2), (255, 0, 0), 2)
        plt.imshow(img_orig)
        plt.title("Original")
        plt.axis('off')
        
        # 增强图像
        plt.subplot(1, 2, 2)
        img_aug = augmented['image']
        for bbox in augmented['bboxes']:
            x, y, bw, bh = bbox
            x1 = int((x - bw/2) * self.img_size)
            y1 = int((y - bh/2) * self.img_size)
            x2 = int((x + bw/2) * self.img_size)
            y2 = int((y + bh/2) * self.img_size)
            cv2.rectangle(img_aug, (x1, y1), (x2, y2), (0, 255, 0), 2)
        plt.imshow(img_aug)
        plt.title("Augmented")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    def augment_dataset(self, augment_per_image=2, debug=False):
        """执行数据增强
        :param augment_per_image: 每张图像的增强次数
        :param debug: 是否开启调试模式（可视化）
        """
        # 1. 复制验证集和测试集（保持不变）
        print("复制验证集和测试集...")
        for subset in ['val', 'test']:
            src_dir = os.path.join(self.dataset_root, subset)
            dst_dir = os.path.join(self.output_root, subset)
            if os.path.exists(src_dir):
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
        
        # 2. 处理训练集
        print("增强训练集...")
        train_image_dir = os.path.join(self.dataset_root, 'train', 'images')
        train_label_dir = os.path.join(self.dataset_root, 'train', 'labels')
        
        image_files = [f for f in os.listdir(train_image_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in tqdm(image_files, desc="Processing"):
            img_path = os.path.join(train_image_dir, img_file)
            label_path = os.path.join(train_label_dir, 
                                    os.path.splitext(img_file)[0] + '.txt')
            
            # 加载原始数据
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bboxes, class_ids = self._load_yolo_annotations(label_path)
            
            # 复制原始数据到输出目录
            base_name = os.path.splitext(img_file)[0]
            shutil.copy(img_path, os.path.join(self.output_root, 'train', 'images', img_file))
            shutil.copy(label_path, os.path.join(self.output_root, 'train', 'labels', f"{base_name}.txt"))
            
            # 生成增强数据
            for i in range(augment_per_image):
                try:
                    augmented = self.transform(
                        image=image.copy(),
                        bboxes=bboxes.copy(),
                        class_labels=class_ids.copy()
                    )
                    
                    # 调试可视化
                    if debug and i == 0:
                        self._visualize_augmentation(image, bboxes, augmented)
                    
                    # 跳过无效增强
                    if len(augmented['bboxes']) == 0:
                        continue
                    
                    # 保存增强结果
                    aug_img = augmented['image']
                    aug_bboxes = np.array(augmented['bboxes'])
                    aug_class_ids = np.array(augmented['class_labels'])
                    
                    # 保存图像
                    aug_img_path = os.path.join(
                        self.output_root, 'train', 'images',
                        f"{base_name}_aug{i}.jpg"
                    )
                    cv2.imwrite(aug_img_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                    
                    # 保存标签
                    aug_label_path = os.path.join(
                        self.output_root, 'train', 'labels',
                        f"{base_name}_aug{i}.txt"
                    )
                    self._save_yolo_annotations(aug_label_path, aug_bboxes, aug_class_ids)
                
                except Exception as e:
                    print(f"Error augmenting {img_file}: {str(e)}")
        
        # 3. 更新数据集配置文件
        self._update_yaml_config()
        print("\n数据增强完成！")

    def _update_yaml_config(self):
        """更新YOLO数据集配置文件"""
        src_yaml = os.path.join(self.dataset_root, 'data.yaml')
        dst_yaml = os.path.join(self.output_root, 'data.yaml')
        
        if os.path.exists(src_yaml):
            with open(src_yaml, 'r') as f:
                data = yaml.safe_load(f)
            
            # 更新路径
            data['train'] = 'train/images'
            data['val'] = 'val/images'
            if 'test' in data:
                data['test'] = 'test/images'
            
            with open(dst_yaml, 'w') as f:
                yaml.dump(data, f, sort_keys=False)

if __name__ == "__main__":
    # 配置路径
    DATASET_ROOT = "dataset_split"  # 原始数据集目录
    OUTPUT_ROOT = "augmented_dataset_a"  # 输出目录
    
    # 初始化增强器
    augmentor = YOLOAugmentor(
        dataset_root=DATASET_ROOT,
        output_root=OUTPUT_ROOT,
        img_size=320
    )
    
    # 执行增强（开启调试模式可视化第一个样本）
    augmentor.augment_dataset(augment_per_image=2, debug=False)