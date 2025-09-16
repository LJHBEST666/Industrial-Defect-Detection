import os
import random
import shutil
from tqdm import tqdm

def split_dataset(images_dir, labels_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    划分数据集为训练集、验证集和测试集
    
    参数:
        images_dir: 原始图片目录
        labels_dir: 原始标签目录（YOLO格式）
        output_dir: 输出根目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子（确保可复现）
    """
    # 检查比例总和是否为1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例总和必须等于1"
    
    # 创建输出目录结构
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
    
    # 获取所有图片文件（确保与标签匹配）
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    label_files = [f.replace(os.path.splitext(f)[1], '.txt') for f in image_files]
    
    # 检查图片和标签是否一一对应
    for img, lbl in zip(image_files, label_files):
        if not os.path.exists(os.path.join(labels_dir, lbl)):
            raise FileNotFoundError(f"标签文件缺失: {lbl}")
    
    # 随机打乱数据集（固定随机种子保证可复现）
    random.seed(seed)
    paired_files = list(zip(image_files, label_files))
    random.shuffle(paired_files)
    
    # 计算划分点
    total = len(paired_files)
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)
    
    # 划分数据集
    splits_files = {
        'train': paired_files[:train_end],
        'val': paired_files[train_end:val_end],
        'test': paired_files[val_end:]
    }
    
    # 复制文件到对应目录
    for split, files in splits_files.items():
        print(f"正在处理 {split} 集 ({len(files)} 个样本)...")
        for img_file, label_file in tqdm(files, desc=split):
            # 复制图片
            src_img = os.path.join(images_dir, img_file)
            dst_img = os.path.join(output_dir, split, 'images', img_file)
            shutil.copy2(src_img, dst_img)
            
            # 复制标签
            src_label = os.path.join(labels_dir, label_file)
            dst_label = os.path.join(output_dir, split, 'labels', label_file)
            shutil.copy2(src_label, dst_label)
    
    print("\n数据集划分完成！")
    print(f"总样本数: {total}")
    print(f"训练集: {len(splits_files['train'])} ({len(splits_files['train'])/total:.1%})")
    print(f"验证集: {len(splits_files['val'])} ({len(splits_files['val'])/total:.1%})")
    print(f"测试集: {len(splits_files['test'])} ({len(splits_files['test'])/total:.1%})")
    
    # 生成data.yaml文件（YOLOv5/v7/v8兼容格式）
    class_names = get_class_names(labels_dir)  # 从标签文件自动获取类别
    yaml_content = f"""
# 数据集配置文件
path: {os.path.abspath(output_dir)}
train: train/images
val: val/images
test: test/images

# 类别数
nc: {len(class_names)}

# 类别名称列表
names: {class_names}
"""
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        f.write(yaml_content)
    print("\n生成的YAML配置文件已保存到: data.yaml")

def get_class_names(labels_dir):
    """从标签文件自动提取所有类别ID"""
    class_ids = set()
    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            with open(os.path.join(labels_dir, label_file), 'r') as f:
                for line in f.readlines():
                    if line.strip():
                        class_id = int(line.strip().split()[0])
                        class_ids.add(class_id)
    return [str(i) for i in sorted(class_ids)]  # 按ID排序

if __name__ == "__main__":
    # 配置路径
    IMAGES_DIR = r"NEU-DET/IMAGES"  # 原始图片目录
    LABELS_DIR = r"NEU-DET/labels"  # YOLO标签目录
    OUTPUT_DIR = r"dataset_split"   # 输出目录
    
    # 执行划分 (比例可调整)
    split_dataset(
        images_dir=IMAGES_DIR,
        labels_dir=LABELS_DIR,
        output_dir=OUTPUT_DIR,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )