import os
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple

def parse_xml_annotation(xml_path: str) -> Tuple[int, int, List[Dict]]:
    """解析XML标注文件"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    annotations = []
    for obj in root.iter('object'):
        class_name = obj.find('name').text.lower()  # 转为小写统一格式
        bbox = obj.find('bndbox')
        annotations.append({
            'class': class_name,
            'bbox': [
                float(bbox.find('xmin').text),
                float(bbox.find('ymin').text),
                float(bbox.find('xmax').text),
                float(bbox.find('ymax').text)
            ]
        })
    
    return width, height, annotations

def voc_to_yolo_bbox(bbox: List[float], img_width: int, img_height: int) -> List[float]:
    """VOC转YOLO格式"""
    xmin, ymin, xmax, ymax = bbox
    width = xmax - xmin
    height = ymax - ymin
    return [
        (xmin + width/2) / img_width,   # x_center
        (ymin + height/2) / img_height,  # y_center
        width / img_width,               # width
        height / img_height              # height
    ]

def convert_xml_to_yolo_txt(xml_path: str, output_dir: str, class_mapping: Dict[str, int]):
    """单个文件转换"""
    try:
        width, height, annotations = parse_xml_annotation(xml_path)
        output_lines = []
        
        for ann in annotations:
            class_name = ann['class']
            if class_name not in class_mapping:
                print(f"警告: 发现未映射类别 '{class_name}'，已自动跳过")
                continue
                
            yolo_bbox = voc_to_yolo_bbox(ann['bbox'], width, height)
            line = f"{class_mapping[class_name]} {' '.join([f'{x:.6f}' for x in yolo_bbox])}"
            output_lines.append(line)
        
        # 写入文件
        txt_filename = os.path.splitext(os.path.basename(xml_path))[0] + ".txt"
        with open(os.path.join(output_dir, txt_filename), 'w') as f:
            f.write('\n'.join(output_lines))
            
    except Exception as e:
        print(f"处理文件 {xml_path} 时出错: {str(e)}")

def batch_convert_xml_to_yolo(xml_dir: str, output_dir: str, class_mapping: Dict[str, int]):
    """批量转换"""
    os.makedirs(output_dir, exist_ok=True)
    
    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith('.xml'):
            convert_xml_to_yolo_txt(
                os.path.join(xml_dir, xml_file),
                output_dir,
                class_mapping
            )
    
    print(f"转换完成！结果保存在: {output_dir}")
    print(f"类别映射关系: {class_mapping}")

if __name__ == "__main__":
    # NEU-DET 钢材缺陷数据集类别映射
    NEU_CLASS_MAPPING = {
        "crazing": 0,      # 裂纹
        "inclusion": 1,    # 夹杂
        "patches": 2,      # 斑块
        "pitted_surface": 3,  # 点蚀
        "rolled-in_scale": 4,  # 轧入氧化皮
        "scratches": 5     # 划痕
    }
    
    # 配置路径
    XML_DIR = r"NEU-DET\ANNOTATIONS"
    OUTPUT_DIR = r"labels"
    
    # 执行转换
    batch_convert_xml_to_yolo(
        xml_dir=XML_DIR,
        output_dir=OUTPUT_DIR,
        class_mapping=NEU_CLASS_MAPPING
    )