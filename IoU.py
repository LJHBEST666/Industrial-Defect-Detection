def calculate_iou(box1, box2):
    """
    计算两个边界框的IoU
    参数:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    返回:
        iou值
    """
    # 计算交集区域的坐标
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # 计算交集区域面积
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # 计算两个框各自的面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 计算并集区域面积
    union = area1 + area2 - intersection
    
    # 计算IoU
    iou = intersection / union if union != 0 else 0.0
    
    return iou
x = calculate_iou([1, 1, 3, 3], [2, 2, 4, 4])
print(x)