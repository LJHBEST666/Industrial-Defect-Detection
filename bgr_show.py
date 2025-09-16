import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取彩色图像（OpenCV 默认 BGR 顺序）
image = cv2.imread('2.png')
if image is None:
    print("无法加载图片，请检查路径是否正确")
    exit()

# 分离 BGR 通道（不转换为 RGB）
b, g, r = cv2.split(image)  # OpenCV 默认顺序：B, G, R

# 创建全零矩阵用于合并单通道
zeros = np.zeros_like(b)

# 重建各个通道的彩色表示（BGR 顺序）
blue_channel = cv2.merge([b, zeros, zeros])    # 蓝色在 B 通道
green_channel = cv2.merge([zeros, g, zeros])   # 绿色在 G 通道
red_channel = cv2.merge([zeros, zeros, r])     # 红色在 R 通道

# 使用 Matplotlib 显示（需转换为 RGB 格式）
plt.figure(figsize=(15, 10))

# 显示原始图像（BGR 转 RGB）
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # 仅显示时转换
plt.title('Original Image (BGR in OpenCV)')
plt.axis('off')

# 显示蓝色通道（B 通道）
plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(blue_channel, cv2.COLOR_BGR2RGB))  # 转换显示
plt.title('Blue Channel (B in BGR)')
plt.axis('off')

# 显示绿色通道（G 通道）
plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(green_channel, cv2.COLOR_BGR2RGB))  # 转换显示
plt.title('Green Channel (G in BGR)')
plt.axis('off')

# 显示红色通道（R 通道）
plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(red_channel, cv2.COLOR_BGR2RGB))  # 转换显示
plt.title('Red Channel (R in BGR)')
plt.axis('off')

plt.tight_layout()
plt.show()