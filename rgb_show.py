import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取彩色图像
image = cv2.imread('1.png')
if image is None:
    print("无法加载图片，请检查路径是否正确")
    exit()

# 将BGR转换为RGB以便正确显示
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 分离各个颜色通道 - 注意现在顺序是R,G,B
r, g, b = cv2.split(image_rgb)  # 修改了这里的顺序

# 创建全零矩阵用于合并单通道
zeros = np.zeros_like(r)

# 重建各个通道的彩色表示 - 保持RGB顺序
red_channel = cv2.merge([r, zeros, zeros])    # 红色在R通道
green_channel = cv2.merge([zeros, g, zeros])  # 绿色在G通道
blue_channel = cv2.merge([zeros, zeros, b])   # 蓝色在B通道

# 使用Matplotlib显示结果
plt.figure(figsize=(15, 10))

# 显示原始图像
plt.subplot(2, 2, 1)
plt.imshow(image_rgb)  # 这里改为显示image_rgb而不是原始的BGR图像
plt.title('Original RGB Image')
plt.axis('off')

# 显示红色通道
plt.subplot(2, 2, 2)
plt.imshow(red_channel)
plt.title('Red Channel')
plt.axis('off')

# 显示绿色通道
plt.subplot(2, 2, 3)
plt.imshow(green_channel)
plt.title('Green Channel')
plt.axis('off')

# 显示蓝色通道
plt.subplot(2, 2, 4)
plt.imshow(blue_channel)
plt.title('Blue Channel')
plt.axis('off')

plt.tight_layout()
plt.show()