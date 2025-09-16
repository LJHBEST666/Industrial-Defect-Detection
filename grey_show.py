import cv2
import matplotlib.pyplot as plt

# 读取图片
image_path = '1.png'  # 替换为你的图片路径
image = cv2.imread(image_path)

# 检查图片是否成功加载
if image is None:
    print("无法加载图片，请检查路径是否正确")
    exit()

# 获取并打印图片尺寸和通道数
height, width, channels = image.shape
print(f"图片尺寸: 高={height}像素, 宽={width}像素")
print(f"通道数: {channels} (BGR彩色图)")

# 将BGR彩色图转换为灰度图
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用Matplotlib显示图片
plt.figure(figsize=(10, 5))

# 显示原图(需要将BGR转换为RGB)
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Color Image (RGB)')
plt.axis('off')

# 显示灰度图
plt.subplot(1, 2, 2)
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

plt.tight_layout()
# plt.show()

# 计算灰度直方图
hist = cv2.calcHist([gray_image], [0], None, [256], [0,256])

# 绘制直方图
plt.figure()
plt.title("灰度直方图")
plt.xlabel("像素强度")
plt.ylabel("像素数量")
plt.plot(hist)
plt.xlim([0,256])
plt.show()

