from matplotlib import pyplot as plt  # 展示图片
import numpy as np  # 数值处理
import cv2  # opencv库
from sklearn.linear_model import LinearRegression, Ridge, Lasso  # 回归分析


def noise_mask_image(img, noise_ratio=[0.8,0.4,0.6]):
    """
    根据题目要求生成受损图片
    :param img: cv2 读取图片,而且通道数顺序为 RGB
    :param noise_ratio: 噪声比率，类型是 List,，内容:[r 上的噪声比率,g 上的噪声比率,b 上的噪声比率]
                        默认值分别是 [0.8,0.4,0.6]
    :return: noise_img 受损图片, 图像矩阵值 0-1 之间，数据类型为 np.array,
             数据类型对象 (dtype): np.double, 图像形状:(height,width,channel),通道(channel) 顺序为RGB
    """
    # 受损图片初始化
    noise_img = None
    # -------------实现受损图像答题区域-----------------
    height, width, channels = img.shape
    noise_masks = []
    for ratio in noise_ratio:
        mask = np.random.choice([0, 1], size=(height, width), p=[ratio, 1 - ratio])
        noise_masks.append(mask)
    noise_img = np.stack([img[:, :, i] * noise_masks[i] for i in range(channels)], axis=2)
    # -----------------------------------------------

    return noise_img


def get_noise_mask(noise_img):
    """
    获取噪声图像，一般为 np.array
    :param noise_img: 带有噪声的图片
    :return: 噪声图像矩阵
    """
    # 将图片数据矩阵只包含 0和1,如果不能等于 0 则就是 1。
    return np.array(noise_img != 0, dtype='double')


def restore_image(noise_img, size=8):
    """
    使用 你最擅长的算法模型 进行图像恢复。
    :param noise_img: 一个受损的图像
    :param size: 输入区域半径，长宽是以 size*size 方形区域获取区域, 默认是 4
    :return: res_img 恢复后的图片，图像矩阵值 0-1 之间，数据类型为 np.array,
            数据类型对象 (dtype): np.double, 图像形状:(height,width,channel), 通道(channel) 顺序为RGB
    """
    # 恢复图片初始化，首先 copy 受损图片，然后预测噪声点的坐标后作为返回值。
    res_img = np.copy(noise_img)

    # 获取噪声图像
    noise_mask = get_noise_mask(noise_img)

    # -------------实现图像恢复代码答题区域----------------------------
    noise_mask = get_noise_mask(noise_img)
    
    height, width, channels = noise_img.shape

    # 初始化线性回归模型
    #model = LinearRegression()
    model = Ridge(alpha=0.1)
    # 对每个通道进行处理
    for c in range(channels):
        # 遍历每个像素
        for i in range(height):
            for j in range(width):
                if noise_mask[i, j, c] == 0:  # 如果是噪声点（掩码为 0）
                    # 获取周围区域的边界
                    half_size = size // 2
                    top = max(i - half_size, 0)
                    bottom = min(i + half_size + 1, height)
                    left = max(j - half_size, 0)
                    right = min(j + half_size + 1, width)

                    # 提取周围区域的坐标和像素值
                    region = noise_img[top:bottom, left:right, c]
                    coords = []
                    values = []

                    for x in range(region.shape[0]):
                        for y in range(region.shape[1]):
                            if region[x,y] != 0:
                                coords.append([x+top, y+left])  # 计算全局坐标
                                values.append(region[x, y])

                    coords = np.array(coords)
                    values = np.array(values)

                    # 训练线性回归模型
                    model.fit(coords, values)

                    # 预测当前噪声点的像素值
                    predicted_value = model.predict([[i, j]])[0]

                    #predicted_value = np.clip(predicted_value, 0, 1)
                    if predicted_value > 1:
                        predicted_value = 1
                    if predicted_value < 0:
                        predicted_value = 0

                    # 恢复噪声点
                    res_img[i, j, c] = predicted_value
    
    
    # res_img = normalization(res_img)


    # ---------------------------------------------------------------

    return res_img
