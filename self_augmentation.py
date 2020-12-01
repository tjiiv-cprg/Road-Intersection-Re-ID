from __future__ import absolute_import

from torchvision.transforms import *

from PIL import Image
import cv2
import random
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear as demosaic
import numpy as np
import math


class GBRG2RGB(object):
    """convert the GBRG image to RGB PIL image.
        Args:
            GBRG image.
        Returns:
            RGB image
    """
    def __init__(self, p=1):
        self.probability = p

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img
        img = demosaic(img, 'gbrg')
        return Image.fromarray(img.astype('uint8')).convert('RGB')


class Add_haze(object):
    """ Randomly add haze in image.
    Args:
         p: The probability that the add_haze operation will be performed.
         alpha: light of the haze
         beta: density of the haze
    """

    def __init__(self, p=0.3):
        self.probability = p

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        alpha = 0.7 + 0.3 * np.random.rand(1)[0]
        beta = 0.03 + 0.04 * np.random.rand(1)[0]
        # alpha = 0.7 + 0.3 * 0.5
        # beta = 0.03 + 0.04 * 1
        img = np.array(img)
        size_h, size_w, size_c = img.shape
        A_light = np.array([255, 255, 255]) * alpha
        size = math.sqrt(max(size_h, size_w))
        center = (size_h // 2, size_w // 2)
        t_map = np.ones((size_h, size_w))
        for j in range(size_h):
            for l in range(size_w):
                d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
                t_map[j][l] = math.exp(-beta * d)
        t_map = t_map[:, :, np.newaxis]
        t_map = t_map.repeat(3, axis=2)

        A_light = A_light[np.newaxis, :]
        A_light = A_light.repeat(size_w, axis=0)
        A_light = A_light[np.newaxis, :, :]
        A_light = A_light.repeat(size_h, axis=0)

        haze_img = t_map * img + (1 - t_map) * A_light

        return Image.fromarray(haze_img.astype('uint8')).convert('RGB')


class Add_rain(object):
    """ Randomly add rain in image.
    Args:
         >>> input: img
             p:      The probability that the add_rain operation will be performed.
             value:  The number of raindrops
             length: The length of raindrops
             angle： Angle of the rain，anti_clock is positive
             w:      The weight of raindrops
             beta：  The light of rain
         >>> output: raining_img
        
    """

    def __init__(self, p=0.3):
        self.probability = p

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        value = 200
        w = 1
        length = 10
        angle = -30 + 60 * np.random.rand(1)[0]
        beta = 0.7 + 0.3 * np.random.rand(1)[0]

        all_rain = ["light", "middle", "heavy"]
        which_rain = random.sample(all_rain, 1)

        if which_rain[0] == "light":
            value = 200
            w = 1
            length = 10
        elif which_rain[0] == "middle":
            value = 400
            w = 3
            length = 20
        elif which_rain[0] == "heavy":
            value = 600
            w = 5
            length = 30

        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        noise = np.random.uniform(0, 256, img.shape[0:2])
        # 控制噪声水平，取浮点数，只保留最大的一部分作为噪声
        v = value * 0.01
        noise[np.where(noise < (256 - v))] = 0

        # 噪声做初次模糊
        k = np.array([[0, 0.1, 0],
                      [0.1, 8, 0.1],
                      [0, 0.1, 0]])

        noise = cv2.filter2D(noise, -1, k)

        trans = cv2.getRotationMatrix2D((length / 2, length / 2), angle - 45, 1 - length / 100.0)
        dig = np.diag(np.ones(length))  # 生成对焦矩阵
        k = cv2.warpAffine(dig, trans, (length, length))  # 生成模糊核
        k = cv2.GaussianBlur(k, (w, w), 0)  # 高斯模糊这个旋转后的对角核，使得雨有宽度
        # k = k / length                         #是否归一化
        blurred = cv2.filter2D(noise, -1, k)  # 用刚刚得到的旋转后的核，进行滤波
        # 转换到0-255区间
        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        blurred = np.array(blurred, dtype=np.uint8)

        rain = np.expand_dims(blurred, 2)
        rain_result = img.copy()  # 拷贝一个掩膜
        rain = np.array(rain, dtype=np.float32)  # 数据类型变为浮点数，后面要叠加，防止数组越界要用32位
        rain_result[:, :, 0] = rain_result[:, :, 0] * (255 - rain[:, :, 0]) / 255.0 + beta * rain[:, :, 0]
        rain_result[:, :, 1] = rain_result[:, :, 1] * (255 - rain[:, :, 0]) / 255 + beta * rain[:, :, 0]
        rain_result[:, :, 2] = rain_result[:, :, 2] * (255 - rain[:, :, 0]) / 255 + beta * rain[:, :, 0]

        return Image.fromarray(cv2.cvtColor(rain_result, cv2.COLOR_BGR2RGB))


class AddPepperNoise(object):
    """增加椒盐噪声
    Args:
        snr （float）: Signal Noise Rate
        p (float): 概率值，依概率执行该操作
    """

    def __init__(self, p=0.9):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.uniform(0, 1) < self.p:
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            signal_pct = 0.7 + 0.3 * int(np.random.rand(1)[0] * 100)/100  # self.snr
            noise_pct = (1 - signal_pct)
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255
            img_[mask == 2] = 0
            return Image.fromarray(img_.astype('uint8')).convert('RGB')
        else:
            return img


class AddGaussianNoise(object):

    def __init__(self, p=0.9):
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            amplitude = 10 + 20 * np.random.rand(1)[0]
            N = amplitude * np.random.normal(loc=0, scale=1, size=(h, w, 1))
            N = np.repeat(N, c, axis=2)
            img_ = N + img_
            img_[img_ > 255] = 255
            img_ = Image.fromarray(img_.astype('uint8')).convert('RGB')
            return img_
        else:
            return img
