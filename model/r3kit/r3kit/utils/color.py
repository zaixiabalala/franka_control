import random
import numpy as np


def rgb2hsv(img:np.ndarray) -> np.ndarray:
    # img: [N, 3]
    r, g, b = img[:, 0], img[:, 1], img[:, 2]
    maxc = np.max(img, axis=-1)
    minc = np.min(img, axis=-1)
    v = maxc
    deltac = maxc - minc
    s = deltac / (maxc + 1e-7)
    s = np.where(maxc == 0, 0, s)
    rc = (maxc - r) / (deltac + 1e-7)
    gc = (maxc - g) / (deltac + 1e-7)
    bc = (maxc - b) / (deltac + 1e-7)
    h = np.where(maxc == minc, 0, np.where(
        maxc == r, 60 * (bc - gc), np.where(
            maxc == g, 120 + 60 * (rc - bc), 240 + 60 * (gc - rc))))
    h = np.where(h < 0, h + 360, h)
    return np.stack([h, s, v], axis=-1)

def hsv2rgb(img:np.ndarray) -> np.ndarray:
    # img: [N, 3]
    h, s, v = img[:, 0], img[:, 1], img[:, 2]
    h = h / 60
    i = np.floor(h)
    f = h - i
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    i = i.astype(np.int32)
    i = i % 6
    rgb = np.empty(img.shape, dtype=img.dtype)
    rgb[:, 0] = np.where(i == 0, v, np.where(i == 1, q, np.where(i == 2, p, np.where(i == 3, p, np.where(i == 4, t, v)))))
    rgb[:, 1] = np.where(i == 0, t, np.where(i == 1, v, np.where(i == 2, v, np.where(i == 3, q, np.where(i == 4, p, p)))))
    rgb[:, 2] = np.where(i == 0, p, np.where(i == 1, p, np.where(i == 2, t, np.where(i == 3, v, np.where(i == 4, v, q)))))
    return rgb


def color_jitter(img:np.ndarray, brightness:float=0., contrast:float=0., saturation:float=0., hue:float=0., fixed:bool=False) -> np.ndarray:
    # img: [N, 3]
    if brightness != 0:
        if not fixed:
            factor = random.uniform(1 - brightness, 1 + brightness)
        else:
            factor = 1 + brightness
        img = img * factor

    if contrast != 0:
        mean = np.mean(img, axis=0)
        if not fixed:
            factor = random.uniform(1 - contrast, 1 + contrast)
        else:
            factor = 1 + contrast
        img = (img - mean[None, :]) * factor + mean[None, :]

    if saturation != 0:
        gray = np.dot(img, [0.2989, 0.5870, 0.1140])
        if not fixed:
            factor = random.uniform(1 - saturation, 1 + saturation)
        else:
            factor = 1 + saturation
        img = (img - gray[:, None]) * factor + gray[:, None]

    if hue != 0:
        hsv = rgb2hsv(img)
        if not fixed:
            factor = random.uniform(-hue, hue)
        else:
            factor = hue
        hue_shift = factor * 360
        hsv[:, 0] = (hsv[:, 0] + hue_shift) % 360
        img = hsv2rgb(hsv)

    img = np.clip(img, 0, 1)
    return img


if __name__ == '__main__':
    import cv2
    image = cv2.imread('temp.png')
    cv2.imshow('original', image)
    cv2.waitKey(0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.
    image = image.astype(np.float32)            # (H, W, 3)
    hsv = hsv2rgb(rgb2hsv(image.reshape(-1, 3))).reshape(image.shape)
    hsv = (hsv * 255).astype(np.uint8)
    hsv = cv2.cvtColor(hsv, cv2.COLOR_RGB2BGR)
    cv2.imshow('hsv', hsv)
    cv2.waitKey(0)
    for _ in range(10):
        jitter = color_jitter(image.reshape(-1, 3), brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1).reshape(image.shape)
        jitter = (jitter * 255).astype(np.uint8)
        jitter = cv2.cvtColor(jitter, cv2.COLOR_RGB2BGR)
        cv2.imshow('jitter', jitter)
        cv2.waitKey(0)
