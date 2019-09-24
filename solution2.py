import numpy as np
import matplotlib.pyplot as plt

def gauss_kernel(window_size = 5, sigma = 3):
    mid = (int)(window_size / 2)
    kernel = np.zeros((window_size, window_size))
    for i in range(window_size):
        for j in range(window_size):
            diff = np.sqrt((i - mid) ** 2 + (j - mid) ** 2)
            kernel[i, j] = np.exp(-(diff ** 2) / (2 * sigma ** 2))
    return kernel / np.sum(kernel)

def gauss_filter(img, window_size = 5, sigma = 3):
    img2 = np.zeros_like(img)
    kernel = gauss_kernel(window_size, sigma)
    p = window_size//2
    for k in range(img.shape[2]):
        for i in range(p, img.shape[0] - p):
            for j in range(p, img.shape[1] - p):
                window = img[i - p: i + p + 1, j - p: j + p + 1, k]
                img2[i, j, k] = (kernel * window).sum()
    return img2


def main():
    img = plt.imread("img.png")[:, :, :3]
    img2 = gauss_filter(img)

    fig, axs = plt.subplots(1,2)
    axs[0].imshow(img)
    axs[1].imshow(img2)
    plt.show()


if __name__ == "__main__":
    main()