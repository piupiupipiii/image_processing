import cv2
import numpy as np
from matplotlib import pyplot as plt


def main():
    img = cv2.imread('big.jpeg')
    cv2.imshow('gambar1', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
    test = np.array(
        [
            [1, 2, 2],
            [3, 3, 3],
            [1, 1, 4],
        ]
    )

    img1 = cv2.imread('C:\\Users\\Silvy Nur Azkia\\Downloads\\buku_blur.jpg')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    hasil = Filtering.Konvolusi(img1, test)
    plt.imshow(hasil, cmap='gray', interpolation='bicubic')
    plt.xticks([], plt.ystick([]))
    plt.show()
    cv2.waitKey()
    cv2.destroyAllWindows()
