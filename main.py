import cv2


def main():
    img = cv2.imread('big.jpeg')
    cv2.imshow('gambar1', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
