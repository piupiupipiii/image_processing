import math
import sys
import cv2
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow
from PyQt5.uic import loadUi
from matplotlib import pyplot as plt

class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('GUI.ui', self)
        self.Image = None
        self.button_loadCitra.clicked.connect(self.fungsi)
        self.button_prosesCitra.clicked.connect(self.grayscale)
        self.actionOperasi_Pencerahan.triggered.connect(self.brightness)
        self.actionSimple_Contrast.triggered.connect(self.contrast)
        self.actionContrast_Streching.triggered.connect(self.contrastStreching)
        self.actionNegatif.triggered.connect(self.negatif)
        self.actionBiner.triggered.connect(self.biner)


        self.actionHistogram_Grayscale.triggered.connect(self.grayHistogram)

    def fungsi(self):
        self.Image = cv2.imread('meng.jpg')
        self.displayImage(self.Image, self.label)

    def grayscale(self):
        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] +
                                     0.587 * self.Image[i, j, 1] +
                                     0.114 * self.Image[i, j, 2], 0, 255)
        self.Image = gray
        self.displayImage(gray, self.label_2)

    def brightness(self):
        # agar menghindari error ketika melewati proses Grayscaling Citra
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)

        except:
            pass

        H, W = self.Image.shape[:2]
        brightness = 80
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = np.clip(a + brightness, 0, 255)

                self.Image.itemset((i, j), b)
        self.displayImage(self.Image, self.label)

    def contrast(self):
        # agar menghindari error ketika melewati proses Grayscaling Citra
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)

        except:
            pass

        H, W = self.Image.shape[:2]
        contrast = 1.7
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = np.clip(a * contrast, 0, 255)

                self.Image.itemset((i, j), b)
        self.displayImage(self.Image, self.label)

    def contrastStreching(self):
        # agar menghindari error ketika melewati proses Grayscaling Citra
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)

        except:
            pass

        H, W = self.Image.shape[:2]
        minV = np.min(self.Image)
        maxV = np.max(self.Image)
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = float(a - minV) /(maxV - minV) * 255

                self.Image.itemset((i, j), b)
        self.displayImage(self.Image, self.label)

    def negatif(self):
        # agar menghindari error ketika melewati proses Grayscaling Citra
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)

        except:
            pass

        H, W = self.Image.shape[:2]
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                b = math.ceil(255 - a)

                self.Image.itemset((i, j), b)
        self.displayImage(self.Image, self.label_2)

    def grayHistogram(self):
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(
                    0.299 * self.Image[i, j, 0] +
                    0.587 * self.Image[i, j, 1] +
                    0.114 * self.Image[i, j, 2] , 0, 255)

        self.Image = gray
        self.displayImage(self.Image, self.label_2)
        hist, bins = np.histogram(self.Image.ravel(), 256, [0, 256])
        plt.plot(hist)
        plt.title('Histogram Grayscale')
        plt.show()


    def biner(self):
        # agar menghindari error ketika melewati proses Grayscaling Citra
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)

        except:
            pass

        H, W = self.Image.shape[:2]
        for i in range(H):
            for j in range(W):
                a = self.Image.item(i, j)
                if a == 180:
                    b = 0
                elif a < 180:
                    b = 1
                else:
                    b = 255

                self.Image.itemset((i, j), b)
        self.displayImage(self.Image, self.label_2)

    def displayImage(self, image, label):
        qformat = QImage.Format_Indexed8

        if len(image.shape) == 3:
            if image.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(image.data, image.shape[1], image.shape[0],
                     image.strides[0], qformat)

        img = img.rgbSwapped()

        label.setPixmap(QPixmap.fromImage(img))


app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle('Pertemuan 1')
window.show()
sys.exit(app.exec_())
