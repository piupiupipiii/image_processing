import math
import sys
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QDialog
from PyQt5.uic import loadUi
from PyQt5.QtCore import pyqtSlot
from matplotlib import pyplot as plt

class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('GUI.ui', self)
        self.Image = None
        self.button_loadCitra.clicked.connect(self.fungsi)
        #operasi titik
        self.button_prosesCitra.clicked.connect(self.grayscale)
        self.actionOperasi_Pencerahan.triggered.connect(self.brightness)
        self.actionSimple_Contrast.triggered.connect(self.contrast)
        self.actionContrast_Streching.triggered.connect(self.contrastStreching)
        self.actionNegatif.triggered.connect(self.negatif)
        self.actionBiner.triggered.connect(self.biner)

        #operasi histogram
        self.actionHistogram_Grayscale.triggered.connect(self.grayHistogram)
        self.actionHistogram_RGB.triggered.connect(self.RGBHistogram)
        self.actionHistogram_Equalization.triggered.connect(self.EqualHistogram)

        #operasi geometri
        self.actionTranslasi.triggered.connect(self.translasi)
        self.action90_Derajat.triggered.connect(self.rotasi90derajat)
        self.action2x.triggered.connect(self.zoomIn2x)
        self.actionZoom_Out.triggered.connect(self.zoomout)
        self.actionCrop.triggered.connect(self.crop)

        #operasi aritmatika
        self.actionTambah_dan_Kurang.triggered.connect(self.aritmatika_tambah)
        self.actionKali_dan_Bagi.triggered.connect(self.aritmatika_kalibagi())


        #operasi boolean
        self.actionOperasi_AND.triggered.connect(self.boolean_AND)
        self.actionOperasi_OR.triggered.connect(self.boolean_OR)
        self.actionOperasi_XOR.triggered.connect(self.boolean_XOR)

#function operasi titik
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


#function operasi histogram
    def grayHistogram(self):
        try:
            self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.Image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        # for i in range(H):
        #     for j in range(W):
        #         gray[i, j] = np.clip(
        #             0.299 * self.Image[i, j, 0] +
        #             0.587 * self.Image[i, j, 1] +
        #             0.114 * self.Image[i, j, 2] , 0, 255)

        # self.Image = gray
        self.displayImage(gray, self.label_2)
        # self.Image = gray
        self.displayImage(self.Image, self.label_2)
        hist, bins = np.histogram(self.Image.ravel(), 255, (0,255))
        plt.plot(hist)
        plt.title('Histogram Grayscale')
        # plt.hist(self.Image.ravel(), 255, [0, 255])
        plt.show()

    @pyqtSlot()
    def RGBHistogram(self):
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histo = cv2.calcHist([self.Image], [i], None, [255], [0, 255])
            plt.plot(histo, color=col)
        plt.xlim([0, 255])
        plt.show()


    @pyqtSlot()
    def EqualHistogram(self):
        hist, bins = np.histogram(self.Image.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        self.Image = cdf[self.Image]
        self.displayImage(self.Image, self.label_2)

        plt.plot(cdf_normalized, color='b')
        plt.hist(self.Image.flatten(), 256, [0, 256], color='r')
        plt.xlim([0, 256])
        plt.legend(('cdf', 'histogram'), loc='upper left')
        plt.show()

#operasi geometri
    def translasi(self):
        h, w = self.Image.shape[:2]
        quarter_h, quarter_w = h / 4, w / 4
        T = np.float_([[1, 0, quarter_w], [0, 1, quarter_h]])
        img = cv2.warpAffine(self.Image, T, (w, h))
        self.Image = img
        self.displayImage(self.Image, self.label_2)

    def rotasi90derajat(self):
        self.rotasi(90)

    def rotasi(self, degree):
        h, w = self.Image.shape[:2]
        rotationMatrix = cv2.getRotationMatrix2D((w / 2, h / 2), degree, 0.7)
        cos = np.abs(rotationMatrix[0, 0])
        sin = np.abs(rotationMatrix[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        rotationMatrix[0, 2] += (nW / 2) - w / 2
        rotationMatrix[1, 2] += (nH / 2) - h / 2
        rot_image = cv2.warpAffine(self.Image, rotationMatrix, (nW, nH))
        self.Image = rot_image
        self.displayImage(self.Image, self.label_2)


    def zoomIn2x(self):
        self.zoomIn(self.Image)

    def zoomIn(self, Image):
        skala = 2
        resize_image = cv2.resize(self.Image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original', self.Image)
        cv2.imshow('Zoom In', resize_image)
        cv2.waitKey()

    def zoomout(self):
        skala = 0.5
        resize_image = cv2.resize(self.Image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original',self.Image)
        cv2.imshow('Zoom Out', resize_image)
        cv2.waitKey()

    def crop(self):
        h, w = self.Image.shape[:2]
        start_row, start_col = int(h * 0.1)
        end_row, end_col = int(h * 0.5), int(w * 0.5)
        crop = self. Image[start_row:end_row, start_col:end_col]
        cv2.imshow("Original", self.Image)
        cv2.imshow("Crop Image", crop)

#operasi aritmatika
    def aritmatika_tambah(self):
        image1 = cv2.imread('mengcling.jpg', 0)
        image2 = cv2.imread('mengkacamata.jpg', 0)
        image_tambah = image1 + image2
        image_kurang = image1 - image2
        cv2.imshow('Image 1 Original', image1)
        cv2.imshow('Image 2 Original', image2)
        cv2.imshow('Image Tambah', image_tambah)
        cv2.imshow('Image Kurang', image_kurang)
        cv2.waitKey()

    def aritmatika_kalibagi(self):
        image1 = cv2.imread('mengcling.jpg', 0)
        image2 = cv2.imread('mengkacamata.jpg', 0)
        image_tambah = image1 * image2
        image_kurang = image1 / image2
        cv2.imshow('Image 1 Original', image1)
        cv2.imshow('Image 2 Original', image2)
        cv2.imshow('Image Kali', image_tambah)
        cv2.imshow('Image Bagi', image_kurang)
        cv2.waitKey()

    def boolean_AND(self):
        image1 = cv2.imread('mengcling.jpg', 0)
        image2 = cv2.imread('mengkacamata.jpg', 0)
        operasi = cv2.bitwise_and(image1, image2)
        cv2.imshow("Image 1 Original", image1)
        cv2.imshow("Image 2 Original", image2)
        cv2.imshow("Image Operasi AND", operasi)
        cv2.waitKey()

    def boolean_OR(self):
        image1 = cv2.imread('mengcling.jpg', 0)
        image2 = cv2.imread('mengkacamata.jpg', 0)
        operasi = cv2.bitwise_or(image1, image2)
        cv2.imshow("Image 1 Original", image1)
        cv2.imshow("Image 2 Original", image2)
        cv2.imshow("Image Operasi AND", operasi)
        cv2.waitKey()

    def boolean_XOR(self):
        image1 = cv2.imread('mengcling.jpg', 0)
        image2 = cv2.imread('mengkacamata.jpg', 0)
        operasi = cv2.bitwise_xor(image1, image2)
        cv2.imshow("Image 1 Original", image1)
        cv2.imshow("Image 2 Original", image2)
        cv2.imshow("Image Operasi AND", operasi)
        cv2.waitKey()

    def displayImage(self, Image, label):
        qformat = QImage.Format_Indexed8

        if len(Image.shape) == 3:
            if Image.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(Image.data, Image.shape[1], Image.shape[0],
                     Image.strides[0], qformat)

        img = img.rgbSwapped()

        label.setPixmap(QPixmap.fromImage(img))


app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle('Pertemuan 1')
window.show()
sys.exit(app.exec_())
