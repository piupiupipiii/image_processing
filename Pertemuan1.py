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
        self.action3x.triggered.connect(self.zoomIn3x)
        self.action4x.triggered.connect(self.zoomIn4x)
        self.actionZoom_Out.triggered.connect(self.zoomout)
        self.actionCrop.triggered.connect(self.crop)

        #operasi aritmatika
        self.actionTambah_dan_Kurang.triggered.connect(self.aritmatika_tambah)
        self.actionKali_dan_Bagi.triggered.connect(self.aritmatika_kalibagi)


        #operasi boolean
        self.actionOperasi_AND.triggered.connect(self.boolean_and)
        self.actionOperasi_OR.triggered.connect(self.boolean_or)
        self.actionOperasi_XOR.triggered.connect(self.boolean_xor)

        self.actionfiltering.triggered.connect(self.konvol)
        self.actionmean.triggered.connect(self.mean)

#function operasi titik
    def fungsi(self):
        self.Image = cv2.imread('meng.jpg')
        self.displayImage(self.Image, self.label)

    def grayscale(self):
        H, W = self.Image.shape[:2] #untuk nyari koordinat image
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
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(
                    0.299 * self.Image[i, j, 0] +
                    0.587 * self.Image[i, j, 1] +
                    0.114 * self.Image[i, j, 2] , 0, 255)

        self.Image = gray
        self.displayImage(gray, self.label_2)
        hist, bins = np.histogram(self.Image.ravel(), 255, (0,255))
        plt.plot(hist)
        plt.title('Histogram Grayscale')
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
        h, w = self.Image.shape[:2] #variabel tinggi dan lebar
        quarter_h, quarter_w = h / 4, w / 4 #nyimpen nilai seperempat dari tinggi dan lebar
        T = np.float_([[1, 0, quarter_w], [0, 1, quarter_h]]) #menggunakan matrik 2 x3
        img = cv2.warpAffine(self.Image, T, (w, h)) #ditranslasi menggunakan matriks T
        self.Image = img
        self.displayImage(self.Image, self.label_2)

    def rotasi90derajat(self):
        self.rotasi(90)

    def rotasi(self, degree):
        h, w = self.Image.shape[:2]
        rotationMatrix = cv2.getRotationMatrix2D((w / 2, h / 2), degree, 0.7) #untuk menghitung rotasi berdasarkan titik tengahnya
        cos = np.abs(rotationMatrix[0, 0])
        sin = np.abs(rotationMatrix[0, 1])
        nW = int((h * sin) + (w * cos)) #hasil rotasi yg dikali citra
        nH = int((h * cos) + (w * sin)) #hasil rotasi
        rotationMatrix[0, 2] += (nW / 2) - w / 2 #agar tetap di tengah
        rotationMatrix[1, 2] += (nH / 2) - h / 2
        rot_image = cv2.warpAffine(self.Image, rotationMatrix, (nW, nH))
        self.Image = rot_image
        self.displayImage(self.Image, self.label_2)


    def zoomIn2x(self):
        skala = 2
        resize_image = cv2.resize(self.Image, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC) #nentuin zoom berdasakrkan skala dan bikin jadi bagus meskipun zoom
        cv2.imshow('Original', self.Image)
        cv2.imshow('Zoom In', resize_image)
        cv2.waitKey()

    def zoomIn3x(self):
        skala = 3
        resize_image = cv2.resize(self.Image, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original', self.Image)
        cv2.imshow('Zoom In', resize_image)
        cv2.waitKey()

    def zoomIn4x(self):
        skala = 4
        resize_image = cv2.resize(self.Image, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original', self.Image)
        cv2.imshow('Zoom In', resize_image)
        cv2.waitKey()


    def zoomout(self):
        skala = 0.5
        original = self.Image
        resize_image = cv2.resize(self.Image, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC)
        self.Image = resize_image  # Perbarui self.Image dengan citra yang di-zoomout
        self.displayImage(resize_image, self.label_2)
        cv2.imshow('Original', original)
        cv2.imshow('Zoom Out', resize_image)
        cv2.waitKey()

    def crop(self):
        H, W = self.Image.shape[:2]
        start_row, start_col = 150, 150
        end_row, end_col = H - 150, W - 150
        cropped_img = self.Image[start_row:end_row, start_col:end_col]
        cv2.imshow("Cropped Image", cropped_img)
        cv2.waitKey()

#operasi aritmatika
    def aritmatika_tambah(self):
        image1 = cv2.imread('mengcling.jpg', 0)
        image2 = cv2.imread('mengkacamata.jpg', 0)

        image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))

        image_tambah = image1 + image2
        image_kurang = image1 - image2 #hasilnya akan menampilkan gambar yang terlihat negatif
        cv2.imshow('Image 1 Original', image1)
        cv2.imshow('Image 2 Original', image2)
        cv2.imshow('Image Tambah', image_tambah)
        cv2.imshow('Image Kurang', image_kurang)
        cv2.waitKey()

    def aritmatika_kalibagi(self):
        image1 = cv2.imread('mengcling.jpg', 0)
        image2 = cv2.imread('mengkacamata.jpg', 0)

        image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0])) #biar gambar sama

        image_kali = image1 * image2
        image_bagi = image1 / image2
        cv2.imshow('Image 1 Original', image1)
        cv2.imshow('Image 2 Original', image2)
        cv2.imshow('Image Kali', image_kali)
        cv2.imshow('Image Bagi', image_bagi)
        cv2.waitKey()

    def boolean_and(self):
        image1 = cv2.imread('mengcling.jpg', 0)
        image2 = cv2.imread('mengkacamata.jpg', 0)

        image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))

        operasi = cv2.bitwise_and(image1, image2) #bitwise dipake buat menggabungkan citra dengan pake piksel mana yanga akan diambil berdasarkan operasi and
        cv2.imshow("Image 1 Original", image1)
        cv2.imshow("Image 2 Original", image2)
        cv2.imshow("Image Operasi AND", operasi)
        cv2.waitKey()

    def boolean_or(self):
        image1 = cv2.imread('mengcling.jpg', 0)
        image2 = cv2.imread('mengkacamata.jpg', 0)

        image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))

        operasi = cv2.bitwise_or(image1, image2)
        cv2.imshow("Image 1 Original", image1)
        cv2.imshow("Image 2 Original", image2)
        cv2.imshow("Image Operasi AND", operasi)
        cv2.waitKey()

    def boolean_xor(self):
        image1 = cv2.imread('mengcling.jpg', 0)
        image2 = cv2.imread('mengkacamata.jpg', 0)

        image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))

        operasi = cv2.bitwise_xor(image1, image2)
        cv2.imshow("Image 1 Original", image1)
        cv2.imshow("Image 2 Original", image2)
        cv2.imshow("Image Operasi AND", operasi)
        cv2.waitKey()

    def konvol(self):
        original_image = cv2.imread('buku_blur.jpg', 1)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

        cv2.imshow("Image Original", original_image)
        cv2.waitKey()

        # Kernel
        kernel = np.array([[6, 0, -6],
                           [6, 1, -6],
                           [6, 0, -6]])

        filtered_image = self.convolution(original_image, kernel)

        cv2.imshow("Image Filtered", filtered_image)
        cv2.waitKey()
        self.Image = filtered_image

        self.displayImage(filtered_image, self.label_2)

    def FilterB(self):
        img = cv2.imread('buku_blur.jpg', 1)  # Mengambil citra dari gambar yg sudah ada
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Ubah citra masukan menjadi grayscale

        # Kernel
        kernel = np.array([[6, 0, -6],
                           [6, 1, -6],
                           [6, 0, -6]])

        img_out = self.convolution(img, kernel)  # Lakukan konvolusi
        self.displayImage(2)  # Menampilkan citra hasil
        self.Image = img_out  # Simpan citra hasil di self.Image

    def convolution(self, X, F):
        height, width = X.shape
        kernel_height, kernel_width = F.shape
        H = kernel_height // 2
        W = kernel_width // 2
        out = np.zeros_like(X)

        for i in range(H + 1, height - H):
            for j in range(W + 1, width - W):
                Sum = 0
                for k in range(-H, H + 1):
                    for l in range(-W, W + 1):
                        a = X[i + k, j + l]
                        w = F[H + k, W + l]
                        Sum += (w * a)
                out[i, j] = Sum
        return out

    def mean(self):
        image1 = cv2.imread('buku_blur.jpg', 1)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

        cv2.imshow("Image Original", image1)
        cv2.waitKey()

        kernel = np.array([[1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1]])

        image2 = self.convolution(image1, kernel)

        cv2.imshow("Image Filtered", image2)
        cv2.waitKey()
        self.Image = image2

        self.displayImage(image2, self.label_2)


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

        if window == 1:
            self.labelAwal.setPixmap(QPixmap.fromImage(img))
            self.labelAwal.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.labelAwal.setScaledContents(True)
        if window == 2:
            self.labelGUI.setPixmap(QPixmap.fromImage(img))
            self.labelGUI.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.labelGUI.setScaledContents(True)

        label.setPixmap(QPixmap.fromImage(img))


app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle('Pertemuan 1')
window.show()
sys.exit(app.exec_())

# if __name__ == '__main__':
#     test = np.array(
#         [
#             [1,2,2],
#             [3,3,3],
#             [1,1,4],
#         ]
#         )
#
#     img1 = cv2.imread('C:\\Users\\Silvy Nur Azkia\\Downloads\\buku_blur.jpg')
#     img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#     hasil = Filtering.Konvolusi(img1, test)
#     plt.imshow(hasil, cmap='gray', interpolation='bicubic')
#     plt.xticks([], plt.ystick([]))
#     plt.show()
#     cv2.waitKey()
#     cv2.destroyAllWindows()