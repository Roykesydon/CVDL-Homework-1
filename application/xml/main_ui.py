# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(996, 698)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(40, 20, 161, 331))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.loadFolder = QtWidgets.QPushButton(self.groupBox)
        self.loadFolder.setGeometry(QtCore.QRect(20, 42, 121, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.loadFolder.setFont(font)
        self.loadFolder.setObjectName("loadFolder")
        self.loadImageL = QtWidgets.QPushButton(self.groupBox)
        self.loadImageL.setGeometry(QtCore.QRect(20, 140, 121, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.loadImageL.setFont(font)
        self.loadImageL.setObjectName("loadImageL")
        self.loadImageR = QtWidgets.QPushButton(self.groupBox)
        self.loadImageR.setGeometry(QtCore.QRect(20, 240, 121, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.loadImageR.setFont(font)
        self.loadImageR.setObjectName("loadImageR")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(230, 20, 241, 331))
        self.groupBox_2.setObjectName("groupBox_2")
        self.findCorners = QtWidgets.QPushButton(self.groupBox_2)
        self.findCorners.setGeometry(QtCore.QRect(20, 30, 201, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.findCorners.setFont(font)
        self.findCorners.setObjectName("findCorners")
        self.findIntrinsic = QtWidgets.QPushButton(self.groupBox_2)
        self.findIntrinsic.setGeometry(QtCore.QRect(20, 70, 201, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.findIntrinsic.setFont(font)
        self.findIntrinsic.setObjectName("findIntrinsic")
        self.groupBox_4 = QtWidgets.QGroupBox(self.groupBox_2)
        self.groupBox_4.setGeometry(QtCore.QRect(20, 120, 201, 91))
        self.groupBox_4.setObjectName("groupBox_4")
        self.extrinsicNumber = QtWidgets.QComboBox(self.groupBox_4)
        self.extrinsicNumber.setGeometry(QtCore.QRect(70, 20, 69, 22))
        self.extrinsicNumber.setObjectName("extrinsicNumber")
        self.extrinsicNumber.addItem("")
        self.extrinsicNumber.addItem("")
        self.extrinsicNumber.addItem("")
        self.extrinsicNumber.addItem("")
        self.extrinsicNumber.addItem("")
        self.extrinsicNumber.addItem("")
        self.extrinsicNumber.addItem("")
        self.extrinsicNumber.addItem("")
        self.extrinsicNumber.addItem("")
        self.extrinsicNumber.addItem("")
        self.extrinsicNumber.addItem("")
        self.extrinsicNumber.addItem("")
        self.extrinsicNumber.addItem("")
        self.extrinsicNumber.addItem("")
        self.extrinsicNumber.addItem("")
        self.findExtrinsic = QtWidgets.QPushButton(self.groupBox_4)
        self.findExtrinsic.setGeometry(QtCore.QRect(20, 50, 161, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.findExtrinsic.setFont(font)
        self.findExtrinsic.setObjectName("findExtrinsic")
        self.findDistortion = QtWidgets.QPushButton(self.groupBox_2)
        self.findDistortion.setGeometry(QtCore.QRect(20, 230, 201, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.findDistortion.setFont(font)
        self.findDistortion.setObjectName("findDistortion")
        self.showResult = QtWidgets.QPushButton(self.groupBox_2)
        self.showResult.setGeometry(QtCore.QRect(20, 280, 201, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.showResult.setFont(font)
        self.showResult.setObjectName("showResult")
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(500, 20, 231, 331))
        self.groupBox_3.setObjectName("groupBox_3")
        self.showWordsHorizontal = QtWidgets.QPushButton(self.groupBox_3)
        self.showWordsHorizontal.setGeometry(QtCore.QRect(20, 100, 191, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.showWordsHorizontal.setFont(font)
        self.showWordsHorizontal.setObjectName("showWordsHorizontal")
        self.showWordsVertical = QtWidgets.QPushButton(self.groupBox_3)
        self.showWordsVertical.setGeometry(QtCore.QRect(20, 160, 191, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.showWordsVertical.setFont(font)
        self.showWordsVertical.setObjectName("showWordsVertical")
        self.wordsText = QtWidgets.QTextEdit(self.groupBox_3)
        self.wordsText.setGeometry(QtCore.QRect(20, 40, 191, 31))
        self.wordsText.setObjectName("wordsText")
        self.groupBox_5 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_5.setGeometry(QtCore.QRect(750, 20, 211, 331))
        self.groupBox_5.setObjectName("groupBox_5")
        self.stereoDisparityMap = QtWidgets.QPushButton(self.groupBox_5)
        self.stereoDisparityMap.setGeometry(QtCore.QRect(20, 150, 171, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.stereoDisparityMap.setFont(font)
        self.stereoDisparityMap.setObjectName("stereoDisparityMap")
        self.groupBox_6 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_6.setGeometry(QtCore.QRect(210, 370, 211, 271))
        self.groupBox_6.setObjectName("groupBox_6")
        self.loadImage1 = QtWidgets.QPushButton(self.groupBox_6)
        self.loadImage1.setGeometry(QtCore.QRect(20, 30, 171, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.loadImage1.setFont(font)
        self.loadImage1.setObjectName("loadImage1")
        self.loadImage2 = QtWidgets.QPushButton(self.groupBox_6)
        self.loadImage2.setGeometry(QtCore.QRect(20, 90, 171, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.loadImage2.setFont(font)
        self.loadImage2.setObjectName("loadImage2")
        self.keypoints = QtWidgets.QPushButton(self.groupBox_6)
        self.keypoints.setGeometry(QtCore.QRect(20, 150, 171, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.keypoints.setFont(font)
        self.keypoints.setObjectName("keypoints")
        self.matchedKeypoints = QtWidgets.QPushButton(self.groupBox_6)
        self.matchedKeypoints.setGeometry(QtCore.QRect(20, 210, 171, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.matchedKeypoints.setFont(font)
        self.matchedKeypoints.setObjectName("matchedKeypoints")
        self.groupBox_7 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_7.setGeometry(QtCore.QRect(450, 370, 491, 271))
        self.groupBox_7.setObjectName("groupBox_7")
        self.loadImage = QtWidgets.QPushButton(self.groupBox_7)
        self.loadImage.setGeometry(QtCore.QRect(20, 20, 171, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.loadImage.setFont(font)
        self.loadImage.setObjectName("loadImage")
        self.showAugmentedImages = QtWidgets.QPushButton(self.groupBox_7)
        self.showAugmentedImages.setGeometry(QtCore.QRect(20, 70, 171, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.showAugmentedImages.setFont(font)
        self.showAugmentedImages.setObjectName("showAugmentedImages")
        self.showModelStructure = QtWidgets.QPushButton(self.groupBox_7)
        self.showModelStructure.setGeometry(QtCore.QRect(20, 120, 171, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.showModelStructure.setFont(font)
        self.showModelStructure.setObjectName("showModelStructure")
        self.showAccuracyAndLoss = QtWidgets.QPushButton(self.groupBox_7)
        self.showAccuracyAndLoss.setGeometry(QtCore.QRect(20, 170, 171, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.showAccuracyAndLoss.setFont(font)
        self.showAccuracyAndLoss.setObjectName("showAccuracyAndLoss")
        self.inference = QtWidgets.QPushButton(self.groupBox_7)
        self.inference.setGeometry(QtCore.QRect(20, 220, 171, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.inference.setFont(font)
        self.inference.setObjectName("inference")
        self.label = QtWidgets.QLabel(self.groupBox_7)
        self.label.setGeometry(QtCore.QRect(230, 29, 91, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.predictText = QtWidgets.QLabel(self.groupBox_7)
        self.predictText.setGeometry(QtCore.QRect(290, 29, 71, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.predictText.setFont(font)
        self.predictText.setText("")
        self.predictText.setObjectName("predictText")
        self.inferenceImage = QtWidgets.QLabel(self.groupBox_7)
        self.inferenceImage.setGeometry(QtCore.QRect(230, 80, 128, 128))
        self.inferenceImage.setText("")
        self.inferenceImage.setObjectName("inferenceImage")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 996, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "Load Image"))
        self.loadFolder.setText(_translate("MainWindow", "Load Folder"))
        self.loadImageL.setText(_translate("MainWindow", "Load Image_L"))
        self.loadImageR.setText(_translate("MainWindow", "Load Image_R"))
        self.groupBox_2.setTitle(_translate("MainWindow", "1. Calibration"))
        self.findCorners.setText(_translate("MainWindow", "1.1 Find Corners"))
        self.findIntrinsic.setText(_translate("MainWindow", "1.2 Find Intrinsic"))
        self.groupBox_4.setTitle(_translate("MainWindow", "1.3 Find Extrinsic"))
        self.extrinsicNumber.setItemText(0, _translate("MainWindow", "1"))
        self.extrinsicNumber.setItemText(1, _translate("MainWindow", "2"))
        self.extrinsicNumber.setItemText(2, _translate("MainWindow", "3"))
        self.extrinsicNumber.setItemText(3, _translate("MainWindow", "4"))
        self.extrinsicNumber.setItemText(4, _translate("MainWindow", "5"))
        self.extrinsicNumber.setItemText(5, _translate("MainWindow", "6"))
        self.extrinsicNumber.setItemText(6, _translate("MainWindow", "7"))
        self.extrinsicNumber.setItemText(7, _translate("MainWindow", "8"))
        self.extrinsicNumber.setItemText(8, _translate("MainWindow", "9"))
        self.extrinsicNumber.setItemText(9, _translate("MainWindow", "10"))
        self.extrinsicNumber.setItemText(10, _translate("MainWindow", "11"))
        self.extrinsicNumber.setItemText(11, _translate("MainWindow", "12"))
        self.extrinsicNumber.setItemText(12, _translate("MainWindow", "13"))
        self.extrinsicNumber.setItemText(13, _translate("MainWindow", "14"))
        self.extrinsicNumber.setItemText(14, _translate("MainWindow", "15"))
        self.findExtrinsic.setText(_translate("MainWindow", "1.3 Find Extrinsic"))
        self.findDistortion.setText(_translate("MainWindow", "1.4 Find Distortion"))
        self.showResult.setText(_translate("MainWindow", "1.5 Show Result"))
        self.groupBox_3.setTitle(_translate("MainWindow", "2. Augmented Reality"))
        self.showWordsHorizontal.setText(
            _translate("MainWindow", "2.1 show words on board")
        )
        self.showWordsVertical.setText(
            _translate("MainWindow", "2.2 show words vertical")
        )
        self.groupBox_5.setTitle(_translate("MainWindow", "3. Stereo disparity map"))
        self.stereoDisparityMap.setText(
            _translate("MainWindow", "3.1 stereo disparity map")
        )
        self.groupBox_6.setTitle(_translate("MainWindow", "4. SIFT"))
        self.loadImage1.setText(_translate("MainWindow", "Load Image1"))
        self.loadImage2.setText(_translate("MainWindow", "Load Image2"))
        self.keypoints.setText(_translate("MainWindow", "4.1 Keypoints"))
        self.matchedKeypoints.setText(_translate("MainWindow", "4.2 Matched Keypoints"))
        self.groupBox_7.setTitle(_translate("MainWindow", "5. VGG19"))
        self.loadImage.setText(_translate("MainWindow", "Load Image"))
        self.showAugmentedImages.setText(
            _translate("MainWindow", "5.1 Show Augmented Images")
        )
        self.showModelStructure.setText(
            _translate("MainWindow", "5.2 Show Model Structure")
        )
        self.showAccuracyAndLoss.setText(
            _translate("MainWindow", "5.3 Show Accuracy and Loss")
        )
        self.inference.setText(_translate("MainWindow", "5.4 Inference"))
        self.label.setText(_translate("MainWindow", "Predict ="))