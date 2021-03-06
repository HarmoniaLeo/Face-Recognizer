# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitle2.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
import cv2


class Ui_Dialog(object):
    def setupUi(self, Dialog,mainDialog,img,vec,label,p):
        self.Dialog=Dialog
        self.mainDialog=mainDialog
        self.vec=vec
        self.Dialog.setObjectName("Dialog")
        self.Dialog.resize(221, 324)
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(10, 10, 200, 200))
        self.label.setText("")
        self.label.setObjectName("label")
        self.predictLabel = QtWidgets.QLabel(self.Dialog)
        self.predictLabel.setGeometry(QtCore.QRect(10, 230, 201, 16))
        self.predictLabel.setObjectName("predictLabel")
        self.markEdit = QtWidgets.QLineEdit(self.Dialog)
        self.markEdit.setGeometry(QtCore.QRect(10, 260, 201, 21))
        self.markEdit.setObjectName("markEdit")
        self.addButton = QtWidgets.QPushButton(self.Dialog)
        self.addButton.setGeometry(QtCore.QRect(10, 290, 201, 32))
        self.addButton.setObjectName("addButton")

        self.retranslateUi(self.Dialog,label,p)
        QtCore.QMetaObject.connectSlotsByName(self.Dialog)

        self.addButton.clicked.connect(self.addButtonClicked)

        height, width, depth = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = QtGui.QImage(img.data, width, height, width * depth, QtGui.QImage.Format_RGB888)
        img=img.scaled(self.label.height(),self.label.width())
        img=QtGui.QPixmap(img)
        self.label.setPixmap(img)

    def retranslateUi(self, Dialog,label,p):
        _translate = QtCore.QCoreApplication.translate
        self.Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.predictLabel.setText(_translate("Dialog", "Predict: "+label+" ( {0:.2f} ) ".format(p)))
        self.markEdit.setPlaceholderText(_translate("Dialog", "Input the true name of the face"))
        self.addButton.setText(_translate("Dialog", "Add to dataset"))

    def addButtonClicked(self):
        self.mainDialog.addButtonClicked(self.vec,self.markEdit.text())
        self.Dialog.close()