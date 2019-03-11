import sys
import ALG.loaddata as alg

from PyQt5 import QtGui
from PyQt5.QtWidgets import (QMainWindow, QAction, QApplication, QSlider, QSpinBox, QLabel)
from PyQt5.QtCore import Qt

global picsrc

class UI(QMainWindow):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        global picsrc
        picsrc = "../IMG/1.jpg"
        self.statusBar()

        # 添加滚动栏和微调框以及图片位置
        lb1 = QLabel('1',self)
        lb2 = QLabel('100',self)
        lb1.setGeometry(20,550,30,20)
        lb2.setGeometry(140,550,30,20)
        pic = QLabel(self)
        picdata = QtGui.QPixmap(picsrc)
        pic.setPixmap(picdata)
        pic.setGeometry(10,10,640,240)

        self.sld = QSlider(Qt.Horizontal,self)
        self.sld.setFocusPolicy(Qt.NoFocus)
        self.sld.setGeometry(30,550,100,20)
        self.sld.setMinimum(1)
        self.sld.setMaximum(100)

        self.sld.valueChanged.connect(self.changeValue)


        self.spin = QSpinBox(self)
        self.spin.setRange(1,100)
        self.spin.setGeometry(200,550,70,20)

        self.spin.valueChanged.connect(self.changeValue)


        # 添加菜单栏
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(alg.loadmat('../DATA/14.mat'))

        viewMenu = menubar.addMenu('&View')


        functionMenu = menubar.addMenu('&Function')

        processing = menubar.addMenu('&Processing')

        self.setGeometry(300, 300,1000, 600)
        self.setWindowTitle('Thermal-imaging')
        self.show()

    def changeValue(self, value):
        global picsrc
        self.spin.setValue(value)
        self.sld.setValue(value)
        picsrc = '../IMG/'+str(value) + ".jpg"
        print(picsrc)






if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = UI()
    sys.exit(app.exec_())