import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from Ui_MainWindow import Ui_MainWindow

class MainWindow:
    def __init__(self):
        self.main_win = QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.main_win)

        self.ui.stackedWidget.setCurrentWidget(self.ui.page_home)

        self.ui.pushButton_home.clicked.connect(self.showHome)
        self.ui.pushButton_ProjectManagement.clicked.connect(self.showProjectManagement)
        self.ui.pushButton_Preprocessing.clicked.connect(self.showPreprocessing)
        self.ui.pushButton_MLModeling.clicked.connect(self.showMLModeling)
        self.ui.pushButton_BehaviorAnalysis.clicked.connect(self.showBehaviorAnalysis)
        self.ui.pushButton_Help.clicked.connect(self.showHelp)

    def show(self):
        self.main_win.show()

    def showHome(self):
        self.ui.stackedWidget.setCurrentWidget(self.ui.page_home)

    def showProjectManagement(self):
        self.ui.stackedWidget.setCurrentWidget(self.ui.page_ProjectManagement)

    def showPreprocessing(self):
        self.ui.stackedWidget.setCurrentWidget(self.ui.page_Preprocessing)

    def showMLModeling(self):
        self.ui.stackedWidget.setCurrentWidget(self.ui.page_MLModeling)

    def showBehaviorAnalysis(self):
        self.ui.stackedWidget.setCurrentWidget(self.ui.page_BehaviorAnalysis)

    def showHelp(self):
        self.ui.stackedWidget.setCurrentWidget(self.ui.page_Help)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())