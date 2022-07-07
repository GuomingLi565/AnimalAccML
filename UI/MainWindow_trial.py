import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from Ui_MainWindow import Ui_MainWindow

class MainWindow:
    def __init__(self):
        self.main_win = QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.main_win)

        self.ui.stackedWidget.setCurrentWidget(self.ui.page_home)

        self.ui.Blue_btn.clicked.connect(self.showBlue)
        self.ui.Red_btn.clicked.connect(self.showRed)
        self.ui.Yellow_btn.clicked.connect(self.showYellow)

    def show(self):
        self.main_win.show()

    def showBlue(self):
        self.ui.stackedWidget.setCurrentWidget(self.ui.page_blue)

    def showRed(self):
        self.ui.stackedWidget.setCurrentWidget(self.ui.page_red)

    def showYellow(self):
        self.ui.stackedWidget.setCurrentWidget(self.ui.page_yellow)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())