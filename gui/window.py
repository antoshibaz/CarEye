import sys

from PyQt5 import QtWidgets
from PyQt5 import uic


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("MainForm.ui", self)
        # cp = QtWidgets.QDesktopWidget().availableGeometry()
        # self.setGeometry(cp)

        # okButton = QtWidgets.QPushButton("OK")
        # cancelButton = QtWidgets.QPushButton("Cancel")
        # hbox = QtWidgets.QHBoxLayout()
        # hbox.addStretch(1)
        # hbox.addWidget(okButton)
        # hbox.addWidget(cancelButton)
        # vbox = QtWidgets.QVBoxLayout()
        # vbox.addStretch(1)
        # vbox.addLayout(hbox)
        # self.mainWidget.setLayout(vbox)


def main():
    app = QtWidgets.QApplication(sys.argv)
    form = MainWindow()
    form.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
