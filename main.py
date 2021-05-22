import sys
from PyQt5.QtWidgets import QApplication, QDialog
import myUi

if __name__=='__main__':
    app = QApplication(sys.argv)
    MainWindow = QDialog()
    ui = myUi.Ui_Dialog()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())