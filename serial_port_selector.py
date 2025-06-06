import os
import sys
from PyQt6.QtWidgets import QApplication, QDialog
from serial.tools.list_ports import comports
from PyQt6 import uic
from PyQt6.QtWidgets import QMessageBox

# UI 파일 경로 설정
ui_file_path = os.path.join(os.path.dirname(__file__), "res", "findComPort.ui")

try:
    form_findComPort = uic.loadUiType(ui_file_path)[0]
except FileNotFoundError:
    print(f"Error: UI file not found at {ui_file_path}")
    sys.exit(1)

class SerialPortSelector(QDialog, form_findComPort):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.selected_port = None  # 전역 변수 대신 인스턴스 변수 사용
        
        self.populate_ports()
        self.pushButton_1.clicked.connect(self.populate_ports)
        self.pushButton_2.clicked.connect(self.handle_confirm)
        self.pushButton_3.clicked.connect(self.reject)
    
    def populate_ports(self):
        ports = [port.device for port in comports()]
        self.comboBox.clear()  # 기존 항목 제거
        self.comboBox.addItems(ports)
    
    def handle_confirm(self):
        self.selected_port = self.comboBox.currentText()
        if self.selected_port:
            self.accept()
    
    def auto_select_cp2104_port(self):
        """
        자동으로 cp2104가 포함된 포트를 선택합니다.
        """
        ports = comports()
        for port in ports:
            desc = port.description.lower()
            hwid = port.hwid.lower()
            if "cp210" in desc or "silicon labs" in desc or "10c4:ea60" in hwid:
                self.selected_port = port.device
                self.comboBox.setCurrentText(port.device)
                print(f"자동으로 선택된 포트: {port.device}")
                return
        QMessageBox.information(self, "정보", "cp2104 포트를 찾을 수 없습니다.")
    
    @staticmethod
    def launch(parent=None):
        selector = SerialPortSelector(parent)
        result = selector.exec()
        if result:
            return selector.selected_port
        return None

if __name__ == "__main__":
    # PyQt 애플리케이션 초기화
    app = QApplication(sys.argv)

    # 포트 선택 다이얼로그 실행
    selector = SerialPortSelector()
    selector.show()
    # 애플리케이션 실행
    sys.exit(app.exec())