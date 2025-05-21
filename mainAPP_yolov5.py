from PyQt6 import uic
from PyQt6.QtWidgets import QApplication, QDialog, QLineEdit, QPushButton, QLabel
from PyQt6.QtGui import QPixmap, QImage
import sys
import cv2
import os
import csv
import shutil
import glob
from yolov5.detect import run
from motion_controller import execute_motion
from serial_port_selector import SerialPortSelector

class ParkingApp(QDialog):
    def __init__(self):
        super().__init__()
        uic.loadUi("D:\\2025_robot\\20250514_humanoid_yolov5\\res\\parking.ui", self)

        self.name_input = self.findChild(QLineEdit, "name_put")
        self.phone_input = self.findChild(QLineEdit, "phone_put")
        self.memo_input = self.findChild(QLineEdit, "memo_put")
        self.picture_name_input = self.findChild(QLineEdit, "picture_put")
        self.save_button = self.findChild(QPushButton, "save")
        self.picture_button = self.findChild(QPushButton, "picture")
        self.cam_label = self.findChild(QLabel, "cam")
        self.port_label = self.findChild(QLabel, "lblPort")
        self.port_select_button = self.findChild(QPushButton, "pushButton_6")

        self.save_button.clicked.connect(self.save_data)
        self.picture_button.clicked.connect(self.capture_picture)
        self.port_select_button.clicked.connect(self.open_port_selector)

        self.selected_port = None

        self.image_save_path = "image"
        if not os.path.exists(self.image_save_path):
            os.makedirs(self.image_save_path)

        self.cap = cv2.VideoCapture(0)
        self.current_frame = None
        self.timer_active = True
        self.start_webcam()

    def open_port_selector(self):
        selector = SerialPortSelector()
        if selector.exec():
            self.selected_port = selector.selected_port
            if self.selected_port:
                self.port_label.setText(self.selected_port)
                print(f"선택된 포트: {self.selected_port}")
            else:
                self.port_label.setText("None")

    def start_webcam(self):
        def update_frame():
            if self.timer_active:
                ret, frame = self.cap.read()
                if ret:
                    detected_frame = self.detect_objects(frame)

                    rgb_image = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                    pixmap = QPixmap.fromImage(qt_image)
                    self.cam_label.setPixmap(pixmap.scaled(self.cam_label.width(), self.cam_label.height()))

        self.timer = self.startTimer(30)
        self.timerEvent = lambda event: update_frame()

    def get_latest_exp_dir(self):
        exp_dirs = glob.glob("runs/detect/exp*")
        if not exp_dirs:
            return None
        return max(exp_dirs, key=os.path.getctime)

    def detect_objects(self, frame):
        temp_image_path = "temp_frame.jpg"
        cv2.imwrite(temp_image_path, frame)

        output_dir = "runs/detect"
        run(
            weights="D:\\2025_robot\\20250514_humanoid_yolov5\\best.pt",
            source=temp_image_path,
            project="runs/detect",
            name="exp",
            exist_ok=True,
            save_txt=True,
            save_conf=True
        )

        exp_dir = self.get_latest_exp_dir()
        if exp_dir is None:
            print("[오류] YOLO 탐지 결과 디렉토리가 없습니다.")
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
            return frame

        detected_image_path = os.path.join(exp_dir, os.path.basename(temp_image_path))
        detected_frame = frame

        if os.path.exists(detected_image_path):
            detected_frame = cv2.imread(detected_image_path)
            self.current_frame = detected_frame

            # 탐지된 객체에 따라 모션 실행
            self.detect_and_execute_motion(detected_image_path)

            os.remove(detected_image_path)

        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

        # exp_dir 삭제 (주의: 다른 곳에서 쓰는 중이라면 삭제하면 안됨)
        if os.path.exists(exp_dir):
            shutil.rmtree(exp_dir)

        return detected_frame

    def detect_and_execute_motion(self, detected_img_path):
        if not self.selected_port:
            print("포트가 선택되지 않았습니다. 동작을 실행하지 않습니다.")
            return

        label_path = os.path.join("runs", "detect", "exp", "labels", os.path.basename(detected_img_path).replace(".jpg", ".txt"))

        if not os.path.exists(label_path):
            print(f"[경고] 탐지 결과 라벨 파일이 없습니다: {label_path}")
            return

        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            classes = [line.split()[0] for line in lines]

            print("탐지된 클래스:", classes)

            if "0" in classes:
                print("빈자리 탐지됨 → 18번 동작 실행")
                execute_motion(self.selected_port, 18)
            if "1" in classes:
                print("찬자리 탐지됨 → 22번 동작 실행")
                execute_motion(self.selected_port, 22)

    def capture_picture(self):
        if self.current_frame is not None:
            picture_name = "cam_save.jpg"
            file_name = os.path.join(self.image_save_path, picture_name)
            cv2.imwrite(file_name, self.current_frame)
            print(f"사진이 저장되었습니다: {file_name}")
            self.picture_name_input.setText(file_name)
        else:
            print("현재 프레임이 None입니다. 웹캠이 제대로 작동하는지 확인하세요.")

    def save_data(self):
        name = self.name_input.text()
        phone = self.phone_input.text()
        memo = self.memo_input.text()
        image_file = self.picture_name_input.text()

        if name.strip() and phone.strip() and os.path.exists(image_file):
            file_exists = os.path.isfile("parking_data.csv")
            try:
                with open("parking_data.csv", "a", newline="", encoding="utf-8-sig") as file:
                    writer = csv.writer(file)
                    if not file_exists:
                        writer.writerow(["이름", "전화번호", "메모", "사진"])
                    writer.writerow([name, phone, memo, image_file])
                print(f"저장된 데이터: 이름={name}, 전화번호={phone}, 메모={memo}, 사진={image_file}")
            except Exception as e:
                print(f"데이터 저장 중 오류 발생: {e}")
        else:
            print("이름, 전화번호, 사진은 필수 입력 항목입니다.")
            return

        QApplication.quit()

    def closeEvent(self, event):
        self.cap.release()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    parking_app = ParkingApp()
    parking_app.show()
    sys.exit(app.exec())