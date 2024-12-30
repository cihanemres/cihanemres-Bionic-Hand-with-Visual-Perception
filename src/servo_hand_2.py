import cv2
import numpy as np
import ElizlemeUnitesi as eiu
import pyfirmata2
import time
import json
import tkinter as tk
from tkinter import simpledialog, ttk
import serial.tools.list_ports

class EnhancedCalibration:
    def __init__(self, port):
        self.servo_controller = None
        self.port = port
        self.detector = eiu.elizleyici(detection_confidence=0.9)
        self.calibration_data = {"min": [], "max": [], "wrist": {"min_length": None, "max_length": None}}
        self.calibrated = False
        self.servo_angles = {
            "basparmak": 90,
            "isaretparmak": 90,
            "ortaparmak": 90,
            "yuzukparmak": 90,
            "serceparmak": 90,
            "bilek": 90
        }
        self.saved_once = False
        self.wrist_last_update = time.time()

    def setup_servo_controller(self):
        self.servo_controller = pyfirmata2.Arduino(self.port)
        self.servo_pins = {
            "basparmak": self.servo_controller.get_pin("d:3:s"),
            "isaretparmak": self.servo_controller.get_pin("d:5:s"),
            "ortaparmak": self.servo_controller.get_pin("d:6:s"),
            "yuzukparmak": self.servo_controller.get_pin("d:9:s"),
            "serceparmak": self.servo_controller.get_pin("d:10:s"),
            "bilek": self.servo_controller.get_pin("d:11:s")
        }
        print("Servo controller initialized")

    def display_instructions(self, img, message, y_offset=30):
        lines = message.split("\n")
        for i, line in enumerate(lines):
            cv2.putText(img, line, (10, y_offset + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def collect_calibration_data(self, lm_list, mode):
        if not lm_list:
            return False

        if mode == "wrist_min":
            length = np.linalg.norm(np.array(lm_list[4][1:]) - np.array(lm_list[20][1:]))
            self.calibration_data["wrist"]["min_length"] = length
        elif mode == "wrist_max":
            length = np.linalg.norm(np.array(lm_list[4][1:]) - np.array(lm_list[20][1:]))
            self.calibration_data["wrist"]["max_length"] = length
        elif mode in ["min", "max"]:
            finger_lengths = [
                np.linalg.norm(np.array(lm_list[4][1:]) - np.array(lm_list[0][1:])),
                np.linalg.norm(np.array(lm_list[8][1:]) - np.array(lm_list[0][1:])),
                np.linalg.norm(np.array(lm_list[12][1:]) - np.array(lm_list[0][1:])),
                np.linalg.norm(np.array(lm_list[16][1:]) - np.array(lm_list[0][1:])),
                np.linalg.norm(np.array(lm_list[20][1:]) - np.array(lm_list[0][1:]))
            ]
            self.calibration_data[mode] = finger_lengths

        return True

    def save_calibration(self):
        if not self.saved_once:
            with open("calibration_data.json", "w") as file:
                json.dump(self.calibration_data, file)
            print("Calibration data saved to calibration_data.json")
            self.saved_once = True

    def clear_calibration_data(self):
        self.calibration_data = {"min": [], "max": [], "wrist": {"min_length": None, "max_length": None}}
        with open("calibration_data.json", "w") as file:
            json.dump(self.calibration_data, file)
        print("Calibration data cleared")
        self.saved_once = False

    def calculate_wrist_angle(self, lm_list):
        if not self.calibration_data["wrist"]["min_length"] or not self.calibration_data["wrist"]["max_length"]:
            return 90

        current_time = time.time()

        length = np.linalg.norm(np.array(lm_list[4][1:]) - np.array(lm_list[20][1:]))
        thumb_y_diff = abs(lm_list[4][2] - lm_list[0][2])  # Difference in Y-axis for thumb movement

        if thumb_y_diff > 20:  # Threshold for thumb movement
            if current_time - self.wrist_last_update < 1:  # Add a delay for wrist updates
                return self.servo_angles["bilek"]  # Maintain current angle

        angle = np.interp(
            length,
            [self.calibration_data["wrist"]["min_length"], self.calibration_data["wrist"]["max_length"]],
            [0, 180]
        )
        self.wrist_last_update = current_time  # Update last wrist adjustment time
        return angle

    def update_servo_angles(self, lm_list):
        if not lm_list or not self.calibrated:
            return

        finger_lengths = [
            np.linalg.norm(np.array(lm_list[4][1:]) - np.array(lm_list[0][1:])),
            np.linalg.norm(np.array(lm_list[8][1:]) - np.array(lm_list[0][1:])),
            np.linalg.norm(np.array(lm_list[12][1:]) - np.array(lm_list[0][1:])),
            np.linalg.norm(np.array(lm_list[16][1:]) - np.array(lm_list[0][1:])),
            np.linalg.norm(np.array(lm_list[20][1:]) - np.array(lm_list[0][1:]))
        ]

        finger_names = ["basparmak", "isaretparmak", "ortaparmak", "yuzukparmak", "serceparmak"]
        for i, length in enumerate(finger_lengths):
            if self.calibration_data['min'] and self.calibration_data['max']:
                angle = np.interp(length, [self.calibration_data['min'][i], self.calibration_data['max'][i]], [0, 180])
                self.servo_angles[finger_names[i]] = angle

        self.servo_angles["bilek"] = self.calculate_wrist_angle(lm_list)

    def display_servo_angles(self, img):
        y_offset = 400
        for name, angle in self.servo_angles.items():
            cv2.putText(img, f"{name}: {angle:.1f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            y_offset += 30

    def update_servos(self):
        for name, angle in self.servo_angles.items():
            if name in self.servo_pins:
                self.servo_pins[name].write(angle)

    def run_calibration(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            print("Failed to access camera")
            return

        calibration_mode = None
        start_time = None

        while True:
            ret, img = cap.read()
            if not ret:
                print("Failed to capture frame")
                continue

            img = self.detector.find_hands(img)
            lm_list = self.detector.find_position(img, draw=True)

            if calibration_mode is not None:
                if calibration_mode == 'open':
                    self.display_instructions(img, "Open your hand and hold steady\nCalibration will start in 2 seconds")
                    if start_time is None:
                        start_time = time.time()
                    elif time.time() - start_time >= 2:
                        if self.collect_calibration_data(lm_list, 'max'):
                            self.display_instructions(img, "Open hand calibration is complete", y_offset=120)
                            time.sleep(2)
                            calibration_mode = None
                            start_time = None
                elif calibration_mode == 'close':
                    self.display_instructions(img, "Close your hand and hold steady\nCalibration will start in 2 seconds")
                    if start_time is None:
                        start_time = time.time()
                    elif time.time() - start_time >= 2:
                        if self.collect_calibration_data(lm_list, 'min'):
                            self.display_instructions(img, "Close hand calibration is complete", y_offset=120)
                            time.sleep(2)
                            calibration_mode = None
                            start_time = None
                elif calibration_mode == 'wrist_min':
                    self.display_instructions(img, "Turn your wrist to the minimum position\nCalibration will start in 2 seconds")
                    if start_time is None:
                        start_time = time.time()
                    elif time.time() - start_time >= 2:
                        if self.collect_calibration_data(lm_list, 'wrist_min'):
                            self.display_instructions(img, "Wrist minimum calibration is complete", y_offset=120)
                            time.sleep(2)
                            calibration_mode = None
                            start_time = None
                elif calibration_mode == 'wrist_max':
                    self.display_instructions(img, "Turn your wrist to the maximum position\nCalibration will start in 2 seconds")
                    if start_time is None:
                        start_time = time.time()
                    elif time.time() - start_time >= 2:
                        if self.collect_calibration_data(lm_list, 'wrist_max'):
                            self.display_instructions(img, "Wrist maximum calibration is complete", y_offset=120)
                            time.sleep(2)
                            calibration_mode = None
                            start_time = None

                if len(self.calibration_data['min']) and len(self.calibration_data['max']):
                    self.calibrated = True
                    self.save_calibration()
            else:
                self.display_instructions(img, "Press 'o' for open calibration\nPress 'c' for close calibration\nPress 's' to reset calibration\nPress 'm' for wrist minimum calibration\nPress 'x' for wrist maximum calibration")

            if self.calibrated:
                self.update_servo_angles(lm_list)
                self.update_servos()
                self.display_servo_angles(img)

            cv2.imshow("Calibration", img)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('o'):
                calibration_mode = 'open'
                start_time = None
                self.saved_once = False
            elif key == ord('c'):
                calibration_mode = 'close'
                start_time = None
                self.saved_once = False
            elif key == ord('s'):
                self.clear_calibration_data()
                self.display_instructions(img, "Calibration data cleared", y_offset=120)
                time.sleep(2)
            elif key == ord('m'):
                calibration_mode = 'wrist_min'
                start_time = None
                self.saved_once = False
            elif key == ord('x'):
                calibration_mode = 'wrist_max'
                start_time = None
                self.saved_once = False
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Select Arduino Port")

    ports = [port.device for port in serial.tools.list_ports.comports()]

    if not ports:
        print("No COM ports found. Exiting.")
        root.destroy()
        exit()

    selected_port = tk.StringVar(value=ports[0])

    def on_confirm():
        root.destroy()

    ttk.Label(root, text="Select Arduino Port:").pack(pady=10)
    port_menu = ttk.Combobox(root, textvariable=selected_port, values=ports, state="readonly")
    port_menu.pack(pady=5)
    ttk.Button(root, text="Confirm", command=on_confirm).pack(pady=10)

    root.mainloop()

    port = selected_port.get()

    if port:
        calibrator = EnhancedCalibration(port)
        calibrator.setup_servo_controller()
        calibrator.run_calibration()
    else:
        print("No port selected. Exiting.")
