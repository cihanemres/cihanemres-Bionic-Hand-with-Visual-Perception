import cv2
import mediapipe as mp
import time
import numpy as np


class elizleyici():
    def __init__(self, mode: bool = False, max_hands: int = 2, model_complexity: int = 1,
                 detection_confidence: float = 0.9, tracking_confidence: float = 0.9):
        self.mode = mode
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.model_complexity,
                                         self.detection_confidence, self.tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils
        self.tip_ids = [4, 8, 12, 16, 20]  # Parmak uçlarının landmark ID'leri

    def find_hands(self, img, draw: bool = True) -> cv2.Mat:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_no: int = 0, draw: bool = True) -> list:
        lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return lm_list

    def fingers_up(self, lm_list) -> list:
        fingers = []
        if len(lm_list) == 0:
            return fingers

        # Başparmak
        if lm_list[self.tip_ids[0]][1] > lm_list[self.tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 parmak
        for id in range(1, 5):
            if lm_list[self.tip_ids[id]][2] < lm_list[self.tip_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def get_hand_center(self, lm_list) -> tuple:
        if len(lm_list) == 0:
            return None
        x_coords = [lm[1] for lm in lm_list]
        y_coords = [lm[2] for lm in lm_list]
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))
        return (center_x, center_y)


def main():
    p_time = 0
    c_time = 0
    cap = cv2.VideoCapture(0)
    detector = elizleyici()
    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        lm_list = detector.find_position(img)

        if len(lm_list) != 0:
            fingers = detector.fingers_up(lm_list)
            print(f"Fingers up: {fingers}")

            center = detector.get_hand_center(lm_list)
            if center:
                cv2.circle(img, center, 8, (0, 255, 0), cv2.FILLED)
                cv2.putText(img, f"Center: {center}", (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        cv2.putText(img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()