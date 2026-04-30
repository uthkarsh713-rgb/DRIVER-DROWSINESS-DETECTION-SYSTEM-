import cv2
import dlib
import numpy as np
import time
import os
from tensorflow.keras.models import load_model
from pygame import mixer
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

# === INITIAL SETUP ===
try:
    mixer.init()
    mixer.music.load(os.path.join("project", "music.wav"))
except Exception as e:
    print(f"[ERROR] Audio load failed: {e}")

try:
    model = load_model("drowsiness_cnn_model.h5")
except Exception as e:
    print(f"[ERROR] Model load failed: {e}")
    exit()

face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(os.path.join("models", "shape_predictor_68_face_landmarks.dat"))

(lStart, lEnd) = (42, 48)  # Right eye
(rStart, rEnd) = (36, 42)  # Left eye

flag = 0
frame_check = 10
alarm_trigger_time = None
alarm_duration = 5  # seconds

# === FUNCTION TO PROCESS EYE IMAGE ===
def process_eye(eye_img):
    try:
        eye_img = cv2.resize(eye_img, (24, 24))
        eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
        eye_img = eye_img.astype("float32") / 255.0
        eye_img = eye_img.reshape(1, 24, 24, 1)
        return eye_img
    except Exception as e:
        print(f"[WARNING] Eye processing failed: {e}")
        return None

# === PULL OVER ANIMATION ===
def pull_over_animation():
    fig, ax = plt.subplots(figsize=(5, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("gray")
    ax.set_title("PARKED SAFELY", fontsize=14, color='lime')

    ax.plot([2, 2], [0, 10], 'w-', linewidth=2)
    ax.plot([8, 8], [0, 10], 'w-', linewidth=2)
    for i in range(0, 11, 2):
        ax.plot([5, 5], [i, i + 1], 'w--', linewidth=2)

    car = patches.Rectangle((4.5, 9), 1, 2, color='blue', ec='black')
    ax.add_patch(car)
    alert_text = ax.text(2.5, 8, "", fontsize=12, color='red', weight='bold')

    def update(frame_num):
        x_start = 4.5
        y_pos = 9 - frame_num * 0.4
        x_pos = max(1.3, x_start - max(0, (frame_num - 5) * 0.3))

        if frame_num < 5:
            alert_text.set_text("DROWSINESS DETECTED!")
            if not mixer.music.get_busy():
                mixer.music.play(-1)
        elif frame_num < 20:
            alert_text.set_text("PULLING OVER FOR SAFETY")
        else:
            alert_text.set_text("")

        if frame_num == 24:
            mixer.music.stop()
            alert_text.set_text("")

        car.set_xy((x_pos, y_pos))
        return car, alert_text

    ani = animation.FuncAnimation(fig, update, frames=25, interval=300, blit=True, repeat=False)
    plt.show(block=False)
    plt.pause(8)
    plt.close()

# === CAMERA LOOP ===
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Webcam not accessible.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    for face in faces:
        shape = landmark_predictor(gray, face)
        shape_np = np.array([[p.x, p.y] for p in shape.parts()])

        leftEye = shape_np[lStart:lEnd]
        rightEye = shape_np[rStart:rEnd]

        def crop_eye(eye):
            x1, y1 = np.min(eye[:, 0]), np.min(eye[:, 1])
            x2, y2 = np.max(eye[:, 0]), np.max(eye[:, 1])
            margin = 5
            return frame[max(0, y1 - margin):y2 + margin, max(0, x1 - margin):x2 + margin]

        left_eye_img = crop_eye(leftEye)
        right_eye_img = crop_eye(rightEye)

        if left_eye_img.size == 0 or right_eye_img.size == 0:
            print("[INFO] Eye not properly detected. Skipping frame.")
            continue

        cv2.imshow("Left Eye", left_eye_img)
        cv2.imshow("Right Eye", right_eye_img)

        left_input = process_eye(left_eye_img)
        right_input = process_eye(right_eye_img)

        if left_input is None or right_input is None:
            continue

        left_pred = model.predict(left_input)[0][0]
        right_pred = model.predict(right_input)[0][0]

        # Display live predictions
        text = f"Left: {left_pred:.2f} | Right: {right_pred:.2f}"
        print(text)
        cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Use prediction threshold < 0.5 for closed eyes
        both_eyes_closed = left_pred < 0.5 and right_pred < 0.5

        if both_eyes_closed:
            flag += 1
        else:
            flag = max(0, flag - 1)

        if flag >= frame_check:
            cv2.putText(frame, "******** DROWSY ********", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if not mixer.music.get_busy():
                mixer.music.play(-1)
                alarm_trigger_time = time.time()

    if alarm_trigger_time and time.time() - alarm_trigger_time > alarm_duration:
        cap.release()
        cv2.destroyAllWindows()
        pull_over_animation()
        break

    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        mixer.music.stop()
        break

cap.release()
cv2.destroyAllWindows()
