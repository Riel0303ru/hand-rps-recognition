import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# Load model
model = tf.keras.models.load_model("model/rps_model.h5")
class_names = ['paper', 'rock', 'scissors']
img_size = (224, 224)

# Setup Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Game logic: siapa menang
def decide_winner(move1, move2):
    if move1 == move2:
        return "DRAW"
    elif (move1 == "rock" and move2 == "scissors") or \
         (move1 == "paper" and move2 == "rock") or \
         (move1 == "scissors" and move2 == "paper"):
        return "LEFT WINS"
    else:
        return "RIGHT WINS"

# Kamera
cap = cv2.VideoCapture(0)
cv2.namedWindow("🎮 RPS Game", cv2.WINDOW_NORMAL)
cv2.resizeWindow("🎮 RPS Game", 1280, 720)

print("🎮 Mainkan dengan 2 tangan! Tekan 'q' buat keluar.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Kamera error.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    moves = []  # buat nyimpen hasil prediksi (max 2 tangan)
    positions = []  # posisi tangan (buat nulis teks)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Bounding box
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min = int(min(x_coords) * w) - 20
            y_min = int(min(y_coords) * h) - 20
            x_max = int(max(x_coords) * w) + 20
            y_max = int(max(y_coords) * h) + 20

            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)

            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size == 0:
                continue

            resized = cv2.resize(hand_img, img_size)
            norm_img = resized / 255.0
            input_img = np.expand_dims(norm_img, axis=0)

            prediction = model.predict(input_img, verbose=0)[0]
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction) * 100

            label = f"{predicted_class.upper()} ({confidence:.1f}%)"
            cv2.rectangle(frame, (x_min, y_min-30), (x_max, y_min), (0, 255, 0), -1)
            cv2.putText(frame, label, (x_min + 5, y_min - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            moves.append(predicted_class)
            positions.append((x_min, y_min - 40))

    # Kalau 2 tangan muncul
    if len(moves) == 2:
        winner = decide_winner(moves[0], moves[1])
        result_text = f"{moves[0].upper()} vs {moves[1].upper()} — {winner}"
        cv2.putText(frame, result_text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    cv2.imshow("🎮 RPS Game", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
