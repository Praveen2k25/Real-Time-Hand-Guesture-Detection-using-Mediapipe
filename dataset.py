import cv2
import mediapipe as mp
import numpy as np
import os


DATASET_PATH = 'gesture_landmark_dataset'
GESTURE_NAME = 'Hand-open'  
NUM_SAMPLES = 1000  
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils


gesture_folder = os.path.join(DATASET_PATH, GESTURE_NAME)
os.makedirs(gesture_folder, exist_ok=True)

count = 0

while count < NUM_SAMPLES:
    _, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
        
            landmark_list = []
            for lm in hand_landmarks.landmark:
                landmark_list.append([lm.x, lm.y, lm.z])  
            
            np.savetxt(f'{gesture_folder}/{count}.csv', landmark_list, delimiter=',')
            count += 1
            print(f"Saved sample {count}/{NUM_SAMPLES}")
            if count >= NUM_SAMPLES:
                break

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Capture Gesture", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
