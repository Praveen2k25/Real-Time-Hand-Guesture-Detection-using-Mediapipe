import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder


mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

mp_draw = mp.solutions.drawing_utils


model = load_model('Real-Time-Hand-Guesture-Detection-using-Mediapipe\models.h5')


le = LabelEncoder()
le.classes_ = np.load('Real-Time-Hand-Guesture-Detection-using-Mediapipe\classes.npy', allow_pickle=True)
print("Class labels:", le.classes_)  


cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
          
            landmark_list = []
            for lm in hand_landmarks.landmark:
                cx, cy = lm.x, lm.y  
                landmark_list.append([cx, cy, lm.z])  

            
            landmark_flattened = np.array(landmark_list).flatten()

            
            if landmark_flattened.size == 63: 
                landmark_flattened = landmark_flattened.reshape(1, 21, 3) 

                
                prediction = model.predict(landmark_flattened)
                print("Raw prediction probabilities:", prediction)  
                gesture_index = np.argmax(prediction)  
                gesture_name = le.inverse_transform([gesture_index])[0]  

              
                cv2.putText(img, gesture_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                print("Predicted gesture index:", gesture_index)
                print("Predicted gesture name:", gesture_name) 

           
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

   
    cv2.imshow("Gesture Recognition", img)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
