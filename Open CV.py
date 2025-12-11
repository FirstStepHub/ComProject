import cv2
import mediapipe as mp
import time
from pyfirmata2 import Arduino

board = Arduino('COM5')

LED_PINS = [13, 12, 11, 10, 9] 

mp_hand = mp.solutions.hands
hands = mp_hand.Hands(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5,
    max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

tip_ids = [4, 8, 12, 16, 20] 

def control_leds(count):
    
    for i in range(len(LED_PINS)):
        pin = LED_PINS[i]
        
        if i < count:
            board.digital[pin].write(1)
        else:
            board.digital[pin].write(0)

def NumFinger(lm_list):
    
    if not lm_list: 
        return 0

    fingers = []
    
    for id in range(1, 5): 
        if lm_list[tip_ids[id]][2] < lm_list[tip_ids[id] - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)

    if lm_list[tip_ids[0]][1] < lm_list[tip_ids[0] - 1][1]:
        fingers.insert(0, 1)
    else:
        fingers.insert(0, 0)

    return sum(fingers)

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.flip(img, 1)
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    lm_list = []
    current_count = 0 

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hand.HAND_CONNECTIONS)
            
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
            
            current_count = NumFinger(lm_list)
            
            control_leds(current_count)
            
    cv2.putText(
    img,                        # ภาพที่จะเขียน
    f"Number: {current_count}",         # ข้อความ
    (50, 50),                   # ตำแหน่ง x, y
    cv2.FONT_HERSHEY_SIMPLEX,   # ฟร้อนตัวอักษร
    1,                          # ขนาดตัวอักษร (scale)
    (0, 0, 255),                # สี (BGR)
    2                           # ความหนาของตัวอักษร
    )

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for pin in LED_PINS:
    board.digital[pin].write(0)

cap.release()
cv2.destroyAllWindows()