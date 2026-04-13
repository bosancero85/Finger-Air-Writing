import cv2
import numpy as np
import sys

# Robuster MediaPipe Import
try:
    import mediapipe as mp
    from mediapipe.python.solutions import hands as mp_hands
    from mediapipe.python.solutions import drawing_utils as mp_draw
    print("MediaPipe erfolgreich geladen!")
except ImportError:
    try:
        import mediapipe.solutions.hands as mp_hands
        import mediapipe.solutions.drawing_utils as mp_draw
        print("MediaPipe über alternativen Pfad geladen!")
    except Exception as e:
        print(f"Fehler beim Laden von MediaPipe: {e}")
        sys.exit()

# Initialisierung
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Kamera-Setup
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Leinwand (Canvas)
canvas = np.zeros((720, 1280, 3), np.uint8)
px, py = 0, 0
color = (0, 255, 0) # Wir nehmen mal Grün

print("Programm läuft. Drücke 'q' zum Beenden.")

while True:
    success, img = cap.read()
    if not success: break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            # Landmarks für Zeigefinger (8) und Mittelfinger (12)
            h, w, _ = img.shape
            x1, y1 = int(hand_lms.landmark[8].x * w), int(hand_lms.landmark[8].y * h)
            x2, y2 = int(hand_lms.landmark[12].x * w), int(hand_lms.landmark[12].y * h)

            # Check: Welche Finger sind oben?
            index_up = hand_lms.landmark[8].y < hand_lms.landmark[6].y
            middle_up = hand_lms.landmark[12].y < hand_lms.landmark[10].y

            # Modus 1: Auswahl/Pause (Zwei Finger oben)
            if index_up and middle_up:
                px, py = 0, 0
                cv2.circle(img, (x1, y1), 15, (255, 255, 255), cv2.FILLED)
                cv2.putText(img, "Pause / Auswahl", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Modus 2: Zeichnen (Nur Zeigefinger oben)
            elif index_up:
                if px == 0 and py == 0: px, py = x1, y1
                cv2.line(canvas, (px, py), (x1, y1), color, 10)
                px, py = x1, y1
            else:
                px, py = 0, 0

    # Canvas und Kamera-Bild kombinieren
    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, canvas)

    cv2.imshow("Air Writing", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()