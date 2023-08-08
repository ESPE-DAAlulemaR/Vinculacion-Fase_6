import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import os

# Comprobar si el archivo "historial.txt" existe y borrarlo si es necesario
if os.path.exists("historial.txt"):
    os.remove("historial.txt")

# Configuración de la captura de video
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 640)
cap.set(4, 480)

# Inicialización de detectores y clasificadores
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# Parámetros de clasificación

offset = 20
imgSize = 300

# Etiquetas para clasificación
labels = [
    "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "CH", "D", "E", "F", "G", "H", "I", "K", "L", "M" , "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"
]

# Variable para almacenar la última letra detectada
last_detected_letter = None

while True:
    # Captura el siguiente cuadro de video
    success, img = cap.read()

    # Encuentra manos en el cuadro de video
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Recorta y redimensiona la región de interés (ROI) de la mano
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Calcular el aspect ratio
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Realiza la clasificación de gesto
        prediction, index = classifier.getPrediction(imgWhite, draw=False)

        # Obtén la letra actual
        current_letter = labels[index]

        # Mostrar resultados en la imagen
        cv2.rectangle(img, (x - offset, y - offset-50), (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
        cv2.putText(img, current_letter, (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(img, (x-offset, y-offset), (x + w+offset, y + h+offset), (255, 0, 255), 4)

        # Guardar la letra actual en el archivo "historial.txt" si es diferente de la última letra detectada
        if current_letter != last_detected_letter:
            with open("historial.txt", "a") as file:
                file.write(current_letter + "\n")
            last_detected_letter = current_letter

    # Mostrar la imagen resultante en la ventana
    cv2.imshow("Image", img)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
