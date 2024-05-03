import cv2
import numpy as np

# Abrir el video desde la ubicación especificada
video = cv2.VideoCapture('C:/Users/gloperma/Downloads/prueba.mp4')

# Inicializar variables de conteo y estado de liberación
contador = 0
liberado = False

# Bucle principal para procesar cada fotograma del video
while True:
    # Leer el siguiente fotograma del video
    ret, img = video.read()
    # Salir del bucle si no se puede leer más fotogramas
    if not ret:
        break  # Salir del bucle si no se puede leer el fotograma

    # Redimensionar la imagen a un tamaño específico
    img = cv2.resize(img, (640, 480))
    # Convertir la imagen a escala de grises
    imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Aplicar umbralización adaptativa para obtener una imagen binaria
    imgTh = cv2.adaptiveThreshold(imgGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 12)
    # Dilatar la imagen binaria para mejorar la detección de bordes
    kernel = np.ones((8, 8), np.uint8)
    imgDil = cv2.dilate(imgTh, kernel, iterations=2)

    # Definir la región de interés
    x, y, w, h = 390, 100, 30, 150
    recorte = imgDil[y:y+h, x:x+w]
    # Contar el número de píxeles blancos en la región de interés
    brancos = cv2.countNonZero(recorte)

    # Actualizar el contador si se detecta un número suficiente de píxeles blancos y se cumple una condición de liberación
    if brancos > 4000 and liberado:
        contador += 1
    # Actualizar el estado de liberación basado en el número de píxeles blancos
    if brancos < 4000:
        liberado = True
    else:
        liberado = False

    # Dibujar un rectángulo alrededor de la región de interés en el fotograma original
    if not liberado:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
    else:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 4)

    # Dibujar un rectángulo en la imagen binaria
    cv2.rectangle(imgTh, (x, y), (x + w, y + h), (255, 255, 255), 6)

    # Mostrar el número de píxeles blancos y el contador en el fotograma
    cv2.putText(img, str(brancos), (x - 30, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    cv2.rectangle(img, (575, 155), (575 + 88, 155 + 85), (255, 255, 255), -1)
    cv2.putText(img, str(contador), (x + 100, y), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 3)

    # Mostrar el fotograma procesado
    cv2.imshow('video original', img)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# Liberar el recurso del objeto de video y cerrar todas las ventanas
video.release()
cv2.destroyAllWindows()
