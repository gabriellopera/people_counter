import cv2
import numpy as np

# Abrir el video desde la ubicación especificada
video = cv2.VideoCapture('C:/Users/gloperma/Downloads/prueba.mp4')

# Crear el objeto para la sustracción de fondo
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100, detectShadows=True)

# Inicializar variables de seguimiento y conteo
object_centroids = {}  # Diccionario para almacenar los centroides de los objetos detectados
object_id = 0  # ID del objeto
crossed_objects = set()  # Conjunto de objetos que han cruzado la región de interés
rect_color = (255, 0, 0)  # Color inicial del rectángulo (azul)
crossing = False  # Indicador de cruce de persona

# Definir una función para calcular la distancia euclidiana
def euclidean_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

# Definir la región de interés (rectángulo)
roi_x, roi_y, roi_w, roi_h = 390, 100, 30, 150  # Ajustar estas coordenadas según sea necesario

# Bucle principal para procesar cada fotograma del video
while True:
    # Leer el siguiente fotograma del video
    ret, img = video.read()
    # Salir del bucle si no se puede leer más fotogramas
    if not ret:
        break  # Salir del bucle si no se puede leer el fotograma

    # Redimensionar la imagen a un tamaño específico
    img = cv2.resize(img, (640, 480))   

    # Aplicar sustracción de fondo para obtener la máscara de movimiento
    fgmask = fgbg.apply(img)
    
    # Eliminar ruido y sombras de la máscara de movimiento
    kernel = np.ones((5, 5), np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.dilate(fgmask, kernel, iterations=2)

    # Encontrar contornos en la máscara de movimiento
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Lista para almacenar los centroides de los objetos detectados
    detected_centroids = []
    # Dibujar contornos en el fotograma original
    for contour in contours:
        # Filtrar contornos pequeños para evitar falsos positivos
        if cv2.contourArea(contour) > 1000:  # Area para evitar contornos muy pequeños
            x_cont, y_cont, w_cont, h_cont = cv2.boundingRect(contour)
            if 30 < w_cont < 300 and 30 < h_cont < 300:  #Condicion para que los contornos se considere que pertenecen a una persona 
                cv2.rectangle(img, (x_cont, y_cont), (x_cont + w_cont, y_cont + h_cont), (0, 255, 0), 2)
                centroid = (int(x_cont + w_cont / 2), int(y_cont + h_cont / 2))
                detected_centroids.append(centroid)

    # Dibujar el rectángulo de la región de interés
    cv2.rectangle(img, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), rect_color, 4)

    # Actualizar el seguimiento de centroides
    new_object_centroids = {}
    for centroid in detected_centroids:
        found = False
        for obj_id, previous_centroid in object_centroids.items():
            if euclidean_distance(centroid, previous_centroid) < 50:
                new_object_centroids[obj_id] = centroid
                found = True
                break
        if not found:
            new_object_centroids[object_id] = centroid
            object_id += 1

    object_centroids = new_object_centroids

    # Verificar el cruce del rectángulo
    for obj_id, centroid in object_centroids.items():
        if (roi_x < centroid[0] < roi_x + roi_w) and (roi_y < centroid[1] < roi_y + roi_h):
            if obj_id not in crossed_objects:
                crossed_objects.add(obj_id)
                crossing = True
        else:
            if obj_id in crossed_objects:
                crossing = False

    # Actualizar el contador de personas cruzadas
    total_crossed = len(crossed_objects)

    # Cambiar el color del rectángulo según el estado de cruce
    if crossing:
        rect_color = (0, 255, 0)  # Cambiar a verde si alguien está cruzando
    else:
        rect_color = (255, 0, 0)  # Volver a azul si nadie está cruzando

    # Mostrar el contador de personas en el fotograma
    cv2.putText(img, f'Personas: {total_crossed}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar el fotograma procesado y la máscara de fondo
    cv2.imshow('video original', img)
    cv2.imshow('Foreground Mask', fgmask)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# Liberar el recurso del objeto de video y cerrar todas las ventanas
video.release()
cv2.destroyAllWindows()
