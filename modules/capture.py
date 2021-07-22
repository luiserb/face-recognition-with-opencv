"""

    Módulo para capturar imagenes

"""


import os # volvemos a importar la librería os para crear carpetas en nuestro OS

import cv2 # importamos la librería de OpenCV
import imutils # importaremos imutils para configurar las propiedades de la ventana


# creamos la función para capturar imagenes con los parámetros de nombre de usuario y la ruta de nuestro scripts
def captura_de_imagenes(nombre, ruta):
    # creamos una ruta para cada persona
    carpeta_de_usuario = ruta + '/database/' + nombre 

    # Si el usuario no existe crearemos una carpeta con sus imagenes, caso contrario omitirá este paso
    if not os.path.exists(carpeta_de_usuario): 
        os.makedirs(carpeta_de_usuario)
    
    # creamos la variable para usar nuestra camara
    camara = cv2.VideoCapture(0)

    # llamamos al modelo por default de OpenCV para pixelear el rostro
    # descargalo desde https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
    # y guardalo en la carpeta modulos con el nombre: haarcascade_frontalface_default.xml
    modelo_default = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # definimos un contador
    contador = 0

    # creamos un bucle para procesar la toma de fotos
    while True:

        # definimos las propiedades de la ventana de captura y color de fondo(opcional)
        camara_activa, ventana = camara.read()

        # si no hay una  webcam disponible no habrá captura
        if camara_activa == False:
            break
        
        # configuraremos el ancho de nuestra ventana
        ventana = imutils.resize(ventana, width=800)

        # definimos el color gris en nuestra ventana 
        color_gris = cv2.cvtColor(ventana, cv2.COLOR_BGR2GRAY)

        # definimos el marcador de rostro para nuestra ventana
        marcador_de_rostro = ventana.copy()

        # definimos el detector de rostros
        rostros = modelo_default.detectMultiScale(color_gris, scaleFactor=1.1, minNeighbors=5)

        # añadiremos un for para cada imagen 
        for (x, y, w, h) in rostros:

            # definimos el tipo de marcador, ubicación, y el color
            cv2.rectangle(ventana, (x,y), (x+w, y+h), (0, 255, 0), 2)
            cuadro = marcador_de_rostro[y:y+h, x:x+w]

            #definimos las propiedades de la imagen y la interpolación
            foto = cv2.resize(cuadro, (240, 240), interpolation=cv2.INTER_CUBIC)
            
            #definimos el formato de imagen y donde se guardará
            cv2.imwrite(carpeta_de_usuario + "/" + f'{nombre}_{contador}.jpeg', foto)
            
            #sumamos 1 a cada vuelta que dá el bucle
            contador = contador + 1
            
        # introduciremos las propiedades de la ventana
        cv2.imshow(f'Capturando a {nombre}', ventana)
        
        # creamos la variable de tiempo de espera que mostrará un marco pequeño durante 1 ms, después la pantalla se cerrará automáticamente.
        tiempo_de_espera = cv2.waitKey(1)

        # realizaremos la condicional de imagenes, mientras mas número de imagenes será más preciso a la hora de reconocer rostros
        if tiempo_de_espera == 30 or contador >= 201:
            break

    # luego de finalizar el bucle, destruiremos todas las ventanas
    camara.release()
    cv2.destroyAllWindows()