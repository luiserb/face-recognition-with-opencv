""" 

    Ejecutaremos el algoritmo de OpenCV para el reconocimiento

"""

# importaremos nuevamente las librerias
import os
import cv2
import imutils


# definimos la función
def reconocimiento_facial(ruta):
    # guardamos en una lista todos los nombre de cada carpeta de usuario
    nombre_de_usuario = os.listdir(ruta + '/database')
    
    # iniciamos el algoritmo LPHB para leer nuestro modulo previamente creado 
    modelo = cv2.face.LBPHFaceRecognizer_create(
        radius=1, # El radio es utilizado para construir el patrón binario local circular. Cuanto mayor sea el radio, más suave será la imagen pero más información espacial puede obtener.
        neighbors=8, # El número de puntos de muestra para construir un patrón binario local circular. Un valor apropiado es utilizar 8 puntos de muestra.
        grid_x=8, # El número de celdas en la dirección horizontal
        grid_y=8 # El número de celdas en la dirección vertical
    )
    
    modelo.read('MisModelosFaciales.xml')
    
    # definimos nuestro medio de captura
    camara = cv2.VideoCapture(0)

    # volvemos a llamar el archivo .xml por default de OpenCV previamente descargado
    modelo_default = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    print('Haz click en esta ventana y presiona CTRL + C para detener el script')
    
    # inicializamos el bucle
    while True:

        # introducimos un try para manejar errores
        try:
            # volvemos a definir las propiedades de nuestra ventana
            camara_activa, ventana = camara.read()
            if camara_activa == False: 
                break

            ventana = imutils.resize(ventana, width=800)
            
            # definimos otra vez la variable de color y el marco de reconocimiento
            color_gris = cv2.cvtColor(ventana, cv2.COLOR_BGR2GRAY)
            aux_ventana = color_gris.copy()
            
            # definimos el detector de rostros
            rostros = modelo_default.detectMultiScale(color_gris, scaleFactor=1.1, minNeighbors=5)

            # realizamos un for para cada rostro detectado en la variable rostros
            for (x,y,w,h) in rostros:
                marco_detector = aux_ventana[y:y+h,x:x+w]
                marco_detector = cv2.resize(marco_detector, (320, 320), interpolation=cv2.INTER_CUBIC)
                # usamos la función predict para evaluar el valor de confianza
                resultados = modelo.predict(marco_detector)
                
                # agregaremos una condicional con un nivel de confianza promedio y la forma geométrica del detector
                if resultados[1] < 65:
                    cv2.putText(ventana,'{}'.format(nombre_de_usuario[resultados[0]]),(x,y-25) ,2,1.1, (0,255,0), 1, cv2.LINE_AA)
                    cv2.rectangle(ventana, (x,y),(x+w,y+h),(0,255,0),2)
                # si el nivel de confiaza es mayor a 70, lo tomaremos por desconocido
                else:
                    cv2.putText(ventana,'Desconocido', (x,y-20), 2,0.8, (0,0,255),  1, cv2.LINE_AA)
                    cv2.rectangle(ventana, (x,y),(x+w,y+h),(0,0,255),2)

            # agregamos las propiedades a la ventana
            cv2.imshow('Reconocimiento facial con OpenCV',ventana)
            
            # definimos un tiempo de 1ms para actualizar
            tiempo_de_espera = cv2.waitKey(1)
            if tiempo_de_espera == 27:
                break
        
        # silenciamos el error KeyboardInterrupt al cerrar la ventana con CTRL + C
        except KeyboardInterrupt:
            # cerramos todas las ventanas
            camara.release()
            cv2.destroyAllWindows()
            print('By luiserb')
            break
