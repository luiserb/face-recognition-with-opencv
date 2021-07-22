""" 

    Entrenaremos el algoritmo de reconocimiento con los datos obtenidos en la carpeta database
    mediante el algoritmo LBPH 

"""

import os

import cv2
import numpy as np #importaremos la librería numpy para calculos avanzados


# definiremos la funcion con el parámetro de la variable de nuestro directorio
def crear_modelo(ruta):
    # guardaremos en una variable una lista retornando todas las carpeta creadas hasta el momento para luego imprimirlas
    lista_de_datos = os.listdir(ruta + '/database')
    print('Datos obtenidos de:', lista_de_datos)

    nombre_de_usuario = [] # definimos una lista para guardar cada nombre
    lotes = [] # definimos una lista para guardar el número de cada imagen
    contador = 0 # definimos un contador de inicio igual a 0

    # realizamos un for para cada carpeta en nuestra carpeta de datos
    for directorio in lista_de_datos:
        usuario = ruta + '/database/' + directorio + '/'
        print('Codificando...' )

        # realizamos otro for para cada usuario
        for imagen in os.listdir(usuario):
            lotes.append(contador) # agregamos a la lista lotes el número índice de la imagen
            nombre_de_usuario.append(cv2.imread(usuario + '/' + imagen, 0)) # agregamos cada imagen a la lista de nombre de usuario
            recuadro = cv2.imread(usuario + '/' + imagen, 0) # definimos la variable que leerá cada imagen, el 0 corresponde al escalas de grises
            cv2.imshow('Procesando', recuadro) # añadimos la propiedad de la ventana emergente
            cv2.waitKey(35) # definimos el tiempo de espera por cada usuario

        contador = contador + 1 # por cada vuelta sumaremos 1 al contador


    print('Entrenando algoritmo...')
    # definimos el algoritmo de reconocimiento, en este caso será LBPH, y toma los siguiente argumentos
    reconocedor_facial = cv2.face.LBPHFaceRecognizer_create(
        radius=1, # El radio es utilizado para construir el patrón binario local circular. Cuanto mayor sea el radio, más suave será la imagen pero más información espacial puede obtener.
        neighbors=8, # El número de puntos de muestra para construir un patrón binario local circular. Un valor apropiado es utilizar 8 puntos de muestra.
        grid_x=8, # El número de celdas en la dirección horizontal
        grid_y=8 # El número de celdas en la dirección vertical
    )
    reconocedor_facial.train(nombre_de_usuario, np.array(lotes)) # entrenaremos el reconocedor facial mediante el nombre de cada usuario y cálculos de array
    reconocedor_facial.write('MisModelosFaciales.xml') # guardamos nuestro modelo con la extensión .xml
    print('Algoritmo entrenado')
    cv2.destroyAllWindows() # cerramos la ventana