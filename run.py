# -*- coding: utf-8 -*-

#importamos la librería encargada de interactuar con el OS
import os

#importamos las funciones previamente escritas
from modules.capture import captura_de_imagenes
from modules.create_model import crear_modelo
from modules.face_recognizer import reconocimiento_facial

#creamos la variable que retorna nuestra carpeta en nuestro sistema
directiorio_local = os.path.dirname(os.path.abspath(__file__))

# creamos una lista para guardar cada usuario que serán agregados
usuarios = []

# creamos un input y la almacenaremos en una variable para luego agregarala a nuestra lista de usuarios
usuario = input('Agregar nombre de usuario: ')
usuarios.append(usuario.capitalize())
print('Primer usuario agregado: ', usuarios)

# se ejecutará la funcíon de captura de datos al usuario previamente ingresado
captura_de_imagenes(usuario.capitalize(), directiorio_local)

# creamos un bucle para crear varios usuarios si es requerido
while True:
    print('¿Deseas seguir agregando usuario?')
    usuario_secundarios = input("Responde 'y' para decir sí, y 'n' para decir no: ")
    if usuario_secundarios == 'y':
        otro_usuario = input('Agrega otro nombre de usuario: ')
        usuarios.append(otro_usuario.capitalize())
        captura_de_imagenes(otro_usuario.capitalize(), directiorio_local)
        print('Usuarios registrados hasta ahora: ', usuarios)
    elif usuario_secundarios == 'n':
        break
    else:
        print('Respuesta inválida: responde y o n.')


# al finalizar el bucle se procederá a crear el modelo .xml necesario
crear_modelo(directiorio_local)

# se ejucturá la función de reconocimiento que ejecutará el algoritmo de reconocimiento de OpenCV
reconocimiento_facial(directiorio_local)