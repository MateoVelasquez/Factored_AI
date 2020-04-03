"""Preprocesar imagenes

Modulo para realizar un preprocesamiento de la imagen.
"""
import os
import math
import pathlib
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt


# Parámetros de carga
BATCH_SIZE = 32
IMG_H = 480
IMG_W = 640
DATA_DIR = r'C:\Users\MATEO\RepositoriosGIT\Factored_AI\img\train'

# STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)


def build_generador(data_dir, batch_sze=BATCH_SIZE,
                    trgt_sze=(IMG_H, IMG_W), color='rgb'):
    """funcion para crear generador

    Funcion para crear el generador de entrenamiento a partir de
    una carpeta de imagenes dada.
    """
    # Deteccion de las imagenes según la ruta
    data_dir = pathlib.Path(data_dir)
    CLASS_NAMES = np.array([item.name for item in data_dir.glob('*')])
    print(data_dir)
    image_count = len(list(data_dir.glob('*/*.jpeg')))
    print(f'numero de imagenes en {os.path.basename(data_dir)}: {image_count}')
    print(f'clases encontradas: {CLASS_NAMES}')

    # creando el generador
    img_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                           rotation_range=40,
                                                           width_shift_range=0.2,
                                                           height_shift_range=0.2,
                                                           shear_range=0.2,
                                                           zoom_range=0.2)
    train_data_gen = img_gen.flow_from_directory(directory=str(data_dir),
                                                 batch_size=batch_sze,
                                                 shuffle=True,
                                                 target_size=trgt_sze,
                                                 color_mode=color,
                                                 classes=list(CLASS_NAMES),
                                                 class_mode='binary')
    return train_data_gen, CLASS_NAMES


# Inspeccionando el lote de imagenes
def show_batch(image_batch, label_batch, class_names):
    plt.figure(figsize=(10, 10))
    num_img = len(image_batch)
    for n in range(num_img):
        plt.subplot(math.ceil(math.sqrt(num_img)),
                    math.ceil(math.sqrt(num_img)), n+1)
        plt.imshow(image_batch[n])
        plt.title(class_names[int(label_batch[n])])
        # plt.title(class_names[label_batch[n] == 1][0].title())
        plt.axis('off')
    plt.draw()
    plt.show()
    return


if __name__ == "__main__":
    # # Previsualizacion
    generador, class_names = build_generador(DATA_DIR)
    image_batch, label_batch = next(generador)
    show_batch(image_batch, label_batch, class_names)
    # # # Cargando datos con keras
    # list_ds = data.Dataset.list_files(str(pathlib.Path(DATA_DIR)/'*/*'))
    # for f in list_ds.take(5):
    #     print(f.numpy())
