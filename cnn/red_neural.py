""" modelo de red neural convolcional

Modelo de red neuronal basico uno.
modelo convulucional.

"""
import os
import cnn
import numpy as np
import matplotlib.pyplot as plt
from cnn.gen_generator import build_generador
from tensorflow.keras.layers import (Input, Dense, Activation, ZeroPadding2D,
                                     BatchNormalization, Flatten, Conv2D)
from tensorflow.keras.layers import (AveragePooling2D, MaxPooling2D, #noqa
                                     Dropout, GlobalMaxPooling2D, #noqa
                                     GlobalAveragePooling2D) #noqa
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau)


dicact = os.path.split(os.path.dirname(cnn.__file__))[0]
savepath = os.path.join(dicact, 'saved_models')
DEFAULT_NAME = 'alexnet'


def model_build(img_resolucion, cfg, red_name=DEFAULT_NAME, save_model=True):
    """ Construccion del modelo


    """
    # input
    x_in = Input(img_resolucion)
    x = ZeroPadding2D(cfg['padding'])(x_in)

    # convolucional 0
    x = Conv2D(cfg['kernel_num'], cfg['kernel_sze'], strides=cfg['stride'],
               name='convolucion_0')(x)
    x = BatchNormalization(axis=cfg['norm_ejes'], name='batch_normal_0')(x)
    x = Activation(cfg['l1_act'])(x)
    x = MaxPooling2D(cfg['maxpool_sze'], name='max_pooling_0')(x)

    # convolucional 1
    x = Conv2D(cfg['kernel_num'], cfg['kernel_sze'], strides=cfg['stride'],
               name='convolucion_1')(x)
    x = BatchNormalization(axis=cfg['norm_ejes'], name='batch_normal_1')(x)
    x = Activation(cfg['l1_act'])(x)
    x = MaxPooling2D(cfg['maxpool_sze'], name='max_pooling_1')(x)

    # convolucion 2
    x = Conv2D(64, cfg['kernel_sze'], strides=cfg['stride'],
               name='convolucion_2')(x)
    x = BatchNormalization(axis=cfg['norm_ejes'], name='batch_normal_2')(x)
    x = Activation(cfg['l1_act'])(x)
    x = MaxPooling2D(cfg['maxpool_sze'], name='max_pooling_2')(x)

    # Flatten
    x = Flatten()(x)
    # Fullyconected 0
    x = Dense(64, activation=cfg['fc_act'], name='fully_conected_0')(x)
    x = Dropout(0.5)(x)

    # Fullyconected 1
    x = Dense(1, activation=cfg['fc_act'], name='fully_conected_1')(x)

    model_nc = Model(inputs=x_in, outputs=x, name='red_uno')

    # compilación del modelo
    model_cp = model_nc
    model_cp.compile(cfg['optimizer'], cfg['lost_fn'],
                     metrics=cfg['metrica'])
    # Graficando el modelo
    os.makedirs(savepath, exist_ok=True)
    print('Generando imagen del modelo')
    plot_model(model_cp, to_file=os.path.join(savepath, red_name + '.png'))
    if save_model:
        model_cp.save(os.path.join(savepath, red_name + '.h5'))
    return model_cp, model_nc


def train_model(modelo, cfg_train, red_name=DEFAULT_NAME, save_weights=True):
    """Funcion de entrenamiento

    Funcion para entrenar el modelo
    """
    # Creacion de los generadores de data
    trg_size = (cfg_train['IMG_SZE'][0], cfg_train['IMG_SZE'][1])
    train_generador, _ = build_generador(cfg_train['TRAIN_DATA_PATH'],
                                         batch_sze=cfg_train['BATCH_SIZE'],
                                         trgt_sze=trg_size,
                                         color=cfg_train['IMG_COLOR'])
    test_generador, _ = build_generador(cfg_train['TEST_DATA_PATH'],
                                        batch_sze=cfg_train['BATCH_SIZE'],
                                        trgt_sze=trg_size,
                                        color=cfg_train['IMG_COLOR'])
    # # Funciones de callback
    # # Detiene el entrenamiento si no mejora (EarlyStopping),
    # # reduce indice de aprendizaje si no mejora (ReduceLROnPlateau),
    # # https://enmilocalfunciona.io/tratamiento-de-imagenes-usando-imagedatagenerator-en-keras/  #noqa
    early_stop = EarlyStopping(monitor='val_loss', patience=20,
                               verbose=1, min_delta=1e-4)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4,
                                  verbose=1, min_delta=1e-4)
    callbacks_list = [early_stop, reduce_lr]
    nb_test_samples = len(test_generador)
    pasos_fit = (nb_test_samples // cfg_train['BATCH_SIZE'])
    model_history = modelo.fit_generator(train_generador,
                                         epochs=cfg_train['EPOCAS'],
                                         validation_data=test_generador,
                                         validation_steps=pasos_fit,
                                         callbacks=callbacks_list)
    if save_weights:
        savepath = os.path.join(dicact, 'saved_models')
        os.makedirs(savepath, exist_ok=True)
        modelo.save_weights(os.path.join(savepath, red_name + '_pesos.h5'))
    return model_history, modelo


def resultados_graficos(model_history, red_name=DEFAULT_NAME):
    plt.figure(0)
    plt.plot(model_history.history['acc'], 'r')
    plt.plot(model_history.history['val_acc'], 'g')
    plt.xticks(np.arange(0, 20, 1.0))
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.xlabel("Num of Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy vs Validation Accuracy")
    plt.legend(['train', 'validation'])
    plt.savefig(os.path.join(savepath, red_name + '_TaccVacc.png'))

    plt.figure(1)
    plt.plot(model_history.history['loss'], 'r')
    plt.plot(model_history.history['val_loss'], 'g')
    plt.xticks(np.arange(0, 20, 1.0))
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.xlabel("Num of Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Validation Loss")
    plt.legend(['train', 'validation'])
    plt.savefig(os.path.join(savepath, red_name + '_TlossVloss.png'))

    plt.figure(2)
    plt.plot(model_history.history['mse'], 'r')
    plt.plot(model_history.history['val_mse'], 'g')
    plt.xticks(np.arange(0, 20, 1.0))
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.xlabel("Num of Epochs")
    plt.ylabel("MSE")
    plt.title("Training Loss vs Validation Loss")
    plt.legend(['train', 'validation'])
    plt.savefig(os.path.join(savepath, red_name + '_mseTlossVloss.png'))
    # plt.show()


if __name__ == "__main__":
    pass
