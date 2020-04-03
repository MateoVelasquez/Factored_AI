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
from tensorflow.keras import optimizers as opti #noqa
from tensorflow import optimizers as tfopti #noqa
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau)


dicact = os.path.split(os.path.dirname(cnn.__file__))[0]
savepath = os.path.join(dicact, 'saved_models')
DEFAULT_NAME = 'vggnet'


def model_paper(img_resolucion, cfg, red_name='paper', save_model=True):
    x_in = Input(img_resolucion)

    kernelnum = [32, 64, 128, 128]
    fc = [512, 1]

    kernelsze = (3, 3)
    x = Conv2D(kernelnum[0], kernelsze, name='conv3_01', activation='relu')(x_in)
    x = MaxPooling2D(pool_size=(2, 2), name='max_pooling_0')(x)

    x = Conv2D(kernelnum[1], kernelsze, name='conv3_11', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='max_pooling_1')(x)

    x = Conv2D(kernelnum[2], kernelsze, name='conv3_21', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='max_pooling_2')(x)

    x = Conv2D(kernelnum[3], kernelsze, name='conv3_31', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='max_pooling_3')(x)

    x = Flatten()(x)
    x = Dropout(0.6)(x)
    x = Dense(fc[0], activation='relu', name='fc_1')(x)
    x = Dense(fc[1], activation='sigmoid', name='out')(x)

    # Compilacion:
    model_nc = Model(inputs=x_in, outputs=x, name='paper')
    # compilación del modelo
    model_cp = model_nc
    opt = opti.SGD(lr=0.006, decay=1e-6, momentum=0.9, nesterov=True)
    #opt = tfopti.Adam(learning_rate=0.0001)
    model_cp.compile(loss='binary_crossentropy', optimizer=opt,
                     metrics=cfg['metrica'])
    # Graficando el modelo
    os.makedirs(savepath, exist_ok=True)
    print('Generando imagen del modelo')
    plot_model(model_cp, to_file=os.path.join(savepath, red_name + '.png'))
    if save_model:
        model_cp.save(os.path.join(savepath, red_name + '.h5'))
    return model_cp, model_nc


def mdelvgg_build(img_resolucion, cfg, red_name=DEFAULT_NAME, save_model=True):
    """ Construccion del modelo


    """
    ke_vec = [4, 8, 16, 32]
    fc_vec = [256, 256]
    # input
    x_in = Input(img_resolucion)
    # bloque 0
    kernelnum = ke_vec[0]
    kernelsze = (3, 3)
    x = Conv2D(kernelnum, kernelsze, strides=cfg['stride'], name='conv3_01',
               padding='same', activation='relu')(x_in)
    x = Conv2D(kernelnum, kernelsze, strides=cfg['stride'], name='conv3_02',
               padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='max_pooling_0', strides=(2, 2))(x)

    # bloque 1
    kernelnum = ke_vec[1]
    kernelsze = (3, 3)
    x = Conv2D(kernelnum, kernelsze, strides=cfg['stride'], name='conv3_11',
               padding='same', activation='relu')(x_in)
    x = Conv2D(kernelnum, kernelsze, strides=cfg['stride'], name='conv3_12',
               padding='same', activation='relu')(x)
    x = AveragePooling2D(pool_size=(2, 2), name='Avg_pol_1', strides=(2, 2))(x)

    # bloque 2
    kernelnum = ke_vec[2]
    kernelsze = (3, 3)
    x = Conv2D(kernelnum, kernelsze, strides=cfg['stride'], name='conv3_21',
               padding='same', activation='relu')(x_in)
    x = Conv2D(kernelnum, kernelsze, strides=cfg['stride'], name='conv3_22',
               padding='same', activation='relu')(x)
    x = AveragePooling2D(pool_size=(2, 2), name='Avg_pol_2', strides=(2, 2))(x)

    # # Bloque 3
    kernelnum = ke_vec[3]
    kernelsze = (3, 3)
    x = Conv2D(kernelnum, kernelsze, strides=cfg['stride'], name='conv3_31',
               padding='same', activation='relu')(x_in)
    x = Conv2D(kernelnum, kernelsze, strides=cfg['stride'], name='conv3_32',
               padding='same', activation='relu')(x)
    x = Conv2D(kernelnum, kernelsze, strides=cfg['stride'], name='conv3_33',
               padding='same', activation='relu')(x)
    x = AveragePooling2D(pool_size=(2, 2), name='Avg_pol_3', strides=(2, 2))(x)

    # # bloque 3
    # kernelnum = int(256*preescaler)
    # kernelsze = (3, 3)
    # x = ZeroPadding2D(cfg['padding'])(x)
    # x = Conv2D(kernelnum, kernelsze, strides=cfg['stride'],
    #            name='conv3_31')(x)
    # x = Conv2D(kernelnum, kernelsze, strides=cfg['stride'],
    #            name='conv3_32')(x)
    # # x = BatchNormalization(axis=cfg['norm_ejes'], name='batch_normal_3')(x)
    # x = Activation(cfg['l1_act'])(x)
    # x = MaxPooling2D(cfg['maxpool_sze'], name='max_pooling_3')(x)

    # Flatten
    x = Flatten()(x)
    # Fullyconected 0
    x = Dense(fc_vec[0], activation=cfg['l1_act'], name='fully_conected_0')(x)
    x = Dropout(0.5)(x)
    # Fullyconected 1
    x = Dense(fc_vec[1], activation=cfg['l1_act'], name='fully_conected_1')(x)
    x = Dropout(0.2)(x)
    # output0
    x = Dense(2, activation=cfg['fc_act'], name='Salida')(x)

    # Compilacion:
    model_nc = Model(inputs=x_in, outputs=x, name='red_uno')
    # compilación del modelo
    model_cp = model_nc
    opt = opti.SGD(lr=0.08, decay=1e-6, momentum=0.9, nesterov=True)
    #opt = tfopti.Adam(learning_rate=0.001)
    model_cp.compile(loss='sparse_categorical_crossentropy', optimizer=opt,
                     metrics=cfg['metrica'])
    # Graficando el modelo
    os.makedirs(savepath, exist_ok=True)
    print('Generando imagen del modelo')
    plot_model(model_cp, to_file=os.path.join(savepath, red_name + '.png'))
    if save_model:
        model_cp.save(os.path.join(savepath, red_name + '.h5'))
    return model_cp, model_nc


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
    x = Dense(1000, activation=cfg['fc_act'], name='fully_conected_0')(x)
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
    early_stop = EarlyStopping(monitor='val_loss', patience=15,
                               verbose=1, min_delta=1e-4)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10,
                                  verbose=1, min_delta=1e-4)
    callbacks_list = [early_stop, reduce_lr]
    callbacks_list = [early_stop]
    nb_test_samples = len(test_generador)
    pasos_fit = (nb_test_samples // cfg_train['BATCH_SIZE'])
    model_history = modelo.fit(train_generador,
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
    #plt.xticks(np.arange(0, 60, 1.0))
    plt.yticks(np.arange(0, 1, 0.1))
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.xlabel("Num of Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy vs Validation Accuracy")
    plt.legend(['train', 'validation'])
    plt.savefig(os.path.join(savepath, red_name + '_TaccVacc.png'))

    plt.figure(1)
    plt.plot(model_history.history['loss'], 'r')
    plt.plot(model_history.history['val_loss'], 'g')
    #plt.xticks(np.arange(0, 60, 1.0))
    plt.yticks(np.arange(0, 1, 0.1))
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.xlabel("Num of Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Validation Loss")
    plt.legend(['train', 'validation'])
    plt.savefig(os.path.join(savepath, red_name + '_TlossVloss.png'))

    plt.figure(2)
    plt.plot(model_history.history['mse'], 'r')
    plt.plot(model_history.history['val_mse'], 'g')
    #plt.xticks(np.arange(0, 60, 1.0))
    plt.yticks(np.arange(0, 1, 0.1))
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.xlabel("Num of Epochs")
    plt.ylabel("MSE")
    plt.title("Training Loss vs Validation Loss")
    plt.legend(['train', 'validation'])
    plt.savefig(os.path.join(savepath, red_name + '_mseTlossVloss.png'))
    # plt.show()


if __name__ == "__main__":
    modelopaper = model_paper()
    print(modelopaper.summary())
