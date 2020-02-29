""" modelo de red neural convolcional

Modelo de red neuronal basico uno.

"""
from tensorflow.keras.layers import (Input, Dense, Activation, ZeroPadding2D,
                          BatchNormalization, Flatten, Conv2D) #noqa
from tensorflow.keras.layers import (AveragePooling2D, MaxPooling2D, Dropout, #noqa
                          GlobalMaxPooling2D, GlobalAveragePooling2D) #noqa
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model


def model_build(in_shp, cfg):
    """ Construccion del modelo

    Función para construirelur el modelo
    """
    # convolucional
    x_in = Input(in_shp)
    print(cfg['padding'])
    x = ZeroPadding2D(cfg['padding'])(x_in)
    x = Conv2D(cfg['kernel_num'], cfg['kernel_sze'], strides=cfg['stride'],
               name='convolucion_0')(x)
    x = BatchNormalization(axis=cfg['norm_ejes'], name='batch_normal_0')(x)
    x = Activation(cfg['l1_act'])(x)
    x = MaxPooling2D(cfg['maxpool_sze'], name='max_pooling')(x)

    # Fullyconected
    x = Flatten()(x)
    x = Dense(1, activation=cfg['fc_act'], name='fully_conected')(x)
    model_nc = Model(inputs=x_in, outputs=x, name='red_uno')

    # compilación del modelo
    model_cp = model_nc
    model_cp.compile(cfg['optimizer'], cfg['lost_fn'],
                     metrics=cfg['metrica'])
    print('Generando imagen del modelo')
    plot_model(model_cp)
    return model_cp, model_nc


if __name__ == "__main__":
    parametros = {
        'padding': (4, 4),
        'kernel_num': 32,
        'kernel_sze': (7, 7),
        'stride': (1, 1),
        'norm_ejes': 3,
        'l1_act': 'relu',
        'maxpool_sze': (2, 2),
        'fc_act': 'sigmoid',
        'optimizer': 'adam',
        'lost_fn': 'binary_crossentropy',
        'metrica': ['accuracy']
    }
    resolucion = (20, 20, 3)
    model_build(resolucion, parametros)

    pass
