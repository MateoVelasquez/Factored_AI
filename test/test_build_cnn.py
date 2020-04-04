import os
import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
ruta = os.path.abspath(os.path.normpath('..'))
sys.path.append(ruta)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cnn.red_neural as red #noqa
from cnn.gen_generator import build_generador #noqa

cfg_train = {
    'TRAIN_DATA_PATH': r'C:\Users\MATEO\RepositoriosGIT\Factored_AI\img\train',
    'TEST_DATA_PATH': r'C:\Users\MATEO\RepositoriosGIT\Factored_AI\img\test',
    'VAL_DATA_PATH': r'C:\Users\MATEO\RepositoriosGIT\Factored_AI\img\val',
    'BATCH_SIZE': 1,
    'EPOCAS': 50,
    'IMG_SZE': (200, 200, 3),  # (heigth, width, channel)
    'IMG_COLOR': "rgb"
}


def train_red():
    # Parametros de configuracion de la red
    parametros = {
        'padding': (1, 1),
        'kernel_num': 32,
        'kernel_sze': (3, 3),
        'stride': (1, 1),
        'norm_ejes': 3,
        'l1_act': 'relu',
        'maxpool_sze': (2, 2),
        'fc_act': 'softmax',
        'optimizer': 'adam',
        'lost_fn': 'categorical_crossentropy',
        'metrica': ['acc', 'mse']
    }
    cnn_red, _ = red.model_paper(cfg_train['IMG_SZE'], parametros,
                                 save_model=True)
    historial, modelo = red.train_model(cnn_red, cfg_train)
    red.resultados_graficos(historial)


def prediccion(red_name, img_to_predict):
    cnn = load_model(os.path.join(ruta, 'saved_models', red_name + '.h5'))
    cnn.load_weights(os.path.join(ruta, 'saved_models',
                                  red_name + '_pesos.h5'))

    img = load_img(img_to_predict, target_size=(cfg_train['IMG_SZE'][1],
                                                cfg_train['IMG_SZE'][0]),
                   color_mode=cfg_train['IMG_COLOR'])
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    arreglo = cnn.predict(img)
    print(arreglo)
    # resultado = arreglo[0]
    # print('\n')
    # if resultado[0] < 0.4:
    #     print(f'Probabilidad {resultado[0]}. El paciente no tiene neumonia')
    # else:
    #     print(f'Probabilidad {resultado[0]}. El paciente pude tener neumonia')


def evaluar(red_name):
    cnn = load_model(os.path.join(ruta, 'saved_models', red_name + '.h5'))
    cnn.load_weights(os.path.join(ruta, 'saved_models',
                                  red_name + '_pesos.h5'))
    
    dim = cfg_train['IMG_SZE'][:-1]

    val_generador, _ = build_generador(cfg_train['VAL_DATA_PATH'],
                                       batch_sze=cfg_train['BATCH_SIZE'],
                                       trgt_sze=dim,
                                       color=cfg_train['IMG_COLOR'])
    result = cnn.evaluate(val_generador)
    dict(zip(cnn.metrics_names, result))
    pass



def model_summary():
    parametros = {'metrica': ['acc', 'mse']}
    modelopaper,_ = red.model_paper(cfg_train['IMG_SZE'], parametros,
                                  save_model=False)
    print(modelopaper.summary())

if __name__ == "__main__":
    #resumen red
    #model_summary()

    # # crear y entrenar la red:
    #train_red()

    # # Evaluate
    #evaluar('paper')

    # # hacer una prediccion:
    #file_path = r'C:\Users\MATEO\RepositoriosGIT\Factored_AI\img\val\PNEUMONIA\person1952_bacteria_4883.jpeg' #noqa
    file_path = r'C:\Users\MATEO\RepositoriosGIT\Factored_AI\img\val\NORMAL\NORMAL2-IM-1437-0001.jpeg' #noqa
    prediccion('paper', file_path)
