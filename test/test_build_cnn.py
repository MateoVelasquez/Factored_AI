import os
import sys
ruta = os.path.abspath(os.path.normpath('..'))
sys.path.append(ruta)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cnn.red_neural as red #noqa

cfg_train = {
    'TRAIN_DATA_PATH': r'C:\Users\MATEO\RepositoriosGIT\Factored_AI\img\train',
    'TEST_DATA_PATH': r'C:\Users\MATEO\RepositoriosGIT\Factored_AI\img\test',
    'VAL_DATA_PATH': r'C:\Users\MATEO\RepositoriosGIT\Factored_AI\img\val',
    'BATCH_SIZE': 32,
    'EPOCAS': 50,
    'IMG_SZE': (224, 224, 1),  # (heigth, width, channel)
    'IMG_COLOR': "grayscale"
}


def test_red():
    # Parametros de configuracion de la red
    parametros = {
        'padding': (4, 4),
        'kernel_num': 32,
        'kernel_sze': (3, 3),
        'stride': (1, 1),
        'norm_ejes': 3,
        'l1_act': 'relu',
        'maxpool_sze': (2, 2),
        'fc_act': 'sigmoid',
        'optimizer': 'adam',
        'lost_fn': 'binary_crossentropy',
        'metrica': ['acc', 'mse']
    }
    cnn_red, _ = red.model_build(cfg_train['IMG_SZE'], parametros,
                                 save_model=True)
    historial, modelo = red.train_model(cnn_red, cfg_train)
    red.resultados_graficos(historial)


if __name__ == "__main__":
    test_red()
