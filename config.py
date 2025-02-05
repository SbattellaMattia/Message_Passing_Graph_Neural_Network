from os.path import join, dirname
from os import getcwd, makedirs
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

DIR_PATH = dirname(join(getcwd(), __file__))
DATASET_PATH = join(DIR_PATH, 'dataset')

"""
TO_SERVER_PATH viene utilizzato nel caso in cui, per motivi di spazio insufficienti,
vogliamo salvare i dati degli esperimenti in una posizione nel file system differente
dalla directory del progetto
"""
TO_SERVER_PATH = None
# TO_SERVER_PATH = '/dev/shm'


def get_result_path(ds_name, model):
    if TO_SERVER_PATH is None:
        result_path = join(DIR_PATH, 'NAP', 'results', model, ds_name)
    else:
        result_path = join(TO_SERVER_PATH,'NAP', 'results', model, ds_name)
    makedirs(result_path, exist_ok=True)
    return result_path

# TRAINING PARAMETERS
SEED = 42
TRAIN_SPLIT = 0.67
PATIENCE = 20
THRESHOLD=1
NO_REPEAT=True
EPOCH = [200]

BATCH_SIZE = [64]

#PROVE PROF
#LEARNING_RATE = [1e-2, 1e-3, 1e-4]
#DROPOUT = [0.1, 0.2]

#PROVE MIE
LEARNING_RATE = [1e-4, 1e-3]
DROPOUT = [0.2, 0.1]

# ----------

# MODEL PARAMETERS

#PROVE PROF
#GRAPH_CONV_LAYERS = [2,3,5,7]
#NUM_NEURONS = [32,64]
#K_VALUES = [3, 5, 7, 30]


#PROVA MIA
GRAPH_CONV_LAYERS = [2,5,7]
NUM_NEURONS = [64,128]
K_VALUES = [3,7,10]


GRID_COMBINATIONS = []
for batch_size in BATCH_SIZE:
    for epochs in EPOCH:
        for k in K_VALUES:
            for num_neurons in NUM_NEURONS:
                for graph_conv_layers in GRAPH_CONV_LAYERS:
                    for dropout_rate in DROPOUT:
                        for learning_rate in LEARNING_RATE:
                            tmp_graph_conv_layer_size = ''
                            data_comb = (f'{epochs}_epochs_{learning_rate}_lr_{batch_size}_'
                                         f'bs_{dropout_rate}_dropout_{graph_conv_layers}_graph_conv_layers'
                                         f'_{num_neurons}_num_neurons_{k}_k')
                            GRID_COMBINATIONS.append({
                                'path': data_comb,
                                'batch_size': batch_size,
                                'epochs': epochs,
                                'learning_rate': learning_rate,
                                'dropout': dropout_rate,
                                'graph_conv_layers': graph_conv_layers,
                                'num_neurons': num_neurons,
                                'k': k})
# ----------
