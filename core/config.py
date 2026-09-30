
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # /root/A

class Config:

    DATASET = 'A'  #  A, B, C, D, E

    
    DATA_PATH = os.path.join(BASE_DIR, 'datasets', DATASET, 'developmentSet.xlsx')
    TEST_DATA_PATH = os.path.join(BASE_DIR, 'datasets', DATASET, 'testSet.xlsx')


    SAVE_DIR = './results'


    #'CNN4', 'GoogleNet', 'ResNet9', 'ResNet18', 'VGG16'
    MODEL_NAME = 'ResNet18'

    IN_CHANNELS = 1
    NUM_CLASSES = None
    NUM_HIDDEN_UNITS = 128

    EPOCHS = 200          # >= PATIENCE，保证早停有机会触发
    BATCH_SIZE = 256
    LR = 0.0005
    WEIGHT_DECAY = 1e-4
    PATIENCE = 30

    METHODS = 4

    L1 = 0.7
    L2 = 0.8
    TEMPERATURE = 0.4
    ALPHA = 0.2


    COSINE_SHIFT = True

    CLASS_BALANCED_INFONCE = True

    USE_MULTI_PROTOTYPE = True
    FINCH_DISTANCE = 'cosine'   

    N_BOOTSTRAP = 1000    
    BOOTSTRAP_SEED = 42

    
    USE_RAY = True              
    RAY_SEARCH_ITER = 500      
    RAY_RANDOM_STEPS = 50       

    #   'continuous' —  (loguniform/uniform)，
    #   'discrete' 
    RAY_SPACE_MODE = 'continuous'

    RAY_PRUNE_FOLD1 = True      
    RAY_MIN_FOLD1_UAR = 0.60    
    RAY_SEARCH_OBJECTIVE = 'best_fold'
    RAY_SEARCH_HIDDEN_UNITS = True   

    
    MAX_PROTOTYPES_PER_CLASS = 10
    MIN_SAMPLES_PER_PROTOTYPE = 5

    FINCH_WARMUP_EPOCHS = 0

    
    DCE_TAU_INIT = 1.0        
    DCE_TAU_LEARNABLE = True 
    DCE_TAU_MIN = 0.05
    DCE_TAU_MAX = 5.0
    RAY_SEARCH_DCE_TAU = False


    SEED = 42
    DEVICE = 'cuda'


def refresh_dataset_paths(cfg):
    cfg.DATA_PATH = os.path.join(BASE_DIR, 'datasets', cfg.DATASET, 'developmentSet.xlsx')
    cfg.TEST_DATA_PATH = os.path.join(BASE_DIR, 'datasets', cfg.DATASET, 'testSet.xlsx')
