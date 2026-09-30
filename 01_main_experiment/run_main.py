"""
run_main.py
"""

import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'core')) 

from core.config import Config                     
from core.config import refresh_dataset_paths     
import core.train_5fold_dce as engine             

Config = engine.Config


def main():
    parser = argparse.ArgumentParser(description='CAMPL')
    parser.add_argument('--dataset', default='A', choices=['A', 'B', 'C', 'D', 'E'])
    parser.add_argument('--model', default='CNN4',
                        choices=['CNN4', 'GoogleNet', 'ResNet9', 'ResNet18', 'VGG16'])
    parser.add_argument('--ray', action='store_true',
                        help='use Ray Tune')
    parser.add_argument('--dev-path', default=None, help='')
    parser.add_argument('--test-path', default=None, help='')
    args = parser.parse_args()

    Config.DATASET = args.dataset
    refresh_dataset_paths(Config)  
    if args.dev_path:
        Config.DATA_PATH = args.dev_path
    if args.test_path:
        Config.TEST_DATA_PATH = args.test_path
    Config.MODEL_NAME = args.model
    Config.METHODS = 4                       
    Config.USE_RAY = args.ray
    Config.SAVE_DIR = os.path.join(HERE, 'results')

    if Config.USE_RAY:
        engine.run_ray_tune()
    else:
        fold_results, best_fold_idx, num_classes = engine.run_5fold_cv()
        engine.save_cv_summary(fold_results, best_fold_idx, Config)
        engine.test_best_model(num_classes)


if __name__ == '__main__':
    main()
