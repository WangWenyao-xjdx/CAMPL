"""
run_main.py — 主实验 (CAMPL, METHODS=4)
=========================================
主模型 = 最优传统骨干（默认 CNN4）+ 完整损失组合（L1*DCE + L2*InfoNCE-alpha）。

执行顺序说明（重要）:
  1. 先运行超参数搜索:   python run_main.py --ray --dataset A
     （Ray Tune 在开发集五折验证 UAR 上搜索；trial 绝不接触测试集；
       搜索结束后自动用最优配置重跑五折并仅评估一次测试集，
       最优配置保存为 results/.../ray_best_config.txt）
  2. 把 ray_best_config.txt 中的最优超参数写入 core/config.py
     （BATCH_SIZE / LR / L1 / L2 / TEMPERATURE / ALPHA），然后固定超参
     在全部五个数据集上复跑:  python run_main.py --dataset A ... E

输出: results/{DATASET}/{MODEL_NAME}/method_4/
  results.txt / best_model.pth / best_fold_centers.npz / best_fold_train.npz /
  best_fold_val.npz / best_fold_val_predictions.npz / test_predictions.npz
  （Ray 模式另有 ray_best_config.txt, ray_trials_summary.csv, ray_trials/）
"""

import os
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'core')) 

from core.config import Config                     # noqa: E402
from core.config import refresh_dataset_paths     # noqa: E402
import core.train_5fold_dce as engine              # noqa: E402

# 【修复】engine 内部是 `from config import Config`（顶层模块 config），与上面的
# `from core.config import Config`（模块 core.config）是两个不同的模块对象——
# 此前驱动脚本对 Config 的修改（如 --dataset B）根本不会传入引擎，导致所有
# 数据集/模型参数都被忽略、永远跑数据集 A。统一改用引擎实际读取的 Config。
Config = engine.Config


def main():
    parser = argparse.ArgumentParser(description='主实验: CAMPL (METHODS=4)')
    parser.add_argument('--dataset', default='A', choices=['A', 'B', 'C', 'D', 'E'])
    parser.add_argument('--model', default='CNN4',
                        choices=['CNN4', 'GoogleNet', 'ResNet9', 'ResNet18', 'VGG16'])
    parser.add_argument('--ray', action='store_true',
                        help='启用 Ray Tune 超参数搜索（阶段1；仅对主骨干运行一次）')
    parser.add_argument('--dev-path', default=None, help='覆盖开发集 canonical 文件路径')
    parser.add_argument('--test-path', default=None, help='覆盖测试集 canonical 文件路径')
    args = parser.parse_args()

    Config.DATASET = args.dataset
    refresh_dataset_paths(Config)  # 修复：切换数据集后重建 DATA_PATH/TEST_DATA_PATH
    if args.dev_path:
        Config.DATA_PATH = args.dev_path
    if args.test_path:
        Config.TEST_DATA_PATH = args.test_path
    Config.MODEL_NAME = args.model
    Config.METHODS = 4                        # 完整 CAMPL
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
