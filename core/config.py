"""
config.py
=========
集中管理所有参数，方便统一修改。
只需修改此文件中的数值，无需改动模型和训练代码。

【存储结构说明】
普通模式: results/{DATASET}/{MODEL_NAME}/method_{METHODS}/
Ray模式:  results/{DATASET}/{MODEL_NAME}/method_{METHODS}/ray_trials/{trial_id}/
"""

import os

# 获取 config.py 所在目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # /root/A

class Config:
    # ==================== 数据路径 ====================
    # 预处理（基线校正/平滑/单条光谱归一化）与开发/测试划分已由作者完成；
    # 这里直接指向预处理完成的 canonical 文件
    #（col0=类别标签, col1=患者编号, col2+=光谱强度，可选首行波数）。
    # 建议使用「基线校正+平滑+单条光谱 z-score」版本的数据（见 README）。
    DATASET = 'A'  # 可选: A, B, C, D, E
    # DATASET = 'B'  # 可选: A, B, C, D, E
    # DATASET = 'C'  # 可选: A, B, C, D, E
    # DATASET = 'D'  # 可选: A, B, C, D, E
    # DATASET = 'E'  # 可选: A, B, C, D, E
    
    DATA_PATH = os.path.join(BASE_DIR, 'datasets', DATASET, 'developmentSet.xlsx')
    TEST_DATA_PATH = os.path.join(BASE_DIR, 'datasets', DATASET, 'testSet.xlsx')

    # ==================== 保存根目录 ====================
    SAVE_DIR = './results'

    # ==================== 模型选择 ====================
    # 传统骨干: 'CNN4', 'GoogleNet', 'ResNet9', 'ResNet18', 'VGG16'
    # 现代基线: 'AttentionCNN1D', 'Transformer1D', 'Mamba1D'
    MODEL_NAME = 'ResNet18'

    # ==================== 模型参数 ====================
    IN_CHANNELS = 1
    NUM_CLASSES = None
    NUM_HIDDEN_UNITS = 128

    # ==================== 训练参数 ====================
    EPOCHS = 200          # >= PATIENCE，保证早停有机会触发
    BATCH_SIZE = 256
    LR = 0.0005
    WEIGHT_DECAY = 1e-4
    PATIENCE = 10

    # ==================== 损失函数组合控制 ====================
    # METHODS:
    #   -1 -> 基线：仅使用普通分类头 (fc) 的交叉熵，不使用原型/DCE 头
    #   0  -> 只计算 prototypical_loss
    #   1  -> 只计算 DCE 损失
    #   2 -> 只计算 InfoNCE-alpha 损失
    #   3  -> L1 * DCE + L2 * InfoNCE
    #   4  -> L1 * DCE + L2 * InfoNCE-alpha
    METHODS = 1

    # ==================== 损失权重与超参数 ====================
    L1 = 0.7
    L2 = 0.8
    TEMPERATURE = 0.4
    ALPHA = 0.2

    # ==================== InfoNCE 公式开关 ====================
    # COSINE_SHIFT=True (默认): 余弦相似度定义在 [-1,1]，
    #   先平移 s01 = (s + 1) / 2 到 [0,1]，再计算 s01 ** alpha（InfoNCE-alpha）。
    # COSINE_SHIFT=False: 旧行为（clamp 负相似度到 0，再 pow(alpha)）。
    COSINE_SHIFT = True
    # CLASS_BALANCED_INFONCE=True (默认): InfoNCE 分母按类聚合，
    #   每个类先对其 FINCH 原型做 logsumexp 再减去 log(该类原型数)，
    #   避免原型多的类在分母中被过度加权。
    CLASS_BALANCED_INFONCE = True

    # ==================== 多原型设置 ====================
    # True: 使用 FINCH 对每个类聚类生成多原型
    # False: 回退为单原型（每类1个均值原型，与旧逻辑等价）
    USE_MULTI_PROTOTYPE = False
    FINCH_DISTANCE = 'cosine'   # FINCH 距离度量: 'cosine', 'euclidean', 'cityblock' 等

    # ==================== 评估指标 ====================
    N_BOOTSTRAP = 1000    # bootstrap 95% CI 重采样次数
    BOOTSTRAP_SEED = 42

    # ==================== Ray Tune 参数搜索控制 ====================
    USE_RAY = True              # True: 启用 Ray Tune 贝叶斯搜索; False: 普通五折
    RAY_SEARCH_ITER = 200        # 贝叶斯搜索总迭代次数（包含初始随机步数）
    RAY_RANDOM_STEPS = 15        # 贝叶斯优化前的初始随机探索步数 (TPESampler n_startup_trials)
    # 搜索空间模式:
    #   'continuous' — 连续采样 (loguniform/uniform)，搜索更细，但会得到 1.232e-4 这类任意精度取值
    #   'discrete'   — 离散网格采样 (tune.choice)，取值干净（如 1e-4/3e-4/1e-3），便于论文报告与复现
    RAY_SPACE_MODE = 'discrete'
    # trial 提前剪枝：fold 1 验证 UAR 低于阈值时立即终止该 trial，进入下一个 trial；
    # 被终止的 trial 以 FAIL 状态结束，TPE 不会用其超参数引导后续采样。
    RAY_PRUNE_FOLD1 = True      # True: 启用 fold-1 剪枝
    RAY_MIN_FOLD1_UAR = 0.73    # fold 1 验证 UAR 阈值
    # 搜索目标（优化指标）:
    #   'best_fold' — 五折中验证 UAR 最高的一折（与"最终模型取 best fold"的选择协议一致）
    #   'mean'      — 五折平均验证 UAR（更稳健，抗单折运气）
    # 两种指标在每个 trial 中都会记录到 ray_trials_summary.csv，仅是搜索目标不同。
    RAY_SEARCH_OBJECTIVE = 'best_fold'
    # 是否把原型特征维度 NUM_HIDDEN_UNITS 纳入搜索（默认 6 维可能是性能瓶颈）
    RAY_SEARCH_HIDDEN_UNITS = True   # True 时在 {8,16,32,64,128} 中搜索

    # ==================== 原型数量控制 ====================
    # FINCH 最细分区可能为每类生成过多原型，用以下两个参数收敛原型数量:
    #   MAX_PROTOTYPES_PER_CLASS: 每类原型数上限。在 FINCH 层级分区中自动选取
    #       聚类数 <= 上限的最细分区；若最粗分区仍超上限，保留成员最多的前 K 个簇。
    #       None = 不限制（旧行为，取最细分区）。
    #   MIN_SAMPLES_PER_PROTOTYPE: 成员少于此数的簇不作为原型（噪声簇过滤）；
    #       若过滤后某类为空，自动回退为该类的均值原型。
    MAX_PROTOTYPES_PER_CLASS = 5
    MIN_SAMPLES_PER_PROTOTYPE = 5
    # FINCH 预热: 先用普通分类头 CE 训练 backbone 若干 epoch，
    # 再在训练过的特征上运行 FINCH（论文公式(1) 的 f_theta 指已训练网络）。
    # 0 = 不预热（旧行为：在随机初始化特征上聚类，亚类结构无意义）。
    # 建议 10–30。
    FINCH_WARMUP_EPOCHS = 20

    # ---- DCE 可学习温度 tau_dce（logits = class_sim / tau_dce）----
    # 原始 DCE 把余弦相似度（[-1,1]）直接送入 log_softmax，决策余量很小。
    # tau 以 log 参数化并 clamp 到 [DCE_TAU_MIN, DCE_TAU_MAX]。
    # DCE_TAU_INIT=1.0 时与旧行为完全一致。
    DCE_TAU_INIT = 1.0        # tau 初值
    DCE_TAU_LEARNABLE = True  # True: tau 作为可学习参数；False: 固定为 DCE_TAU_INIT
    DCE_TAU_MIN = 0.05
    DCE_TAU_MAX = 5.0
    # True 时在 Ray 网格中搜索 tau 初值（一般不需要：tau 本身可学习；
    # 仅当 DCE_TAU_LEARNABLE=False、想搜索"固定 tau"时开启）
    RAY_SEARCH_DCE_TAU = False

    # ==================== 其他 ====================
    SEED = 42
    DEVICE = 'cuda'


def refresh_dataset_paths(cfg):
    """--dataset 切换后重建数据路径。
    DATA_PATH/TEST_DATA_PATH 在类定义时按默认 DATASET='A' 固化，
    切换数据集后必须调用本函数重建，否则仍读取数据集 A。"""
    cfg.DATA_PATH = os.path.join(BASE_DIR, 'datasets', cfg.DATASET, 'developmentSet.xlsx')
    cfg.TEST_DATA_PATH = os.path.join(BASE_DIR, 'datasets', cfg.DATASET, 'testSet.xlsx')
