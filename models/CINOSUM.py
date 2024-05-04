from .model_builder import *
from .generate import *
import torch
from types import SimpleNamespace

args_dict = {
    "encoder": 'transformer',
    "mode": 'test',
    "bert_data_path": '/',
    "model_path": '../',
    "result_path": '../',
    "temp_dir": '/',
    "bert_config_path": '/',
    "batch_size": 1000,
    "use_interval": True,
    "hidden_size": 2048,
    "ff_size": 512,
    "heads": 4,
    "inter_layers": 2,
    "rnn_size": 1024,
    "param_init": 0,
    "param_init_glorot": True,
    "dropout": 0.1,
    "optim": 'adam',
    "lr": 1,
    "beta1": 0.9,
    "beta2": 0.999,
    "decay_method": '',
    "warmup_steps": 10000,
    "max_grad_norm": 0,
    "save_checkpoint_steps": 5,
    "accum_count": 1,
    "world_size": 1,
    "report_every": 1,
    "train_steps": 1000,
    "recall_eval": False,
    "visible_gpus": '-1',
    "gpu_ranks": '0',
    "log_file": '/',
    "dataset": '',
    "seed": 3407,
    "test_all": False,
    "test_from": '',
    "train_from": '',
    "report_rouge": True,
    "block_trigram": True
}



def initialize_args():
    args = SimpleNamespace(**args_dict)
    return args

class CINOSUM(nn.Module):
    """
    CINOSUM类，继承自PyTorch的nn.Module，用于多语言摘要生成任务。

    参数:
    - device (str): 指定模型运行的设备，默认为'cpu'。
    - model_path (str): 指定预训练模型的路径，默认为None值。
    - len_pred (int): 生成摘要的预期长度，默认为2。
    - language (str): 模型处理的目标语言，默认为'zh'（中文）。
    - load_pretrained_bert (bool): 是否加载训练的模型，默认为True。

    属性:
    - args: 通过initialize_args函数初始化的参数对象。
    - len_pred: 生成摘要的长度。
    - language: 模型处理的语言。
    - model_path: 预训练模型的文件路径。
    - config: 使用XLMRobertaConfig加载的模型配置对象。
    - model: Summarizer模型实例。
    - device: 模型运行的设备。
    - if_load: 标记模型是否已加载，避免重复加载。

    方法:
    - __init__: 类的构造函数，用于初始化模型及其相关配置。

    注意:
    - 若model_path为空或None，则会跳过加载模型的步骤，使用随机初始化的模型。
    - 需要事先导入PyTorch相关模块和XLMRobertaConfig，以及定义Summarizer模型。
    """
    def __init__(self, device='cpu', model_path = None, len_pred = 2, language = 'zh', load_pretrained_bert = True):
        super(CINOSUM, self).__init__()

        self.args = initialize_args()

        self.len_pred = len_pred
        self.language = language
        self.model_path = model_path

        self.config = XLMRobertaConfig.from_pretrained("hfl/cino-large-v2")
        self.model = Summarizer(self.args, device, xlmr_config=self.config)
        self.device = device
        
        self.if_load = False
        #不使用训练过的模型,使用随机初始化的模型
        if model_path == '' or model_path == None:
            print('Use a random model')
            self.if_load = True
        
        
    def Extractive(self, contexts, batch_size = 1, language = 'zh', len_pred = 2):
        """
        运行抽取式摘要过程。

        参数:
        - self: 类实例的引用。
        - contexts (list of str): 需要进行摘要提取的文本列表。
        - batch_size (int, optional): 在模型推理过程中一次处理的文本数量，默认为1。
        - language (str, optional): 输入文本的语言，默认为'zh'（中文）。
        - len_pred (int, optional): 预期生成摘要的句子数量，默认为2。

        返回:
        - condidate_list (list of str): 模型生成的摘要列表。

        过程描述:
        - 首先检查模型是否已加载，如果尚未加载，则加载模型并设置为评估模式。
        - 使用生成函数生成摘要，该函数接受模型、文本、设备等参数。
        - 返回生成的摘要列表。

        """
        if self.if_load == False:
            checkpoint = torch.load(self.model_path, map_location=lambda storage, loc: storage)
            self.model.load_cp(checkpoint)
            self.model.eval()
            self.if_load = True
            print(f'load model success. load from{self.model_path}')
        condidate_list = generate(self.model, contexts, self.device, batch_size=batch_size, len_pred=len_pred, language=language)
        
        return condidate_list
    

        