import torch

class Config():
    def __init__(self):
        self.num_classes = 10
        self.num_clients = 20
        self.prototype_size = 512
        self.select_rate = 1  # fraction of clients per communication round
        self.nums_g_epoch = 50 # number of pretraining round
        self.nums_ft_epoch = 500 # number of fine-tuning round
        self.dataset = 'mnist'
        self.model = 'cnn'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batchsize = 32
        self.partition = 'pathological'
        self.learning_rate = 0.01
        self.num_proxy_data = 100 * self.num_classes
        self.l_epoch = 100 # local update
        self.prox_cfft = 0.01 # punish term coefficient of fedprox
        self.finetune = False
        self.algorithm = 'Ratioloss'
        self.imb_factor = 0.01
        self.m_class_fraction = 0.5
        self.longtail_type = 'exp' # 'step' & 'exp'

    def init_configuration(self, args):
        if args.algorithm:
            self.algorithm = args.algorithm
        if args.num_ft_epoch:
            self.finetune = True
            self.nums_ft_epoch = args.num_ft_epoch
        if args.num_g_epoch:
            self.nums_g_epoch = args.num_g_epoch
        if args.model:
            self.model = args.model


conf = Config()