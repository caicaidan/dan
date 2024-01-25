import torch
# import numpy as np
from .EEGNet import EEGNet
from .DeepConvNet import DCN
from .ShallowConvNet import ShallowConvNet as SCN





class Private(torch.nn.Module):
    def __init__(self, args):
        super(Private, self).__init__()
        _, nchans, ntimes = args.inputsize
        latent_dim = args.latent_dim
        if args.model == 'EEGNet':
            self.task_out = torch.nn.ModuleList(
                [EEGNet(args) for _ in range(args.num_subjects)])
        elif args.model == 'SCN':
            self.task_out = torch.nn.ModuleList(
                [SCN(args) for _ in range(args.num_subjects)])
        elif args.model == 'DCN':
            self.task_out = torch.nn.ModuleList(
                [DCN(args, latent_dim=args.latent_dim, kernel_1=64, kernel_2=16, dropout=args.dropout, block_out_channels=[25, 25, 50, 100, 200])
 for _ in range(args.num_subjects)])

    def forward(self, x_p, task_id):
        x_p = self.task_out[task_id](x_p)
        return x_p.view(x_p.size(0), -1)


class Shared(torch.nn.Module):

    def __init__(self, args):
        super(Shared, self).__init__()
        _, nchans, ntimes = args.inputsize
        latent_dim = args.latent_dim
        if args.model == 'EEGNet':
            self.model = EEGNet(args)
        elif args.model == 'SCN':
            self.model = SCN(args)
        elif args.model == 'DCN':
            self.model = DCN(args, latent_dim=latent_dim, kernel_1=64, kernel_2=16, dropout=args.dropout,
                            block_out_channels=[25, 25, 50, 100, 200])

            # self.model = DCN(args)

    def forward(self, x_s):
        # x_s = torch.rand(self.batch_size, 22, 1001) # 为deepconvnet的forward函数提供输入修改
        return self.model(x_s)


# 1. 继承torch.nn.Module类
# 2. 在__init__()中定义网络需要的操作层
# 3. 在forward()中定义前向传播的运算
# 4. 可以根据需要定义其他函数或者类方法
# 5. 一般不需要定义反向传播函数，系统会自动实现反向传播
class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        ncha, size, _ = args.inputsize  # [1, 22, 1001]
        self.task_cls = args.task_cls  # [[0, 4], [1, 4], [2, 4], [3, 4], [4, 4], [5, 4], [6, 4], [7, 4], [8, 4]]
        self.latent_dim = args.latent_dim  # 64
        self.num_tasks = args.num_subjects  # 9
        self.hidden1 = args.hidden_dim[0]
        self.hidden2 = args.hidden_dim[1]

        self.samples = args.samples  # 30 # make sure  use memory or not
        self.shared = Shared(args)  # shared network
        self.private = Private(args)  # private network
        # nn. ModuleList
        # 它是一个储存不同 module，并自动将每个 module 的 parameters 添加到网络之中的容器。你可以把任意 nn.Module 的子类 (比如 nn.Conv2d, nn.Linear 之类的)
        # 加到这个 list 里面，方法和 Python 自带的 list 一样，无非是 extend，append 等操作。
        # ModuleList中的module参数会自动注册到Module中，但是普通的list不会

        # nn.Sequential内部实现了forward函数，因此可以不用写forward函数。而nn.ModuleList则没有实现内部forward函数。
        self.head = torch.nn.ModuleList()  # 9个任务，每个任务一个head , head是一个线性层 9个线性层 9个任务 9个head
        for i in range(self.num_tasks):
            self.head.append(  # nn.Sequential里面的模块按照顺序进行排列的，所以必须确保前一个模块的输出大小和下一个模块的输入大小是一致的
                torch.nn.Sequential(
                    torch.nn.Linear(self.latent_dim * 2, self.hidden1),
                    # nn.Linear(in_feature,out_feature,bias), Linear()函数通常用于设置网络中的全连接层
                    torch.nn.ELU(inplace=True),
                    # elu(x) = x if x >= 0 alpha * (exp(x) - 1) if x < 0 , inplace=True 会改变输入的数据 ，否则不会改变原输入，只会产生新的输出. ELU函数在 x >= 0 时返回其本身，而在 x < 0 时采用指数增长的方式进行平滑。
                    torch.nn.Dropout(),
                    # Dropout是一种正则化技术，通过在训练过程中阻止神经元节点间的联合适应性来减少过拟合。Dropout在训练过程中随机选择一些神经元节点不参与训练，但是在测试过程中则使用所有的神经元节点。Dropout的实现非常简单，只需要在每次训练过程中随机将一些神经元节点的输出值设为0即可。
                    torch.nn.Linear(self.hidden1, self.hidden2),
                    torch.nn.ELU(inplace=True),
                    torch.nn.Linear(self.hidden2, self.task_cls[i][1])
                ))

    def forward(self, x_s, x_p, tt, task_id):
        x_s = self.shared(x_s)  # 128
        x_p = self.private(x_p, task_id)  # 128
        x = torch.cat([x_p, x_s], dim=1)  # (32, 256)
        return torch.stack([self.head[tt[i]].forward(x[i]) for i in range(x.size(0))])

    def get_encoded_ftrs(self, x_s, x_p, task_id):
        return self.shared(x_s), self.private(x_p, task_id)

    def print_model_size(self):
        count_P = sum(p.numel() for p in self.private.parameters() if p.requires_grad)
        count_S = sum(p.numel() for p in self.shared.parameters() if p.requires_grad)
        count_H = sum(p.numel() for p in self.head.parameters() if p.requires_grad)

        print('Num parameters in S       = %s ' % (self.pretty_print(count_S)))
        print('Num parameters in P       = %s,  per task = %s ' % (
        self.pretty_print(count_P), self.pretty_print(count_P / self.num_tasks)))
        print('Num parameters in p       = %s,  per task = %s ' % (
        self.pretty_print(count_H), self.pretty_print(count_H / self.num_tasks)))
        print('Num parameters in P+p     = %s ' % self.pretty_print(count_P + count_H))
        print('-------------------------->   Total architecture size: %s parameters (%sB)' %
              (self.pretty_print(count_S + count_P + count_H), self.pretty_print(4 * (count_S + count_P + count_H))))

    def pretty_print(self, num):
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

############################################unchanged######################################################
# class Private(torch.nn.Module):
#     def __init__(self, args):
#         super(Private, self).__init__()
#         _, nchans, ntimes = args.inputsize
#         latent_dim = int(args.latent_dim)
#         if args.model == 'EEGNet':
#             self.task_out = torch.nn.ModuleList(
#                 [EEGNet(nchans=nchans, latent_dim=latent_dim) for _ in range(args.num_subjects)])
#         elif args.model == 'SCN':
#             self.task_out = torch.nn.ModuleList(
#                 [SCN(nchans=nchans, latent_dim=latent_dim) for _ in range(args.num_subjects)])
#         elif args.model == 'DCN':
#             self.task_out = torch.nn.ModuleList(
#                 [DCN(nchans=nchans, latent_dim=latent_dim) for _ in range(args.num_subjects)])

#     def forward(self, x_p, task_id):
#         x_p = self.task_out[task_id](x_p)
#         return x_p.view(x_p.size(0), -1)


# class Shared(torch.nn.Module):

#     def __init__(self, args):
#         super(Shared, self).__init__()
#         _, nchans, ntimes = args.inputsize
#         latent_dim = int(args.latent_dim)
#         if args.model == 'EEGNet':
#             self.model = EEGNet(nchans=nchans, ntimes=ntimes, latent_dim=latent_dim, dropout=0.)
#         elif args.model == 'SCN':
#             self.model = SCN(nchans=nchans, ntimes=ntimes, latent_dim=latent_dim, dropout=0.)
#         elif args.model == 'DCN':
#             self.model = DCN(nchans=nchans, latent_dim=latent_dim)

#     def forward(self, x_s):
#         # x_s = torch.rand(self.batch_size, 22, 1001) # 为deepconvnet的forward函数提供输入修改
#         return self.model(x_s)



# class Net(torch.nn.Module):
#     def __init__(self, args):
#         super(Net, self).__init__()
#         ncha, size, _ = args.inputsize  # [1, 22, 1001]
#         self.task_cls = args.task_cls  # [[0, 4], [1, 4], [2, 4], [3, 4], [4, 4], [5, 4], [6, 4], [7, 4], [8, 4]]
#         self.latent_dim = args.latent_dim  # 64
#         self.num_tasks = args.num_subjects  # 9
#         self.hidden1 = args.hidden_dim[0]
#         self.hidden2 = args.hidden_dim[1]

#         self.samples = args.samples  # 30 # make sure  use memory or not
#         self.shared = Shared(args)  # shared network
#         self.private = Private(args)  # private network
#         self.head = torch.nn.ModuleList()  # 9个任务，每个任务一个head , head是一个线性层 9个线性层 9个任务 9个head
#         for i in range(self.num_tasks):
#             self.head.append(  
#                 torch.nn.Sequential(
#                     torch.nn.Linear(self.latent_dim * 2, self.hidden1),
#                     torch.nn.ELU(inplace=True),
#                     torch.nn.Dropout(),
#                     torch.nn.Linear(self.hidden1, self.hidden2),
#                     torch.nn.ELU(inplace=True),
#                     torch.nn.Linear(self.hidden2, self.task_cls[i][1])
#                 ))

#     def forward(self, x_s, x_p, tt, task_id):
#         x_s = self.shared(x_s)  # 128
#         x_p = self.private(x_p, task_id)  # 128
#         x = torch.cat([x_p, x_s], dim=1)  # (32, 256)
#         return torch.stack([self.head[tt[i]].forward(x[i]) for i in range(x.size(0))])

#     def get_encoded_ftrs(self, x_s, x_p, task_id):
#         return self.shared(x_s), self.private(x_p, task_id)

#     def print_model_size(self):
#         count_P = sum(p.numel() for p in self.private.parameters() if p.requires_grad)
#         count_S = sum(p.numel() for p in self.shared.parameters() if p.requires_grad)
#         count_H = sum(p.numel() for p in self.head.parameters() if p.requires_grad)

#         print('Num parameters in S       = %s ' % (self.pretty_print(count_S)))
#         print('Num parameters in P       = %s,  per task = %s ' % (
#         self.pretty_print(count_P), self.pretty_print(count_P / self.num_tasks)))
#         print('Num parameters in p       = %s,  per task = %s ' % (
#         self.pretty_print(count_H), self.pretty_print(count_H / self.num_tasks)))
#         print('Num parameters in P+p     = %s ' % self.pretty_print(count_P + count_H))
#         print('-------------------------->   Total architecture size: %s parameters (%sB)' %
#               (self.pretty_print(count_S + count_P + count_H), self.pretty_print(4 * (count_S + count_P + count_H))))

#     def pretty_print(self, num):
#         magnitude = 0
#         while abs(num) >= 1000:
#             magnitude += 1
#             num /= 1000.0
#         return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

