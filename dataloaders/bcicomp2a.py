from __future__ import print_function
import torch
import scipy.io
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import torch.utils.data
import numpy as np
import scipy.io

class BCICompIV2a():

    def __init__(self, root, sub, memory, task_num, train=True):
        """
        Initialize the dataset loader.

        Args:
            root (str): Root directory of the dataset.
            sub (int): Subject number.
            memory (dict): Memory object for task-specific data.
            task_num (int): Number of tasks.
            train (bool, optional): Whether to load training data. Defaults to True.
        """


        self.train = train
        self.root = root
        data_dict = scipy.io.loadmat(self.root + 'new_train/' + str(sub) + '.mat')  # load training data

        ##################################for 2 class###########################################

        # # Extract training data and targets
        # trainX = np.array(data_dict['trainX'])
        # trainY = np.array(data_dict['trainY']).squeeze()

        # # Transpose data
        # raw_data = np.transpose(trainX, (2, 1, 0))
        # raw_targets = list(trainY)

        # # Filter data for labels 0 and 1
        # mask = [(target == 0) or (target == 1) for target in raw_targets]
        # self.data = raw_data[mask]
        # self.targets = [raw_targets[i] for i, m in enumerate(mask) if m]

        # print(f"Data shape: {self.data.shape}, Targets shape: {len(self.targets)}")

        # # Set task module labels and task discriminator labels
        # self.tt = [task_num] * len(self.data)  # task module labels
        # self.td = [task_num + 1] * len(self.data)  # task discriminator labels

        # if train and memory is not None:
        #     for task_id in range(task_num):
        #         for i in range(len(memory[task_id]['x'])):
        #             self.data = np.append(self.data, [memory[task_id]['x'][i]], axis=0)
        #             self.targets.append(memory[task_id]['y'][i])
        #             self.tt.append(memory[task_id]['tt'][i])
        #             self.td.append(memory[task_id]['td'][i])
        ##################################for 4 class###########################################
                    

        trainX = np.array(data_dict['trainX'])
        trainY = np.array(data_dict['trainY']).squeeze()

        # 转置数据
        self.data = np.transpose(trainX, (2, 1, 0))
        self.targets = list(trainY)

        print(f"Data shape: {self.data.shape}, Targets shape: {len(self.targets)}")

        # 设置任务模块标签和任务鉴别器标签
        self.tt = [task_num] * len(self.data)  # 任务模块标签
        self.td = [task_num + 1] * len(self.data)  # 鉴别器标签

        if train and memory is not None:
            for task_id in range(task_num):
                for i in range(len(memory[task_id]['x'])):
                    self.data = np.append(self.data, [memory[task_id]['x'][i]], axis=0)
                    self.targets.append(memory[task_id]['y'][i])
                    self.tt.append(memory[task_id]['tt'][i])
                    self.td.append(memory[task_id]['td'][i])
                    
    #################################################################################

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (subject_data, target) where target is index of the target class.
        """
        data, target, tt, td = self.data[index], int(self.targets[index]), self.tt[index], self.td[index]

        return data, target, tt, td



    def __len__(self):
        return len(self.data)


class DatasetGen(object):
    """docstring for DatasetGen"""

    def __init__(self, args):
        super(DatasetGen, self).__init__()
        self.use_memory = args.use_memory

        self.num_workers = args.num_workers
        self.pin_memory = True
        self.seed = args.seed
        np.random.seed(self.seed)

        self.batch_size = args.batch_size
        self.pc_valid = args.pc_valid
        self.root = args.data_dir
        self.latent_dim = args.latent_dim
        self.num_samples = args.samples # 30 # make sure  use memory or not

        self.task_cls = [[s, args.num_classes] for s in range(args.num_subjects)]  # [[0, 2], [1, 2], [2, 2], [3, 2],.., [8, 2]]  -> unnecessary. remove later.
        self.subject_queue = np.random.permutation(np.arange(1, args.num_subjects+1)).tolist()  # shuffle order of subjects 1-9  e.g., [2, 1, 4, 3, 5, 6, 7, 8, 9]
        self.task_ids = [[s] for s in range(args.num_subjects)]  # [[0], [1], [2], [3], [4], [5], [6], [7], [8]]

        self.dataloaders = {}
        self.train_set = {}
        self.test_set = {}
        self.train_split = {}

        self.task_memory = {}
        for i in range(args.num_subjects):
            self.task_memory[i] = {}
            self.task_memory[i]['x'] = []
            self.task_memory[i]['y'] = []
            self.task_memory[i]['tt'] = []
            self.task_memory[i]['td'] = []

    def get(self, task_id):

        self.dataloaders[task_id] = {}
        sys.stdout.flush()
        if task_id == 0:
            memory = None
        else:
            memory = self.task_memory

        self.train_set[task_id] = BCICompIV2a(root=self.root, sub=self.subject_queue[task_id], memory=memory, task_num=task_id)

        # 假设 self.pc_valid 和 self.pc_test 分别是验证集和测试集的比例
        self.pc_valid = 1.5/ 10  # 20% of the dataset for the validation set
        self.pc_test = 1.5/ 10  # 20% of the dataset for the test set
        split_valid = int(np.floor(self.pc_valid * len(self.train_set[task_id])))
        split_test = int(np.floor(self.pc_test * len(self.train_set[task_id])))

        # 确保训练集、验证集和测试集的总和等于原始数据集的大小
        split_train = len(self.train_set[task_id]) - split_valid - split_test

        # 分割数据集
        train_split, valid_split, test_split = torch.utils.data.random_split(
            self.train_set[task_id], [split_train, split_valid, split_test]
        )

        train_loader = torch.utils.data.DataLoader(
            train_split, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers,
            pin_memory=self.pin_memory, drop_last=True
        )

        # 对于验证集和测试集，您可以使用与训练集相同的 batch_size，除非有特殊需求
        valid_loader = torch.utils.data.DataLoader(
            valid_split, batch_size=int(self.batch_size/4),
            shuffle=False, num_workers=self.num_workers,
            pin_memory=self.pin_memory, drop_last=True
        )

        test_loader = torch.utils.data.DataLoader(
            test_split, batch_size=int(self.batch_size/2),
            shuffle=False, num_workers=self.num_workers,
            pin_memory=self.pin_memory, drop_last=True
        )

        self.dataloaders[task_id]['train'] = train_loader
        self.dataloaders[task_id]['valid'] = valid_loader
        self.dataloaders[task_id]['test'] = test_loader
        self.dataloaders[task_id]['name'] = f'BCI-COMP2A-task{task_id}-sub{self.subject_queue[task_id]}'

        shape = self.train_set[task_id].data.shape[:]
        print(f"Training set size:      {len(train_loader.dataset)}  EEG signals of shape {shape}")
        print(f"Validation set size:    {len(valid_loader.dataset)}  EEG signals of shape {shape}")
        print(f"Test set size:          {len(test_loader.dataset)}  EEG signals of shape {shape}")

        # If the flag use_memory is set to 'yes' and the number of samples to store in memory is greater than 0, it updates the memory with samples from the current task.
        if self.use_memory == 'yes' and self.num_samples > 0:
            self.update_memory(task_id)

        return self.dataloaders




    def update_memory(self, task_id):
        num_samples_per_subjects = self.num_samples // len(self.task_ids[task_id])
        mem_class_mapping = {i: i for i, c in enumerate(self.task_ids[task_id])}

        if task_id == 0:
            memory = None
        else:
            memory = self.task_memory

        # Looping over each class in the current task
        for i in range(len(self.task_ids[task_id])):
            dataset = BCICompIV2a(root=self.root, sub=self.subject_queue[task_id], memory=memory, task_num=task_id, train=True)

            data_loader = torch.utils.data.DataLoader(dataset, batch_size=1,num_workers=self.num_workers,pin_memory=self.pin_memory)

            randind = torch.randperm(len(data_loader.dataset))[:num_samples_per_subjects]  # randomly choosenum_samples_per_subjects samples from the current task

            # Adding the selected samples to memory
            for ind in randind:
                self.task_memory[task_id]['x'].append(data_loader.dataset[ind][0])
                self.task_memory[task_id]['y'].append(mem_class_mapping[i])
                self.task_memory[task_id]['tt'].append(data_loader.dataset[ind][2])
                self.task_memory[task_id]['td'].append(data_loader.dataset[ind][3])

        print ('Memory updated by adding {} data'.format(len(self.task_memory[task_id]['x'])))




