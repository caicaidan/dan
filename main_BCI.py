import os,argparse,time
import numpy as np
from omegaconf import OmegaConf
from copy import deepcopy
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import utils

tstart=time.time()

# Arguments
parser = argparse.ArgumentParser(description='Adversarial Continual Learning...')
# Load the config file
parser.add_argument('--config',  type=str, default='./configs/config_bcicomp2b.yml')
flags = parser.parse_args()
args = OmegaConf.load(flags.config)

########################################################################################################################

# Args -- Experiment
if args.experiment=='bci-intuitive':
    from dataloaders import intuitiveDataset as datagenerator
elif args.experiment=='bci-competition-IV2a':
    from dataloaders import bcicomp2a as datagenerator
elif args.experiment=='bci-competition-IV2b':
    from dataloaders import bcicomp2b as datagenerator
else:
    raise NotImplementedError

from sil import SIL as approach



# Args -- Network
if 'bci' in args.experiment:
    from networks import mlp_sil_BCI as network
# elif args.experiment == 'haptic':
#     from networks import alexnet_acl as network
else:
    raise NotImplementedError

########################################################################################################################

def run(args, run_id):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

        # Faster run but not deterministic:
        # torch.backends.cudnn.benchmark = True

        # To get deterministic results_2a that match with paper at cost of lower speed:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Data loader
    print('Instantiate data generators and model...')
    dataloader = datagenerator.DatasetGen(args)
    # args.task_cls, args.inputsize = dataloader.task_cls, dataloader.inputsize
    args.task_cls = dataloader.task_cls
    # if args.experiment == 'multidatasets': args.lrs = dataloader.lrs # multidatasets is not used in this paper

    # Model
    net = network.Net(args)
    net = net.to(args.device)
    net.print_model_size()

    # Approach
    appr=approach(net,args,network=network)              # (network.Net, args, network=network.shared, network=network.private)


    # Loop tasks
    acc=np.zeros((len(args.task_cls),len(args.task_cls)),dtype=np.float32)
    lss=np.zeros((len(args.task_cls),len(args.task_cls)),dtype=np.float32)

    for t, ncls in args.task_cls:
        print('*'*250)
        dataset = dataloader.get(t)
        print(' '*105, 'Dataset {:2d} ({:s})'.format(t+1,dataset[t]['name']))
        print('*'*250)

        # Train
        appr.train(t,dataset[t])  # 将t和dataset[t]传入train函数中
        print('-'*250)
        print()
        for u in range(t+1):
            # Load previous model and replace the shared module with the current one u=0,1,2,3,4,5,6,7,8 t=0,1,2,3,4,5,6,7,8 load the previous model to test---- u=2 t=3 ---- u=3 t=4 ...
            test_model = appr.load_model(u)
            test_rest = appr.test(dataset[u]['test'], u, model=test_model) # 将dataset[u]['test'], u, model=test_model传入test函数中

            print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u, dataset[u]['name'], test_rest['loss_t'], test_rest['acc_t']))
            acc[t, u] = test_rest['acc_t']
            lss[t, u] = test_rest['loss_t']

        # Save
        print()
        print('Saved accuracies at '+os.path.join(args.checkpoint,args.output))
        np.savetxt(os.path.join(args.checkpoint,args.output),acc,'%.6f')

    #Extract embeddings to plot in tensorboard for miniimagenet
    if args.tsne == 'yes' and args.experiment == 'bci-competition-IV2a':
        appr.get_tsne_embeddings_first_ten_tasks(dataset, model=appr.load_model(t))
        appr.get_tsne_embeddings_last_three_tasks(dataset, model=appr.load_model(t))

    avg_acc, gem_bwt = utils.print_log_acc_bwt(args.task_cls, acc, lss, output_path=args.checkpoint, run_id=run_id)

    return avg_acc, gem_bwt

#######################################################################################################################


def main(args):
    utils.make_directories(args)
    utils.save_code(args)

    print('=' * 100)
    print('Arguments =')
    for arg in vars(args):
        print('\t' + arg + ':', getattr(args, arg))
    print('=' * 100)

    accuracies, forgetting = [], []
    for n in range(args.num_runs):
        args.seed = n
        args.output = '{}_{}_tasks_seed_{}.txt'.format(args.experiment, args.num_subjects, args.seed)
        print ("args.output: ", args.output)            # args.output:  bci-competition-IV2a_9_tasks_seed_0.txt

        print (" >>>> Run #", n)
        acc, bwt = run(args, n)    # 将args和n传入run函数中
        accuracies.append(acc)
        forgetting.append(bwt)


    print('*' * 100)
    print ("Average over {} runs: ".format(args.num_runs))
    print ('AVG ACC: {:5.4f}% \pm {:5.4f}'.format(np.array(accuracies).mean(), np.array(accuracies).std()))
    print ('AVG BWT: {:5.2f}% \pm {:5.4f}'.format(np.array(forgetting).mean(), np.array(forgetting).std()))


    print ("All Done! ")
    print('[Elapsed time = {:.1f} min]'.format((time.time()-tstart)/(60)))
    utils.print_time()

# def test_trained_model(args, final_model_id):
#     args.seed = 0
#
#     print('Instantiate data generators and model...')
#     dataloader = datagenerator.DatasetGen(args)
#     args.taskcla, args.inputsize = dataloader.taskcla, dataloader.inputsize
#     # if args.experiment == 'multidatasets': args.lrs = dataloader.lrs
#
#     def get_model(final_model_id, test_data_id):
#         # Load the test model
#         test_net = network.Net(args)
#         checkpoint_test = torch.load(os.path.join(args.checkpoint, 'model_{}.pth.tar'.format(test_data_id)))
#         test_net.load_state_dict(checkpoint_test['model_state_dict'])
#
#         # Load your final trained model
#         net = network.Net(args)
#         checkpoint = torch.load(os.path.join(args.checkpoint, 'model_{}.pth.tar'.format(final_model_id)))
#         net.load_state_dict(checkpoint['model_state_dict'])
#
#         # # Change the shared module with the final model's shared module
#         final_shared = deepcopy(net.shared.state_dict())
#         test_net.shared.load_state_dict(final_shared)
#         test_net = test_net.to(args.device)
#
#         return test_net
#
#     for t,ncla in args.taskcla:
#         print('*'*250)
#         dataset = dataloader.get(t)
#         print(' '*105, 'Dataset {:2d} ({:s})'.format(t+1,dataset[t]['name']))
#         print('*'*250)
#
#         # Model
#         test_model = get_model(final_model_id, test_data_id=t)
#
#         # Approach
#         appr = approach(test_model, args, network=network)
#
#         # Test
#         test_res = appr.inference(dataset[t]['test'], t, model=test_model)
#
#         print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.4f}% <<<'.format(t, dataset[t]['name'],
#                                                                                           test_res['loss_t'],
#                                                                                           test_res['acc_t']))

#######################################################################################################################
if __name__ == '__main__':
    main(args)
    # test_trained_model(args, final_model_id=4)








# # Arguments
# parser = argparse.ArgumentParser(description='Subject/Domain-Incremental Learning for MI-EEG Classification')
# # Load the config file
# parser.add_argument('--config',  type=str, default='./configs/config_bcicomp2b.yml')
# flags = parser.parse_args()
# args = OmegaConf.load(flags.config)

# ########################################################################################################################

# # Args -- Experiment
# if args.experiment=='bci-intuitive':
#     from dataloaders import intuitiveDataset as datagenerator
# elif args.experiment=='bci-competition-IV2a':
#     from dataloaders import bcicomp2a as datagenerator
# elif args.experiment=='bci-competition-IV2b':
#     from dataloaders import bcicomp2b as datagenerator
# else:
#     raise NotImplementedError

# from sil import SIL as approach



# # Args -- Network
# if 'bci' in args.experiment:
#     from networks import mlp_sil_BCI as network
# # elif args.experiment == 'haptic':
# #     from networks import alexnet_sil as network
# else:


#     raise NotImplementedError

# ########################################################################################################################

# def run(args, run_id):
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(args.seed)

#         # Faster run but not deterministic:
#         # torch.backends.cudnn.benchmark = True

#         # To get deterministic results that match with paper at cost of lower speed:
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False

#     # Data loader
#     print('Instantiate data generators and model...')
#     dataloader = datagenerator.DatasetGen(args)
#     # args.task_cls, args.inputsize = dataloader.task_cls, dataloader.inputsize
#     args.task_cls = dataloader.task_cls
#     # if args.experiment == 'multidatasets': args.lrs = dataloader.lrs # multidatasets is not used in this paper

#     # Model
#     net = network.Net(args)
#     net = net.to(args.device)
#     net.print_model_size()

#     # Approach
#     appr=approach(net,args,network=network)              # (network.Net, args, network=network.shared, network=network.private)


#     # Loop tasks
#     acc=np.zeros((len(args.task_cls),len(args.task_cls)),dtype=np.float32)
#     lss=np.zeros((len(args.task_cls),len(args.task_cls)),dtype=np.float32)

#     for t, ncls in args.task_cls:
#         print('*'*250)
#         dataset = dataloader.get(t)
#         print(' '*105, 'Dataset {:2d} ({:s})'.format(t+1,dataset[t]['name']))
#         print('*'*250)

#         # Train
#         appr.train(t,dataset[t])  # 将t和dataset[t]传入train函数中
#         print('-'*250)
#         print()
#         for u in range(t+1):
#             # Load previous model and replace the shared module with the current one u=0,1,2,3,4,5,6,7,8 t=0,1,2,3,4,5,6,7,8 load the previous model to test---- u=2 t=3 ---- u=3 t=4 ...
#             test_model = appr.load_model(u)
#             test_rest = appr.test(dataset[u]['test'], u, model=test_model) # 将dataset[u]['test'], u, model=test_model传入test函数中

#             print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u, dataset[u]['name'], test_rest['loss_t'], test_rest['acc_t']))
#             acc[t, u] = test_rest['acc_t']
#             lss[t, u] = test_rest['loss_t']

#         # Save
#         print()
#         print('Saved accuracies at '+os.path.join(args.checkpoint,args.output))
#         np.savetxt(os.path.join(args.checkpoint,args.output),acc,'%.6f')

#     #Extract embeddings to plot in tensorboard for miniimagenet
#     if args.tsne == 'yes' and args.experiment == 'bci-competition-IV2a':
#         appr.get_tsne_embeddings_first_ten_tasks(dataset, model=appr.load_model(t))
#         appr.get_tsne_embeddings_last_three_tasks(dataset, model=appr.load_model(t))

#     avg_acc, gem_bwt = utils.print_log_acc_bwt(args.task_cls, acc, lss, output_path=args.checkpoint, run_id=run_id)

#     return avg_acc, gem_bwt

# #######################################################################################################################


# def main(args):
#     utils.make_directories(args)
#     utils.save_code(args)

#     print('=' * 100)
#     print('Arguments =')
#     for arg in vars(args):
#         print('\t' + arg + ':', getattr(args, arg))
#     print('=' * 100)

#     accuracies, forgetting = [], []
#     for n in range(args.num_runs):
#         args.seed = n
#         args.output = '{}_{}_tasks_seed_{}.txt'.format(args.experiment, args.num_subjects, args.seed)
#         print ("args.output: ", args.output)            # args.output:  bci-competition-IV2a_9_tasks_seed_0.txt

#         print (" >>>> Run #", n)
#         acc, bwt = run(args, n)    # 将args和n传入run函数中
#         accuracies.append(acc)
#         forgetting.append(bwt)


#     print('*' * 100)
#     print ("Average over {} runs: ".format(args.num_runs))
#     print ('AVG ACC: {:5.4f}% \pm {:5.4f}'.format(np.array(accuracies).mean(), np.array(accuracies).std()))
#     print ('AVG BWT: {:5.2f}% \pm {:5.4f}'.format(np.array(forgetting).mean(), np.array(forgetting).std()))


#     print ("All Done! ")
#     print('[Elapsed time = {:.1f} min]'.format((time.time()-tstart)/(60)))
#     utils.print_time()

# # def test_trained_model(args, final_model_id):
# #     args.seed = 0
# #
# #     print('Instantiate data generators and model...')
# #     dataloader = datagenerator.DatasetGen(args)
# #     args.taskcla, args.inputsize = dataloader.taskcla, dataloader.inputsize
# #     # if args.experiment == 'multidatasets': args.lrs = dataloader.lrs
# #
# #     def get_model(final_model_id, test_data_id):
# #         # Load the test model
# #         test_net = network.Net(args)
# #         checkpoint_test = torch.load(os.path.join(args.checkpoint, 'model_{}.pth.tar'.format(test_data_id)))
# #         test_net.load_state_dict(checkpoint_test['model_state_dict'])
# #
# #         # Load your final trained model
# #         net = network.Net(args)
# #         checkpoint = torch.load(os.path.join(args.checkpoint, 'model_{}.pth.tar'.format(final_model_id)))
# #         net.load_state_dict(checkpoint['model_state_dict'])
# #
# #         # # Change the shared module with the final model's shared module
# #         final_shared = deepcopy(net.shared.state_dict())
# #         test_net.shared.load_state_dict(final_shared)
# #         test_net = test_net.to(args.device)
# #
# #         return test_net
# #
# #     for t,ncla in args.taskcla:
# #         print('*'*250)
# #         dataset = dataloader.get(t)
# #         print(' '*105, 'Dataset {:2d} ({:s})'.format(t+1,dataset[t]['name']))
# #         print('*'*250)
# #
# #         # Model
# #         test_model = get_model(final_model_id, test_data_id=t)
# #
# #         # Approach
# #         appr = approach(test_model, args, network=network)
# #
# #         # Test
# #         test_res = appr.inference(dataset[t]['test'], t, model=test_model)
# #
# #         print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.4f}% <<<'.format(t, dataset[t]['name'],
# #                                                                                           test_res['loss_t'],
# #                                                                                           test_res['acc_t']))

# #######################################################################################################################
# if __name__ == '__main__':
#     main(args)
#     # test_trained_model(args, final_model_id=4)

