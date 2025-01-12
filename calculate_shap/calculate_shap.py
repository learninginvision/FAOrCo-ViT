import argparse
import importlib
from utils import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_command_line_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-project', type=str, default="orco")
    parser.add_argument('-dataset', type=str, default='cifar100',
                        choices=['mini_imagenet', 'cub200', 'cifar100'])
    parser.add_argument('-dataroot', type=str, default="../datasets/")
    parser.add_argument('-save_path_prefix', "-prefix", type=str, default="")

    parser.add_argument('-gpu', default='0')
    parser.add_argument('-num_workers', type=int, default=8)
    parser.add_argument('-seed', type=int, default=1)
    parser.add_argument('-debug', action='store_true')
    parser.add_argument('-resume', action='store_true')
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-gamma', type=float, default=0.1)
    parser.add_argument('-temperature', type=int, default=16)

    parser.add_argument('-start_session', type=int, default=1)
    parser.add_argument('-model_dir', type=str, help='loading model parameter from a specific dir')

    parser.add_argument('-epochs_base', type=int, default=100)
    parser.add_argument('-epochs_joint', type=int, default=100)
    parser.add_argument('-epochs_target_gen', type=int, default=1000)
    parser.add_argument('-lr_base', type=float, default=0.1)
    parser.add_argument('-lr_base_encoder', type=float, default=0.1)
    parser.add_argument('-lr_new', type=float, default=0.1) 
    parser.add_argument('-optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam', 'adamw', "mtadam"])
    parser.add_argument('-optimizer_joint', type=str, default='sgd',
                        choices=['sgd', 'adam', 'adamw', "mtadam"])    
    parser.add_argument('-schedule', type=str, default='Step',
                        choices=['Step', 'Milestone', 'Cosine'])

    parser.add_argument('-reserve_mode', type=str, default='all', 
                        choices=["all", "full"]) 

    parser.add_argument('-joint_schedule', type=str, default='Milestone',
                        choices=['Milestone', 'Cosine'])
    parser.add_argument('-base_schedule', type=str, default='Milestone',
                        choices=['Milestone', 'Cosine'])

    parser.add_argument('-decay', type=float, default=0.0005)
    parser.add_argument('-decay_new', type=float, default=0) 

    parser.add_argument('-cos_n_lam', type=float, default=0.5) 
    parser.add_argument('-cos_b_lam', type=float, default=0.0) 

    parser.add_argument('-sup_lam', type=float, default=1)
    parser.add_argument('-cos_lam', type=float, default=1)
    parser.add_argument('-simplex_lam', type=float, default=1)

    parser.add_argument('-perturb_epsilon_base', type=float, default=9e-05)
    parser.add_argument('-perturb_epsilon_inc', type=float, default=9e-05)
    parser.add_argument('-perturb_offset', type=float, default=0.5)

    parser.add_argument('-base_mode', type=str, default='ft_dot',
                        choices=['ft_dot', 'ft_cos', 'ft_l2', "ft_dot_freeze"]) 
    parser.add_argument('-fine_tune_backbone_base', action='store_true', help='')
    parser.add_argument("-proj_type", type=str, default="proj",
                        choices=["proj", "proj_ncfscil"])

    parser.add_argument('-batch_size_pretrain', type=int, default=128)
    parser.add_argument('-batch_size_base', type=int, default=128)
    parser.add_argument('-batch_size_supcon_base', type=int, default=128)
    # parser.add_argument('-batch_size_new', type=int, default=0, help='set 0 will use all the availiable training image for new')
    parser.add_argument('-batch_size_replay', type=int, default=0, help='set 0 will use all the availiable training image for new')
    parser.add_argument('-batch_size_test', type=int, default=10) 
    
    parser.add_argument('-drop_last_batch', action="store_true", help="Drops the last batch if not equal to the assigned batch size")
    parser.add_argument('-exemplars_count', type=int, default=-1)
    
    parser.add_argument('-rand_aug_sup_con', action='store_true', help='')
    parser.add_argument('-prob_color_jitter', type=float, default=0.8)
    parser.add_argument('-min_crop_scale', type=float, default=0.2)

    parser.add_argument('-warmup_epochs_base', type=int, default=3)
    parser.add_argument('-warmup_epochs_inc', type=int, default=10)
    
    # --pretrained params ---
    parser.add_argument('-pretrain_epochs', type=int, default=50)
    parser.add_argument('-pretrain_schedule', type=str, default='Cosine',
                        choices=['Milestone', 'Cosine'])
    parser.add_argument('-pretrain_epochs_max', type=int, default=100)
    parser.add_argument('-warmup_epochs_pretrain', type=int, default=3)
    parser.add_argument('-pretrain_lr', type=float, default=0.05)
    parser.add_argument('-pretrain_optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam', 'adamw', "mtadam"])
    parser.add_argument('-alpha', type=float, default=0.2)
    
    
    # --MAB params ---
    parser.add_argument('-model', type=str, default='vit_base_patch16_224_dino', help='Name of model to train')
    parser.add_argument('-pretrained', default=True, help='Load pretrained model or not')
    parser.add_argument('-encoder_outdim', type=int, default=768)
    parser.add_argument('-num_heads', type=int, default=8)
    parser.add_argument('-proj_hidden_dim', type=int, default=2048)
    parser.add_argument('-proj_output_dim', type=int, default=128)
    
    # --augmentation params ---
    parser.add_argument('-base_aug', action='store_true')
    parser.add_argument('-inc_aug', action='store_true')
    parser.add_argument('-max_delta_base', type=float, default=0.1)
    parser.add_argument('-max_delta_inc', type=float, default=1.0)
    parser.add_argument('-base_num_aug', type=int, default=1)
    parser.add_argument('-inc_num_aug', type=int, default=1)
    parser.add_argument('-include_original', action='store_true')
    
    # --replay params ---
    parser.add_argument('-num_replay', type=int, default=5)
    
    parser.add_argument('-base_model_dir', type=str)
    parser.add_argument('-output_dir', type=str)
    
    parser.add_argument('-stop_task1', action='store_true')
    parser.add_argument('-simple_aug', action='store_true')
    
    parser.add_argument('-cur_session', type=int, default=1)
    parser.add_argument('-baseline', action='store_true')
    
    
    
    
    return parser

if __name__ == '__main__':
    from cifar import CIFAR100  
    from gradientshap import Gradient
    import torch.nn as nn
    import os.path as osp
    from tqdm import tqdm
    from Network_vit_inc import ORCONET_ViT_Inc
    
    # Parse Arguments
    parser = get_command_line_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    args.num_gpu = set_gpu(args)
    
    args.base_class = 60
    args.num_classes = 100
    args.way = 5
    args.shot = 5
    args.sessions = 9
    # /home/lilipan/ling/FSCIL/orco_vit_v4/checkpoint/mini_imagenet/orco_vit/new_multi_Gauss_resize_noaug/11-12-22-07-44_pretrained_lr_0.01_prbs_128_base_lr_0.05_bbs_128_new_lr_0.05_incbs_0_num_aug_2_delta_base_0.5_delta_inc_2.0/session0_max_acc.pth
    # args.output_dir = "/home/lilipan/ling/FSCIL/orco_vitt_cifar100_stoptask1/baseline_and_best/baseline/"

    # for session in range(args.sessions):
    
    model = ORCONET_ViT_Inc(args, mode=args.base_mode)
    model = nn.DataParallel(model, list(range(args.num_gpu)))                        
    model = model.cuda()
    
    current_model_state_dict = torch.load(osp.join(args.output_dir, f'session{args.cur_session}_max_acc.pth'))['model_state_dict']
    model.load_state_dict(current_model_state_dict)

        
    class_list = np.arange(args.base_class + args.way * args.cur_session)
    
    for cls in tqdm(class_list, desc=f'Calcualte SHAP in Seession {args.cur_session}'):
        
        testset = CIFAR100(root=args.dataroot, train=False, cal_shap=True, index=cls)
        testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.batch_size_test, shuffle=False, 
                                            num_workers=0, pin_memory=True)

        model.eval()
        tqdm_gen = tqdm(testloader, desc=f'Calcualte SHAP in class {cls}')
        concatenated_shap_values_array = np.empty((0,3,224,224))
        for i, batch in enumerate(tqdm_gen):
            data, test_label = [_.cuda() for _ in batch]
            # data_numpy = data.cpu().numpy()
            # logits = model(data.reshape(1,-1))
            selected_classes = [cls]
            e = Gradient(model, data)
            # e = shap.SamplingExplainer(model,data_numpy.reshape(data_numpy.shape[0], -1)) 
            shap_values, index = e.shap_values(data, nsamples=50, ranked_outputs=selected_classes, output_rank_order="custom")#用所有的样本计算shapley value
            
            shap_values_array = np.array(shap_values).squeeze()
            concatenated_shap_values_array = np.concatenate((concatenated_shap_values_array, shap_values_array), axis=0)
            # print(shap_values_array)
            
        if args.baseline:       
            shap_save_path = f'calculate_shap/shapley_value/noaug/model_{args.cur_session}/'
        else:
            shap_save_path = f'calculate_shap/shapley_value/aug/model_{args.cur_session}/'
        if not os.path.exists(shap_save_path):
            os.makedirs(shap_save_path)
        np.save(f'{shap_save_path}/class{cls}_shap_values.npy', concatenated_shap_values_array)
