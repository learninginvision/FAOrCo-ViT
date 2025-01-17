import argparse
import importlib
from utils import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_command_line_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-project', type=str, default="orco")
    parser.add_argument('-dataset', type=str, default='mini_imagenet',
                        choices=['mini_imagenet', 'cub200', 'cifar100'])
    parser.add_argument('-dataroot', type=str, default="../datasets")
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
    parser.add_argument('-batch_size_test', type=int, default=100) 
    
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
    parser.add_argument('-model', type=str, default='vit_base_patch16_224_mocov3', help='Name of model to train')
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
    parser.add_argument('-inc_num_aug', type=int, default=2)
    parser.add_argument('-include_original', action='store_true')
    
    # --replay params ---
    parser.add_argument('-num_replay', type=int, default=5)
    
    parser.add_argument('-base_model_dir', type=str)
    parser.add_argument('-output_dir', type=str)
    
    
    
    return parser

if __name__ == '__main__':
    # Parse Arguments
    parser = get_command_line_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    args.num_gpu = set_gpu(args)
    
    # Trainer initialization
    trainer = importlib.import_module('models.%s.fscil_trainer' % (args.project)).FSCILTrainer(args)
    trainer.test()