import sys
import os
sys.path.append(os.path.dirname(__file__))

from base import Trainer
import os.path as osp
import torch.nn as nn
from copy import deepcopy
import datetime
import helper
from supcon import *
import utils
import dataloader.data_utils as data_utils
# from Network import ORCONET
from Network_vit import ORCONET_ViT
from Network_vit_inc import ORCONET_ViT_Inc

import random

import time
from pathlib import Path
from collections import OrderedDict
import yaml

from tqdm import tqdm

def save_args_to_yaml(args, output_file):
    args_dict = OrderedDict(vars(args))
    with open(output_file, 'w') as yaml_file:
        yaml.dump(args_dict, yaml_file)

class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()                                                                            # Setting logs and artefact paths.
        self.args = data_utils.set_up_datasets(self.args)                                               # Data wrapper inside the args object, passed throughout this file
                
        self.model = ORCONET_ViT(self.args, mode=self.args.base_mode)
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))                        
        self.model = self.model.cuda()

        self.best_model_dict = {}
        if self.args.model_dir is not None:                                                             # Loading pretrained model, Note that for CUB we use an imagenet pretrained model            
            state_dict = torch.load(self.args.model_dir)["state_dict"]
            # Adapt keys to match network
            for k,v in state_dict.items():
                if "backbone" in k:
                    self.best_model_dict[k.replace("backbone", "module.encoder")] = v
                if "projector" in k:
                    self.best_model_dict["module." + k] = v

    def get_dataloader(self, session):
        if session == 0:
            trainset, trainloader, testloader = data_utils.get_base_dataloader(self.args)
        else:
            trainset, trainloader, testloader = data_utils.get_new_dataloader(self.args, session)
        return trainset, trainloader, testloader

    def train_phase1(self):
        
        trainset, base_trainloader, base_testloader = data_utils.get_supcon_base_dataloader(self.args, pretrain=True)
                
        self.model.module.pretrain_base(base_trainloader)
        
    
    def train_phase2(self):
        """
            Base Alignment Phase: We aim to align the base dataset $D^0$ to the pseudo-targets through our OrCo loss. 
        """
        base_set, _, base_testloader = self.get_dataloader(0)
        save_model_dir = os.path.join(self.args.output_dir, 'session0_max_acc.pth')
                
        if len(self.best_model_dict):
            self.model.load_state_dict(self.best_model_dict, strict=False)
        
        save_mab_path = os.path.join(self.args.output_dir, 'best_pretrained_mab.pth')
        save_projector_path = os.path.join(self.args.output_dir, 'best_pretrained_projector.pth')
        
        if os.path.exists(save_mab_path) and os.path.exists(save_projector_path):
            mab_checkpoint = torch.load(save_mab_path)
            projector_checkpoint = torch.load(save_projector_path)
            self.model.module.mab.load_state_dict(mab_checkpoint, strict=False)
            self.model.module.projector.load_state_dict(projector_checkpoint, strict=False)
                         
        # Compute the Mean Prototypes (Indicated as \mu_j in the paper) from the projection head
        best_prototypes = helper.get_base_prototypes(base_set, base_testloader.dataset.transform, self.model, self.args)        
        
        print("===Compute Pseudo-Targets and Class Assignment===")
        self.model.module.fc.find_reseverve_vectors_all()

        print("===[Phase-2] Started!===")
        self.model.module.fc.assign_base_classifier(best_prototypes)        # Assign classes to the optimal pseudo target
       
        
        _, sup_trainloader, _ = data_utils.get_supcon_base_dataloader(self.args)
        self.model.module.update_base(sup_trainloader, base_testloader)

        # Save Phase-1 model
        # torch.save(dict(params=self.model.state_dict()), save_model_dir)
        
        torch.save(
            {
                'model_state_dict': self.model.state_dict(),  # 保存模型参数
                'random_state': torch.get_rng_state(),       # 保存CPU的随机数生成器状态
                'cuda_random_state': torch.cuda.get_rng_state_all(),  # 保存GPU的随机数生成器状态
                'numpy_random_state': np.random.get_state(), # 保存NumPy的随机数生成器状态
                'python_random_state': random.getstate(),    # 保存Python的随机数生成器状态
            },
            save_model_dir
        )
        
        self.best_model_dict = deepcopy(self.model.state_dict())

        # # Compute Phase-2 Accuracies
        out = helper.test(self.model, base_testloader, 0, self.args, 0)
        best_va = out[0]

        # Log the Phase-2 Accuracies
        print(f"[Phase-2] Accuracy: {best_va*100:.3f}")
        self.trlog['max_acc'][0] = float('%.3f' % (best_va * 100))

    def train(self):
        t_start_time = time.time()

        # Train Stats
        if self.args.base_aug or self.args.inc_aug:
            self.args.output_dir = os.path.join(self.args.save_path, datetime.datetime.now().__format__('%m-%d-%H-%M-%S') + \
                                            f'base_aug_{self.args.base_num_aug}_delta_base_{self.args.max_delta_base}_'
                                            f'inc_aug_{self.args.inc_num_aug}_delta_inc_{self.args.max_delta_inc}'
                                            )
        
        else:
            self.args.output_dir = os.path.join(self.args.save_path, datetime.datetime.now().__format__('%m-%d-%H-%M-%S') + \
                                            'no_augmentation'
                                            )
        
        Path(self.args.output_dir).mkdir(parents=True, exist_ok=True)
        config_path = os.path.join(self.args.output_dir, 'config.yaml')
        save_args_to_yaml(self.args, config_path)
        
        result_list = [self.args]
        
        self.train_phase1()
        
        # # Base Alignment (Phase 2)
        self.train_phase2()

        # Few-Shot Alignment (Phase 3)
        for session in range(1, self.args.sessions):
        # for session in range(1, 2):
            # Load base model/previous incremental session model
            self.model.load_state_dict(self.best_model_dict, strict = True)

            # Load data for this session
            train_set, trainloader, testloader = self.get_dataloader(session)

            print("\n\n===[Phase-3][Session-%d] Started!===" % session)
            # self.model.eval() # Following CEC

            # # Assignment            
            self.model.module.update_targets(trainloader, testloader, np.unique(train_set.targets), session-1)
                        
            # Alignment
            _, sessionloader = data_utils.get_supcon_replay_dataloader(self.args, session)
            self.model.module.update_incremental(sessionloader, session)

            # Compute scores
            tsa, novel_cw, base_cw = helper.test(self.model, testloader, 0, self.args, session)

            # Save Accuracies and Means
            self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
            self.trlog['max_novel_acc'][session] = float('%.3f' % (novel_cw * 100))
            self.trlog['max_base_acc'][session] = float('%.3f' % (base_cw * 100))
            self.trlog["max_hm"][session] = float('%.3f' % (utils.hm(base_cw, novel_cw) * 100))

            # Save the final model
            # save_model_dir = os.path.join(self.args.save_path, 'session' + str(session) + '_max_acc.pth')
            save_model_dir = os.path.join(self.args.output_dir, 'session' + str(session) + '_max_acc.pth')
            # torch.save(dict(params=self.model.state_dict()), save_model_dir)
            
            torch.save(
                {
                    'model_state_dict': self.model.state_dict(),  
                    'random_state': torch.get_rng_state(),       
                    'cuda_random_state': torch.cuda.get_rng_state_all(), 
                    'numpy_random_state': np.random.get_state(), 
                    'python_random_state': random.getstate(),   
                },
                save_model_dir
            )
            
            
            self.best_model_dict = deepcopy(self.model.state_dict())
            print('===[Phase-3][Session-%d] Saving model to :%s===' % (session, save_model_dir))

            out_string = 'Session {}, test Acc {:.3f}, test_novel_acc {:.3f}, test_base_acc {:.3f}, hm {:.3f}'\
                .format(
                    session, 
                    self.trlog['max_acc'][session], 
                    self.trlog['max_novel_acc'][session], 
                    self.trlog['max_base_acc'][session], 
                    self.trlog["max_hm"][session]
                    )
            print(out_string)

            result_list.append(out_string)
        
        self.exit_log(result_list)

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        print('Total time used %.3f mins' % total_time)

    def exit_log(self, result_list):
        # Remove the firsat dummy harmonic mean value
        del self.trlog['max_hm'][0]
        del self.trlog['max_hm_cw'][0]

        result_list.append('Base Session Best Epoch {}\n'.format(self.trlog['max_acc_epoch']))

        result_list.append("Top 1 Accuracy: ")
        result_list.append(self.trlog['max_acc'])
        
        result_list.append("Harmonic Mean: ")
        result_list.append(self.trlog['max_hm'])

        result_list.append("Base Test Accuracy: ")
        result_list.append(self.trlog['max_base_acc'])

        result_list.append("Novel Test Accuracy: ")
        result_list.append(self.trlog['max_novel_acc'])

        average_harmonic_mean = np.array(self.trlog['max_hm']).mean()
        result_list.append("Average Harmonic Mean Accuracy: ")
        result_list.append(average_harmonic_mean)

        average_acc = np.array(self.trlog['max_acc']).mean()
        result_list.append("Average Accuracy: ")
        result_list.append(average_acc)

        performance_decay = self.trlog['max_acc'][0] - self.trlog['max_acc'][-1]
        result_list.append("Performance Decay: ")
        result_list.append(performance_decay)

        print(f"\n\nacc: {self.trlog['max_acc']}")
        print(f"avg_acc: {average_acc:.3f}")
        print(f"hm: {self.trlog['max_hm']}")
        print(f"avg_hm: {average_harmonic_mean:.3f}")
        print(f"pd: {performance_decay:.3f}")
        print(f"base: {self.trlog['max_base_acc']}")
        print(f"novel: {self.trlog['max_novel_acc']}")    
        utils.save_list_to_txt(os.path.join(self.args.output_dir, 'results.txt'), result_list)


    def set_save_path(self):
        self.args.save_path = '%s/' % self.args.dataset                      
        self.args.save_path = self.args.save_path + '%s/' % self.args.project

        # Appending a user defined prefix to the folder.
        if self.args.save_path_prefix:
            self.args.save_path = self.args.save_path + self.args.save_path_prefix 

        self.args.save_path = os.path.join('checkpoint_wd0.0', self.args.save_path)
        utils.ensure_path(self.args.save_path)
    
    
    def train_incremental(self):
        t_start_time = time.time()

        # Train Stats
        if self.args.base_aug or self.args.inc_aug:
            self.args.output_dir = os.path.join(self.args.save_path, datetime.datetime.now().__format__('%m-%d-%H-%M-%S') + \
                                            f'base_aug_{self.args.base_num_aug}_delta_base_{self.args.max_delta_base}_'
                                            f'inc_aug_{self.args.inc_num_aug}_delta_inc_{self.args.max_delta_inc}'
                                            )
        
        else:
            self.args.output_dir = os.path.join(self.args.save_path, datetime.datetime.now().__format__('%m-%d-%H-%M-%S') + \
                                            'no_augmentation'
                                            ) 
        
        Path(self.args.output_dir).mkdir(parents=True, exist_ok=True)
        config_path = os.path.join(self.args.output_dir, 'config.yaml')
        save_args_to_yaml(self.args, config_path)
        
        result_list = [self.args]
                
        self.model = ORCONET_ViT_Inc(self.args, mode=self.args.base_mode)
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))                        
        self.model = self.model.cuda()
        
        base_model_path = os.path.join(self.args.base_model_dir, 'session0_max_acc.pth')
        # self.model.load_state_dict(torch.load(base_model_path)["params"], strict=True)
        # self.model.load_state_dict(torch.load(base_model_path)["params"], strict=False)
        
        checkpoint = torch.load(base_model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        torch.set_rng_state(checkpoint['random_state'])        
        torch.cuda.set_rng_state_all(checkpoint['cuda_random_state']) 
        np.random.set_state(checkpoint['numpy_random_state'])        
        random.setstate(checkpoint['python_random_state'])
        
        self.best_model_dict = deepcopy(self.model.state_dict())
                
        base_set, _, base_testloader = self.get_dataloader(0)
        out = helper.test(self.model, base_testloader, 0, self.args, 0)
        best_va = out[0]

        # Log the Phase-2 Accuracies
        print(f"[Phase-2] Accuracy: {best_va*100:.3f}")
        self.trlog['max_acc'][0] = float('%.3f' % (best_va * 100))

        # Few-Shot Alignment (Phase 3)
        for session in range(1, self.args.sessions):
        # for session in range(1, 2):
            # Load base model/previous incremental session model
            self.model.load_state_dict(self.best_model_dict, strict = True)

            # Load data for this session
            train_set, trainloader, testloader = self.get_dataloader(session)

            print("\n\n===[Phase-3][Session-%d] Started!===" % session)
            # self.model.eval() # Following CEC

            # Assignment
            self.model.module.update_targets(trainloader, testloader, np.unique(train_set.targets), session-1)

            # Alignment
            #_, jointloader = data_utils.get_supcon_joint_dataloader(self.args, session)
            _, sessionloader = data_utils.get_supcon_replay_dataloader(self.args, session)
            
            self.model.module.update_incremental(sessionloader, session)
            # self.model.module.update_incremental_original(sessionloader, session)
            
            # print(self.model.module.projector.omega_w_fc1)
            # print(self.model.module.projector.omega_w_fc2)

            # Compute scores
            tsa, novel_cw, base_cw = helper.test(self.model, testloader, 0, self.args, session)

            # Save Accuracies and Means
            self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
            self.trlog['max_novel_acc'][session] = float('%.3f' % (novel_cw * 100))
            self.trlog['max_base_acc'][session] = float('%.3f' % (base_cw * 100))
            self.trlog["max_hm"][session] = float('%.3f' % (utils.hm(base_cw, novel_cw) * 100))

            # Save the final model
            # save_model_dir = os.path.join(self.args.save_path, 'session' + str(session) + '_max_acc.pth')
            save_model_dir = os.path.join(self.args.output_dir, 'session' + str(session) + '_max_acc.pth')
            # torch.save(dict(params=self.model.state_dict()), save_model_dir)
            torch.save(
                {
                    'model_state_dict': self.model.state_dict(),  # 保存模型参数
                    'random_state': torch.get_rng_state(),       # 保存CPU的随机数生成器状态
                    'cuda_random_state': torch.cuda.get_rng_state_all(),  # 保存GPU的随机数生成器状态
                    'numpy_random_state': np.random.get_state(), # 保存NumPy的随机数生成器状态
                    'python_random_state': random.getstate(),    # 保存Python的随机数生成器状态
                },
                save_model_dir
            )
            
            self.best_model_dict = deepcopy(self.model.state_dict())
            print('===[Phase-3][Session-%d] Saving model to :%s===' % (session, save_model_dir))

            out_string = 'Session {}, test Acc {:.3f}, test_novel_acc {:.3f}, test_base_acc {:.3f}, hm {:.3f}'\
                .format(
                    session, 
                    self.trlog['max_acc'][session], 
                    self.trlog['max_novel_acc'][session], 
                    self.trlog['max_base_acc'][session], 
                    self.trlog["max_hm"][session]
                    )
            print(out_string)

            result_list.append(out_string)
        
        self.exit_log(result_list)

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        print('Total time used %.3f mins' % total_time)
        
    
    def test(self):
        # Load the best model
        
        output_file_path = os.path.join(self.args.output_dir, "session_acc.txt")
        
        for session in range(1, self.args.sessions):
            self.args.cur_session = session
            self.model = ORCONET_ViT_Inc(self.args, mode=self.args.base_mode)
            self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))                        
            self.model = self.model.cuda()
            
            current_model_state_dict = torch.load(osp.join(self.args.output_dir, f'session{session}_max_acc.pth'))['model_state_dict']
            self.model.load_state_dict(current_model_state_dict)
            
            _, _, testloader = self.get_dataloader(session)
            
            tsa, novel_cw, base_cw, session_acc = helper.test_v2(self.model, testloader, 0, self.args, session)
            
            print(f'Session {session} - Top-1 Acc: {tsa*100:.3f}, Novel Acc: {novel_cw*100:.3f}, Base Acc: {base_cw*100:.3f}')
            print("Session Accuracy: ", session_acc)
            
            with open(output_file_path, 'a') as f:
                f.write(f'Session {session} - Top-1 Acc: {tsa*100:.3f}, Novel Acc: {novel_cw*100:.3f}, Base Acc: {base_cw*100:.3f}\n')
                f.write(f"Session Accuracy: {session_acc}\n")
    
    @torch.no_grad()
    def save_features_for_tsne(self):
        self.model.eval()
        
        for session in range(0, self.args.sessions):
            
            self.args.cur_session = session
            self.model = ORCONET_ViT_Inc(self.args, mode=self.args.base_mode)
            self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))                        
            self.model = self.model.cuda()
            
            current_model_state_dict = torch.load(osp.join(self.args.output_dir, f'session{session}_max_acc.pth'))['params']
            self.model.load_state_dict(current_model_state_dict)
            
            _, sup_trainloader, _ = data_utils.get_supcon_base_dataloader(self.args, session=session)
        
            self.model.module.store_prototype(sup_trainloader, session=session, args=self.args)
            
            train_features_all = []
            train_labels_all = [] 
            replay_features_all = []
            replay_labels_all = []
            
            for idx, batch in enumerate(tqdm(sup_trainloader, desc="Processing Batches")):
                    images, label = batch
                    images = torch.cat([images[0], images[1]], dim=0).cuda()
                    train_label = label.cuda()
                    
                    train_features = self.model.module.encode(images, aug=self.args.base_aug, pretrain=True, store=True)
                    
                    # if self.args.base_aug:
                    #     nviews = 2 * (self.args.pretrain_num_aug)
                    # else:
                    #     nviews = 2
                    
                    train_nviews = 2 * (self.args.pretrain_num_aug)
                    replay_nviews = 2 * (self.args.num_augmentations)
                        
                    train_label = train_label.repeat(train_nviews)
                    
                        
                    # sample_label = self.args.base_class + (self.args.way * (session-1))
                    sample_label = self.args.base_class
                    replay_features, replay_labels = self.model.module.calculator.sample_groups(num_classes = sample_label, num_replay=self.args.num_replay, nview = replay_nviews)
                    
                    replay_features = torch.cat(replay_features, dim=0).cuda()
                    replay_labels = torch.cat(replay_labels, dim=0).cuda()
                    
                    train_features_all.append(train_features.data.cpu().numpy())
                    train_labels_all.append(train_label.data.cpu().numpy())
                    replay_features_all.append(replay_features.data.cpu().numpy())
                    replay_labels_all.append(replay_labels.data.cpu().numpy())
            
            train_features_all = np.concatenate(train_features_all, axis=0)
            train_labels_all = np.concatenate(train_labels_all, axis=0)
            replay_features_all = np.concatenate(replay_features_all, axis=0)
            replay_labels_all = np.concatenate(replay_labels_all, axis=0)
            
            np.save(os.path.join(self.args.output_dir, 'features_labels.npy'),
                    arr={
                        'train_features': train_features_all,
                        'train_labels': train_labels_all,
                        'replay_features': replay_features_all,
                        'replay_labels': replay_labels_all,
                    })
            
    def plot_tsne(self):
        
        from sklearn.manifold import TSNE
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        
        save_path = osp.join(self.args.output_dir, 'features_labels.npy')
        
        data = np.load(save_path, allow_pickle=True).item()
        
        train_features_all = data['train_features']
        train_labels_all = data['train_labels']
        replay_features_all = data['replay_features']
        replay_labels_all = data['replay_labels']
        
        class_list = np.arange(60)
        
        for cls in class_list:
            train_index = np.where(train_labels_all == cls)[0]
            replay_index = np.where(replay_labels_all == cls)[0]
            
            train_features = train_features_all[train_index]
            replay_features = replay_features_all[replay_index]
            
            train_labels = train_labels_all[train_index]
            replay_labels = replay_labels_all[replay_index]
            
            
            # 将两组特征联合
            all_features = np.concatenate([train_features, replay_features], axis=0)
            all_labels = np.concatenate([train_labels, replay_labels], axis=0)

            # 使用 t-SNE 进行联合降维
            all_features_embedded = TSNE(
                n_components=2,
                perplexity=30,
                early_exaggeration=1,
                learning_rate='auto',
                verbose=1
            ).fit_transform(all_features)

            # 分离降维后的结果
            train_features_embedded = all_features_embedded[:len(train_features)]
            replay_features_embedded = all_features_embedded[len(train_features):]
            
            plt.figure(figsize=(10, 10))

                        # 绘制训练特征点
            plt.scatter(
                x=train_features_embedded[:, 0],
                y=train_features_embedded[:, 1],
                c='blue',  # 使用固定颜色蓝色
                marker='o',
                edgecolors='black',
                linewidths=1,
                alpha=0.8,
                s=20,
                label='Train Features'
            )

            # 绘制 Replay 特征点
            plt.scatter(
                x=replay_features_embedded[:, 0],
                y=replay_features_embedded[:, 1],
                c='red',  # 使用固定颜色红色
                marker='o',
                edgecolors='black',
                linewidths=1,
                alpha=0.8,
                s=20,
                label='Replay Features'
            )

            # 设置图例、样式和保存
            plt.legend()
            plt.axis('off')
            plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
            plt.savefig(f'tsne_train_and_replay_features_class{cls}.png')
            # plt.show()
    
    def test_rank(self):
        
        output_path = 'checkpoint/cifar100/orco_vit/noaug/11-21-22-36-41_pretrained_lr_0.1_prbs_128_base_lr_0.1_bbs_128_new_lr_0.1_incbs_0_num_aug_2_delta_base_0.5_delta_inc_2.0'
        
        model_state_dict = torch.load(osp.join(output_path, f'session{8}_max_acc.pth'))['params']
        # session_1_model_state_dict = torch.load(osp.join(output_path, f'session{1}_max_acc.pth'))['params']
        # session_2_model_state_dict = torch.load(osp.join(output_path, f'session{2}_max_acc.pth'))['params']
        
        delta_w_fc1 = model_state_dict['module.projector.delta_w_fc1']
        # session_1_fc1_weight = session_1_model_state_dict['module.projector.delta_w_fc1']
        # session_2_fc1_weight = session_2_model_state_dict['module.projector.delta_w_fc1']
        
        delta_W_0 = delta_w_fc1[0]
        delta_W_1 = delta_w_fc1[1]
        delta_W_2 = delta_w_fc1[2]
        
        rank_0 = torch.linalg.matrix_rank(delta_W_0)
        rank_1 = torch.linalg.matrix_rank(delta_W_1)
        
        print(rank_0)
        print(rank_1)
        
        ...