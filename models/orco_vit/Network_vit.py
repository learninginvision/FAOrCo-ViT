import sys
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch.utils
sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F
from scheduler.lr_scheduler import LinearWarmupCosineAnnealingLR
import supcon


from helper import *
from utils import *
from copy import deepcopy

from tqdm import tqdm
import dataloader.data_utils as data_utils
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment

from timm.models import create_model
import models.vision_transformer
from MAB import MultiAttentionBlock, MultiAttentionBlock_V2
# from projector import Projector
from projector import Projector_delta_W, Porjector_complex, Projector

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from feature_aug import brightness_augmentation, brightness_augmentation_v2, brightness_augmentation_v3

from torch.utils.tensorboard import SummaryWriter

# from torch.optim.lr_scheduler import MultiStepLR

class PseudoTargetClassifier(nn.Module):
    def __init__(self, args, num_features):
        super().__init__()
        
        self.args = args
        self.num_features = num_features        # Input dimension for all classifiers

        # Classifier for the base classes
        self.base_fc = nn.Linear(self.num_features, self.args.base_class, bias=False)       # Note the entire number of classes are already added

        # Set of all classifiers
        self.classifiers = nn.Sequential(self.base_fc)
                
        # Register buffer for the pseudo targets. Assume the total number of classes
        self.num_classes = self.args.num_classes
        self.n_inc_classes = self.args.num_classes - self.args.base_class

        # Number of generated pseudo targets
        if self.args.reserve_mode in ["all"]:
            self.reserve_vector_count = self.num_classes
        elif self.args.reserve_mode in ["full"]:
            self.reserve_vector_count = self.num_features

        # Storing the generated pseudo targets (reserved vectors)
        self.register_buffer("rv", torch.randn(self.reserve_vector_count, self.num_features))

        self.temperature = 1.0

    def compute_angles(self, vectors):
        proto = vectors.cpu().numpy()
        dot = np.matmul(proto, proto.T)
        dot = dot.clip(min=0, max=1)
        theta = np.arccos(dot)
        np.fill_diagonal(theta, np.nan)
        theta = theta[~np.isnan(theta)].reshape(theta.shape[0], theta.shape[1] - 1)
        
        avg_angle_close = theta.min(axis = 1).mean()
        avg_angle = theta.mean()

        return np.rad2deg(avg_angle), np.rad2deg(avg_angle_close)

    def get_assignment(self, cost):
        """Tak array with cosine scores and return the output col ind """
        _, col_ind = linear_sum_assignment(cost, maximize = True)
        return col_ind

    def get_classifier_weights(self, uptil = -1):
        output = []
        for i, cls in enumerate(self.classifiers.children()):
            if uptil >= 0 and uptil < i + 1:
                break
            output.append(cls.weight.data)
        return torch.cat(output, axis = 0)

    def assign_base_classifier(self, base_prototypes):
        # Normalise incoming prototypes
        base_prototypes = normalize(base_prototypes)
        target_choice_ix = self.reserve_vector_count

        cost = cosine_similarity(base_prototypes.cpu(), self.rv.cpu()[:target_choice_ix])
        col_ind = self.get_assignment(cost)
        new_fc_tensor = self.rv[col_ind]

        # Create fixed linear layer
        self.classifiers[0].weight.data = new_fc_tensor

        # Remove from the final rv
        all_idx = np.arange(self.rv.shape[0])
        self.rv = self.rv[all_idx[~np.isin(all_idx, col_ind)]]

    def assign_novel_classifier(self, new_prototypes):  
        # Normalise incoming prototypes
        new_prototypes = normalize(new_prototypes)
        target_choice_ix = self.reserve_vector_count

        cost = cosine_similarity(new_prototypes.cpu(), self.rv.cpu()[:target_choice_ix])
        col_ind = self.get_assignment(cost)
        new_fc_tensor = self.rv[col_ind]

        # Creating and appending a new classifier from the given reserved vectors
        new_fc = nn.Linear(new_fc_tensor.shape[1], new_fc_tensor.shape[0], bias=False).cuda()
        new_fc.weight.data.copy_(new_fc_tensor)
        self.classifiers.append(new_fc.cuda())

        # Maintaining the pseudo targets. Self.rv contains only the unassigned vectors
        all_idx = np.arange(self.rv.shape[0])
        self.rv = self.rv[all_idx[~np.isin(all_idx, col_ind)]]


    def find_reseverve_vectors_all(self):
        points = torch.randn(self.reserve_vector_count, self.num_features).cuda()
        points = normalize(points)
        points = torch.nn.Parameter(points)

        opt = torch.optim.SGD([points], lr=1)
        
        best_angle = 0
        tqdm_gen = tqdm(range(self.args.epochs_target_gen))

        for _ in tqdm_gen:
            # Compute the cosine similarity.
            sim = F.cosine_similarity(points[None,:,:], points[:,None,:], dim=-1)
            # l = torch.log(torch.exp(sim/self.temperature).sum(axis = 1)).sum() / points.shape[0]
            contrastive_loss = torch.log(torch.exp(sim/self.temperature).sum(axis = 1)).sum() / points.shape[0]

            # loss = contrastive_loss + 0.1 * orthogonality_loss
            loss = contrastive_loss

            
            loss.backward()
            opt.step()
            points.data = normalize(points.data)

            curr_angle, curr_angle_close = self.compute_angles(points.detach())
            if curr_angle > best_angle: # best angle is the angle with which the separation is maximised
                best_angle = curr_angle

            tqdm_gen.set_description(f"Loss = {loss:.5f}, Best Avg Angle (deg): {best_angle:.3f}, Average Angle rv+base [close]: {curr_angle_close:.3f}")

        # Setting Reserved vectors
        self.rv = points.data

    def forward(self, x):
        return self.get_logits(x)
        
    def get_logits(self, encoding, session = 0):
        output = []
        for i, cls in enumerate(self.classifiers.children()):
            out = F.linear(F.normalize(encoding, p=2, dim=-1), F.normalize(cls.weight, p=2, dim=-1))
            out = out / self.temperature
            output.append(out)
        output = torch.cat(output, axis = 1)
        return output



class ORCONET_ViT(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args
        
        
        self.encoder = create_model(self.args.model, self.args.pretrained)
        
        self.mab = MultiAttentionBlock(dim=self.args.encoder_outdim, num_heads=self.args.num_heads)
        
        self.projector = Projector(self.args.encoder_outdim, self.args.proj_hidden_dim, self.args.proj_output_dim)

        # Final classifier. This hosts the pseudo targets, all and classification happens here
        self.fc = PseudoTargetClassifier(self.args, self.args.proj_output_dim)
        
        self.best_mab = None
        self.best_projector = None
        
    def set_projector(self):
        self.best_projector = deepcopy(self.projector.state_dict())
    
    def set_mab_projector(self, save_mab_path=None, save_projector_path=None):

        self.mab.load_state_dict(torch.load(save_mab_path))
        self.projector.load_state_dict(torch.load(save_projector_path))
    
    def reset_projector(self):
        self.projector.load_state_dict(self.best_projector)
        
    def set_mab(self):
        self.best_mab = deepcopy(self.mab.state_dict())

    def reset_mab(self):
        self.mab.load_state_dict(self.best_mab)
        
    def forward_metric(self, x, task_id=-1, train=False):
        # Get projection output
        g_x = self.encode(x)

        # Get similarity scores between classifier prototypes and g_x
        sim = self.fc.get_logits(g_x, 0)

        return sim, g_x
    
    def augment(self, x, delta):
        cls_token = x[:, 0, :]
        patch_token = x[:, 1:, :]
        aug_patch_token = brightness_augmentation_v3(patch_token, max_delta=delta)
        x_aug = torch.cat((cls_token.unsqueeze(1), aug_patch_token), dim=1)
        return x_aug
    
    
    def get_trainable_params(self):
        trainable_params = [(name, param.shape) for name, param in self.named_parameters() if param.requires_grad]
        for name, shape in trainable_params:
            print(f"Parameter: {name}, Shape: {shape}")
    

    def encode(self, x, aug=False, base=False, includ_original=False):
         
        encodings = self.encoder(x)
        
        original_x = encodings['x']
        
        if aug:
            all_x_aug = []
            if base:
                for _ in range(self.args.base_num_aug):
                    x_aug = self.augment(original_x, self.args.max_delta_base)
                    all_x_aug.append(x_aug)
            else:
                for _ in range(self.args.inc_num_aug):
                    x_aug = self.augment(original_x, self.args.max_delta_inc)
                    all_x_aug.append(x_aug)
            
            if includ_original:
                all_x_aug = torch.cat([original_x] + all_x_aug, dim=0)
            else:
                all_x_aug = torch.cat(all_x_aug, dim=0)
                
            mab_output = self.mab(all_x_aug)[:, 0, :]
            x = self.projector(mab_output)
            
        else:
            mab_output = self.mab(original_x)[:, 0, :]
            x = self.projector(mab_output)
            
        return x
    

    def forward(self, input, **kwargs):
        if self.mode == "backbone":  # Pass only through the backbone
            input = self.encoder(input)
            return input
        if self.mode == 'encoder':
            input = self.encode(input, **kwargs)
            return input
        elif self.mode not in ['encoder', "backbone"]:
            input, encoding = self.forward_metric(input, **kwargs)
            return input, encoding
        else:
            raise ValueError('Unknown mode')

    def get_class_avg(self,data,label,class_list):
        """
            Using the exemplars available during training to instantiate the classifier for the novel setting
        """
        new_fc=[]
        cov_list = []
        for class_index in class_list:
            data_index=(label==class_index).nonzero().squeeze(-1)
            embedding=data[data_index]

            # Compute covariance matrix again
            cov_this = np.cov(normalize(embedding).cpu(), rowvar=False)
            cov_list.append(cov_this)

            proto=embedding.mean(0)     # Mean along the 0th axis of all the embeddings for this particular class index
            new_fc.append(proto)

        new_fc_tensor=torch.stack(new_fc,dim=0)

        return new_fc_tensor, cov_list

    def update_targets(self, trainloader, testloader, class_list, session):
        for batch in trainloader:
            data, label = [_.cuda() for _ in batch]
            data=self.encode(data).detach()

        new_prototypes, _ = self.get_class_avg(data, label, class_list)
        
        # return new_prototypes

        # Assign a new novel classifier from the given reseve vectors
        self.fc.assign_novel_classifier(new_prototypes)
    
    def update_targets_v2(self, trainloader, class_list, args):
        with torch.no_grad():
            for batch in trainloader:
                data, label = batch
                data = torch.cat([data[0], data[1]], dim=0).cuda()
                label = label.cuda()
                data=self.encode(data, aug=args.inc_aug).detach()
                label  = label.repeat(2 * args.num_augmentations)

        new_prototypes, _ = self.get_class_avg(data, label, class_list)
        
        # Assign a new novel classifier from the given reseve vectors
        self.fc.assign_novel_classifier(new_prototypes)

    def get_optimizer_new(self, optimized_parameters):
        if self.args.optimizer_joint == "sgd":
            optimizer = torch.optim.SGD(optimized_parameters,lr=self.args.lr_new, momentum=0.9, dampening=0.9, weight_decay=self.args.decay_new)
        elif self.args.optimizer_joint == 'adam':
            optimizer = torch.optim.Adam(optimized_parameters, lr=self.args.lr_new, weight_decay=1e-4)
        return optimizer

    def get_optimizer_base(self, optimized_parameters):        
        if self.args.optimizer == "sgd":
            optimizer = torch.optim.SGD(optimized_parameters, lr=self.args.lr_base, momentum=0.9, dampening=0.9, weight_decay=self.args.decay)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(optimized_parameters, lr=self.args.lr_base, weight_decay=1e-4)
        return optimizer
    
    def get_optimizer_pretrain(self, optimized_parameters):        
        if self.args.pretrain_optimizer == "sgd":
            optimizer = torch.optim.SGD(optimized_parameters, lr=self.args.pretrain_lr, momentum=0.9, dampening=0.9, weight_decay=self.args.decay)
        elif self.args.pretrain_optimizer == 'adam':
            optimizer = torch.optim.Adam(optimized_parameters, lr=self.args.pretrain_lr, weight_decay=1e-4)
        return optimizer

    def test_pseudo_targets(self, fc, testloader, epoch, session):
        """
            Get the testing score for the fc that is being currently trained
        """
        test_class = self.args.base_class + session * self.args.way     # Final class idx that we could be in the labels this session
        va = Averager()

        self.eval()
        with torch.no_grad():
            for batch in testloader:
                data, test_label = [_.cuda() for _ in batch]
                encoding = self.encode(data).detach()
                logits = fc.get_logits(encoding)
                logits = logits[:, :test_class]
                acc = count_acc(logits, test_label)
                va.add(acc)
            va = va.item()

        metrics = {
            "va": va
        }

        return metrics
    
    def select_criterion(self):
        return nn.CrossEntropyLoss()

    def criterion_forward(self, criterion, logits, label):
        return criterion(logits, label)
    
    def pull_loss(self, label_rep, novel_class_start, criterion, logits):
        novel_classes_idx = torch.argwhere(label_rep >= novel_class_start).flatten()
        base_classes_idx = torch.argwhere(label_rep < novel_class_start).flatten()
        novel_loss = base_loss = 0
        if novel_classes_idx.numel() != 0:
            novel_loss = self.criterion_forward(criterion, logits[novel_classes_idx, :], label_rep[novel_classes_idx])                            
        if base_classes_idx.numel() != 0:
            base_loss = self.criterion_forward(criterion, logits[base_classes_idx, :], label_rep[base_classes_idx])
        cos_loss = (novel_loss*self.args.cos_n_lam) + (base_loss*self.args.cos_b_lam)
        return cos_loss
    

    def pretrain_base(self, baseloader):
        
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False
        
        for name, param in self.projector.named_parameters():
            print(name)
        
        optimized_parameters = [
                {'params': self.projector.parameters()},
                {'params': self.mab.parameters()}
            ]    
        
        
        self.get_trainable_params()
                    
        optimizer = self.get_optimizer_pretrain(optimized_parameters)
        
        # Setting up the scheduler
        if self.args.pretrain_schedule == "Milestone":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80],gamma=self.args.gamma)
        elif self.args.pretrain_schedule == "Cosine":
            warmup_epochs = self.args.warmup_epochs_pretrain
            min_lr = 1e-5
            scheduler = LinearWarmupCosineAnnealingLR(
                    optimizer, 
                    warmup_epochs=warmup_epochs, 
                    max_epochs=self.args.pretrain_epochs_max,
                    warmup_start_lr=3e-05 if warmup_epochs > 0 else self.args.pretrain_lr,
                    eta_min=min_lr)
        
        scl = supcon.SupConLoss()
        sscl = supcon.SupConLoss() 
        
        best_acc = 0
        
        writer = SummaryWriter(log_dir=self.args.output_dir)
        
        with torch.enable_grad():
            
            for epoch in tqdm(range(self.args.pretrain_epochs), desc="Training"):
                for param_group in optimizer.param_groups:
                    print(f'Learning rate: {param_group["lr"]}')
                total_loss = 0
                with tqdm(enumerate(baseloader), total=len(baseloader), desc="Processing Batches") as pbar:
                    for idx, (images, label) in pbar:
                                                
                        images = torch.cat([images[0], images[1]], dim=0).cuda()
                        label=label.cuda()

                        features = self.encode(images, aug=self.args.base_aug, base=True, includ_original=self.args.include_original)
                        features = normalize(features)
                        
                        if self.args.base_aug:
                            if self.args.include_original:
                                split_size = features.shape[0] // (2 * (self.args.base_num_aug + 1)) # add original feature
                            else:
                                split_size = features.shape[0] // (2 * (self.args.base_num_aug))
                        else:
                            split_size = features.shape[0] // 2
                        split_features = torch.split(features, split_size, dim=0)
                        features_ = torch.cat([t.unsqueeze(1) for t in split_features], dim=1)
                                            
                        loss = (1 - self.args.alpha) * scl(features_, label) + self.args.alpha * sscl(features_)
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        total_loss += loss.item()
                        
                        pbar.set_postfix(loss=loss.item())
                        writer.add_scalar('Loss/pretrain', loss.item(), epoch * len(baseloader) + idx)
                                   
                acc = self.test_pretrain()
                
                if best_acc is None or best_acc < acc:
                    best_acc = acc
                    best_mab = deepcopy(self.mab.state_dict())
                    best_projector = deepcopy(self.projector.state_dict())
                    torch.save(best_mab, os.path.join(self.args.output_dir, 'best_pretrained_mab.pth'))    
                    torch.save(best_projector, os.path.join(self.args.output_dir, 'best_pretrained_projector.pth'))
                
                scheduler.step()
               
        
    
    def update_base(self, baseloader, testloader):
        
        for name, param in self.encoder.named_parameters():
            param.requires_grad = False
        
        for name, param in self.mab.named_parameters():
            param.requires_grad = True
        
        optimized_parameters = [
                {'params': self.projector.parameters()},
                {'params': self.mab.parameters()}
            ]
        
                
        self.get_trainable_params()
        
        optimizer = self.get_optimizer_base(optimized_parameters)

        # Setting up the scheduler
        if self.args.base_schedule == "Milestone":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80],gamma=self.args.gamma)
        elif self.args.base_schedule == "Cosine":
            warmup_epochs = self.args.warmup_epochs_base
            min_lr = 1e-5
            scheduler = LinearWarmupCosineAnnealingLR(
                    optimizer, 
                    warmup_epochs=warmup_epochs, 
                    max_epochs=self.args.epochs_base,
                    warmup_start_lr=3e-05 if warmup_epochs > 0 else self.args.lr_base,
                    eta_min=min_lr)
            
        scl = supcon.SupConLoss()
        xent = self.select_criterion()

        best_acc = 0
        best_projector = None

        # Targets for PSCL
        target_prototypes = torch.nn.Parameter(self.fc.rv)
        target_prototypes.requires_grad = False
        target_labels = torch.arange(self.fc.reserve_vector_count - self.args.base_class).cuda() + self.args.base_class

        # Variables for Orthogonality Loss
        assigned_targets = self.fc.get_classifier_weights().detach().clone()
        assigned_targets_label = torch.arange(self.args.base_class)
        unassigned_targets = self.fc.rv.detach().clone()
        
        writer = SummaryWriter(log_dir=self.args.output_dir)

        with torch.enable_grad():
            tqdm_gen = tqdm(range(self.args.epochs_base))
            for epoch in tqdm_gen:
                total_loss = 0
                ta = Averager()
                for idx, batch in enumerate(baseloader):
                # for idx, batch in enumerate(tqdm(baseloader, desc="Processing Batches")):
                    images, label = batch
                                        
                    images = torch.cat([images[0], images[1]], dim=0).cuda()
                    label=label.cuda()
                                            
                    features = self.encode(images, aug=self.args.base_aug, base=True, includ_original=self.args.include_original)
                    features = normalize(features)
                    
                    if self.args.base_aug:
                        if self.args.include_original:
                            nviews = 2 * (self.args.base_num_aug + 1)
                        else:
                            nviews = 2 * (self.args.base_num_aug)
                    else:
                        nviews = 2
                    
                    # assert features.shape[0] % (2 * self.args.pretrain_num_aug) == 0
                    # split_size = features.shape[0] // (2 * self.args.pretrain_num_aug)
                    split_size = features.shape[0] // nviews
                    split_features = torch.split(features, split_size, dim=0)

                    # PSCL Loss
                    bsz = pbsz = label.shape[0]
                    perturbed_targets, target_labels_ = perturb_targets_norm_count(target_prototypes.clone(), target_labels.clone(), pbsz, 
                                                                                   nviews=nviews, epsilon = self.args.perturb_epsilon_base,
                                                                                   offset=self.args.perturb_offset)
                    features_add_pertarget = []
                    for i in range(len(split_features)):
                        f_add_pert = torch.cat((split_features[i], perturbed_targets[i]), axis = 0)
                        features_add_pertarget.append(f_add_pert.unsqueeze(1))
                    features_add_pertarget = torch.cat(features_add_pertarget, dim=1)
                    
                    label_ = torch.cat((label, target_labels_))
                    loss = self.args.sup_lam * scl(features_add_pertarget, label_)


                    # Cross Entropy
                    label_rep = label.repeat(nviews)
                    logits = self.fc(features)
                    loss += self.args.cos_lam * self.criterion_forward(xent, logits, label_rep)
                    
                    # Orthogonality Loss
                    # orth_loss = simplex_loss(f1, label, assigned_targets, assigned_targets_label, unassigned_targets)
                    # orth_loss = simplex_loss(split_features[0], label, assigned_targets, assigned_targets_label, unassigned_targets)
                    orth_loss = simplex_loss(features, label_rep, assigned_targets, assigned_targets_label, unassigned_targets)
                    loss += self.args.simplex_lam * orth_loss
                
                    ta.add(count_acc(logits, label_rep))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    
                    writer.add_scalar('Loss/base alignment', loss.item(), epoch * len(baseloader) + idx)

                    out_string = f"Epoch: {epoch}|[{idx}/{len(baseloader)}], Training Accuracy (Base): {ta.item()*100:.3f}, Validation Accuracy (Base): {best_acc*100:.3f}"
                    tqdm_gen.set_description(out_string)

                # Model Saving
                test_out = self.test_pseudo_targets(self.fc, testloader, epoch, 0)
                va = test_out["va"]
                
                if best_acc is None or best_acc < va:
                    best_acc = va
                    best_projector = deepcopy(self.projector.state_dict())

                out_string = f"Epoch: {epoch}, Training Accuracy (Base): {ta.item()*100:.3f}, Validation Accuracy (Base): {va*100:.3f}"
                tqdm_gen.set_description(out_string)
                
                scheduler.step()

        # Setting to best validation projection head
        self.projector.load_state_dict(best_projector, strict=True)

    def update_incremental(self, jointloader, session):        
        for name, param in self.encoder.named_parameters():
                param.requires_grad = False
        
        if self.args.stop_task1:     
            if session == 1:     
                for name, param in self.mab.named_parameters():
                    param.requires_grad = True
                
                optimized_parameters = [
                        {'params': self.projector.parameters()},
                        {'params': self.mab.parameters()}
                        ]
            else:
                for name, param in self.mab.named_parameters():
                    param.requires_grad = False
                
                optimized_parameters = [
                        {'params': self.projector.parameters()},
                        ]
        else:
            optimized_parameters = [
                    {'params': self.projector.parameters()},
                    {'params': self.mab.parameters()}
                    ]
        
        
        self.get_trainable_params()
        
        optimizer = self.get_optimizer_new(optimized_parameters)
        
        # Setting up the Cosine Scheduler
        warmup_epochs = self.args.warmup_epochs_inc
        min_lr = 0
        scheduler = LinearWarmupCosineAnnealingLR(
                optimizer, 
                warmup_epochs=warmup_epochs, 
                max_epochs=self.args.epochs_joint,
                warmup_start_lr=3e-05 if warmup_epochs > 0 else self.args.lr_new,
                eta_min=min_lr)

        sc_criterion = supcon.SupConLoss()
        pull_criterion = self.select_criterion()

        best_projector = None
        novel_class_start = self.args.base_class

        self.eval()

        # Target prototypes contains assigned targets and unassigned targets (except base)
        target_prototypes = torch.nn.Parameter(torch.cat((self.fc.get_classifier_weights(uptil=session+1)[self.args.base_class:].clone(), self.fc.rv.clone())))
        target_prototypes.requires_grad = False
        target_labels = torch.arange(self.fc.reserve_vector_count - self.args.base_class).cuda() + self.args.base_class

        # For Orthogonality loss
        assigned_targets = self.fc.get_classifier_weights().detach().clone()
        assigned_targets_label = torch.arange(self.args.base_class + (self.args.way * session))
        unassigned_targets = self.fc.rv.detach().clone()

        out_string = ""
        
        writer = SummaryWriter(log_dir=self.args.output_dir)
        

        with torch.enable_grad():
            tqdm_gen = tqdm(range(self.args.epochs_joint))
            for epoch in tqdm_gen:
                                
                total_loss = 0
                ta = Averager()
                for idx, batch in enumerate(jointloader):
                # for idx, batch in enumerate(tqdm(jointloader, desc="Processing Batches")):
                    images, label = batch
                    
                    inc_idx=torch.where(label>=60)[0]
                    base_idx = torch.where(label<60)[0]
                    
                    
                    if len(inc_idx) and len(base_idx):
                        label_base = label[base_idx].cuda()
                        label_inc = label[inc_idx].cuda()
                        
                        images_base = torch.cat([images[0][base_idx], images[1][base_idx]], dim=0).cuda()
                        images_inc = torch.cat([images[0][inc_idx], images[1][inc_idx]], dim=0).cuda()
                        #label = label.cuda()
                        
                        features_inc = self.encode(images_inc, aug=self.args.inc_aug)#(inc_aug*2*n,768)
                        features_inc = normalize(features_inc)
                        
                        features_base = self.encode(images_base, aug=False)#(2*n,768)
                        features_base = normalize(features_base)
                        
                        # if self.args.inc_aug:
                        nviews_inc = 2 * (self.args.inc_num_aug)
                        # else:
                        nviews = 2

                        split_size_inc = features_inc.shape[0] // nviews_inc
                        split_feat_inc = torch.split(features_inc, split_size_inc, dim=0)
                        split_feat_inc_view_1 = torch.cat(split_feat_inc[:self.args.inc_num_aug], dim=0)
                        split_feat_inc_view_2 = torch.cat(split_feat_inc[self.args.inc_num_aug:], dim=0)
                        
                        split_size_base = features_base.shape[0] // nviews
                        split_feat_base = torch.split(features_base, split_size_base, dim=0)
                        
                        split_feat_view_1 = torch.cat((split_feat_inc_view_1, split_feat_base[0]), dim=0)
                        split_feat_view_2 = torch.cat((split_feat_inc_view_2, split_feat_base[1]), dim=0)
                        split_feat = [split_feat_view_1, split_feat_view_2]
                        
                        label = torch.cat((label_inc.repeat(self.args.inc_num_aug), label_base), dim=0)
                    
                    
                    elif len(inc_idx) == 0:
                        label_base = label[base_idx].cuda()
                        
                        images_base = torch.cat([images[0][base_idx], images[1][base_idx]], dim=0).cuda()
                        
                        features_base = self.encode(images_base, aug=False)#(2*n,768)
                        features_base = normalize(features_base)
                        
                        # if self.args.inc_aug:
                        nviews_inc = 2 * (self.args.inc_num_aug)
                        # else:
                        nviews = 2
                        
                        split_size_base = features_base.shape[0] // nviews
                        split_feat_base = torch.split(features_base, split_size_base, dim=0)
                        
                        split_feat = split_feat_base
                        
                        label = label_base
                    
                    else :
                        label_inc = label[inc_idx].cuda()
                        
                        images_inc = torch.cat([images[0][inc_idx], images[1][inc_idx]], dim=0).cuda()
                        
                        features_inc = self.encode(images_inc, aug=self.args.inc_aug)#(inc_aug*2*n,768)
                        features_inc = normalize(features_inc)
                        
                        nviews_inc = 2 * (self.args.inc_num_aug)
                        nviews = 2

                        split_size_inc = features_inc.shape[0] // nviews_inc
                        split_feat_inc = torch.split(features_inc, split_size_inc, dim=0)
                        split_feat_inc_view_1 = torch.cat(split_feat_inc[:self.args.inc_num_aug], dim=0)
                        split_feat_inc_view_2 = torch.cat(split_feat_inc[self.args.inc_num_aug:], dim=0)
                        
                        split_feat = [split_feat_inc_view_1, split_feat_inc_view_2]
                        
                        label = label_inc.repeat(self.args.inc_num_aug)
                    
                    
                    bsz = pbsz = label.shape[0]
                    perturbed_targets, target_labels_ = perturb_targets_norm_count(target_prototypes.clone(), target_labels.clone(), pbsz, 
                                                                                   nviews=nviews, epsilon = self.args.perturb_epsilon_base, 
                                                                                   offset=self.args.perturb_offset)
                    
                    features_add_pertarget = []
                    for i in range(nviews):
                        f_add_pert = torch.cat((split_feat[i], perturbed_targets[i]), axis = 0)#128+128
                        features_add_pertarget.append(f_add_pert.unsqueeze(1))
                    features_add_pertarget = torch.cat(features_add_pertarget, dim=1)
                    label_ = torch.cat((label, target_labels_))#128+128
                    pscl = self.args.sup_lam * sc_criterion(features_add_pertarget, label_)

                    # Cross Entropy
                    features = torch.cat(split_feat, dim=0)
                    label_rep = label.repeat(nviews)
                    
                    logits = self.fc(features)
                    # label_rep = label.repeat(2 * (self.args.num_augmentations + 1))
                    # label_rep = label.repeat(nviews)
                    xent_loss = self.args.cos_lam * self.pull_loss(label_rep, novel_class_start, pull_criterion, logits)#pull_criterion=nn.CrossEntropyLoss()


                    # # Computing simplex loss
                    new_ixs = torch.argwhere(label_rep >= self.args.base_class).flatten()
                    orth_loss = self.args.simplex_lam * simplex_loss(features[new_ixs], label_rep[new_ixs], assigned_targets, assigned_targets_label, unassigned_targets)             
                    
                    # new_ixs = torch.argwhere(label >= self.args.base_class).flatten()
                    # orth_loss = self.args.simplex_lam * simplex_loss(split_feat[0][new_ixs], label[new_ixs], assigned_targets, assigned_targets_label, unassigned_targets)             
                    
                    # Combined Loss
                    loss = pscl + xent_loss + orth_loss
                    
                    ta.add(count_acc(logits, label_rep))
                    total_loss += loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    writer.add_scalar(f'Loss/{session}_incremental loss', loss.item(), epoch * len(jointloader) + idx)
                    writer.add_scalar(f'Loss/{session}_pscl loss', pscl.item(), epoch * len(jointloader) + idx)
                    writer.add_scalar(f'Loss/{session}_xent loss', xent_loss.item(), epoch * len(jointloader) + idx)
                    writer.add_scalar(f'Loss/{session}_orth loss', orth_loss.item(), epoch * len(jointloader) + idx)

                # Model Saving
                out_string = 'Session: {}, Epoch: {}|, Training Loss (Joint): {:.3f}, Training Accuracy (Joint): {:.3f}'\
                    .format(
                            session, 
                            epoch,
                            total_loss,
                            float('%.3f' % (ta.item() * 100.0))
                            )
                tqdm_gen.set_description(out_string)
                
                scheduler.step()

        
    def test_pretrain(self):
        
        self.eval()
        
        # trainset, _, testloader = data_utils.get_dataloader(self.args, 0)
        trainset, _, testloader = data_utils.get_dataloader(self.args, 0)
        
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
        trainloader.dataset.transform = testloader.dataset.transform
        
        train_features, train_labels = self.extract_feature(trainloader)
        test_features, test_labels = self.extract_feature(testloader)
        
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(train_features, train_labels)
        
        test_pred = knn.predict(test_features)
        
        accuracy = accuracy_score(test_labels, test_pred)
        print(f'KNN Classification Accuracy: {accuracy * 100:.2f}%')
        return accuracy
        
    
    def extract_feature(self, dataloader):
        self.eval()
        features = []
        labels = []
        with torch.no_grad():
            tqdm_gen = tqdm(dataloader)
            tqdm_gen.set_description("Generating Features: ")
            for i, batch in enumerate(tqdm_gen, 1):
                images, label = batch
                feats = self.encode(images.cuda()).view(images.size(0), -1)
                features.append(feats.cpu())
                labels.append(label)
        return torch.cat(features).numpy(), torch.cat(labels).numpy()
        