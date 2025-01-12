import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
import math


class Projector(nn.Module):
    def __init__(self, input_dim = 768, hidden_dim = 2048, output_dim = 128):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
          
    def forward(self, x):
        h = self.fc1(x)
        output = self.fc2(self.relu(h))
        
        return output



class Projector_delta_W(nn.Module):
    def __init__(self, input_dim = 768, hidden_dim = 2048, output_dim = 128, args=None):
        super(Projector_delta_W, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_task = args.sessions
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        self.delta_w_fc1 = nn.Parameter(torch.zeros((self.num_task, self.input_dim, self.hidden_dim)), requires_grad=True)
        self.delta_w_fc2 = nn.Parameter(torch.zeros((self.num_task, self.hidden_dim, self.output_dim)), requires_grad=True)

                                                 
        for param in self.fc1.parameters():
            param.requires_grad = False
        
        for param in self.fc2.parameters():
            param.requires_grad = False
          
        self._init_weights()
            
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.fc1.weight, std=.02)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        
        nn.init.trunc_normal_(self.fc2.weight, std=.02)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)
        
        nn.init.zeros_(self.delta_w_fc1)
        nn.init.zeros_(self.delta_w_fc2)
            
    def cal_delta_w(self, task_id=-1, train=False):
        
        if train:
            assert isinstance(task_id, int)         
            if task_id > 0:
                with torch.no_grad():
                    fc1_delta_w_his = torch.sum(self.delta_w_fc1[:task_id], dim=0)
                    fc2_delta_w_his = torch.sum(self.delta_w_fc2[:task_id], dim=0)         
                
                fc1_delta_w = fc1_delta_w_his + self.delta_w_fc1[task_id]  
                fc2_delta_w = fc2_delta_w_his + self.delta_w_fc2[task_id]           
            else:
                fc1_delta_w = self.delta_w_fc1[task_id]
                fc2_delta_w = self.delta_w_fc2[task_id]
                                  
        else:
            assert isinstance(task_id, int)
            
            with torch.no_grad():
                
                fc1_delta_w = torch.sum(self.delta_w_fc1[:task_id + 1], dim=0)
                fc2_delta_w = torch.sum(self.delta_w_fc2[:task_id + 1], dim=0)
                        
        return fc1_delta_w, fc2_delta_w
    
    def ortho_loss(self, task_id=-1):        
        
        pre_fc1_delta_w = self.delta_w_fc1[:task_id].detach().permute(0, 2, 1).reshape(-1, self.input_dim)
        
        pre_fc2_delta_w = self.delta_w_fc2[:task_id].detach().permute(0, 2, 1).reshape(-1, self.hidden_dim)
        
        dot_matrix_fc1 = torch.matmul(pre_fc1_delta_w, self.delta_w_fc1[task_id]) 
        dot_matrix_fc2 = torch.matmul(pre_fc2_delta_w, self.delta_w_fc2[task_id])
        
        
        loss = torch.norm(dot_matrix_fc1, p='fro') + torch.norm(dot_matrix_fc2, p='fro')
        loss = loss.cuda()
            
        return loss
    
    def ortho_loss_v2(self, task_id=-1):        
        
        pre_fc1_delta_w = self.delta_w_fc1[:task_id].detach()
        concat_fc1_delta_w = torch.cat((pre_fc1_delta_w, self.delta_w_fc1[task_id].unsqueeze(0)), dim=0)
        concat_fc1_delta_w = concat_fc1_delta_w.permute(2, 0, 1)
        dot_matrix_fc1 = torch.bmm(concat_fc1_delta_w, concat_fc1_delta_w.transpose(1, 2))
        mask_fc1 = 1 - torch.eye(task_id + 1, device=dot_matrix_fc1.device).unsqueeze(0).expand(self.hidden_dim, -1, -1)
        mask_dot_product_fc1 = dot_matrix_fc1 * mask_fc1
        
        
        pre_fc2_delta_w = self.delta_w_fc2[:task_id].detach()
        concat_fc2_delta_w = torch.cat((pre_fc2_delta_w, self.delta_w_fc2[task_id].unsqueeze(0)), dim=0)
        concat_fc2_delta_w = concat_fc2_delta_w.permute(2, 0, 1)
        dot_matrix_fc2 = torch.bmm(concat_fc2_delta_w, concat_fc2_delta_w.transpose(1, 2))
        mask_fc2 = 1 - torch.eye(task_id + 1, device=dot_matrix_fc2.device).unsqueeze(0).expand(self.output_dim, -1, -1)
        mask_dot_product_fc2 = dot_matrix_fc2 * mask_fc2
        
        loss = torch.norm(mask_dot_product_fc1, p='fro') + torch.norm(mask_dot_product_fc2, p='fro')
        loss = loss.cuda()
            
        return loss
    
    def to_device(self):
        self.delta_w_fc1 = self.delta_w_fc1.cuda()
        self.delta_w_fc2 = self.delta_w_fc2.cuda()
            
    
    def forward(self, x, task_id=0, train=False):
        
        self.to_device()
         
        fc1_delta_w, fc2_delta_w = self.cal_delta_w(task_id=task_id, train=train)
        
        # fc1_delta_w_output = torch.einsum('bd, dz->bz', x, fc1_delta_w)
        w_fc1 = self.fc1.weight.t() + fc1_delta_w
        
        h = torch.einsum('bd, dz->bz', x, w_fc1) + self.fc1.bias
        
        h = self.relu(h)
        
        w_fc2 = self.fc2.weight.t() + fc2_delta_w
        
        output = torch.einsum('bd, dz->bz', h, w_fc2) + self.fc2.bias
        
        # fc2_delta_w_output = torch.einsum('bd, dz->bz', h, fc2_delta_w)
        
        # output = self.fc2(h) + fc2_delta_w_output
        
        return output
    
class Projector_delta_W_omega(nn.Module):
    
    def __init__(self, input_dim = 768, hidden_dim = 2048, output_dim = 128, args=None):
        super(Projector_delta_W_omega, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_task = args.sessions
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        self.delta_w_fc1 = nn.Parameter(torch.zeros((self.num_task, self.input_dim, self.hidden_dim)), requires_grad=True)
        self.delta_w_fc2 = nn.Parameter(torch.zeros((self.num_task, self.hidden_dim, self.output_dim)), requires_grad=True)

        
        self.omega_w_fc1 = nn.Parameter(torch.ones(self.num_task), requires_grad=True)
        self.omega_w_fc2 = nn.Parameter(torch.ones(self.num_task), requires_grad=True)
                                         
        for param in self.fc1.parameters():
            param.requires_grad = False
        
        for param in self.fc2.parameters():
            param.requires_grad = False
          
        self._init_weights()
            
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.fc1.weight, std=.02)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        
        nn.init.trunc_normal_(self.fc2.weight, std=.02)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)
        
        nn.init.zeros_(self.delta_w_fc1)
        nn.init.zeros_(self.delta_w_fc2)
        
        nn.init.ones_(self.omega_w_fc1)
        nn.init.ones_(self.omega_w_fc2)
    
    def cal_delta_w(self, task_id=-1, train=False):
        
        if train:
            assert isinstance(task_id, int)         
            if task_id > 0:
                with torch.no_grad():
                    fc1_delta_w_his= self.delta_w_fc1[:task_id]
                    fc2_delta_w_his = self.delta_w_fc2[:task_id]   
                
                fc1_delta_w = (self.omega_w_fc1[:task_id].reshape(-1, 1, 1) * fc1_delta_w_his).sum(dim=0) + self.delta_w_fc1[task_id]
                fc2_delta_w = (self.omega_w_fc2[:task_id].reshape(-1, 1, 1) * fc2_delta_w_his).sum(dim=0) + self.delta_w_fc2[task_id]
                         
            else:
                fc1_delta_w = self.delta_w_fc1[task_id]
                fc2_delta_w = self.delta_w_fc2[task_id]
                                  
        else:
            assert isinstance(task_id, int)
            
            with torch.no_grad():
                
                fc1_delta_w = (self.omega_w_fc1[:task_id + 1].reshape(-1, 1, 1) * self.delta_w_fc1[:task_id + 1]).sum(dim=0)
                fc2_delta_w = (self.omega_w_fc2[:task_id + 1].reshape(-1, 1, 1) * self.delta_w_fc2[:task_id + 1]).sum(dim=0)
                        
        return fc1_delta_w, fc2_delta_w
    
    def ortho_loss(self, task_id=-1):        
        
        pre_fc1_delta_w = self.delta_w_fc1[:task_id].detach().permute(0, 2, 1).reshape(-1, self.input_dim)
        
        pre_fc2_delta_w = self.delta_w_fc2[:task_id].detach().permute(0, 2, 1).reshape(-1, self.hidden_dim)
        
        dot_matrix_fc1 = torch.matmul(pre_fc1_delta_w, self.delta_w_fc1[task_id]) 
        dot_matrix_fc2 = torch.matmul(pre_fc2_delta_w, self.delta_w_fc2[task_id]) 
        
        loss = torch.norm(dot_matrix_fc1, p='fro') + torch.norm(dot_matrix_fc2, p='fro')
        loss = loss.cuda()
            
        return loss
    
    def ortho_loss_v2(self, task_id=-1):        
        
        pre_fc1_delta_w = self.delta_w_fc1[:task_id].detach()
        concat_fc1_delta_w = torch.cat((pre_fc1_delta_w, self.delta_w_fc1[task_id].unsqueeze(0)), dim=0)
        concat_fc1_delta_w = concat_fc1_delta_w.permute(2, 0, 1)
        dot_matrix_fc1 = torch.bmm(concat_fc1_delta_w, concat_fc1_delta_w.transpose(1, 2))
        mask = 1 - torch.eye(self.num_task, device=dot_matrix_fc1.device).unsqueeze(0).expand(self.hidden_dim, -1, -1)
        mask_dot_product_fc1 = dot_matrix_fc1 * mask
        
        
        pre_fc2_delta_w = self.delta_w_fc2[:task_id].detach()
        concat_fc2_delta_w = torch.cat((pre_fc2_delta_w, self.delta_w_fc2[task_id].unsqueeze(0)), dim=0)
        concat_fc2_delta_w = concat_fc2_delta_w.permute(2, 0, 1)
        dot_matrix_fc2 = torch.bmm(concat_fc2_delta_w, concat_fc2_delta_w.transpose(1, 2))
        mask = 1 - torch.eye(self.num_task, device=dot_matrix_fc2.device).unsqueeze(0).expand(self.output_dim, -1, -1)
        mask_dot_product_fc2 = dot_matrix_fc2 * mask
        
        loss = torch.norm(mask_dot_product_fc1, p='fro') + torch.norm(mask_dot_product_fc2, p='fro')
        loss = loss.cuda()
            
        return loss
    
    def to_device(self):
        self.delta_w_fc1 = self.delta_w_fc1.cuda()
        self.delta_w_fc2 = self.delta_w_fc2.cuda()
        self.omega_w_fc1 = self.omega_w_fc1.cuda()
        self.omega_w_fc2 = self.omega_w_fc2.cuda()
            
    
    def forward(self, x, task_id=0, train=False):
        
        self.to_device()
         
        fc1_delta_w, fc2_delta_w = self.cal_delta_w(task_id=task_id, train=train)
        
        # fc1_delta_w_output = torch.einsum('bd, dz->bz', x, fc1_delta_w)
        w_fc1 = self.fc1.weight.t() + fc1_delta_w
        
        h = torch.einsum('bd, dz->bz', x, w_fc1) + self.fc1.bias
        
        h = self.relu(h)
        
        w_fc2 = self.fc2.weight.t() + fc2_delta_w
        
        output = torch.einsum('bd, dz->bz', h, w_fc2) + self.fc2.bias
        
        # fc2_delta_w_output = torch.einsum('bd, dz->bz', h, fc2_delta_w)
        
        # output = self.fc2(h) + fc2_delta_w_output
        
        return output

class Porjector_complex(nn.Module):
    def __init__(self, input_dim = 768, hidden_dim = 2048, output_dim = 128, args=None):
        super(Porjector_complex, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_task = args.sessions
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.gelu = nn.GELU()
        
        
        
        self.delta_w_fc1 = nn.Parameter(torch.zeros((self.num_task, self.input_dim, self.hidden_dim)), requires_grad=True)
        self.delta_w_fc2 = nn.Parameter(torch.zeros((self.num_task, self.hidden_dim, self.hidden_dim)), requires_grad=True)
        self.delta_w_fc3 = nn.Parameter(torch.zeros((self.num_task, self.hidden_dim, self.output_dim)), requires_grad=True)
        
                                         
        for param in self.fc1.parameters():
            param.requires_grad = False
        
        for param in self.fc2.parameters():
            param.requires_grad = False
        
        for param in self.fc3.parameters():
            param.requires_grad = False
          
        self._init_weights()
            
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.fc1.weight, std=.02)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        
        nn.init.trunc_normal_(self.fc2.weight, std=.02)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)
                   
        nn.init.trunc_normal_(self.fc3.weight, std=.02)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc3.bias)
        
        nn.init.zeros_(self.delta_w_fc1)
        nn.init.zeros_(self.delta_w_fc2)
        nn.init.zeros_(self.delta_w_fc3)
        
    
    def cal_delta_w(self, task_id=-1, train=False):
        
        if train:
            assert isinstance(task_id, int)         
            if task_id > 0:
                with torch.no_grad():
                    fc1_delta_w_his = torch.sum(self.delta_w_fc1[:task_id], dim=0)
                    fc2_delta_w_his = torch.sum(self.delta_w_fc2[:task_id], dim=0) 
                    fc3_delta_w_his = torch.sum(self.delta_w_fc3[:task_id], dim=0)        
                
                fc1_delta_w = fc1_delta_w_his + self.delta_w_fc1[task_id]  
                fc2_delta_w = fc2_delta_w_his + self.delta_w_fc2[task_id]  
                fc3_delta_w = fc3_delta_w_his + self.delta_w_fc3[task_id]         
            else:
                fc1_delta_w = self.delta_w_fc1[task_id]
                fc2_delta_w = self.delta_w_fc2[task_id]
                fc3_delta_w = self.delta_w_fc3[task_id]
                                  
        else:
            assert isinstance(task_id, int)
            
            with torch.no_grad():
                
                fc1_delta_w = torch.sum(self.delta_w_fc1[:task_id + 1], dim=0)
                fc2_delta_w = torch.sum(self.delta_w_fc2[:task_id + 1], dim=0)
                fc3_delta_w = torch.sum(self.delta_w_fc3[:task_id + 1], dim=0)
                        
        return fc1_delta_w, fc2_delta_w, fc3_delta_w
    
    def ortho_loss(self, task_id=-1):        
        
        pre_fc1_delta_w = self.delta_w_fc1[:task_id].detach().permute(0, 2, 1).reshape(-1, self.input_dim)
        
        pre_fc2_delta_w = self.delta_w_fc2[:task_id].detach().permute(0, 2, 1).reshape(-1, self.hidden_dim)
        
        pre_fc3_delta_w = self.delta_w_fc3[:task_id].detach().permute(0, 2, 1).reshape(-1, self.hidden_dim)
        
        dot_matrix_fc1 = torch.matmul(pre_fc1_delta_w, self.delta_w_fc1[task_id]) 
        dot_matrix_fc2 = torch.matmul(pre_fc2_delta_w, self.delta_w_fc2[task_id]) 
        dot_matrix_fc3 = torch.matmul(pre_fc3_delta_w, self.delta_w_fc3[task_id])
        
        loss = torch.norm(dot_matrix_fc1, p='fro') + torch.norm(dot_matrix_fc2, p='fro') + torch.norm(dot_matrix_fc3, p='fro')
        loss = loss.cuda()
            
        return loss
    
    def ortho_loss_v2(self, task_id=-1):        
        
        pre_fc1_delta_w = self.delta_w_fc1[:task_id].detach()
        concat_fc1_delta_w = torch.cat((pre_fc1_delta_w, self.delta_w_fc1[task_id].unsqueeze(0)), dim=0)
        concat_fc1_delta_w = concat_fc1_delta_w.permute(2, 0, 1)
        dot_matrix_fc1 = torch.bmm(concat_fc1_delta_w, concat_fc1_delta_w.transpose(1, 2))
        mask = 1 - torch.eye(self.num_task, device=dot_matrix_fc1.device).unsqueeze(0).expand(self.hidden_dim, -1, -1)
        mask_dot_product_fc1 = dot_matrix_fc1 * mask
        
        
        pre_fc2_delta_w = self.delta_w_fc2[:task_id].detach()
        concat_fc2_delta_w = torch.cat((pre_fc2_delta_w, self.delta_w_fc2[task_id].unsqueeze(0)), dim=0)
        concat_fc2_delta_w = concat_fc2_delta_w.permute(2, 0, 1)
        dot_matrix_fc2 = torch.bmm(concat_fc2_delta_w, concat_fc2_delta_w.transpose(1, 2))
        mask = 1 - torch.eye(self.num_task, device=dot_matrix_fc2.device).unsqueeze(0).expand(self.output_dim, -1, -1)
        mask_dot_product_fc2 = dot_matrix_fc2 * mask
        
        loss = torch.norm(mask_dot_product_fc1, p='fro') + torch.norm(mask_dot_product_fc2, p='fro')
        loss = loss.cuda()
            
        return loss
    
    def to_device(self):
        self.delta_w_fc1 = self.delta_w_fc1.cuda()
        self.delta_w_fc2 = self.delta_w_fc2.cuda()
        self.delta_w_fc3 = self.delta_w_fc3.cuda()
            
    
    def forward(self, x, task_id=0, train=False):
        
        self.to_device()
         
        fc1_delta_w, fc2_delta_w, fc3_delta_w = self.cal_delta_w(task_id=task_id, train=train)
        
        w_fc1 = self.fc1.weight.t() + fc1_delta_w
        
        h = torch.einsum('bd, dz->bz', x, w_fc1) + self.fc1.bias
        
        h = self.gelu(h)
          
        w_fc2 = self.fc2.weight.t() + fc2_delta_w
        
        z = torch.einsum('bd, dz->bz', h, w_fc2) + self.fc2.bias
        
        z = self.gelu(z)
        
        w_fc3 = self.fc3.weight.t() + fc3_delta_w
        
        output = torch.einsum('bd, dz->bz', z, w_fc3) + self.fc3.bias

        
        return output
        