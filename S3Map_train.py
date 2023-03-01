#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 01:13:04 2021

@author: Fenqiang Zhao

@contact: zhaofenqiang0221@gmail.com
"""


import numpy as np
import glob

import torch
import os
import math

from sphericalunet.model import Unet
from sphericalunet.utils.interp_torch import convert2DTo3D, getEn, diffeomorp, get_bi_inter
from sphericalunet.utils.utils import get_neighs_order, get_vertex_dis
from sphericalunet.utils.vtk import read_vtk

from utils_spherical_mapping import ResampledInnerSurf, computeMetrics

from torch.utils.tensorboard import SummaryWriter
#os.system('rm -rf /pine/scr/f/e/fenqiang/dHCP/scripts/log/*')

NUM_NEIGHBORS = 6

###############################################################################

dataset = 'cereblm'    # 'BCP', 'ADNI', 'NEO', 'dHCP' 'cereblm'
initial_mapping = 'FaSu'  # 'SIP' 'CF'  'FaSu'
n_vertex = 10242

device = torch.device('cpu') # torch.device('cpu'), or torch.device('cuda:0')

if initial_mapping == 'SIP':
    
    if n_vertex == 10242:
        weight_distan = 1.0
        weight_area = 0.8
        weight_smooth = 6.0
        deform_scale = 10.0
    elif n_vertex == 40962:
        weight_distan = 1.0
        weight_area = 1.0
        weight_angle = 0.0
        weight_smooth = 6.0
        deform_scale = 30.0
    elif n_vertex == 163842:
        weight_distan = 1.0
        weight_area = 0.8
        weight_smooth = 5.0
        deform_scale = 30.0
    else:
        print("error")
        
elif initial_mapping == 'FaSu':
    
    if n_vertex == 10242:
        weight_distan = 1.0
        weight_area = 1.0
        weight_smooth = 200
        weight_angle = 0.2
        deform_scale = 10.0
    elif n_vertex == 40962:
        weight_distan = 1.0
        weight_area = 1.2
        weight_angle = 0.2
        weight_smooth = 20.0
        deform_scale = 30.0
    elif n_vertex == 163842:
        weight_distan = 1.0
        weight_area = 0.8
        weight_smooth = 8.0
        deform_scale = 30.0
    else:
        print("error")     
      
 
elif initial_mapping == 'CF':
    
    if n_vertex == 10242:
        weight_distan = 1.0
        weight_area = 0.8
        weight_smooth = 10.0
        deform_scale = 10.0
    elif n_vertex == 40962:
        weight_distan = 1.0
        weight_area = 0.8
        weight_smooth = 20.0
        deform_scale = 30.0
    elif n_vertex == 163842:
        weight_distan = 1.0
        weight_area = 0.8
        weight_smooth = 20.0
        deform_scale = 30.0
    else:
        print("error")     
    
    
batch_size = 1
learning_rate = 0.005

writer = SummaryWriter('/pine/scr/f/e/fenqiang/dHCP/scripts/log/SMnet_ver'+ str(n_vertex) + '_' + \
               dataset+ '_' +initial_mapping+'_area'+ str(weight_area) +'_angle'+ str(weight_angle)+ '_dist'+ str(weight_distan) +'_smooth'+ str(weight_smooth) +'.log')


#max_disp = get_vertex_dis(n_vertex)/100.0 * 10

###############################################################################
    
if dataset == 'BCP':
    if initial_mapping == 'SIP':
        if n_vertex == 10242:
            files = sorted(glob.glob('/pine/scr/f/e/fenqiang/BCP_SpheMap/*/*.SIP.RespInner.npy'))
        elif n_vertex == 40962:
            files = sorted(glob.glob('/pine/scr/f/e/fenqiang/BCP_SpheMap/*/*.SIP.10242moved.OrigSpheMoved.RespInner.npy'))
        elif n_vertex == 163842:
            files = sorted(glob.glob('/pine/scr/f/e/fenqiang/BCP_SpheMap/*/*.SIP.10242moved.OrigSpheMoved.40962moved.OrigSpheMoved.RespInner.npy'))
        else:
            print("error")
    elif initial_mapping == 'FaSu':
        if n_vertex == 10242:
            files = sorted(glob.glob('/pine/scr/f/e/fenqiang/BCP_SpheMap/*/*.FaSu.RespInner.npy'))
        elif n_vertex == 40962:
            files = sorted(glob.glob('/pine/scr/f/e/fenqiang/BCP_SpheMap/*/*.FaSu.10242moved.OrigSpheMoved.RespInner.npy'))
        elif n_vertex == 163842:
            files = sorted(glob.glob('/pine/scr/f/e/fenqiang/BCP_SpheMap/*/*.FaSu.10242moved.OrigSpheMoved.40962moved.OrigSpheMoved.RespInner.npy'))
        else:
            print("error")
        
    elif initial_mapping == 'CF':
        if n_vertex == 10242:
            files = sorted(glob.glob('/pine/scr/f/e/fenqiang/BCP_SpheMap/*/*.CF.RespInner.npy'))
        elif n_vertex == 40962:
            files = sorted(glob.glob('/pine/scr/f/e/fenqiang/BCP_SpheMap/*/*.CF.10242moved.OrigSpheMoved.RespInner.npy'))
        elif n_vertex == 163842:
            files = sorted(glob.glob('/pine/scr/f/e/fenqiang/BCP_SpheMap/*/*.CF.10242moved.OrigSpheMoved.40962moved.OrigSpheMoved.RespInner.npy'))
        else:
            print("error")    
            
        
elif dataset == 'dHCP':
    if initial_mapping == 'SIP':
        if n_vertex == 10242:
            files = sorted(glob.glob('/pine/scr/f/e/fenqiang/dHCP/*/*.SIP.RespInner.npy'))
        elif n_vertex == 40962:
            files = sorted(glob.glob('/pine/scr/f/e/fenqiang/dHCP/*/*.SIP.10242moved.OrigSpheMoved.RespInner.npy'))
        elif n_vertex == 163842:
            files = sorted(glob.glob('/pine/scr/f/e/fenqiang/dHCP/*/*.SIP.10242moved.OrigSpheMoved.40962moved.OrigSpheMoved.RespInner.npy'))
        else:
            print("error")
    elif initial_mapping == 'FaSu':
        if n_vertex == 10242:
            files = sorted(glob.glob('/pine/scr/f/e/fenqiang/dHCP/*/*.FaSu.RespInner.npy'))
        elif n_vertex == 40962:
            files = sorted(glob.glob('/pine/scr/f/e/fenqiang/dHCP/*/*.FaSu.10242moved.OrigSpheMoved.RespInner.npy'))
        elif n_vertex == 163842:
            files = sorted(glob.glob('/pine/scr/f/e/fenqiang/dHCP/*/*.FaSu.10242moved.OrigSpheMoved.40962moved.OrigSpheMoved.RespInner.npy'))
        else:
            print("error")
        
    elif initial_mapping == 'CF':
        if n_vertex == 10242:
            files = sorted(glob.glob('/pine/scr/f/e/fenqiang/dHCP/*/*.CF.RespInner.npy'))
        elif n_vertex == 40962:
            files = sorted(glob.glob('/pine/scr/f/e/fenqiang/dHCP/*/*.CF.10242moved.OrigSpheMoved.RespInner.npy'))
        elif n_vertex == 163842:
            files = sorted(glob.glob('/pine/scr/f/e/fenqiang/dHCP/*/*.CF.10242moved.OrigSpheMoved.40962moved.OrigSpheMoved.RespInner.npy'))
        else:
            print("error")    
        
        
elif dataset == 'cereblm':
    if n_vertex == 10242:
        files = sorted(glob.glob('/pine/scr/f/e/fenqiang/ForJiale_cerebellum/data/NORMAL*/NORMAL*.FastSurfer.RespInner.npy'))
    elif n_vertex == 163842:
        files = sorted(glob.glob('/pine/scr/f/e/fenqiang/dHCP/*/*.FaSu.10242moved.OrigSpheMoved.40962moved.OrigSpheMoved.RespInner.npy'))
    else:
        print("error")

    

    

#elif dataset == 'ADNI':
#    if initial_mapping == '':
#        if n_vertex == 10242:
#            files = sorted(glob.glob('/media/ychenp/fq/spherical_mapping/data/ADNI1/*/*/*.Sphe0.RespInner.npy'))
#        elif n_vertex == 40962:
#            files = sorted(glob.glob('/media/ychenp/fq/spherical_mapping/data/ADNI1/*/*/*.Sphe0.RespSphe.10242moved.OrigSpheMoved.RespInner.npy'))
#        elif n_vertex == 163842:
#            files = sorted(glob.glob('/media/ychenp/fq/spherical_mapping/data/ADNI1/*/*/*.Sphe0.RespSphe.10242moved.OrigSpheMoved.RespSphe.40962moved.OrigSpheMoved.RespInner.npy'))
#        else:
#            print("error")
#            
#    elif initial_mapping == 'FaSu':
#        if n_vertex == 10242:
#            # files = sorted(glob.glob('/media/ychenp/fq/spherical_mapping/data/ADNI1/*/*/*.Sphere.FaSu.RespInner.npy'))
#            files = sorted(glob.glob('/media/ychenp/fq/spherical_mapping/data/ADNI1/*/*/*.FaSu.RespSphe.10242moved.OrigSpheMoved.RespInner.npy'))
#        elif n_vertex == 40962:
#            files = sorted(glob.glob('/media/ychenp/fq/spherical_mapping/data/ADNI1/*/*/*.FaSu.RespSphe.10242moved.OrigSpheMoved.10242moved.OrigSpheMoved.RespInner.npy'))
#        elif n_vertex == 163842:
#            files = sorted(glob.glob('/media/ychenp/fq/spherical_mapping/data/ADNI1/*/*/*.FaSu.RespSphe.10242moved.OrigSpheMoved.10242moved.OrigSpheMoved.40962moved.OrigSpheMoved.RespInner.npy'))
#        else:
#            print("error")
#            
#            
#    elif initial_mapping == 'CF':
#        if n_vertex == 10242:
#            files = sorted(glob.glob('/media/ychenp/fq/spherical_mapping/data/ADNI1/*/*/*.CF.RespInner.npy'))
#        elif n_vertex == 40962:
#            files = sorted(glob.glob('/media/ychenp/fq/spherical_mapping/data/ADNI1/*/*/*.CF.10242moved.OrigSpheMoved.RespInner.npy'))
#        elif n_vertex == 163842:
#            files = sorted(glob.glob('/media/ychenp/fq/spherical_mapping/data/ADNI1/*/*/*.CF.10242moved.OrigSpheMoved.40962moved.OrigSpheMoved.RespInner.npy'))
#        else:
#            print("error")
            

test_files = [ files[x] for x in range(int(len(files)*0.2)) ]
#train_files = [ files[x] for x in range(int(len(files)*0.8), len(files)) ]
train_files = test_files

train_dataset = ResampledInnerSurf(train_files, n_vertex)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
# val_dataset = ResampledInnerSurf(test_files, n_vertex)
# val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

if n_vertex == 10242:
    level = 6
elif n_vertex == 40962:
    level = 7
elif n_vertex == 163842:
    level = 8
else:
    print("error")

model = Unet(in_ch=12, out_ch=2, level=level, n_res=4, rotated=0)
print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#############################################################

def get_learning_rate(epoch):
    limits = [100, 500, 2000]
    lrs = [1, 0.5, 0.05, 0.005]
    assert len(lrs) == len(limits) + 1
    for lim, lr in zip(limits, lrs):
        if epoch < lim:
            return lr * learning_rate
    return lrs[-1] * learning_rate


neigh_orders = get_neighs_order('/proj/ganglilab/users/Fenqiang/sunetpkg/lib/python3.7/site-packages/sphericalunet/utils/neigh_indices/adj_mat_order_'+ str(n_vertex) +'_rotated_0.mat')
neigh_sorted_orders = neigh_orders.reshape((n_vertex, 7))
neigh_sorted_orders = np.concatenate((np.arange(n_vertex)[:, np.newaxis], neigh_sorted_orders[:, 0:6]), axis=1)
template = read_vtk('/proj/ganglilab/users/Fenqiang/sunetpkg/lib/python3.7/site-packages/sphericalunet/utils/neigh_indices/sphere_'+ str(n_vertex) +'_rotated_0.vtk')
fixed_xyz = torch.from_numpy(template['vertices']).to(device) / 100.0
grad_filter = torch.ones((7, 1), dtype=torch.float32, device=device)
grad_filter[6] = -6    
bi_inter = get_bi_inter(n_vertex, device)[0]
En = getEn(n_vertex, device)[0]


# dataiter = iter(train_dataloader)
# feat, file = dataiter.next()

for epoch in range(5000):
    lr = get_learning_rate(epoch)
    optimizer.param_groups[0]['lr'] = lr
    print("learning rate = {}".format(lr))
    
    for batch_idx, (feat, file) in enumerate(train_dataloader):
        model.train()
        file = file[0]
        feat = feat.squeeze().to(device)  
        
        inner_dist, inner_area, inner_angle = computeMetrics(feat, neigh_sorted_orders, device)

        data = torch.cat((inner_dist, inner_area), 1)  # features
        
        deform_2d = model(data) / deform_scale  # 10.0 for 10242, 30.0 for 40962
        deform_3d = convert2DTo3D(deform_2d, En, device)
        # torch.linalg.norm(deform_3d, dim=1).mean()
        # deform_3d = deform_ratio.unsqueeze(2).repeat(1,1,3) * orig_sphere_vector_CtoN
        # deform_3d = torch.sum(deform_3d, dim=1)
   
        # moved_sphere_loc = fixed_xyz + deform_3d 
        # moved_sphere_loc = moved_sphere_loc / torch.linalg.norm(moved_sphere_loc, dim=1, keepdim=True).repeat(1,3)
            
        # diffeomorphic implementation
        velocity_3d = deform_3d/math.pow(2, 6)
        moved_sphere_loc = diffeomorp(fixed_xyz, velocity_3d, 
                                      num_composition=6, bi=True, 
                                      bi_inter=bi_inter, 
                                      device=device)
        
        moved_dist, moved_area, moved_angle = computeMetrics(moved_sphere_loc, neigh_sorted_orders, device)
        
        # tmp = (mvoed_oriented_area_perc < -1e-7).nonzero(as_tuple=True)
        # print("after corrected: ", tmp[0].shape[0]/3)
        # print("original negative triangle: ", ((orig_sphere_area_perc <  -1e-7).sum()/3).item(), "after corrected: ", tmp[0].shape[0]/3)
        # dist_scale = torch.sum(moved_dist * inner_dist)/torch.sum(torch.square(moved_dist))  # scale moved sphere to find the minimum distance distortion
        # print("current distance scale: ", dist_scale)
        # loss_dist = torch.sum(torch.square(moved_dist - inner_dist)) / 10000.
        # loss_area = torch.sum(torch.square(moved_area - inner_area)) / 10000.
       
        loss_dist = torch.mean(torch.abs(moved_dist - inner_dist) / (inner_dist+1e-12))
        loss_area = torch.mean(torch.abs(moved_area - inner_area) / (inner_area+1e-12))
        loss_angle = torch.mean(torch.abs(moved_angle - inner_angle) / (inner_angle+1e-12))
       
        # loss_dist = torch.mean(torch.abs(moved_dist - inner_dist))
        # loss_area = torch.mean(torch.abs(moved_area - inner_area))
        
        loss_smooth = torch.abs(torch.mm(deform_3d[:,[0]][neigh_orders].view(n_vertex, 7), grad_filter)) + \
                      torch.abs(torch.mm(deform_3d[:,[1]][neigh_orders].view(n_vertex, 7), grad_filter)) + \
                      torch.abs(torch.mm(deform_3d[:,[2]][neigh_orders].view(n_vertex, 7), grad_filter))
        loss_smooth = torch.mean(loss_smooth)
        
        loss = weight_distan * loss_dist + \
               weight_area * loss_area + \
               weight_angle * loss_angle + \
               weight_smooth * loss_smooth 
               

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print("[Epoch {}/{}] [loss_dist: {:5.4f}] [loss_area: {:5.4f}] [loss_angle: {:5.4f}] [loss_smooth: {:5.4f}]".format(epoch, 
                                        batch_idx, loss_dist,
                                        loss_area, loss_angle, loss_smooth))
        # print("[loss_dist_rel: {:5.4f}] [loss_area_rel: {:5.4f}]".format(loss_dist_rel, 
        #                                 loss_area_rel))
        
        writer.add_scalars('Train/loss', {'loss_dist': loss_dist*weight_distan,
                                          'loss_area': loss_area*weight_area,
                                          'loss_angle': loss_angle*weight_angle,
                                          'loss_smooth': loss_smooth*weight_smooth},
                                          epoch*len(train_dataloader)+batch_idx)
        
        if loss_dist > 1 or loss_area > 1:
            print(file)
        
    

    torch.save(model.state_dict(), '/pine/scr/f/e/fenqiang/dHCP/scripts/trained_model/SMnet_ver'+ str(n_vertex) + '_' + \
               dataset+ '_' +initial_mapping+'_area'+ str(weight_area) +'_angle'+ str(weight_angle) +'_dist'+ str(weight_distan) +'_smooth'+ str(weight_smooth) +'.mdl')
