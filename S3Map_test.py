#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 02:39:34 2021

@author: Fenqiang Zhao

@contact: zhaofenqiang0221@gmail.com
"""

import argparse
import numpy as np
import torch
import math
import os
import shutil

from sphericalunet.model import SUnet
from sphericalunet.utils.interp_torch import convert2DTo3D, getEn, diffeomorp_torch, get_bi_inter
from sphericalunet.utils.utils import get_neighs_order
from sphericalunet.utils.vtk import read_vtk, write_vtk
from sphericalunet.utils.spherical_mapping import inflateSurface, projectionOntoSphe, ResampledInnerSurfVtk, computeMetrics, \
    computeNegArea, computeAndWriteDistortionOnOrig, computeAndWriteDistortionOnRespSphe, computeAndSaveDistortionFile
   
abspath = os.path.abspath(os.path.dirname(__file__))


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='S3Map algorithm for mapping a cortical surface to sphere',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--file', '-f', default='None', required=True, help="the full path of the inner surface in vtk format, containing vertices and faces")
    parser.add_argument('--folder', '-folder', default='None', help="a subject-specific folder for storing the output results")
    parser.add_argument('--config', '-c', default=None,  required=True,
                        help="Specify the config file for spherical mapping. An example can be found in the same folder named as S3Map_Config_3level.yaml")
    parser.add_argument('--model_path', default='None', help="full path for finding all trained models")
    parser.add_argument('--device', default='GPU', choices=['GPU', 'CPU'], 
                        help='The device for running the model.')


    args = parser.parse_args()
    file = args.file
    folder = args.folder
    n_vertex = int(args.n_vertex)
    model_path = args.model_path
    device = args.device
    model_path = args.model_path
        
    print('file: ', file)
    print('folder: ', folder)
    
    # check device
    if device == 'GPU':
        device = torch.device('cuda:0')
    elif device =='CPU':
        device = torch.device('cpu')
    else:
        raise NotImplementedError('Only support GPU or CPU device')
        
    
    if not os.path.isfile(os.path.join(folder, file.split('/')[-1])):
        shutil.copyfile(file, os.path.join(folder, file.split('/')[-1]))
    file = os.path.join(folder, file.split('/')[-1])
   
    # inflate surface
    surf = read_vtk(file)
    inflated_vertices = inflateSurface(surf['vertices'], surf['faces'][:, 1:], iter_num=200, lamda=0.8)
    surf['vertices'] = inflated_vertices
    write_vtk(surf, file.replace('.vtk', '.inflated.vtk'))
    print('Surface inflation done. Inflated surface saved to ', file.replace('.vtk', '.inflated.vtk'))

    # initial spherical mapping    
    projectionOntoSphe(file.replace('.vtk', '.inflated.vtk'))

    
    n_vertexs = [10242, 40962, 163842]
    
    
    for n_vertex in n_vertexs:
        if n_vertex == 10242:
            deform_scale = 10.0
            level = 6
        elif n_vertex == 40962:
            deform_scale = 30.0
            level = 7
        elif n_vertex == 163842:
            deform_scale = 30.0
            level = 8
        else:
            raise NotImplementedError("vertex number is not correct.")
  
        std_sphe_moved_file = file.replace('.vtk', '.SIP.RespSphe.vtk')
        orig_sphe_file = file.replace('.vtk', '.SIP.vtk')
        out_name = file.replace('.vtk', '.SIP.RespSphe.Moved.RespSphe.vtk')
        
        train_files = [ file ]
        train_dataset = ResampledInnerSurfVtk(train_files, n_vertex)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, pin_memory=True)
           
        model = SUnet(in_ch=12, out_ch=2, level=level, n_res=4, rotated=0)
        print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
        model.to(device)
        model.load_state_dict(torch.load(model_path+'SMnet_ver'+str(n_vertex)+'.mdl'))
        
        neigh_orders = get_neighs_order(n_vertex)
        neigh_sorted_orders = neigh_orders.reshape((n_vertex, 7))
        neigh_sorted_orders = np.concatenate((np.arange(n_vertex)[:, np.newaxis], neigh_sorted_orders[:, 0:6]), axis=1)
        template = read_vtk(abspath+'/neigh_indices/sphere_'+ str(n_vertex) +'_rotated_0.vtk')
        fixed_xyz = torch.from_numpy(template['vertices']).to(device) / 100.0
        bi_inter = get_bi_inter(n_vertex, device)[0]
        En = getEn(n_vertex, device)[0]
                    
        model.eval()
        for batch_idx, (feat, file) in enumerate(train_dataloader):
            # t1 = time.time()
            file = file[0]
            print(file)
            with torch.no_grad():
                feat = feat.squeeze().to(device)  
                inner_dist, inner_area, _ = computeMetrics(feat, neigh_sorted_orders, device)
                data = torch.cat((inner_dist, inner_area), 1)  # features
                deform_2d = model(data) / deform_scale  
                deform_3d = convert2DTo3D(deform_2d, En, device)
                velocity_3d = deform_3d/math.pow(2, 6)
                moved_sphere_loc = diffeomorp(fixed_xyz, velocity_3d, 
                                              num_composition=6, bi=True, 
                                              bi_inter=bi_inter, 
                                              device=device)
                
            orig_surf = read_vtk(file)
    
            orig_sphere = {'vertices': fixed_xyz.cpu().numpy() * 100.0,
                            'faces': template['faces'],
                            'deformation': deform_3d.cpu().numpy() * 100.0}
            write_vtk(orig_sphere, file.replace('.vtk', 
                                                '.RespSphe.'+ str(n_vertex) +'deform.vtk'))
            
            corrected_sphere = {'vertices': moved_sphere_loc.cpu().numpy() * 100.0,
                                'faces': template['faces']}
            neg_area = computeNegArea(corrected_sphere['vertices'], corrected_sphere['faces'][:, 1:])
            print("corrected negative areas: ", neg_area)
            write_vtk(corrected_sphere, file.replace('.vtk', 
                                                     '.RespSphe.'+ str(n_vertex) +'moved.vtk'))
        
        print('Coreect distortion on', n_vertex, 'level is done.')
        
        
        # postprocessing
        moved_surf = read_vtk(std_sphe_moved_file)
        orig_surf = read_vtk(orig_sphe_file)
        inner_surf = read_vtk(file)
        orig_sphere_moved_ver = computeAndWriteDistortionOnOrig(template, orig_surf, moved_surf, inner_surf, neigh_orders, out_name)
        print("original sphere moved done!")
         
        template_163842 = read_vtk(abspath+'/neigh_indices/sphere_163842_rotated_0.vtk')
        computeAndWriteDistortionOnRespSphe(orig_sphere_moved_ver, template_163842, inner_surf, out_name.replace('.vtk', '.RespInner.vtk'))
        print("original sphere moved and resampled done!")
     