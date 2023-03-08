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

from sphericalunet.models.models import SUnet
from sphericalunet.utils.interp_torch import convert2DTo3D, getEn, diffeomorp_torch, get_bi_inter
from sphericalunet.utils.interp_numpy import resampleSphereSurf
from sphericalunet.utils.utils import get_neighs_order, get_template
from sphericalunet.utils.vtk import read_vtk, write_vtk
from sphericalunet.utils.spherical_mapping import InflateSurface, projectOntoSphere, ResampledInnerSurfVtk, computeMetrics, \
    computeNegArea, computeAndWriteDistortionOnOrig, computeAndWriteDistortionOnRespSphe, computeAndSaveDistortionFile
   
abspath = os.path.abspath(os.path.dirname(__file__))


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='S3Map algorithm for mapping a cortical surface to sphere',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--file', '-f', default='None', required=True, 
                        help="the full path of the inner surface in vtk format, containing vertices and faces")
    parser.add_argument('--hemi', '-hemi', default='None', required=True, 
                        help="the hemisphere of the input inner cortical surface")
    parser.add_argument('--folder', '-folder', default='None', required=True,
                        help="a subject-specific folder for storing the output results")
    parser.add_argument('--config', '-c', default=None,
                        help="Specify the config file for spherical mapping."+\
                            " An example can be found in the same folder named as S3Map_Config_3level.yaml"+\
                                "If not given, default is 3 level with 10,242, 40,962, 163,842 vertices, respectively.")
    parser.add_argument('--model_path','-model_path', default='None', help="full path for finding all trained models")
    parser.add_argument('--device', default='GPU', choices=['GPU', 'CPU'], 
                        help='The device for running the model.')


    args = parser.parse_args()
    file = args.file
    folder = args.folder
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
    print('Starting inflating surface...')
    surf = read_vtk(file)
    surf['vertices'], surf['sulc']  = InflateSurface(surf['vertices'], surf['faces'][:, 1:], iter_num=3000, lamda=0.99, save_sulc=True, scale=True)
    write_vtk(surf, file.replace('.vtk', '.Inflated.vtk'))
    print('Surface inflation done. Inflated surface saved to ', file.replace('.vtk', '.inflated.vtk'))

    # initial spherical mapping    
    projectOntoSphere(file.replace('.vtk', '.Inflated.vtk'))
    print('Initial spherical mapping done.')

    
    # starting distortion correction    
    print("Starting distortion correction...")
    n_vertexs = [10242, 40962, 163842]
    
    for n_vertex in n_vertexs:
        n_level = n_vertexs.index(n_vertex)
        print("Multi-level distottion correction -", n_level+1, "level with", n_vertex, "vertices.")
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
  
        if level == 6:
            resp_inner_file = file.replace('.vtk', '.SIP.RespInner.vtk')
        else:
            resp_inner_file = file.replace('.vtk', '.SIP.RespSphe.'+str(n_vertexs[n_level-1])+'moved.RespInner.vtk')
        train_files = [ resp_inner_file ]
        train_dataset = ResampledInnerSurfVtk(train_files, n_vertex)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, pin_memory=True)
           
        model = SUnet(in_ch=12, out_ch=2, level=level, n_res=4, rotated=0, complex_chs=8)
        print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
        model.to(device)
        model.load_state_dict(torch.load(os.path.join(model_path, 'SMnet_ver'+str(n_vertex)+'.mdl')))
        
        neigh_orders = get_neighs_order(n_vertex)
        neigh_sorted_orders = neigh_orders.reshape((n_vertex, 7))
        neigh_sorted_orders = np.concatenate((np.arange(n_vertex)[:, np.newaxis], neigh_sorted_orders[:, 0:6]), axis=1)
        template = get_template(n_vertex)
        fixed_xyz = torch.from_numpy(template['vertices']).to(device) / 100.0
        bi_inter = get_bi_inter(n_vertex, device)[0]
        En = getEn(n_vertex, device)[0]
                    
        model.eval()
        for batch_idx, (feat, resp_inner_file) in enumerate(train_dataloader):
            # t1 = time.time()
            resp_inner_file = resp_inner_file[0]
            print("Current inner surface under processing is:", resp_inner_file)
            with torch.no_grad():
                feat = feat.squeeze().to(device)  
                inner_dist, inner_area, _ = computeMetrics(feat, neigh_sorted_orders, device)
                data = torch.cat((inner_dist, inner_area), 1)  # features
                deform_2d = model(data) / deform_scale  
                deform_3d = convert2DTo3D(deform_2d, En, device)
                velocity_3d = deform_3d/math.pow(2, 6)
                moved_sphere_loc = diffeomorp_torch(fixed_xyz, velocity_3d, 
                                                  num_composition=6, bi=True, 
                                                  bi_inter=bi_inter, 
                                                  device=device)
                
            # orig_sphere = {'vertices': fixed_xyz.cpu().numpy() * 100.0,
            #                 'faces': template['faces'],
            #                 'deformation': deform_3d.cpu().numpy() * 100.0}
            # write_vtk(orig_sphere, file.replace('.vtk', 
            #                                     '.RespSphe.'+ str(n_vertex) +'deform.vtk'))
            
            neg_area = computeNegArea(moved_sphere_loc.cpu().numpy() * 100.0, template['faces'][:, 1:])
            print("Corrected negative areas: ", neg_area)
        print('Correct distortion on', n_vertex, 'vertices is done.')
        
        
        # postprocessing
        orig_inner_surf = read_vtk(file)
        orig_sphe_surf = read_vtk(file.replace('.vtk', '.SIP.vtk'))
        orig_sphere_moved = resampleSphereSurf(template['vertices'], 
                                                orig_sphe_surf['vertices'],
                                                moved_sphere_loc.cpu().numpy() * 100.0,
                                                neigh_orders=neigh_orders)
        orig_sphere_moved = 100 * orig_sphere_moved / np.linalg.norm(orig_sphere_moved, axis=1)[:, np.newaxis]
        moved_orig_sphe_surf = {'vertices': orig_sphere_moved,
                                'faces': orig_inner_surf['faces'],
                                'sulc': orig_inner_surf['sulc']}
        write_vtk(moved_orig_sphe_surf, file.replace('.vtk', '.SIP.RespSphe.'+str(n_vertex)+'moved.OrigSpheMoved.vtk'))
        print("Move original sphere done.")
         
        template_163842 = get_template(163842)
        computeAndWriteDistortionOnRespSphe(orig_sphere_moved, template_163842, 
                                            orig_inner_surf, 
                                            file.replace('.vtk', 
                                                         '.SIP.RespSphe.'+str(n_vertex)+'moved.RespInner.vtk'))
        print("Resample original inner and sphere surface done!")
     