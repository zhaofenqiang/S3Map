#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 01:44:25 2021

@author: Fenqiang Zhao
"""

import torch
import os
import numpy as np

from functools import reduce

from .vtk import read_vtk, write_vtk
from .interp_numpy import resampleSphereSurf
from .utils import get_neighs_order

abspath = os.path.abspath(os.path.dirname(__file__))


neigh_orders_163842 = get_neighs_order(163842)
neigh_sorted_orders_163842 = neigh_orders_163842.reshape((163842, 7))
neigh_sorted_orders_163842 = np.concatenate((np.arange(163842)[:, np.newaxis], neigh_sorted_orders_163842[:, 0:6]), axis=1)

# neigh_orders_40962 = get_neighs_order(40962)
# neigh_sorted_orders_40962 = neigh_orders_40962.reshape((40962, 7))
# neigh_sorted_orders_40962 = np.concatenate((np.arange(40962)[:, np.newaxis], neigh_sorted_orders_40962[:, 0:6]), axis=1)

# neigh_orders_10242 = get_neighs_order(10242)
# neigh_sorted_orders_10242 = neigh_orders_10242.reshape((10242, 7))
# neigh_sorted_orders_10242 = np.concatenate((np.arange(10242)[:, np.newaxis], neigh_sorted_orders_10242[:, 0:6]), axis=1)

# template_10242 = read_vtk(abspath + '/neigh_indices/sphere_10242_rotated_0.vtk')
# template_40962 = read_vtk(abspath + '/neigh_indices/sphere_40962_rotated_0.vtk')
template_163842 = read_vtk(abspath + '/neigh_indices/sphere_163842_rotated_0.vtk')



def inflateSurface(vertices, faces, iter_num=20, lamda=0.8, vertex_has_faces=None):
    """
    
    using the formula in 2.1 in "a quantitative comparison of three methods for inflating cortical meshes"
    V_t = (1-lamda) * V + lamda * V_N_i
    
    faces shape is Nx3

    """
    num_vertices = len(vertices)
    
    if vertex_has_faces == None:
        vertex_has_faces = []
        for j in range(num_vertices):
            vertex_has_faces.append([])
        for j in range(len(faces)):
            face = faces[j]
            vertex_has_faces[face[0]].append(j)
            vertex_has_faces[face[1]].append(j)
            vertex_has_faces[face[2]].append(j)
    max_len = 0
    for i in range(num_vertices):
        if len(vertex_has_faces[i]) > max_len:
            max_len = len(vertex_has_faces[i]) 
    print("max_len: ", max_len)
    
    neighs = np.zeros((num_vertices, max_len), dtype=np.int32) + len(faces)
    for i in range(num_vertices):
        neighs[i, 0:len(vertex_has_faces[i])] = np.array(vertex_has_faces[i])
    
    v_t = vertices
    for i in range(iter_num):
        face_center = v_t[faces].mean(1)
        face_area = computeFaceWiseArea(v_t, faces)
        face_cen_weighted = face_area[:, np.newaxis].repeat(3,1) * face_center
        face_area = np.append(face_area, 0)
        face_cen_weighted = np.concatenate((face_cen_weighted, np.array([[0,0,0]])), axis=0)
        face_center = face_cen_weighted[neighs].sum(1) / face_area[neighs].sum(1, keepdims=True)
        
        v_t = (1 - lamda) * v_t + lamda * face_center
            
    return v_t



def projectionOntoSphe(file, compute_distortion=False):
    """
    Surface inflation and projection for spherical mapping of cortical surfaces
    Here is the projection and resampling code

    Parameters
    ----------
    file : should be sufficently inflated surface, ending in ".inflated.vtk"

    Returns
    -------
    None.

    """
    print("Starting projection onto sphere...")
    print(file)
    
    inflated_surf = read_vtk(file)
    if compute_distortion:
        inner_surf = read_vtk(file.replace('.inflated.vtk', '.vtk'))
        assert len(inflated_surf['vertices']) == len(inner_surf['vertices'])
        assert (inner_surf['faces'] == inflated_surf['faces']).sum() == len(inflated_surf['faces']) *4
    
    inflated_ver = inflated_surf['vertices']
    
    inflated_ver = inflated_ver - inflated_ver.mean(0)     #centralize
    inflated_ver = 100 * inflated_ver / np.linalg.norm(inflated_ver, axis=1).mean()  #normalize to 100 cortical shape
    sphere_0_ver = 100 * inflated_ver / np.linalg.norm(inflated_ver, axis=1)[:, np.newaxis].repeat(3, axis=1)
    
    if compute_distortion:
        computeAndWriteDistortionOnOrigSphe(sphere_0_ver, inner_surf, file.replace('.inflated.vtk', '.SIP.vtk'))
        print('projection onto sphere done') 
        computeAndWriteDistortionOnRespSphe(sphere_0_ver, template_163842, inner_surf,
                                            file.replace('.inflated.vtk', '.SIP.RespInner.vtk'))
        print('projection onto sphere resampling done')
    else:
        inflated_surf['vertices'] = sphere_0_ver
        write_vtk(inflated_surf, file.replace('.inflated.vtk', '.SIP.vtk'))
        print('projection onto sphere done') 

    

def computeAndWriteDistortionOnOrig(template, orig_surf, moved_surf, inner_surf, neigh_orders, out_name):
    orig_sphere_moved = resampleSphereSurf(template['vertices'], 
                                           orig_surf['vertices'],
                                           moved_surf['vertices'],
                                           neigh_orders=neigh_orders)
    orig_sphere_moved = 100 * orig_sphere_moved / np.linalg.norm(orig_sphere_moved, axis=1)[:, np.newaxis]
    computeAndWriteDistortionOnOrigSphe(orig_sphere_moved, inner_surf, out_name)
    return orig_sphere_moved


def computeAndWriteDistortionOnOrigSphe(orig_sphere_moved, inner_surf, file_name):
    faces = inner_surf['faces'][:, 1:]
    num_neg_tri = computeNegArea(orig_sphere_moved, faces)
    print("negative areas of original sphere: ", num_neg_tri)
    angle_dis, dist_dis, area_dis = computeDistortionOnOrigMesh(inner_surf['vertices'], orig_sphere_moved, faces)
    orig_sphere_surf = {'vertices': orig_sphere_moved,
                        'faces': inner_surf['faces'],
                        'sulc':  inner_surf['sulc'],
                        'curv':  inner_surf['curv'],
                        'angle_dis': angle_dis,
                        'dist_dis': dist_dis,
                        'area_dis': area_dis}
    write_vtk(orig_sphere_surf, file_name)
    
    
def computeAndWriteDistortionOnRespSphe(orig_sphere_moved, template, inner_surf, file_name, compute_distortion=False):
    resampled_feat = resampleSphereSurf(orig_sphere_moved, template['vertices'], 
                                        np.concatenate((inner_surf['vertices'], 
                                                        inner_surf['sulc'][:, np.newaxis],
                                                        inner_surf['curv'][:, np.newaxis]), axis=1),
                                        faces=inner_surf['faces'], threshold=1e-8, ring_threshold=4)
    if compute_distortion:
        dist_dis, area_dis = computeDistortionOnRegularMesh(resampled_feat[:, 0:3], template['vertices'], neigh_sorted_orders_163842)
    
    resampled_inner_surf = {'vertices': resampled_feat[:, 0:3],
                            'faces': template['faces'],
                            'sulc': resampled_feat[:,-2],
                            'curv': resampled_feat[:,-1],
                            'dist_dis': dist_dis,
                            'area_dis': area_dis
                            }
    write_vtk(resampled_inner_surf, file_name)
    
    # in case that resampled sphere already exists
    if os.path.exists(file_name.replace('RespInner.vtk', 'RespSphe.vtk')):
        resampled_sphere_surf = read_vtk(file_name.replace('RespInner.vtk', 'RespSphe.vtk'))
    else:
        resampled_sphere_surf = {'vertices': template['vertices'],
                                 'faces': template['faces']
                                }
    resampled_sphere_surf['sulc'] = resampled_feat[:,-2]
    resampled_sphere_surf['curv'] = resampled_feat[:,-1]
    resampled_sphere_surf['dist_dis'] = area_dis
    resampled_sphere_surf['area_dis'] = area_dis
    
    write_vtk(resampled_sphere_surf, file_name.replace('RespInner.vtk', 'RespSphe.vtk'))
    
        

def computeAndSaveDistortionFile(file1, file2):
    """
    

    file1 is the original surface

    """
    surf1 = read_vtk(file1)
    surf2 = read_vtk(file2)
    
    vert1 = surf1['vertices']
    vert2 = surf2['vertices']
    face1 = surf1['faces']
    face2 = surf2['faces']
    
    assert len(vert1) == len(vert2), 'error'
    assert (face1 == face2).sum() == len(face1) * 4, 'error'
    
    faces = face1[:, 1:]
    dist_dis, angle_dis, area_dis = evaluateDistortionBetweenTwoSurfaces(vert1, vert2, faces)  # vert1 is original surface
        
    dic = {'dist_dis': dist_dis,
           'angle_dis': angle_dis,
           'area_dis': area_dis}
    
    np.save(file2.replace('.vtk', '.npy'), dic)
    


class InflatedSurf(torch.utils.data.Dataset):

    def __init__(self, files):
        self.files = files

    def __getitem__(self, index):
        file = self.files[index]
        data = np.load(file)
        
        ver_loc = data[:, 0:3].astype(np.float32)
        ver_loc = ver_loc - ver_loc.mean(0)     #centralize
        ver_loc = ver_loc / np.linalg.norm(ver_loc, axis=1).mean()  #normalize to 1 cortical shape
        
        neigh_sorted_orders = data[:, 3:].astype(np.int64)
        neigh_sorted_orders = np.concatenate((np.arange(len(ver_loc))[:, np.newaxis], 
                                              neigh_sorted_orders), axis=1)
        
        return ver_loc, neigh_sorted_orders, file

    def __len__(self):
        return len(self.files)
    
    
class ResampledInnerSurf(torch.utils.data.Dataset):

    def __init__(self, files, n_vertex):
        self.files = files
        self.n_vertex = n_vertex

    def __getitem__(self, index):
        file = self.files[index]
        data = np.load(file)
        
        feat = data.astype(np.float32)
        feat = feat - feat.mean(0)     #centralize
        feat = feat / np.linalg.norm(feat, axis=1).mean() # normalize size
        
        return feat[0:self.n_vertex, :], file

    def __len__(self):
        return len(self.files)
    
    
class ResampledInnerSurfVtk(torch.utils.data.Dataset):

    def __init__(self, files, n_vertex):
        self.files = files
        self.n_vertex = n_vertex

    def __getitem__(self, index):
        file = self.files[index]
        data = read_vtk(file)
        data = data['vertices']
        feat = data.astype(np.float32)
        feat = feat - feat.mean(0)     #centralize
        feat = feat / np.linalg.norm(feat, axis=1).mean() # normalize size
        
        return feat[0:self.n_vertex, :], file

    def __len__(self):
        return len(self.files)
    

        
def computeMetrics(vertices, neigh_sorted_orders, device, NUM_NEIGHBORS=6):
    """
    compute the metrics of surfaces for constructing the objective/loss function

    Parameters
    ----------
    vertices : TYPE
        DESCRIPTION.
    neigh_sorted_orders : Nx(NUM_NEIGHBORS+1)
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    vector_CtoN = vertices[neigh_sorted_orders][:, 1:, :] - vertices[neigh_sorted_orders][:, [0], :].repeat(1, NUM_NEIGHBORS, 1)
    
    # geodesic distance, Nx6, center vetex to neighbor vertex distance
    distan = torch.linalg.norm(vector_CtoN, dim=2)
    distan[0:12, -1] = distan[0:12, 0:-1].mean(1)
    
    # triangle area, NxNUM_NEIGHBORS
    area = torch.zeros((vertices.shape[0], NUM_NEIGHBORS), device=device)
    angle = torch.zeros((vertices.shape[0], NUM_NEIGHBORS), device=device)
    for i in range(NUM_NEIGHBORS):
        c = vertices
        if i < NUM_NEIGHBORS-1:
            a = vertices[neigh_sorted_orders][:, i+1, :]
            b = vertices[neigh_sorted_orders][:, i+2, :]
            angle[:, i] = torch.acos(torch.clamp(torch.sum((a-c)*(b-c),dim=1)/distan[:,i]/distan[:,i+1], min=-0.9999999, max=0.9999999))
        else:
            a = vertices[neigh_sorted_orders][:, -1, :]
            a[0:12, :] = vertices[neigh_sorted_orders][0:12, -2, :]
            b = vertices[neigh_sorted_orders][:, 1, :]
            angle[:, i] = torch.acos(torch.clamp(torch.sum((a-c)*(b-c),dim=1)/distan[:,-1]/distan[:,0],min=-0.9999999, max=0.9999999))
        cros_vec = torch.cross(a-c, b-c, dim=1)
        area[:, i] = 1/2 * torch.linalg.norm(cros_vec, dim=1)
   
    # normalize distance and area using total distance and area
    distan_perc = distan/distan.sum() * 100000.0
    area_perc = area/area.sum() * 100000.0
    
    # normalize angle using market share 
    angle = angle / angle.sum(1, keepdim=True)
    
    return distan_perc, area_perc, angle


def computeMetrics_np(vertices, neigh_sorted_orders, NUM_NEIGHBORS=6):
    vector_CtoN = vertices[neigh_sorted_orders][:, 1:, :] - np.repeat(vertices[neigh_sorted_orders][:, [0], :], NUM_NEIGHBORS, axis=1)
    
    # geodesic distance, Nx6, center vetex to neighbor vertex distance
    distan = np.linalg.norm(vector_CtoN, axis=2)
    
    # triangle area, NxNUM_NEIGHBORS
    area = np.zeros((vertices.shape[0], NUM_NEIGHBORS))
    for i in range(NUM_NEIGHBORS):
        if i < NUM_NEIGHBORS-1:
            a = vertices[neigh_sorted_orders][:, i+1, :]
            b = vertices[neigh_sorted_orders][:, i+2, :]
        else:
            a = vertices[neigh_sorted_orders][:, -1, :]
            b = vertices[neigh_sorted_orders][:, 1, :]
        c = vertices
        cros_vec = np.cross(a-c, b-c)
        area[:, i] = 1/2 * np.linalg.norm(cros_vec, axis=1)

    # compute angles, for miccai 2022 figures
    # vector_CtoN[0:12, -1, 0] = 1
    # angle = np.zeros((vertices.shape[0], NUM_NEIGHBORS))
    # for i in range(NUM_NEIGHBORS):
    #     if i < NUM_NEIGHBORS-1:
    #         v01 = vector_CtoN[:, i, :]
    #         v02 = vector_CtoN[:, i+1, :]
    #     else:
    #         v01 = vector_CtoN[:, -1, :]
    #         v02 = vector_CtoN[:, 0, :]

    #     angle[:, i] = np.arccos(np.clip(np.sum(v01 * v02, axis=1) / np.linalg.norm(v01, axis=1) / np.linalg.norm(v02, axis=1), -1, 1))
    # # normalize angles for each vertex
    # angle = angle / angle.sum(1, keepdims=True) * 360
    
    # return distan/distan.sum() * 100000.0, area/area.sum() * 100000.0, angle
    return distan/distan.sum() * 100000.0, area/area.sum() * 100000.0


    
def computeNegArea(vertices, faces):
    """
    simple code for checking triangles intersections on sphere,
    only work for sphere, may work for inflated surface,
    but not work for inner surface
    """
    assert faces.shape[1] == 3, "faces' shape[1] should be Nx3, not Nx4"
    vertices = vertices - vertices.mean(0)
    v = vertices[faces]
    a = v[:, 0, :]
    b = v[:, 1, :]
    c = v[:, 2, :]
    area = 1/2* np.sum(np.multiply(np.cross(b-a, c-a), a/np.linalg.norm(a, axis=1, keepdims=True)), axis=1)
    return (area < -1e-10).sum()


def computeAngles(vertices, faces, vertex_has_angles):
    num_vers = vertices.shape[0]
    num_faces = faces.shape[0]
    angle = np.zeros((num_faces, 3))
    
    v1 = vertices[faces[:,0], :]
    v2 = vertices[faces[:,1], :]
    v3 = vertices[faces[:,2], :]
    
    v0_12 = v2 - v1
    v0_23 = v3 - v2
    v0_13 = v3 - v1
    
    tmp = reduce(np.intersect1d, (np.where(np.linalg.norm(v0_12, axis=1) > 0), 
                                  np.where(np.linalg.norm(v0_13, axis=1) > 0), 
                                  np.where(np.linalg.norm(v0_23, axis=1) > 0)))

    angle[:, 0][tmp] = np.arccos(np.clip(np.sum(v0_12 * v0_13, axis=1)[tmp] / np.linalg.norm(v0_12, axis=1)[tmp] / np.linalg.norm(v0_13, axis=1)[tmp], -1, 1))
    angle[:, 1][tmp] = np.arccos(np.clip(np.sum(-v0_12 * v0_23, axis=1)[tmp] / np.linalg.norm(-v0_12, axis=1)[tmp] / np.linalg.norm(v0_23, axis=1)[tmp], -1, 1))
    angle[:, 2][tmp] = np.arccos(np.clip(np.sum(v0_23 * v0_13, axis=1)[tmp] / np.linalg.norm(v0_23, axis=1)[tmp] / np.linalg.norm(v0_13, axis=1)[tmp], -1, 1))
    angle = np.reshape(angle, (num_faces*3, 1))
    angle[np.where(angle==0)] = angle.mean()
    
    # normalize angles for each vertex
    for j in range(num_vers):
        tmp = angle[vertex_has_angles[j]].sum()
        angle[vertex_has_angles[j]] = angle[vertex_has_angles[j]] / tmp
        
    return angle  


def computeFaceWiseArea(vertices, faces):
    """
    compute face-wise area

    Parameters
    ----------
    vertices : TYPE
        DESCRIPTION.
    faces : TYPE
        DESCRIPTION.

    Returns
    -------
    area : TYPE
        DESCRIPTION.

    """
    v1 = vertices[faces[:,0], :]
    v2 = vertices[faces[:,1], :]
    v3 = vertices[faces[:,2], :]
    area = np.linalg.norm(np.cross(v2-v1, v3-v1), axis=1)/2.0
    return area         


def computeDistance(vertices, edges):
    v1 = vertices[edges[:,0], :]
    v2 = vertices[edges[:,1], :]
    dis_12 = np.linalg.norm(v2 - v1, axis=1)
    return dis_12


def computeDistortionOnRegularMesh(ver1, ver2, neigh_sorted_orders):
    """
    assume ver1 is inner or inflated surface, ver2 are sphere surface
    
    This order will affect results when computing relative distortion where 
    ver1 will be divided.

    Parameters
    ----------
    ver1 : TYPE
        DESCRIPTION.
    ver2 : TYPE
        DESCRIPTION.
    neigh_sorted_orders : TYPE
        DESCRIPTION.

    Returns
    -------
    dist_dis, area_dis, 0:6 is for each triangle of each vertex
    7 dim is for total for each vertex.

    """
    n_vertex = len(ver1)
    n_vertex2 = len(ver2)
    assert n_vertex == n_vertex2
    assert n_vertex in [2562, 10242, 40962, 163842]
    dist_dis = np.zeros((n_vertex, 7))
    area_dis = np.zeros((n_vertex, 7))
    dist1, area1 = computeMetrics_np(ver1, neigh_sorted_orders)
    dist2, area2 = computeMetrics_np(ver2, neigh_sorted_orders)
    
    tmp = np.where(dist1==0)
    if len(tmp[0])/n_vertex > 0.002:
        print("A lot of values is 0 in distance", len(tmp[0]))
    dist1[tmp] = dist1.mean()
    
    tmp = np.where(dist2==0)
    if len(tmp[0])/n_vertex > 0.002:
        print("A lot of values is 0 in distance", len(tmp[0]))
    dist2[tmp] = dist2.mean()
    
    tmp = np.where(area1==0)
    if len(tmp[0])/n_vertex > 0.002:
        print("A lot of values is 0 in area", len(tmp[0]))
    area1[tmp] = area1.mean()
    
    tmp = np.where(area2==0)
    if len(tmp[0])/n_vertex > 0.002:
        print("A lot of values is 0 in area", len(tmp[0]))
    area2[tmp] = area2.mean()
    
    dist_dis[:, 0:6] = (dist2 - dist1)/dist1
    area_dis[:, 0:6] = (area2 - area1)/area1
    dist_dis[:, -1] = np.mean(dist_dis[:, 0:6], axis=1)
    area_dis[:, -1] = np.mean(area_dis[:, 0:6], axis=1)
    
    return dist_dis, area_dis



def evaluateDistortionBetweenTwoSurfaces(vert1, vert2, faces):
    """
    Difference with computeDistortionOnOrigMesh (vertex-wise): edge-weise, angle-wise, area-wise

    Parameters
    ----------
    vert1 : TYPE
        DESCRIPTION.
    vert2 : TYPE
        DESCRIPTION.
    faces : TYPE
        DESCRIPTION.

    Returns
    -------
    dist_dis : TYPE
        DESCRIPTION.
    angle_dis : TYPE
        DESCRIPTION.
    area_dis : TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """
    
    num_vers = vert1.shape[0]
    num_faces = faces.shape[0]
    
    # angle distortion
    vertex_has_angles = []
    for j in range(num_vers):
        vertex_has_angles.append([])
    for j in range(num_faces):
        face = faces[j]
        vertex_has_angles[face[0]].append(j*3)
        vertex_has_angles[face[1]].append(j*3+1)
        vertex_has_angles[face[2]].append(j*3+2)
    angle1 = computeAngles(vert1, faces, vertex_has_angles)
    angle2 = computeAngles(vert2, faces, vertex_has_angles)
    angle_dis = (angle2 - angle1)*360
    
    # area distortion
    area1 = computeArea(vert1, faces)
    area2 = computeArea(vert2, faces)
    area_scale = area1.sum()/area2.sum()
    # print("area_scale: ", area_scale)
    area_dis = (area2*area_scale - area1)/area1
    
    # edge distance distortion
    edges = np.zeros((num_faces*3, 2), dtype=np.int64) - 1
    for j in range(num_faces):
        face = faces[j]
        edges[j*3] = [face[0], face[1]]
        edges[j*3+1] = [face[1], face[2]]
        edges[j*3+2] = [face[0], face[2]]
    dist1 = computeDistance(vert1, edges)
    dist2 = computeDistance(vert2, edges)
    dist_scale = dist1.sum()/dist2.sum()
    # print("dist_scale: ", dist_scale)
    dist_dis = (dist2*dist_scale - dist1)/dist1
    
    print("dist_dis, angle_dis, area_dis: ", np.mean(np.abs(dist_dis)), np.mean(np.abs(angle_dis)), np.mean(np.abs(area_dis)))
    return dist_dis, angle_dis, area_dis



def computeDistortionOnOrigMesh(inflated_ver, sphere_ver, faces):
    """
    

    Parameters
    ----------
    inflated_ver : TYPE
        DESCRIPTION.
    sphere_ver : TYPE
        DESCRIPTION.
    faces : N x 3
        faces.shape[1] == 3

    Returns
    -------
    angle_dis : TYPE
        DESCRIPTION.
    dist_dis : TYPE
        DESCRIPTION.
    area_dis : TYPE
        DESCRIPTION.

    """
    assert faces.shape[1] == 3
    
    num_vers = inflated_ver.shape[0]
    num_faces = faces.shape[0]
    
    # angle distortion
    vertex_has_angles = []
    for j in range(num_vers):
        vertex_has_angles.append([])
    for j in range(num_faces):
        face = faces[j]
        vertex_has_angles[face[0]].append(j*3)
        vertex_has_angles[face[1]].append(j*3+1)
        vertex_has_angles[face[2]].append(j*3+2)
        
    angle1 = computeAngles(inflated_ver, faces, vertex_has_angles)
    angle2 = computeAngles(sphere_ver, faces, vertex_has_angles)
    angle_dis = np.zeros(num_vers)
    for j in range(num_vers):
        a1 = angle1[vertex_has_angles[j]]
        a2 = angle2[vertex_has_angles[j]]
        angle_dis[j] = np.mean(np.abs(a1-a2)) * 360
    
    
    # area distortion
    vertex_has_faces = []
    for j in range(num_vers):
        vertex_has_faces.append([])
    for j in range(num_faces):
        face = faces[j]
        vertex_has_faces[face[0]].append(j)
        vertex_has_faces[face[1]].append(j)
        vertex_has_faces[face[2]].append(j)
        
    # check no repeat faces
    # for l in vertex_has_faces:
    #     if len(set(l)) != len(l):
    #         print("error!")

    sphere_area = computeArea(sphere_ver, faces)
    inflated_area = computeArea(inflated_ver, faces)
    sphere_area = sphere_area*(inflated_area.sum()/sphere_area.sum())
    
    sphere_area_vert = np.zeros(num_vers)
    inflated_area_vert = np.zeros(num_vers)
    for j in range(num_vers):
        sphere_area_vert[j] = sphere_area[vertex_has_faces[j]].sum()
        inflated_area_vert[j] = inflated_area[vertex_has_faces[j]].sum()
    area_dis = (sphere_area_vert - inflated_area_vert) / inflated_area_vert


    # edge distance distortion
    edges = np.zeros((num_faces*3, 2), dtype=np.int64) - 1
    for j in range(num_faces):
        face = faces[j]
        edges[j*3] = [face[0], face[1]]
        edges[j*3+1] = [face[1], face[2]]
        edges[j*3+2] = [face[0], face[2]]
    
    vertex_has_edges = []
    for j in range(num_vers):
        vertex_has_edges.append([])
    for j in range(len(edges)):
        e = edges[j]
        vertex_has_edges[e[0]].append(j)
        vertex_has_edges[e[1]].append(j)
        
    # check edges times=2
    # for j in vertex_has_edges:
    #     if len(j) % 2 != 0:
    #         print('error')
        
    sphere_dis_12 = computeDistance(sphere_ver, edges)
    inflated_dis_12 = computeDistance(inflated_ver, edges)
    sphere_dis_12 = sphere_dis_12 * (inflated_dis_12.sum()/sphere_dis_12.sum())

    sphere_dis_vert = np.zeros(num_vers)
    inflated_dis_vert = np.zeros(num_vers)
    for j in range(num_vers):
        sphere_dis_vert[j] = sphere_dis_12[vertex_has_edges[j]].sum()/2.0
        inflated_dis_vert[j] = inflated_dis_12[vertex_has_edges[j]].sum()/2.0
    dist_dis = (sphere_dis_vert - inflated_dis_vert) / inflated_dis_vert

    return angle_dis, dist_dis, area_dis


