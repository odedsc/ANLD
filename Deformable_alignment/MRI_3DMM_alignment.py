# import packages
from Alignment_functions import *

import numpy as np
import trimesh
import networkx as nx
import pandas as pd
from datetime import datetime
import os
from scipy.optimize import minimize
import open3d as o3d
import sys
from sklearn.neighbors import NearestNeighbors

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset',
    type=str,
    default='ADNI',
    help='MRI datasets: IXI / ADNI')
parser.add_argument(
    '--region',
    type=str,
    default='all',
    help='The region we align according to: all/head or face')
parser.add_argument(
    '--subject',
    type=int,
    default=-1,
    help='The subject we align: subject index or all subjects (-1)')
parser.add_argument(
    '--iterations',
    type=int,
    default=3,
    help='Number of iterations')
parser.add_argument(
    '--save',
    action='store_true',
    default=False,
    help='Save output: True or False')

args = parser.parse_args()

print(f'Running alignment for {args.dataset} dataset over {args.region} region')

# # Loading the 3DMM

models_folder = './datasets/TMS/3DMM/UHM_models/'
modules_folder = '/Landmarks and masks/'

# Original model including everything
UHM = load_3DMM(models_folder, modules_folder, False)

# Lighter model including everything but eyes, teeth and inner mouth cavity
light_UHM = load_3DMM(models_folder, modules_folder, True)

# Matching landmark indices between models
light_UHM_facial_landmarks = []

for current_landmark_index, current_landmark_vertex in enumerate(UHM['Landmarks']['indices']):
    original_model_coordinates = UHM['Mean_CCS'][current_landmark_vertex]
    new_model_vertex_diffs = np.linalg.norm(light_UHM['Mean_CCS']-original_model_coordinates, axis=1)
    light_UHM['Landmarks']['indices'].append(np.argmin(new_model_vertex_diffs))
    light_UHM['Landmarks']['names'].append(UHM['Landmarks']['names'][current_landmark_index])
light_UHM['Landmarks']['cranial_landmark_indices']=np.arange(0, len(light_UHM['Landmarks']['indices'])-len(UHM['Landmarks']['indices']))
light_UHM['Landmarks']['facial_landmark_indices']=np.arange(0, len(UHM['Landmarks']['indices']))+light_UHM['Landmarks']['cranial_landmark_indices'].size

nasion_index = 52241
light_UHM['Landmarks']['indices'].append(nasion_index)
light_UHM['Landmarks']['names'].append('nasion')

inion_index = 36323
light_UHM['Landmarks']['indices'].append(36323)
light_UHM['Landmarks']['names'].append('inion')

light_UHM['Landmarks']['additional_landmark_indices']=np.arange(len(light_UHM['Landmarks']['indices'])-2, len(light_UHM['Landmarks']['indices']))

light_UHM['Trimesh'] = trimesh.Trimesh(vertices=light_UHM['Mean_CCS'], faces=light_UHM['Trilist'], process=False)

cranial_annotation_indices = [7, 13, 89, 90]
input_landmarks = ['1', '2', '3', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27',
                   '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42',
                   '43', '44', '45', '46', '47', '48']
input_landmarks = [int(current_landmark) for current_landmark in input_landmarks]

# # MR image

# ## Definitions

# ### Parameters

registration_scale_factor = 0.001

shape_all_cranial_landmark_indices = cranial_annotation_indices

skin_estimated_density = 0.5e-3

correspondent_max_distance_threshold = 25*skin_estimated_density
correspondent_max_distance_ratio_threshold = 2

neighbors_radius = 2*skin_estimated_density

use_perpendicular_plane_distance = True

use_injective_correspondence = True

mask_exp_coeff = 1

downsampling_rate = 1

enable_early_stop = True

weights_lambda = 1e-3
lambdas = [weights_lambda]

light_UHM['Mean'] = light_UHM['Mean']*registration_scale_factor
light_UHM['Mean_CCS'] = light_UHM['Mean_CCS']*registration_scale_factor
light_UHM['Eigenvectors_CCS'] = light_UHM['Eigenvectors_CCS']*registration_scale_factor
light_UHM['Trimesh'].vertices = light_UHM['Trimesh'].vertices*registration_scale_factor

dataset_folder = f'./datasets/TMS/MRI_datasets/{args.dataset}/'
images_folder = dataset_folder+'Processed/'
all_subjects_folders = os.listdir(images_folder)
all_subjects_folders.sort()

if args.region=='all' or args.region=='head':
    region_folder = 'Data_Chamfer_10/'
else:
    region_folder = 'Face_based_Chamfer_distances/'

for selected_subject_index in np.arange(len(all_subjects_folders)):
    selected_subject = all_subjects_folders[selected_subject_index]
    if args.subject==-1 and os.path.exists(dataset_folder+region_folder+selected_subject+'.xlsx'):
        print(f"Already have subject: {selected_subject} data")
        continue
    if args.subject!=-1:
        selected_subject = all_subjects_folders[args.subject]
        selected_subject_index = args.subject
    #try:
    print(f"Started working on subject: {selected_subject}, {selected_subject_index+1}/{len(all_subjects_folders)}")
    current_subject_start_time = datetime.now()
    
    if args.dataset=='IXI':
        selected_reconstruction = 'm2m_' + selected_subject + '_no_cat/' #'_only_T1_no_cat/' / '_no_cat/
    else: #args.dataset=='ADNI':
        if 'Scaled_2' in selected_subject:
            selected_reconstruction = 'm2m_' + '_'.join(selected_subject.split('_')[:3]) + '_no_cat/' #'_only_T1_no_cat/' / '_no_cat/
        else:
            selected_reconstruction = 'm2m_' + selected_subject + '_no_cat/' #'_only_T1_no_cat/' / '_no_cat/
            
    #segments_folder = 'm2m_mesh/'
    selected_folder = images_folder + selected_subject + '/' + selected_reconstruction# + segments_folder

    selected_segment_filename = selected_folder + 'skin.stl'

    MRI = {}
    MRI['Mesh'] = trimesh.load_mesh(selected_segment_filename)
    skin_mesh_vertices = MRI['Mesh'].vertices
    skin_mesh_vertices[:, 0] = -1*skin_mesh_vertices[:, 0]
    skin_mesh_vertices[:, [2, 1]] = skin_mesh_vertices[:, [1, 2]]
    MRI['Mesh'].vertices = skin_mesh_vertices
    MRI['Mesh'].vertices = np.array(MRI['Mesh'].vertices)*registration_scale_factor

    vertical_crop_coefficient = 0.15
    shape_crop_indices, shape_y_value_crop, _ = vertical_crop_by_coefficient(light_UHM['Mean_CCS'], vertical_crop_coefficient)
    skin_crop_indices, skin_y_value_crop, _ = vertical_crop_by_coefficient(np.array(MRI['Mesh'].vertices), vertical_crop_coefficient)

    if args.region=='all' or args.region=='head':
        shape_EEG_10_20_landmark_indices_after_crop, shape_EEG_10_20_landmarks_remain_after_crop = find_new_indices(
            light_UHM['Mean_CCS'], shape_crop_indices,
            np.take(light_UHM['Landmarks']['indices'], light_UHM['Landmarks']['cranial_landmark_indices']))
            

        shape_facial_landmark_indices_after_crop, shape_facial_landmarks_remain_after_crop = find_new_indices(
            light_UHM['Mean_CCS'], shape_crop_indices,
            np.take(light_UHM['Landmarks']['indices'], light_UHM['Landmarks']['facial_landmark_indices']))

        shape_cranial_landmark_indices_after_crop, shape_cranial_landmarks_remain_after_crop = find_new_indices(
            light_UHM['Mean_CCS'], shape_crop_indices,
            np.take(light_UHM['Landmarks']['indices'], shape_all_cranial_landmark_indices))
        
        shape_crop_indices = np.delete(shape_crop_indices,
                                np.concatenate((shape_EEG_10_20_landmark_indices_after_crop, shape_facial_landmark_indices_after_crop, 
                                                shape_cranial_landmark_indices_after_crop)).astype(int))
    
    skin_downsampling_indices = np.unique(np.random.choice(skin_crop_indices,
                                                            size=int(skin_crop_indices.shape[0]/downsampling_rate), replace=False))
    
    skin_mesh_tree, skin_mesh_vertices_neighbors_num, skin_mesh_vertices_neighbors_num_mean = get_neighbors(
        np.array(MRI['Mesh'].vertices), skin_downsampling_indices, neighbors_radius)
    
    shape_downsampling_indices = np.unique(np.random.choice(shape_crop_indices,
                                                            size=int(shape_crop_indices.shape[0]/downsampling_rate), replace=False))
    
    shape_mesh_tree, shape_mesh_vertices_neighbors_num, shape_mesh_vertices_neighbors_num_mean = get_neighbors(
        light_UHM['Mean_CCS'], shape_downsampling_indices, neighbors_radius)
    
    desired_mean_density = min(skin_mesh_vertices_neighbors_num_mean, shape_mesh_vertices_neighbors_num_mean)
    
    skin_mesh_tree_reduced, current_skin_mesh_vertices_neighbors_num, current_skin_remaining_indices = reduce_points_density(
        skin_mesh_tree, skin_mesh_vertices_neighbors_num, desired_mean_density, neighbors_radius, 0.01)
    if len(current_skin_remaining_indices)<skin_mesh_vertices_neighbors_num.shape[0]:
        skin_remaining_indices = []
        for current_vertex_index, current_vertex_position in enumerate(skin_mesh_tree_reduced.data):
            skin_remaining_indices.append(np.where(skin_mesh_tree.data==current_vertex_position)[0][0])
        skin_remaining_indices = np.array(skin_remaining_indices)
    else:
        skin_remaining_indices = np.arange(np.array(skin_mesh_vertices_neighbors_num).shape[0])

    shape_mesh_tree_reduced, current_shape_mesh_vertices_neighbors_num, current_shape_remaining_indices = reduce_points_density(
        shape_mesh_tree, shape_mesh_vertices_neighbors_num, desired_mean_density, neighbors_radius, 0.01)
    if len(current_shape_remaining_indices)<shape_mesh_vertices_neighbors_num.shape[0]:
        shape_remaining_indices = []
        for current_vertex_index, current_vertex_position in enumerate(shape_mesh_tree_reduced.data):
            shape_remaining_indices.append(np.where(shape_mesh_tree.data==current_vertex_position)[0][0])
        shape_remaining_indices = np.array(shape_remaining_indices)
    else:
        shape_remaining_indices = np.arange(np.array(shape_mesh_vertices_neighbors_num).shape[0])
    skin_remaining_indices = np.unique(skin_remaining_indices)
    skin_registration_indices = skin_downsampling_indices[skin_remaining_indices]
    skin_registration_indices = np.sort(np.unique(skin_registration_indices))

    shape_remaining_indices = np.unique(shape_remaining_indices)
    if args.region=='all' or args.region=='head':
        shape_registration_indices = np.concatenate((np.take(light_UHM['Landmarks']['indices'], light_UHM['Landmarks']['cranial_landmark_indices'])[shape_EEG_10_20_landmarks_remain_after_crop],
                                                    np.take(light_UHM['Landmarks']['indices'], light_UHM['Landmarks']['facial_landmark_indices'])[shape_facial_landmarks_remain_after_crop],
                                                    np.take(light_UHM['Landmarks']['indices'], shape_all_cranial_landmark_indices)[shape_cranial_landmarks_remain_after_crop],
                                                    shape_downsampling_indices[shape_remaining_indices]))
    else:
        shape_registration_indices = shape_downsampling_indices[shape_remaining_indices]
    shape_registration_indices = np.sort(np.unique(shape_registration_indices))

    # Point Cloud face cropshape_remaining_indices = np.unique(shape_remaining_indices)
    crop_depth_coefficient = 0.32
    
    shape_registration_indices_face_crop = depth_crop_by_coefficient(light_UHM['Mean_CCS'], crop_depth_coefficient, shape_registration_indices)
    skin_registration_indices_face_crop = depth_crop_by_coefficient(np.array(MRI['Mesh'].vertices), crop_depth_coefficient, skin_registration_indices)
    
    skin_registration_trimesh = o3d.geometry.TriangleMesh()
    skin_registration_trimesh.vertices = o3d.utility.Vector3dVector(MRI['Mesh'].vertices)
    skin_registration_trimesh.triangles = o3d.utility.Vector3iVector(MRI['Mesh'].faces)
    skin_registration_trimesh.vertex_normals = o3d.utility.Vector3dVector(MRI['Mesh'].vertex_normals)
    if args.region=='all' or args.region=='head':
        skin_registration_trimesh.remove_vertices_by_index(
            np.delete(np.arange(np.array(skin_registration_trimesh.vertices).shape[0]), skin_registration_indices))
    else:
        skin_registration_trimesh.remove_vertices_by_index(
            np.delete(np.arange(np.array(skin_registration_trimesh.vertices).shape[0]), skin_registration_indices_face_crop))

    skin_registration_pc = o3d.geometry.PointCloud()
    skin_registration_pc.points = skin_registration_trimesh.vertices
    skin_registration_pc.normals = skin_registration_trimesh.vertex_normals

    shape_registration_trimesh = o3d.geometry.TriangleMesh()
    shape_registration_trimesh.vertices = o3d.utility.Vector3dVector(light_UHM['Trimesh'].vertices)
    shape_registration_trimesh.triangles = o3d.utility.Vector3iVector(light_UHM['Trilist'])
    shape_registration_trimesh.vertex_normals = o3d.utility.Vector3dVector(light_UHM['Trimesh'].vertex_normals)
    
    if args.region=='all' or args.region=='head':
        shape_registration_trimesh.remove_vertices_by_index(
            np.delete(np.arange(np.array(shape_registration_trimesh.vertices).shape[0]), shape_registration_indices))
    else:
        shape_registration_trimesh.remove_vertices_by_index(
            np.delete(np.arange(np.array(shape_registration_trimesh.vertices).shape[0]), shape_registration_indices_face_crop))

    shape_registration_pc = o3d.geometry.PointCloud()
    shape_registration_pc.points = shape_registration_trimesh.vertices
    shape_registration_pc.normals = shape_registration_trimesh.vertex_normals

    shape_pc_EEG_10_20_landmark_indices = []
    if args.region=='all' or args.region=='head':
        for current_landmark_index in np.take(light_UHM['Landmarks']['indices'], light_UHM['Landmarks']['cranial_landmark_indices']):
            current_index = np.where(np.array(shape_registration_pc.points)==
                                    np.array(light_UHM['Trimesh'].vertices)[current_landmark_index, :])
            if np.size(current_index)>0:
                shape_pc_EEG_10_20_landmark_indices.append(current_index[0][0])

        shape_pc_EEG_10_20_landmark_indices = np.array(shape_pc_EEG_10_20_landmark_indices)

        shape_pc_facial_landmark_indices = []

        for current_landmark_index in np.take(light_UHM['Landmarks']['indices'], light_UHM['Landmarks']['facial_landmark_indices'])[shape_facial_landmarks_remain_after_crop]:
            current_index = np.where(np.array(shape_registration_pc.points)==
                                    np.array(light_UHM['Trimesh'].vertices)[current_landmark_index, :])
            if np.size(current_index)>0:
                shape_pc_facial_landmark_indices.append(current_index[0][0])
        
        shape_pc_facial_landmark_indices = np.array(shape_pc_facial_landmark_indices)

        shape_pc_cranial_landmark_indices = []

        for current_landmark_index in np.take(light_UHM['Landmarks']['indices'], shape_all_cranial_landmark_indices)[shape_cranial_landmarks_remain_after_crop]:
            current_index = np.where(np.array(shape_registration_pc.points)==
                                    np.array(light_UHM['Trimesh'].vertices)[current_landmark_index, :])
            if np.size(current_index)>0:
                shape_pc_cranial_landmark_indices.append(current_index[0][0])

        shape_pc_cranial_landmark_indices = np.array(shape_pc_cranial_landmark_indices)
    
    skin_tip_of_the_nose_index = np.argmax(np.array(skin_registration_pc.points)[:, 2])
    skin_tip_of_the_nose_position = np.array(skin_registration_pc.points)[skin_tip_of_the_nose_index, :]

    shape_tip_of_the_nose_index = np.argmax(np.array(shape_registration_pc.points)[:, 2])
    shape_tip_of_the_nose_position = np.array(shape_registration_pc.points)[shape_tip_of_the_nose_index, :]

    skin_translation = shape_tip_of_the_nose_position-skin_tip_of_the_nose_position
    skin_registration_pc.points = o3d.utility.Vector3dVector(np.array(skin_registration_pc.points)+skin_translation)

    skin_z_span = max(np.array(skin_registration_trimesh.vertices)[:, 2])-min(np.array(skin_registration_trimesh.vertices)[:, 2])
    shape_z_span = max(np.array(shape_registration_trimesh.vertices)[:, 2])-min(np.array(shape_registration_trimesh.vertices)[:, 2])

    skin_registration_pc.points = o3d.utility.Vector3dVector(np.array(skin_registration_pc.points)*shape_z_span/skin_z_span)
    
    manual_translation = skin_translation
    manual_translation_matrix = np.array([[1, 0, 0, manual_translation[0]],
                                                [0, 1, 0, manual_translation[1]],
                                                [0, 0, 1, manual_translation[2]],
                                                [0, 0, 0, 1]])

    manual_scale = shape_z_span/skin_z_span
    manual_scale_matrix = np.array([[manual_scale, 0, 0, 0],
                                                [0, manual_scale, 0, 0],
                                                [0, 0, manual_scale, 0],
                                                [0, 0, 0, 1]])
    if args.region=='all' or args.region=='head':
        eigen_vec_registration = light_UHM['Eigenvectors_CCS'][shape_registration_indices, :, :].reshape(
            -1, light_UHM['Eigenvectors_CCS'].shape[-1])
        mean_shape_registration_flattened = (light_UHM['Mean'].reshape(-1,3))[shape_registration_indices].flatten().reshape(-1,1)
    else:
        eigen_vec_registration = light_UHM['Eigenvectors_CCS'][shape_registration_indices_face_crop, :, :].reshape(
            -1, light_UHM['Eigenvectors_CCS'].shape[-1])
        mean_shape_registration_flattened = (light_UHM['Mean'].reshape(-1,3))[shape_registration_indices_face_crop].flatten().reshape(-1,1)
    
    np.random.seed(0)
    weights = np.random.rand(light_UHM['Eigenvectors_CCS'].shape[-1])

    if enable_early_stop:
        previous_loss = 1e6
    
    opt_start_time = datetime.now()

    icp_results = []

    for current_opt_step in range(args.iterations):
        current_opt_step_start_time = datetime.now()

        skin_registration_pc, icp_result, _ = ICP_and_transform(skin_registration_pc, shape_registration_pc, correspondent_max_distance_threshold)
        icp_results.append(icp_result)

        skin_registration_points = np.array(skin_registration_pc.points)

        skin_correspondence_indices_icp = np.array(icp_result.correspondence_set)[:, 0]
        shape_correspondence_indices_icp = np.array(icp_result.correspondence_set)[:, 1]
        
        if use_injective_correspondence:
            injective_correspondence_set = create_injective_correspondence(skin_registration_pc, skin_correspondence_indices_icp,
                                                                            shape_registration_pc, shape_correspondence_indices_icp)
            skin_correspondence_indices = injective_correspondence_set[:, 0]
            shape_correspondence_indices = injective_correspondence_set[:, 1] 
        else:
            skin_correspondence_indices = skin_correspondence_indices_icp
            shape_correspondence_indices = shape_correspondence_indices_icp

        results = minimize(head_mesh_align,
                            x0=weights.flatten(),
                            args=(skin_registration_points,
                                    skin_correspondence_indices,
                                    shape_correspondence_indices,
                                    lambdas,
                                    mean_shape_registration_flattened,
                                    eigen_vec_registration,
                                    light_UHM['EigenValues'],
                                    ),
                            tol=1e-20,
                            options={'disp': False, 'maxiter': 10},
                        )
        
        weights = results['x'].reshape(-1, 1)

        shape_registration_points = get_adjusted_shape_CCS(weights, mean_shape_registration_flattened,
                                                            eigen_vec_registration, light_UHM['EigenValues'])

        shape_registration_pc.points = o3d.utility.Vector3dVector(shape_registration_points)

        current_opt_step_end_time = datetime.now()
        print(f"Optimization step {current_opt_step} took: {current_opt_step_end_time-current_opt_step_start_time}") 

        if 1:
            result_skin_points = skin_registration_points[skin_correspondence_indices_icp, :]
            result_shape_points = shape_registration_points[shape_correspondence_indices_icp, :]
            result_distances = result_shape_points - result_skin_points
            result_distances_norm = np.linalg.norm(result_distances, axis=1)
            result_distance_loss = 1e3*np.mean(result_distances_norm)
            result_weights_normalization = lambdas[0]*np.linalg.norm(weights)
            result_total_loss = result_distance_loss
            print(f"Loss: {result_total_loss}")

            if use_injective_correspondence:
                injective_correspondence_skin_points = skin_registration_points[injective_correspondence_set[:, 0], :]
                injective_correspondence_shape_points = shape_registration_points[injective_correspondence_set[:, 1], :]

                injective_correspondence_distances = injective_correspondence_shape_points - injective_correspondence_skin_points
                injective_correspondence_distances_norm = np.linalg.norm(injective_correspondence_distances, axis=1)
                injective_correspondence_distance_loss = 1e3*np.mean(injective_correspondence_distances_norm)
                injective_correspondence_weights_normalization = lambdas[0]*np.linalg.norm(weights)
                injective_correspondence_mask_ones_percentage = max(len(skin_correspondence_indices)/skin_registration_points.shape[0],
                                                len(shape_correspondence_indices)/shape_registration_points.shape[0])
                injective_correspondence_distance_loss_mask = injective_correspondence_distance_loss/(injective_correspondence_mask_ones_percentage**mask_exp_coeff)
                injective_correspondence_total_loss = injective_correspondence_distance_loss# + injective_correspondence_weights_normalization
                print(f"Injective correspondence loss: {injective_correspondence_total_loss}")

        if enable_early_stop:
            if result_total_loss>1.01*previous_loss:
                print(f"Early stopping at step {current_opt_step}")
                break
            else:
                previous_loss = result_total_loss

    opt_end_time = datetime.now()
    print(f"{current_opt_step+1} optimization steps took: {opt_end_time-opt_start_time}")

    
    transformation_array = np.zeros((4*5, 4))

    transformation_array[:4, :] = manual_translation_matrix
    transformation_array[4:8, :] = manual_scale_matrix
    for i in range(len(icp_results)):
        transformation_array[(i+2)*4:(i+3)*4, :] = icp_results[i].transformation

    if args.region=='all' or args.region=='head':
        shape_points_head = np.array(shape_registration_pc.points)
        skin_points_head = np.array(skin_registration_pc.points)
    
        chamfer_distance_over_head, _, _ = get_chamfer_distance(shape_points_head, skin_points_head)
        chamfer_distance_over_head/=registration_scale_factor
    
        eigen_vec_registration_face = light_UHM['Eigenvectors_CCS'][shape_registration_indices_face_crop, :, :].reshape(-1, light_UHM['Eigenvectors_CCS'].shape[-1])
        mean_shape_registration_flattened_face = (light_UHM['Mean'].reshape(-1,3))[shape_registration_indices_face_crop].flatten().reshape(-1,1)

        shape_points_face = get_adjusted_shape_CCS(weights, mean_shape_registration_flattened_face, eigen_vec_registration_face, light_UHM['EigenValues'])
        
        face_skin_registration_trimesh = o3d.geometry.TriangleMesh()
        face_skin_registration_trimesh.vertices = o3d.utility.Vector3dVector(MRI['Mesh'].vertices)
        face_skin_registration_trimesh.triangles = o3d.utility.Vector3iVector(MRI['Mesh'].faces)
        face_skin_registration_trimesh.vertex_normals = o3d.utility.Vector3dVector(np.array(MRI['Mesh'].vertex_normals))

        face_skin_registration_trimesh.remove_vertices_by_index(
            np.delete(np.arange(np.array(face_skin_registration_trimesh.vertices).shape[0]), skin_registration_indices_face_crop))
        
        face_skin_registration_pc = o3d.geometry.PointCloud()
        face_skin_registration_pc.points = face_skin_registration_trimesh.vertices
        face_skin_registration_points = np.array(face_skin_registration_pc.points)

        number_of_transformations=2+len(icp_results)
        coordinates = []
        coordinates.append(face_skin_registration_points)
        for i in range(number_of_transformations):
            current_transformation_matrix = transformation_array[4*i:4*(i+1), :]
            current_extended_coordinates = np.concatenate((coordinates[i], np.ones((coordinates[i].shape[0], 1))), axis=1).T
            current_product = (current_transformation_matrix@current_extended_coordinates).T
            coordinates.append(current_product[:, :3])

        skin_points_face = coordinates[-1]

        chamfer_distance_over_face, _, _ = get_chamfer_distance(shape_points_face, skin_points_face)
        chamfer_distance_over_face/=registration_scale_factor
    else:
        shape_points_face = np.array(shape_registration_pc.points)
        skin_points_face = np.array(skin_registration_pc.points)
    
        chamfer_distance_over_face, _, _ = get_chamfer_distance(shape_points_face, skin_points_face)
        chamfer_distance_over_face/=registration_scale_factor

        eigen_vec_registration_head = light_UHM['Eigenvectors_CCS'][shape_registration_indices, :, :].reshape(-1, light_UHM['Eigenvectors_CCS'].shape[-1])
        mean_shape_registration_flattened_head = (light_UHM['Mean'].reshape(-1,3))[shape_registration_indices].flatten().reshape(-1,1)

        head_shape_points = get_adjusted_shape_CCS(weights, mean_shape_registration_flattened_head, eigen_vec_registration_head, light_UHM['EigenValues'])

        head_skin_registration_trimesh = o3d.geometry.TriangleMesh()
        head_skin_registration_trimesh.vertices = o3d.utility.Vector3dVector(MRI['Mesh'].vertices)
        head_skin_registration_trimesh.triangles = o3d.utility.Vector3iVector(MRI['Mesh'].faces)
        head_skin_registration_trimesh.vertex_normals = o3d.utility.Vector3dVector(np.array(MRI['Mesh'].vertex_normals))

        head_skin_registration_trimesh.remove_vertices_by_index(
            np.delete(np.arange(np.array(head_skin_registration_trimesh.vertices).shape[0]), skin_registration_indices))
        
        head_skin_registration_pc = o3d.geometry.PointCloud()
        head_skin_registration_pc.points = head_skin_registration_trimesh.vertices
        head_skin_registration_points = np.array(head_skin_registration_pc.points)

        number_of_transformations=2+len(icp_results)
        coordinates = []
        coordinates.append(head_skin_registration_points)
        for i in range(number_of_transformations):
            current_transformation_matrix = transformation_array[4*i:4*(i+1), :]
            current_extended_coordinates = np.concatenate((coordinates[i], np.ones((coordinates[i].shape[0], 1))), axis=1).T
            current_product = (current_transformation_matrix@current_extended_coordinates).T
            coordinates.append(current_product[:, :3])

        head_skin_points = coordinates[-1]

        chamfer_distance_over_head, _, _ = get_chamfer_distance(head_shape_points,  head_skin_points)
        chamfer_distance_over_head/=registration_scale_factor


    chamfer_distances_df = pd.DataFrame(
        data=np.array([chamfer_distance_over_face.detach().numpy(), chamfer_distance_over_head.detach().numpy()]),
        index=['chamfer_distance_over_head', 'chamfer_distance_over_face']
    )

    # Create a Pandas Excel writer using XlsxWriter as the engine
    if args.region!='all' and args.region!='head':
        chamfer_distances_folder = dataset_folder+region_folder+selected_subject
        writer = pd.ExcelWriter(chamfer_distances_folder+'.xlsx')
        # Write each dataframe to a different worksheet
        chamfer_distances_df.to_excel(writer, sheet_name='Chamfer distances')
        # Close the Pandas Excel writer and output the Excel file
        writer.save()
        # get Chamfer distances and skip the rest of the code
        continue

    shape_facial_landmarks_crop = light_UHM['Landmarks']['facial_landmark_indices'][np.where(
        light_UHM['Mean_CCS'][np.take(light_UHM['Landmarks']['indices'], light_UHM['Landmarks']['facial_landmark_indices']), 1]>shape_y_value_crop)[0]]-20

    assymetric_indices = []

    for current_facial_index in range(17):
        current_facial_index += 1
        complementery_facial_index = 18-current_facial_index
        if current_facial_index in shape_facial_landmarks_crop and complementery_facial_index not in shape_facial_landmarks_crop:
            i = np.where(shape_facial_landmarks_crop==current_facial_index)[0]
            assymetric_indices.append(i)

    shape_facial_landmark_indices_crop = np.delete(np.arange(shape_facial_landmarks_crop.shape[0]), assymetric_indices)
    shape_facial_landmark_indices_final = shape_pc_facial_landmark_indices[shape_facial_landmark_indices_crop]


    skin_facial_landmark_indices, skin_missing_facial_landmark_indices = annotate_using_another_point_cloud(
        skin_registration_pc, shape_registration_pc, shape_facial_landmark_indices_final, correspondent_max_distance_threshold, use_perpendicular_plane_distance=True)     

    skin_corresponding_EEG_10_20_landmark_indices, skin_missing_EEG_10_20_landmark_indices = annotate_using_another_point_cloud(
        skin_registration_pc, shape_registration_pc, shape_pc_EEG_10_20_landmark_indices, correspondent_max_distance_threshold, use_perpendicular_plane_distance=True)  

    skin_cranial_landmark_indices, skin_missing_cranial_landmark_indices = annotate_using_another_point_cloud(
        skin_registration_pc, shape_registration_pc, shape_pc_cranial_landmark_indices, correspondent_max_distance_threshold, use_perpendicular_plane_distance=True)

    coordinates_columns_names = ['indices', 'x', 'y', 'z']


    skin_coordinates_indices = np.concatenate((skin_corresponding_EEG_10_20_landmark_indices, skin_facial_landmark_indices, skin_cranial_landmark_indices[-2:])).reshape(-1, 1)
    skin_coordinates_positions = np.array(skin_registration_pc.points)[skin_coordinates_indices.squeeze(), :]

    skin_coordinates_data = np.concatenate((skin_coordinates_indices, skin_coordinates_positions), axis=1)

    skin_coordinates_indices_names = [
        [np.array(light_UHM['Landmarks']['names'])[light_UHM['Landmarks']['cranial_landmark_indices']][i] for i in light_UHM['Landmarks']['cranial_landmark_indices'] if i not in skin_missing_EEG_10_20_landmark_indices],
        list(
            np.delete(np.take(np.array(light_UHM['Landmarks']['names'])[np.concatenate((light_UHM['Landmarks']['facial_landmark_indices'], light_UHM['Landmarks']['additional_landmark_indices']))],
                                np.delete(shape_facial_landmarks_crop, assymetric_indices)-1),
                        skin_missing_facial_landmark_indices)
        ),
        np.take(light_UHM['Landmarks']['names'], cranial_annotation_indices[-2:])
    ]

    skin_coordinates_indices_names = [current_name for current_names_group in skin_coordinates_indices_names for current_name in current_names_group]

    skin_coordinates_df = pd.DataFrame(data=skin_coordinates_data,
                                        index=skin_coordinates_indices_names,
                                        columns=coordinates_columns_names)        
    
    skin_original_coordinates_positions = np.array(skin_registration_trimesh.vertices)[skin_coordinates_indices.squeeze(), :]

    skin_original_coordinates_data = np.concatenate((skin_coordinates_indices, skin_original_coordinates_positions), axis=1)

    skin_original_coordinates_df = pd.DataFrame(data=skin_original_coordinates_data,
                                                index=skin_coordinates_indices_names,
                                                columns=coordinates_columns_names)
                                                

    shape_coordinates_indices = np.concatenate((shape_pc_EEG_10_20_landmark_indices, shape_pc_facial_landmark_indices)).reshape(-1, 1)
    shape_coordinates_positions = np.array(shape_registration_pc.points)[shape_coordinates_indices.squeeze(), :]

    shape_coordinates_data = np.concatenate((shape_coordinates_indices, shape_coordinates_positions), axis=1)

    shape_coordinates_indices_names = [
        np.array(light_UHM['Landmarks']['names'])[light_UHM['Landmarks']['cranial_landmark_indices']],
        list(
            np.take(np.array(light_UHM['Landmarks']['names'])[np.concatenate((light_UHM['Landmarks']['facial_landmark_indices'], light_UHM['Landmarks']['additional_landmark_indices']))],
                    np.delete(shape_facial_landmarks_crop, assymetric_indices)-1)
        )
    ]

    shape_coordinates_indices_names = [current_name for current_names_group in shape_coordinates_indices_names for current_name in current_names_group]

    shape_coordinates_df = pd.DataFrame(data=shape_coordinates_data,
                                        index=shape_coordinates_indices_names,
                                        columns=coordinates_columns_names)

    ## Creating normals dataframes
    
    normals_columns_names = ['x', 'y', 'z']
    
    skin_coordinates_normals = np.array(skin_registration_pc.normals)[skin_coordinates_indices.squeeze(), :]

    skin_normals_df = pd.DataFrame(data=skin_coordinates_normals,
                                        index=skin_coordinates_indices_names,
                                        columns=normals_columns_names)
    
    skin_original_coordinates_normals = np.array(skin_registration_trimesh.vertex_normals)[skin_coordinates_indices.squeeze(), :]

    skin_original_normals_df = pd.DataFrame(data=skin_original_coordinates_normals,
                                        index=skin_coordinates_indices_names,
                                        columns=normals_columns_names)

    # ## Creating stats dataframes

    inverse_transformation_array = np.zeros((4*5, 4))

    inverse_transformation_array[:4, :] = get_affine_transformation_matrix_inverse(manual_translation_matrix)
    inverse_transformation_array[4:8, :] = get_affine_transformation_matrix_inverse(manual_scale_matrix)
    for i in range(len(icp_results)):
        inverse_transformation_array[(i+2)*4:(i+3)*4, :] = get_affine_transformation_matrix_inverse(transformation_array[(i+2)*4:(i+3)*4, :])

    inverse_transformation_df = pd.DataFrame(data=inverse_transformation_array)

    #http://planning.cs.uiuc.edu/node102.html#eqn:yprmat , http://planning.cs.uiuc.edu/node103.html
    if len(icp_results)==1:
        final_combined_icp_transformation = icp_results[0].transformation
    else:
        for i in range(len(icp_results)-1):
            if i==0:
                current_combined_icp_transformation = icp_results[i].transformation
            current_combined_icp_transformation = icp_results[i+1].transformation@current_combined_icp_transformation
        final_combined_icp_transformation = current_combined_icp_transformation

        
    transformation_scale = 1
    for i in range(int(inverse_transformation_array.shape[0]/4)):
        current_inverse_transformation = inverse_transformation_array[4*i:4*(i+1), :]
        current_x_scale = np.linalg.norm(current_inverse_transformation[:3, 0])
        current_y_scale = np.linalg.norm(current_inverse_transformation[:3, 1])
        current_z_scale = np.linalg.norm(current_inverse_transformation[:3, 2])
        current_scale = np.mean(np.array([current_x_scale, current_y_scale, current_z_scale]))
        transformation_scale = transformation_scale*current_scale

    skin_yaw_angle = np.rad2deg(np.arctan(final_combined_icp_transformation[1,0]/final_combined_icp_transformation[0,0]))
    skin_pitch_angle = np.rad2deg(np.arctan(-final_combined_icp_transformation[2,0]/np.sqrt(final_combined_icp_transformation[2,1]**2+final_combined_icp_transformation[2,2]**2)))
    skin_roll_angle = np.rad2deg(np.arctan(final_combined_icp_transformation[2,1]/final_combined_icp_transformation[2,2]))


    skin_final_tip_of_the_nose_position = np.array(skin_registration_pc.points)[skin_tip_of_the_nose_index, :]
    shape_final_tip_of_the_nose_position =  np.array(shape_registration_pc.points)[shape_tip_of_the_nose_index, :]
    final_tip_of_the_nose_distance = np.linalg.norm(skin_final_tip_of_the_nose_position-shape_final_tip_of_the_nose_position)


    shape_pc_EEG_10_20_landmark_positions = np.array(shape_registration_pc.points)[shape_pc_EEG_10_20_landmark_indices, :]

    
    alignment_stats_data = [
        result_total_loss,
        final_tip_of_the_nose_distance,
        transformation_scale,
        int(np.array(skin_registration_pc.points).shape[0]),
        int(np.array(shape_registration_pc.points).shape[0]),
        int(skin_correspondence_indices.shape[0]),
        int(shape_correspondence_indices.shape[0]),
        int(downsampling_rate),
        skin_yaw_angle,
        skin_pitch_angle,
        skin_roll_angle,
        correspondent_max_distance_threshold,
        int(correspondent_max_distance_ratio_threshold),
        neighbors_radius,
        int(mask_exp_coeff)
    ]

    alignment_stats_indices_names = [
        'final_loss',
        'final_tip_of_the_nose_distance',
        'transformation_scale',
        'skin_number_of_points',
        'shape_number_of_points',
        'full_skin_correspondence_set_size',
        'full_shape_correspondence_set_size',
        'downsampling_rate',
        'skin_yaw_angle',
        'skin_pitch_angle',
        'skin_roll_angle',
        'correspondent_max_distance_threshold',
        'correspondent_max_distance_ratio_threshold',
        'neighbors_radius',
        'mask_exp_coeff'
    ]

    if use_injective_correspondence:
        alignment_stats_data.append(injective_correspondence_total_loss)
        alignment_stats_indices_names.append('injective_correspondence_final_loss')

    alignment_stats_df = pd.DataFrame(data=alignment_stats_data,
                                        index=alignment_stats_indices_names,
                                        columns=['value'])        
    
    
    # ## Get geodesic distances

    decimation_percentages = np.array([1])
    fixed_landmarks = True

    cranial_landmark_names_no_missing = [(np.take(light_UHM['Landmarks']['indices'], cranial_annotation_indices))[i] for i in range(len(cranial_annotation_indices)) if i not in skin_missing_cranial_landmark_indices]

    EEG_10_20_landmark_names_no_missing = [(np.array(light_UHM['Landmarks']['names'])[light_UHM['Landmarks']['cranial_landmark_indices']])[i] for i in light_UHM['Landmarks']['cranial_landmark_indices'] if i not in skin_missing_EEG_10_20_landmark_indices]

    distances_to_find_names = ['A1_A2', 'nasion_inion',
                                'Fp1_F7', 'F7_T3', 'T3_T5', 'T5_O1', 'O1_O2', 'O2_T6', 'T6_T4', 'T4_F8', 'F8_Fp2', 'Fp2_Fp1']



    skin_cranial_geodesic_distances = np.zeros((len(distances_to_find_names),))

    skin_original_density_trimesh = trimesh.Trimesh(vertices=np.array(skin_registration_pc.points),
                                                    faces=np.array(skin_registration_trimesh.triangles),
                                                    process=False)

    original_density_skin_trimesh = skin_original_density_trimesh

    current_skin_trimesh = skin_original_density_trimesh

    current_skin_graph = trimesh.graph.vertex_adjacency_graph(current_skin_trimesh)
    current_skin_graph_weights, _ = get_graph_weights(current_skin_graph, current_skin_trimesh)

    for i, current_distance_landmarks in enumerate(distances_to_find_names):
        try:
            current_distance_landmarks = current_distance_landmarks.split('_')
            current_landmark_name_i = current_distance_landmarks[0]
            current_landmark_name_j = current_distance_landmarks[1]

            if i<2:
                current_landmark_index_i = skin_cranial_landmark_indices[cranial_landmark_names_no_missing.index(current_landmark_name_i)]
                current_landmark_index_j = skin_cranial_landmark_indices[cranial_landmark_names_no_missing.index(current_landmark_name_j)]
            else:
                current_landmark_index_i = skin_corresponding_EEG_10_20_landmark_indices[EEG_10_20_landmark_names_no_missing.index(current_landmark_name_i)]
                current_landmark_index_j = skin_corresponding_EEG_10_20_landmark_indices[EEG_10_20_landmark_names_no_missing.index(current_landmark_name_j)]

            current_geodesic_distance = nx.dijkstra_path_length(current_skin_graph_weights,
                                                                current_landmark_index_i, 
                                                                current_landmark_index_j, 
                                                                weight='weight')
        except:
            print(f"cranial landmarks {current_distance_landmarks} geodesic distance failed")
            current_geodesic_distance = 0

        skin_cranial_geodesic_distances[i] = current_geodesic_distance    
        
    skin_geodesic_distances_df = pd.DataFrame(data=skin_cranial_geodesic_distances,
                                                index=distances_to_find_names
                                                )                                     

    if args.save==True:
        # Saving dataframes

        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(dataset_folder+region_folder+selected_subject+'.xlsx')

        # Write each dataframe to a different worksheet.
        skin_coordinates_df.to_excel(writer, sheet_name='Skin coordinates')
        skin_original_coordinates_df.to_excel(writer, sheet_name='Skin original coordinates')
        #skin_simnibs_coordinates_df.to_excel(writer, sheet_name='Skin SimNIBS coordinates')
        skin_normals_df.to_excel(writer, sheet_name='Skin normals')
        skin_original_normals_df.to_excel(writer, sheet_name='Skin original normals')
        shape_coordinates_df.to_excel(writer, sheet_name='Shape coordinates')
        #shape_corresponding_coordinates_df.to_excel(writer, sheet_name='Shape corresponding coordinates')
        skin_geodesic_distances_df.to_excel(writer, sheet_name='Skin distances')
        #skin_original_geodesic_distances_df.to_excel(writer, sheet_name='Skin original distances')
        inverse_transformation_df.to_excel(writer, sheet_name='Inverse transformations')
        alignment_stats_df.to_excel(writer, sheet_name='Stats')
        chamfer_distances_df.to_excel(writer, sheet_name='Chamfer distances')

        # Close the Pandas Excel writer and output the Excel file.
        writer.close()

    current_subject_end_time = datetime.now()

    print(f"Finished subject: {selected_subject}, process time was: {current_subject_end_time-current_subject_start_time}, loss is: {result_total_loss}")
    if args.subject!=-1:
        break