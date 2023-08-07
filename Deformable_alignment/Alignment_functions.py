# import packages
import pickle
import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
import scipy.spatial as spatial
import torch
import networkx as nx

def get_affine_transformation_matrix_inverse(affine_transformation_matrix):
    # getting the affine transformation matrix inverse so we would be able to transform the points from the 3DMM coordinate system to the original one
    P = affine_transformation_matrix[:3, :3]
    v = affine_transformation_matrix[:3, 3]
    inverse_matrix = np.zeros((affine_transformation_matrix.shape))
    inverse_matrix[:3, :3] = np.linalg.inv(P)
    inverse_matrix[:3, 3] = -np.linalg.inv(P)@v
    inverse_matrix[3, :3] = np.zeros((1, 3))
    inverse_matrix[3, 3] = 1
    return inverse_matrix

def get_adjusted_shape_CCS(weights, mean_shape_flattened, eigen_vec, eigen_val):
    # getting a new shape according to an added deformation defined by desired eigen-decomposition weights
    eigen_val_vec = eigen_val.reshape(weights.shape)
    if torch.is_tensor(weights) and torch.is_tensor(eigen_val_vec):
        eigen_vec_multipliers = torch.multiply(weights, eigen_val_vec)
    else:
        eigen_vec_multipliers = np.multiply(weights, eigen_val_vec)
    added_deformation = (eigen_vec @ eigen_vec_multipliers).reshape(-1, 1)

    new_shape = mean_shape_flattened + added_deformation #+ added_deformation_2
    #print(added_deformation.shape)

    new_shape_CCS = new_shape.reshape(-1,3)

    return new_shape_CCS

def ICP_and_transform(source_pc, target_pc,
                      correspondent_max_distance_threshold,
                      max_iteration=int(1e2), relative_rmse=1e-7, relative_fitness=1e-7,
                      roll_rotation_angle_threshold=None, pitch_rotation_angle_threshold=None, yaw_rotation_angle_threshold=None, number_of_iterations=10):
    # run generalized ICP algorithm with desired parameters and update the source point cloud according to the algorithm transformation
    if roll_rotation_angle_threshold is not None or pitch_rotation_angle_threshold is not None or yaw_rotation_angle_threshold is not None:
        translation = np.zeros(3)
        
        for i in range(number_of_iterations):
            existing_threshold_number = 0
            under_threshold_number = 0
            if i>0:
                translation = np.random.normal(size=3, scale=0.005)
            source_pc.translate(translation)

            icp_result = o3d.pipelines.registration.registration_generalized_icp(source=source_pc,
                                                                            target=target_pc,
                                                                            max_correspondence_distance=correspondent_max_distance_threshold,
                                                                            estimation_method=o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
                                                                            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                                                                                max_iteration=max_iteration,
                                                                                relative_rmse=relative_rmse,
                                                                                relative_fitness=relative_fitness
                                                                            ))

            if i==0:
                transformation_matrix = icp_result.transformation

            _, _, rotation_angles = decompose_transformation_matrix(icp_result.transformation.copy())

            if np.all(rotation_angles==0):
                source_pc.translate(-translation)
                continue

            if roll_rotation_angle_threshold is not None:
                existing_threshold_number += 1
                #roll_rotation_angle = np.mod(rotation_angles[2], 1*np.pi)
                roll_rotation_angle = rotation_angles[1]*180/np.pi
                if np.abs(roll_rotation_angle) < roll_rotation_angle_threshold:
                    #print(f'Iteration {i} Roll rotation angle is below threshold: {roll_rotation_angle} degrees')
                    under_threshold_number += 1
                #else:
                    #print(f'Iteration {i} Roll rotation angle is too large: {roll_rotation_angle} degrees')

            if yaw_rotation_angle_threshold is not None:
                existing_threshold_number += 1
                #yaw_rotation_angle = np.mod(rotation_angles[1], 2*np.pi)
                yaw_rotation_angle = rotation_angles[2]*180/np.pi
                if np.abs(yaw_rotation_angle) < yaw_rotation_angle_threshold:
                    under_threshold_number += 1
                #else:
                #    print(f'Iteration {i} Yaw rotation angle is too large: {yaw_rotation_angle} degrees')

            if pitch_rotation_angle_threshold is not None:
                existing_threshold_number += 1
                #pitch_rotation_angle = np.mod(rotation_angles[0], 2*np.pi)
                pitch_rotation_angle = rotation_angles[0]*180/np.pi
                if np.abs(pitch_rotation_angle) < pitch_rotation_angle_threshold:
                    under_threshold_number += 1
                #else:
                #    print(f'Iteration {i} Pitch rotation angle is too large: {pitch_rotation_angle} degrees')
            
            if under_threshold_number==existing_threshold_number or i==number_of_iterations-1:
                #icp_result = icp_result.copy()
                transformation_matrix = icp_result.transformation.copy()
                transformation_matrix[:-1, -1] = transformation_matrix[:-1, -1]+translation
                break
            else:
                source_pc.translate(-translation)

    else:
        icp_result = o3d.pipelines.registration.registration_generalized_icp(source=source_pc,
                                                                        target=target_pc,
                                                                        max_correspondence_distance=correspondent_max_distance_threshold,
                                                                        estimation_method=o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
                                                                        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                                                                            max_iteration=max_iteration,
                                                                            relative_rmse=relative_rmse,
                                                                            relative_fitness=relative_fitness
                                                                        ))
        
        transformation_matrix = icp_result.transformation.copy()

    source_pc.transform(transformation_matrix)

    return source_pc, icp_result, transformation_matrix

def head_mesh_align(weights, skin_mesh_icp, skin_points_indices, shape_points_indices, lambdas,
                    mean_shape_flattened, eigen_vec, eigen_val):
    # getting the new deformed shape distance from the source point cloud
    new_shape_CCS = get_adjusted_shape_CCS(weights, mean_shape_flattened, eigen_vec, eigen_val)

    current_skin_points = skin_mesh_icp[skin_points_indices, :]

    aligned_shape_points = new_shape_CCS[shape_points_indices, :]

    distances = aligned_shape_points - current_skin_points
    distances_norm = np.linalg.norm(distances, axis=1)

    distance_loss = np.mean(distances_norm)

    total_loss = distance_loss
    return total_loss

def perpendicular_plane_distance(line_first_point, line_second_point, third_point):
    # get the closest distance between a line (defined by line_first_point and line_second_point) and a point in space (third_point)
    line_vector = line_first_point-line_second_point
    distance = np.linalg.norm(np.outer(np.dot(third_point-line_second_point, line_vector)/np.dot(line_vector, line_vector),
                                       line_vector)+line_second_point-third_point, axis=1)[0]
    return distance

def get_chamfer_distance(x, y, metric='l2', disance_threshold=None): # https://gist.github.com/sergeyprokudin/c4bf4059230da8db8256e36524993367
    """
    Chamfer distance between two point clouds
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    disance_threshold: float, default None
        if not None, only points with distance smaller than disance_threshold will be considered
    
    Returns
    -------
    chamfer_dist: float
        computed Chamfer distance
    mean_x_to_y: float
        mean distance from x to y
    mean_y_to_x: float
        mean distance from y to x
    """

    x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
    min_y_to_x = torch.tensor(x_nn.kneighbors(y)[0], requires_grad=True)

    y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
    min_x_to_y = torch.tensor(y_nn.kneighbors(x)[0], requires_grad=True)

    if disance_threshold is not None:
        min_y_to_x=min_y_to_x[min_y_to_x<disance_threshold]
        min_x_to_y=min_x_to_y[min_x_to_y<disance_threshold]

    chamfer_dist = torch.mean(min_y_to_x) + torch.mean(min_x_to_y)
        
    return chamfer_dist, min_x_to_y, min_y_to_x

def load_annotations(models_folder, module_to_load):
    # load the annotations from the annotations file

    # load modules (landmarks and masks)
    modules_folder = models_folder + '/Landmarks and masks/'

    landmark_indices = []
    landmark_names = []
    #landmark_groups = []

    for current_module_name in module_to_load:
        module_file = open(modules_folder + current_module_name + '.pkl', 'rb')
        current_module = pickle.load(module_file)
        module_file.close()
        if current_module_name=='EEG_10_20':
            current_module_names = list(current_module.keys())
            current_module = np.asarray(list(current_module.values()))
        else:
            current_module_names = list(map(str, 1+np.arange(len(current_module))))

        landmark_indices.append(current_module)
        landmark_names.append(current_module_names)
        #landmark_groups.append(np.arange(len(current_module)))

    # turn list of lists into one list
    landmark_indices = [item for items in landmark_indices for item in items]
    landmark_names = [item for items in landmark_names for item in items]
    num_of_landmarks = len(landmark_indices)
    
    landmarks = {}
    landmarks['indices'] = landmark_indices
    landmarks['names'] = landmark_names
    landmarks['landmarks_num'] = num_of_landmarks
    
    return landmarks

def load_3DMM(models_folder, modules_folder, is_light):
    # load the 3DMM model and its fields from the 3DMM file
    modules_folder = models_folder + '/Landmarks and masks/'
    
    if is_light==False:
        model_name = 'head_model_global_align'
        module_to_load = ['68_land_idxs'] # has been annotated with Multi-Pie 68 facial landmarks
    else:
        model_name = 'head_model_global_align_no_mouth_and_eyes'
        module_to_load = ['EEG_10_20'] # we annotated it with these landmarks
    
    model_file = open(models_folder + model_name + '.pkl', 'rb')
    model_dict = pickle.load(model_file)
    model_file.close()
    
    # turning coordinates system to be in millimeter units
    scale_factor = 100 # https://en.wikipedia.org/wiki/Telecanthus#cite_note-ent-2 - euclidean distance in original model is 0.3: np.linalg.norm(mean_shape_CCS[1214]-mean_shape_CCS[13746])
    
    model_dict['Mean'] = scale_factor*model_dict['Mean']
    model_dict['Mean_CCS'] = model_dict['Mean'].reshape(-1,3)
    model_dict['Eigenvectors_CCS'] =  model_dict['Eigenvectors'].reshape(-1, 3, model_dict['Eigenvectors'].shape[1])
    model_dict['Eigenvectors_num'] =  model_dict['Eigenvectors'].shape[1]
    
    landmarks = load_annotations(models_folder, module_to_load)
    
    model_dict['Landmarks'] = landmarks
    
    return model_dict

def vertical_crop_by_coefficient(vertices_coordinates, crop_vertical_coefficient):
    # crop the point cloud vertically according to the crop_vertical_coefficient relatively to the distance between the tip of the nose and the "highest" vertical point
    tip_of_nose_index = np.argmax(vertices_coordinates[:, 2])
    tip_of_nose_y_value = vertices_coordinates[tip_of_nose_index, 1]
    highest_y_value = np.max(vertices_coordinates[:, 1])

    y_value_crop = tip_of_nose_y_value-crop_vertical_coefficient*(highest_y_value-tip_of_nose_y_value)
    crop_remaining_indices = np.where(vertices_coordinates[:, 1]>y_value_crop)[0]
    
    return crop_remaining_indices, y_value_crop, tip_of_nose_y_value

def depth_crop_by_coefficient(vertices_coordinates, crop_depth_coefficient, registration_indices):
    # crop the point cloud in the depth direction according to the crop_depth_coefficient relatively to the depth
    z_value_crop = np.max(vertices_coordinates[:, 2])-crop_depth_coefficient*(np.max(vertices_coordinates[:, 2])-np.min(vertices_coordinates[:, 2]))
    crop_indices_face = np.where((vertices_coordinates[:, 2]>z_value_crop))[0]
    registration_indices_face_crop = np.intersect1d(registration_indices, crop_indices_face)
    
    return registration_indices_face_crop

def find_new_indices(original_shape, crop_indices, desired_original_indices):
    # find the indices of the landmarks in a partial or cropped point cloud according to their position in the original point cloud
    indices_after_crop = []
    remain_after_crop = []
    for i, current_original_index in enumerate(desired_original_indices):
        current_index = np.where(original_shape[crop_indices, :]==
                                 original_shape[current_original_index, :])
        if np.size(current_index)>0:
            indices_after_crop.append(current_index[0][0])
            remain_after_crop.append(i)
    indices_after_crop = np.array(indices_after_crop)

    return indices_after_crop, remain_after_crop

def find_original_indices(original_shape, partial_shape, partial_shape_indices, closest_instead_of_missing=False):
    # find the indices of the landmarks in a point cloud according to their position in the partial or cropped point cloud
    indices_before_crop = []
    for current_index in partial_shape_indices:
        current_original_index = np.where(partial_shape[current_index, :]==original_shape)
        if np.size(current_original_index)>0:
            indices_before_crop.append(current_original_index[0][0])
        else:
            if closest_instead_of_missing:
                distances = np.linalg.norm(partial_shape[current_index, :]-original_shape, axis=1)
                indices_before_crop.append(np.argmin(distances))
            else:
                indices_before_crop.append(np.nan)
    indices_before_crop = np.array(indices_before_crop)

    return indices_before_crop

def get_neighbors(vertices_coordinates, downsampling_indices, neighbors_radius):
    # find the neighbors in the a given radius for each point in the point cloud
    mesh_tree = spatial.KDTree(vertices_coordinates[downsampling_indices, :])

    mesh_vertices_neighbors = mesh_tree.query_ball_tree(mesh_tree, neighbors_radius)

    mesh_vertices_neighbors_num = []

    for _, current_mesh_vertex_neighbors in enumerate(mesh_vertices_neighbors):
        mesh_vertices_neighbors_num.append(len(current_mesh_vertex_neighbors))

    mesh_vertices_neighbors_num = np.array(mesh_vertices_neighbors_num)
    mesh_vertices_neighbors_num_mean = np.mean(mesh_vertices_neighbors_num)
    
    return mesh_tree, mesh_vertices_neighbors_num, mesh_vertices_neighbors_num_mean

def reduce_points_density(mesh_tree, mesh_vertices_neighbors_num, desired_mean_density, neighbors_radius, iteration_remove_rate):
    # reduce the points density in the point cloud by removing points in high density regions with respect to the desired mean density across the point cloud
    current_remaining_indices = list(np.arange(np.array(mesh_vertices_neighbors_num).shape[0]))

    while np.mean(mesh_vertices_neighbors_num)>desired_mean_density:
        current_remaining_indices = list(np.arange(np.array(mesh_vertices_neighbors_num).shape[0]))

        indices_to_remove = np.argsort(mesh_vertices_neighbors_num)[-int(iteration_remove_rate*len(mesh_vertices_neighbors_num)):]

        for index_to_remove in indices_to_remove:
            #skin_removed_indices.append(index_to_remove)
            current_remaining_indices.remove(index_to_remove)

        mesh_tree = spatial.KDTree(mesh_tree.data[current_remaining_indices, :])

        current_mesh_vertices_neighbors = mesh_tree.query_ball_tree(mesh_tree, neighbors_radius)

        mesh_vertices_neighbors_num = []

        for _, current_mesh_vertex_neighbors in enumerate(current_mesh_vertices_neighbors):
            mesh_vertices_neighbors_num.append(len(current_mesh_vertex_neighbors))    
    
    return mesh_tree, mesh_vertices_neighbors_num, current_remaining_indices

def create_injective_correspondence(source_registration_pc, source_correspondence_indices_icp,
                                    target_registration_pc, target_correspondence_indices_icp, use_perpendicular_plane_distance=True):
    # create an injective correspondence set between two point clouds utilizing an existing correspondence set (established by ICP)
    source_injective_values_array = np.unique(source_correspondence_indices_icp)
    target_injective_values_array = np.unique(target_correspondence_indices_icp)

    if target_injective_values_array.shape[0]>source_injective_values_array.shape[0]:
        injective_values_array = source_injective_values_array
        all_values_array = source_correspondence_indices_icp
        corresponding_all_values_array = target_correspondence_indices_icp
    else:
        injective_values_array = target_injective_values_array
        all_values_array = target_correspondence_indices_icp
        corresponding_all_values_array = source_correspondence_indices_icp

    injective_correspondence_set = np.zeros((injective_values_array.shape[0], 2), dtype=int)

    for i, current_index in enumerate(injective_values_array):
        current_indices = np.where(all_values_array==current_index)[0]
        if current_indices.size>1:
            current_distances = []
            for j in current_indices:
                if use_perpendicular_plane_distance:
                    try:
                        current_distance = perpendicular_plane_distance(np.array(target_registration_pc.points)[target_correspondence_indices_icp[j], :],
                                                                        np.array(target_registration_pc.points)[target_correspondence_indices_icp[j], :]+np.array(target_registration_pc.normals)[target_correspondence_indices_icp[j], :],
                                                                        np.array(source_registration_pc.points)[source_correspondence_indices_icp[j], :]
                                                                       )
                    except:
                        print(j, target_correspondence_indices_icp[j])
                        print(np.array(target_registration_pc.points)[target_correspondence_indices_icp[j], :])
                        print(np.array(target_registration_pc.normals)[target_correspondence_indices_icp[j], :])

                else:
                    current_distance = np.linalg.norm(np.array(target_registration_pc.points)[target_correspondence_indices_icp[j], :]-
                                                      np.array(source_registration_pc.points)[source_correspondence_indices_icp[j], :])

                current_distances.append(current_distance)

            current_corresponding_index = corresponding_all_values_array[current_indices[np.argmin(np.array(current_distances))]]
        else:
            current_corresponding_index = corresponding_all_values_array[current_indices[0]]

        current_correspondents = np.array([current_corresponding_index, current_index])

        injective_correspondence_set[i, :] = current_correspondents.astype(int)
    
    return injective_correspondence_set

def annotate_using_another_point_cloud(source_registration_pc, target_registration_pc, target_landmark_indices, correspondent_max_distance_threshold,
                                       use_perpendicular_plane_distance=True):
    # annotate the source point cloud using the target point cloud by finding the closest point in the target point cloud to each point in the source point cloud
    source_landmark_indices = []
    source_missing_landmark_indices = []

    for i, current_target_landmark_index in enumerate(target_landmark_indices):
        current_target_landmark_position = np.array(target_registration_pc.points)[current_target_landmark_index, :]

        if use_perpendicular_plane_distance:
            source_current_landmark_potential_indices = np.argsort(np.linalg.norm(np.array(source_registration_pc.points)-current_target_landmark_position, axis=1))[:10]
            current_landmark_perpendicular_plane_distances = []
            for current_potential_index in source_current_landmark_potential_indices:
                current_landmark_perpendicular_plane_distance = perpendicular_plane_distance(current_target_landmark_position,
                                                                                             current_target_landmark_position+np.array(target_registration_pc.normals)[current_target_landmark_index, :],
                                                                                             np.array(source_registration_pc.points)[current_potential_index, :]
                                                                                      )
                current_landmark_perpendicular_plane_distances.append(current_landmark_perpendicular_plane_distance)

            current_landmark_lowest_distance_index = np.argmin(current_landmark_perpendicular_plane_distances)
            if current_landmark_perpendicular_plane_distances[current_landmark_lowest_distance_index]<correspondent_max_distance_threshold:
                source_current_landmark_index = source_current_landmark_potential_indices[current_landmark_lowest_distance_index]
                source_landmark_indices.append(source_current_landmark_index)
            else:
                source_missing_landmark_indices.append(i)

        else:
            current_landmark_distances = np.linalg.norm(np.array(source_registration_pc.points)-current_target_landmark_position, axis=1)
            current_landmark_lowest_distance_index = np.argmin(current_landmark_distances)
            if current_landmark_distances[current_landmark_lowest_distance_index]<correspondent_max_distance_threshold:
                source_current_landmark_index = current_landmark_lowest_distance_index
                source_landmark_indices.append(source_current_landmark_index)
            else:
                source_missing_landmark_indices.append(i)

    source_landmark_indices = np.array(source_landmark_indices)
    
    return source_landmark_indices, source_missing_landmark_indices

def get_graph_weights(graph, trimesh):
    # fill in the missing edges so the adjacency matrix is not missing any node
    if np.array(graph.nodes()).shape[0]!=np.array(trimesh.vertices).shape[0]:
        #find missing nodes and add them to the graph
        missing_node_indices = np.setdiff1d(np.arange(np.array(trimesh.vertices).shape[0]), np.array(graph.nodes()))
        for i in missing_node_indices:
            graph.add_edge(i, i, weight=0)

    # fill in the isolated nodes so the adjacency matrix is not missing them
    number_of_isolates=nx.number_of_isolates(graph)
    if number_of_isolates>0:
        isolated_node_indices = list(nx.isolates(graph))
        for i in isolated_node_indices:
            graph.add_edge(i, i, weight=0)

    # get the distances along all mesh edges by treating the mesh as a graph and computing its weights
    graph_edges = graph.edges()

    graph_weights_matrix = nx.adjacency_matrix(graph, nodelist=np.array(graph.nodes)).todense()
    for _, current_edge in enumerate(graph_edges):
        euclidean_diff = trimesh.vertices[current_edge[1], :] - trimesh.vertices[current_edge[0], :]
        current_euclidean_distance = np.linalg.norm(euclidean_diff)
        graph[current_edge[0]][current_edge[1]]['weight'] = current_euclidean_distance
        graph_weights_matrix[current_edge[0], current_edge[1]] = current_euclidean_distance
        graph_weights_matrix[current_edge[1], current_edge[0]] = current_euclidean_distance
    return graph, graph_weights_matrix

def get_closest_point(original_trimesh_index, current_trimesh, current_decimated_trimesh):
    # get the closest point of a decimated mesh to a specific point on the original (not decimated) mesh
    original_point_coordinates = current_trimesh.vertices[original_trimesh_index, :]
    current_decimated_trimesh_vertices = np.asarray([current_decimated_trimesh.vertices]).squeeze()
    all_distances = np.linalg.norm(original_point_coordinates-current_decimated_trimesh_vertices, axis=1)
    new_index = np.argmin(all_distances)
    new_point = current_decimated_trimesh_vertices[new_index, :]
    return new_index, new_point

def decompose_transformation_matrix(transformation_matrix):
    translation_coeffs = transformation_matrix[:-1, -1]
    transformation_matrix[:-1, -1] = 0

    x_scale_coeff = np.linalg.norm(transformation_matrix[:-1, 0])
    y_scale_coeff = np.linalg.norm(transformation_matrix[:-1, 1])
    z_scale_coeff = np.linalg.norm(transformation_matrix[:-1, 2])
    scale_coeffs = np.array([x_scale_coeff, y_scale_coeff, z_scale_coeff])
    transformation_matrix[:-1, 0] = transformation_matrix[:-1, 0]/x_scale_coeff
    transformation_matrix[:-1, 1] = transformation_matrix[:-1, 1]/y_scale_coeff
    transformation_matrix[:-1, 2] = transformation_matrix[:-1, 2]/z_scale_coeff

    rotation_matrix = transformation_matrix[:-1, :-1]

    alpha = np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2])
    beta = np.arctan2(-rotation_matrix[2,0], np.sqrt(rotation_matrix[2,1]**2 + rotation_matrix[2,2]**2))
    gamma = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])
    rotation_coeefs = np.array([alpha, beta, gamma])

    return translation_coeffs, scale_coeffs, rotation_coeefs