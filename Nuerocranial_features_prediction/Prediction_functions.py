import numpy as np
import os
import trimesh
import copy
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# get the landmarks indices in the new shape
def get_new_CCS_indices(new_shape_CCS, old_shape_CCS, landmarks):
    new_indices = np.zeros(len(landmarks)).astype(int)
    for current_landmark_index in range(len(landmarks)):
        new_indices[current_landmark_index] = np.where(new_shape_CCS==old_shape_CCS[landmarks[current_landmark_index], :])[0][0]
    return new_indices

# get the landmarks indices in the new shape trimesh in case they are not the same as in the vertices array
def get_trimesh_indices(shape_trimesh, shape_CCS, landmarks):
    trimesh_indices = np.zeros(len(landmarks)).astype(int)
    for current_landmark_index in range(len(landmarks)):
        try:
            trimesh_indices[current_landmark_index] = np.where(shape_CCS[landmarks[current_landmark_index], :]==
                                                               shape_trimesh.vertices)[0][0]
        except:
            print(current_landmark_index, "landmark wasn't found in the trimesh")
            trimesh_indices[current_landmark_index] = 0
        #print(trimesh_indices[current_landmark_index])
    return trimesh_indices

# get the landmarks indices in the vertices array in case they are not the same as in the trimesh
def get_CCS_indices(shape_trimesh, shape_CCS, landmarks):
    CCS_indices = np.zeros(len(landmarks)).astype(int)
    for current_landmark_index in range(len(landmarks)):
        CCS_indices[current_landmark_index] = np.where(shape_CCS==np.array(shape_trimesh.vertices[landmarks[current_landmark_index], :]))[0][0]
    return CCS_indices

# get the trimesh graph weights as euclidean distances between the nodes
def get_graph_weights(shape_graph, shape_trimesh, num_digits_round=10):
    shape_graph_edges = shape_graph.edges()
    for current_index, current_edge in enumerate(shape_graph_edges):
        euclidean_diff = shape_trimesh.vertices[current_edge[1], :] - shape_trimesh.vertices[
            current_edge[0], :]
        current_euclidean_distance = np.linalg.norm(euclidean_diff).round(num_digits_round+2)
        shape_graph[current_edge[0]][current_edge[1]]['euclidean_distance'] = current_euclidean_distance
    return shape_graph

# get the closest point of certain shape vertices in another shape 
def get_closest_point(original_trimesh_index, current_shape_trimesh, current_shape_decimated_trimesh):
    original_point_coordinates = current_shape_trimesh.vertices[original_trimesh_index, :]
    current_shape_decimated_trimesh_vertices = np.asarray([current_shape_decimated_trimesh.vertices]).squeeze()
    all_distances = np.linalg.norm(original_point_coordinates-current_shape_decimated_trimesh_vertices, axis=1)
    new_index = np.argmin(all_distances)
    new_point = current_shape_decimated_trimesh_vertices[new_index, :]
    return new_index, new_point

# define the nn model class for the regression task - Multi Layer Perceptron
class MLP_3(nn.Module):
    def __init__(self, inputs_size, output_size, first_hidden_layer_size=2**7, second_hidden_layer_size=2**5, third_hidden_layer_size=2**3):
        super().__init__()
        #self.layers = nn.Sequential(
        self.inputs_size = inputs_size
        
        self.fc1 = nn.Linear(inputs_size, first_hidden_layer_size)
        self.bn1 = nn.BatchNorm1d(first_hidden_layer_size)
        self.d1 = nn.Dropout(p=0.3, inplace=False)
        
        self.fc2 = nn.Linear(first_hidden_layer_size, second_hidden_layer_size)
        self.bn2 = nn.BatchNorm1d(second_hidden_layer_size)
        self.d2 = nn.Dropout(p=0.2, inplace=False)
        
        self.fc3 = nn.Linear(second_hidden_layer_size, third_hidden_layer_size)
        self.bn3 = nn.BatchNorm1d(third_hidden_layer_size)
        self.d3 = nn.Dropout(p=0.2, inplace=False)
        
        self.fc4 = nn.Linear(third_hidden_layer_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x) #torch.tanh(x)
        x = self.bn1(x)
        x = self.d1(x)
        
        x = self.fc2(x)
        x = F.leaky_relu(x) #torch.tanh(x)
        x = self.bn2(x)
        x = self.d2(x)
        
        x = self.fc3(x)
        x = F.leaky_relu(x) #torch.tanh(x)
        x = self.bn3(x)
        x = self.d3(x)

        x = self.fc4(x)
        return x
    
# find the most recent trained model for a predicting selected feature in a folder
def find_most_recent_trained_files(trained_folder_path, desired_feature_name, fold_index=-1):
    trained_folder_and_features_set_files = trained_folder_path
    trained_folder_files = os.listdir(trained_folder_and_features_set_files)

    potential_files_creation_time = []
    potential_files_index = []
    
    for i, current_filename in enumerate(trained_folder_files):
        #print(current_filename)
        if desired_feature_name in current_filename and 'model' in current_filename:# and desired_decimation_percentage in current_filename:
            if fold_index != -1:# and len(current_filename)>30:
                if fold_index==int(current_filename[-7]):
                    current_filename_creation_time = current_filename[:17]
                    potential_files_index.append(i)
                    potential_files_creation_time.append(current_filename_creation_time)
            else:
                current_filename_creation_time = current_filename[:17]
                potential_files_index.append(i)
                potential_files_creation_time.append(current_filename_creation_time)
    
    most_recent_trained_model_filename = trained_folder_files[potential_files_index[np.argsort(potential_files_creation_time)[-1]]]
    print(most_recent_trained_model_filename)
    most_recent_trained_scaler_filename = most_recent_trained_model_filename[:-6]+'_scaler.pkl'
    most_recent_trained_documentation_filename = most_recent_trained_model_filename[:-6]+'_documentation.txt'
    
    most_recent_trained_files = [most_recent_trained_model_filename, most_recent_trained_scaler_filename, most_recent_trained_documentation_filename]
    
    return most_recent_trained_files

# run the model for a single epoch
def run_model(model, dataloader, loss_function, optimizer, output_size, mode):
    current_loss = 0.0
    if mode=='train':
        model.train()
    else:
        model.eval()
    # Iterate over the DataLoader for training data
    for i, data in enumerate(dataloader):
        # Get and prepare inputs
        inputs, targets = data
        inputs, targets = inputs.float().to(device), targets.float().to(device)
                
        targets = targets.reshape((targets.shape[0], output_size))
        if mode=='train':
            # Zero the gradients
            optimizer.zero_grad()
        # Perform forward pass
        current_outputs = model(inputs)
        # Compute loss
        loss = loss_function(current_outputs, targets)
        if mode=='test':
            if i==0:
                outputs = current_outputs
            else:
                outputs = torch.cat((outputs, current_outputs), dim=0)
        if mode=='train':
            # Perform backward pass
            loss.backward()
            # Perform optimization
            optimizer.step()
        #if mode!='test':
        current_loss += loss.item()
        
    if mode=='test':
        return current_loss, model, outputs
    else:
        return current_loss, model

# runs MRI model training and prediction
def MRI_model_choose_and_predict(mlp, X_train, y_train, X_test, y_test, mode='train', n_jobs_num=1, num_of_epochs=500):
    np.random.seed(1)
    
    validation_losses = []
    test_losses = []
    validation = False
    batch_size = 2**3
    
    if mode == 'train':
        lr = 5e-3
    elif mode == 'fine_tune':
        lr = 2.5e-3
    else:
        lr = 0

    loss_function = nn.MSELoss() #nn.L1Loss
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    
    if y_test.shape[1]>1:
        output_size = 3
    else:
        output_size = 1
    
    if mode != 'test_only':
        validation = False # False / True
        if np.array_equal(y_test, np.array([-1, -1, -1]).reshape(1, 3)): #only training on the entire dataset
            validation = False
        triggered = False

        train_input = torch.tensor(X_train.astype(np.float32))
        train_target = torch.tensor(y_train.astype(np.float32)) 

        train_tensor = TensorDataset(train_input, train_target)

        if validation==True:
            train_set_percentage = 0.8
            last_validation_loss = 1e6
            lowest_validation_loss = 1e6
            patience = 4
            #tolerance = 0.001*1e-3
            trigger_times = 0

            train_set_size = int(np.round(train_set_percentage*train_input.shape[0]))
            valid_set_size = train_input.shape[0]-train_set_size
            
            if train_set_size%batch_size==1 and valid_set_size%batch_size!=0:
                train_set_size -= 1
                valid_set_size += 1
            elif valid_set_size%batch_size==1 and train_set_size%batch_size!=0:
                train_set_size += 1
                valid_set_size -= 1
            elif train_set_size%batch_size==1:
                train_set_size -= 1
                train_tensor = TensorDataset(train_input[:-1, :], train_target[:-1, :])
            elif valid_set_size%batch_size==1:
                valid_set_size -= 1
                train_tensor = TensorDataset(train_input[:-1, :], train_target[:-1, :])

            train_tensor, valid_tensor = random_split(train_tensor, [train_set_size, valid_set_size])

            validloader = DataLoader(dataset=valid_tensor, batch_size=batch_size, shuffle=False)

        trainloader = DataLoader(dataset=train_tensor, batch_size=batch_size, shuffle=False)
    
    if np.array_equal(y_test, np.array([-1, -1, -1]).reshape(1, 3))==False:
        test_input = torch.tensor(X_test.astype(np.float32))
        test_target = torch.tensor(y_test.astype(np.float32))

        test_tensor = TensorDataset(test_input, test_target) 
        testloader = DataLoader(dataset=test_tensor, batch_size=batch_size, shuffle=False)

    # Run the training loop
    for epoch in range(num_of_epochs):
        if mode != 'test_only':
            if triggered==True:
                continue

            train_loss, mlp = run_model(mlp.to(device), trainloader, loss_function, optimizer, output_size, mode='train')

            if validation==True:
                with torch.no_grad():
                    validation_loss, _ = run_model(mlp.to(device), validloader, loss_function, optimizer, output_size, mode='validation')
                    validation_losses.append(validation_loss)

                    last_validation_loss = validation_loss
                    if validation_loss<lowest_validation_loss:
                        lowest_validation_loss=validation_loss
                        lowest_validation_loss_model=copy.deepcopy(mlp)
                        lowest_validation_loss_epoch=epoch

                    if last_validation_loss < validation_loss and epoch>50:
                        trigger_times += 1
                        if trigger_times > patience:
                            triggered = True
                            print(('Early stopping at epoch '+str(epoch)))
                    else:
                        trigger_times = 0

            test_loss , _, _ = run_model(mlp.to(device), testloader, loss_function, optimizer, output_size, mode='test')
            scheduler.step()

        # Disable grad
        if np.array_equal(y_test, np.array([-1, -1, -1]).reshape(1, 3))==False:
            with torch.no_grad():
                test_loss , _, predictions = run_model(mlp.to(device), testloader, loss_function, optimizer, output_size, mode='test')
                test_losses.append(test_loss)
                #print(epoch, test_loss)

            
    if mode != 'test_only' and np.array_equal(y_test, np.array([-1, -1, -1]).reshape(1, 3))==False:
        lowest_validation_loss_epoch=num_of_epochs-1
        lowest_validation_loss_model=copy.deepcopy(mlp)
        print(f"lowest_validation_loss_epoch is {num_of_epochs-1}")

        if validation==True:
            with torch.no_grad():
                _ , _, predictions = run_model(lowest_validation_loss_model.to(device), testloader, loss_function, optimizer, output_size, mode='test')
    
    if mode != 'test_only' and np.array_equal(y_test, np.array([-1, -1, -1]).reshape(1, 3))==False:#validation==True:
        return predictions, validation_losses, test_losses, lowest_validation_loss_model, lowest_validation_loss_epoch
    elif np.array_equal(y_test, np.array([-1, -1, -1]).reshape(1, 3)):
        return mlp
    else:
        return predictions, validation_losses, test_losses, 'a', 'b'

# transform a given prediction coordinates from between the subject's and 3DMM space
def transform_to_original_space(df, prediction, subject_id):
    number_of_transformations=5
    coordinates = []
    coordinates.append(prediction)
    for i in range(number_of_transformations):
        current_transformation_matrix = df.iloc[(number_of_transformations-1-i)*4:(number_of_transformations-i)*4, np.arange(4*subject_id, 4*(subject_id+1))].values
        current_extended_coordinates = np.concatenate((np.array(coordinates[i]).reshape(1, coordinates[i].shape[0]), np.ones((1, 1))), axis=1).T
        current_product = current_transformation_matrix@current_extended_coordinates
        coordinates.append(current_product[:3])
    return coordinates[-1]

# project a given distance into the perpendicular and tangent components
def project_distances(plane_normal, plane_point, other_point):
    plane_normal = plane_normal/np.linalg.norm(plane_normal)
    distance_vector = other_point-plane_point
    perpendicular_distance = np.dot(distance_vector, plane_normal)
    plane_projected_other_point = other_point - perpendicular_distance*plane_normal
    tangent_distance = np.linalg.norm(plane_projected_other_point-plane_point)
    return perpendicular_distance, tangent_distance

# change keys of an ordered dictionary for compatibility with older versions of pytorch
def change_keys(OrderedDict):
    newOrderedDict = OrderedDict.copy()
    for i, key in enumerate(OrderedDict):
        key, value = newOrderedDict.popitem(False)
        if 'module.' in key:
            key = key.replace('module.', '')
            newOrderedDict[key] = value
        else:
            newOrderedDict[key] = value
    return newOrderedDict

'''
def figure_saver(fig, media_folder, filename):
    pio.kaleido.scope.mathjax = None

    timestamp_string = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    timestamp_string = timestamp_string.replace('_2022_', '_22_')

    figure_folder = media_folder+"/Cranium_estimation_paper/Figures/Predictions/"
    figure_filename = f"{filename}_{timestamp_string}" #???"experiment_"+str(experiment_number)+"_"+timestamp_string+"_"+filename
    figure_filetype = ".eps"

    figure_path = figure_folder + figure_filename + figure_filetype

    fig.layout.title = ""
    
    fig.layout.margin.t = 0#0.075*fig.layout.height
    fig.layout.margin.b = 0
    fig.layout.margin.l = 0
    fig.layout.margin.r = 10#0.05*fig.layout.width

    pio.write_image(fig, figure_path)#, scale=5)
    pio.write_json(fig, file=figure_folder+figure_filename+".json" ,validate=True, pretty=True, remove_uids=False, engine='json')

def figure_saver(experiment_number, arrays_list, regressor, save_figures=True, save_arrays=True):
    if regressor=='MLP' or regressor=='pytorch_MLP':
        regressor_name = 'pytorch_MLP'

    timestamp_string = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    timestamp_string = timestamp_string.replace('_2022_', '_22_')

    figure_folder = media_folder+"/Cranium_estimation_paper/Figures/Predictions/"+regressor_name+"/std/"
    figure_filetype = ".jpg"
    figure_filename = "experiment_"+str(experiment_number)+"_"+timestamp_string+"_"+filename

    figure_path = figure_folder + figure_filename + figure_filetype

    fig.layout.title = ""

    fig.update_layout(yaxis = dict(dtick = 0.25))  
    fig.update_layout(yaxis_range=[0,3])

    if save_figures:
        pio.write_image(fig, figure_path, scale=5)

    figure_folder = media_folder+"/Cranium_estimation_paper/Figures/Predictions/"+regressor_name+"/no_std/"
    figure_filetype = ".jpg"
    figure_filename = "experiment_"+str(experiment_number)+"_"+timestamp_string+"_"+filename

    figure_path = figure_folder + figure_filename + figure_filetype
    
    no_std_traces = []
    
    for current_trace_index in range(len(fig.data)):
        if np.mod(current_trace_index, 3)==0:
            no_std_traces.append(fig.data[current_trace_index])

    fig.data = no_std_traces

    fig.update_layout(yaxis_range=[0,2])

    if save_figures:
        pio.write_image(fig, figure_path, scale=5)
    
    if save_arrays:
        array_folder = media_folder+"/Cranium_estimation_paper/Figures/Predictions/"+regressor_name+"/"
        array_filetype = '.npy'

        array_path = array_folder + figure_filename + array_filetype

        with open(array_path, 'wb') as file:
            for i in range(len(arrays_list)):
                np.save(file, arrays_list[i])
'''

# runs Synthetic model training and prediction
def Synthetic_model_choose_and_predict(regressor_name, X_train, y_train, X_test, y_test, n_jobs_num=1):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'Using {torch.cuda.get_device_name(0)}')
    else:
        device = torch.device("cpu")
        print('Using CPU')
            
    if regressor_name=='pytorch_MLP' or regressor_name=='MLP':
        
        test_losses = []
        
        validation = False # False / True
        validation_losses = []
        
        batch_size = 2**3
        lr = 5e-3

        triggered = False

        if y_test.shape[1]>1:
            output_size = 3
        else:
            output_size = 1
        
        train_target = torch.tensor(y_train.astype(np.float32)) 
        train_input = torch.tensor(X_train.astype(np.float32))
        
        train_tensor = TensorDataset(train_input, train_target)
                
        if validation==True:
            train_set_percentage = 0.8
            lowest_validation_loss = 1e6

            train_set_size = int(np.round(train_set_percentage*train_input.shape[0]))
            valid_set_size = train_input.shape[0]-train_set_size

            train_tensor, valid_tensor = random_split(train_tensor, [train_set_size, valid_set_size])

            validloader = DataLoader(dataset=valid_tensor, batch_size=batch_size, shuffle=False)
            
        trainloader = DataLoader(dataset=train_tensor, batch_size=batch_size, shuffle=False)

        test_input = torch.tensor(X_test.astype(np.float32))
        test_target = torch.tensor(y_test.astype(np.float32))

        test_tensor = TensorDataset(test_input, test_target) 
        testloader = DataLoader(dataset=test_tensor, batch_size=batch_size, shuffle=False)

        loss_function = nn.MSELoss()
        if output_size>1:
            mlp = MLP_3(X_train.shape[1], output_size)
            mlp = nn.DataParallel(mlp, device_ids=[0])
            print('MLP')
        else:
            mlp = MLP_3(X_train.shape[1], output_size)
            mlp = nn.DataParallel(mlp, device_ids=[0])
            print('MLP')
        optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.99)#, verbose=True)

        # Run the training loop
        for epoch in range(1000):
            #print(epoch)
            if triggered==True:
                continue
                
            train_loss, regressor_model = run_model(mlp.to(device), trainloader, loss_function, optimizer, output_size, mode='train')
            
            if validation==True:
                with torch.no_grad():
                    validation_loss, _ = run_model(mlp.to(device), validloader, loss_function, optimizer, output_size, mode='validation')
                    validation_losses.append(validation_loss)
                    
                    last_validation_loss = validation_loss
                    if validation_loss<lowest_validation_loss:
                        lowest_validation_loss=validation_loss
                        lowest_validation_loss_model=copy.deepcopy(mlp)
                        lowest_validation_loss_epoch=epoch

            scheduler.step()
            
            # Disable grad
            with torch.no_grad():
                test_loss , _, current_predictions = run_model(mlp.to(device), testloader, loss_function, optimizer, output_size, mode='test')
                test_losses.append(test_loss)
                
        if validation==True:
            print(f"lowest_validation_loss_epoch is {lowest_validation_loss_epoch}")
        else:
            print(f"lowest_validation_loss_epoch is {epoch}")
        
        if validation==True:
            with torch.no_grad():
                _ , _, lowest_validation_loss_predictions = run_model(lowest_validation_loss_model.to(device), testloader, loss_function, optimizer, output_size, mode='test')
    
    if validation==True:
        return lowest_validation_loss_predictions, lowest_validation_loss_model, validation_losses, test_losses
    else:
        return current_predictions, regressor_model, validation_losses, test_losses

# load results arrays from the most recent experiment
def experiment_arrays_loader(folder, regressor_name, experiment_number):
    array_folder = folder+"/Cranium_estimation_paper/Figures/Predictions/"+regressor_name+"/"

    files = []
    for current_file in os.listdir(array_folder):
        if current_file.startswith(("experiment_"+str(experiment_number))) and current_file.endswith(".npy"):
            files.append(current_file)

    files.sort()
    last_file_created = files[-1]

    with open(last_file_created, 'rb') as file:
        MSE_array = np.load(file)
        std_array = np.load(file)
        
    return MSE_array, std_array

# train and test models for each landmark using coordinates only as features
def coordinates_by_coordinates_regression(df, train_indices, test_indices, features_to_use, predicted_feature_index, 
                                         landmarks_names, regressor_name='MLP'):
    y_train = np.asarray(df.loc[train_indices, ([landmarks_names[predicted_feature_index]])])#.reshape((-1, 3))
    y_test = np.asarray(df.loc[test_indices, ([landmarks_names[predicted_feature_index]])])#.reshape((-1, 3))
    
    features_to_use = np.take(np.asarray(landmarks_names), features_to_use)
    X_train = np.asarray(df.loc[train_indices, features_to_use])
    X_test = np.asarray(df.loc[test_indices, features_to_use])

    means = np.mean(X_train, axis=0)
    
    X_standard_scaler = StandardScaler()
    X_train = X_standard_scaler.fit_transform(X_train)
    X_test = X_standard_scaler.transform(X_test)
    
    current_predictions, regressor_model, validation_losses, test_losses = Synthetic_model_choose_and_predict(regressor_name, X_train, y_train, X_test, y_test)

    euclidean_errors = np.linalg.norm(current_predictions.to("cpu")-y_test, axis=1)
    squared_euclidean_errors = euclidean_errors**2
    mean_squared_euclidean_error = np.mean(squared_euclidean_errors)
    std_euclidean_errors = np.std(euclidean_errors)
   
    return regressor_model, mean_squared_euclidean_error, squared_euclidean_errors, std_euclidean_errors, X_standard_scaler, means, validation_losses, test_losses

# train and test models for each landmark using coordinates and geodesic distances as features
def coordinates_by_coordinates_and_geodesic_distance_regression(coordinates_df, geodesic_df, train_indices, test_indices, features_to_use,
                                                                predicted_feature_index, landmarks_names, regressor_name='MLP'):
    y_train = np.asarray(coordinates_df.loc[train_indices, ([landmarks_names[predicted_feature_index]])])
    y_test = np.asarray(coordinates_df.loc[test_indices, ([landmarks_names[predicted_feature_index]])])
    
    features_to_use = np.take(np.asarray(landmarks_names), features_to_use)
    
    coordinates_train = np.asarray(coordinates_df.loc[train_indices, features_to_use])
    coordinates_test = np.asarray(coordinates_df.loc[test_indices, features_to_use])
    
    geodesic_data = np.zeros((geodesic_df.shape[0], 3))
    for i in range(geodesic_df.shape[0]):
        geodesic_data[i, :2] = np.array(geodesic_df.iloc[i, :2])
        geodesic_data[i, 2] = np.sum(np.array(geodesic_df.iloc[i, 2:]))
        
    geodesic_train = geodesic_data[train_indices, :]
    geodesic_test = geodesic_data[test_indices, :]

    X_train = np.concatenate((coordinates_train, geodesic_train), axis=1)
    print(coordinates_train.shape, geodesic_train.shape, X_train.shape)
    X_test = np.concatenate((coordinates_test, geodesic_test), axis=1)
    
    X_standard_scaler = StandardScaler()
    X_train = X_standard_scaler.fit_transform(X_train)
    X_test = X_standard_scaler.transform(X_test)
    
    means = np.mean(X_train, axis=0)
       
    current_predictions, regressor_model, validation_losses, test_losses = Synthetic_model_choose_and_predict(regressor_name, X_train, y_train, X_test, y_test)
    
    euclidean_errors = np.linalg.norm(current_predictions.to("cpu")-y_test, axis=1)
    squared_euclidean_errors = euclidean_errors**2
    mean_squared_euclidean_error = np.mean(squared_euclidean_errors)
    std_euclidean_errors = np.std(euclidean_errors)
   
    return regressor_model, mean_squared_euclidean_error, squared_euclidean_errors, std_euclidean_errors, X_standard_scaler, means, validation_losses, test_losses