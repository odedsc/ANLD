# import packages
import subprocess
import os
import time
import numpy as np
from datetime import datetime

# definitions
processed_folder = './MRI_datasets/IXI/Processed/'
processed_subjects_folders = os.listdir(processed_folder)
processed_subjects_folders.sort()

T1_images_folder = './MRI_datasets/IXI/Raw/T1/'
all_subjects_T1_folders = os.listdir(T1_images_folder)
all_subjects_T1_folders.sort()

T2_images_folder = './MRI_datasets/IXI/Raw/T2/'

simnibs_path_export_command = 'SIMNIBS_BIN="/home/oded/SimNIBS-3.2/bin"; PATH="/home/oded/miniconda3/bin:$PATH"; export PATH=${PATH}:${SIMNIBS_BIN}'

num_parallel_processes = 6

# run simnibs segmentation on the subjects MR images
num_of_subjects = len(all_subjects_T1_folders)
currently_running_processes = []
currently_running_subjects_indices = []
currently_running_subjects_times = []
current_subject_index = 0

while current_subject_index<num_of_subjects:
    current_subject_name = all_subjects_T1_folders[current_subject_index]
    current_subject_ID = current_subject_name.rsplit('-', 1)[0]
    if current_subject_ID in processed_subjects_folders:
        print(f"Subject {current_subject_ID} already processed")
        current_subject_index += 1
        continue
        
    for process_index in np.arange(len(currently_running_processes))[::-1]: # Check the currently_running_processes in reverse order
        if currently_running_processes[process_index].poll() is not None: # If the process hasn't finished will return None
            del currently_running_processes[process_index] # Remove from list - this is why we needed reverse order
            print(f"Finished working on {currently_running_subjects_indices[process_index]}, Process time was: {datetime.now()-currently_running_subjects_times[process_index]}")
            del currently_running_subjects_indices[process_index]
            del currently_running_subjects_times[process_index]
    
    if len(currently_running_processes)<num_parallel_processes and current_subject_ID not in processed_subjects_folders: # More to do and some spare slots        
        T1_image_filename = current_subject_ID+"-T1.nii.gz"
        T2_image_filename = current_subject_ID+"-T2.nii.gz"
        
        os.mkdir(processed_folder+current_subject_ID)
        os.chdir(processed_folder+current_subject_ID)
        
        current_headreco_command = f"headreco all --no-cat {current_subject_ID}_no_cat {T1_images_folder+T1_image_filename} {T2_images_folder+T2_image_filename}"
        
        current_command = simnibs_path_export_command + f';{current_headreco_command}'

        current_process = subprocess.Popen(current_command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        print(f"Started working on {current_subject_ID}")
        currently_running_processes.append(current_process)
        currently_running_subjects_indices.append(current_subject_ID)
        currently_running_subjects_times.append(datetime.now())
        current_subject_index += 1
    
    time.sleep(60)

print("Done!")