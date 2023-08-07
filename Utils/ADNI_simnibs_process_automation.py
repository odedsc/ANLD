# import packages
import subprocess
import os
import time
import numpy as np
from datetime import datetime

# definitions
processed_folder = './MRI_datasets/ADNI/Processed/'
processed_subjects_folders = os.listdir(processed_folder)
processed_subjects_folders.sort()

images_folder = './MRI_datasets/ADNI/Raw/'
all_subjects_folders = os.listdir(images_folder)
all_subjects_folders.sort()



simnibs_path_export_command = 'SIMNIBS_BIN="/home/oded/SimNIBS-3.2/bin"; PATH="/home/oded/miniconda3/bin:$PATH"; export PATH=${PATH}:${SIMNIBS_BIN}'

num_parallel_processes = 6

# run simnibs segmentation on the subjects MR images
num_of_subjects = len(all_subjects_folders)
currently_running_processes = []
currently_running_subjects_indices = []
currently_running_subjects_times = []
current_subject_index = 0

while current_subject_index<num_of_subjects:
    current_subject_ID = all_subjects_folders[current_subject_index]
    if current_subject_ID in processed_subjects_folders or current_subject_ID+'_Scaled_2' in processed_subjects_folders:
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
        subject_image_filenames = []
                
        for root, dirs, files in os.walk(images_folder+current_subject_ID):
            for file in files:
                if file.endswith(".nii"):
                    subject_image_filenames.append(root+'/'+file)
                
        if len(subject_image_filenames)==1:
            image_filename = subject_image_filenames[0]
        else:
            for current_filename in subject_image_filenames:
                if 'scaled_2' in current_filename or 'Scaled_2' in current_filename:
                    image_filename = current_filename
                
        if 'scaled_2' not in image_filename and 'Scaled_2' not in image_filename:
            os.mkdir(processed_folder+current_subject_ID)
            os.chdir(processed_folder+current_subject_ID)
        else:
            os.mkdir(processed_folder+current_subject_ID+'_Scaled_2')
            os.chdir(processed_folder+current_subject_ID+'_Scaled_2')
        
        current_headreco_command = f"headreco all --no-cat {current_subject_ID}_no_cat {image_filename}"
        
        current_command = simnibs_path_export_command + f';{current_headreco_command}'

        current_process = subprocess.Popen(current_command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        print(f"Started working on {current_subject_ID}")
        currently_running_processes.append(current_process)
        currently_running_subjects_indices.append(current_subject_ID)
        currently_running_subjects_times.append(datetime.now())
        current_subject_index += 1
    
    time.sleep(60)

print("Done!")