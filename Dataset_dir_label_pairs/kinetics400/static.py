import os
root_dir = "/media/ps/Data_disk_2/yuan_space/kinetics-400-jpg/train_256"
# root_dir = "/media/ps/Data_disk_2/yuan_space/kinetics-400-data/raw-part/compress/train_256"
min =99999999999
max = -1
for action_name in os.listdir(root_dir):
    action_dir = os.path.join(root_dir,action_name)
    for file_name in os.listdir(action_dir):
        file_path  = os.path.join(action_dir,file_name)
        file_len = len(os.listdir(file_path))
        if min>file_len:
            min = file_len
        if max<file_len:
            max = file_len
print(min,max)
