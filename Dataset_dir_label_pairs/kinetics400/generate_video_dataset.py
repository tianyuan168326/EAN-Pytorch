import os
train_dir = "/media/ps/Data_disk_2/yuan_space/kinetics-400-jpg/train_256"
val_dir = "/media/ps/Data_disk_2/yuan_space/kinetics-400-jpg/val_256"
action_names = os.listdir(train_dir)
action_dict = {}
for action_i in range(len(action_names)):
    action_dict.update({action_names[action_i]:str(action_i)})
# root_dir = "/media/ps/Data_disk_2/yuan_space/kinetics-400-data/raw-part/compress/train_256"
open("./category.txt","w").writelines(list(map(lambda n: n+"\n",action_names)))

train_items = []
for action_name in os.listdir(train_dir):
    action_dir = os.path.join(train_dir,action_name)
    for file_name in os.listdir(action_dir):
        file_path  = os.path.join(action_dir,file_name)
        file_len = len(os.listdir(file_path))
        file_sub_path = os.path.join("train_256",action_name,file_name)
        train_item = file_sub_path +" "+ str(file_len) +" " + action_dict[action_name]+"\n"
        train_items += [train_item]
        # break
open("./train.txt","w").writelines(train_items)


val_items = []
for action_name in os.listdir(val_dir):
    action_dir = os.path.join(val_dir,action_name)
    for file_name in os.listdir(action_dir):
        file_path  = os.path.join(action_dir,file_name)
        file_len = len(os.listdir(file_path))
        file_sub_path = os.path.join("val_256",action_name,file_name)
        train_item = file_sub_path +" "+ str(file_len) +" " + action_dict[action_name]+"\n"
        val_items += [train_item]
        # break

open("./val.txt","w").writelines(val_items)