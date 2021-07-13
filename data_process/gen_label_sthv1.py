# processing the raw data of the video datasets (Something-something and jester)
# generate the meta files:
#   category.txt:               the list of categories.
#   train_videofolder.txt:      each row contains [videoname num_frames classIDX]
#   val_videofolder.txt:        same as above
#
# Bolei Zhou, Dec.2 2017
#
#
import os
import pdb
root_dir = "/media/ps/Data_disk_2/yuan_space/sth-sth-v1/"
dataset_name = 'something-something-v1' # 'jester-v1'
with open(root_dir+'%s-labels.csv'% dataset_name) as f:
    lines = f.readlines()
categories = []
for line in lines:
    line = line.rstrip()
    categories.append(line)
categories = sorted(categories)
with open('category.txt','w') as f:
    f.write('\n'.join(categories))

dict_categories = {}
for i, category in enumerate(categories):
    dict_categories[category] = i

files_input = [root_dir+'%s-validation.csv'%dataset_name,root_dir+'%s-train.csv'%dataset_name]
files_output = ['val_videofolder.txt','train_videofolder.txt']
for (filename_input, filename_output) in zip(files_input, files_output):
    with open(filename_input) as f:
        lines = f.readlines()
    folders = []
    idx_categories = []
    for line in lines:
        line = line.rstrip()
        items = line.split(';')
        # print(items,dict_categories[items[1]],items[1])
        folders.append(items[0])
        idx_categories.append(os.path.join(str(dict_categories[items[1]])))
    output = []
    for i in range(len(folders)):
        curFolder = folders[i]
        curIDX = idx_categories[i]
        print(curFolder,curIDX)
        # counting the number of frames in each video folders
        dir_files = os.listdir(os.path.join('/home/ps/sthv1/20bn-%s'%dataset_name, curFolder))
        output.append('%s %d %d'%(str(curFolder), len(dir_files), int(curIDX)))
        print('%d/%d'%(i, len(folders)))
    with open(filename_output,'w') as f:
        f.write('\n'.join(output))