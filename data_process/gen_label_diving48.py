import os
import json

dataset_name = 'Diving48'
root_dir = '/data_video/diving48/frames'

files_input = ["/data_video/diving48/Diving48_test.json","/data_video/diving48/Diving48_train.json"]
files_output = ['/data/tianyuan/AST/data_process/diving48/val_videofolder.txt','/data/tianyuan/AST/data_process/diving48/train_videofolder.txt']
for (filename_input, filename_output) in zip(files_input, files_output):
    with open(os.path.join(root_dir, dataset_name, filename_input), 'r') as f:
        data = json.load(f)
    output = []
    for i in range(len(data)):
        output.append('%s %d %d'%(data[i]['vid_name'], data[i]['end_frame'], data[i]['label']))
        print('%d/%d'%(i, len(data)))
    with open(os.path.join(root_dir, dataset_name, filename_output),'w') as f:
        f.write('\n'.join(output))