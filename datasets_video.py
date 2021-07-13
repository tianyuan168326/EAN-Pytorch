import os
import torch
import torchvision
import torchvision.datasets as datasets


def return_somethingv1(ROOT_DATASET):
    filename_categories = 'Dataset_dir_label_pairs/somethingv1/category.txt'
    # root_data = "/home/ps/sthv1/20bn-something-something-v1"
    root_data = "/data_video/sthv1_frames/sth-sth-v1/20bn-something-something-v1"
    filename_imglist_train = 'Dataset_dir_label_pairs/somethingv1/train.txt'
    filename_imglist_val = 'Dataset_dir_label_pairs/somethingv1/valid.txt'
    prefix = '{:05d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix

def return_kinetics400(ROOT_DATASET):
    filename_categories = 'Dataset_dir_label_pairs/kinetics400/category.txt'
    root_data = "/data_video/important_dataset/kinetics-400-jpg"
    filename_imglist_train = 'Dataset_dir_label_pairs/kinetics400/train.txt'
    filename_imglist_val = 'Dataset_dir_label_pairs/kinetics400/valid.txt'
    prefix = 'image_{:05d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix
def return_kinetics200(ROOT_DATASET):
    filename_categories = 'kinetics200/category.txt'
    root_data = "/data_video/important_dataset/kinetics-400-jpg"
    filename_imglist_train = 'kinetics200/train_k200.txt'
    filename_imglist_val = 'kinetics200/test_k200.txt'
    prefix = 'image_{:05d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix
def return_diving48(ROOT_DATASET):
    filename_categories = 'Dataset_dir_label_pairs/diving48/category.txt'
    root_data = "/data_video/diving48/frames"
    filename_imglist_train = 'Dataset_dir_label_pairs/diving48/train_videofolder.txt'
    filename_imglist_val = 'Dataset_dir_label_pairs/diving48/val_videofolder.txt'
    prefix = 'frames{:05d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix
def return_somethingv2(ROOT_DATASET):
    filename_categories =  'Dataset_dir_label_pairs/somethingv2/category.txt'
    root_data = "/data_video/important_dataset/sthv2-frames"
    # root_data = "/media/ps/SSD/tianyuan/sthv2-frames"
    filename_imglist_train = 'Dataset_dir_label_pairs/somethingv2/train.txt'
    filename_imglist_val = 'Dataset_dir_label_pairs/somethingv2/valid.txt'
    prefix = '{:06d}.jpg'
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_dataset(dataset,ROOT_DATASET):
    dict_single = { 
        'somethingv1':return_somethingv1,
        'somethingv2':return_somethingv2,
        'k400':return_kinetics400,
        "k200":return_kinetics200,
        "diving48":return_diving48,
     }
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](ROOT_DATASET)
    else:
        raise ValueError('Unknown dataset '+dataset)

    file_imglist_train = os.path.join(ROOT_DATASET, file_imglist_train)
    file_imglist_val = os.path.join(ROOT_DATASET, file_imglist_val)
    file_categories = os.path.join(ROOT_DATASET, file_categories)
    
    with open(file_categories) as f:
        lines = f.readlines()
    categories = [item.rstrip() for item in lines]
    return categories, file_imglist_train, file_imglist_val, root_data, prefix

