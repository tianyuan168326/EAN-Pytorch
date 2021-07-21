import torch.utils.data as data
import torch

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import random

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class VideoDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments, image_tmpl, new_length = 1, transform=None, 
                 random_shift=True, test_mode=False,dataset="",
                 multi_clip_test = False,
                 dense_sample=False,
                 num_clips=1,number_id = None):

        self.root_path = root_path
        self.list_file = list_file
        self.new_length = new_length
        self.num_segments = num_segments
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.multi_clip_test = multi_clip_test
        self.dataset = dataset
        self.num_clips = num_clips
        self.dense_sample = dense_sample  # using dense sample as I3D
        if self.dense_sample:
            print('=> Using dense sample for the dataset...')
        self.number_id = number_id
        self._parse_list()

    def _load_image(self, directory, idx):
        try:
            return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]
        except Exception:
            print(('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx))))
            return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')]
    def _parse_list(self):
        # check the frame number is large >3:
        # usualy it is [video_id, num_frames, class_idx]
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        tmp = [item for item in tmp if int(item[1])>=3]
        if self.number_id:
            tmp = [item for item in tmp if int(item[0]) == self.number_id]
        self.video_list = [VideoRecord(item) for item in tmp]
        print(('video number:%d'%(len(self.video_list))))

    # def _sample_indices(self, record):
    #     """

    #     :param record: VideoRecord
    #     :return: list
    #     """
    #     if self.dense_sample:  # i3d dense sample
    #         sample_pos = max(1, 1 + record.num_frames - 64)
    #         t_stride = 64 // self.num_segments
    #         start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
    #         offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
    #         return np.array(offsets) + 1
    #     ###### TSN style
    #     average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
    #     # print(average_duration,record.num_frames)
    #     offsets = []
    #     if average_duration > 0:
    #         offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
    #     elif record.num_frames > self.num_segments:
    #         if((len(video_list) - self.new_length + 1) >= self.num_segments):
    #             offsets += list(np.sort(randint(len(video_list) - self.new_length + 1, size=self.num_segments)))
    #         else:
    #             offsets += list(np.sort(randint(len(video_list) - 5 + 1, size=self.num_segments)))
    #     else:
    #         # offsets += list(np.zeros((self.num_segments,)))
    #         offsets = list(range(record.num_frames))
    #         # offsets = np.array(offsets)
    #         offsets_padding = np.zeros((self.num_segments - record.num_frames,)).tolist()
    #         offsets = offsets_padding+offsets
    #         offsets = np.array(offsets)
    #     print(offsets)
    #     return offsets + 1
    def _sample_indices(self, record):
        """
        :param record: VideoRecord
        :return: list
        """
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:  # normal sample
            average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                                  size=self.num_segments)
            elif record.num_frames > self.num_segments:
                offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1
    def _get_val_indices(self, record):
        # print("_get_val_indices")
        # exit()
        """Sampling for validation set
        Sample the middle frame from each video segment
        """
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        ###TSN style
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        elif record.num_frames > self.num_segments:
            offsets = list(range(self.num_segments))
            # offsets_padding = np.zeros((self.num_segments - record.num_frames,)).tolist()
            # offsets = offsets_padding+offsets
            offsets = np.array(offsets)
        else:
            offsets = list(range(record.num_frames))
            offsets_padding = np.zeros((self.num_segments - record.num_frames,)).tolist()
            offsets = offsets_padding+offsets
            offsets = np.array(offsets)
        return offsets + 1
    
    def _get_k400_train_indices(self, record):
        interval = 8
        # if record.num_frames > self.num_segments*interval:
        #     begin_index = random.randint(0,record.num_frames-self.num_segments*interval)
        #     offsets = np.array([int(begin_index + interval * x) for x in range(self.num_segments)])
        # else:
        #     offsets = np.zeros((self.num_segments,))
        # # print("offsets",offsets)
        # return offsets + 1
        sample_pos = max(1, 1 + record.num_frames - 64)
        t_stride = 64 // self.num_segments
        start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
        offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
        # print(offsets)
        return np.array(offsets) + 1
    def _get_k400_val_indices(self, record):
        # interval = 8
        # if record.num_frames > self.num_segments*interval:
        #     begin_index = (record.num_frames-self.num_segments*interval)//2
        #     offsets = np.array([int(begin_index + interval * x) for x in range(self.num_segments)])
        # else:
        #     offsets = np.zeros((self.num_segments,))
        # return offsets + 1
        sample_pos = max(1, 1 + record.num_frames - 64)
        t_stride = 64 // self.num_segments
        start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
        offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
        return np.array(offsets) + 1

    def _get_test_indices(self, record):
        if self.dense_sample:
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        ### TSN
        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
        if self.num_clips == 1:
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)]) + 1
            # if record.num_frames > self.num_segments + self.new_length - 1:
            #     tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            #     offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            # elif record.num_frames > self.num_segments:
            #     offsets = list(range(self.num_segments))
            #     # offsets_padding = np.zeros((self.num_segments - record.num_frames,)).tolist()
            #     # offsets = offsets_padding+offsets
            #     offsets = np.array(offsets)
            # else:
            #     offsets = list(range(record.num_frames))
            #     offsets_padding = np.zeros((self.num_segments - record.num_frames,)).tolist()
            #     offsets = offsets_padding+offsets
            #     offsets = np.array(offsets)
            # offsets = np.array(offsets) + 1

        elif self.num_clips == 2:
            offsets = [np.array([int(tick * x) for x in range(self.num_segments)])+1,
                           np.array([int(tick * x + tick / 2.0) for x in range(self.num_segments)]) + 1]
        elif self.num_clips == 3:
            offsets = [np.array([int(tick * x) for x in range(self.num_segments)])+1,
                           np.array([int(tick * x + tick*1.0 / 3.0) for x in range(self.num_segments)]) + 1,
                           np.array([int(tick * x + tick*2.0 / 3.0) for x in range(self.num_segments)]) + 1]
        elif self.num_clips == 5:
            offsets = [np.array([int(tick * x) for x in range(self.num_segments)])+1,
                           np.array([int(tick * x + tick*1.0 / 5.0) for x in range(self.num_segments)]) + 1,
                           np.array([int(tick * x + tick*2.0 / 5.0) for x in range(self.num_segments)]) + 1,
                           np.array([int(tick * x + tick*3.0 / 5.0) for x in range(self.num_segments)]) + 1,
                           np.array([int(tick * x + tick*4.0 / 5.0) for x in range(self.num_segments)]) + 1,
                           ]
        return offsets

    def __getitem__(self, index):
        record = self.video_list[index]
        # check this is a legit video folder
        while not os.path.exists(os.path.join(self.root_path, record.path, self.image_tmpl.format(1))):
            print("not exist",(os.path.join(self.root_path, record.path, self.image_tmpl.format(1))))
            exit()
            index = np.random.randint(len(self.video_list))
            record = self.video_list[index]
        # if self.dataset == "k400":
        if False:
            segment_indices = self._get_k400_train_indices(record) if self.random_shift else self._get_k400_val_indices(record)
        else:
            if not self.test_mode:
                segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
            else: ### test set
                if self.multi_clip_test:
                    segment_indices = self._sample_indices(record)
                else:
                    segment_indices = self._get_test_indices(record)
        return self.get(record, segment_indices)

    def get(self, record, indices):
        if self.num_clips > 1:
            process_data_final = []
            for k in range(self.num_clips):
                images = list()
                for seg_ind in indices[k]:
                    p = int(seg_ind)
                    for i in range(self.new_length):
                        seg_imgs = self._load_image(record.path, p)
                        images.extend(seg_imgs)
                        if p < record.num_frames:
                            p += 1

                process_data, label = self.transform((images, record.label))
                process_data_final.append(process_data)
            process_data_final = torch.stack(process_data_final, 0)#
            return process_data_final, label

        else:
            images = list()
            for seg_ind in indices:
                p = int(seg_ind)
                for i in range(self.new_length):
                    seg_imgs = self._load_image(record.path, p)
                    images.extend(seg_imgs)
                    if p < record.num_frames:
                        p += 1

            process_data, label = self.transform((images, record.label))
            return process_data, label

    def __len__(self):
        # print("len(self.video_list)",len(self.video_list))
        # exit()
        return len(self.video_list)
