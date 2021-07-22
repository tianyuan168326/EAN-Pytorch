import time
import sys
import os
import numpy as np
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix
from dataset import VideoDataSet
from transforms import *
import datasets_video
import pdb
from torch.nn import functional as F
import sys


from opts import parser

args = parser.parse_args()
os.environ['ds'] = args.dataset
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
         correct_k = correct[:k].view(-1).float().sum(0)
         res.append(correct_k.mul_(100.0 / batch_size))
    return res

categories, train_list, val_list, root_path, prefix = datasets_video.return_dataset(args.dataset,args.root_path)

if args.dataset == 'somethingv1':
    num_class = 174
    args.rgb_prefix = ''
    rgb_read_format = "{:05d}.jpg"
elif args.dataset == 'somethingv2':
    num_class = 174
    args.rgb_prefix = ''
    rgb_read_format = "{:05d}.jpg"
elif args.dataset == 'k400':
    num_class = 400
    args.rgb_prefix = 'image'
    rgb_read_format = '{:05d}.jpg'
elif args.dataset == 'diving48':
    num_class = 48
    args.rgb_prefix = 'frames'
    rgb_read_format = "{:05d}.jpg"
else:
    raise ValueError('Unknown dataset '+args.dataset)


import importlib
lib = importlib.import_module(args.agg_type)
TemporalModel = getattr(lib,"TemporalModel")

if args.dataset == 'somethingv1' or args.dataset == 'somethingv2':
		# label transformation for left/right categories
		# please refer to labels.json file in sometingv2 for detail.
		target_transforms = {86:87,87:86,93:94,94:93,166:167,167:166}
else:
    target_transforms = None

net_o = TemporalModel(0,args.input_size,num_class, args.num_segments, model = args.type, backbone=args.arch, 
                    alpha = args.alpha, beta = args.beta, 
                    dropout = args.dropout, target_transforms = target_transforms)

net = torch.nn.DataParallel(net_o.cuda())
if args.resume:
    if os.path.isfile(args.resume):
        print(("=> loading checkpoint '{}'".format(args.resume)))
        checkpoint = torch.load(args.resume,map_location='cpu')
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        print("best_prec1 in model",best_prec1)
        ckeck = checkpoint['state_dict']
        ckeck_copy = ckeck.copy()
        ks  =ckeck.keys()

        net.load_state_dict(ckeck_copy, strict=True)
        print(("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch'])))
    else:
        print(("=> no checkpoint found at '{}'".format(args.resume)))

# checkpoint = torch.load(args.resume)
# print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

# base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
# net.load_state_dict(base_dict, strict=False)

# if args.test_crops == 1:
#     cropping = torchvision.transforms.Compose([
#         GroupScale((net.scale_size,net.scale_size)),
#         GroupCenterCrop(net.input_size),
#     ])
# elif args.test_crops == 10:
#     cropping = torchvision.transforms.Compose([
#         GroupOverSample(net.input_size, net.scale_size)
#     ])
# elif args.test_crops == 5:
#     cropping = torchvision.transforms.Compose([
#         GroupFiveCrops(net.input_size, net.scale_size)
#     ])
# else:
#     raise ValueError("Only 1 and 10 crops are supported while we got {}".format(args.test_crops))
crop_size = net_o.crop_size
scale_size = net_o.scale_size
input_mean = net_o.input_mean
input_std = net_o.input_std
new_length = 5
from transforms import *
normalize = GroupNormalize(input_mean, input_std)
test_ds = VideoDataSet(root_path, val_list, num_segments=args.num_segments,
				   image_tmpl=prefix,new_length=new_length,
				   random_shift=False,
                   test_mode = True,
				   transform=torchvision.transforms.Compose([
					GroupScale(int(scale_size)),
					   GroupCenterCrop(crop_size),
					   Stack(roll=(args.arch in ['BNInception','InceptionV3'])),
					   ToTorchFormatTensor(div=(args.arch not in ['BNInception','InceptionV3'])),
					   normalize,
				   ]),num_clips = args.num_clips)

data_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=1, shuffle=False,
        num_workers=8, pin_memory=False)

devices = [0]
net.eval()
data_gen = enumerate(data_loader)
total_num = len(data_loader.dataset)
output = []
output_scores = []
def eval_video(video_data):
    i, data, label = video_data
    num_crop = args.test_crops

    if args.num_clips > 1:
        input_var = data.view(-1, 3, data.size(3), data.size(4))
    else:
        input_var = data.view(-1, 3, data.size(2), data.size(3))

    rst,_ = net(input_var)

    rst = rst.data.cpu().numpy().copy()
    rst = rst.reshape(-1, 1, num_class)
    return i, rst, label[0]


max_num = args.max_num if args.max_num > 0 else len(data_loader.dataset)
# max_num = 10

top1 = AverageMeter()
top5 = AverageMeter()

with torch.no_grad():
    for i, (data, label) in data_gen:
        if i >= max_num:
            break
        rst = eval_video((i, data, label))
        output.append(rst[1:])
        output_scores.append(np.mean(rst[1], axis=0))
        prec1, prec5 = accuracy(torch.from_numpy(np.mean(rst[1], axis=0)), label, topk=(1, 5))
        top1.update(prec1, 1)
        top5.update(prec5, 1)
        print('video {} done, total {}/{},  moving Prec@1 {:.3f} Prec@5 {:.3f}'.format(i, i+1,
                                                                        total_num,
                                                                         top1.avg, top5.avg))
   

video_pred = [np.argmax(np.mean(x[0], axis=0)) for x in output]

video_labels = [x[1] for x in output]


cf = confusion_matrix(video_labels, video_pred).astype(float)

cls_cnt = cf.sum(axis=1)
cls_hit = np.diag(cf)

cls_acc = cls_hit / cls_cnt

print('-----Evaluation of {} is finished------'.format(args.resume))
print('Overall Prec@1 {:.02f}% Prec@5 {:.02f}%'.format(top1.avg, top5.avg))

if True:

    # order_dict = {e:i for i, e in enumerate(sorted(name_list))}
    reorder_output = [None] * len(output)
    reorder_label = [None] * len(output)
    reorder_pred = [None] * len(output)
    output_csv = []
    for i in range(len(output)):
        reorder_output[i] = output_scores[i]
        reorder_label[i] = video_labels[i]
        reorder_pred[i] = video_pred[i]
    save_name = args.agg_type + '_clips_' + str(args.num_clips) + '.npz'
    print(save_name)
    np.savez(save_name, scores=reorder_output, labels=reorder_label, predictions=reorder_pred, cf=cf)