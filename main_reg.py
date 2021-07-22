import argparse
import os
import sys
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import time
import shutil
import torch
import torch.nn as nn
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm_ as clip_grad_norm

from dataset import VideoDataSet
from transforms import *
from opts import parser
import datasets_video

dist.init_process_group(backend='nccl')  


best_prec1 = 0

def main():
	global args, best_prec1
	args = parser.parse_args()
	import importlib
	lib = importlib.import_module(args.agg_type)
	TemporalModel = getattr(lib,"TemporalModel")
	check_rootfolders()

	categories, train_list, val_list, root_path, prefix = datasets_video.return_dataset(args.dataset,args.root_path)
	num_class = len(categories)


	global store_name 
	store_name = '_'.join([args.type, args.dataset, args.arch, 'segment%d'% args.num_segments, args.store_name])
	print(('storing name: ' + store_name))
	os.environ['ds'] = args.dataset
	if args.dataset == 'somethingv1' or args.dataset == 'somethingv2':
		# label transformation for left/right categories
		# please refer to labels.json file in sometingv2 for detail.
		target_transforms = {86:87,87:86,93:94,94:93,166:167,167:166}
	else:
		target_transforms = None

	ft_action = args.ft_action

	model = TemporalModel(ft_action,args.input_size,num_class, args.num_segments, model = args.type, backbone=args.arch, 
						alpha = args.alpha, beta = args.beta, 
						dropout = args.dropout, target_transforms = target_transforms)

	crop_size = model.crop_size
	scale_size = model.scale_size
	input_mean = model.input_mean
	input_std = model.input_std
	policies = get_optim_policies(model,ft_action)
	train_augmentation = model.get_augmentation()

	local_rank = args.local_rank
	torch.cuda.set_device(local_rank)
	if torch.cuda.is_available():
		model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[local_rank],\
			 broadcast_buffers=True, find_unused_parameters=True)


	

	if args.resume:
		if os.path.isfile(args.resume):
			print(("=> loading checkpoint '{}'".format(args.resume)))
			checkpoint = torch.load(args.resume,map_location='cpu')
			
			args.start_epoch = checkpoint['epoch']
			best_prec1 = checkpoint['best_prec1']
			ckeck = checkpoint['state_dict']
			ckeck_copy = ckeck.copy()
			ks  =ckeck.keys()
			# exit()
			model.load_state_dict(ckeck_copy,strict=False)
			print(("=> loaded checkpoint '{}' (epoch {})"
				  	.format(args.evaluate, checkpoint['epoch'])))
		else:
			print(("=> no checkpoint found at '{}'".format(args.resume)))

	
	cudnn.benchmark = True

	# Data loading code
	# normalize = GroupNormalize(input_mean, input_std)
	train_ds = VideoDataSet(root_path, train_list, num_segments=args.num_segments,
				   image_tmpl=prefix,new_length=5,
				   transform=torchvision.transforms.Compose([
					   train_augmentation,
					   Stack(roll=(args.arch in ['BNInception','InceptionV3'])),
					   ToTorchFormatTensor(div=(args.arch not in ['BNInception','InceptionV3'])),
					#    normalize,
				   ]))
	
	train_loader = torch.utils.data.DataLoader(
		train_ds,
		batch_size=args.batch_size, drop_last=True,
		num_workers=args.workers, pin_memory=True,sampler=DistributedSampler(train_ds,shuffle=True))
	test_ds = VideoDataSet(root_path, val_list, num_segments=args.num_segments,
				   image_tmpl=prefix,new_length=5,
				   random_shift=False,
				   transform=torchvision.transforms.Compose([
					   GroupScale(int(scale_size)),
					   GroupCenterCrop(crop_size),
					   Stack(roll=(args.arch in ['BNInception','InceptionV3'])),
					   ToTorchFormatTensor(div=(args.arch not in ['BNInception','InceptionV3'])),
					   normalize,
				   ]))
	val_loader = torch.utils.data.DataLoader(
		test_ds,
		batch_size=args.batch_size, shuffle=False,
		num_workers=args.workers, pin_memory=True,drop_last=True,sampler=DistributedSampler(test_ds))

	# define loss function (criterion) and optimizer
	criterion = torch.nn.CrossEntropyLoss()
	
	for group in policies:
		print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
			group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))
	# exit()

	if args.optim == "sgd":
		optimizer = torch.optim.SGD(policies,
								args.lr,
								momentum=args.momentum,
								weight_decay=args.weight_decay)
	if args.optim == "adam":
		optimizer = torch.optim.Adam(policies,
								args.lr,
								weight_decay=args.weight_decay)

	log_training = open(os.path.join(args.checkpoint_dir,'log', '%s.csv' % store_name), 'w')
	if args.evaluate:
		prec1 = validate(val_loader, model, criterion, log_training)
		is_best = prec1 > best_prec1
		best_prec1 = max(prec1, best_prec1)
		save_checkpoint({
				'epoch': args.start_epoch,
				'arch': args.arch,
				'state_dict': model.state_dict(),
				'best_prec1': best_prec1,
			}, is_best)


		return

	
	
	for epoch in range(args.start_epoch, args.epochs):
		# adjust learning rate
		
		adjust_learning_rate(optimizer, epoch, args.lr_steps)
		
		# train for one epoch
		train(train_loader, model, criterion, optimizer, epoch, log_training,ft_action)
		if dist.get_rank() == 0:
			save_checkpoint({
					'epoch': epoch + 1,
					'arch': args.arch,
					'state_dict': model.state_dict(),
					'best_prec1': 0,
				}, False)
		# evaluate on validation set
		if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
			prec1 = validate(val_loader, model, criterion, (epoch + 1) * len(train_loader), log_training)
			prec1 = prec1.cpu()
			if not type(best_prec1) == type(1): 
				best_prec1 = best_prec1.cpu()
			if dist.get_rank() == 0:
				# remember best prec@1 and save checkpoint
				is_best = prec1 > best_prec1
				best_prec1 = max(prec1, best_prec1)
				save_checkpoint({
					'epoch': epoch + 1,
					'arch': args.arch,
					'state_dict': model.state_dict(),
					'best_prec1': best_prec1,
				}, is_best)
EPOCH  =5
def reduce_tensor(tensor):
	rt = tensor.clone()
	dist.all_reduce(rt, op=dist.reduce_op.SUM)
	rt /= dist.get_world_size()
	return rt
def train(train_loader, model, criterion, optimizer, epoch, log,ft_action):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	reg_losses = AverageMeter()
	rep_losses1 = AverageMeter()
	rep_losses2 = AverageMeter()
	rep_losses3 = AverageMeter()
	reg_loss_fuses = AverageMeter()
	prob_pens = AverageMeter()

	top1 = AverageMeter()
	top5 = AverageMeter()

	# switch to train mode
	model.train()

	end = time.time()
	train_loader.sampler.set_epoch(epoch)
	for i, (input, target) in enumerate(train_loader):
		# measure data loading time
		data_time.update(time.time() - end)
		input = input.cuda(non_blocking = True)
		target = target.cuda(non_blocking=True)
		
		rec_output2,reg_loss = model(input)
		rec_output_avg = rec_output2
		loss_action_final2 = criterion(rec_output2, target)
		rep_losses1.update(loss_action_final2.item(), input.size(0))
		loss_action_final = loss_action_final2
		# reg_losses.update(reg_loss.item(), input.size(0))
		loss_sum = (loss_action_final)
		# loss_sum = loss_sum.mean()
		losses.update(loss_sum.item(), input.size(0))
		prec1, prec5 = accuracy(rec_output_avg, target, topk=(1,5))

		prec1 = prec1[0]
		prec5 = prec5[0]
		
		# torch.distributed.barrier()
		# reduced_acc1 = reduce_tensor(prec1)
		# reduced_acc5 = reduce_tensor(prec5)
		top1.update(prec1, input.size(0))
		top5.update(prec5, input.size(0))
			
		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss_sum.backward()


		if args.clip_gradient is not None:
			total_norm = clip_grad_norm(model.parameters(), args.clip_gradient)
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0 and '0' in str(loss_sum.device):

			output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
					'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
					'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					'Reg_Loss {reg_loss.val:.4f} ({reg_loss.avg:.4f})\t'
				 	'rep_losses1 {rep_losses1.val:.4f} ({rep_losses1.avg:.4f})\t'
				 	'rep_losses2 {rep_losses2.val:.4f} ({rep_losses2.avg:.4f})\t'
				 	'rep_losses3 {rep_losses3.val:.4f} ({rep_losses3.avg:.4f})\t'
				 	'prob_pens {prob_pen.val:.4f} ({prob_pen.avg:.4f})\t'
					'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
					'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
						epoch, i, len(train_loader), batch_time=batch_time,
						data_time=data_time, loss=losses, reg_loss = reg_losses,
					   rep_losses1 = rep_losses1,
					   rep_losses2 = rep_losses2,
					   rep_losses3 = rep_losses3,
					   prob_pen = prob_pens,
						top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr']))
			print(output)
			log.write(output + '\n')
			log.flush()
		
		# break

def validate(val_loader, model, criterion, iter, log = None):
	batch_time = AverageMeter()
	losses = AverageMeter()
	rep_losses1 = AverageMeter()
	rep_losses2 = AverageMeter()
	rep_losses3 = AverageMeter()
	reg_losses = AverageMeter()
	reg_loss_fuses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()
	rep_losses = AverageMeter()

	# switch to evaluate mode
	model.eval()

	end = time.time()
	with torch.no_grad():
		for i, (input, target) in enumerate(val_loader):
		
		

			input = input.cuda(non_blocking = True)
			target = target.cuda(non_blocking = True)
				# compute output
			rec_output2,reg_loss = model(input)
			rec_output_avg = rec_output2
			loss_action_final2 = criterion(rec_output2, target)
			rep_losses1.update(loss_action_final2.item(), input.size(0))
			loss_action_final = loss_action_final2
			# reg_losses.update(reg_loss.item(), input.size(0))
			loss_sum = (loss_action_final)
			# loss_sum = loss_sum.mean()
			losses.update(loss_sum.item(), input.size(0))
			prec1, prec5 = accuracy(rec_output_avg, target, topk=(1,5))

			prec1 = prec1[0]
			prec5 = prec5[0]
			
			torch.distributed.barrier()
			reduced_acc1 = reduce_tensor(prec1)
			reduced_acc5 = reduce_tensor(prec5)
			top1.update(reduced_acc1, input.size(0))
			top5.update(reduced_acc5, input.size(0))
			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % args.print_freq == 0 and "0" in str(loss_sum.device):
				output = ('Test: [{0}/{1}]\t'
				 	 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				 	 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				 	 'reg_losses {reg_loss.val:.4f} ({reg_loss.avg:.4f})\t'
				 	 'reg_loss_fuses {reg_loss_fuse.val:.4f} ({reg_loss_fuse.avg:.4f})\t'
				 	 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
				 	 'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
				 	  i, len(val_loader), batch_time=batch_time, loss=losses,reg_loss=reg_losses,
					   reg_loss_fuse = reg_loss_fuses,
				  	 top1=top1, top5=top5))
				print(output)
				if log is not None:
					log.write(output + '\n')
					log.flush()
			# break
	output = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
		  .format(top1=top1, top5=top5, loss=losses))
	print(output)
	output_best = '\nBest Prec@1: %.3f'%(best_prec1)
	print(output_best)
	if log is not None:
		log.write(output + ' ' + output_best + '\n')
		log.flush()

	return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
	if dist.get_rank() == 0:
		torch.save(state, '%s/%s_checkpoint.pth.tar' % (os.path.join(args.checkpoint_dir,'checkpoint'), store_name))
		if is_best:
			shutil.copyfile('%s/%s_checkpoint.pth.tar' % (os.path.join(args.checkpoint_dir,'checkpoint'), store_name),
			'%s/%s_best.pth.tar' % (os.path.join(args.checkpoint_dir,'checkpoint'), store_name))
		if state['epoch'] %10 == 0:
			shutil.copyfile(
				'%s/%s_checkpoint.pth.tar' % (os.path.join(args.checkpoint_dir,'checkpoint'), store_name),
			'%s/%s_checkpoint_%d.pth.tar' % (os.path.join(args.checkpoint_dir,'checkpoint'), store_name,state['epoch'])
			)

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

def adjust_learning_rate(optimizer, epoch, lr_steps):
	"""Sets the learning rate to the initial LR decayed by 10 """
	decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
	if epoch <EPOCH :
		# decay *=0.1
		decay *=1
	lr = args.lr * decay
	decay = args.weight_decay
	print("lr",lr)
	# exit()
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr * param_group['lr_mult']
		param_group['weight_decay'] = decay * param_group['decay_mult']


def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	with torch.no_grad():
		maxk = max(topk)
		batch_size = target.size(0)
		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.reshape(1, -1).expand_as(pred))

		res = []
		for k in topk:
			correct_k = correct[:k].reshape(-1).float().sum(0,keepdim = True)
			res.append(correct_k.mul_(100.0 / batch_size))
		return res

def check_rootfolders():
	"""Create log and model folder"""
	folders_util = [args.checkpoint_dir,os.path.join(args.checkpoint_dir,'log'), os.path.join(args.checkpoint_dir,'checkpoint')]
	for folder in folders_util:
		if not os.path.exists(folder):
			print(('creating folder ' + folder))
			os.mkdir(folder)

def get_optim_policies(model,ft_action):
	# return model.parameters()
	first_conv_weight = []
	first_conv_bias = []
	scale_conv_weight = []
	scale_conv_bias = []
	normal_weight = []
	normal_bias = []
	bn = []

	conv_cnt = 0
	bn_cnt = 0
	other_p = []
	for n,m in model.named_parameters():
		if "pos_embedding" in n:
			other_p.append(m)
	for n,m in model.named_modules():
		# if not ("new_fc" in n or "sparse_transformer" in n):
		# 	continue
		# if "pos_embedding" in n:
		if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv3d):
			ps = list(m.parameters())
			conv_cnt += 1
			if conv_cnt <= 3:
				first_conv_weight.append(ps[0])
				if len(ps) == 2:
					first_conv_bias.append(ps[1])
			else:
				normal_weight.append(ps[0])
				if len(ps) == 2:
					normal_bias.append(ps[1])
		
		elif isinstance(m, torch.nn.Linear):
			ps = list(m.parameters())
			normal_weight.append(ps[0])
			if len(ps) == 2:
				normal_bias.append(ps[1])

		elif isinstance(m,nn.BatchNorm3d) or isinstance(m,nn.BatchNorm2d) or isinstance(m,nn.BatchNorm1d) :
			bn_cnt += 1
			bn.extend(list(m.parameters()))
	
		elif len(m._modules) == 0:
			if len(list(m.parameters())) > 0:
				raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))
	return [
			{'params': first_conv_weight, 'lr_mult':  1, 'decay_mult': 1,
			 'name': "first_conv_weight"},
			{'params': first_conv_bias, 'lr_mult':  2, 'decay_mult': 0,
			 'name': "first_conv_bias"},
			{'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
			 'name': "normal_weight"},
			{'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
			 'name': "normal_bias"},
			 {'params': scale_conv_weight, 'lr_mult': 10, 'decay_mult': 1,
			 'name': "scale_conv_weight"},
			{'params': scale_conv_bias, 'lr_mult': 20, 'decay_mult': 0,
			 'name': "scale_conv_bias"},
			{'params': bn, 'lr_mult': 1, 'decay_mult': 0,
			 'name': "BN scale/shift"},
			{'params': other_p, 'lr_mult': 1, 'decay_mult': 1,
			 'name': "other_p"},
		]



if __name__ == '__main__':
	main()
