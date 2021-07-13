import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of action recognition models")
parser.add_argument('--dataset', type=str,
                   default = 'somethingv1')
parser.add_argument('--root_path', type = str, default = '../',
                    help = 'root path to video dataset folders')
parser.add_argument('--store_name', type=str, default="")
parser.add_argument('--local_rank', type=int, default="0")
parser.add_argument('--augname', type=str, default="")


# ========================= Model Configs ==========================
parser.add_argument('--type', type=str, default="GST",
# choices=['GST','R3D','S3D','STCeption','STCeption_Reg'],
                    help = 'type of temporal models, currently support GST,Res3D and S3D')
parser.add_argument('--arch', type=str, default="resnet50",
                    help = 'backbone networks, currently only support resnet')
parser.add_argument('--agg_type', type=str, default="TSN",
                    help = 'video classifier,default TSN')
                    
parser.add_argument('--num_segments', type=int, default=8)
parser.add_argument('--fp', type=int, default=32, help = 'numerical precision')
parser.add_argument('--alpha', type=int, default=4, help = 'spatial temporal split for output channels')
parser.add_argument('--beta', type=int, default=2, choices=[1,2], help = 'channel splits for input channels, 1 for GST-Large and 2 for GST')

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=24, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[50, 60], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--dropout', '--dp', default=0.3, type=float,
                    metavar='dp', help='dropout ratio')
parser.add_argument('--warmup', default=False, action="store_true", help='if using warm up')
parser.add_argument('--warmup_epoch', default=10, type=int,
                    metavar='N', help='warmup_epoch (default: 10)')
parser.add_argument('--nesterov', default=False, action="store_true", help='if using nesterov')

#========================= Optimizer Configs ==========================
parser.add_argument('--optim',type=str,  required=False,
                    help = 'sgd or adam')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=3e-4, type=float,   
                    metavar='W', help='weight decay (default: 3e-4)')
parser.add_argument('--clip-gradient', '--gd', default=20, type=float,
                    metavar='W', help='gradient norm clipping (default: 20)')

# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')

parser.add_argument('--eval-freq', '-ef', default=1, type=int,
                    metavar='N', help='evaluation frequency (default: 1)')

# ========================= Input Data Configs ==========================
parser.add_argument('--input_size', '-i', default=112, type=int,
                    metavar='N', help='input size default 112')

# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=5, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--resume_student', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--resume_teacher', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--resume_precode', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--resume_tpre', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
                    
parser.add_argument('--resume_action', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--resume_ds_model', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--distill_resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
                    
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--checkpoint_dir',type=str,  required=False,
                    help = 'folder to restore checkpoint and training log')

#==============================For multi crop test================
parser.add_argument('-ts', '--test_segments', default=-1, type=int, metavar='N')
parser.add_argument('-tc', '--test_crops', default=1, type=int, metavar='N')
parser.add_argument('-nc', '--num_clips', default=1, type=int, metavar='N')
parser.add_argument('--dense_sample', default=False, action="store_true", help='use dense sample for video dataset')
parser.add_argument('--multi_clip_test', default=False, action="store_true", help='multi clip test')
parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--shift', default=False, action="store_true", help='use shift for models')
parser.add_argument('--ft_action', default=False, action="store_true", help='ft_action')