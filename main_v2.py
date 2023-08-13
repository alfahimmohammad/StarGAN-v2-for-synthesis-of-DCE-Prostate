import os
import argparse

from munch import Munch
from torch.backends import cudnn
import torch

from dataset_prostate import *
# from dataset import get_test_loader
from core.solver_new import Solver


def str2bool(v):
    return v.lower() in ('true')


def subdirs(dname):
    return [d for d in os.listdir(dname)
            if os.path.isdir(os.path.join(dname, d))]

def create_data_loaders(args):
        
    acc_factors = args.acceleration_factor#.split(',')
    mask_types = args.mask_type#.split(',')
    dataset_type_1 = args.dataset_type_1#.split(',')  
    dataset_type_2 = args.dataset_type_2
    
    loader = torch.utils.data.DataLoader(SliceDataProstate(args.train_path,mode='train'),batch_size=args.batch_size,shuffle=True)
    
#     ref_loader = torch.utils.data.DataLoader(SliceDataref(args.train_path,acc_factors,dataset_type_2,mask_types,'train'),batch_size=1,shuffle=False)
    
    test_loader = torch.utils.data.DataLoader(SliceDataProstate(args.validation_path,mode='test'),batch_size=1,shuffle=False)

    return  loader, test_loader #input_loader , ref_loader 

def main(args):
    print(args)
#     cudnn.benchmark = True
    torch.manual_seed(args.seed)

    solver = Solver(args)

    if args.mode == 'train':
#         assert len(subdirs(args.train_path)) == args.num_domains
#         assert len(subdirs(args.validation_path)) == args.num_domains
#         input_loader, test_loader, ref_loader = create_data_loaders(args)
        loader, test_loader = create_data_loaders(args)

#         loaders = Munch(src=input_loader,
#                         ref = ref_loader,
#                         val= test_loader)
        loaders = Munch(data=loader,
                        val= test_loader)
        solver.train(loaders)
    elif args.mode == 'sample':
#         assert len(subdirs(args.src_dir)) == args.num_domains
#         assert len(subdirs(args.ref_dir)) == args.num_domains
#         input_loader, test_loader, ref_loader = create_data_loaders(args)
        loader, test_loader = create_data_loaders(args)
#         loaders = Munch(src=input_loader,
#                         ref = ref_loader)
        loaders = Munch(data=test_loader)
        solver.sample(loaders)
    
    elif args.mode == 'eval':
        solver.evaluate()
    elif args.mode == 'align':
        from core.wing import align_faces
        align_faces(args, args.inp_dir, args.out_dir)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image resolution')
    parser.add_argument('--num_domains', type=int, default=3,
                        help='Number of domains')
    parser.add_argument('--latent_dim', type=int, default=16,
                        help='Latent vector dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension of mapping network')
    parser.add_argument('--style_dim', type=int, default=64,
                        help='Style code dimension')

    # weight for objective functions
    parser.add_argument('--lambda_reg', type=float, default=1,
                        help='Weight for R1 regularization')
    parser.add_argument('--lambda_cyc', type=float, default=1,
                        help='Weight for cyclic consistency loss')
    parser.add_argument('--lambda_sty', type=float, default=1,
                        help='Weight for style reconstruction loss')
    parser.add_argument('--lambda_ds', type=float, default=1,
                        help='Weight for diversity sensitive loss')
    parser.add_argument('--ds_iter', type=int, default=100000,
                        help='Number of iterations to optimize diversity sensitive loss')
    parser.add_argument('--w_hpf', type=float, default=1,
                        help='weight for high-pass filtering')

    # training arguments
    parser.add_argument('--randcrop_prob', type=float, default=0.5,
                        help='Probabilty of using random-resized cropping')
    parser.add_argument('--total_iters', type=int, default=120000,
                        help='Number of total iterations')
    parser.add_argument('--resume_iter', type=int, default=0,
                        help='Iterations to resume training/testing')
    parser.add_argument('--batch_size', type=int, default=4, 
                        help='Batch size for training') #8
    parser.add_argument('--val_batch_size', type=int, default=1,
                        help='Batch size for validation') #32
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for D, E and G')
    parser.add_argument('--f_lr', type=float, default=1e-6,
                        help='Learning rate for F')
    parser.add_argument('--beta1', type=float, default=0.0,
                        help='Decay rate for 1st moment of Adam')
    parser.add_argument('--beta2', type=float, default=0.99,
                        help='Decay rate for 2nd moment of Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--num_outs_per_domain', type=int, default=1,
                        help='Number of generated images per domain during sampling')
    # misc
#     parser.add_argument('--mode', type=str, required=True,
#                         choices=['train', 'sample', 'eval', 'align'],

#                         help='This argument is used in solver')
    parser.add_argument('--mode',type=str, default = 'train')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers used in DataLoader')
    parser.add_argument('--seed', type=int, default=777,
                        help='Seed for random number generator')

    # directory for training
    parser.add_argument('--train_path',type=str,default='/media/Data16T/MRI/datasets/ProstateX_dce_h5')
    parser.add_argument('--validation_path',type=str,default='/media/Data16T/MRI/datasets/ProstateX_dce_h5')
    parser.add_argument('--acceleration_factor',type=str,default=['4x'])
    parser.add_argument('--dataset_type_1',type=str,default=['DCE_01','DCE_02','DCE_03'])
    parser.add_argument('--dataset_type_2',type=str,default=['DCE_01','DCE_02','DCE_03'])
    parser.add_argument('--mask_type',type=str,default=['cartesian'])
    
    
    parser.add_argument('--sample_dir', type=str, default='exp_Dce1Dce2-Dce2Dce3_reference/sample_dir',
                        help='Directory for saving generated images')
    parser.add_argument('--checkpoint_dir', type=str, default='exp_Dce1Dce2-Dce2Dce3_reference/checkpoint_dir',
                        help='Directory for saving network checkpoints')

    # directory for calculating metrics
    parser.add_argument('--eval_dir', type=str, default='exp_Dce1Dce2-Dce2Dce3_reference/eval_dir',
                        help='Directory for saving metrics, i.e., FID and LPIPS')

    # directory for testing
    parser.add_argument('--result_dir', type=str, default='exp_Dce1Dce2-Dce2Dce3_reference/result_dir',
                        help='Directory for saving generated images and videos')
#     parser.add_argument('--src_dir', type=str, default='assets/representative/celeba_hq/src',
#                         help='Directory containing input source images')
#     parser.add_argument('--ref_dir', type=str, default='assets/representative/celeba_hq/ref',
#                         help='Directory containing input reference images')
    parser.add_argument('--inp_dir', type=str, default='exp_Dce1Dce2-Dce2Dce3_reference/input_dir',
                        help='input directory when aligning faces')
    parser.add_argument('--out_dir', type=str, default='exp_Dce1Dce2-Dce2Dce3_reference/output_dir',
                        help='output directory when aligning faces')
    
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Which device to train on. Set to "cuda" to use the GPU')

    # face alignment
    parser.add_argument('--wing_path', type=str, default=None)
    parser.add_argument('--lm_path', type=str, default=None)

    # step size
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--sample_every', type=int, default=2500)#2500
    parser.add_argument('--save_every', type=int, default=5000) 
#     parser.add_argument('--save_every', type=int, default=500) 
    parser.add_argument('--eval_every', type=int, default=5000)#10000
#     parser.add_argument('--eval_every', type=int, default=500)

    args = parser.parse_args()
    main(args)
