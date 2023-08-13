import pathlib
import random
import numpy as np
import h5py
from torch.utils.data import Dataset
import torch
from skimage import feature
import os 
import glob
import pathlib
import random
import numpy as np
import h5py
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch
from skimage import feature
import os 
from munch import Munch

class InputFetcher:
    def __init__(self, loader, loader_ref=None, latent_dim=16, mode=''):
        self.loader = loader
        self.loader_ref = loader_ref
        self.latent_dim = latent_dim
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mode = mode

    def _fetch_inputs(self):
        try:
            x, y = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, y = next(self.iter)
        return x, y

    def _fetch_refs(self):
        try:
            x, y = next(self.iter_ref)
            x2 = x 
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader_ref)
            x, y = next(self.iter_ref)
            x2 = x
        return x, x2 , y

    def __next__(self):
        x, y = self._fetch_inputs()
        if self.mode == 'train':
            x_ref, x_ref2, y_ref = self._fetch_refs()
            z_trg = torch.randn(x.size(0), self.latent_dim)
            z_trg2 = z_trg
            inputs = Munch(x_src=x, y_src=y, y_ref=y_ref,
                           x_ref=x_ref, x_ref2=x_ref2,
                           z_trg=z_trg, z_trg2=z_trg2)
        elif self.mode == 'val':
            x_ref, y_ref = self._fetch_inputs()
            inputs = Munch(x_src=x, y_src=y,
                           x_ref=x_ref, y_ref=y_ref)
        elif self.mode == 'test':
            inputs = Munch(x=x, y=y)
        else:
            raise NotImplementedError

        return Munch({k: v.to(self.device)
                      for k, v in inputs.items()})

#####################################################################################

class InputFetcher_v2:
    def __init__(self, loader, latent_dim=16, mode=''):
        self.loader = loader
        self.latent_dim = latent_dim
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mode = mode
    
    def _fetch_imgs(self):
        if self.mode == 'train':
            try:
                x_src, y_src, x_ref, y_ref = next(self.iter)
            except (AttributeError, StopIteration):
                self.iter = iter(self.loader)
                x_src, y_src, x_ref, y_ref = next(self.iter)
                
            return x_src, y_src, x_ref, y_ref
        else:
            try:
                x_src, y_src, x_ref, y_ref, src_domain, trg_domain = next(self.iter)
            except (AttributeError, StopIteration):
                self.iter = iter(self.loader)
                x_src, y_src, x_ref, y_ref, src_domain, trg_domain = next(self.iter)
                
            return x_src, y_src, x_ref, y_ref, src_domain, trg_domain 

    def __next__(self):
        
        if self.mode == 'train':
            x_src, y_src, x_ref, y_ref = self._fetch_imgs()
            z_trg = torch.randn(x_src.size(0), self.latent_dim)
            z_trg2 = z_trg
            inputs = Munch(x_src=x_src, y_src=y_src, y_ref=y_ref,
                           x_ref=x_ref, x_ref2=x_ref,
                           z_trg=z_trg, z_trg2=z_trg2)
        elif self.mode == 'val':
            x_src, y_src, x_ref, y_ref, src_domain, trg_domain = self._fetch_imgs()
            z_trg = torch.randn(x_src.size(0), self.latent_dim)
            inputs = Munch(x_src=x_src, y_src=y_src,
                           x_ref=x_ref, y_ref=y_ref,
                           z_trg=z_trg, src_domain=src_domain[0], trg_domain=trg_domain[0])
        elif self.mode == 'test':
            x_src, y_src, x_ref, y_ref = self._fetch_imgs()
            inputs = Munch(x=x_src, y=y_src)
        else:
            raise NotImplementedError

#         return Munch({k: v.to(self.device)
#                       if type(v) != str else k: v for k, v in inputs.items()})
        dic = {}
        for k,v in inputs.items():
            if type(v) != str:
                dic[k]= v.to(self.device)
            else:
                dic[k]= v
                
        return Munch(dic)

#####################################################################################

class SliceData(Dataset):
    
    def __init__(self,root,acc_factors,dataset_types,mask_types,train_or_valid,mode='train'):
        self.examples = []
        escapes = ''.join([chr(char) for char in range(1, 32)])
        translator = str.maketrans('', '', escapes) 
        self.filenames = []
        self.examples = []
        self.mode = mode
        
        dataroot = os.path.join(root,dataset_types[0]).translate(translator)
        for mask_type in mask_types:
            newroot = os.path.join(dataroot,mask_type,train_or_valid).translate(translator)
            for acc_factor in acc_factors:
                #print("acc_factor: ", acc_factor)
                files = list(pathlib.Path(os.path.join(newroot,'acc_{}'.format(acc_factor)).translate(translator)).iterdir())
                for fname in sorted(files):
                    with h5py.File(fname,'r') as hf:
                        fsvol = hf['volfs']
                        num_slices = fsvol.shape[2]
                        #print("acc_factor: ",acc_factor)
                        #acc_factor = float(acc_factor[:-1].replace("_","."))
                        filename = str(fname).split("/")[-1]
                        self.filenames += [(filename, slice) for slice in range(num_slices)]
        
        
        for trg_domain in dataset_types:
            src_domains = [x for x in dataset_types if x != trg_domain]
            for src_domain in src_domains:
                for (filename, slice) in self.filenames:
                    trg_root = os.path.join(root,trg_domain).translate(translator)
                    src_root = os.path.join(root,src_domain).translate(translator)
                    for mask_type in mask_types:
                        trg_newroot = os.path.join(trg_root,mask_type,train_or_valid).translate(translator)
                        src_newroot = os.path.join(src_root,mask_type,train_or_valid).translate(translator)
                        for acc_factor in acc_factors:
                            #print("acc_factor: ", acc_factor)
                            trg_file = os.path.join(trg_newroot,'acc_{}'.format(acc_factor),filename).translate(translator)
                            src_file = os.path.join(src_newroot,'acc_{}'.format(acc_factor),filename).translate(translator)
                            self.examples += [(trg_file, src_file, slice, acc_factor, mask_type, trg_domain, src_domain)]
                            
                            
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        trg_file, src_file, slice, acc_factor, mask_type, trg_domain, src_domain = self.examples[i]

        if trg_domain=='mrbrain_t1':
            c_trg = torch.tensor([0])
        elif trg_domain=='mrbrain_ir':
            c_trg = torch.tensor([1])
        elif trg_domain=='mrbrain_flair':
            c_trg = torch.tensor([2])

        if src_domain=='mrbrain_t1':
            c_src = torch.tensor([0])
        elif src_domain=='mrbrain_ir':
            c_src = torch.tensor([1])
        elif src_domain=='mrbrain_flair':
            c_src = torch.tensor([2])


        with h5py.File(trg_file, 'r') as data:
            trg_img  = data['volfs'][:,:,slice].astype(np.float64)

        with h5py.File(src_file, 'r') as data:
            src_img  = data['volfs'][:,:,slice].astype(np.float64)

        if self.mode == 'train':
            return torch.from_numpy(src_img), c_src, torch.from_numpy(trg_img), c_trg
        else:
            return torch.from_numpy(src_img), c_src, torch.from_numpy(trg_img), c_trg, src_domain, trg_domain

#####################################################################################

class SliceDatainput(Dataset):

    def __init__(self, root, acc_factors,dataset_type_1,mask_types,train_or_valid): 
        self.examples = []
        escapes = ''.join([chr(char) for char in range(1, 32)])
        translator = str.maketrans('', '', escapes) 
        
#         print(f'datatype is {dataset_types},{mask_types},{train_or_valid},{acc_factors}')
        

        for dataset_type in dataset_type_1:
            dataroot = os.path.join(root,dataset_type).translate(translator)
            for mask_type in mask_types:
                newroot = os.path.join(dataroot,mask_type,train_or_valid).translate(translator)
                for acc_factor in acc_factors:
                    #print("acc_factor: ", acc_factor)
                    files = list(pathlib.Path(os.path.join(newroot,'acc_{}'.format(acc_factor)).translate(translator)).iterdir())
                    for fname in sorted(files):
                        with h5py.File(fname,'r') as hf:
                            fsvol = hf['volfs']
                            num_slices = fsvol.shape[2]
                            #print("acc_factor: ",acc_factor)
                            #acc_factor = float(acc_factor[:-1].replace("_","."))
                            self.examples += [(fname, slice, acc_factor, mask_type, dataset_type) for slice in range(num_slices)]
   

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice, acc_factor, mask_type, dataset_type = self.examples[i]
        
        if dataset_type=='mrbrain_t1':
            c = torch.tensor([0])
#             c=0
#             c = torch.tensor(np.array([1, 0, 0]))

              
        elif dataset_type=='mrbrain_ir':
            c = torch.tensor([1])
#             c=1
#             c = torch.tensor(np.array([0, 1, 0]))
        elif dataset_type=='mrbrain_flair':
            c = torch.tensor([2])
#             c=2
#             c = torch.tensor(np.array([0, 0, 1]))
#         else: c = torch.tensor(np.array([0, 0]))
            
    
        with h5py.File(fname, 'r') as data:
#             key_img = 'img_volus_{}'.format(acc_factor).translate(translator)
            source_img  = data['volfs'][:,:,slice].astype(np.float64)
            
            
            return torch.from_numpy(source_img),c

#######################################################################################
class SliceDataref(Dataset):

    def __init__(self, root, acc_factors,dataset_type_2,mask_types,train_or_valid): 
        self.examples = []
        escapes = ''.join([chr(char) for char in range(1, 32)])
        translator = str.maketrans('', '', escapes) 
        
#         print(f'datatype is {dataset_types},{mask_types},{train_or_valid},{acc_factors}')
        

        for dataset_type in dataset_type_2:
            dataroot = os.path.join(root,dataset_type).translate(translator)
            for mask_type in mask_types:
                newroot = os.path.join(dataroot,mask_type,train_or_valid).translate(translator)
                for acc_factor in acc_factors:
                    #print("acc_factor: ", acc_factor)
                    files = list(pathlib.Path(os.path.join(newroot,'acc_{}'.format(acc_factor)).translate(translator)).iterdir())
                    for fname in sorted(files):
                        with h5py.File(fname,'r') as hf:
                            fsvol = hf['volfs']
                            num_slices = fsvol.shape[2]
                            #print("acc_factor: ",acc_factor)
                            #acc_factor = float(acc_factor[:-1].replace("_","."))
                            self.examples += [(fname, slice, acc_factor, mask_type, dataset_type) for slice in range(num_slices)]
   

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice, acc_factor, mask_type, dataset_type = self.examples[i]
        
        
        if dataset_type=='mrbrain_t1':
            c = torch.tensor([0])
#             c=0
#             c = torch.tensor(np.array([1, 0, 0]))
        elif dataset_type=='mrbrain_ir':
            c = torch.tensor([1])
#             c=1
#             c = torch.tensor(np.array([0, 1, 0]))
        elif dataset_type=='mrbrain_flair':
            c = torch.tensor([2])
        
#             c=2
#             c = torch.tensor(np.array([0, 0, 1]))   
    
        with h5py.File(fname, 'r') as data:
#             key_img = 'img_volus_{}'.format(acc_factor).translate(translator)
            source_img  = data['volfs'][:,:,slice].astype(np.float64)
            
            
            return torch.from_numpy(source_img),c
########################################################################################
class SliceDataref2(Dataset):

    def __init__(self, root, acc_factors,dataset_type_1,mask_types,train_or_valid): 
        self.examples = []
        escapes = ''.join([chr(char) for char in range(1, 32)])
        translator = str.maketrans('', '', escapes) 
        
#         print(f'datatype is {dataset_types},{mask_types},{train_or_valid},{acc_factors}')
        

        for dataset_type in dataset_type_1:
            dataroot = os.path.join(root,dataset_type).translate(translator)
            for mask_type in mask_types:
                newroot = os.path.join(dataroot,mask_type,train_or_valid).translate(translator)
                for acc_factor in acc_factors:
                    #print("acc_factor: ", acc_factor)
                    files = list(pathlib.Path(os.path.join(newroot,'acc_{}'.format(acc_factor)).translate(translator)).iterdir())
                    for fname in sorted(files):
                        with h5py.File(fname,'r') as hf:
                            fsvol = hf['volfs']
                            num_slices = fsvol.shape[2]
                            #print("acc_factor: ",acc_factor)
                            #acc_factor = float(acc_factor[:-1].replace("_","."))
                            self.examples += [(fname, slice, acc_factor, mask_type, dataset_type) for slice in range(num_slices)]
   

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice, acc_factor, mask_type, dataset_type = self.examples[i]
        
#         if dataset_type=='mrbrain_t1':
#             c = torch.tensor([0])
# #             c=0
# #             c = torch.tensor(np.array([1, 0, 0]))

              
#         elif dataset_type=='mrbrain_ir':
#             c = torch.tensor([1])
# #             c=1
# #             c = torch.tensor(np.array([0, 1, 0]))
#         elif dataset_type=='mrbrain_flair':
#             c = torch.tensor([2])
# #             c=2
# #             c = torch.tensor(np.array([0, 0, 1]))
# #         else: c = torch.tensor(np.array([0, 0]))
            
    
        with h5py.File(fname, 'r') as data:
#             key_img = 'img_volus_{}'.format(acc_factor).translate(translator)
            source_img  = data['volfs'][:,:,slice].astype(np.float64)
            source_img = torch.from_numpy(source_img).unsqueeze(0)
            source_img = (source_img - source_img.min())/(source_img.max() - source_img.min())
            source_img = F.pad(input=source_img, pad=(8, 8, 8, 8), mode='constant', value=0)
            
            
            return source_img
        
class SliceDataDev(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root,acc_factor,dataset_type,mask_type,mask_path):

        escapes = ''.join([chr(char) for char in range(1, 32)])
        translator = str.maketrans('', '', escapes)
        files = list(pathlib.Path(root.translate(translator)).iterdir())
        self.examples = []
        self.mask_path = mask_path.translate(translator) 
        
        for fname in sorted(files):
            with h5py.File(fname,'r') as hf:
                fsvol = hf['volfs']
                num_slices = fsvol.shape[2]
                self.examples += [(fname, slice, acc_factor,mask_type,dataset_type) for slice in range(num_slices)]

          

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        
        fname, slice, acc_factor,mask_type, dataset_type = self.examples[i]
        escapes = ''.join([chr(char) for char in range(1, 32)])
        translator = str.maketrans('', '', escapes)
    
        with h5py.File(fname, 'r') as data:

            key_img = 'img_volus_{}'.format(acc_factor).translate(translator) 
            key_kspace = 'kspace_volus_{}'.format(acc_factor).translate(translator) 
            input_img  = data[key_img][:,:,slice]
            input_kspace  = data[key_kspace][:,:,slice]
            input_kspace = npComplexToTorch(input_kspace)
            target = data['volfs'][:,:,slice]
 
            if dataset_type == 'cardiac':
#                 Cardiac dataset should be padded,150 becomes 160. # this can be commented for kirby brain 
                input_img  = np.pad(input_img,(5,5),'constant',constant_values=(0,0))
                target = np.pad(target,(5,5),'constant',constant_values=(0,0))

            mask = np.load(os.path.join(self.mask_path,dataset_type,mask_type,'mask_{}.npy'.format(acc_factor)).translate(translator))
            return torch.from_numpy(input_img), input_kspace, torch.from_numpy(target), torch.from_numpy(mask), str(fname.name),slice

# class SliceData(Dataset):
        
# #     """
# #     A PyTorch Dataset that provides access to MR image slices.
# #         """

#     def __init__(self, root,dataset_type,train_or_valid): 
        
#         escapes = ''.join([chr(char) for char in range(1, 32)])
#         translator = str.maketrans('', '', escapes)
#         newroot = os.path.join(root,dataset_type,train_or_valid).translate(translator)
# #         print(f'newroot is {newroot}')
#         files = list(pathlib.Path(newroot.translate(translator)).iterdir())
        
#         self.examples = []
# #         self.mask_path = mask_path.translate(translator) 
        
#         for fname in sorted(files):
#             with h5py.File(fname,'r') as hf:
#                 fsvol = hf['T1']
#                 num_slices = fsvol.shape[2]
#                 self.examples += [(fname, slice,dataset_type) for slice in range(num_slices)]
          



#     def __len__(self):
#         return len(self.examples)

#     def __getitem__(self, i):
        
#         fname, slice, dataset_type = self.examples[i] 
#         escapes = ''.join([chr(char) for char in range(1, 32)])
#         translator = str.maketrans('', '', escapes)
    
#         with h5py.File(fname, 'r') as data:


#             input_img  = data['T1'][:,:,slice].astype(np.float64)
#             target = data['FLR'][:,:,slice].astype(np.float64)# converting to double
#             mydict ={}
#             mydict['T1']=[torch.from_numpy(input_img),torch.tensor(np.array([1, 0, 0]))]
#             mydict['FLR']=[torch.from_numpy(target),torch.tensor(np.array([0, 1, 0]))]

#             return mydict#, input_kspace, torch.from_numpy(target),torch.from_numpy(mask)

