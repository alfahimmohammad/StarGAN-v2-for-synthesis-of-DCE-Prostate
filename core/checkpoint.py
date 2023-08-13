"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import torch


class CheckpointIO(object):
    def __init__(self, fname_template, data_parallel=False, **kwargs):
        os.makedirs(os.path.dirname(fname_template), exist_ok=True)
        self.fname_template = fname_template
        self.module_dict = kwargs
        self.data_parallel = data_parallel

    def register(self, **kwargs):
        self.module_dict.update(kwargs)

    def save(self, step):
        fname = self.fname_template.format(step)
        print('Saving checkpoint into %s...' % fname)
        outdict = {}
        for name, module in self.module_dict.items():
#             print("name",name)
#             print("module",module)
            if self.data_parallel:
                outdict[name] = module.module.state_dict()
            else:
                outdict[name] = module.state_dict()
#         print("checkpoint",outdict , fname)    
        torch.save(outdict, fname)

    def load(self, step):
        fname = self.fname_template.format(step)
        assert os.path.exists(fname), fname + ' does not exist!'
        print('Loading checkpoint from %s...' % fname)
        if torch.cuda.is_available():
#             print('blaa1')
            module_dict = torch.load(fname)
        else:
#             print('blaa2')
            module_dict = torch.load(fname, map_location=torch.device('cpu'))
        
        for name, module in self.module_dict.items():
#             print('blaa3')
            model_wts = module.state_dict()
#             for (tw,mw) in zip(module_dict[name],model_wts):
#                 print('blaa4')
#                 tmp = tw.split('.')
#                 print(tmp)
#                 x = tmp.pop(0)
#                 print('blaa5')
#                 res = ".".join([str(item) for item in tmp])
#                 print(res, mw)
#                 if res == mw:
#                     print(f'Loading weight {tw} to {mw}')
#                     model_wts[mw] = module_dict[name][tw]
            
            if self.data_parallel:
#                 print('blaa6')
                module.module.load_state_dict(model_wts)
            else:
#                 print('blaa7')
                module.load_state_dict(model_wts)
