# flake8: noqa
import os.path as osp

#from edited_scripts.basicsr_train import train_pipeline
from edited_scripts.basicsr import train_pipeline
import edited_scripts.realesrgan_data_realesrgan_paired_dataset

#import realesrgan.archs
#import realesrgan.data
#import realesrgan.models

if __name__ == '__main__':
    
    params = {
        "opt": "C:\\repos\\Real-ESRGAN\\options\\experements\\train_realesr-general-x4v3_pairdata.yml",
        "launcher": 'none', #'none', 'pytorch', 'slurm'
        "auto_resume": True,
        "debug": True,
        "local_rank": 0,
        "force_yml": None
    }

    #root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    root_path = "C:\\repos\\Real-ESRGAN\\"
    train_pipeline(params, root_path)