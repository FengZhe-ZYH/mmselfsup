_base_ = 'mae_vit-base-p16_8xb512-amp-coslr-300e_in1k.py'

# Initialize continuous pre-training from existing MAE model
load_from = 'https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-800e_in1k/mae_vit-base-p16_8xb512-coslr-800e-fp16_in1k_20220825-5d81fbc4.pth'

# Override dataset settings for PerioXrays Dataset
train_dataloader = dict(
    dataset=dict(
        type='mmcls.CustomDataset',
        data_root='',  
        data_prefix=dict(img_path='/hdd1/zyh/Datasets/PerioXrays/coco/images/train2017/'),
        ann_file='data/perioxrays_train.txt'
    )
)

# Do not resume training state, so it starts at epoch 0 with the loaded weights
resume = False
