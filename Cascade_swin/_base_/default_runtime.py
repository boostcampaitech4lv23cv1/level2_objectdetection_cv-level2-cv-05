# import wandb
import datetime
now = (datetime.datetime.now().replace(microsecond=0) + datetime.timedelta(hours=9)).strftime("%m-%d %H:%M")

checkpoint_config = dict(max_keep_ckpts=1, interval=1)

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
        dict(type='WandbLoggerHook',
         init_kwargs={
            'project': 'trash_detection',
            'entity': 'aidatapkt',
            'name' : f'final_CascadeRcnn_swinb384_fold3{now}_pkt',
            'tags': ['swinb384', 'mulitiscale 512~1024, 64', 'best augmentation'] 
            },
         interval=10)]
        )
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
