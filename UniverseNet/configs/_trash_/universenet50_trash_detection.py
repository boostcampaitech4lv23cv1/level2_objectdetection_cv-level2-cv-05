_base_ = [
    './models/universenet50_2008d_soft.py',
    './datasets/coco_trash_detection_1.py',
    './schedules/schedule_1x.py', './default_runtime.py'
]

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-5)

runner = dict(type='EpochBasedRunner', max_epochs=21)

fp16 = dict(loss_scale=512.)
