# dataset settings
dataset_type = 'CocoDataset'
data_root = '../../dataset/'
train = 'train.json'
train_multilabel_sgkfold = '2___train_MultiStfKFold.json'
val_multilabel_sgkfold = '2___val_MultiStfKFold.json'
test = "test.json"

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

albu_train_transforms = [
    dict(
        type='Cutout',
        num_holes=8, 
        max_h_size=48, 
        max_w_size=48, 
        p=0.5),    
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='HorizontalFlip',
                p=1.0),
            dict(
                type='VerticalFlip',
                p=1.0),
            dict(
                type='RandomRotate90',
                p=1.0),
                ],
        p=0.5),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='HueSaturationValue',
                p=1.0),
            dict(
                type='CLAHE',
                p=1.0),
                ],
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[-0.2, 0.4],
        contrast_limit=[-0.5, 0.5],
        p=0.2),
]

train_pipeline = [
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='Mosaic', img_scale=(1024, 1024), pad_val=img_norm_cfg["mean"][::-1], prob=0.3),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(
        type="Resize",
        img_scale=[(512 + 128 * i, 512 + 128 * i) for i in range(9)],
        multiscale_mode="value",
        keep_ratio=True,
    ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels'],
    )
]

valid_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale = (512,512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(512 + 128 * i, 512 + 128 * i) for i in range(9)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + train_multilabel_sgkfold,
            img_prefix=data_root,
            classes=classes,
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
            ]
        ),
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + val_multilabel_sgkfold,
        img_prefix=data_root,
        classes=classes,
        pipeline=valid_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + test,
        img_prefix=data_root,
        classes=classes,
        pipeline=test_pipeline))

evaluation = dict(interval=1, save_best='bbox_mAP_50', metric='bbox')
