# # dataset settings
# dataset_type = 'CocoDataset'
# data_root = 'data/coco/'
# img_scale=(1024,1024)
# # img_scale=(512,512)
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# albu_train_transforms = [
#     dict(
#         type='VerticalFlip',
#     ),
#     # dict(
#     #     type='ShiftScaleRotate'
#     # ),
#     dict(
#         type='OneOf',
#         transforms=[
#             dict(type='RandomBrightnessContrast',
#             brightness_limit=[0.1,0.3],
#             contrast_limit=[0.1,0.3],
#             p = 1.0),
#             dict(type='CLAHE',p=1.0)
#         ],
#         p = 0.1),
#     dict(
#         type='RandomRotate90', p=0.5
#     ),
#     dict(
#         type='RandomResizedCrop',
#         height=1024,
#         width=1024,
#         scale=(0.5,1.0),
#         p = 0.2
#     ),
#     dict(
#         type='OneOf',
#         transforms=[
#             dict(type='Blur', blur_limit=3, p=1.0),
#             dict(type='MedianBlur', blur_limit=3, p=1.0)
#         ],
#         p=0.1),
    
    
# ]
# train_pipeline = [
#     # dict(type='LoadImageFromFile'),
#     # dict(type='LoadAnnotations', with_bbox=True),
#     # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    
#     dict(
#         type='Albu',
#         transforms=albu_train_transforms,
#         bbox_params=dict(
#             type='BboxParams',
#             format='pascal_voc',
#             label_fields=['gt_labels'],
#             min_visibility=0.0,
#             filter_lost_elements=True),
#         keymap={
#             'img': 'image',
#             'gt_bboxes': 'bboxes'
#         },
#         update_pad_shape=False,
#         skip_img_without_anno=True),
    

#     # dict(type='MixUp',
#     #      img_scale = img_scale,
#     #      ratio_range=(0.8,1.6),
#     #      pad_val=114.0),
    
#     dict(type='Mosaic',img_scale=img_scale,
#          bbox_clip_border=False,
#          skip_filter=False,
#          pad_val=114.0),
#     dict(type='RandomAffine',
#          scaling_ratio_range=(0.1,2),
#          border=(-img_scale[0]//2,-img_scale[1]//2)),
    
#     dict(type='RandomFlip', flip_ratio=0.5),
   
    
    
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
# ]

# dataset settings
dataset_type = 'CocoDataset'
data_root = '../../dataset/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
albu_train_transforms = [
    # dict(
    #     type='RandomResizedCrop',
    #     height=512,
    #     width=512,
    #     scale=[0.2,1.0],
    #     ratio=[0.7,1.5],
    #     interpolation=1,
    #     p=0.5),
    
    # dict(
    #     type='Cutout',
    #     num_holes=4, 
    #     max_h_size=96, 
    #     max_w_size=128, 
    #     p=0.5),
    
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
    # dict(type='RandomAffine',
    #      scaling_ratio_range=(0.1,2),
    #      border=(-img_scale[0]//2,-img_scale[1]//2)),
    dict(type='RandomFlip', flip_ratio=0.0),
    
    # dict(type='MixUp', pad_val=img_norm_cfg["mean"][::-1]),
    dict(
        type="Resize",
        img_scale=[(512 + 64 * i, 512 + 64 * i) for i in range(9)],
        multiscale_mode="value",
        keep_ratio=True,
    ),

    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels'],
        # meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg',
        #            'pad_shape', 'scale_factor')
    )
]


val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
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

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        # flip=False,
        flip=True,
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
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/instances_train2017.json',
            img_prefix=data_root + 'train2017/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
            ]
            ),
        pipeline=train_pipeline
        ),
        # type=dataset_type,
        # ann_file=data_root + 'annotations/instances_train2017.json',
        # img_prefix=data_root + 'train2017/',
        # pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
