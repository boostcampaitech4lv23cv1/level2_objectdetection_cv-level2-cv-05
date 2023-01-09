# optimizer
# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type='AdamW', lr=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[16, 22])

# lr_config = dict(
#     policy = 'CosineAnnealing',
#     warmup='linear',
#     warmup_iters=1000,
#     warmup_ratio=1.0/10,
#     min_lr_ratio = 1e-5)

lr_config = dict(
    policy='CosineRestart',
    warmup='linear',
    warmup_iters=1099,
    warmup_ratio=0.001,
    periods=[5495, 5495, 6594, 8792, 8792],
    restart_weights=[1, 0.85, 0.75, 0.7, 0.6],
    by_epoch=False,
    min_lr=5e-6
    )


runner = dict(type='EpochBasedRunner', max_epochs=24)
