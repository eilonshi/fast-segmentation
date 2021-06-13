# bisenet v2
cfg = dict(
    num_aux_heads=4,
    lr_start=1e-3,
    momentum=0.9,
    optimizer_betas=(0.9, 0.999),
    weight_decay=1e-4,
    warmup_iters=50,
    max_iter=300000,
    checkpoint_iters=10,
    message_iters=10,
    scales=[0.25, 2.],
    crop_size=(512, 768),
    ims_per_gpu=2,
    use_fp16=True,
    use_sync_bn=False,
)
