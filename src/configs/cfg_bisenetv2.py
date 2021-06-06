# bisenet v2
cfg = dict(
    model_type='bisenetv2',
    num_aux_heads=4,
    lr_start=1e-3,
    momentum=0.9,
    optimizer_betas=(0.9, 0.999),
    weight_decay=1e-4,
    warmup_iters=50,
    max_iter=300000,
    checkpoint_iters=1000,
    message_iters=10,
    im_root='/home/bina/PycharmProjects/tevel-segmentation/data',
    train_im_anns='/home/bina/PycharmProjects/tevel-segmentation/data/train.txt',
    val_im_anns='/home/bina/PycharmProjects/tevel-segmentation/data/val.txt',
    demo_im_anns='/home/bina/PycharmProjects/tevel-segmentation/data/demo.txt',
    log_path='/home/bina/PycharmProjects/tevel-segmentation/logs/regular_logs',
    tensorboard_path='/home/bina/PycharmProjects/tevel-segmentation/logs/tensorboard_logs',
    models_path='/home/bina/PycharmProjects/tevel-segmentation/models',
    scales=[0.25, 2.],
    crop_size=(512, 768),
    ims_per_gpu=8,
    use_fp16=True,
    use_sync_bn=False,
)