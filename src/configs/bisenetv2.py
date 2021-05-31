# bisenet v2
cfg = dict(
    model_type='bisenetv2',
    num_aux_heads=4,
    lr_start=1e-4,  # 5e-2,
    momentum=0.9,
    weight_decay=5e-4,
    warmup_iters=500,
    max_iter=20000,
    checkpoint_iters=1000,
    message_iters=50,
    im_root='/home/bina/PycharmProjects/tevel-segmentation/data',
    train_im_anns='/home/bina/PycharmProjects/tevel-segmentation/data/train_small.txt',
    # '/home/bina/PycharmProjects/tevel-segmentation/data/train.txt',
    val_im_anns='/home/bina/PycharmProjects/tevel-segmentation/data/train_small.txt',
    scales=[0.25, 2.],
    crop_size=[512, 1024],
    ims_per_gpu=2,
    use_fp16=True,
    use_sync_bn=False,
    respth='/home/bina/PycharmProjects/tevel-segmentation/models',
    logpth='/home/bina/PycharmProjects/tevel-segmentation/logs',
)
