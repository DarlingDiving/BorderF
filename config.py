class DefaultConfig(object):

    train_data_root = '/media/daidai/data/CNNaug_dataset/train/person'
    val_data_root = '/media/daidai/data/CNNaug_dataset/val/person'
    test_data_root = '/media/daidai/data/CNNaug_dataset/test/wang'
    load_model = True
    loaded_model_name = 'BorderF'
    load_model_path = './Daset/checkpoints/RGBLaplace'
    save_model = True
    save_model_path = './Daset/checkpoints/RGBLaplace'

    seed = 43
    batch_size = 4#14
    use_gpu = True
    gpu_id = '0'
    num_workers = 4
    print_iter = False # print training info every print_freq epochs

    img_size = 256
    optimizer = 'adam' 
    use_sam = False

    max_epoch = 100
    lr = 0.0001
    lr_decay = 0.96
    weight_decay = 0

    #resnext
    cardinality = 8
    depth = 29
    base_width = 64
    widen_factor = 4
    nlabels = 2
    dropout     = 0.0#0.3


    model_name  = 'BorderF'
    mid_loss_weight = 0.5
    train_noise = None
    test_noise  = None
    noise_scale = 0

opt = DefaultConfig()
