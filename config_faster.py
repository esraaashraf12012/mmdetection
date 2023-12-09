# The new config inherits a base config to highlight the necessary modification
_base_ = 'configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
# Modify dataset related settings
data_root = 'data/helm/'
metainfo = {
    'classes': ("head","helmet","person",) ,
     'palette': [
        (220, 20, 60),
        (119, 11, 32),
        (0, 0, 142),
     ] # The palette colors of bounding box masks
}

train_dataloader = dict( #loader for training
    batch_size=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/_annotations.coco.json',
        data_prefix=dict(img='train/')))
val_dataloader = dict( #loader for validation
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='valid/_annotations.coco.json',
        data_prefix=dict(img='valid/')))
test_dataloader =  dict( #loader for testing
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='test/_annotations.coco.json',
        data_prefix=dict(img='test/')))

# model evaluation settings
val_evaluator = dict(ann_file=data_root + 'valid/_annotations.coco.json') #validation evaluator
test_evaluator = dict(ann_file=data_root + 'test/_annotations.coco.json') #test evaluator


train_cfg = dict(max_epochs=12, type='EpochBasedTrainLoop', val_interval=1) #train config , 12 epochs, 1 validation per epoch


