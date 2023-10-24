# The new config inherits a base config to highlight the necessary modification
_base_ = 'configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
# model = dict(
#     roi_head=dict(
#         bbox_head=dict(num_classes=1), mask_head=dict(num_classes=1)))

# Modify dataset related settings
data_root = 'data/helm/'
metainfo = {
    'classes': ("head","helmet","person",)
    # 'palette': [
    #     (220, 20, 60),
    # ]
}

train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/_annotations.coco.json',
        data_prefix=dict(img='train/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='valid/_annotations.coco.json',
        data_prefix=dict(img='valid/')))
test_dataloader =  dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='test/_annotations.coco.json',
        data_prefix=dict(img='test/')))

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + 'valid/_annotations.coco.json')
test_evaluator = dict(ann_file=data_root + 'test/_annotations.coco.json')


train_cfg = dict(max_epochs=12, type='EpochBasedTrainLoop', val_interval=1)

# # We can use the pre-trained Mask RCNN model to obtain higher performance
# load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
