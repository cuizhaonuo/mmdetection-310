import os
import sys

from mmdet.apis import init_detector, inference_detector
from typing import List

print(sys.path)

GPU_DEVICES = 'cuda:0'

class ModelAdapter:
    '''
    {
        boxes: [[x1, y1, x2, y2] .. ],
        scores: [],
        labels: [],
        masks: [[h, w] .. ]
    }
    '''
    def predict(self, input_data):
        pass

class MMLabModelAdapter(ModelAdapter):
    def __init__(self, config_path, checkpoint_path):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.model = init_detector(self.config_path, self.checkpoint_path, device=GPU_DEVICES)
    def predict(self, input_data):
        result = inference_detector(self.model, input_data)
        return result

class MMDetAdapter(MMLabModelAdapter):
    def __init__(self):
        base_dir = '/public/home/zncui/pycharm_project/mmdetection'
        config_path = os.path.join(
            base_dir,
            'work_dirs/co_dino_guo_result/co_dino_5scale_swin_l_16xb1_16e_o365tococo.py'
        )
        checkpoint_path = os.path.join(
            base_dir,
            'work_dirs/co_dino_guo_result/epoch_16.pth'
        )
        super().__init__(config_path, checkpoint_path)

class MMRotateAdapter(MMLabModelAdapter):
    def __init__(self):
        import mmrotate
        base_dir = '/public/home/zncui/pycharm_project/mmrotate'
        config_path = os.path.join(
            base_dir,
            'work_dirs/oriented_reppoints_r50_fpn_40e_dota_ms_le135/oriented_reppoints_r50_fpn_40e_dota_ms_le135.py'
        )
        checkpoint_path = os.path.join(
            base_dir,
            'work_dirs/oriented_reppoints_r50_fpn_40e_dota_ms_le135/epoch_40.pth'
        )
        super().__init__(config_path, checkpoint_path)

class TransformersAdapter(ModelAdapter):
    def __init__(self):
        self.project_dir = '/public/home/zncui/pycharm_project/DINOv3-Mask2former/'
        self.model_dir = os.path.join(
            self.project_dir,
            'dinov3-large-mask2former-instance-ON-diverse-1024/best_model'
        )
    def load_hf_universal_seg(self):

        from transformers import AutoImageProcessor, AutoModelForUniversalSegmentation
        processor = AutoImageProcessor.from_pretrained(self.model_dir)
        model = AutoModelForUniversalSegmentation.from_pretrained(self.model_dir)
        model.to(GPU_DEVICES).eval()
        # id2label 用于可视化的类别名（可为空）
        id2label = getattr(model.config, "id2label", {}) or {}
        id2label = {int(k): v for k, v in id2label.items()} if isinstance(id2label, dict) else {}
        return processor, model, id2label, GPU_DEVICES



class DataLoaderStrategy:
    def collect_data(self):
        pass

class COCODataLoader(DataLoaderStrategy):
    def __init__(self):
        self.coco_path = '/public/home/zncui/datasets/zhaoxian/wheats_rle_coco/coco'
    def collect_data(self):
        self.coco_dict = dict(images=[], annotations=[])
        for pattern in ['val2017', 'test2017']:
            img_dir = os.path.join(self.coco_path, pattern)
            ann_file = os.path.join(self.coco_path, 'annotations', f'instances_{pattern}.json')
            for img_name in os.listdir(img_dir):
                if img_name.endswith('.jpg'):
                    self.coco_dict['images'].append(dict(file_name=os.path.join(img_dir, img_name)))
            self.coco_dict['annotations'].append(ann_file)

class YOLODataLoader(DataLoaderStrategy):
    def __init__(self):
        self.yolo_path = '/public/home/zncui/datasets/wheat-guo-dataset/yolo_wheat_big_image'
    def collect_data(self):
        self.yolo_dict = dict(images=[], labels=[])
        for pattern in ['images']:
            img_dir = os.path.join(self.yolo_path, 'images', pattern)
            label_dir = os.path.join(self.yolo_path, 'labels', pattern)
            for img_name in os.listdir(img_dir):
                if img_name.endswith('.jpg'):
                    self.yolo_dict['images'].append(os.path.join(img_dir, img_name))
                    label_name = img_name.replace('.jpg', '.txt')
                    self.yolo_dict['labels'].append(os.path.join(label_dir, label_name))

class COCOAndYOLODataLoader(DataLoaderStrategy):
    def __init__(self):
        self.coco_loader = COCODataLoader()
        self.yolo_loader = YOLODataLoader()
    def collect_data(self):
        self.coco_loader.collect_data()
        self.yolo_loader.collect_data()
        self.combined_data = {
            'coco': self.coco_loader.coco_dict,
            'yolo': self.yolo_loader.yolo_dict
        }
        return self.combined_data

class BaseInferencePipeline:
    def __init__(self, loader, predictor, postprocess, observers: List):
        self.loader = loader
        self.predictor = predictor
        self.postprocess = postprocess
        self.observers = observers

    def prepare_data(self): return self.loader.iter_batches()

    def run(self):
        for batch in self.prepare_data():
            images = [b["image"] for b in batch]
            preds = self.predictor.predict(images)
            preds = [self.postprocess(p) for p in preds]
            for obs in self.observers:
                obs.on_batch_done(images, preds)
        for obs in self.observers:
            obs.on_pipeline_end()

if __name__ == '__main__':
    model = MMDetAdapter()
    print(model)