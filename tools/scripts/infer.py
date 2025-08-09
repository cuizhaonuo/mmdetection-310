import os
import mmcv
from mmdet.apis import init_detector, inference_detector
from mmdet.visualization import DetLocalVisualizer


base_dir = '/public/home/zncui/pycharm_project/mmdetection'
# 输入图片
in_img = '/public/lab_share/nfs_field/level_1/DJI/P1/zhaoxian2021/20220528/706D6point4m/D6-002-R01L12/DJI_20220528113109_0004_0.png'
# 结果图片
out_img = '/public/home/zncui/pycharm_project/mmdetection/infer_dirs/DJI_20220528113109_0004_0-out.jpg'

if __name__ == '__main__':
    config_file = os.path.join(base_dir, 'work_dirs/co_dino_5scale_swin_l_16xb1_16e_o365tococo/co_dino_5scale_swin_l_16xb1_16e_o365tococo.py')
    checkpoint_file = r'/public/home/zncui/pycharm_project/mmdetection/work_dirs/co_dino_5scale_swin_l_16xb1_16e_o365tococo/epoch_16.pth'
    model = init_detector(config_file, checkpoint_file, device='cuda:1')
    img = mmcv.imread(in_img, flag='color', channel_order='rgb', backend = 'cv2')
    result = inference_detector(model, img)
    print(result)
    print(len(result.pred_instances.bboxes))
    visualizer = DetLocalVisualizer()
    visualizer.add_datasample('detection', img, result, draw_pred=True, show=False, out_file=out_img)

