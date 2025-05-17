import os
import mmcv
from mmdet.apis import init_detector, inference_detector
from mmdet.visualization import DetLocalVisualizer


# mmdetection 安装目录，该值取决于你的安装环境
base_dir = '/public/home/zncui/pycharm_project/mmdetection'
# 输入图片
# in_img = '/public/home/zncui/datasets/wheats_rle_yolo/images/test/D6-017-20230529.jpg'
in_img = '/public/lab_share/nfs_field/level_1/DJI/P1/zhaoxian2021/20220528/706D6point4m/D6-002-R01L12/DJI_20220528113109_0004_0.png'
# in_img = 'demo2.jpg'
# 结果图片
out_img = '/public/home/zncui/pycharm_project/mmdetection/infer_dirs/DJI_20220528113109_0004_0-out.jpg'
# out_img = 'out2.jpg'

if __name__ == '__main__':
    # 模型配置文件
    config_file = os.path.join(base_dir, 'work_dirs/co_dino_5scale_swin_l_16xb1_16e_o365tococo/co_dino_5scale_swin_l_16xb1_16e_o365tococo.py')
    # 模型 checkpoint 文件
    # 下载地址: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
    checkpoint_file = r'/public/home/zncui/pycharm_project/mmdetection/work_dirs/co_dino_5scale_swin_l_16xb1_16e_o365tococo/epoch_16.pth'
    # 构建模型
    model = init_detector(config_file, checkpoint_file, device='cuda:1')
    # 测试单张图片
    # img = mmcv.imread(in_img)
    img = mmcv.imread(in_img, flag='color', channel_order='rgb', backend = 'cv2')
    # 利用模型对图片进行推理
    result = inference_detector(model, img)
    # 打印推理结果
    print(result)
    print(len(result.pred_instances.bboxes))
    # 可视化推理结果并保存图片
    visualizer = DetLocalVisualizer()
    visualizer.add_datasample('detection', img, result, draw_pred=True, show=False, out_file=out_img)

