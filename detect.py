import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box, xywh2xyxy
from ultralytics.utils.ops import non_max_suppression
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def detect(opt):
    # 获取命令行的参数
    source1, source2, weights, view_img, save_txt, imgsz = opt.source1, opt.source2, opt.weights, opt.view_img, opt.save_txt, opt.img_size

    # 判断是保存图片还是视频
    save_img = not opt.nosave and not source1.endswith('.txt')  # save inference images
    webcam = source1.isnumeric() or source1.endswith('.txt') or source1.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # 创建保存目录
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # 初始化电脑硬件，加载显卡或者CPU
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # 加载模型
    model = attempt_load(weights, map_location=device)                          # 加载best.pt
    stride = int(model.stride.max())                                            # 获取模型步长
    imgsz = check_img_size(imgsz, s=stride)                                     # 检查图片尺寸(640 * 640)
    names = model.module.names if hasattr(model, 'module') else model.names     # 获取穴位的名字
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # 加载图片
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source1, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source1, img_size=imgsz, stride=stride)
        dataset2 = LoadImages(source2, img_size=imgsz, stride=stride)

    # # Run inference
    # if device.type != 'cpu':
    #     model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    # 开始图片的推理
    t0 = time.time()
    img_num = 0
    fps_sum = 0
    for (path, img, im0s, vid_cap), (path_, img2, im0s_, vid_cap_) in zip(dataset, dataset2):
        # print(path)
        # print(path_)
        print(f"【侦查报告】RGB图尺寸: {img.shape}, 深度图尺寸: {img2.shape}")

        img = torch.from_numpy(img).to(device)                                                      # 把图片送进显卡
        #img2 = torch.from_numpy(img2).to(device)

        # 将 img2 (深度图) 从 3 通道压缩为 1 通道
        # 假设 img2 形状是 [3, 640, 640]，取第一层变为 [1, 640, 640]
        img2 = torch.from_numpy(img2[:1, :, :]).to(device)

        # 把图片的 0-255 像素值变成0-1 之间的小数
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        img2 = img2.half() if half else img2.float()  # uint8 to fp16/32
        img2 /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img2.ndimension() == 3:
            img2 = img2.unsqueeze(0)

        # 把RGB和Depth拼在一起丢给模型
        t1 = time_synchronized()
        pred = model(torch.cat([img, img2], dim=1), augment=opt.augment)[0]

        # 筛选结果
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # 把结果画在图片上
        for i, det in enumerate(pred):  # detections per image

            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
                p, s, im0_, frame = path, '', im0s_.copy(), getattr(dataset2, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)                                                              # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # print(gn)

            # # ----------------------------------------------------------------------------
            # #  画GT，替换det
            # #
            # # ---------------------------------------------------------------------------
            # annoPath = "/home/fqy/proj/paper/test_result/gt/"
            # annoName  = (path_.split("/")[-1]).split(".")[0] + ".txt"
            # annoPath += annoName
            # # print(annoPath)
            # gt = np.loadtxt(annoPath)
            # gt = gt.reshape((-1, 5))
            # ones = np.ones((gt.shape[0], 1))
            # gt = np.hstack((gt, ones))
            # gt[:, [0,1,2,3,4,5]] = gt[:, [1,2,3,4,5,0]]
            # gt = torch.from_numpy(gt).to(device)
            # # print(gt[:, :4])
            # gt[:, :4] = xywh2xyxy(gt[:, :4]) * 640
            #
            # det = gt

            # print(det)
            # 如果坐标真找到了穴位
            if len(det):
                # 坐标还原，模型是在640尺寸上找的，这里把它按照比例放大回1280尺寸
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # 画框
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or opt.save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        # label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                        # 不需要标签的名字
                        label = None

                        # 在RGB和Depth上画框
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=opt.line_thickness)
                        plot_one_box(xyxy, im0_, label=label, color=colors(c, True), line_thickness=opt.line_thickness)
                        if opt.save_crop:
                            save_one_box(xyxy, im0s, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.6f}s, {1/(t2 - t1):.6f}Hz)')
            # add all the fps
            img_num += 1
            fps_sum += 1/(t2 - t1)

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    save_path_rgb = save_path.split('.')[0] + '_rgb.' + save_path.split('.')[1]
                    save_path_ir = save_path.split('.')[0] + '_ir.' + save_path.split('.')[1]
                    print(save_path_rgb)
                    cv2.imwrite(save_path_rgb, im0)
                    cv2.imwrite(save_path_ir, im0_)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')
    print(f'Average Speed: {fps_sum/img_num:.6f}Hz')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=r'weights\weights\best.pt', help='model.pt path(s)')
    parser.add_argument('--source1', type=str, default=r'test\rgb', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--source2', type=str, default=r'test\ir', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', default=False, action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=True, action='store_true', help='hide confidences')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect(opt=opt)
                strip_optimizer(opt.weights)
        else:
            detect(opt=opt)
