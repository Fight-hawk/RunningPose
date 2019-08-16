import os
import cv2
import ntpath
from opt import opt
from tqdm import tqdm
from fn import getTime
from pPose_nms import write_json
from SPPE.src.main_fast_inference import *
from dataloader_single_thread import VideoLoader, DetectionLoader, DetectionProcessor, DataWriter, Mscoco

args = opt
args.dataset = 'coco'
if __name__ == "__main__":
    videofile = args.video
    mode = args.mode
    if not os.path.exists(args.outputpath):
        os.mkdir(args.outputpath)

    if not len(videofile):
        raise IOError('Error: must contain --video')

    # Load input video
    data_loader = VideoLoader(videofile, batchSize=args.detbatch)
    (fourcc, fps, frameSize) = data_loader.videoinfo()
    data_loader.start()

    # Load detection loader
    print('Loading YOLO model..')
    det_loader = DetectionLoader(data_loader, batchSize=args.detbatch)
    det_loader.start()


    det_processor = DetectionProcessor(det_loader)
    det_processor.start()
    # print(len(det_processor.Q))

    # Load pose model
    pose_dataset = Mscoco()
    if args.fast_inference:
        pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
    else:
        pose_model = InferenNet(4 * 1 + 1, pose_dataset)
    if opt.device == 'GPU':
        pose_model.cuda()
    else:
        pose_model.cpu()
    pose_model.eval()

    runtime_profile = {
        'dt': [],
        'pt': [],
        'pn': []
    }

    # Data writer
    save_path = os.path.join(args.outputpath, ntpath.basename(videofile).split('.')[0] + '.mp4')
    writer = DataWriter(args.save_video, save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frameSize)
    # print(data_loader.length())
    im_names_desc = tqdm(range(data_loader.length()))
    batchSize = args.posebatch
    for i in im_names_desc:
        start_time = getTime()
        with torch.no_grad():
            (inps, orig_img, im_name, boxes, scores, pt1, pt2) = det_processor.read(i)
            if orig_img is None:
                break
            if boxes is None or boxes.nelement() == 0:
                writer.save(None, None, None, None, None, orig_img, im_name.split('/')[-1])
                continue

            ckpt_time, det_time = getTime(start_time)
            runtime_profile['dt'].append(det_time)
            # Pose Estimation

            datalen = inps.size(0)
            leftover = 0
            if (datalen) % batchSize:
                leftover = 1
            num_batches = datalen // batchSize + leftover
            hm = []
            for j in range(num_batches):
                if opt.device == 'GPU':
                    inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)].cuda()
                else:
                    inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)].cpu()
                hm_j = pose_model(inps_j)
                hm.append(hm_j)
            hm = torch.cat(hm)
            ckpt_time, pose_time = getTime(ckpt_time)
            runtime_profile['pt'].append(pose_time)

            hm = hm.cpu().data
            writer.save(boxes, scores, hm, pt1, pt2, orig_img, im_name.split('/')[-1])

            ckpt_time, post_time = getTime(ckpt_time)
            runtime_profile['pn'].append(post_time)
    writer.start()
    final_result = writer.results()
    write_json(final_result, os.path.join(args.outputpath, ntpath.basename(args.video).split('.')[0] + '.json'))