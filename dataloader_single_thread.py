import torch
import torch.utils.data as data
from SPPE.src.utils.img import cropBox, im_to_torch
from opt import opt
from yolo.preprocess import prep_frame
from pPose_nms import pose_nms
from SPPE.src.utils.eval import getPrediction
from yolo.util import dynamic_write_results
from yolo.darknet import Darknet
import cv2
import numpy as np
if opt.vis_fast:
    from fn import vis_frame_fast as vis_frame
else:
    from fn import vis_frame


class VideoLoader:
    def __init__(self, path, batchSize=1):
        # initialize the file video stream and put it into a list
        self.path = path
        self.stream = cv2.VideoCapture(path)
        assert self.stream.isOpened(), 'Cannot capture source'
        self.stopped = False
        self.batchSize = batchSize
        # get the total frames of the video
        # but the the number is not accuate
        self.datalen = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        leftover = 0
        if self.datalen % batchSize:
            leftover = 1
        self.num_batches = self.datalen // batchSize + leftover

        # initialize the list used to store frames read from
        # the video file
        self.Q = list()

    def length(self):
        return len(self.Q)

    def start(self):
        # start to read frames from the file video stream
        self.update()

    def update(self):
        for i in range(self.num_batches):
            img = []
            orig_img = []
            im_name = []
            im_dim_list = []
            for k in range(i*self.batchSize, min((i + 1)*self.batchSize, self.datalen)):
                inp_dim = int(opt.inp_dim)
                (grabbed, frame) = self.stream.read()
                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    return
                img_k, orig_img_k, im_dim_list_k = prep_frame(frame, inp_dim)
                img.append(img_k)
                orig_img.append(orig_img_k)
                im_name.append(k)
                im_dim_list.append(im_dim_list_k)

            with torch.no_grad():
                # Human Detection
                img = torch.cat(img)
                im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)
            self.Q.append((img, orig_img, im_name, im_dim_list))

    def videoinfo(self):
        # get the video info
        fourcc = int(self.stream.get(cv2.CAP_PROP_FOURCC))
        fps = self.stream.get(cv2.CAP_PROP_FPS)
        frameSize = (int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)) // int(2 if opt.resize else 1),
                     int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)) // int(2 if opt.resize else 1))
        return fourcc, fps, frameSize

    def len(self):
        return len(self.Q)


class DetectionLoader:
    def __init__(self, dataloder, batchSize=1):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.det_model = Darknet("yolo/cfg/yolov3-spp.cfg")
        self.det_model.load_weights('models/yolo/yolov3-spp.weights')
        self.det_model.net_info['height'] = opt.inp_dim
        self.det_inp_dim = int(self.det_model.net_info['height'])
        assert self.det_inp_dim % 32 == 0
        assert self.det_inp_dim > 32
        if opt.device == 'GPU':
            self.det_model.cuda()
        else:
            self.det_model.cpu()
        self.det_model.eval()

        self.dataloder = dataloder
        self.batchSize = batchSize
        self.datalen = self.dataloder.length()
        leftover = 0
        if (self.datalen) % batchSize:
            leftover = 1
        self.num_batches = self.datalen // batchSize + leftover
        # initialize the list used to store frames read from
        # the video file
        self.Q = list()

    def start(self):
        # start to dectect person
        self.update()

    def update(self):
        # keep looping the whole dataset
        for i in range(self.num_batches):
            img, orig_img, im_name, im_dim_list = self.dataloder.Q[i]
            with torch.no_grad():
                # Human Detection
                if opt.device == 'GPU':
                    img = img.cuda()
                else:
                    img = img.cpu()

                prediction = self.det_model(img, CUDA=True if opt.device == 'GPU' else False)
                # NMS process
                dets = dynamic_write_results(prediction, opt.confidence,
                                    opt.num_classes, nms=True, nms_conf=opt.nms_thesh)
                if isinstance(dets, int) or dets.shape[0] == 0:
                    for k in range(len(orig_img)):
                        self.Q.append((orig_img[k], im_name[k], None, None, None, None, None))
                    continue
                dets = dets.cpu()
                im_dim_list = torch.index_select(im_dim_list, 0, dets[:, 0].long())
                scaling_factor = torch.min(self.det_inp_dim / im_dim_list, 1)[0].view(-1, 1)

                # coordinate transfer
                dets[:, [1, 3]] -= (self.det_inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
                dets[:, [2, 4]] -= (self.det_inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

                dets[:, 1:5] /= scaling_factor
                for j in range(dets.shape[0]):
                    dets[j, [1, 3]] = torch.clamp(dets[j, [1, 3]], 0.0, im_dim_list[j, 0])
                    dets[j, [2, 4]] = torch.clamp(dets[j, [2, 4]], 0.0, im_dim_list[j, 1])
                boxes = dets[:, 1:5]
                scores = dets[:, 5:6]

            for k in range(len(orig_img)):
                boxes_k = boxes[dets[:, 0] == k]
                if isinstance(boxes_k, int) or boxes_k.shape[0] == 0:
                    self.Q.append((orig_img[k], im_name[k], None, None, None, None, None))
                    continue
                inps = torch.zeros(boxes_k.size(0), 3, opt.inputResH, opt.inputResW)
                pt1 = torch.zeros(boxes_k.size(0), 2)
                pt2 = torch.zeros(boxes_k.size(0), 2)
                # multiply the score with bounding box height
                processed_scores = self.cal_scores(scores, boxes_k)
                self.Q.append((orig_img[k], im_name[k], boxes_k[np.argmax(processed_scores):np.argmax(processed_scores)+1], scores[np.argmax(processed_scores)], inps[np.argmax(processed_scores):np.argmax(processed_scores)+1], pt1[np.argmax(processed_scores):np.argmax(processed_scores)+1], pt2[np.argmax(processed_scores):np.argmax(processed_scores)+1]))

    def cal_scores(self, scores, boxes):
        processed_scores = scores.clone()
        for i in range(boxes.shape[0]):
            processed_scores[i][0] *= abs(boxes[i][1]-boxes[i][3])
        return processed_scores

    def len(self):
        # return list len
        return len(self.Q)


class DetectionProcessor:
    def __init__(self, detectionLoader):
        self.detectionLoader = detectionLoader
        self.datalen = self.detectionLoader.datalen
        # initialize the list used to store data
        self.Q = list()

    def start(self):
        # start to process detection results
        self.update()

    def update(self):
        # keep looping the whole dataset
        for i in range(self.datalen):
            
            with torch.no_grad():
                (orig_img, im_name, boxes, scores, inps, pt1, pt2) = self.detectionLoader.Q[i]
                if boxes is None or boxes.nelement() == 0:
                    self.Q.append((None, orig_img, im_name, boxes, scores, None, None))
                    continue
                inp = im_to_torch(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
                inps, pt1, pt2 = crop_from_dets(inp, boxes, inps, pt1, pt2)
                self.Q.append((inps, orig_img, im_name, boxes, scores, pt1, pt2))

    def len(self):
        # return list len
        return len(self.Q)


class DataWriter:
    def __init__(self, save_video=False,
                savepath='examples/res/1.avi', fourcc=cv2.VideoWriter_fourcc(*'XVID'), fps=25, frameSize=(640,480)):
        if save_video:
            self.stream = cv2.VideoWriter(savepath, fourcc, fps, frameSize)
            assert self.stream.isOpened(), 'Cannot open video for writing'
        self.save_video = save_video
        self.final_result = []
        # initialize the list used to store frames read from
        # the video file
        self.Q = list()
        # the index of the keypoints that we need

    def start(self):
        # start to get the right result
        for i in range(len(self.Q)):
            (boxes, scores, hm_data, pt1, pt2, orig_img, im_name) = self.Q[i]
            orig_img = np.array(orig_img, dtype=np.uint8)
            if boxes is None or boxes.nelement() == 0:
                result = {
                    'imgname': im_name,
                    'result': None,
                }
                self.final_result.append(result)
                if opt.save_video:
                    self.stream.write(orig_img)
            else:
                preds_hm, preds_img, preds_scores = getPrediction(
                    hm_data, pt1, pt2, opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW)

                result = pose_nms(
                    boxes, scores, preds_img, preds_scores)
                if len(result) == 0:
                    result = {
                        'imgname': im_name,
                        'result': None,
                    }
                else:
                    result = {
                        'imgname': im_name,
                        'result': result,
                    }
                self.final_result.append(result)
                if opt.save_video:
                    img = orig_img
                    if opt.vis and result['result'] is not None:
                        img = vis_frame(orig_img, result)
                    self.stream.write(img)

    def save(self, boxes, scores, hm_data, pt1, pt2, orig_img, im_name):
        self.Q.append((boxes, scores, hm_data, pt1, pt2, orig_img, im_name))

    def results(self):
        # return final result
        return self.final_result

    def len(self):
        # return list len
        return len(self.Q)


class Mscoco(data.Dataset):
    def __init__(self, train=True, sigma=1,
                 scale_factor=(0.2, 0.3), rot_factor=40, label_type='Gaussian'):
        self.img_folder = '../data/coco/images'    # root image folders
        self.is_train = train           # training set or test set
        self.inputResH = opt.inputResH
        self.inputResW = opt.inputResW
        self.outputResH = opt.outputResH
        self.outputResW = opt.outputResW
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.label_type = label_type

        self.nJoints_coco = 17
        self.nJoints_mpii = 16
        self.nJoints = 33

        self.accIdxs = (1, 2, 3, 4, 5, 6, 7, 8,
                        9, 10, 11, 12, 13, 14, 15, 16, 17)
        self.flipRef = ((2, 3), (4, 5), (6, 7),
                        (8, 9), (10, 11), (12, 13),
                        (14, 15), (16, 17))

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


def crop_from_dets(img, boxes, inps, pt1, pt2):
    '''
    Crop human from origin image according to Dectecion Results
    '''

    imght = img.size(1)
    imgwidth = img.size(2)
    tmp_img = img
    tmp_img[0].add_(-0.406)
    tmp_img[1].add_(-0.457)
    tmp_img[2].add_(-0.480)
    for i, box in enumerate(boxes):
        upLeft = torch.Tensor((float(box[0]), float(box[1])))
        bottomRight = torch.Tensor((float(box[2]), float(box[3])))

        ht = bottomRight[1] - upLeft[1]
        width = bottomRight[0] - upLeft[0]

        scaleRate = 0.3

        upLeft[0] = max(0, upLeft[0] - width * scaleRate / 2)
        upLeft[1] = max(0, upLeft[1] - ht * scaleRate / 2)
        bottomRight[0] = max(
            min(imgwidth - 1, bottomRight[0] + width * scaleRate / 2), upLeft[0] + 5)
        bottomRight[1] = max(
            min(imght - 1, bottomRight[1] + ht * scaleRate / 2), upLeft[1] + 5)

        try:
            inps[i] = cropBox(tmp_img.clone(), upLeft, bottomRight, opt.inputResH, opt.inputResW)
        except IndexError:
            print(tmp_img.shape)
            print(upLeft)
            print(bottomRight)
            print('===')
        pt1[i] = upLeft
        pt2[i] = bottomRight

    return inps, pt1, pt2
