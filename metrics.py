#coding=utf-8

import numpy as np
import os
import json
from locate_key_frame import locate_frame
import cv2
from PIL import Image, ImageDraw, ImageFont
import math
import ntpath
import sys
from collections import deque as dq
import codecs
import argparse

'''
This is a set method evaluate running posture
'''

keypoints2index = {
    'lshoulder': 5,
    'rshoulder': 6,
    'lelbow': 7,
    'relbow': 8,
    'lwrist': 9,
    'rwrist': 10,
    'lhip': 11,
    'rhip': 12,
    'lknee': 13,
    'rknee': 14,
    'lankle': 15,
    'rankle': 16,
    'middle_hip': 23,
    'neck': 24,
    'upper_neck': 25,
    'head_top': 26
}


def cal_angle_v2(x, y):
    '''
    :param x:
    :param y:
    :return:
    '''
    a1 = math.atan2(x[1], x[0]) if math.atan2(x[1], x[0]) >= 0 else math.atan2(x[1], x[0]) + 2 * np.pi
    a2 = math.atan2(y[1], y[0]) if math.atan2(y[1], y[0]) >= 0 else math.atan2(y[1], y[0]) + 2 * np.pi
    return (a2 - a1) * 180 / np.pi


def cal_angle(x, y):
    '''
    calculate the angle between the 2d vector x and y.
    :param x: 1D array like.
    :param y: 1D array like.
    :return: angle between x and y.
    '''
    x = np.array(x)
    y = np.array(y)
    Lx = np.sqrt(x.dot(x))
    Ly = np.sqrt(y.dot(y))
    cos_angle = x.dot(y) / (Lx*Ly)
    #convert to angle
    angle = np.arccos(cos_angle) * 360 / 2 / np.pi
    return angle


def put_text(img, text, location, font_size, color):
    cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(cv2img)
    draw = ImageDraw.Draw(pilimg)
    font = ImageFont.truetype('/System/Library/Fonts/PingFang.ttc', font_size, encoding='utf-8')
    draw.text(location, text, color, font=font)
    image = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    return image


def neutral_position(keypoints, angle_threshold, confidence_threshold=0.05):
    '''
    计算头部与躯干的夹角，判断是否过于前倾.
    :param keypoints: keypoints.
    :param angle_threshold: standard angle.
    :param confidence_threshold: keypoints confidence must more than this param, default is 0.05.
    :return: angle between head and body.
    '''
    # get three keypoints {head_top, middel_hip, upper_neck}
    head_top = get_coordinate(keypoints, 'head_top')
    middle_hip = get_coordinate(keypoints, 'middle_hip')
    upper_neck = get_coordinate(keypoints, 'upper_neck')
    if head_top[-1] <= confidence_threshold or \
        middle_hip[-1] <= confidence_threshold or\
            upper_neck[-1] < confidence_threshold:
        return 0, False
    x = head_top[:2] - upper_neck[:2]
    y = upper_neck[:2] - middle_hip[:2]
    angle = cal_angle(x, y)
    return angle, angle > angle_threshold


def stand_knee(keypoints, confidence_threshold=0.05):
    langle, rangle = 0, 0
    lankle = get_coordinate(keypoints, 'lankle')
    lknee = get_coordinate(keypoints, 'lknee')
    lhip = get_coordinate(keypoints, 'lhip')
    rankle = get_coordinate(keypoints, 'rankle')
    rknee = get_coordinate(keypoints, 'rknee')
    rhip = get_coordinate(keypoints, 'rhip')
    if lankle[-1] > confidence_threshold and lknee[-1] > confidence_threshold and lhip[-1] > confidence_threshold:
        langle = cal_angle(lhip[:2] - lknee[:2], lankle[:2] - lknee[:2])
    if rankle[-1] > confidence_threshold and rknee[-1] > confidence_threshold and rhip[-1] > confidence_threshold:
        rangle = cal_angle(rhip[:2]-rknee[:2], rankle[:2]-rknee[:2])
    return langle, rangle


def step_frequency(stand_frame_indexes, fps=42):
    return (len(stand_frame_indexes)-1)/(abs(stand_frame_indexes[0][1]-stand_frame_indexes[-1][1])/fps)*60


def get_coordinate(keypoints, name):
    return np.array(keypoints[keypoints2index[name]*3: keypoints2index[name]*3+3])


def swing_arm(keypoints, confidence_threshold=0.05):
    lwrist = get_coordinate(keypoints, 'lwrist')
    lelbow = get_coordinate(keypoints, 'lelbow')
    lshoulder = get_coordinate(keypoints, 'lshoulder')
    if lwrist[-1] > confidence_threshold and lelbow[-1] > confidence_threshold and lshoulder[-1] > confidence_threshold:
        angle = cal_angle(lshoulder[:2] - lelbow[:2], lwrist[:2] - lelbow[:2])
    return 0 if angle is None else angle


def hip_flexibility(keypoints, angle_threshold=70, confidence_threshold=0.05):
    angle = 0
    lhip = get_coordinate(keypoints, 'lhip')
    lknee = get_coordinate(keypoints, 'lknee')
    rhip = get_coordinate(keypoints, 'rhip')
    rknee = get_coordinate(keypoints, 'rknee')
    if lhip[-1] > confidence_threshold and lknee[-1] > confidence_threshold and rhip[-1] > confidence_threshold and rknee[-1] > confidence_threshold:
        angle = cal_angle(lknee[:2]-lhip[:2], rknee[:2]-rhip[:2])
    return angle, angle > angle_threshold


def leg_angle(keypoints, angle_threshold=25, confidence_threshold=0.05, support_leg='left'):
    result = (0, False)
    lhip = get_coordinate(keypoints, 'lhip')
    lknee = get_coordinate(keypoints, 'lknee')
    rhip = get_coordinate(keypoints, 'rhip')
    rknee = get_coordinate(keypoints, 'rknee')
    if lhip[-1] > confidence_threshold and lknee[-1] > confidence_threshold \
        and rhip[-1] > confidence_threshold and rknee[-1] > confidence_threshold:
        if support_leg == 'right':
            angle = cal_angle_v2(lknee[:2]-lhip[:2], rknee[:2]-rhip[:2])
        else:
            angle = cal_angle_v2(rknee[:2]-rhip[:2], lknee[:2]-lhip[:2])
        result = (angle, angle > angle_threshold)
    return result



def stand_point(keypoints, left=True, angle_threshold=70):
    if left:
        lhip = np.array(keypoints[keypoints2index['middle_hip']*3: keypoints2index['middle_hip']*3+3])
        lankle = np.array(keypoints[keypoints2index['lankle']*3: keypoints2index['lankle']*3+3])
        angle = cal_angle(lhip[:2]-lankle[:2], np.array([1, 0]))
        return angle, angle > angle_threshold
    else:
        rhip = np.array(keypoints[keypoints2index['middle_hip'] * 3: keypoints2index['middle_hip'] * 3 + 3])
        rankle = np.array(keypoints[keypoints2index['rankle'] * 3: keypoints2index['rankle'] * 3 + 3])
        angle = cal_angle(rhip[:2]-rankle[:2], np.array([1, 0]))
        return angle, angle > angle_threshold


def vertical_amplitude(keypoints, confidence_threshold=0.05):
    result = {'left': None, 'right': None}
    lshoulder = get_coordinate(keypoints, 'lshoulder')
    lelbow = get_coordinate(keypoints, 'rshoulder')
    lwrist = get_coordinate(keypoints, 'lwrist')
    rshoulder = get_coordinate(keypoints, 'rshoulder')
    relbow = get_coordinate(keypoints, 'relbow')
    rwrist = get_coordinate(keypoints, 'rwrist')
    if lshoulder[-1] > confidence_threshold and lelbow[-1] > confidence_threshold and lwrist[-1] > confidence_threshold:
        angle = cal_angle(lshoulder[:2]-lelbow[:2], lwrist[:2]-lelbow[:2])
        result['left'] = angle
    if rshoulder[-1] > confidence_threshold and relbow[-1] > confidence_threshold and rwrist[-1] > confidence_threshold:
        angle = cal_angle(rshoulder[:2]-relbow[:2], rwrist[:2]-relbow[:2])
        result['right'] = angle
    return result


def draw(keypoints, frame):
    p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),
               # Nose, LEye, REye, LEar, REar
               (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),
               # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
               (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255)]
    line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                  (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                  (77, 222, 255), (255, 156, 127),
                  (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]
    l_pair = [
        (5, 25), (6, 25),
        (5, 7), (7, 9), (6, 8), (8, 10),
        (11, 13), (12, 14), (13, 15), (14, 16),
        (11, 23), (12, 23), (23, 24), (24, 25), (25, 26)
    ]
    part_line = {}
    # Draw keypoints
    for n, value in enumerate(keypoints2index.values()):
        if keypoints[value*3+2] < 0.05:
            continue
        cor_x, cor_y = int(keypoints[value*3]), int(keypoints[value*3+1])
        part_line[value] = (cor_x, cor_y)
        cv2.circle(frame, (cor_x, cor_y), 4, p_color[n], -1)
    # Draw limbs
    for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                cv2.line(frame, start_xy, end_xy, line_color[i],  3)


class KalmanTracker:
    def __init__(self, measurementMatDim, transitionMatDim):
        self.kalman = cv2.KalmanFilter(measurementMatDim, transitionMatDim)
        self.kalman.measurementMatrix = np.array([[1, 0], [0, 1]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0], [0, 1]], np.float32)
        self.kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.03

    def track(self, x, y):
        current_measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kalman.correct(current_measurement)
        current_prediction = self.kalman.predict()
        return current_prediction[0], current_prediction[1]

    def correct(self, x, y):
        current_measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kalman.correct(current_measurement)


def cal_x_var(x, max_len):
    assert max_len % 2 != 0, 'lenght must be odd'
    x_var = [sys.maxsize for _ in range(x[-1][1])]
    x_deq = dq(maxlen=max_len)
    for i in range(max_len-1):
        x_deq.append(x[i][0])
    for i in range(int((max_len-1)/2), len(x)-int((max_len-1)/2)-1):
        x_deq.append(x[i+int((max_len-1)/2)+1][0])
        x_var[x[i][1]] = np.var(x_deq)
    return x_var


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--kalman_filter', default=False, action='store_true', help='add kalman filter to all data')
    parser.add_argument('-v', '--video_path', default='', type=str, help='path of the input video')
    parser.add_argument('-j', '--json_path', default='', type=str, help='path of the input json file')
    parser.add_argument('-o', '--output_dir', default='./output', type=str, help='direction to save output video')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arg()
    video_path = args.video_path
    num_keypoints = 33
    font_size = 30
    blue = (255, 255, 0)
    red = (255, 0, 0)
    with codecs.open(args.json_path, 'r', encoding='utf-8') as r:
        data = json.load(r)
    if args.kalman_filter:
        trackers = {}
        for key in keypoints2index.keys():
            trackers[key] = KalmanTracker(2, 2)
        skipNo = 0
        skipIndex = 0

    read_stream = cv2.VideoCapture(video_path)
    assert read_stream.isOpened(), 'can not open the video!'
    length = int(read_stream.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = int(read_stream.get(cv2.CAP_PROP_FOURCC))
    fps = read_stream.get(cv2.CAP_PROP_FPS)
    keypoint_data = []
    for i in range(num_keypoints):
        keypoint_data.append({})

    for i, result in enumerate(data):
        if result['keypoints'] is None:
            continue
        if args.kalman_filter:
            for key, value in keypoints2index.items():
                if result['keypoints'][value * 3 + 2] > 0.05:
                    x, y = trackers[key].track(result['keypoints'][value * 3],
                                               result['keypoints'][value * 3 + 1])
                    if skipIndex < skipNo * 16:
                        skipIndex += 1
                    else:
                        data[i]['keypoints'][value*3], data[i]['keypoints'][value*3+1] = x[0], y[0]
        for j in range(num_keypoints):
            if result['keypoints'][j * 3 + 2] > 0.05:
                keypoint_data[j][i] = (result['keypoints'][j * 3], -result['keypoints'][j * 3 + 1])
    foot_stand_frame_indexes = locate_frame(keypoint_data, len(data))
    left_foot_stand_frame_indexes = foot_stand_frame_indexes[0]
    right_foot_stand_frame_indexes = foot_stand_frame_indexes[1]
    frameSize = (int(read_stream.get(cv2.CAP_PROP_FRAME_WIDTH)), int(read_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    lindex = 0
    rindex = 0

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    write_steam = cv2.VideoWriter(os.path.join(args.output_dir, ntpath.basename(args.video_path).split('.')[0] + '.mp4'), cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, frameSize)
    assert write_steam.isOpened(), 'can not open video'

    cv2.namedWindow('running', cv2.WINDOW_NORMAL)
    stand_knee_list = []
    stand_point_list = []
    hip_angle_list = []

    max_shoulder_angle = 0
    min_shoulder_angle = 180

    for i in range(length):
        ret, frame = read_stream.read()
        if not ret:
            break
        if i >= len(data):
            break
        if data[i]['keypoints'] is not None:

            # 最大伸肩角、最小曲肩角
            '''
            暂时 放弃 摆臂角度受手腕的高度影响，不准确
            angle = swing_arm(data[i]['keypoints'])
            if angle != 0:
                frame = put_text(frame, '最大伸肩角:%.2f %s' % (max(max_shoulder_angle, angle), '摆臂靠前' if max(max_shoulder_angle, angle) > 90 else ''), (20, 230), font_size, red if max(max_shoulder_angle, angle) > 90 else blue)
                max_shoulder_angle = max(max_shoulder_angle, angle)
                frame = put_text(frame, '最小曲肩角:%.2f %s' % (min(min_shoulder_angle, angle), '摆臂靠后' if min(min_shoulder_angle, angle) < 20 else ''), (20, 260), font_size, red if min(max_shoulder_angle, angle) < 20 else blue)
                min_shoulder_angle = min(min_shoulder_angle, angle)
            '''
            # 中立位角
            anlge, flag = neutral_position(data[i]['keypoints'], angle_threshold=10)
            if flag:
                frame = put_text(frame, '中立位角:%.2f, 前倾过度' % anlge, (20, 20), font_size, red)
            else:
                frame = put_text(frame, '中立位角:%.2f' % anlge, (20, 20), font_size, blue)

            # 髋关节灵活性
            angle, flag = hip_flexibility(keypoints=data[i]['keypoints'])
            hip_angle_list.append(angle)
            frame = put_text(frame, '髋关节角: %.2f, 最大值: %.2f' % (angle, max(hip_angle_list)), (20, 80), font_size, blue)

            if len(left_foot_stand_frame_indexes) > 0 and i >= left_foot_stand_frame_indexes[lindex][0] -5 and i < left_foot_stand_frame_indexes[lindex][0]:

                langle, rangle = stand_knee(keypoints=data[i]['keypoints'])
                frame = put_text(frame, '落地伸膝角: %.2f %s' % (langle, '过大' if langle > 165 else ''), (20, 50), font_size, red if langle > 165 else blue)
                stand_knee_list.append(langle)

                angle, flag_ = stand_point(data[i]['keypoints'], angle_threshold=70, left=True)
                frame = put_text(frame, '落地角:%.2f %s' % (angle, '落地点靠前' if flag_ else ''), (20, 170), font_size,
                                 red if flag_ else blue)
                stand_point_list.append(angle)

            if len(left_foot_stand_frame_indexes) > 0 and left_foot_stand_frame_indexes[lindex][0] <= i and left_foot_stand_frame_indexes[lindex][1] >= i :

                frame = put_text(frame, '左脚落地', (20, 110), font_size, red)

                if i == left_foot_stand_frame_indexes[lindex][0]:
                    langle, rangle = stand_knee(keypoints=data[i]['keypoints'])
                    stand_knee_list.append(langle)
                    frame = put_text(frame, '落地伸膝角: %.2f %s, 最大值: %.2f' % (langle, '过大' if langle > 165 else '', max(stand_knee_list)), (20, 50), font_size, red if langle > 165 else blue)
                    stand_knee_list = []
                    angle, flag_ = stand_point(data[i]['keypoints'], angle_threshold=70, left=True)
                    stand_point_list.append(angle)
                    frame = put_text(frame, '落地角: %.2f %s, 最小值: %.2f' % (
                    angle, '落地点靠前' if flag_ else '', min(stand_point_list)), (20, 170), font_size,
                                     red if flag_ else blue)
                    stand_point_list = []

                if i == left_foot_stand_frame_indexes[lindex][2]:
                    angle, flag = leg_angle(data[i]['keypoints'], angle_threshold=25, support_leg='left')

                    if flag:
                        frame = put_text(frame, '收腿角:%.2f, 收腿太慢' % angle, (20, 140), font_size, red)
                    else:
                        frame = put_text(frame, '收腿角:%.2f' % angle, (20, 140), font_size, red)
                if i == left_foot_stand_frame_indexes[lindex][1] and lindex < len(left_foot_stand_frame_indexes) - 1:
                    lindex += 1

            if len(right_foot_stand_frame_indexes) > 0 and i >= right_foot_stand_frame_indexes[rindex][0] -5 and i < right_foot_stand_frame_indexes[rindex][0]:
                langle, rangle = stand_knee(keypoints=data[i]['keypoints'])
                frame = put_text(frame, '落地伸膝角: %.2f %s' % (rangle, '过大' if rangle > 165 else ''), (20, 50), font_size, red if rangle > 165 else blue)
                stand_knee_list.append(rangle)

                angle, flag_ = stand_point(data[i]['keypoints'], angle_threshold=70, left=False)
                stand_point_list.append(angle)
                frame = put_text(frame, '落地角:%.2f %s' % (angle, '落地点靠前' if flag_ else ''), (20, 170), font_size,
                                 red if flag_ else blue)

            if len(right_foot_stand_frame_indexes) > 0 and right_foot_stand_frame_indexes[rindex][0] <= i and right_foot_stand_frame_indexes[rindex][1] >= i:

                frame = put_text(frame, '右脚落地', (20, 110), font_size, red)

                if i == right_foot_stand_frame_indexes[rindex][0]:
                    langle, rangle = stand_knee(keypoints=data[i]['keypoints'])
                    stand_knee_list.append(rangle)
                    frame = put_text(frame, '落地伸膝角: %.2f %s, 最大值: %.2f' % (rangle, '过大' if rangle > 165 else '', max(stand_knee_list)), (20, 50), font_size, red if rangle > 165 else blue)
                    stand_knee_list = []

                if i == right_foot_stand_frame_indexes[rindex][2]:
                    angle, flag = leg_angle(data[i]['keypoints'], angle_threshold=25, support_leg='right')
                    if flag:
                        frame = put_text(frame, '收腿角:%.2f,  收腿太慢' % angle, (20, 140), font_size, red)
                    else:
                        frame = put_text(frame, '收腿角:%.2f' % angle, (20, 140), font_size, red)
                    angle, flag_ = stand_point(data[i]['keypoints'], angle_threshold=70, left=False)
                    frame = put_text(frame, '落地角:%.2f' % angle, (20, 170), font_size, red)
                    if not flag_:
                        frame = put_text(frame, '落地角:%.2f, 落地点靠前' % angle, (20, 170), font_size, red)
                if i == right_foot_stand_frame_indexes[rindex][1] and rindex < len(right_foot_stand_frame_indexes) - 1:
                    rindex += 1

            # 球棍模型
            draw(keypoints=data[i]['keypoints'], frame=frame)

            # 帧号
            frame = put_text(frame, '帧号:%d' % i, (20, 200), font_size, blue)
            cv2.imshow('running', frame)
            write_steam.write(frame)
        else:
            cv2.imshow('running', frame)
            write_steam.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    write_steam.release()
    read_stream.release()
    cv2.destroyAllWindows()