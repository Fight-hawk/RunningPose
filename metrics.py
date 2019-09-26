#coding=utf-8

import numpy as np
import json
from locate_key_frame import *
import cv2
from PIL import Image, ImageDraw, ImageFont
import math
import sys
from collections import deque as dq
import codecs
'''
This is a set method evaluate running posture
'''

keypoints2index = {
    'lshouder': 5,
    'rshouder': 6,
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
    # float is not accuracy, so the value maybe higher than 1, that is an invalid value in method arccos
    cos_angle = 0.9999 if cos_angle > 1 else cos_angle
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


def neutral_position(head_top, middle_hip, upper_neck, angle_threshold, confidence_threshold=0.05):
    '''
    计算头部与躯干的夹角，判断是否过于前倾.
    :param keypoints: keypoints.
    :param angle_threshold: standard angle.
    :param confidence_threshold: keypoints confidence must more than this param, default is 0.05.
    :return: angle between head and body.
    '''
    x = np.array(head_top) - np.array(upper_neck)
    y = np.array(upper_neck) - np.array(middle_hip)
    angle = cal_angle(x, y)
    return angle, angle > angle_threshold


def stand_knee(lankle, lknee, lhip, rankle, rknee, rhip):
    langle = cal_angle(np.array(lhip) - np.array(lknee), np.array(lankle) - np.array(lknee))
    rangle = cal_angle(np.array(rhip) - np.array(rknee), np.array(rankle) - np.array(rknee))
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


def hip_flexibility(lhip, lknee, rhip, rknee, angle_threshold=70):
    angle = cal_angle(np.array(lknee) - np.array(lhip), np.array(rknee) - np.array(rhip))
    return angle, angle > angle_threshold


def leg_angle(lhip, lknee, rhip, rknee, angle_threshold=25, support_leg='left'):
    if support_leg == 'right':
        angle = cal_angle_v2(np.array(rknee)-np.array(rhip), np.array(lknee) - np.array(lhip))
    else:
        angle = cal_angle_v2(np.array(lknee) - np.array(lhip), np.array(rknee) - np.array(rhip))
    return angle, angle > angle_threshold


def stand_point(hip, ankle, left=True, angle_threshold=70):
    if left:
        angle = cal_angle(np.array(hip)-np.array(ankle), np.array([1, 0]))
    else:
        angle = cal_angle(np.array(hip)-np.array(ankle), np.array([1, 0]))
    return angle, angle < angle_threshold


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


def dist(a,b):
    return math.sqrt((a[0]-b[0])*(a[0]-b[0])+(a[1]-b[1])*(a[1]-b[1]))


if __name__ == '__main__':
    # video_path = '/Users/peijiexu/Downloads/runresult/1566290253686573.mp4'
    video_path = '/Users/kanhaipeng/Downloads/simple0917/044.mp4'
    finished = False
    font_size = 25
    num_keypoints = 33
    blue = (0, 0, 255)
    red = (255, 0, 0)
    with codecs.open('/Users/kanhaipeng/Downloads/simple0917/044.json', 'r',encoding='utf-8') as r:
        data = json.load(r)
    read_stream = cv2.VideoCapture(video_path)
    assert read_stream.isOpened(), 'can not open the video!'
    length = int(read_stream.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = int(read_stream.get(cv2.CAP_PROP_FOURCC))
    fps = read_stream.get(cv2.CAP_PROP_FPS)
    width = int(read_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(read_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    l_ankle_y_list = []
    r_ankle_y_list = []
    l_ankle_x_list = []
    r_ankle_x_list = []
    l_knee_x_list = []
    r_knee_x_list = []
    keypoint_data = []
    for i in range(num_keypoints):
        keypoint_data.append({})
    for i, result in enumerate(data):
        if result['keypoints'] is None:
            continue
        else:
            for j in range(num_keypoints):
                if result['keypoints'][j*3+2] > 0.05:
                    keypoint_data[j][i] = (result['keypoints'][j*3],-result['keypoints'][j*3+1])
    istreadmill = False
    if istreadmill:
        foot_stand_frame_indexes = tread_locate_frame(keypoint_data,len(data))
    else:
        foot_stand_frame_indexes = locate_frame(keypoint_data,len(data))

    left_foot_stand_frame_indexes = foot_stand_frame_indexes[0]
    right_foot_stand_frame_indexes = foot_stand_frame_indexes[1]

    frameSize = (int(read_stream.get(cv2.CAP_PROP_FRAME_WIDTH)), int(read_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    lindex = 0
    rindex = 0
    cv2.namedWindow('running', cv2.WINDOW_NORMAL)

    stand_knee_list = []
    stand_point_list = []
    hip_angle_list = []

    for i in range(length):
        ret, frame = read_stream.read()
        if not ret:
            break
        if i >= len(data):
            break
        if data[i]['keypoints'] is not None:

            # 中立位角
            if i in keypoint_data[keypoints2index['head_top']] and  \
                i in keypoint_data[keypoints2index['middle_hip']] and \
                 i in keypoint_data[keypoints2index['upper_neck']]:
                anlge, flag = neutral_position(keypoint_data[keypoints2index['head_top']][i],
                                               keypoint_data[keypoints2index['middle_hip']][i],
                                               keypoint_data[keypoints2index['upper_neck']][i], angle_threshold=10)
                if flag:
                    frame = put_text(frame, '中立位角:%.2f, 前倾过度' % anlge, (20, 20), font_size, red)
                else:
                    frame = put_text(frame, '中立位角:%.2f' % anlge, (20, 20), font_size, blue)

            # 髋关节灵活性
            if i in keypoint_data[keypoints2index['lhip']] and \
                i in keypoint_data[keypoints2index['lknee']] and \
                 i in keypoint_data[keypoints2index['rhip']] and \
                  i in keypoint_data[keypoints2index['rknee']]:
                angle, flag = hip_flexibility(lhip=keypoint_data[keypoints2index['lhip']][i],
                                              lknee=keypoint_data[keypoints2index['lknee']][i],
                                              rhip=keypoint_data[keypoints2index['rhip']][i],
                                              rknee=keypoint_data[keypoints2index['rknee']][i])
                hip_angle_list.append(angle)
                frame = put_text(frame, '髋关节角: %.2f, 最大值: %.2f' % (angle, max(hip_angle_list)), (20, 80), font_size,
                                 blue)

            if len(left_foot_stand_frame_indexes) > 0 \
                    and left_foot_stand_frame_indexes[lindex][0]-5 <= i < left_foot_stand_frame_indexes[lindex][0]:
                if i in keypoint_data[keypoints2index['lankle']] and \
                    i in keypoint_data[keypoints2index['lknee']] and \
                     i in keypoint_data[keypoints2index['lhip']] and \
                      i in keypoint_data[keypoints2index['rankle']] and \
                       i in keypoint_data[keypoints2index['rknee']] and \
                        i in keypoint_data[keypoints2index['rhip']]:
                    langle, rangle = stand_knee(lankle=keypoint_data[keypoints2index['lankle']][i],
                                                lknee=keypoint_data[keypoints2index['lknee']][i],
                                                lhip=keypoint_data[keypoints2index['lhip']][i],
                                                rankle=keypoint_data[keypoints2index['rankle']][i],
                                                rknee=keypoint_data[keypoints2index['rknee']][i],
                                                rhip=keypoint_data[keypoints2index['rhip']][i])
                    frame = put_text(frame, '落地伸膝角: %.2f %s' % (langle, '过大' if langle > 165 else ''), (20, 50), font_size, red if langle > 165 else blue)
                    stand_knee_list.append(langle)
                if i in keypoint_data[keypoints2index['lankle']] and \
                    i in keypoint_data[keypoints2index['lhip']]:
                    angle, flag_ = stand_point(ankle=keypoint_data[keypoints2index['lankle']][i],
                                               hip=keypoint_data[keypoints2index['lhip']][i],
                                               angle_threshold=70, left=True)
                    frame = put_text(frame, '落地角:%.2f %s' % (angle, '落地点靠前' if flag_ else ''), (20, 170), font_size,
                                     red if flag_ else blue)
                    stand_point_list.append(angle)

            if len(left_foot_stand_frame_indexes) > 0 and \
                    left_foot_stand_frame_indexes[lindex][0] <= i <= left_foot_stand_frame_indexes[lindex][1]:
                llength = left_foot_stand_frame_indexes[lindex][1] - left_foot_stand_frame_indexes[lindex][0] + 1

                frame = put_text(frame, '左脚落地', (20, 110), font_size, red)

                if i == left_foot_stand_frame_indexes[lindex][0]:
                    if i in keypoint_data[keypoints2index['lankle']] and \
                        i in keypoint_data[keypoints2index['lknee']] and \
                         i in keypoint_data[keypoints2index['lhip']] and \
                          i in keypoint_data[keypoints2index['rankle']] and \
                           i in keypoint_data[keypoints2index['rknee']] and \
                            i in keypoint_data[keypoints2index['rhip']]:
                        langle, rangle = stand_knee(lankle=keypoint_data[keypoints2index['lankle']][i],
                                                    lknee=keypoint_data[keypoints2index['lknee']][i],
                                                    lhip=keypoint_data[keypoints2index['lhip']][i],
                                                    rankle=keypoint_data[keypoints2index['rankle']][i],
                                                    rknee=keypoint_data[keypoints2index['rknee']][i],
                                                    rhip=keypoint_data[keypoints2index['rhip']][i])
                        stand_knee_list.append(langle)
                        frame = put_text(frame, '落地伸膝角: %.2f %s, 最大值: %.2f' % (langle, '过大' if langle > 165 else '', max(stand_knee_list)), (20, 50), font_size, red if langle > 165 else blue)
                        stand_knee_list = []
                    if i in keypoint_data[keypoints2index['lhip']] and \
                        i in keypoint_data[keypoints2index['lankle']]:
                        angle, flag_ = stand_point(hip=keypoint_data[keypoints2index['lhip']][i],
                                                   ankle=keypoint_data[keypoints2index['lankle']][i],
                                                   angle_threshold=70, left=True)
                        stand_point_list.append(angle)
                        frame = put_text(frame, '落地角: %.2f %s, 最小值: %.2f' % (
                        angle, '落地点靠前' if flag_ else '', min(stand_point_list)), (20, 170), font_size,
                                         red if flag_ else blue)
                        stand_point_list = []

                if i == left_foot_stand_frame_indexes[lindex][2]:
                    if i in keypoint_data[keypoints2index['lhip']] and \
                        i in keypoint_data[keypoints2index['lknee']] and \
                         i in keypoint_data[keypoints2index['rhip']] and \
                          i in keypoint_data[keypoints2index['rknee']]:
                        angle, flag = leg_angle(lhip=keypoint_data[keypoints2index['lhip']][i],
                                                lknee=keypoint_data[keypoints2index['lknee']][i],
                                                rhip=keypoint_data[keypoints2index['rhip']][i],
                                                rknee=keypoint_data[keypoints2index['rknee']][i],
                                                angle_threshold=25, support_leg='left')
                        if flag:
                            frame = put_text(frame, '收腿角:%.2f, 收腿太慢' % angle, (20, 140), font_size, red)
                        else:
                            frame = put_text(frame, '收腿角:%.2f' % angle, (20, 140), font_size, red)
                if i == left_foot_stand_frame_indexes[lindex][1] and lindex < len(left_foot_stand_frame_indexes) - 1:
                    lindex += 1

            if len(right_foot_stand_frame_indexes) > 0 and \
                    right_foot_stand_frame_indexes[rindex][0]-5 <= i < right_foot_stand_frame_indexes[rindex][0]:
                if i in keypoint_data[keypoints2index['lankle']] and \
                    i in keypoint_data[keypoints2index['lknee']] and \
                     i in keypoint_data[keypoints2index['lhip']] and \
                      i in keypoint_data[keypoints2index['rankle']] and \
                       i in keypoint_data[keypoints2index['rknee']] and \
                        i in keypoint_data[keypoints2index['rhip']]:
                    langle, rangle = stand_knee(lankle=keypoint_data[keypoints2index['lankle']][i],
                                                lknee=keypoint_data[keypoints2index['lknee']][i],
                                                lhip=keypoint_data[keypoints2index['lhip']][i],
                                                rankle=keypoint_data[keypoints2index['rankle']][i],
                                                rknee=keypoint_data[keypoints2index['rknee']][i],
                                                rhip=keypoint_data[keypoints2index['rhip']][i])
                    frame = put_text(frame, '落地伸膝角: %.2f %s' % (rangle, '过大' if rangle > 165 else ''), (20, 50), font_size,
                                     red if rangle > 165 else blue)
                    stand_knee_list.append(rangle)
                if i in keypoint_data[keypoints2index['rankle']] and \
                    i in keypoint_data[keypoints2index['rhip']]:
                    angle, flag_ = stand_point(ankle=keypoint_data[keypoints2index['rankle']][i],
                                               hip=keypoint_data[keypoints2index['rhip']][i],
                                               angle_threshold=70, left=False)
                    stand_point_list.append(angle)
                    frame = put_text(frame, '落地角:%.2f %s' % (angle, '落地点靠前' if flag_ else ''), (20, 170), font_size,
                                     red if flag_ else blue)

            if len(right_foot_stand_frame_indexes) > 0 and \
                    right_foot_stand_frame_indexes[rindex][0] <= i <= right_foot_stand_frame_indexes[rindex][1]:

                frame = put_text(frame, '右脚落地', (20, 110), font_size, red)

                if i == right_foot_stand_frame_indexes[rindex][0]:
                    if i in keypoint_data[keypoints2index['lankle']] and \
                        i in keypoint_data[keypoints2index['lknee']] and \
                         i in keypoint_data[keypoints2index['lhip']] and \
                          i in keypoint_data[keypoints2index['rankle']] and \
                           i in keypoint_data[keypoints2index['rknee']] and \
                            i in keypoint_data[keypoints2index['rhip']]:
                        langle, rangle = stand_knee(lankle=keypoint_data[keypoints2index['lankle']][i],
                                                    lknee=keypoint_data[keypoints2index['lknee']][i],
                                                    lhip=keypoint_data[keypoints2index['lhip']][i],
                                                    rankle=keypoint_data[keypoints2index['rankle']][i],
                                                    rknee=keypoint_data[keypoints2index['rknee']][i],
                                                    rhip=keypoint_data[keypoints2index['rhip']][i])
                        stand_knee_list.append(rangle)
                        frame = put_text(frame, '落地伸膝角: %.2f %s, 最大值: %.2f' % (
                        rangle, '过大' if rangle > 165 else '', max(stand_knee_list)), (20, 50), font_size,
                                         red if rangle > 165 else blue)
                        stand_knee_list = []
                    if i in keypoint_data[keypoints2index['rankle']] and \
                        i in keypoint_data[keypoints2index['rhip']]:
                        angle, flag_ = stand_point(hip=keypoint_data[keypoints2index['rhip']][i],
                                                   ankle=keypoint_data[keypoints2index['rankle']][i],
                                                   angle_threshold=70, left=False)
                        stand_point_list.append(angle)
                        frame = put_text(frame, '落地角: %.2f %s, 最小值: %.2f' % (
                            angle, '落地点靠前' if flag_ else '', min(stand_point_list)), (20, 170), font_size,
                                         red if flag_ else blue)
                        stand_point_list = []

                if i == right_foot_stand_frame_indexes[rindex][2]:
                    if i in keypoint_data[keypoints2index['lhip']] and \
                        i in keypoint_data[keypoints2index['lknee']] and \
                         i in keypoint_data[keypoints2index['rhip']] and \
                          i in keypoint_data[keypoints2index['rknee']]:
                        angle, flag = leg_angle(lhip=keypoint_data[keypoints2index['lhip']][i],
                                                lknee=keypoint_data[keypoints2index['lknee']][i],
                                                rhip=keypoint_data[keypoints2index['rhip']][i],
                                                rknee=keypoint_data[keypoints2index['rknee']][i],
                                                support_leg='right')
                        if flag:
                            frame = put_text(frame, '收腿角:%.2f,  收腿太慢' % angle, (20, 140), font_size, red)
                        else:
                            frame = put_text(frame, '收腿角:%.2f' % angle, (20, 140), font_size, red)

                if i == right_foot_stand_frame_indexes[rindex][1] and rindex < len(right_foot_stand_frame_indexes) - 1:
                    rindex += 1

            draw(keypoints=data[i]['keypoints'], frame=frame)
            cv2.imshow('running', frame)
            cv2.waitKey(0)
        else:
            cv2.imshow('running', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    read_stream.release()
    cv2.destroyAllWindows()

'''
1.落地区间
2.关键点 检测反
3.关键帧
'''