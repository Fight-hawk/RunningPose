import json
import os
import matplotlib
import math
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import cv2

"""Detect peaks in data based on their amplitude and other features."""

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

def merge(list):
    res = []
    if(len(list) <= 0):
        return res
    cur = list[0][0]
    l = []
    for i in range(len(list)):
        for j in range(0,len(list[i])):
            l.append(list[i][j])
            cur = list[i][j]
        if(i + 1 >= len(list)):
            if(len(l) >= 5):
                res.append(l)
            return res
        if(list[i+1][0] - cur <= 5):
            for j in range(cur+1,list[i+1][0]):
                l.append(j)
        else:
            if(len(l) >= 5):
                res.append(l)
            l = []
    return res

def locate_zone(data,length):
    lenlist = []
    for i in range(length):
        l = []
        if i in data:
            for j in range(length):
                if j in data:
                    dist = math.sqrt((data[i][0] - data[j][0]) * (data[i][0] - data[j][0]) + (data[i][1] - data[j][1]) * (data[i][1] - data[j][1]))
                    l.append(dist)
                else:
                    l.append(-1)
            lenlist.append(l)
        else:
            lenlist.append([])
    avg = 0.0
    c = 0
    for i in range(length-1):
        if i in data and i+1 in data:
            dist = lenlist[i][i+1]
            avg += dist
            c += 1
    avg /= c
    sum = 0
    count = []
    c = 0
    for i in range(length):
        if i in data:
            num = 0
            for j in range(length):
                if j in data:
                    d = lenlist[i][j]
                    if(d < avg):
                        num += 1
            count.append(num)
            sum += num
            c += 1
        else:
            count.append(-1)
    sum /= c

    result = []
    l = []
    for i in range(length):
        if i in data and count[i] >= sum * 1.5:
            l.append(i)
        else:
            if(len(l) > 0):
                result.append(l)
                l = []
    if(len(l) > 0):
        result.append(l)
    result = merge(result)
    return result

def locate(zone,length,ankle,knee):
    ranges = []
    for i in range(len(zone)):
        ranges.append((zone[i][0], zone[i][-1]))
    result = []
    for i in range(len(ranges)):
        minx = 100000
        minindex = -1
        for j in range(ranges[i][0], ranges[i][1] + 1):
            if j in ankle and j in knee:
                dx = ankle[j][0] - knee[j][0]
                if dx < 0:
                    dx = 100000
            else:
                continue
            if dx < minx:
                minx = dx
                minindex = j
        result.append((ranges[i][0], ranges[i][1], minindex))
    return result


def locate_frame(keypoint_data,length):
    if(length == 0):
        return []
    left = locate_zone(keypoint_data[keypoints2index['lankle']],length)
    right = locate_zone(keypoint_data[keypoints2index['rankle']],length)
    l = locate(left,length,keypoint_data[keypoints2index['lankle']],keypoint_data[keypoints2index['lknee']])
    r = locate(right,length,keypoint_data[keypoints2index['rankle']],keypoint_data[keypoints2index['rknee']])
    print(l)
    print(r)
    return (l,r)



