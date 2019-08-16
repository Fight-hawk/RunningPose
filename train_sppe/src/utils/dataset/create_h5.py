import json
import numpy as np
import h5py

# with open('../../../data/person_keypoints_val2017.json', 'r') as file:
#     id = None
#     json_obj = json.load(file)
#     image_names = []
#     annotations = []
#     parts = []
#     for annotation in json_obj['annotations']:
#         annotations.append([annotation['bbox']])
#         imgname = '%012d.jpg'%(annotation['image_id'])
#         imgname = [ord(ch) for ch in imgname]
#         image_names.append(imgname)
#         keypoints = []
#         # print(len(annotation['keypoints']))
#         for i in range(0, len(annotation['keypoints']), 3):
#             keypoints.append([annotation['keypoints'][i], annotation['keypoints'][i+1]])
#         parts.append(keypoints)

image_names = []
annotations = []
parts = []
with open('keepland.json', 'r') as file:
    json_obj = json.load(file)
    for obj in json_obj:
        image_name = obj['img_paths'].split('/')[-1].rjust(16, '0')
        image_name = [ord(ch) for ch in image_name]
        image_names.append(image_name)
        keypoints = obj['joint_self']
        for keypoint in keypoints:
            del keypoint[-1]
        parts.append(keypoints)
        annotations.append([obj['bbox']])

with open('COCO_train16k.json', 'r') as file:
    json_obj = json.load(file)
    for obj in json_obj['root']:
        image_name = obj['img_paths'].split('/')[-1]
        if image_name == '000000570624.jpg':
            print('have')
        image_name = [ord(ch) for ch in image_name]
        image_names.append(image_name)
        keypoints = obj['joint_self']
        for keypoint in keypoints:
            del keypoint[-1]
        parts.append(keypoints)
        annotations.append([obj['bbox']])
image_names = np.array(image_names)
annotations = np.array(annotations)
parts = np.array(parts)
indices = np.arange(image_names.shape[0])
np.random.shuffle(indices)
image_names = image_names[indices]
annotations = annotations[indices]
parts = parts[indices]
print(annotations.shape)
print(image_names.shape)
print(parts.shape)
h5file = h5py.File('annot_coco.h5', 'w')
h5file['imgname'] = image_names
h5file['part'] = parts
h5file['bndbox'] = annotations
h5file.close()