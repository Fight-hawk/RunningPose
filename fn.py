import cv2


def vis_frame_fast(frame, im_res):
    '''
    frame: frame image
    im_res: im_res of predictions
    format: coco or mpii

    return rendered image
    '''
    # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist, LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, middle hip, 1, 2, upper neck， head top
    p_index = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 0, 3]
    p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),
               # Nose, LEye, REye, LEar, REar
               (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),
               # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
               (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255),
               (203, 56, 234), (233, 22, 13)]
    line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                  (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                  (77, 222, 255), (255, 156, 127),
                  (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]
    l_pair = [
        # (0, 1), (0, 2), (1, 3), (2, 4),  # Headx
        (0, 14), (1, 14),
        (0, 2), (2, 4), (1, 3), (3, 5),
        (6, 8), (7, 9), (8, 10), (9, 11),
        (6, 12), (7, 12), (12, 13), (13, 14), (14, 15)
    ]

    # im_name = im_res['imgname'].split('/')[-1]
    img = frame
    for human in im_res['result']:
        part_line = {}
        kp_preds = human['keypoints'][p_index]
        kp_scores = human['kp_score'][p_index]
        bboxes = human['bboxes']
        up_left, right_bottom = (int(bboxes[0][0]), int(bboxes[0][1])), (int(bboxes[0][2]), int(bboxes[0][3]))
        bbox_scores = float(human['bbox_scores'])
        # Draw bboxes
        cv2.rectangle(img, up_left, right_bottom, (33, 253, 123), 4)
        cv2.putText(img, 'score:%.4f' % bbox_scores, up_left, cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 1)
        # Draw keypoints
        for n in range(kp_scores.shape[0]):
            if kp_scores[n] <= 0.05:
                continue
            cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
            part_line[n] = (cor_x, cor_y)
            cv2.circle(img, (cor_x, cor_y), 4, p_color[n], -1)
        # Draw limbs
        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                cv2.line(img, start_xy, end_xy, line_color[i], 2*(kp_scores[start_p] + kp_scores[end_p]) + 1)
    return img
