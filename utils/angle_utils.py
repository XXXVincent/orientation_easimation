import numpy as np

PI = 3.14159
P2 = [[712., 0, 640.], [0, 712., 360.], [0., 0., 1.]]

def check_angle(angle):
    if angle > PI:
        return angle - 2 * PI
    elif angle < -PI:
        return angle + 2 * PI
    else:
        return angle


def get_new_alpha(alpha):
    """utils.py:15
    change the range of orientation from [-pi, pi] to [0, 2pi]
    :param alpha: original orientation in KITTI
    :return: new alphautils.py:15utils.py:15
    """
    new_alpha = float(alpha) + np.pi / 2.
    if new_alpha < 0:
        new_alpha = new_alpha + 2. * np.pi
        # make sure angle lies in [0, 2pi]
    new_alpha = new_alpha - int(new_alpha / (2. * np.pi)) * (2. * np.pi)

    return new_alpha


def compute_anchors(alpha, bins, overlap):
    anchors = []
    wedge = 2. * PI / bins
    l_index = int(alpha / wedge)  # angle/pi
    r_index = l_index + 1

    # (angle - l_index*pi) < pi/2 * 1.05 = 1.65
    if (alpha - l_index * wedge) < wedge / 2 * (1 + overlap / 2):
        anchors.append([l_index, alpha - l_index * wedge])

    # (r*pi + pi - angle) < pi/2 * 1.05 = 1.65
    if (r_index * wedge - alpha) < wedge / 2 * (1 + overlap / 2):
        anchors.append([r_index % bins, alpha - r_index * wedge])

    return anchors


def compute_orientaion(P2, bbox2d, alpha):
    cx, cy = P2[0][2], P2[1][2]
    fx = P2[0][0]
    cx_bbox = bbox2d[0][0] + (bbox2d[2][0] - bbox2d[0][0])/2
    d = np.sqrt((cx_bbox - cx) ** 2)
    rot_ray = np.arctan(d / fx)
    # global = alpha + ray

    # rot_global = alpha + rot_ray
    if cx_bbox >= cx:
        rot_global=alpha - rot_ray
    else:
        rot_global= alpha + rot_ray
    assert -PI < check_angle(alpha) < PI
    rot_global = np.round(rot_global, 2)

    return rot_global


def recover_angle(bin_confidence, bin_anchor, bin_num):
    # select anchor from bins
    max_anc = np.argmax(bin_confidence)
    anchors = bin_anchor[max_anc]
    # compute the angle offset
    if anchors[1] > 0:
        angle_offset = np.arccos(anchors[0])
    else:
        angle_offset = -np.arccos(anchors[0])

    # add the angle offset to the center ray of each bin to obtain the local orientation
    wedge = 2 * np.pi / bin_num
    angle = angle_offset + max_anc * wedge

    # angle - 2pi, if exceed 2pi
    angle_l = angle % (2 * np.pi)

    # change to ray back to [-pi, pi]
    angle = angle_l - np.pi / 2
    if angle > np.pi:
        angle -= 2 * np.pi
    angle = np.round(angle, 2)

    return angle


def save_result(info, Alpha, rot_y, i,mode):
    ImageID = info['image'][0].split('.')[0]
    Class = info['class'][0]
    dimGT = info['dimensions']
    w = str(dimGT[0].item())
    h = str(dimGT[1].item())
    l = str(dimGT[2].item())
    BOX_2D = info['bbox2d']
    left_x = str(BOX_2D[0].item())
    left_y = str(BOX_2D[1].item())
    right_x = str(BOX_2D[2].item())
    right_y = str(BOX_2D[3].item())
    Loc = info['locations']
    location_x = str(Loc[0].item())
    location_y = str(Loc[1].item())
    location_z = str(Loc[2].item())
    Alpha = np.round(Alpha.item(), 2)
    rot_y = np.round(rot_y.item(), 2)

    if mode=='predict':
        # save_path = "output_pred/" #!
        save_path = "utils/kitti_eval_offline/results/test_set/" #!
    else:
        save_path = "utils/kitti_eval_offline/results/data/" #!

    save_path = save_path + ImageID + ".txt"
    with open(save_path, "a", encoding='utf-8') as f:
        line = Class + " 0.00 " + "0 " + str(Alpha) + " " + left_x + " " + \
               left_y + " " + right_x + " " + right_y + " " + w + \
               " " + h + " " + l + " " + location_x + " " \
               + location_y + " " + location_z + " " + str(rot_y) + " 1.0 " + "\n"

        f.writelines(line)
        f.close()
    print('object %d finished'%i)


