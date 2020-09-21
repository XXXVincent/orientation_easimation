import os
import csv
import argparse
import numpy as np
import xml.etree.ElementTree as ET

PI = 3.14159
K = [[712., 0, 640.], [0, 712., 360.], [0., 0., 1.]]


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def check_last_line_break(predict_txt):
    f = open(predict_txt, 'rb+')
    try:
        f.seek(-1, os.SEEK_END)
    except:
        pass
    else:
        if f.__next__() == b'\n':
            f.seek(-1, os.SEEK_END)
            f.truncate()
    f.close()


def degree2rad(degree):
    return degree * PI / 180


def check_angle(angle):
    if angle > PI:
        return angle - 2 * PI
    elif angle < -PI:
        return angle + 2 * PI
    else:
        return angle


def roty2alpha(K, rot_y, bbox2d):
    cx, cy = K[0][2], K[1][2]
    fx = K[0][0]
    cx_bbox = bbox2d[0] + (bbox2d[2] - bbox2d[0])/2
    d = np.sqrt((cx_bbox - cx) ** 2 )
    rot_ray = np.arctan(d / fx)

    if cx_bbox >= cx:
        alpha = rot_y + rot_ray
    else:
        alpha = rot_y - rot_ray
    alpha_new = check_angle(alpha)
    assert -PI < alpha_new < PI
    return alpha_new


class XML2KITTI():
    def __init__(self, sourcedir,desdir, camera='front'):
        self.xml_dir = os.path.join(sourcedir, 'xml')
        self.angle_dir = os.path.join(sourcedir, 'angle')
        self.match_file = os.path.join(sourcedir, 'match_list.txt')
        self.new_dir = desdir
        self.camera = camera

        self.template = "{name} 0.00 0 {alpha} {xmin}.00 {ymin}.00 {xmax}.00 {ymax}.00 " \
                        "0.0 0.0 0.0 0.0 0.0 0.0 {roty} 0.0"

    @property
    def match_img_file(self):
        '''
        Match same image sequence with different names.
        '''
        with open(self.match_file, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            matches = {row[0][9:-4]: row[1][10:-4]
                       for line, row in enumerate(reader)}

        return matches

    def get_angles(self, angle_file):
        with open(angle_file, 'r') as f:
            reader = csv.reader(f, delimiter=' ')
            try:
                angles = {row[0]: row[1] for line, row in enumerate(reader)
                         if row[1] != 'null'}
                return angles
            except IndexError:
                print("Error in ", angle_file)


    def get_objects(self, angle_file, xml_file, min_obj_size=56*56):
        objs = []

        xml_content = open(xml_file, 'r').read()
        objects_roots = ET.fromstring(xml_content)
        objects = objects_roots.findall('object')

        angles = self.get_angles(angle_file)
        for object in objects:
            id = object.find("difficult").text
            for k2, v2 in angles.items():
                if id == k2:
                    if self.camera == 'front':
                        rot_y = round(degree2rad(float(v2)), 4)
                    elif self.camera == 'left':
                        rot_y = round(degree2rad(float(v2)), 4) - PI/2
                    elif self.camera == 'right':
                        rot_y = round(degree2rad(float(v2)), 4) + PI/2
                    else:
                        raise NameError("camera not recognized", self.camera)
                    xmin = object.find("bndbox").find("xmin").text
                    xmax = object.find("bndbox").find("xmax").text
                    ymin = object.find("bndbox").find("ymin").text
                    ymax = object.find("bndbox").find("ymax").text
                    bbox = [float(xmin), float(ymin), float(xmax), float(ymax)]
                    if (bbox[3]-bbox[1])*(bbox[2]-bbox[0])<min_obj_size:
                        continue
                    alpha = roty2alpha(K, rot_y, bbox)
                    objs.append({
                        "name": object.find("name").text,
                        "alpha": round(alpha, 4),
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax,
                        "roty": round(degree2rad(float(v2)), 4),
                    })

        return objs

    def fill_template(self, object):
        return self.template.format(**object)

    def export_kitti(self):
        matches = self.match_img_file
        for k1, v1 in matches.items():
            # k1 = '334812020'
            # v1 = '000309'
            xml_file = os.path.join(self.xml_dir, v1) + '.xml'
            angle_file = os.path.join(self.angle_dir, k1) + '.txt'
            if not os.path.exists(angle_file):
                continue
            objects = self.get_objects(angle_file, xml_file)

            # assert objects

            label_file = os.path.join(self.new_dir, v1) + '.txt'
            with open(label_file, 'w') as f:
                for object in objects:
                    row = self.fill_template(object)
                    f.write(row + "\n")

            check_last_line_break(label_file)
            print('Generating label file {}'.format(label_file))


def main(args):
    xml2kitti = XML2KITTI(args.sourcedir,args.desdir, args.camera)
    xml2kitti.export_kitti()

    print("Finished transfer")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=str, default='front')
    parser.add_argument( '--sourcedir', type=str, default='./')
    parser.add_argument( '--desdir', type=str, default='../orientation/utils/kitti_eval_offline/label')
    args = parser.parse_args()

    main(args)
