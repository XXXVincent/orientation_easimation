import os
import csv
import random
import copy
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

from utils import angle_utils

TYPE_ID_CONVERSION = {
    # 'Car': 1,
    # 'Cyclist': 2,
    # 'Pedestrian': 3,
    'car': 1,
}

PI = 3.14159


class KITTIDataset(Dataset):
    def __init__(self, root, cfg, mode):
        self.mode=mode
        self.mean = [102.9801, 115.9465, 122.7717]
        self.std = [1., 1., 1.]
        if mode !='predict':
            self.image_path = os.path.join(root, 'image')
            self.label_path = os.path.join(root, 'label')
        #eval
        else:
            self.image_path = os.path.join(root, 'test_debug', 'image')
            self.label_path = os.path.join(root, 'test_debug', 'label')

        self.image_list = sorted(os.listdir(self.image_path))
        self.label_list = sorted(os.listdir(self.label_path))
        assert len(self.image_list) == len(self.label_list), ('img:', self.image_list, 'label: ', self.label_list)


        self.classes = TYPE_ID_CONVERSION

        self.bins = cfg.BIN
        self.overlapping = cfg.OVERLAPPING
        self.jitter = cfg.JITTER
        # we need to load all annotations into buffer at first
        self.annotations = self.load_annotations()
        self.length = len(self.annotations)
        print('datalen: %d'%self.length)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        anno = copy.deepcopy(self.annotations[idx])
        img = Image.open(os.path.join(self.image_path, anno['image']))
        bbox2d = anno['bbox2d']
        cropped_box = self.crop_image(img, bbox2d,self.mode)
        if self.mode=='train':
            alpha = angle_utils.check_angle(anno['alpha'])
            new_alpha = angle_utils.get_new_alpha(alpha)
            if random.random() < 0.5:
                cropped_box = F.hflip(cropped_box)
                new_alpha = 2 * PI - new_alpha

            cropped_box = F.to_tensor(cropped_box)[[2, 1, 0]] * 255
            cropped_box = F.normalize(cropped_box, mean=self.mean, std=self.std)
            new_alpha = torch.tensor(new_alpha)
            confidence = torch.zeros(self.bins)
            angle_offset = torch.zeros(self.bins, 2)
            anchors = angle_utils.compute_anchors(new_alpha, self.bins, self.overlapping)
            for anchor in anchors:
                angle_offset[anchor[0], :] = torch.tensor(
                    [torch.cos(anchor[1]), torch.sin(anchor[1])])
                confidence[anchor[0]] = 1.
                confidence /= confidence.sum()

            return cropped_box, dict(confidence=confidence,
                                     angle_offset=angle_offset, )
        else:
            cropped_box = F.to_tensor(cropped_box)[[2, 1, 0]]*255
            cropped_box = F.normalize(cropped_box, mean=self.mean, std=self.std)
            return cropped_box, anno

    def load_annotations(self):
        """
        change the range of orientation from [-pi, pi] to [0, 2pi]
        :param alpha: original orientation in KITTI
        :return: new alpha
        """
        result = []
        for number, label_file in enumerate(self.label_list):
            field_names = ['type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'dh', 'dw',  #!
                           'dl', 'lx', 'ly', 'lz', 'ry']
            with open(os.path.join(self.label_path, label_file), 'r') as csv_file:
                reader = csv.DictReader(csv_file, delimiter=' ', fieldnames=field_names)  # delimiter：分割符

                for line, row in enumerate(reader):
                    if row['type'] in self.classes:
                        x1 = float(row['xmin'])
                        x2 = float(row['xmax'])
                        y1 = float(row['ymin'])
                        y2 = float(row['ymax'])

                        dimensions = [float(row['dh']), float(row['dw']), float(row['dl'])]
                        locations = [float(row['lx']), float(row['ly']), float(row['lz'])]
                        if x2 < x1:
                            raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
                        if y2 < y1:
                            raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

                        result.append({'image': self.image_list[number],
                                       'bbox2d': [x1, y1, x2, y2],
                                       'class': row['type'],  # class name
                                       'label': self.classes[row['type']],
                                       'alpha': float(row['alpha']),  # label of class
                                       'dimensions': dimensions,  # 3D size
                                       'locations': locations,  # 3D locations
                                       'rot_y': float(row['ry']),  # global rotation
                                       'scale': 1.  # scale of input image, default 1
                                       })

        return result

    def crop_image(self, img, bbox2d, mode):
        if mode=='train':
            xmin = bbox2d[0] + random.randint(-self.jitter, self.jitter + 1)
            ymin = bbox2d[1] + random.randint(-self.jitter, self.jitter + 1)
            xmax = bbox2d[2] + random.randint(-self.jitter, self.jitter + 1)
            ymax = bbox2d[3] + random.randint(-self.jitter, self.jitter + 1)

            w, h = img.size

            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(w - 1, xmax), min(h - 1, ymax)
        else:
            xmin = bbox2d[0]
            ymin = bbox2d[1]
            xmax = bbox2d[2]
            ymax = bbox2d[3]
        cropped_box = F.crop(img, ymin, xmin, ymax - ymin + 1, xmax - xmin + 1)
        cropped_box = F.resize(cropped_box, size=(56, 56))

        return cropped_box




# if __name__ == '__main__':
    # from torch.utils.data import DataLoader
    #
    # root = '/home/user/Desktop/zechen_laptop/home/user/github_project/orientation/data/kitti/training'
    # cfg = {'bins': 2,
    #        'overlapping': 10,
    #        'jitter': 3,
    #        'batches': 4, }
    # data_set = KITTIDataset(root, cfg)
    # dataloader = DataLoader(data_set, batch_size=4, shuffle=True)
    #
    # for idx, (inputs, labels) in enumerate(dataloader):
    #     print(inputs)
    #     print(labels)
