import cv2
import os
import math
img_dir = r'C:\Users\vincent.xu\Desktop\orientation\dataset\visualized\front\image'
label_dir = r'C:\Users\vincent.xu\Desktop\orientation\dataset\visualized\front\label'
save_dir = img_dir
# save_dir = r'C:\Users\vincent.xu\Desktop\temp\vis\car_large_vis'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
font = cv2.FONT_HERSHEY_SIMPLEX

def read_rot_angles(txt_file):
    with open(txt_file, 'r') as f:
        content = f.readlines()
    anno = []
    for line in content:
        line_content = line.split(' ')
        obj_bbox = line_content[4:8]
        obj_relativ_angle = line_content[3]
        obj_global_angle = line_content[-2]
        obj_bbox.append(obj_relativ_angle)
        obj_bbox.append(obj_global_angle)
        anno.append(obj_bbox)
    return anno
for file in os.listdir(img_dir):
    file_series = file.split('.')[0]
    img = cv2.imread(os.path.join(img_dir, file))
    file_annos = read_rot_angles(os.path.join(label_dir, file_series+'.txt'))
    for anno in file_annos:
        anno = list(map(eval, anno))
        bbox = list(map(int, anno[0:4]))
        relative_angle = anno[4]/math.pi*180
        global_angle = anno[5]/math.pi*180
        imgzi = cv2.rectangle(img, (bbox[2], bbox[3]), (bbox[0], bbox[1]), (0, 255, 0), 1)
        # imgzi = cv2.putText(img, , (50, 300), font, 1.2, (255, 255, 255), 2)
        imgzi = cv2.putText(img,'angle_r: '+"{:.3f}".format(relative_angle), (bbox[0], bbox[1]+5),cv2.FONT_HERSHEY_SIMPLEX,0.4, (255, 255, 0) )
        imgzi = cv2.putText(img, 'angle_g: '+"{:.3f}".format(global_angle), (bbox[0], bbox[1]+15),cv2.FONT_HERSHEY_SIMPLEX,0.4, (255, 255, 0) )
    cv2.imwrite(os.path.join(save_dir, file), imgzi)



def read_rot_angles(txt_file):
    with open(txt_file, 'r') as f:
        content = f.readlines()
    anno = []
    for line in content:
        line_content = line.split(' ')
        obj_bbox = line_content[4:8]
        obj_relativ_angle = line_content[3]
        obj_global_angle = line_content[-2]
        obj_bbox.append(obj_relativ_angle)
        obj_bbox.append(obj_global_angle)
        anno.append(obj_bbox)
    return anno
