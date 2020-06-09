import os
import shutil

dataset_list = r'C:\Users\vincent.xu\Desktop\orientation\dataset\fisheye\right_match_list.txt'
image_dir = r'I:\Public\Beijing\wjz\20200521\right_new\right'
label_dir = r'C:\Users\vincent.xu\Desktop\orientation\dataset\right\label'
save_dir = r'C:\Users\vincent.xu\Desktop\orientation\dataset\fisheye\right'

with open(dataset_list, 'r') as f:
    dataset = f.readlines()

for line in dataset:
    img_file_name = line.split(',')[-1].split('=')[-1].strip('\n')
    old_label_file_name = line.split(',')[0].split('=')[-1].replace('jpg', 'txt')
    img_file_name = old_label_file_name.replace('txt', 'bmp')
    new_label_file_name = img_file_name.replace('jpg', 'txt')
    try:
        shutil.copyfile(os.path.join(image_dir, img_file_name), os.path.join(save_dir, img_file_name))
        # shutil.copyfile(os.path.join(label_dir, old_label_file_name), os.path.join(save_dir, 'label', new_label_file_name))
        # print('Finished: ', old_label_file_name, '->', new_label_file_name)
    except FileNotFoundError:
        print('no matched file found-> ', img_file_name)

