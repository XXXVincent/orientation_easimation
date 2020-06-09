import os
import cv2
import math
def visual(img_path, gt_path,pred_path):

    img_list=sorted(os.listdir(img_path))
    obj_finished=0
    for i,img in enumerate(img_list):
        img_id=img.split('.')[0]
        pred_txt=os.path.join(pred_path,img_id+'.txt')
        gt_txt=os.path.join(gt_path,img_id+'.txt')
        labels_pred=get_label(pred_txt)
        labels_gt=get_label(gt_txt)
        for j,label_pred in enumerate(labels_pred):
            label_pred=label_pred.split(' ')
            bbx_pred=label_pred[4:8]
            rot_pred = label_pred[-3]
            alpha_pred=label_pred[3]
            for k,label_gt in enumerate(labels_gt):
                label_gt=label_gt.split(' ')
                bbx_gt=label_gt[4:8]
                if iou(bbx_pred,bbx_gt,thresh=0.7):
                    if label_gt[-1]!='\n':
                        rot_gt = label_gt[-2]
                    else:
                        rot_gt=label_gt[-3]
                    alpha_gt=label_gt[3]
                    break
                else:
                    rot_gt=' '
                    alpha_gt=' '
            draw(img_path+img,bbx_pred,alpha_pred,alpha_gt,rot_pred,rot_gt)
            obj_finished+=1
            print('  object %d finished.'%obj_finished)
        print('image %s finished.'%img_id)

    return

def get_label(txt_path):
    with open(txt_path,'r') as f:
        labels=f.readlines()
        f.close()

    return labels

def iou(bbx_pred, bbx_gt,thresh):
    x1_pred = float(bbx_pred[0])
    y1_pred = float(bbx_pred[1])
    x2_pred = float(bbx_pred[2])
    y2_pred = float(bbx_pred[3])
    x1_gt = float(bbx_gt[0])
    y1_gt = float(bbx_gt[1])
    x2_gt = float(bbx_gt[2])
    y2_gt = float(bbx_gt[3])

    pred_w=x2_pred-x1_pred
    pred_h=y2_pred-y1_pred
    gt_w=x2_gt-x1_gt
    gt_h=y2_gt-y1_gt

    x1,y1 = max(x1_pred,x1_gt),max(y1_pred,y1_gt)
    x2,y2 = min(x2_pred,x2_gt),min(y2_pred,y2_gt)


    inter_w=max(0,x2-x1)
    inter_h=max(0,y2-y1)
    inter_area = inter_w * inter_h
    union_area = (pred_w * pred_h + 1e-16) + gt_w * gt_h - inter_area

    return (inter_area / union_area) >= thresh



def draw(path,bbx,alpha_pred,alpha_gt,rot_pred,rot_gt):
    x1 = int(float(bbx[0]))
    y1 = int(float(bbx[1]))
    x2 = int(float(bbx[2]))
    y2 = int(float(bbx[3]))
    # rot_pred=float(rot_pred)
    # alpha_pred=float(alpha_pred)
    # rot_pred=round(rot_pred,2)
    # alpha_pred=round(alpha_pred,2)
    '''cv2'''
    img=cv2.imread(path)


    # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    # text = 'rot: %f gt: %s alpha: %f gt: %s' % (rot_pred, rot_gt,alpha_pred,alpha_gt)
    text_1 = 'alpha: %.2f' % (float(alpha_pred)/math.pi*180)
    text_2 =  'gt: %.2f'  %(float(alpha_gt)/math.pi*180)
    cv2.rectangle(img, (x1,y1), (x2, y2), (255,255,0), 1)
    cv2.putText(img, text_1, (x1, y1-15), font, 0.5, (0, 255, 255), 1)
    cv2.putText(img, text_2, (x1, y1), font, 0.5, (0, 255, 255), 1)

    cv2.imwrite(path, img)

def main():
    img_path = r'C:\Users\vincent.xu\Desktop\orientation\orientation_easimation\data\image_viz\\'
    gt_path = r'C:\Users\vincent.xu\Desktop\orientation\orientation_easimation\data\label'
    pred_path = r'C:\Users\vincent.xu\Desktop\orientation\orientation_easimation\utils\kitti_eval_offline\results\test_set'
    # save_path = r'C:\Users\vincent.xu\Desktop\orientation\dataset\kitti_dataset\label_check\front'
    visual(img_path, gt_path, pred_path)

if __name__=='__main__':
    main()







