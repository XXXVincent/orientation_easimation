import torch
import os
from utils import dataset
from torch.utils.data import DataLoader
import numpy as np
from config import _C as cfg
from utils.model import MobileNetV2,Model
from utils import angle_utils
from PIL import Image
from PIL import ImageFile, ImageDraw, ImageFont

PI = 3.14159
P2 = [[712., 0, 640.], [0, 712., 360.], [0., 0., 1.]]



def main():

    device=cfg.DEVICE
    if cfg.DEBUG:
        data = dataset.KITTIDataset(cfg.PATH, cfg, mode='predict', save_debug_pic=True)
    else:
        data = dataset.KITTIDataset(cfg.PATH,cfg,mode='predict')
    val_dataloader = DataLoader(data, cfg.VAL_BATCH , shuffle=False)

    # Initialize the model
    mobilenetv2 = MobileNetV2()
    model = Model(features=mobilenetv2.features, bins=cfg.BIN).to(device)
    model_list = os.listdir(cfg.MODEL_DIR)
    # model.load_state_dict(torch.load(cfg.MODEL_DIR + "/%s" % sorted(model_list)[-1]))

    model.load_state_dict(torch.load(cfg.MODEL_DIR + '/2binmodel_2020-06-04-10-38-19.pth', map_location=torch.device('cpu')))
    print(sorted(model_list)[-1])
    model.eval()
    if cfg.DEBUG:
        debug_count = 0
        debug_save_dir = r'C:\Users\vincent.xu\Desktop\orientation\orientation_easimation\data\debug\compare'
        error_pred_dir = r'C:\Users\vincent.xu\Desktop\orientation\orientation_easimation\data\debug\error'
        for i, (batch,info, debug_pic, gt_conf) in enumerate(val_dataloader):
            # batch, centerAngle, info = data.EvalBatch()
            batch = torch.FloatTensor(batch).to(device)

            [orient, conf] = model(batch)
            orient = orient.cpu().data.numpy()
            conf = conf.cpu().data.numpy()
            alpha = angle_utils.recover_angle(conf.squeeze(0), orient.squeeze(0), cfg.BIN)
            rot_y = angle_utils.compute_orientaion(P2, info['bbox2d'], alpha)
            angle_utils.save_result(info, alpha, rot_y, i,mode='predict')
            debug_pic = Image.fromarray(debug_pic[0].numpy()).convert('RGB')
            draw = ImageDraw.Draw(debug_pic)
            # draw.text((5, 5), 'GT_bin:' + str(np.argmax(conf)), fill=(255, 0, 0))
            draw.text((5, 15), 'pred_bin:' + str(np.argmax(conf)), fill=(255, 0, 0))

            debug_pic.save(os.path.join(debug_save_dir, str(debug_count).zfill(6) + '.jpg'))
            if gt_conf != np.argmax(conf):
                debug_pic.save(os.path.join(error_pred_dir, str(debug_count).zfill(6) + '.jpg'))

            debug_count+=1

    else:
        for i, (batch, info) in enumerate(val_dataloader):
            # batch, centerAngle, info = data.EvalBatch()
            batch = torch.FloatTensor(batch).to(device)

            [orient, conf] = model(batch)
            orient = orient.cpu().data.numpy()
            conf = conf.cpu().data.numpy()
            alpha = angle_utils.recover_angle(conf.squeeze(0), orient.squeeze(0), cfg.BIN)
            rot_y = angle_utils.compute_orientaion(P2, info['bbox2d'], alpha)
            angle_utils.save_result(info, alpha, rot_y, i, mode='predict')



if __name__ == '__main__':
    main()
