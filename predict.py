import torch
import os
from utils import dataset
from torch.utils.data import DataLoader

from config import _C as cfg
from utils.model import MobileNetV2,Model
from utils import angle_utils


PI = 3.14159
P2 = [[712., 0, 640.], [0, 712., 360.], [0., 0., 1.]]



def main():

    device=cfg.DEVICE

    data = dataset.KITTIDataset(cfg.PATH,cfg,mode='predict')
    val_dataloader = DataLoader(data, cfg.VAL_BATCH , shuffle=False)

    # Initialize the model
    mobilenetv2 = MobileNetV2()
    model = Model(features=mobilenetv2.features, bins=cfg.BIN).to(device)
    model_list = os.listdir(cfg.MODEL_DIR)
    # model.load_state_dict(torch.load(cfg.MODEL_DIR + "/%s" % sorted(model_list)[-1]))

    model.load_state_dict(torch.load(cfg.MODEL_DIR + '/model_2020-05-13-10-06-12.pth'))
    print(sorted(model_list)[-1])
    model.eval()

    for i, (batch,info) in enumerate(val_dataloader):
        # batch, centerAngle, info = data.EvalBatch()
        batch = torch.FloatTensor(batch).to(device)

        [orient, conf] = model(batch)
        orient = orient.cpu().data.numpy()
        conf = conf.cpu().data.numpy()
        print(orient)
        print(conf)

        alpha = angle_utils.recover_angle(conf.squeeze(0), orient.squeeze(0), cfg.BIN)
        rot_y = angle_utils.compute_orientaion(P2, info['bbox2d'], alpha)
        angle_utils.save_result(info, alpha, rot_y, i,mode='predict')



if __name__ == '__main__':
    main()
