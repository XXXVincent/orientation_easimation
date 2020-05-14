import os
import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from config import _C as cfg
from utils import dataset
from utils.model import (
    MobileNetV2,
    Model,
    binary_cross_entropy_one_hot,
    OrientationLoss
)
def train(cfg):

    #load data
    train_dataset = dataset.KITTIDataset(root=cfg.PATH, cfg=cfg,mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.TRAIN_BATCH, shuffle=True)

    # Initialize the model
    mobilenetv2 = MobileNetV2()
    model = Model(features=mobilenetv2.features, bins=cfg.BIN).to(cfg.DEVICE)

    # If specified we start from pth
    model_list = [x for x in sorted(os.listdir(cfg.MODEL_DIR)) if x.endswith(".pth")]
    # model_list=[]
    if not model_list:
        print("No previous model found, start training!")
        mobilenetv2_model = torch.load('./model/mobilenet_v2.pth.tar')
        mobilenetv2.load_state_dict(mobilenetv2_model)
    else:
        print("Find previous model %s" % model_list[-1])
        # model.load_state_dict(torch.load(cfg.MODEL_DIR + "/%s" % model_list[-1], map_location=torch.device(cfg.DEVICE)))


    opt_SGD = torch.optim.SGD(model.parameters(), lr=0.001)

    writer = SummaryWriter(cfg.LOG_DIR )
    for i in range(cfg.EPOCH):
        model.train()
        print('Epoch %d'%i)
        loss_epoch = 0
        conf_loss_epoch = 0
        orient_loss_epoch = 0
        for batch_idx, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(cfg.DEVICE)
            confidence = labels['confidence'].to(cfg.DEVICE)
            angle_offset = labels['angle_offset'].to(cfg.DEVICE)

            [orient, conf] = model(inputs)

            conf_loss = binary_cross_entropy_one_hot(conf, confidence)
            orient_loss = OrientationLoss(orient, angle_offset)
            loss = conf_loss + cfg.WEIGHT* orient_loss
            loss_epoch +=loss
            conf_loss_epoch += conf_loss
            orient_loss_epoch += orient_loss

            opt_SGD.zero_grad()
            loss.backward()
            opt_SGD.step()

            print('Batch %d' % batch_idx)
            print('Training loss: ', loss.item())
            print('Confidence loss: ', conf_loss.item())
            print('Orientation loss: ', orient_loss.item())

        loss_epoch=loss_epoch/(batch_idx + 1)
        conf_loss_epoch=conf_loss_epoch/(batch_idx + 1)
        orient_loss_epoch=orient_loss_epoch/(batch_idx + 1)

            # print('Batch %d'%batch_idx)
            # print('Training loss: ', loss.item())
            # print('Confidence loss: ', conf_loss.item())
            # print('Orientation loss: ', orient_loss.item())
        writer.add_scalar('Training loss: ', loss_epoch.item(), i)
        writer.add_scalar('Confidence loss: ', conf_loss_epoch.item(), i)
        writer.add_scalar('Orientation  loss: ', orient_loss_epoch.item(), i)

        #log process
        pass



        if i % 10 == 0:
            now = datetime.datetime.now()
            now_s = now.strftime("%Y-%m-%d-%H-%M-%S")
            name = cfg.MODEL_DIR + "/{}_model_{}.pth".format(cfg.BIN,now_s)
            torch.save(model.state_dict(), name)


    return


def main():
    # output_dir = cfg.OUTPUT_DIR
    # if output_dir:
    #     mkdir(output_dir)
    # logger = setup_logger("deep3dbox", output_dir)
    # logger.info("Loaded configuration:\n{}".format(cfg))

     train(cfg)

if __name__ == '__main__':
    main()
