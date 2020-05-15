
import os
import json
import argparse
import random
import numpy as np

import torch
import torch.utils.data as tdata
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms, datasets

import pointnet_model
import data_loader
import descriptors_set

def training_loop(source_folder, device, adam_parameters, shelduler_parameters, batch_size=32, epoch_number=10, validation_frac=0.2, val_step=10, reg_lambda=0.001
                  min_pts=50, cls_num=2, descriptor_size=128, pointnet_feature_transform, save_model_to):
    """
        source_folder: folder with training examples (in .npy format)

        validation_frac: what part of dataset should be used for validation( float in (0, 1) )

        min_pts: minimun number of points in bounding box

        cls_num: number of classes (should match number of folders in source_folder)

        descriptor_size:
    """
    point_net_cls = pointnet_model.PointNetCls(k=cls_num, descriptor_size=descriptor_size,
                                               feature_transform=pointnet_feature_transform)

    train_loader, val_loader = data_loader.create_train_val_data_loaders(source_folder, num_of_workers=2, batch_size=batch_size
                                                                         validation_frac=validation_frac, min_pts=min_pts)

    best_scores_descriptors = descriptors_set.DescriptorsSet(max_number=30,
                                                             descriptor_size=descriptor_size)

    if existing_model_path != "":
        point_net_cls.load_state_dict(torch.load(existing_model_path))

    optimizer = optim.Adam(point_net_cls.parameters(), **adam_parameters)

    scheduler = optim.lr_scheduler.StepLR(optimizer, **shelduler_parameters)

    point_net_cls.cuda()

    num_batches = len(train_loader)

    for epoch in range(epoch_number):
        scheduler.step()

        for i, [points, target] in enumerate(train_loader):

            points = points.squeeze().transpose(2, 1)

            points, target = points.to(device, dtype=torch.float), target.to(device) # tensors to GPU

            optimizer.zero_grad()

            point_net_cls = point_net_cls.train()

            pred, trans, trans_feat, descriptors = point_net_cls(points)

            loss = F.nll_loss(pred, target)

            if pointnet_feature_transform:
                loss += feature_transform_regularizer(trans_feat) * reg_lambda

            loss.backward()

            optimizer.step()

            pred_choice = pred.data.max(1)[1]

            correct = pred_choice.eq(target.data).cpu()

            best_scores_descriptors.add_batch(np.exp(pred.cpu().detach().numpy())[:, 1],
                                              correct.numpy(), descriptors.cpu().numpy())

            print('[epoch #{0}: {1}/{2}] train loss: {3} batch accuracy: {4}'.format(epoch, i, num_batches, loss.item(), correct.sum().item() / float(batch_size)))

            if i % val_step == 0:
                j, [points, target] = next(enumerate(val_loader))

                points = points.squeeze().transpose(2, 1)

                points, target = points.cuda(), target.cuda()

                point_net_cls = point_net_cls.eval()

                pred, _, _, _ = point_net_cls(points)

                loss = F.nll_loss(pred, target)

                pred_choice = pred.data.max(1)[1]

                correct = pred_choice.eq(target.data).cpu().sum()

                recall = pred_choice[target == 1].cpu().sum().item() / ((target == 1).cpu().sum().item() + 1e-10)

                precision = pred_choice[target == 1].cpu().sum().item() / ((pred_choice == 1).cpu().sum().item() + 1e-10)

                print(
                      '[epoch #{0}: {1}/{2}] validation loss: {3} batch accuracy: {4}, batch recall: {5}, batch precision: {6}'.format(
                      epoch, i, num_batches, loss.item(), correct.item()/float(batch_size), recall, precision
                      )
                     )
        torch.save(point_net_cls.state_dict(), os.path.join(save_model_to, "cls_model_new_{0}.pth".format(epoch)))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()

    parser.add_argument('--source_path', required=True, help='Source path with training data')
    parser.add_argument('--model_save_path', required=True, help='Path to folder where model checkpoints will be saved')

    parser.add_argument('--validation_frac', type=float, defaut=0.2, help='Relative size of validation subset to whole dataset')
    parser.add_argument('--epoch_number', type=int, default=10, help='Number of epoch to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Size of batch')
    parser.add_argument('--descriptor_size', type=int, default=128, help='Length of PointNet descriptor')
    parser.add_argument('--pointnet_feature_transform', action='store_true', help='Should use transformed features from intermediate layers')
    parser.add_argument('--manual_seed', type=int, defaut=42, help='Random seed for training process')
    parser.add_argument('--min_pts', type=int, defaut=50, help='Minimal points in bounding box')
    parser.add_argument('--val_step', type=int, defaut=10, help='Performs testing on validation subset every val_step batches')
    parser.add_argument('--reg_lambda', type=float, defaut=0.001, help='Regularization coefficient for transformed features')
    parser.add_argument('--adam_parameters', type=json.loads, default={"lr": 0.001, "betas" : [0.9, 0.999]},
                        help='Parameters for ADAM optimizer (lr and betas) e.g. {"lr": 0.001, "betas" : [0.9, 0.999]}')
    parser.add_argument('--shelduler_parameters', type=json.loads, default= {"step_size": 20, "gamma": 0.5},
                        help='Parameters for optimizer lr shelduler'
                        )

    args = parser.parse_args()

    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)

    training_loop(
                  source_folder=args.source_path,
                  device=device,
                  validation_frac=args.validation_frac,
                  adam_parameters=args.adam_parameters,
                  shelduler_parameters=args.shelduler_parameters,
                  min_pts=args.min_pts,
                  cls_num=len(os.listdir(source_folder)),
                  epoch_number=args.epoch_number,
                  val_step=args.val_step,
                  descriptor_size=args.descriptor_size,
                  reg_lambda=args.reg_lambda,
                  pointnet_feature_transform=args.pointnet_feature_transform,
                  batch_size=args.batch_size,
                  save_model_to=args.model_save_path
                 )


if __name__=="__main__":
    main()
