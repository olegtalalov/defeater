import os
import numpy as np
import torch
import torch.utils.data as tdata
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms, datasets


class MinPointSampler(object):
    """Transfromation that samples pts_num number of points from the input point cloud"""

    def __init__(self, pts_num, replace_flag=False):
        self.pts_num = pts_num
        self.replace_flag = replace_flag

    def __call__(self, point_cloud):
        return point_cloud[np.random.choice(point_cloud.shape[0], size=self.pts_num, replace=self.replace_flag), :]

class PointScaler(object):
    """Scales point cloud to a new one with mean = 0 and maximum vector length of 1"""
    def __call__(self, point_cloud):
        return (point_cloud - np.mean(point_cloud, axis=0)) / np.linalg.norm(point_cloud, axis=1).max()


class RndPointsAugmentations(object):
    """
    Performs 2 types of point data augmentation:
    Random jittering via uniformly distributed noise and random rotation along z-axis
    """
    def __init__(self, jitter_a=0, jitter_b=0.2):
        self.jitter_a = jitter_a
        self.jitter_b = jitter_b

    def __call__(self, point_cloud):
        theta = np.random.uniform(0, np.pi*2)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        point_cloud[:, :2] = point_cloud[:, :2].dot(rotation_matrix)
        return point_cloud + np.random.normal(self.jitter_a, self.jitter_b, size=point_cloud.shape)


def create_train_val_data_loaders(data_dir, *, min_pts=75, batch_size=32, validation_frac=0.2, num_of_workers=0):
    """
    Return pair of pytorch dataloaders for train and validation sets.
    """
    # sample => scale => (if train) random jitter and random rotation along z axis => transform to Pytorch tensor

    mps_transform = MinPointSampler(min_pts, replace_flag=True)
    pt_scaler = PointScaler()
    points_aug = RndPointsAugmentations(jitter_b=0.3)

    train_transforms = transforms.Compose([
                                           mps_transform,
                                           points_aug,
                                           pt_scaler,
                                           transforms.ToTensor()
                                          ])

    val_transforms = transforms.Compose([
                                        mps_transform,
                                        pt_scaler,
                                        transforms.ToTensor()
                                       ])

    train_data = datasets.DatasetFolder(data_dir, loader=lambda x: np.load(x).astype(np.float32), extensions=("npy"),
                                        transform=train_transforms)

    val_data = datasets.DatasetFolder(data_dir, loader=lambda x: np.load(x).astype(np.float32), extensions=("npy"),
                                      transform=val_transforms)

    dataset_len = len(train_data)

    indices = np.arange(dataset_len)

    val_abs_size = np.int(np.floor(validation_frac * dataset_len))

    np.random.shuffle(indices)

    train_id, val_id = indices[val_abs_size:], indices[:val_abs_size]

    all_dataset = train_data.samples.copy()
    all_targets = train_data.targets.copy()

    train_data.samples = [all_dataset[i] for i in train_id]
    train_data.targets = [all_targets[i] for i in train_id]

    train_weight = 1 / np.array([(np.array(train_data.targets) == tgt).sum() for tgt in np.unique(train_data.targets)])

    train_samples_weight = torch.tensor([train_weight[tgt] for tgt in train_data.targets])

    train_sampler = tdata.WeightedRandomSampler(train_samples_weight, len(train_samples_weight))

    train_loader = tdata.DataLoader(train_data,
                                    sampler=train_sampler, batch_size=batch_size, num_workers=num_of_workers, drop_last=True)

    if validation_frac > 0:
        val_data.samples = [all_dataset[i] for i in val_id]
        val_data.targets = [all_targets[i] for i in val_id]

        # readjust probabilities for unbalanced classes

        val_weight = 1 / np.array([(np.array(val_data.targets) == tgt).sum() for tgt in np.unique(val_data.targets)])


        val_samples_weight = torch.tensor([val_weight[tgt] for tgt in val_data.targets])

        val_sampler = tdata.WeightedRandomSampler(val_samples_weight, len(val_samples_weight))

        val_loader = tdata.DataLoader(val_data,
                                        sampler=val_sampler, batch_size=batch_size, num_workers=num_of_workers, drop_last=True)
    else:
        val_loader = None

    return train_loader, val_loader


class FilterClassDataset(datasets.vision.VisionDataset):
    """Adapted DatasetFolder class that allows to filter class folders"""
    def __init__(self, root, loader, extensions=None, transform=None, target_transform=None, is_valid_file=None, cls_to_filter=["TYPE_PEDESTRIAN"]):
        super(FilterClassDataset, self).__init__(root, transform=transform,
                                                 target_transform=target_transform)
        self.loader = loader

        self.extensions = extensions
        self.classes = cls_to_filter
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.samples = self.make_dataset(root, self.class_to_idx, extensions=self.extensions)
        self.targets = [s[1] for s in self.samples]

    def make_dataset(self, dir, class_to_idx, extensions=None, is_valid_file=None):
        fn_cls_pair = []
        dir = os.path.expanduser(dir)
        if not ((extensions is None) ^ (is_valid_file is None)):
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
        if extensions is not None:
            def is_valid_file(x):
                return x.lower().endswith(extensions)
        for target in sorted(class_to_idx.keys()):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = (path, class_to_idx[target])
                        fn_cls_pair.append(item)

        return fn_cls_pair

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)


def track_dataloader(data_dir, min_pts=75, batch_size=32, num_of_workers=0, cls_to_filter=["TYPE_PEDESTRIAN"], drop_last=False):
    """Dataloader for descriptor construction"""
    mps_transform = MinPointSampler(min_pts, True)
    pt_scaler = PointScaler()

    data_transforms = transforms.Compose([
                                          mps_transform,
                                          pt_scaler,
                                          transforms.ToTensor()
                                      ])

    data = FilterClassDataset(data_dir, loader=lambda x: np.load(x, mmap_mode="r"),
                                                                    extensions=("npy"),
                                        transform=data_transforms, cls_to_filter=cls_to_filter)

    seq_sampler = tdata.SequentialSampler(data)

    loader = tdata.DataLoader(data,
                   sampler=seq_sampler, batch_size=batch_size, drop_last=drop_last, num_workers=num_of_workers)

    return loader
