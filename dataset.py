import torch
from armory.data import datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform
import random
import numpy as np
import pdb
import transforms as T
import glob
import os
from torchvision.transforms import functional as F
import pdb


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def randint(low, high):
    if low == high:
        return low
    else:
        return np.random.randint(low, high)



class COCODatasetATPD(Dataset):
    def __init__(self, data_dir, output_size=500, patch_size=100, random_crop=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            random crop the image to the shorter side, and then size to output_size if random_crop
        """
        self.files = glob.glob(os.path.join(data_dir, "*.pt"))
        self.output_size = output_size
        self.patch_size = patch_size
        self.random_crop = random_crop    # whether or not random cropping

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data = torch.load(file)
        x, xmin, ymin = data['x'], data['xmin'], data['ymin']

        # x: np.array 1 h w c
        h = x.shape[1]
        w = x.shape[2]
        img_size = min(h, w)  # take the shorter side as output
        flag = h == img_size   # if h is the shorter side
        if self.random_crop:
            if flag:
                start_h = 0
                start_w = randint(0, w - h)
            else:
                start_h = randint(0, h - w)
                start_w = 0
            image = F.to_tensor(x[0, start_h:start_h + img_size, start_w:start_w + img_size, :])
            image = F.resize(image, [self.output_size, self.output_size])
            xmin = randint(0, self.output_size - self.patch_size)
            ymin = randint(0, self.output_size - self.patch_size)
            mask = np.zeros([image.shape[1], image.shape[2]])
            mask = torch.from_numpy(mask).unsqueeze(0)
            mask_adv = np.zeros([image.shape[1], image.shape[2]])
            mask_adv[ymin:ymin + self.patch_size, xmin:xmin + self.patch_size] = 1
            mask_adv = torch.from_numpy(mask_adv).unsqueeze(0)

        else:
            start_h = 0
            start_w = 0
            image = F.to_tensor(x[0, start_h:start_h + img_size, start_w:start_w + img_size, :])
            image = F.resize(image, [self.output_size, self.output_size])
            xmin = 0
            ymin = 0
            mask = np.zeros([image.shape[1], image.shape[2]])
            mask = torch.from_numpy(mask).unsqueeze(0)
            mask_adv = np.zeros([image.shape[1], image.shape[2]])
            mask_adv[ymin:ymin + self.patch_size, xmin:xmin + self.patch_size] = 1
            mask_adv = torch.from_numpy(mask_adv).unsqueeze(0)

        return image, mask, mask_adv




class COCODatasetAT(Dataset):
    def __init__(self, data_dir, p_clean=0.3, output_size=500, patch_size=100, random_crop=False, random_location=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            random crop the image to the shorter side, and then size to output_size if random_crop
        """
        self.files = glob.glob(os.path.join(data_dir, "*.pt"))
        self.output_size = output_size
        self.patch_size = patch_size
        self.p_clean = p_clean
        self.random_crop = random_crop    # whether or not random cropping
        self.random_location = random_location    # whether or not random cropping

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data = torch.load(file)
        x, y, xmin, ymin = data['x'], data['y'], data['xmin'], data['ymin']

        # x: np.array 1 h w c

        h = x.shape[1]
        w = x.shape[2]
        img_size = min(h, w)  # take the shorter side as output
        flag = h == img_size   # if h is the shorter side
        if self.random_crop:
            if flag:
                start_h = 0
                start_w = randint(0, w - h)
            else:
                start_h = randint(0, h - w)
                start_w = 0
            image = F.to_tensor(x[0, start_h:start_h+img_size, start_w:start_w+img_size, :])
            image = F.resize(image, [self.output_size, self.output_size])
            xmin = randint(0, self.output_size - self.patch_size)
            ymin = randint(0, self.output_size - self.patch_size)
        else:
            image = F.to_tensor(x[0])
            if self.random_location:
                xmin = randint(0, image.shape[2] - self.patch_size)
                ymin = randint(0, image.shape[1] - self.patch_size)

        mask = np.zeros([image.shape[1], image.shape[2]])
        mask = torch.from_numpy(mask).unsqueeze(0)
        mask_adv = np.zeros([image.shape[1], image.shape[2]])
        mask_adv[ymin:ymin+self.patch_size, xmin:xmin+self.patch_size] = 1
        mask_adv = torch.from_numpy(mask_adv).unsqueeze(0)
        return image, mask, mask_adv, xmin, ymin, y


class COCODataset(Dataset):
    def __init__(self, data_dir, p_clean=0.3, output_size=500, patch_size=100, random_crop=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            random crop the image to the shorter side, and then size to output_size if random_crop
        """
        self.files = glob.glob(os.path.join(data_dir, "*.pt"))
        self.output_size = output_size
        self.patch_size = patch_size
        self.p_clean = p_clean
        self.random_crop = random_crop    # whether or not random cropping

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data = torch.load(file)
        x, x_adv, xmin, ymin = data['x'], data['x_adv'], data['xmin'], data['ymin']

        # x: np.array 1 h w c
        xmax = xmin + self.patch_size
        ymax = ymin + self.patch_size
        h = x.shape[1]
        w = x.shape[2]
        img_size = min(h, w)  # take the shorter side as output
        flag = h == img_size   # if h is the shorter side
        if np.random.uniform() < self.p_clean:
            if self.random_crop:
                if flag:
                    start_h = 0
                    start_w = randint(0, w - h)
                else:
                    start_h = randint(0, h - w)
                    start_w = 0
                image = F.to_tensor(x[0, start_h:start_h+img_size, start_w:start_w+img_size, :])
                image = F.resize(image, [self.output_size, self.output_size])
            else:
                image = F.to_tensor(x[0])
            mask = np.zeros([image.shape[1], image.shape[2]])
            mask = torch.from_numpy(mask).unsqueeze(0)
        else:
            if self.random_crop:
                mask = np.zeros([h, w])
                mask[ymin:ymax, xmin:xmax] = 1
                mask = torch.from_numpy(mask).unsqueeze(0)
                if flag:
                    start_h = 0
                    start_w = randint(max(0, xmax - img_size), min(xmin, w-img_size))
                else:
                    start_h = randint(max(0, ymax - img_size), min(ymin, h-img_size))
                    start_w = 0
                image = F.to_tensor(x_adv[0, start_h:start_h + img_size, start_w:start_w + img_size, :])
                image = F.resize(image, [self.output_size, self.output_size])
                mask = mask[:, start_h:start_h + img_size, start_w:start_w + img_size]
                mask = F.resize(mask, [self.output_size, self.output_size])
                mask = (mask > 0.5).float()
            else:
                image = F.to_tensor(x_adv[0])
                mask = np.zeros([image.shape[1], image.shape[2]])
                mask[ymin:ymax, xmin:xmax] = 1
                mask = torch.from_numpy(mask).unsqueeze(0)
        return image, mask


class XViewDatasetATE2E(Dataset):
    def __init__(self, data_dir, patch_size=100, eval=False, output_size=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.files = glob.glob(os.path.join(data_dir, "*.pt"))
        self.patch_size = patch_size
        self.eval = eval
        self.output_size = output_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data = torch.load(file)
        x = data['x']
        y = data['y']

        image = F.to_tensor(x[0])
        # print(image.shape)
        if self.output_size:
            image = F.resize(image, [self.output_size, self.output_size])
        mask = np.zeros([image.shape[1], image.shape[2]])
        mask = torch.from_numpy(mask)
        mask = mask.unsqueeze(0)

        if self.eval:
            xmin = data['xmin']
            ymin = data['ymin']
        else:
            h = x.shape[1]
            w = x.shape[2]
            xmin = np.random.randint(0, h - self.patch_size)
            ymin = np.random.randint(0, w - self.patch_size)

        xmax = xmin + self.patch_size
        ymax = ymin + self.patch_size

        mask_adv = np.zeros([image.shape[1], image.shape[2]])
        mask_adv[ymin:ymax, xmin:xmax] = 1
        mask_adv = torch.from_numpy(mask_adv)
        mask_adv = mask_adv.unsqueeze(0)

        return image, mask, mask_adv, xmin, ymin, y


class XViewDatasetAT(Dataset):
    def __init__(self, data_dir, output_size=None, patch_size=100):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.files = glob.glob(os.path.join(data_dir, "*.pt"))
        self.output_size = output_size
        self.patch_size = patch_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data = torch.load(file)
        x = data['x']

        image = F.to_tensor(x[0])
        if self.output_size:
            image = F.resize(image, [self.output_size, self.output_size])
        mask = np.zeros([image.shape[1], image.shape[2]])
        mask = torch.from_numpy(mask)
        mask = mask.unsqueeze(0)

        h = x.shape[1]
        w = x.shape[2]
        xmin = np.random.randint(0, h - self.patch_size)
        ymin = np.random.randint(0, w - self.patch_size)

        xmax = xmin + self.patch_size
        ymax = ymin + self.patch_size

        if self.output_size:
            scale = self.output_size / float(h)
            xmin = int(np.floor(xmin*scale))
            xmax = int(np.ceil(xmax*scale))
            ymin = int(np.floor(ymin*scale))
            ymax = int(np.ceil(ymax*scale))

        mask_adv = np.zeros([image.shape[1], image.shape[2]])
        mask_adv[ymin:ymax, xmin:xmax] = 1
        mask_adv = torch.from_numpy(mask_adv)
        mask_adv = mask_adv.unsqueeze(0)

        return image, mask, mask_adv, xmin, ymin



class XViewDataset(Dataset):
    def __init__(self, data_dir, p_clean=0.3, output_size=500, patch_size=100):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.files = glob.glob(os.path.join(data_dir, "*.pt"))
        self.output_size = output_size
        self.patch_size = patch_size
        self.p_clean = p_clean

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data = torch.load(file)
        x, x_adv, xmin, ymin = data['x'], data['x_adv'], data['xmin'], data['ymin']
        if np.random.uniform() < self.p_clean:
            image = F.to_tensor(x[0])
            image = F.resize(image, [self.output_size, self.output_size])
            mask = np.zeros([self.output_size, self.output_size])
            mask = torch.from_numpy(mask)
            mask = mask.unsqueeze(0)
        else:
            h = x_adv.shape[1]
            w = x_adv.shape[2]
            scale_h = self.output_size / float(h)
            scale_w = self.output_size / float(w)
            xmax = xmin + self.patch_size
            ymax = ymin + self.patch_size

            xmin = int(np.floor(xmin*scale_w))
            xmax = int(np.ceil(xmax*scale_w))
            ymin = int(np.floor(ymin*scale_h))
            ymax = int(np.ceil(ymax*scale_h))

            mask = np.zeros([self.output_size, self.output_size])
            mask[ymin:ymax, xmin:xmax] = 1
            mask = torch.from_numpy(mask)
            mask = mask.unsqueeze(0)

            image = F.to_tensor(x_adv[0])
            image = F.resize(image, [self.output_size, self.output_size])
        return image, mask



class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size=500):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        targets = sample['target']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        target = targets
        target['area'] = target['area'] * new_h * new_w / (h * w)
        target['boxes'][:, 0] = target['boxes'][:, 0] * new_h / h
        target['boxes'][:, 1] = target['boxes'][:, 1] * new_w / w
        target['boxes'][:, 2] = target['boxes'][:, 2] * new_h / h
        target['boxes'][:, 3] = target['boxes'][:, 3] * new_w / w
        #print(image.shape)
        return {'image': image, 'target': target}


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        x = sample['image']
        y = sample['target']
        if random.uniform(0, 1) <= 2:#self.p:
            #pdb.set_trace()
            w = x.shape[1]
            #print(x.shape)
            #print(w)
            image = np.fliplr(x).copy()
            targets = y
            targets['boxes'][:, 1] = w - y['boxes'][:, 3]
            targets['boxes'][:, 3] = w - y['boxes'][:, 1]
        else:
            image = x
            targets = y
        return {'image': image, 'target': targets}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        x = sample['image']
        y = sample['target']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = x.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        target = {'boxes': torch.from_numpy(y['boxes']), 'area': torch.from_numpy(y['area']),
                  'id': torch.from_numpy(y['id']), 'image_id': torch.from_numpy(y['image_id']),
                  'labels': torch.from_numpy(y['labels']),
                  'iscrowd': torch.zeros((y['id'].shape[0]), dtype=torch.int64)}

        return {'image': image, 'target': target}

