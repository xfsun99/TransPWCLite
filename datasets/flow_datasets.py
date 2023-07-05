import os

import imageio
import numpy as np
import random
from path import Path
from abc import abstractmethod, ABCMeta
from torch.utils.data import Dataset
from utils.flow_utils import load_flow
import scipy.io as scio

class ImgSeqDataset(Dataset, metaclass=ABCMeta):
    def __init__(self, root, n_frames, input_transform=None, co_transform=None,
                 target_transform=None, ap_transform=None):
        self.root = Path(root)
        self.n_frames = n_frames
        self.input_transform = input_transform
        self.co_transform = co_transform
        self.ap_transform = ap_transform
        self.target_transform = target_transform
        self.samples = self.collect_samples()

    @abstractmethod
    def collect_samples(self):
        pass

    def _load_sample(self, s):
        images = s['imgs']
        images = [scio.loadmat(self.root / p)['image'].astype(np.float32) for p in images]
        for i in range(2):
            if images[i].shape[-1] !=3:
                images[i] = images[i][:,:,1:]
        images = [imageio.core.util.Array(image) for image in images]
        target = {}
        if 'flow' in s:
            target['flow'] = load_flow(self.root / s['flow'])

        return images, target

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        images, target = self._load_sample(self.samples[idx])

        if self.co_transform is not None:
            # In unsupervised learning, there is no need to change target with image
            images, _ = self.co_transform(images, {})
        if self.input_transform is not None:
            images = [self.input_transform(i) for i in images]
        data = {'img{}'.format(i + 1): p for i, p in enumerate(images)}

        if self.ap_transform is not None:
            imgs_ph = self.ap_transform(
                [data['img{}'.format(i + 1)].clone() for i in range(self.n_frames)])
            for i in range(self.n_frames):
                data['img{}_ph'.format(i + 1)] = imgs_ph[i]

        if self.target_transform is not None:
            for key in self.target_transform.keys():
                target[key] = self.target_transform[key](target[key])
        data['target'] = target
        return data


class SintelRaw(ImgSeqDataset):
    def __init__(self, root, n_frames=2, transform=None, co_transform=None):
        super(SintelRaw, self).__init__(root, n_frames, input_transform=transform,
                                        co_transform=co_transform)

    def collect_samples(self):
        scene_list = self.root.dirs()
        ids = ['1','2','3','4','5','6','7']
        samples = []
        for scene in scene_list:
            for i in ids:
                img_list = (scene / i).files('*.mat')
                img_list.sort()
                for st in range(0, len(img_list) - self.n_frames + 1,2):
                    seq = img_list[st:st + self.n_frames]
                    sample = {'imgs': [self.root.relpathto(file) for file in seq]}
                    samples.append(sample)
        return samples


class Ultra(ImgSeqDataset):
    def __init__(self, root, n_frames=2, type='us1', split='training',
                 subsplit='trainval', with_flow=True, ap_transform=None,
                 transform=None, target_transform=None, co_transform=None, ):
        self.dataset_type = type
        self.with_flow = with_flow

        self.split = split
        self.subsplit = subsplit
        root = Path(root) / split
        super(Ultra, self).__init__(root, n_frames, input_transform=transform,
                                     target_transform=target_transform,
                                     co_transform=co_transform, ap_transform=ap_transform)

    def collect_samples(self):
        img_dir = self.root / Path(self.dataset_type)
        flow_dir = self.root / 'flow'

        assert img_dir.isdir() and flow_dir.isdir()
        samples = []

        scenes = [i.splitall()[-1] for i in (self.root / img_dir).dirs()]
        ids = ['1','2','3','4','5','6','7']
        for scene in scenes:
            flow_scene = scene[:-3]
            for idx in ids:
                files = (img_dir / scene / idx).glob('*.mat')
                for i in range(0,len(files),self.n_frames):
                    filename = files[i].splitall()[-1]
                    fid = int(filename[0])
                    s = {'imgs': [img_dir / scene / idx / '{:d}.mat'.format(fid + i) for i in
                                      range(self.n_frames)]}
                    try:
                        t = [p.isfile() for p in s['imgs']]
                        assert all([p.isfile() for p in s['imgs']])

                        if self.with_flow:
                            if self.n_frames == 2:
                                s['flow'] = flow_dir / flow_scene / 'flow_{:d}.mat'.format(int((fid+1)/2))
                            else:
                                raise NotImplementedError(
                                    'n_frames {} with flow or mask'.format(self.n_frames))

                            if self.with_flow:
                                assert s['flow'].isfile()
                    except AssertionError:
                        print('Incomplete sample for: {}'.format(s['imgs'][0]))
                        continue
                    samples.append(s)

        return samples