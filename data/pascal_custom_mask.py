#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json, os, random
from collections import defaultdict
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import PIL
from imutils import paths

from cocoapi.layout2im.utils.data import imagenet_preprocess, Resize
from torch.utils.data import DataLoader


class PascalVocSceneGraphDataset(Dataset):
    def __init__(self, image_dir, instances_json, classes_file, image_size=(64, 64), mask_size=16,
                 normalize_images=True, max_samples=None, min_object_size=0.01,
                 min_objects_per_image=1, max_objects_per_image=8,
                 include_other=False, instance_whitelist=None):
        """
        A PyTorch Dataset for loading Coco and Coco-Stuff annotations and converting
        them to scene graphs on the fly.

        Inputs:
        - image_dir: Path to a directory where images are held
        - instances_json: Path to a directory where JSON annotations are held
        - image_size: Size (H, W) at which to load images. Default (64, 64).
        - mask_size: Size M for object segmentation masks; default 16.
        - normalize_image: If True then normalize images by subtracting ImageNet
          mean pixel and dividing by ImageNet std pixel.
        - max_samples: If None use all images. Other wise only use images in the
          range [0, max_samples). Default None.
        - include_relationships: If True then include spatial relationships; if
          False then only include the trivial __in_image__ relationship.
        - min_object_size: Ignore objects whose bounding box takes up less than
          this fraction of the image.
        - min_objects_per_image: Ignore images which have fewer than this many
          object annotations.
        - max_objects_per_image: Ignore images which have more than this many
          object annotations.
        - include_other: If True, include COCO-Stuff annotations which have category
          "other". Default is False, because I found that these were really noisy
          and pretty much impossible for the system to model.
        - instance_whitelist: None means use all instance categories. Otherwise a
          list giving a whitelist of instance category names to use.
        """
        super(Dataset, self).__init__()

        self.image_dir = image_dir
        self.mask_size = mask_size
        self.max_samples = max_samples
        self.normalize_images = normalize_images
        self.set_image_size(image_size)
        self.vocab = {
            'object_name_to_idx': {},
            'pred_name_to_idx': {},
        }
        self.classes = []
        annotations = list(paths.list_files(os.path.join(instances_json), validExts=(".xml")))

        # with open(instances_json, 'r') as f:
        #     instances_data = json.load(f)

        self.image_ids = []
        self.image_id_to_filename = {}
        self.image_id_to_size = {}
        new_image_ids = []
        self.image_id_to_objects = defaultdict(list)

        for j, ann in enumerate(annotations):

            tree = ET.parse(ann)
            anno_xml = tree.getroot()
            # anno_json = open(ann, 'r')
            # image_id = anno_xml.find('path').text
            image_id = j
            filename = anno_xml.find('filename').text
            size = anno_xml.findall('size')[0]
            width = size.find('width').text
            height = size.find('height').text
            self.image_ids.append(image_id)
            self.image_id_to_filename[image_id] = filename
            self.image_id_to_size[image_id] = (width, height)

            cls = open(classes_file, 'r')

            object_idx_to_name = {}
            all_instance_categories = []
            for i, category_data in enumerate(cls):
                category_id = i
                category_name = category_data
                all_instance_categories.append(str(category_name[:-1]))
                object_idx_to_name[category_id] = category_name
                self.vocab['object_name_to_idx'][category_name] = category_id

            if instance_whitelist is None:
                instance_whitelist = all_instance_categories
            category_whitelist = set(instance_whitelist)

            for object_data in anno_xml.findall('object'):
                bndbox = object_data.findall('bndbox')[0]
                xmin = bndbox.find('xmin').text
                ymin = bndbox.find('ymin').text
                xmax = bndbox.find('xmax').text
                ymax = bndbox.find('ymax').text
                w = int(xmax) - int(xmin)
                h = int(ymax) - int(ymin)
                # _, _, w, h = object_data['bndbox']
                # Esto no se que es lo que hace exactamente
                W, H = self.image_id_to_size[image_id]
                W = int(W)
                H = int(H)
                box_area = (w * h) / (W * H)
                box_ok = box_area > min_object_size
                object_name = object_data.find('name').text

                if object_name not in self.classes:
                    self.classes.append(object_name)
                object_data.find('name').set("id", str(self.classes.index(object_name)))
                # object_name = object_idx_to_name[object_data['category_id']]
                category_ok = object_name in category_whitelist
                other_ok = object_name != 'other' or include_other
                if box_ok and category_ok and other_ok:
                    self.image_id_to_objects[image_id].append(object_data)

        self.vocab = {
            'object_name_to_idx': {},
            'pred_name_to_idx': {},
        }

        # COCO category labels start at 1, so use 0 for __image__
        self.vocab['object_name_to_idx']['__image__'] = 0

        # Build object_idx_to_name
        name_to_idx = self.vocab['object_name_to_idx']
        # assert len(name_to_idx) == len(set(name_to_idx.values()))
        max_object_idx = max(name_to_idx.values())
        idx_to_name = ['NONE'] * (1 + max_object_idx)
        for name, idx in self.vocab['object_name_to_idx'].items():
            idx_to_name[idx] = name
        self.vocab['object_idx_to_name'] = idx_to_name
        self.num_objects = len(self.vocab['object_idx_to_name'])

        # Prune images that have too few or too many objects
        total_objs = 0
        for image_id in self.image_ids:
            # Hay que comprobar o cambiar esto a un id numerico por que al ser string no puede usarse como clave o asi para esto y da error. Investigar que se puede hacer con esto
            num_objs = len(self.image_id_to_objects[image_id])
            total_objs += num_objs
            if min_objects_per_image <= num_objs <= max_objects_per_image:
                new_image_ids.append(image_id)
        self.image_ids = new_image_ids
        self.vocab['pred_idx_to_name'] = [
            '__in_image__',
            'left of',
            'right of',
            'above',
            'below',
            'inside',
            'surrounding',
        ]
        self.vocab['pred_name_to_idx'] = {}
        for idx, name in enumerate(self.vocab['pred_idx_to_name']):
            self.vocab['pred_name_to_idx'][name] = idx

    def set_image_size(self, image_size):
        print('called set_image_size', image_size)
        transform = [Resize(image_size), T.ToTensor()]
        if self.normalize_images:
            transform.append(imagenet_preprocess())
        self.transform = T.Compose(transform)
        self.image_size = image_size

    def total_objects(self):
        total_objs = 0
        for i, image_id in enumerate(self.image_ids):

            if self.max_samples and i >= self.max_samples:
                break
            num_objs = len(self.image_id_to_objects[image_id])
            total_objs += num_objs
        return total_objs

    def __len__(self):
        if self.max_samples is None:
            return len(self.image_ids)
        return min(len(self.image_ids), self.max_samples)

    def __getitem__(self, index):
        """
        Get the pixels of an image, and a random synthetic scene graph for that
        image constructed on-the-fly from its COCO object annotations. We assume
        that the image will have height H, width W, C channels; there will be O
        object annotations, each of which will have both a bounding box and a
        segmentation mask of shape (M, M). There will be T triples in the scene
        graph.
        Returns a tuple of:
        - image: FloatTensor of shape (C, H, W)
        - objs: LongTensor of shape (O,)
        - boxes: FloatTensor of shape (O, 4) giving boxes for objects in
          (x0, y0, x1, y1) format, in a [0, 1] coordinate system
        - masks: LongTensor of shape (O, M, M) giving segmentation masks for
          objects, where 0 is background and 1 is object.
        - triples: LongTensor of shape (T, 3) where triples[t] = [i, p, j]
          means that (objs[i], p, objs[j]) is a triple.
        """
        image_id = self.image_ids[index]

        filename = self.image_id_to_filename[image_id]
        image_path = os.path.join(self.image_dir, filename)

        with open(image_path, 'rb') as f:
            with PIL.Image.open(f) as image:
                WW, HH = image.size
                image = self.transform(image.convert('RGB'))

        H, W = self.image_size
        objs, boxes, masks = [], [], []

        for object_data in self.image_id_to_objects[image_id]:
            # objs.append(object_data['category_id'])
            objs.append(int(object_data.find('name').get("id")))

            bndbox = object_data.findall('bndbox')[0]
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            w = xmax - xmin
            h = ymax - ymin

            boxes.append(torch.FloatTensor([xmin, ymin, xmax, ymax]))

            # This will give a numpy array of shape (HH, WW)
            mask = torch.zeros(1, H, W)
            # mask = seg_to_mask(object_data['segmentation'], WW, HH)
            mask[:, round(ymin * H):max(round(ymin * H) + 1, round(ymax * H)),
            round(xmin * W):max(round(xmin * W) + 1, round(xmax * W))] = 1
            masks.append(mask)
        # shuffle objs
        O = len(objs)
        rand_idx = list(range(O))
        random.shuffle(rand_idx)

        objs = [objs[i] for i in rand_idx]
        boxes = [boxes[i] for i in rand_idx]
        masks = [masks[i] for i in rand_idx]

        objs = torch.LongTensor(objs)
        boxes = torch.stack(boxes, dim=0)
        masks = torch.stack(masks, dim=0)

        # print(image_path)

        return image, objs, boxes, masks


def coco_collate_fn(batch):
    """
    Collate function to be used when wrapping CocoSceneGraphDataset in a
    DataLoader. Returns a tuple of the following:

    - imgs: FloatTensor of shape (N, C, H, W)
    - objs: LongTensor of shape (O,) giving object categories
    - boxes: FloatTensor of shape (O, 4)
    - masks: FloatTensor of shape (O, M, M)
    - triples: LongTensor of shape (T, 3) giving triples
    - obj_to_img: LongTensor of shape (O,) mapping objects to images
    - triple_to_img: LongTensor of shape (T,) mapping triples to images
    """
    all_imgs, all_objs, all_boxes, all_masks, all_obj_to_img = [], [], [], [], []

    for i, (img, objs, boxes, masks) in enumerate(batch):
        all_imgs.append(img[None])
        O = objs.size(0)
        all_objs.append(objs)
        all_boxes.append(boxes)
        all_masks.append(masks)

        all_obj_to_img.append(torch.LongTensor(O).fill_(i))

    all_imgs = torch.cat(all_imgs)
    all_objs = torch.cat(all_objs)
    all_boxes = torch.cat(all_boxes)
    all_masks = torch.cat(all_masks)
    all_obj_to_img = torch.cat(all_obj_to_img)

    out = (all_imgs, all_objs, all_boxes, all_masks, all_obj_to_img)

    return out


def get_dataloader(batch_size=10, DATASET_DIR='datasets/kangaroo', CLASSES_FILE='datasets/kangaroo/classes.names',
                   instance_whitelist=None, coco_include_other=False, IMAGE_SIZE=(64, 64)):
    coco_train_image_dir = os.path.join(DATASET_DIR, 'JPEGImages')
    # coco_val_image_dir = os.path.join(DATASET_DIR, 'val/JPEGImages')
    coco_train_instances_json = os.path.join(DATASET_DIR, 'Annotations')
    # coco_val_instances_json = os.path.join(DATASET_DIR, 'val/Annotations')

    # coco_train_image_dir = os.path.join(DATASET_DIR, 'JPEGImages')
    # coco_train_instances_json = os.path.join(DATASET_DIR, 'Annotations')

    # coco_train_instances_json = os.path.join(DATASET_DIR, 'annotations/instances_train2017.json')
    # coco_val_instances_json = os.path.join(DATASET_DIR, 'annotations/instances_val2017.json')
    min_object_size = 0.01
    min_objects_per_image = 1
    mask_size = 16

    image_size = IMAGE_SIZE  # (64, 64)
    num_train_samples = None
    num_val_samples = None
    include_relationships = False
    batch_size = batch_size
    shuffle_val = False
    # build datasets
    dset_kwargs = {
        'image_dir': coco_train_image_dir,
        'instances_json': coco_train_instances_json,
        'classes_file': CLASSES_FILE,
        'image_size': image_size,
        'mask_size': mask_size,
        'max_samples': num_train_samples,
        'min_object_size': min_object_size,
        'min_objects_per_image': min_objects_per_image,
        'instance_whitelist': instance_whitelist,
        'include_other': coco_include_other,
        # 'include_relationships': include_relationships,
    }
    train_dset = PascalVocSceneGraphDataset(**dset_kwargs)
    num_objs = train_dset.total_objects()
    num_imgs = len(train_dset)

    print('Training dataset has %d images and %d objects' % (num_imgs, num_objs))
    print('(%.2f objects per image)' % (float(num_objs) / num_imgs))

    # dset_kwargs['image_dir'] = coco_val_image_dir
    # dset_kwargs['instances_json'] = coco_val_instances_json
    # dset_kwargs['max_samples'] = num_val_samples
    # val_dset = PascalVocSceneGraphDataset(**dset_kwargs)

    # assert train_dset.vocab == val_dset.vocab

    # vocab = json.loads(json.dumps(train_dset.vocab))

    # build dataloader
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': 4,
        'shuffle': True,
        'collate_fn': coco_collate_fn,
    }
    train_loader = DataLoader(train_dset, **loader_kwargs)
    #
    # loader_kwargs['shuffle'] = shuffle_val
    # loader_kwargs['num_workers'] = 1
    # val_loader = DataLoader(val_dset, **loader_kwargs)
    return train_loader  # , val_loader

if __name__ == '__main__':
    train_loader, val_loader = get_dataloader(batch_size=32)

    # test reading data
    for i, batch in enumerate(train_loader):
        imgs, objs, boxes, masks, obj_to_img = batch
        print(imgs.shape, objs.shape, boxes.shape, masks.shape, obj_to_img.shape)
        if i == 20: break
