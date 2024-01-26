import os, random
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


class DefaultDataset(Dataset):
    def __init__(
        self, annotation_file, max_sequence_len=10, transform=None, target_transform=None
    ):
        self.slide_labels = pd.read_csv(annotation_file)
        self.max_sequence_len = max_sequence_len
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.slide_labels)

    def __getitem__(self, idx):
        slide_path = self.slide_labels.iloc[idx, 0]
        
        cur_len = 0
        image_list = []
        # TODO: Let's not care about the constant number of chosen images for now
        while cur_len < self.max_sequence_len:
            cur_len += 1

            # TODO: It can be improved - where to look at a slide??
            image_path = random.choice(os.listdir(slide_path))
            image = read_image(slide_path + "/" + image_path)
            image = image / 255

            image_list.append(image)

        joined_images = torch.stack(image_list)

        label = self.slide_labels.iloc[idx, 1].astype("int")
        if self.transform:
            joined_images = self.transform(joined_images)
        if self.target_transform:
            label = self.target_transform(label)
        return joined_images, label
