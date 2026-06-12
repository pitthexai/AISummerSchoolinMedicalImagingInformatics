class JointSegmentationDataset(Dataset):
    def __init__(self, data, num_classes, transforms=None, preprocessing=None):
        self.data = data
        self.num_classes = num_classes
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = cv2.imread(self.data.iloc[idx]['image'])
        mask = cv2.imread(self.data.iloc[idx]['mask'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image.type(torch.FloatTensor), (mask / 255).long()
