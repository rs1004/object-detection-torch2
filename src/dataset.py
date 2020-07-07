from torch.utils.data import Dataset


class PascalVOCDataset(Dataset):
    def __init__(self, purpose, transform=None):
        self.transform = transform
        self.purpose = purpose

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return
