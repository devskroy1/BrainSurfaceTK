from torch.utils.data import Dataset

class DroppedDataset(Dataset):

    def __init__(self, x, pos, y):
        self.samples_x = x
        self.samples_pos = pos
        self.samples_y = y

    def __len__(self):
        return len(self.samples_y)

    def __getitem__(self, idx):
        return self.samples_y[idx]
