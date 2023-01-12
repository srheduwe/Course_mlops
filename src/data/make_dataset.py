import numpy as np
import os
import torch

class mnist():
  def __init__(self, train):
    path = "data/raw/"
    filepaths = os.listdir(path)
    print(path, filepaths)

    if train:
      train_files = [path + i for i in filepaths if i.startswith("train")]
      content = [np.load(f) for f in train_files]
      data = torch.tensor(np.concatenate([c['images'] for c in content])).reshape(-1, 1, 28, 28)
      targets = torch.tensor(np.concatenate([c['labels'] for c in content]))

    else:
      content = np.load(path + "test.npz", allow_pickle=True)
      data = torch.tensor(content['images']).reshape(-1, 1, 28, 28)
      targets = torch.tensor(content['labels'])

    # data = (data - data.mean())/data.std()
    # targets = (targets.float() - targets.float().mean())/targets.float().std()
    # targets = targets.long()
    self.data, self.targets = data, targets

  
  def __len__(self):
      return self.targets.numel()
  
  def __getitem__(self, idx):
      return self.data[idx].float(), self.targets[idx]

if __name__ == "__main__":
    dataset_train = mnist(train=True)
    dataset_test = mnist(train=False)
    torch.save(dataset_train, 'data/processed/dataset_train.pt')
    torch.save(dataset_test, 'data/processed/dataset_test.pt')
    print(dataset_train.data.shape)
    print(dataset_train.targets.shape)
    print(dataset_test.data.shape)
    print(dataset_test.targets.shape)
