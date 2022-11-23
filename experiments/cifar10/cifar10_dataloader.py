
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import torch; torch.manual_seed(0)

def get_cifar10_dataset(batch_size=256, eval_batch_size=256):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    to_numpy = lambda x: x.numpy()
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.Lambda(to_numpy)
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.Lambda(to_numpy)
    ])

    # build train dataset
    train_dataset = datasets.CIFAR10(
        '.', download=True, train=True, transform=train_transform
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        # num_workers=4,
        num_workers=1,
        drop_last=True
    )

    # build eval dataset
    eval_dataset = datasets.CIFAR10(
        '.', download=True, train=False, transform=eval_transform
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=eval_batch_size, shuffle=False,
        # num_workers=4,
        num_workers=1,
        drop_last=True
    )

    train_loader = PostProcessor(train_loader)
    eval_loader = PostProcessor(eval_loader)
    return train_loader, eval_loader

def get_svhn_dataset(batch_size=256, eval_batch_size=256):
    mean = [0.5, 0.5, 0.5]
    std = [1.0, 1.0, 1.0]
    to_numpy = lambda x: x.numpy()
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.Lambda(to_numpy)
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.Lambda(to_numpy)
    ])

    # build train dataset
    train_dataset = datasets.SVHN(
        '.', download=True, split='train', transform=train_transform
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        # num_workers=4,
        num_workers=1,
        drop_last=True
    )

    # build eval dataset
    eval_dataset = datasets.SVHN(
        '.', download=True, split='test', transform=eval_transform
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=eval_batch_size, shuffle=False,
        # num_workers=4,
        num_workers=1,
        drop_last=True
    )

    train_loader = PostProcessor(train_loader)
    eval_loader = PostProcessor(eval_loader)
    return train_loader, eval_loader

class PostProcessor:
    def __init__(self, dataloader):
        self.dataloader = dataloader

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        self.it = iter(self.dataloader)
        return self

    def __next__(self):
        data, labels = next(self.it)
        batch = {
            'image':self._post_data_transform(data),
            'label':self._post_label_transform(labels)
        }
        return batch

    @staticmethod
    def _post_data_transform(x):
        return x.numpy().transpose(0,2,3,1)

    @staticmethod
    def _post_label_transform(x):
        return x.numpy()

def make_continuous_iter(dl):
    it = iter(dl)
    while True:
        try:
            yield from it
        except StopIteration:
            it = iter(dl)

def test():
    train_batches, test_batches = get_cifar10_dataset(batch_size=10)

    for batch in train_batches:
        # print(x.shape)
        # print(x.transpose(0,2,3,1))
        # print(batch)
        print(batch['label'])
        input()
        break

    for batch in train_batches:
        print(batch)
        input()
        break

def test_svhn():
    # train_batches, test_batches = get_svhn_dataset(batch_size=10)
    train_batches, test_batches = get_cifar10_dataset(batch_size=10)

    for batch in train_batches:
        # print(x.shape)
        # print(x.transpose(0,2,3,1))
        # print(batch)
        print(batch['image'].shape)
        print(batch['label'])
        input()
        break


if __name__ == '__main__':
    # test()
    test_svhn()