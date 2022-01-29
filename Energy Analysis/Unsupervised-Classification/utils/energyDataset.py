
from PIL import Image
from typing import List
from torch.tensor import Tensor
from torchvision.transforms import transforms
from torch.utils.data import Dataset

class EnergyDataset(Dataset):
    def __init__(self, img_path: str, transforms: transforms = None) -> None:
        super().__init__()

        self.img_path = img_path
        self.transforms = transforms

        self.images, self.labels = self.get_images(self.img_path)


    def __getitem__(self, index) -> Tensor:
        
        img_path = self.images[index]
        lab = self.labels[index]

        img = Image.open(img_path).convert("RGB")

        if self.transforms is not None:
            img = self.transforms(img)
        else:
            img = transforms.ToTensor()(img)

        return (img, lab)


    def __len__(self) -> None:
        return len(self.images)


    def get_images(self, img_path: str) -> List:
        """获取指定路径下的所有图片"""

        import os
        
        if not os.path.exists(img_path):
            raise f"{img_path} is not exist"

        results = []
        labels = []

        dirs = os.listdir(img_path)
        for d in dirs:
            if d == "alloy":
                # alloy : 0
                alloy_path = os.path.join(img_path, "alloy")
                filenames = os.listdir(alloy_path)
                for filename in filenames:
                    results.append(os.path.join(alloy_path, filename))
                    labels.append(0)
            elif d == "soil":
                # soil : 1
                soil_path = os.path.join(img_path, "soil")
                filenames = os.listdir(soil_path)
                for filename in filenames:
                    results.append(os.path.join(soil_path, filename))
                    labels.append(1)
            elif d == "com-alloy":
                # common-alloy : 2
                com_alloy_path = os.path.join(img_path, "com-alloy")
                filenames = os.listdir(com_alloy_path)
                for filename in filenames:
                    results.append(os.path.join(com_alloy_path, filename))
                    labels.append(2)

        return results, labels
        

if __name__ == "__main__":

    energyDataset = EnergyDataset("/home/linsi/Projects/Energy Analysis/classification/datas/images")

    from torch.utils.data import DataLoader
    energyDataloader = DataLoader(energyDataset, batch_size=1, shuffle=True)

    for imgs in energyDataloader:
        print(imgs) 