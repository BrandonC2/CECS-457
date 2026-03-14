"""
PyTorch — Tensors from Images (Starter)

Tasks:
1) Download/load CIFAR-10 using torchvision.datasets
2) Take ONE image (a PIL Image)
3) Convert it to a torch.Tensor of shape (3, H, W)
4) Display the tensor as an image to verify it looks correct
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torchvision


def get_dataset(root: Path) -> datasets.CIFAR10:
    """
    Returns the CIFAR-10 training set.

    The first time you run this, it will download the dataset to `root`.
    """
    root.mkdir(parents=True, exist_ok=True)
    return datasets.CIFAR10(root=str(root), train=True, download=True, transform=torchvision.transforms.ToTensor())


def pil_to_tensor(pil_img) -> torch.Tensor:
    """
    Convert a PIL image (RGB) to a torch.Tensor with shape (3, H, W).
    """
    to_tensor = transforms.ToTensor()  # float32 in [0, 1], shape (C, H, W)
    return to_tensor(pil_img)


def show_tensor_image(img_tensor: torch.Tensor, title: str, save_path: Path | None = None) -> None:
    """
    TODO: Display a (C, H, W) image tensor using matplotlib.

    Requirements:
    - img_tensor is a torch.Tensor with shape (3, H, W) and values in [0, 1]
    - matplotlib expects (H, W, C), so you must reorder dimensions before plotting

    Hints:
    - Use `img_tensor.permute(1, 2, 0)` to go from (C,H,W) -> (H,W,C)
    - Convert to numpy with `.detach().cpu().numpy()`
    """

    img_np = img_tensor.permute(1,2,0).detach().cpu().numpy()
    plt.imshow(img_np)
    plt.title(title)
    plt.show()

    # raise NotImplementedError("TODO: implement show_tensor_image()")





def image_tensor_to_vector(img_tensor: torch.Tensor) -> torch.Tensor:
    """
    TODO: Convert an image tensor of shape (3, H, W) into a 1D vector of shape (3*H*W,).

    Hints:
    - `img_tensor.flatten()` returns a 1D tensor
    - Keep the order consistent so that reshaping back recovers the original
    """
    # raise NotImplementedError("TODO: implement image_tensor_to_vector()")

    return img_tensor.flatten()


def main() -> None:
    data_root = Path(__file__).resolve().parent / "data"
    dataset = get_dataset(data_root)

    index = 0
    pil_img, label = dataset[index]

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    images, labels = next(iter(data_loader))

    # img_tensor = pil_to_tensor(images).unsqueeze(0)

    layer1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1)

    output = layer1(images)

    print(f"Output tensor shape: {output.shape}")
    print(f"Image tensor shape: {images.shape}")


if __name__ == "__main__":
    main()
