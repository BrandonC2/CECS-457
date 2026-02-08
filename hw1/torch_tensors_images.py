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


def get_dataset(root: Path) -> datasets.CIFAR10:
    """
    Returns the CIFAR-10 training set.

    The first time you run this, it will download the dataset to `root`.
    """
    root.mkdir(parents=True, exist_ok=True)
    return datasets.CIFAR10(root=str(root), train=True, download=True, transform=None)


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

    index = 10
    pil_img, label = dataset[index]

    img_tensor = pil_to_tensor(pil_img)

    # Basic checks (do not remove)
    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.ndim == 3, f"Expected a 3D tensor (C,H,W). Got shape={tuple(img_tensor.shape)}"
    assert img_tensor.shape[0] == 3, f"Expected 3 channels (RGB). Got shape={tuple(img_tensor.shape)}"

    class_name = dataset.classes[label] if hasattr(dataset, "classes") else str(label)
    # TODO: print the label (numeric + class name)
    print(f"numeric={label}, class name={class_name}")

    # TODO: print the min and max values in the tensor
    
    min_tensor_value = img_tensor.min().item()
    max_tensor_value = img_tensor.max().item()

    print(f"Min Value = {min_tensor_value}\nMax Value = {max_tensor_value}")

    # TODO: vectorize the image tensor and print the vector shape
    vec = image_tensor_to_vector(img_tensor)
    # TODO: Print the shape of vec
    print(vec.shape)

    # TODO: reshape vec back to the original image shape and verify it matches the original tensor
    C, H, W = tuple(img_tensor.shape)
    vec = vec.reshape(C, H, W) # (3, 32, 32)
    
    # verify if they're the same
    assert vec.shape == img_tensor.shape, f"Expected equal shape to original tensor.\nGot vec shape={tuple(vec.shape)}. Original Shape={tuple(img_tensor.shape)}"
    show_tensor_image(img_tensor, title=f"CIFAR-10 — {class_name}")


if __name__ == "__main__":
    main()
