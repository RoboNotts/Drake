import torchvision.transforms as transforms
import torch


def preprocessing(image, resize=True, norm=True, size=[360, 480]):
    #preprocessing the image before sending it to the neural network
    if resize:
        image = transforms.functional.resize(image, size)
    transform = transforms.ConvertImageDtype(torch.float32)
    if norm:
        return normalization(transform(image))
    else:
        return transform(image)


def normalization(image):
    #normalization
    transform = transforms.Normalize(
        mean=torch.tensor([0.485, 0.456, 0.406]),
        std=torch.tensor([0.229, 0.224, 0.225]))
    return transform(image)