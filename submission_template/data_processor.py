import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torchvision.transforms import InterpolationMode

from helpers import (
    AugmentationType,
    FlipDirection,
    ImageAugmentations,
    choose_batch_size,
    get_hardware_info,
)


def add_channel_dimension(images):
    if images.ndim == 3:
        return np.expand_dims(images, axis=1)
    return images


class Dataset(TorchDataset):
    def __init__(
        self,
        x,
        y,
        mean,
        std,
        augmentations=None,
        augmentation_probability=0.0,
        augmentation_value_range=None,
        generator=None,
    ):
        if not 0.0 <= augmentation_probability <= 1.0:
            raise ValueError("augmentation_probability must be between 0 and 1")

        self.x = x
        self.y = y
        self.mean = torch.as_tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.as_tensor(std, dtype=torch.float32).view(-1, 1, 1)
        self.augmentations = tuple(augmentations or ())
        self.augmentation_probability = augmentation_probability
        self.augmentation_value_range = augmentation_value_range
        self.generator = generator

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        image = torch.as_tensor(self.x[index], dtype=torch.float32)

        augmentation = self._select_augmentation(image)

        if augmentation is not None:
            image = self._apply_augmentation(image, augmentation)

        image = (image - self.mean) / self.std

        if self.y is None:
            return image

        label = torch.as_tensor(self.y[index], dtype=torch.long)
        return image, label

    def _select_augmentation(self, image):
        if not self.augmentations:
            return None

        probability = torch.rand((), generator=self.generator).item()

        if probability >= self.augmentation_probability:
            return None

        available_augmentations = []

        for augmentation in self.augmentations:
            if augmentation != AugmentationType.RGB:
                available_augmentations.append(augmentation)
            elif image.shape[-3] == 3:
                available_augmentations.append(augmentation)

        selected_index = torch.randint(0, len(available_augmentations), (), generator=self.generator).item()

        return available_augmentations[selected_index]

    def _uniform(self, lower, upper):
        random_value = torch.rand((), generator=self.generator).item()
        return lower + (upper - lower) * random_value

    def _apply_augmentation(self, image, augmentation):
        fill_value = self.mean.flatten().tolist()

        if augmentation == AugmentationType.TRANSLATION:
            return ImageAugmentations.translate(
                image,
                horizontal_percent=self._uniform(-3.0, 3.0),
                vertical_percent=self._uniform(-3.0, 3.0),
                fill_value=fill_value,
                interpolation=InterpolationMode.NEAREST,
            )

        if augmentation == AugmentationType.PIXEL_NOISE:
            return ImageAugmentations.add_pixel_noise(
                image,
                noise_std_percent=self._uniform(1.0, 5.0),
                generator=self.generator,
            )

        if augmentation == AugmentationType.OCCLUSION:
            return ImageAugmentations.occlude(
                image,
                area_percent=self._uniform(1.0, 5.0),
                aspect_ratio=self._uniform(0.75, 1.33),
                fill_value=self.mean,
                generator=self.generator,
            )

        if augmentation == AugmentationType.ROTATION:
            degrees = self._uniform(1.0, 10.0)
            if self._uniform(0.0, 1.0) < 0.5:
                degrees = -degrees
            return ImageAugmentations.rotate(
                image,
                degrees=degrees,
                fill_value=fill_value,
                interpolation=InterpolationMode.BILINEAR,
            )

        if augmentation == AugmentationType.FLIP:
            direction = (
                FlipDirection.HORIZONTAL
                if self._uniform(0.0, 1.0) < 0.5
                else FlipDirection.VERTICAL
            )
            return ImageAugmentations.flip(image, direction=direction)

        if augmentation == AugmentationType.RGB:
            if self.augmentation_value_range is None:
                raise ValueError("RGB augmentation requires augmentation_value_range")
            return ImageAugmentations.adjust_rgb(
                image,
                value_range=self.augmentation_value_range,
                brightness_percent=self._uniform(-5.0, 5.0),
                contrast_percent=self._uniform(-5.0, 5.0),
                saturation_percent=self._uniform(-5.0, 5.0),
                hue_degrees=self._uniform(-5.0, 5.0),
            )

        raise ValueError("unsupported augmentation type: {}".format(augmentation))


class DataProcessor:
    """
    -===================================================================================================================
    INIT ===============================================================================================================
    ====================================================================================================================
    The DataProcessor class will receive the following inputs:
        * train_x: numpy array of shape [n_train_datapoints, channels, height, width], these are the training inputs
        * train_y: numpy array of shape [n_train_datapoints], these are the training labels
        * valid_x: numpy array of shape [n_valid_datapoints, channels, height, width], these are the validation inputs
        * valid_y: numpy array of shape [n_valid_datapoints], these are the validation labels
        * test_x: numpy array of shape [n_valid_datapoints, channels, height, width], these are the test inputs
        * metadata: A dictionary with information about this dataset, with the following keys:
            'num_classes' : The number of output classes in the classification problem
            'codename' : A unique string that represents this dataset
            'input_shape': A tuple describing [n_total_datapoints, channel, height, width] of the input data
            'time_remaining': The amount of compute time left for your submission

    You can modify or add anything into the metadata that you wish, if you want to pass messages between your classes

    """
    def __init__(self, train_x, train_y, valid_x, valid_y, test_x, metadata, clock):
        self.train_x = add_channel_dimension(train_x)
        self.train_y = train_y
        self.valid_x = add_channel_dimension(valid_x)
        self.valid_y = valid_y
        self.test_x = add_channel_dimension(test_x)
        self.metadata = metadata
        self.clock = clock

        self.seed = 42
        self.metadata["seed"] = self.seed

        self.augmentations = [
            AugmentationType.TRANSLATION,
            AugmentationType.PIXEL_NOISE,
            AugmentationType.OCCLUSION,
        ]
        self.augmentation_probability = 0.5
        self.augmentation_value_range = None

        if AugmentationType.RGB in self.augmentations:
            self.augmentation_value_range = (
                float(np.min(self.train_x)),
                float(np.max(self.train_x)),
            )

    """
    ====================================================================================================================
    PROCESS ============================================================================================================
    ====================================================================================================================
    This function will be called, and it expects you to return three outputs:
        * train_loader: A Pytorch dataloader of (input, label) tuples
        * valid_loader: A Pytorch dataloader of (input, label) tuples
        * test_loader: A Pytorch dataloader of (inputs)  <- Make sure shuffle=False and drop_last=False!
        
    See https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader for more info.  
        
    Here, you can do whatever you want to the input data to process it for your NAS algorithm and training functions
    """
    def process(self):
        mean = np.mean(self.train_x, axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
        std = np.std(self.train_x, axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
        std = np.maximum(std, np.finfo(np.float32).eps)

        augmentation_generator = torch.Generator()
        augmentation_generator.manual_seed(self.seed + 1)

        train_dataset = Dataset(
            self.train_x,
            self.train_y,
            mean,
            std,
            augmentations=self.augmentations,
            augmentation_probability=self.augmentation_probability,
            augmentation_value_range=self.augmentation_value_range,
            generator=augmentation_generator,
        )

        valid_dataset = Dataset(self.valid_x, self.valid_y, mean, std)

        test_dataset = Dataset(self.test_x, None, mean, std)

        hardware = get_hardware_info()
        batch_size = choose_batch_size(self.train_x.shape, hardware)
        self.metadata["hardware"] = hardware
        self.metadata["batch_size"] = batch_size

        generator = torch.Generator()
        generator.manual_seed(self.seed)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            generator=generator,
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

        return train_loader, valid_loader, test_loader
