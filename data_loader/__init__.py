from .cityscapes import CitySegmentation
from .cityscapes_texture_loader import cityscapesFormatTextureLoader


datasets = {
    'citys': CitySegmentation,
    'berkeley_campus_cityscapes_format': cityscapesFormatTextureLoader,
}

def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
