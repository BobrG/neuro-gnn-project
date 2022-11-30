import os
import torch
import numpy as np
from pathlib import Path
import torchvision.transforms as transforms


FILENAME_TYPE = {'full': '_T1w_space-MNI152NLin2009cSym_res-1x1x1_T1w',
                 'cropped': '_T1w_space-MNI152NLin2009cSym_desc-Crop_res-1x1x1_T1w', # use only cropped
                 'skull_stripped': '_space-Ixi549Space_desc-skullstripped_T1w',
                 'gm_maps': '_T1w_segm-graymatter_space-Ixi549Space_modulated-off_probability',
                 'shepplogan': '_phantom-SheppLogan'}


def get_nii_path(caps_dict, participant_id, session_id, cohort, preprocessing):

    if cohort not in caps_dict.keys():
        raise ValueError('Cohort names in labels and CAPS definitions do not match.')

    image_path = os.path.join(caps_dict[cohort], participant_id, 'raw_data',
                               participant_id + '_' + session_id +
                               FILENAME_TYPE['cropped'] + '.nii.gz')
    return image_path

class ToTensor(object):
    """Convert image type to Tensor and diagnosis to diagnosis code"""

    def __call__(self, image):
        np.nan_to_num(image, copy=False)
        image = image.astype(float)

        return torch.from_numpy(image[np.newaxis, :]).float()
    
class MinMaxNormalization(object):
    """Normalizes a tensor between 0 and 1"""

    def __call__(self, image):
        return (image - image.min()) / (image.max() - image.min())

def get_transforms(mode, minmaxnormalization=True, data_augmentation=None):
    """
    Outputs the transformations that will be applied to the dataset
    :param mode: (str) input used by the network. Chosen from ['image', 'patch', 'roi', 'slice'].
    :param minmaxnormalization: (bool) if True will perform MinMaxNormalization
    :param data_augmentation: (list[str]) list of data augmentation performed on the training set.
    :return:
    - container transforms.Compose including transforms to apply in train and evaluation mode.
    - container transforms.Compose including transforms to apply in evaluation mode only.
    """
    augmentation_dict = {"None": None}
    if data_augmentation:
        augmentation_list = [augmentation_dict[augmentation] for augmentation in data_augmentation]
    else:
        augmentation_list = []

    if minmaxnormalization:
        transformations_list = [MinMaxNormalization()]
    else:
        transformations_list = []

    all_transformations = transforms.Compose(transformations_list)
    train_transformations = transforms.Compose(augmentation_list)

    return train_transformations, all_transformations
    
