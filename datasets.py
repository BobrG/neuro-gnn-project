import abc
import torch
import pandas as pd
import nibabel as nib
from torch.utils.data import Dataset

from utils import get_nii_path, ToTensor

import logging

logger = logging.getLogger('root')
logger.info('datasets')

class MRIDataset(Dataset):
    """Abstract class for all derived MRIDatasets."""

    def __init__(self, caps_directory, data_file,
                 preprocessing, transformations, labels,
                 augmentation_transformations=None, multi_cohort=False):
        self.caps_dict = self.create_caps_dict(caps_directory, multi_cohort)
        self.transformations = transformations
        self.augmentation_transformations = augmentation_transformations
        self.eval_mode = False
        self.labels = labels
        self.diagnosis_code = {
            'CN': 0,
            'BV': 1,
            'AD': 1,
            'sMCI': 0,
            'pMCI': 1,
            'MCI': 1,
            'unlabeled': -1}
        self.preprocessing = preprocessing

        if not hasattr(self, 'elem_index'):
            raise ValueError(
                "Child class of MRIDataset must set elem_index attribute.")
        if not hasattr(self, 'mode'):
            raise ValueError(
                "Child class of MRIDataset must set mode attribute.")

        # Check the format of the tsv file here
        if isinstance(data_file, str):
            self.df = pd.read_csv(data_file, sep='\t')
        elif isinstance(data_file, pd.DataFrame):
            self.df = data_file
        else:
            raise Exception('The argument data_file is not of correct type.')

        if not multi_cohort:
            self.df["cohort"] = "single"

        mandatory_col = {"participant_id", "session_id"}
        if self.labels:
            mandatory_col.add("diagnosis")
        if multi_cohort:
            mandatory_col.add("cohort")
        if self.elem_index == "mixed":
            mandatory_col.add("%s_id" % self.mode)

        if not mandatory_col.issubset(set(self.df.columns.values)):
            raise Exception("the data file is not in the correct format."
                            "Columns should include %s" % mandatory_col)

        unique_diagnoses = set(self.df.diagnosis)
        unique_codes = set()
        for diagnosis in unique_diagnoses:
            unique_codes.add(self.diagnosis_code[diagnosis])
        self.elem_per_image = self.num_elem_per_image()
        self.size = self[0]['image'].size()

    def __len__(self):
        return len(self.df) * self.elem_per_image

    @staticmethod
    def create_caps_dict(caps_directory, multi_cohort):
        caps_dict = {'single': caps_directory}
        return caps_dict

    def _get_meta_data(self, idx):
        image_idx = idx // self.elem_per_image
        participant = self.df.loc[image_idx, 'participant_id']
        session = self.df.loc[image_idx, 'session_id']
        cohort = self.df.loc[image_idx, 'cohort']

        if self.elem_index is None:
            elem_idx = idx % self.elem_per_image
        elif self.elem_index == "mixed":
            elem_idx = self.df.loc[image_idx, '%s_id' % self.mode]
        else:
            elem_idx = self.elem_index

        if self.labels:
            diagnosis = self.df.loc[image_idx, 'diagnosis']
            label = self.diagnosis_code[diagnosis]
        else:
            label = self.diagnosis_code['unlabeled']

        return participant, session, cohort, elem_idx, label

    def _get_full_image(self):

        participant_id = self.df.loc[0, 'participant_id']
        session_id = self.df.loc[0, 'session_id']
        cohort = self.df.loc[0, 'cohort']

        image_path = get_nii_path(
                self.caps_dict,
                participant_id,
                session_id,
                cohort=cohort,
                preprocessing=self.preprocessing)
        logger.info(image_path)
        image_nii = nib.load(image_path)
        image_np = image_nii.get_fdata()
        
        
        image = ToTensor()(image_np)

        return image

    @abc.abstractmethod
    def __getitem__(self, idx):
        pass

    @abc.abstractmethod
    def num_elem_per_image(self):
        pass

    def eval(self):
        self.eval_mode = True
        return self

    def train(self):
        self.eval_mode = False
        return self
    
    
class MRIDatasetImage(MRIDataset):
    """Dataset of MRI organized in a CAPS folder."""

    def __init__(self, caps_directory, data_df,
                 preprocessing='t1-linear', train_transformations=None,
                 labels=True, all_transformations=None, multi_cohort=False):
        """
        Args:
            caps_directory (string): Directory of all the images.
            data_file (string or DataFrame): Path to the tsv file or DataFrame containing the subject/session list.
            preprocessing (string): Defines the path to the data in CAPS.
            train_transformations (callable, optional): Optional transform to be applied only on training mode.
            labels (bool): If True the diagnosis will be extracted from the given DataFrame.
            all_transformations (callable, options): Optional transform to be applied during training and evaluation.
            multi_cohort (bool): If True caps_directory is the path to a TSV file linking cohort names and paths.
        """
        self.elem_index = None
        self.mode = "image"
        super().__init__(caps_directory, data_df, preprocessing,
                         augmentation_transformations=train_transformations, labels=labels,
                         transformations=all_transformations, multi_cohort=multi_cohort)

    def __getitem__(self, idx):
        participant, session, cohort, _, label = self._get_meta_data(idx)
        image_path =  get_nii_path(self.caps_dict, participant, session, cohort, self.preprocessing)
        image_nii = nib.load(image_path)
        image_np = image_nii.get_fdata()
        image = ToTensor()(image_np)

        if self.transformations:
            image = self.transformations(image)

        if self.augmentation_transformations and not self.eval_mode:
            image = self.augmentation_transformations(image)

        sample = {'image': image, 'label': label, 'participant_id': participant, 'session_id': session,
                  'image_path': image_path}

        return sample

    def num_elem_per_image(self):
        return 1
