import multiprocessing
import shutil
from multiprocessing import Pool

import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw

LABEL_dict = {
    "background": 0,
    "Lower Jawbone": 1,
    "Upper Jawbone": 2,
    "Left Inferior Alveolar Canal": 3,
    "Right Inferior Alveolar Canal": 4,
    "Left Maxillary Sinus": 5,
    "Right Maxillary Sinus": 6,
    "Pharynx": 7,
    "Bridge": 8,
    "Crown": 9,
    "Implant": 10,
    "Upper Right Central Incisor": 11,
    "Upper Right Lateral Incisor": 12,
    "Upper Right Canine": 13,
    "Upper Right First Premolar": 14,
    "Upper Right Second Premolar": 15,
    "Upper Right First Molar": 16,
    "Upper Right Second Molar": 17,
    "Upper Right Third Molar (Wisdom Tooth)": 18,
    "NA1": 19,
    "NA2": 20,
    "Upper Left Central Incisor": 21,
    "Upper Left Lateral Incisor": 22,
    "Upper Left Canine": 23,
    "Upper Left First Premolar": 24,
    "Upper Left Second Premolar": 25,
    "Upper Left First Molar": 26,
    "Upper Left Second Molar": 27,
    "Upper Left Third Molar (Wisdom Tooth)": 28,
    "NA3": 29,
    "NA4": 30,
    "Lower Left Central Incisor": 31,
    "Lower Left Lateral Incisor": 32,
    "Lower Left Canine": 33,
    "Lower Left First Premolar": 34,
    "Lower Left Second Premolar": 35,
    "Lower Left First Molar": 36,
    "Lower Left Second Molar": 37,
    "Lower Left Third Molar (Wisdom Tooth)": 38,
    "NA5": 39,
    "NA6": 40,
    "Lower Right Central Incisor": 41,
    "Lower Right Lateral Incisor": 42,
    "Lower Right Canine": 43,
    "Lower Right First Premolar": 44,
    "Lower Right Second Premolar": 45,
    "Lower Right First Molar": 46,
    "Lower Right Second Molar": 47,
    "Lower Right Third Molar (Wisdom Tooth)": 48
  }

if __name__ == '__main__':
    toothfairy2_data_dir = '/mnt/imagenas/CT/MICCAI/Toothfairy2/Dataset113_ToothFairy2'

    task_id = 999
    task_name = "toothfairy2"
    task_prefix = "toothfairy2"

    foldername = "Dataset%03.0d_%s" % (task_id, task_name)

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)
    # prefix이 같은 애들을 다 불러오는거
    # case_ids = subdirs(han_data_dir, prefix='case', join=False)
    valid_data_list = r'/home/jhhan/02_dev/nnunet/TEMP/MICCAI_VALID_DATALIST.txt'
    with open(valid_data_list, 'r') as f: # remve \n
        case_ids = [line.strip() for line in f]
    
    # for c in case_ids:
    #     # image
    #     shutil.copy(join(toothfairy2_data_dir, 'imagesTr', c + '_0000.nrrd'), join(imagestr, c + '_0000.nrrd'))
    #     # shutil.copy(join(han_data_dir, c, c + "_IMG_MR_T1_aligned.nrrd"), join(imagestr, task_prefix+c[4:] + '_0001.nrrd')) #align
    #     # label
    #     shutil.copy(join(toothfairy2_data_dir, 'labelsTr', c + ".nrrd"), join(labelstr, c + ".nrrd"))

    
    generate_dataset_json(out_base,
                          channel_names={0: 'CBCT'},
                          labels=LABEL_dict,
                          num_training_cases=len(case_ids),
                          file_ending='.nrrd',
                          dataset_release='1.0')
