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
    "A_Carotid_L": 1,
    "A_Carotid_R": 2,
    "Arytenoid": 3,
    "Bone_Mandible": 4,
    "Brainstem": 5,
    "BuccalMucosa": 6,
    "Cavity_Oral": 7,
    "Cochlea_L": 8,
    "Cochlea_R": 9,
    "Cricopharyngeus": 10,
    "Esophagus_S": 11,
    "Eye_AL": 12,
    "Eye_AR": 13,
    "Eye_PL": 14,
    "Eye_PR": 15,
    "Glnd_Lacrimal_L": 16,
    "Glnd_Lacrimal_R": 17,
    "Glnd_Submand_L": 18,
    "Glnd_Submand_R": 19,
    "Glnd_Thyroid": 20,
    "Glottis": 21,
    "Larynx_SG": 22,
    "Lips": 23,
    "OpticChiasm": 24,
    "OpticNrv_L": 25,
    "OpticNrv_R": 26,
    "Parotid_L": 27,
    "Parotid_R": 28,
    "Pituitary": 29,
    "SpinalCord": 30,
}

if __name__ == '__main__':
    han_data_dir = '/usr/DATA/backup_home_dir/jhhan/01_research/01_MICCAI/01_grandchellenge/han_seg/data/raw/HaN-Seg/set_1'

    task_id = 777
    task_name = "HaNSeg2023"
    task_prefix = "hanseg"

    foldername = "Dataset%03.0d_%s" % (task_id, task_name)

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)
    # prefix이 같은 애들을 다 불러오는거
    case_ids = subdirs(han_data_dir, prefix='case', join=False)

    for c in case_ids:
        # image
        shutil.copy(join(han_data_dir, c, c + "_IMG_CT.nrrd"), join(imagestr, task_prefix+c[4:] + '_0000.nrrd'))
        shutil.copy(join(han_data_dir, c, c + "_IMG_MR_T1_aligned.nrrd"), join(imagestr, task_prefix+c[4:] + '_0001.nrrd')) #align
        # label
        shutil.copy(join(han_data_dir, c, c + "_all_rois.seg.nrrd"), join(labelstr, task_prefix+c[4:] + '.nrrd'))

    
    generate_dataset_json(out_base,
                          channel_names={0: 'CT', 1: 'T1'},
                          labels=LABEL_dict,
                          num_training_cases=len(case_ids),
                          file_ending='.nrrd',
                          dataset_release='1.0')
