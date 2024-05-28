import time
import SimpleITK as sitk
import numpy as np
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
np.lib.index_tricks.int = np.uint16
import ants
from os.path import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import json
from toothfairy2_algorithm import ToothFairy2Algorithm

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


def ants_2_itk(image):
    imageITK = sitk.GetImageFromArray(image.numpy().T)
    imageITK.SetOrigin(image.origin)
    imageITK.SetSpacing(image.spacing)
    imageITK.SetDirection(image.direction.reshape(9))
    return imageITK

def itk_2_ants(image):
    image_ants = ants.from_numpy(sitk.GetArrayFromImage(image).T,
                                 origin=image.GetOrigin(),
                                 spacing=image.GetSpacing(),
                                 direction=np.array(image.GetDirection()).reshape(3, 3))
    return image_ants


class Algorithm(ToothFairy2Algorithm):
    def __init__(self):
        super().__init__()

    def predict(self, *, image_ct: ants.ANTsImage, image_mrt1: ants.ANTsImage) -> sitk.Image:
        print("Computing registration", flush=True)
        time0reg= time.time_ns()
        mytx = ants.registration(fixed=image_ct, moving=image_mrt1, type_of_transform='Affine')  #, aff_iterations=(150, 150, 150, 150))
        print(f"Time reg: {(time.time_ns()-time0reg)/1000000000}")
        warped_MR = ants.apply_transforms(fixed=image_ct, moving=image_mrt1,
                                          transformlist=mytx['fwdtransforms'], defaultvalue=image_mrt1.min())
        trained_model_path = join("/opt", "algorithm", "checkpoint", "nnUNet", "Dataset777_HaNSeg2023", "nnUNetTrainer__nnUNetPlans__3d_fullres")
        # trained_model_path = join("/usr/DATA/backup_home_dir/jhhan/01_research/01_MICCAI/01_grandchellenge/han_seg/src/HanSeg_2023/nnUNet/dataset/nnUNet_results", 
        #                           "Dataset777_HaNSeg2023", "nnUNetTrainer__nnUNetPlans__3d_fullres")

        spacing = tuple(map(float,json.load(open(join(trained_model_path, "plans.json"), "r"))["configurations"]["3d_fullres"]["spacing"]))
        ct_image = ants_2_itk(image_ct)
        mr_image = ants_2_itk(warped_MR)
        del image_mrt1
        del warped_MR


        properties = {
            'sitk_stuff':
                {'spacing': ct_image.GetSpacing(),
                 'origin': ct_image.GetOrigin(),
                 'direction': ct_image.GetDirection()
                },
            # the spacing is inverted with [::-1] because sitk returns the spacing in the wrong order lol. Image arrays
            # are returned x,y,z but spacing is returned z,y,x. Duh.
            'spacing': ct_image.GetSpacing()[::-1]
        }
        images = np.vstack([sitk.GetArrayFromImage(ct_image)[None], sitk.GetArrayFromImage(mr_image)[None]]).astype(np.float32)
        fin_origin = ct_image.GetOrigin()
        fin_spacing = ct_image.GetSpacing()
        fin_direction = ct_image.GetDirection()
        fin_size = ct_image.GetSize()
        print(fin_spacing)
        print(spacing)
        print(fin_size)

        old_shape = np.shape(sitk.GetArrayFromImage(ct_image))
        del mr_image
        del ct_image
        # Shamelessly copied from nnUNet/nnunetv2/preprocessing/resampling/default_resampling.py
        # new_shape = np.array([int(round(i / j * k)) for i, j, k in zip(fin_spacing, spacing[::-1], fin_size)])
        predictor = nnUNetPredictor(tile_step_size=0.5, use_mirroring=True, perform_everything_on_gpu=True,
                                    verbose=True, verbose_preprocessing=True,
                                    allow_tqdm=True)
        print(f"Old shape: {old_shape}")
        print(f"New shape: {old_shape}")

        predictor.initialize_from_trained_model_folder(trained_model_path, use_folds="0",
                                                        checkpoint_name="checkpoint_best.pth")

        img_temp = predictor.predict_single_npy_array(images, properties, None, None, False).astype(np.uint8)
        del images
        print("Prediction Done", flush=True)
        output_seg = sitk.GetImageFromArray(img_temp)
        print(f"Seg: {output_seg.GetSize()}, CT: {fin_size}")
        # output_seg.CopyInformation(ct_image)
        output_seg.SetOrigin(fin_origin)
        output_seg.SetSpacing(fin_spacing)
        output_seg.SetDirection(fin_direction)
        print("Got Image", flush=True)
        # save the simpleITK image 
        # sitk.WriteImage(output_seg, str("output_seg.seg.nrrd"), True)
        return output_seg

if __name__ == "__main__":
    time0 = time.time_ns()
    Algorithm().process()
    print((time.time_ns()-time0)/1000000000)
