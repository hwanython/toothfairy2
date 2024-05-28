import ants
import SimpleITK as sitk

from evalutils import SegmentationAlgorithm

import logging
from pathlib import Path
from typing import (
    Optional,
    Pattern,
    Tuple,
)

from pandas import DataFrame
from evalutils.exceptions import FileLoaderError, ValidationError
from evalutils.validators import DataFrameValidator
from evalutils.io import (
    ImageLoader,
)

logger = logging.getLogger(__name__)
task = 'toothfairy2'

class CBCTUniquePathIndicesValidator(DataFrameValidator):
    """
    Validates that the indices from the filenames are unique
    """

    def validate(self, *, df: DataFrame):
        try:
            paths_cbct = df["path_cbct"].tolist()
        except KeyError:
            raise ValidationError(
                "Column `path_cbct` not found in DataFrame."
            )

        if len(set(paths_cbct)) != len(paths_cbct):
            raise ValidationError(
                "The CBCT paths are not unique."
            )


class CBCTUniqueImagesValidator(DataFrameValidator):
    """
    Validates that each image in the set is unique
    """

    def validate(self, *, df: DataFrame):
        try:
            hashes_cbct = df["hash_cbct"].tolist()
        except KeyError:
            raise ValidationError(
                "Column `hash_cbct` not found in DataFrame."
            )

        if len(set(hashes_cbct)) != len(hashes_cbct):
            raise ValidationError(
                "The images are not unique, please submit a unique image for each case."
            )


class ToothFairy2Algorithm(SegmentationAlgorithm):
    def __init__(
        self,
        input_path=Path("./{task}_input/"),
        output_path=Path("./{task}_output/"),
        **kwargs,
    ):
        # Ensure the input and output directories exist
        input_path.mkdir(parents=True, exist_ok=True)
        output_path.mkdir(parents=True, exist_ok=True)
        super().__init__(
            validators=dict(
                input_image=(
                    CBCTUniqueImagesValidator(),
                    CBCTUniquePathIndicesValidator(),
                )
            ),
            input_path=input_path,
            output_path=output_path,
            **kwargs,
        )

    def _load_input_image(self, *, case) -> Tuple[sitk.Image, Path]:
        input_image_file_path_cbct = case["path_cbct"]

        input_image_file_loader = self._file_loaders["input_image"]
        if not isinstance(input_image_file_loader, ImageLoader):
            raise RuntimeError("The used FileLoader was not of subclass ImageLoader")

        # Load the image for this case
        input_image_cbct = ants.image_read(input_image_file_path_cbct.__str__())

        return (
            input_image_cbct,
            input_image_file_path_cbct,
        )

    def process_case(self, *, idx, case):
        # Load and test the image for this case
        (
            input_image_cbct,
            input_image_file_path_cbct,
        ) = self._load_input_image(case=case)

        # Segment nodule candidates
        segmented_nodules = self.predict(image_cbct=input_image_cbct)

        # Write resulting segmentation to output location
        segmentation_path = self._output_path / input_image_file_path_cbct.name.replace(
            "_CBCT", "_seg"
        )
        self._output_path.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(segmented_nodules, str(segmentation_path), True)

        # Write segmentation file path to result.json for this case
        return {
            "outputs": [dict(type="metaio_image", filename=segmentation_path.name)],
            "inputs": [dict(type="metaio_cbct_image", filename=input_image_file_path_cbct.name)],
            "error_messages": [],
        }

    def _load_cases(
        self,
        *,
        folder: Path,
        file_loader: ImageLoader,
        file_filter: Optional[Pattern[str]] = None,
    ) -> DataFrame:
        cases = []

        paths_cbct = sorted(folder.glob("cbct/*"), key=self._file_sorter_key)

        for pth_cbct in paths_cbct:
            if file_filter is None or file_filter.match(str(pth_cbct)):
                try:
                    case_cbct = file_loader.load(fname=pth_cbct)[0]
                    new_cases = [
                        {
                            "hash_cbct": case_cbct["hash"],
                            "path_cbct": case_cbct["path"],
                        }
                    ]
                except FileLoaderError:
                    logger.warning(
                        f"Could not load {pth_cbct.name} using {file_loader}."
                    )
                else:
                    cases += new_cases
            else:
                logger.info(
                    f"Skip loading {pth_cbct.name} because it doesn't match {file_filter}."
                )

        if len(cases) == 0:
            raise FileLoaderError(
                f"Could not load any files in {folder} with {file_loader}."
            )

        return DataFrame(cases)
