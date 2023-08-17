import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pytest
from vpt_core.io.image import ImageSet
from vpt_core.segmentation.seg_result import SegmentationResult

from tests.vpt_plugin_watershed import TEST_DATA_ROOT
from vpt_plugin_watershed.segment import SegmentationMethod
from vpt_plugin_watershed.watershed import key_entity_fill_channel, key_seed_channel


@dataclass(frozen=True)
class Circle:
    x: int
    y: int
    radius: int


def generate_images(image_size: int, cells: List[Circle]) -> Tuple[ImageSet, str, str]:
    dapi = np.ones((image_size, image_size), dtype=np.uint16)
    polyt = np.ones((image_size, image_size), dtype=np.uint16)
    for cell in cells:
        cv2.circle(dapi, (cell.x, cell.y), int(cell.radius * 0.8), (255, 255, 255), -1)
        cv2.circle(polyt, (cell.x, cell.y), int(cell.radius * 1.5), (255, 255, 255), -1)

    nuclear_channel, fill_channel = "DAPI", "PolyT"
    images = ImageSet()
    images[nuclear_channel] = {i: dapi for i in range(3)}
    images[fill_channel] = {i: polyt for i in range(3)}
    return images, nuclear_channel, fill_channel


def get_test_task(file_name: str) -> Dict:
    with open(TEST_DATA_ROOT / file_name, "r") as f:
        data = json.load(f)
        task = data["segmentation_tasks"][0]
    return task


def test_segment_validation_passes_correct() -> None:
    method = SegmentationMethod()
    task = get_test_task("watershed.json")
    method.validate_task(task)


def test_segment_validation_fails_incorrect() -> None:
    method = SegmentationMethod()
    wrong_task = get_test_task("wrong_task.json")
    with pytest.raises(ValueError):
        method.validate_task(wrong_task)


def test_validation_seed_must_match_images() -> None:
    method = SegmentationMethod()
    task = get_test_task("watershed.json")

    # Check with non-matching name
    task["segmentation_parameters"][key_seed_channel] = "fake_channel"
    with pytest.raises(ValueError):
        method.validate_task(task)

    # Check with empty name
    task["segmentation_parameters"][key_seed_channel] = ""
    method.validate_task(task)


def test_validation_fill_must_match_images() -> None:
    method = SegmentationMethod()
    task = get_test_task("watershed.json")

    # Check with non-matching name
    task["segmentation_parameters"][key_entity_fill_channel] = "fake_channel"
    with pytest.raises(ValueError):
        method.validate_task(task)

    # Check with empty name
    task["segmentation_parameters"][key_entity_fill_channel] = ""
    method.validate_task(task)


def test_validation_must_have_entity_type() -> None:
    method = SegmentationMethod()
    task = get_test_task("watershed.json")
    task["entity_types_detected"] = []
    with pytest.raises(ValueError):
        method.validate_task(task)


def test_segment_run_cells() -> None:
    method = SegmentationMethod()
    task = get_test_task("watershed.json")

    TEST_MODEL = str(TEST_DATA_ROOT / "2D_versatile_fluo")
    task["segmentation_parameters"]["stardist_model"] = TEST_MODEL

    cells = [Circle(20, 15, 10), Circle(30, 100, 10), Circle(100, 20, 15), Circle(210, 100, 15)]
    seg_res = method.run_segmentation(
        segmentation_properties=task["segmentation_properties"],
        segmentation_parameters=task["segmentation_parameters"],
        polygon_parameters=task["polygon_parameters"],
        result=["cell"],
        images=generate_images(256, cells)[0],
    )
    for _, z_seg in seg_res.df.groupby(SegmentationResult.z_index_field):
        assert len(z_seg) == len(cells)


def test_segment_run_nuclei() -> None:
    method = SegmentationMethod()
    task = get_test_task("watershed.json")

    TEST_MODEL = str(TEST_DATA_ROOT / "2D_versatile_fluo")
    task["segmentation_parameters"]["stardist_model"] = TEST_MODEL

    cells = [Circle(20, 15, 10), Circle(30, 100, 10), Circle(100, 20, 15), Circle(210, 100, 15)]
    images, seeds_channel, cyto_channel = generate_images(256, cells)
    del images[cyto_channel]
    seg_res = method.run_segmentation(
        segmentation_properties=task["segmentation_properties"],
        segmentation_parameters=task["segmentation_parameters"],
        polygon_parameters=task["polygon_parameters"],
        result=["nuclei"],
        images=images,
    )
    for _, z_seg in seg_res.df.groupby(SegmentationResult.z_index_field):
        assert len(z_seg) == len(cells)


def test_segment_run_two_entities() -> None:
    method = SegmentationMethod()
    task = get_test_task("watershed.json")

    TEST_MODEL = str(TEST_DATA_ROOT / "2D_versatile_fluo")
    task["segmentation_parameters"]["stardist_model"] = TEST_MODEL

    cells = [Circle(20, 15, 10), Circle(30, 100, 10), Circle(100, 20, 15), Circle(210, 100, 15)]
    seg_results = method.run_segmentation(
        segmentation_properties=task["segmentation_properties"],
        segmentation_parameters=task["segmentation_parameters"],
        polygon_parameters=task["polygon_parameters"],
        result=["cells", "nuclei"],
        images=generate_images(256, cells)[0],
    )
    assert isinstance(seg_results, List) and len(seg_results) == 2
    # check that the first result is not covered by the second, so it is a cell
    geom_field = SegmentationResult.geometry_field
    assert not any(seg_results[0].df[geom_field].covered_by(seg_results[1].df[geom_field]))
    for entity_res in seg_results:
        for _, z_seg in entity_res.df.groupby(SegmentationResult.z_index_field):
            assert len(z_seg) == len(cells)


def test_segment_run_unknown_type() -> None:
    method = SegmentationMethod()
    task = get_test_task("watershed.json")

    TEST_MODEL = str(TEST_DATA_ROOT / "2D_versatile_fluo")
    task["segmentation_parameters"]["stardist_model"] = TEST_MODEL

    cells = [Circle(20, 15, 10), Circle(30, 100, 10), Circle(100, 20, 15), Circle(210, 100, 15)]
    seg_res = method.run_segmentation(
        segmentation_properties=task["segmentation_properties"],
        segmentation_parameters=task["segmentation_parameters"],
        polygon_parameters=task["polygon_parameters"],
        result=["fake_type"],
        images=generate_images(256, cells)[0],
    )

    assert type(seg_res) is SegmentationResult
