import json
import os
import pathlib
import queue
import sys
import time
from multiprocessing import Process
from multiprocessing import Queue
from typing import Any
from typing import Dict
from typing import List

import hydra
import openslide
from loguru import logger
from omegaconf import DictConfig

sys.path.append("../../")

from src.tilers import DeepZoomStaticTiler


def get_file_path(dir: pathlib.Path, file_prefix: str) -> pathlib.Path:
    for file in os.listdir(dir):
        file_path = dir / file
        if file.startswith(file_prefix):
            return file_path


def get_slides_path(dir: pathlib.Path) -> List[pathlib.Path]:
    slides_path = []
    for file in os.listdir(dir / "images"):
        file_path = dir / "images" / file
        if os.path.isdir(file_path):
            slides_path.append(file_path)
    return slides_path


def create_file_case_mapping(metadata) -> Dict[str, Any]:
    file_case_mapping = {}
    for file in metadata:
        file_case_mapping[file["file_id"]] = file["associated_entities"][0]["case_id"]
    return file_case_mapping


def create_case_sample_type_mapping(biospecimen) -> Dict[str, Any]:
    case_sample_type_mapping = {}
    for case in biospecimen:
        case_sample_type_mapping[case["case_id"]] = case["samples"][0]["sample_type"]
    return case_sample_type_mapping


def process_slide(slide_queue):
    while True:
        try:
            tiler, slide_dir, file_case_mapping, case_sample_type_mapping, save_dir = slide_queue.get_nowait()
        except queue.Empty:
            break
        else:
            file_name = pathlib.PurePath(slide_dir).name

            sample_case_id = file_case_mapping[file_name]
            sample_type = "normal" if "Normal" in case_sample_type_mapping[sample_case_id] else "tumor"

            slide_path = get_file_path(slide_dir, "TCGA")
            logger.info(f"Processing {file_name}")

            slide = openslide.open_slide(slide_path)
            slide_save_dir = save_dir / sample_type / file_name
            slide_save_dir.mkdir(exist_ok=True, parents=True)

            tiler.process(slide, slide_save_dir)
            logger.info(f"Done {file_name}")

            time.sleep(1)
    return True


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    load_dir = pathlib.Path(cfg.data.load_dir)
    save_dir = pathlib.Path(cfg.data.save_dir)

    logger.info("Loading data")
    slides_path = get_slides_path(load_dir)

    with open(get_file_path(load_dir, "biospecimen.cart"), "r") as ouf:
        biospecimen = json.load(ouf)
        case_sample_type_mapping = create_case_sample_type_mapping(biospecimen)

    with open(get_file_path(load_dir, "metadata.cart"), "r") as ouf:
        metadata = json.load(ouf)
        file_case_mapping = create_file_case_mapping(metadata)

    tiler = DeepZoomStaticTiler(
        cfg.params.tile_size,
        cfg.params.overlap,
        cfg.params.quality,
        cfg.params.background_limit,
        cfg.params.limit_bounds,
    )

    slide_queue = Queue()
    logger.info("Preparing data")
    for slide_dir in slides_path:
        slide_queue.put((tiler, slide_dir, file_case_mapping, case_sample_type_mapping, save_dir))

    logger.info("Tiling images")
    processes = []
    for _ in range(cfg.params.num_cpu):
        p = Process(target=process_slide, args=(slide_queue,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
