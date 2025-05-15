import dataclasses
import jax
import json
import numpy as np
import os
from typing import TypeVar

import openpi.shared.array_typing as at
import openpi.training.config as _config
import openpi.transforms as _transforms
from openpi.models.model import Observation

from .data_loader import create_dataset, TorchDataLoader, DataLoader, TransformedDataset
from .libero_vqa_utils import get_question_template, get_relation_to_robot, find_target_objects, post_process_object


ArrayT = TypeVar("ArrayT", at.Array, jax.ShapeDtypeStruct)

# Real world
_OBJECT_INFO_DIR = '/path/to/real_world_object_info'


@dataclasses.dataclass(frozen=True)
class VQATransform(_transforms.DataTransformFn):
    def __call__(self, data):
        vqa_prompt = ''
        objects = find_target_objects(data['prompt'])

        object_info = data['object_info']
        if not isinstance(object_info, dict):
            object_info_id = np.uint8(object_info.numpy()).tobytes().decode()
            with open(os.path.join(_OBJECT_INFO_DIR, f'{object_info_id}.json'), 'r') as f:
                object_info = json.load(f)
        
        for obj in objects:
            vqa_prompt = vqa_prompt + 'Question: ' + get_question_template('coarse_direction').format(post_process_object(obj)) + ' '
            vqa_prompt = vqa_prompt + 'Answer: ' + get_relation_to_robot(obj, object_info, check_catch=True, check_close=False, mode='coarse_direction') + ';\n'
        
        return {**data, 
                'vqa': vqa_prompt}


def transform_dataset(dataset, data_config, *, skip_norm_stats=False):
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            VQATransform(),
            *data_config.model_transforms.inputs,
        ],
    )


def create_data_loader(
    config,
    *,
    sharding=None,
    skip_norm_stats=False,
    shuffle=False,
    num_batches=None,
    num_workers=0,
):
    data_config = config.data.create(config.assets_dirs, config.model)

    dataset = create_dataset(data_config, config.model)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=config.seed,
    )

    class DataLoaderImpl(DataLoader):
        def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader):
            self._data_config = data_config
            self._data_loader = data_loader

        def data_config(self) -> _config.DataConfig:
            return self._data_config

        def __iter__(self):
            for batch in self._data_loader:
                yield Observation.from_dict(batch), batch["actions"]

    return DataLoaderImpl(data_config, data_loader)
