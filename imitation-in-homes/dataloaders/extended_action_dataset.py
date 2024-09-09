"""
Creating a custom dataset that samples more actions in the future than there are observations.
"""

from typing import List, Tuple, Union

import numpy as np
import torch

from dataloaders.decord_dataset import DecordDataset
from dataloaders.utils import DataLoaderConfig, TrajectorySlice


class ExtendedActionDataset(DecordDataset):
    def __init__(
        self,
        config: DataLoaderConfig,
        num_extra_actions: int = 0,
        only_action_return: bool = False,
        *args,
        **kwargs,
    ):
        self._num_extra_actions = num_extra_actions
        self._k = (config.control_timeskip + 1) * config.fps_subsample
        self._only_action_return = only_action_return
        super().__init__(config, *args, **kwargs)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._total_calls += 1
        # If index is float64, cast to int.
        if isinstance(index, np.float64):
            index = index.astype(np.int64)

        trajectory_slice = self._subslices[index]
        is_padding, actions = self._get_action_slice(trajectory_slice)
        if self._only_action_return:
            return is_padding, actions  # dummy value for obs
        frames = self._get_video_frames(trajectory_slice)
        if self._data_config.use_depth:  # TODO: here make cleaner?
            depths = self._get_depth_frames(trajectory_slice)
            return frames, depths, is_padding, actions
        return frames, is_padding, actions

    def _get_action_slice(
        self, trajectory_slice: TrajectorySlice
    ) -> Union[torch.Tensor, np.ndarray]:
        video_index = trajectory_slice.trajectory_index
        action_indices = self._convert_subslice_to_action_indices(trajectory_slice)
        actionreader = self._get_action_reader(video_index)
        # Now, filter out the indices that are greater than the length of the actionreader.
        num_greater_than_bounds = np.sum(
            (action_indices + self._k) >= len(actionreader)
        )
        if num_greater_than_bounds:
            indices = action_indices[:-num_greater_than_bounds]
        else:
            indices = action_indices
        actions = actionreader.get_batch(indices)
        # And clone the last action for the remaining indices.
        actions = np.concatenate(
            [actions, np.tile(actions[-1], (num_greater_than_bounds, 1))], axis=0
        )
        is_padding = np.zeros(len(actions), dtype=bool)
        if num_greater_than_bounds:
            is_padding[-num_greater_than_bounds:] = True
        return is_padding, actions

    def _convert_subslice_to_action_indices(
        self, subslice: TrajectorySlice
    ) -> Union[List[int], np.ndarray]:
        # Return some extra action indices, some of these will not exist but we will add a
        # mask flag to indicate that.
        subseq_indices = np.arange(
            subslice.start_index,
            subslice.end_index + self._num_extra_actions * subslice.skip,
            subslice.skip,
        )
        return subseq_indices
