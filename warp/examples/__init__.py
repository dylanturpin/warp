# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np


def get_source_directory():
    return os.path.realpath(os.path.dirname(__file__))


def get_asset_directory():
    return os.path.join(get_source_directory(), "assets")


def compute_env_offsets(num_envs: int, env_offset: tuple[float, float, float] | list[float] | np.ndarray = (5.0, 0.0, 5.0), up_axis: str | int = "Y") -> np.ndarray:
    """Return a set of per-environment positional offsets that *tile* the environments on a
    regular grid.

    The logic mirrors the helper used in the simulation examples and is useful when you
    want to duplicate a single articulated *builder* many times so that the resulting
    environments do not overlap.

    Parameters
    ----------
    num_envs : int
        Total number of environments that will be instantiated.
    env_offset : (float, float, float)
        Distance between two consecutive environments *along each world axis*.
        A zero component indicates that the corresponding axis should **not** be used
        when laying out the grid.  For instance, ``(1, 0, 2)`` will tile the
        environments in the X–Z plane while keeping all of them at the same Y height.
    up_axis : str | int
        The world-space *up* axis.  Either ``"X"``, ``"Y"`` or ``"Z"`` (case-insensitive)
        or the corresponding index ``0``, ``1`` or ``2``.  The returned offsets are
        centred around the origin, except along the up-axis where they are only shifted
        positively so that the environments are not placed below the ground plane.

    Returns
    -------
    np.ndarray
        Array of shape ``(num_envs, 3)`` containing the positional offsets for each
        environment.
    """

    env_offset = np.asarray(env_offset, dtype=float)

    # Identify the *active* axes, i.e. those with a non-zero spacing value.
    nonzeros = np.nonzero(env_offset)[0]
    num_dim = nonzeros.shape[0]

    # Early-out when no axis is active – all offsets are zero.
    if num_dim == 0:
        return np.zeros((num_envs, 3), dtype=float)

    # Compute the length of one side of the grid in the active sub-space so that we can
    # accommodate *num_envs* environments.
    side_length = int(np.ceil(num_envs ** (1.0 / num_dim)))

    env_offsets: list[np.ndarray] = []

    if num_dim == 1:
        # Line-lay-out along a single axis (e.g. X only).
        for i in range(num_envs):
            env_offsets.append(i * env_offset)

    elif num_dim == 2:
        # 2-D grid in the plane spanned by the two active axes.
        for i in range(num_envs):
            d0 = i // side_length
            d1 = i % side_length
            offset = np.zeros(3, dtype=float)
            offset[nonzeros[0]] = d0 * env_offset[nonzeros[0]]
            offset[nonzeros[1]] = d1 * env_offset[nonzeros[1]]
            env_offsets.append(offset)

    elif num_dim == 3:
        # 3-D brick-like lay-out.
        for i in range(num_envs):
            d0 = i // (side_length * side_length)
            d1 = (i // side_length) % side_length
            d2 = i % side_length
            offset = np.zeros(3, dtype=float)
            offset[0] = d0 * env_offset[0]
            offset[1] = d1 * env_offset[1]
            offset[2] = d2 * env_offset[2]
            env_offsets.append(offset)

    env_offsets = np.asarray(env_offsets, dtype=float)

    # ------------------------------------------------------------------
    # *Centre* the lay-out so that the *median* of all environments lies at the origin.
    # We do this separately along each axis but keep the *up* axis (e.g. Y) fixed at its
    # original height so that no environment ends up below the ground plane.
    # ------------------------------------------------------------------
    min_offsets = np.min(env_offsets, axis=0)
    correction = min_offsets + (np.max(env_offsets, axis=0) - min_offsets) / 2.0

    # Convert *up_axis* to integer index if given as a string.
    if isinstance(up_axis, str):
        up_axis = "XYZ".index(up_axis.upper())
    correction[up_axis] = 0.0

    env_offsets -= correction

    return env_offsets
