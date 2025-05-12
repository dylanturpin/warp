# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

###########################################################################
#
# Utility script to (re)generate golden reference ``.npz`` files
# consumed by regression tests.
#
###########################################################################

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import warp as wp

# ─────────────────────────────────────────────────────────────────────────────
# Example runner map
# ─────────────────────────────────────────────────────────────────────────────

import warp.examples.sim.example_cartpole as example_cartpole
import warp.examples.sim.example_quadruped as example_quadruped
import warp.examples.sim.example_cloth_self_contact as example_cloth_self_contact
from warp.tests.unittest_utils import get_test_devices

RUN_EXAMPLE: dict[str, callable] = {
    "quadruped": example_quadruped.run_quadruped,
    "cartpole": example_cartpole.run_cartpole,
    "cloth_self_contact": example_cloth_self_contact.run_cloth_self_contact,
}

# ─────────────────────────────────────────────────────────────────────────────
# Test-case specification (flat list, each dict is self-contained)
# ─────────────────────────────────────────────────────────────────────────────

CASES: list[dict] = [
    # ────────── Cart-pole ──────────
    {
        "example": "cartpole",
        "solver": "euler",
        "policy": "sin",
        "num_frames": 10,
        "num_envs": 8,
    },
    {
        "example": "cartpole",
        "solver": "featherstone",
        "policy": "sin",
        "num_frames": 10,
        "num_envs": 8,
    },
    {
        "example": "cartpole",
        "solver": "xpbd",
        "policy": "sin",
        "num_frames": 10,
        "num_envs": 8,
    },
    # Warp-MuJoCo must run on GPU; golden reference is a live MuJoCo-native run
    {
        "example": "cartpole",
        "solver": "mujoco",
        "policy": "sin",
        "num_frames": 10,
        "num_envs": 2,
        "device_filter": lambda d: d.is_cuda,
    },
    # ────────── Quadruped ──────────
    {
        "example": "quadruped",
        "solver": "featherstone",
        "policy": "sin",
        "num_frames": 10,
        "num_envs": 8,
    },
    {
        "example": "quadruped",
        "solver": "xpbd",
        "policy": "sin",
        "num_frames": 10,
        "num_envs": 8,
    },
    {
        "example": "quadruped",
        "solver": "mujoco",
        "policy": "sin",
        "num_frames": 10,
        "num_envs": 2,
        "device_filter": lambda d: d.is_cuda,
    },
    # ────────── Cloth self-contact ──────────
    {
        "example": "cloth_self_contact",
        "solver": "vbd",
        "policy": "none",
        "num_frames": 10,
        "num_envs": 1,
        "state_arrays": ("particle_q", "particle_qd"),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _devices_for_case(case: dict) -> Sequence[wp.context.Device]:
    """Return the devices this *case* should run on, applying any filter."""
    filt = case.get("device_filter")  # may be None
    return [d for d in get_test_devices() if filt is None or filt(d)]


def golden_path(
    *,
    example_name: str,
    solver_name: str,
    policy: str,
    num_envs: int,
    num_frames: int,
    device_str: str,
) -> Path:
    """Return the full path of the golden reference file for the given parameters."""
    return (
        Path(__file__).parent
        / "assets"
        / "golden_states"
        / f"{example_name}_{solver_name}_{policy}policy_{num_envs}envs_{num_frames}steps_{device_str}.npz"
    )


def main() -> None:  # noqa: D401 – simple CLI entry point
    """Generate golden reference ``.npz`` files for every applicable test case."""
    for case in CASES:
        example = case["example"]
        solver = case["solver"]

        # MuJoCo cases compare against a live MuJoCo-native run instead of a file.
        if solver == "mujoco":
            print(f"[Skip] {example}/mujoco – uses native reference, no golden file")
            continue

        run_example = RUN_EXAMPLE[example]

        for device in _devices_for_case(case):
            dev_str = "cpu" if device.is_cpu else "gpu"

            try:
                state = run_example(
                    solver_name=solver,
                    policy=case.get("policy", "none"),
                    num_frames=case.get("num_frames", 100),
                    num_envs=case.get("num_envs", 1),
                    solver_kwargs=case.get("solver_kwargs"),
                    device=device,
                    stage_path=None,
                    enable_timers=False,
                )
            except Exception as exc:  # noqa: BLE001 – keep loop alive
                print(f"[Skip] {example}/{solver} on {device}: {exc}")
                continue

            out_path = golden_path(
                example_name=example,
                solver_name=solver,
                policy=case.get("policy", "none"),
                num_envs=case.get("num_envs", 1),
                num_frames=case.get("num_frames", 100),
                device_str=dev_str,
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out_path, **state)
            rel_path = out_path.relative_to(Path(__file__).parent)
            print(f"[Saved] {rel_path}")

if __name__ == "__main__":
    wp.init()
    main()
