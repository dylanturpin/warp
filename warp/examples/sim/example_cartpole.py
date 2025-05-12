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
# Example Sim Cartpole
#
# Shows how to set up a simulation of a rigid-body cartpole articulation
# from a URDF using the wp.sim.ModelBuilder().
# Note this example does not include a trained policy.
#
###########################################################################

import math
import os

import numpy as np

import warp as wp
import warp.examples
import warp.sim
import warp.sim.render

def _featherstone_factory(model, **kwargs):  # noqa: D401 – simple wrapper
    return wp.sim.FeatherstoneIntegrator(model, use_tile_gemm=False, fuse_cholesky=False, **kwargs)


def _xpbd_factory(model, **kwargs):
    return wp.sim.XPBDIntegrator(**kwargs)


def _semiimplicit_factory(model, **kwargs):
    return wp.sim.SemiImplicitIntegrator(**kwargs)


SOLVER_MAP = {
    "featherstone": _featherstone_factory,
    "xpbd": _xpbd_factory,
    "euler": _semiimplicit_factory,
}

class Example:
    def __init__(
        self,
        stage_path = "example_cartpole.usd",
        num_envs = 8,
        solver_cls = None,
        enable_timers = True,
        policy = "none",
        solver_kwargs = None,
    ):
        self.num_envs = num_envs
        self.enable_timers = enable_timers
        self.sim_time = 0.0
        fps = 60
        self.frame_dt = 1.0 / fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.policy = policy.lower()
        if self.policy not in ("none", "sin"):
            raise ValueError(f"Unsupported policy '{policy}'. Valid options are 'none' or 'sin'.")
        
        # sin policy uses joint targets, requires setting setiffness, damping
        if self.policy == "none":
            joint_stiffness = 0.0
            joint_damping = 0.0
        else:
            joint_stiffness = 200.0
            joint_damping = 2.0

        articulation_builder = wp.sim.ModelBuilder()
        wp.sim.parse_urdf(
            os.path.join(warp.examples.get_asset_directory(), "cartpole.urdf"),
            articulation_builder,
            xform=wp.transform(wp.vec3(), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.5)),
            floating=False,
            density=100,
            armature=0.1,
            stiffness=joint_stiffness,
            damping=joint_damping,
            limit_ke=1.0e4,
            limit_kd=1.0e1,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
        )

        builder = wp.sim.ModelBuilder()

        self.sim_time = 0.0
        fps = 60
        self.frame_dt = 1.0 / fps

        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        positions = warp.examples.compute_env_offsets(self.num_envs, env_offset=(1.0, 0.0, 2.0))

        for i in range(self.num_envs):
            builder.add_builder(
                articulation_builder,
                xform=wp.transform(positions[i], wp.quat_identity()),
            )

            # joint initial positions
            builder.joint_q[-3:] = [0.0, 0.3, 0.0]

        # finalize model
        self.model = builder.finalize()
        self.model.ground = False

        self.model.joint_attach_ke = 1600.0
        self.model.joint_attach_kd = 20.0

        if solver_kwargs is None:
            solver_kwargs = {}

        if solver_cls is None:
            solver_cls = SOLVER_MAP["semiimplicit"]

        self.integrator = solver_cls(self.model, **solver_kwargs)

        if stage_path:
            self.renderer = wp.sim.render.SimRenderer(self.model, stage_path, scaling=15.0)
        else:
            self.renderer = None

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        if self.policy == "sin":
            self._init_sin_policy()

        wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state_0)

        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        if self.policy == "sin":
            wp.launch(
                _generate_sin_targets,
                dim=self.control.joint_act.shape[0],
                inputs=[
                    self._base_targets_w,
                    self._amp_w,
                    self._phase_w,
                    self._time_device,
                    self._freq,
                ],
                outputs=[self.control.joint_act],
                device=self.model.device,
            )

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt, self.control)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.policy == "sin":
            wp.copy(self._time_device, wp.array([self.sim_time], dtype=wp.float32, device=self.model.device))

        with wp.ScopedTimer("step", active=self.enable_timers):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", active=self.enable_timers):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()

    def get_state(self) -> dict:
        """Return key tensors as NumPy arrays (for testing / inspection)."""

        def to_numpy(arr):
            return None if arr is None else arr.numpy()

        return {
            "joint_q": to_numpy(self.state_0.joint_q),
            "joint_qd": to_numpy(self.state_0.joint_qd),
            "body_q": to_numpy(self.state_0.body_q),
            "body_qd": to_numpy(self.state_0.body_qd),
        }

    def _init_sin_policy(self) -> None:
        # slider + 2 hinge joints
        joints_per_env = 3

        # initial target matching initial joint_q
        base_single_env = np.array([0.0, 0.3, 0.0], dtype=np.float32)

        # amplitudes:±0.5 units on the slider, ±0.3 rad on hinges.
        amp_single_env = np.array([0.5, 0.3, 0.3], dtype=np.float32)

        # phase offsets (all zero for simplicity).
        phase_single_env = np.zeros(joints_per_env, dtype=np.float32)

        # tile across environments.
        base_np = np.tile(base_single_env, self.num_envs).astype(np.float32)
        amp_np = np.tile(amp_single_env, self.num_envs).astype(np.float32)
        phase_np = np.tile(phase_single_env, self.num_envs).astype(np.float32)

        self._base_targets_w = wp.array(base_np, dtype=wp.float32, device=self.model.device)
        self._amp_w = wp.array(amp_np, dtype=wp.float32, device=self.model.device)
        self._phase_w = wp.array(phase_np, dtype=wp.float32, device=self.model.device)

        self._time_device = wp.array([0.0], dtype=wp.float32, device=self.model.device)

        self._freq: float = 1.0

        self.control.joint_act = wp.array(base_np, dtype=wp.float32, device=self.model.device)

@wp.kernel
def _generate_sin_targets(
    base: wp.array(dtype=float),
    amp: wp.array(dtype=float),
    phase: wp.array(dtype=float),
    time_val: wp.array(dtype=float),
    freq: float,
    out: wp.array(dtype=float),
):
    tid = wp.tid()
    t = time_val[0]
    out[tid] = base[tid] + amp[tid] * wp.sin(2.0 * wp.pi * freq * t + phase[tid])

def run_cartpole(
    solver_name = "semiimplicit",
    num_frames = 1200,
    num_envs = 8,
    device = None,
    stage_path = None,
    enable_timers = True,
    policy = "none",
    solver_kwargs = None,
):
    solver_factory = SOLVER_MAP.get(solver_name.lower())
    if solver_factory is None:
        raise ValueError(f"Unknown solver '{solver_name}'. Valid keys: {list(SOLVER_MAP)}")

    with wp.ScopedDevice(device):
        example = Example(
            stage_path=stage_path,
            num_envs=num_envs,
            solver_cls=solver_factory,
            enable_timers=enable_timers,
            policy=policy,
            solver_kwargs=solver_kwargs,
        )

        for _ in range(num_frames):
            example.step()
            if example.renderer:
                example.render()

        # Save USD if we rendered frames
        if example.renderer:
            example.renderer.save()

        return example.get_state()




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="example_cartpole.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=100, help="Total number of frames.")
    parser.add_argument("--num_envs", type=int, default=8, help="Total number of simulated environments.")
    parser.add_argument(
        "--solver",
        type=str,
        default="euler",
        choices=list(SOLVER_MAP.keys()),
        help="Which integrator to use for the simulation.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="none",
        choices=("none", "sin"),
        help="Which control policy to apply during the simulation.",
    )

    args = parser.parse_known_args()[0]

    state = run_cartpole(
        solver_name=args.solver,
        num_frames=args.num_frames,
        num_envs=args.num_envs,
        device=args.device,
        stage_path=args.stage_path,
        enable_timers=True,
        policy=args.policy,
    )