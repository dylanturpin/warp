# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact launch metadata, retained resources, and fail-closed graph preparation."""

import ctypes
import dataclasses
import gc
import operator
import unittest
import weakref
from unittest import mock

import numpy as np

import warp as wp
from warp._src import context
from warp._src.types import launch_bounds_t
from warp.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices


@wp.kernel
def write_value(values: wp.array[int], value: int):
    values[wp.tid()] = value


@wp.struct
class ArrayMember:
    values: wp.array[int]


@wp.struct
class NestedMember:
    member: ArrayMember


@wp.kernel
def write_member(value: NestedMember):
    value.member.values[wp.tid()] = 11


@wp.kernel(module="unique")
def copy_tile(source: wp.array[float], destination: wp.array[float]):
    index = wp.tid()
    value = wp.tile_load(source, shape=4, offset=index * 4)
    wp.tile_store(destination, value, offset=index * 4)


@wp.kernel
def stop_loop(condition: wp.array[int], output: wp.array[int]):
    condition[0] = 0
    output[0] = 17


@wp.kernel(module="unique", module_options={"deterministic": wp.DeterministicMode.RUN_TO_RUN})
def deterministic_scatter(source: wp.array[float], indices: wp.array[int], destination: wp.array[float]):
    index = wp.tid()
    wp.atomic_add(destination, indices[index], source[index])


@wp.kernel
def write_index(values: wp.array[int], n: int):
    i = wp.tid()
    values[i] = i + 1 + n * 100


@wp.kernel
def write_scaled(source: wp.array[float], destination: wp.array[float], n: int):
    i = wp.tid()
    destination[i] = source[i] * float(n) + 0.5


@wp.kernel
def write_grid(values: wp.array2d[int], n: int):
    i, j = wp.tid()
    values[i, j] = n


@wp.kernel(grid_stride=False, module="unique")
def write_lean(values: wp.array[int], n: int):
    i = wp.tid()
    values[i] = n + 1


@wp.kernel
def read_reset(reset_counts: wp.array[int], reset_index: wp.array[int], contact_count: wp.array[int]):
    contact_count[0] = reset_counts[reset_index[0]]


@wp.kernel
def narrowphase(count: int, contacts: wp.array[int], touched: wp.array[int]):
    i = wp.tid()
    contacts[i] = count * 1000 + i
    touched[i] = touched[i] + 1


@wp.kernel
def write_pair(values: wp.array[int], first: int, second: int):
    values[wp.tid()] = first * 10 + second


@wp.kernel
def latch_errors(errors: wp.array[int], latch: wp.array[int]):
    i = wp.tid()
    if errors[i] != 0:
        latch[0] = 1


@wp.kernel
def write_volume(values: wp.array3d[int], n: int):
    i, j, k = wp.tid()
    values[i, j, k] = n


def test_capture_packets(test, device):
    with wp.ScopedDevice(device):
        values = wp.zeros(8, dtype=int)
        wp.load_module(device=device)
        with wp.ScopedCapture(record_launches=True) as capture:
            wp.launch(write_value, 8, [values, 3])
            wp.launch(write_value, 0, [values, 99])
        records = wp.capture_get_launches(capture.graph)
        test.assertEqual(len(records), 1)
        record = records[0]
        test.assertIs(record.kernel, write_value)
        test.assertEqual(record.dim, (8,))
        test.assertEqual(record.graph_handle, capture.graph.graph.value)
        test.assertTrue(record.node_handle)
        test.assertEqual(ctypes.c_int.from_buffer_copy(record.parameter_bytes[2]).value, 3)
        bounds = type(launch_bounds_t((0,))).from_buffer_copy(record.parameter_bytes[0])
        test.assertEqual(tuple(bounds.shape), (8,))
        test.assertEqual(bounds.size, 8)
        test.assertTrue(any(owner is values for owner in record.argument_owners))
        test.assertIs(wp.capture_get_launches(capture.graph)[0], record)
        with test.assertRaises(dataclasses.FrozenInstanceError):
            record.dim = (1,)
        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(values.numpy(), 3)


def test_reusable_snapshots(test, device):
    with wp.ScopedDevice(device):
        values = wp.zeros(8, dtype=int)
        first = wp.launch(write_value, 8, [values, 2], record_cmd=True)
        second = wp.zeros_like(values)
        with wp.ScopedCapture(record_launches=True) as capture:
            first.launch()
            first.set_dim(4)
            first.set_param_at_index(0, second)
            first.set_param_at_index(1, 7)
            first.launch()
            first.set_dim(0)
            first.launch()
        records = wp.capture_get_launches(capture.graph)
        test.assertEqual(len(records), 2)
        test.assertEqual([record.dim for record in records], [(8,), (4,)])
        test.assertEqual([ctypes.c_int.from_buffer_copy(r.parameter_bytes[2]).value for r in records], [2, 7])
        test.assertNotEqual(records[0].node_handle, records[1].node_handle)
        test.assertTrue(any(owner is values for owner in records[0].argument_owners))
        test.assertTrue(any(owner is second for owner in records[1].argument_owners))
        before = records[1].parameter_bytes
        first.set_param_at_index(1, 99)
        test.assertEqual(records[1].parameter_bytes, before)
        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(values.numpy(), 2)
        np.testing.assert_allclose(second.numpy(), [7, 7, 7, 7, 0, 0, 0, 0])


def test_struct_resource_lifetime(test, device):
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        with wp.ScopedDevice(device):
            original = wp.zeros(2, dtype=int)
            old_owner = weakref.ref(original)
            nested = NestedMember()
            nested.member.values = original
            wp.load_module(device=device)
            with wp.ScopedCapture(record_launches=True) as capture:
                wp.launch(write_member, 2, [nested])
            record = wp.capture_get_launches(capture.graph)[0]
            packet = record.parameter_bytes
            nested.member.values = wp.zeros_like(original)
            del original
            test.assertIsNotNone(old_owner())
            test.assertEqual(record.parameter_bytes, packet)
            wp.capture_launch(capture.graph)
            np.testing.assert_allclose(old_owner().numpy(), 11)
            np.testing.assert_allclose(nested.member.values.numpy(), 0)
            del record
            test.assertIsNotNone(old_owner())
            capture.graph = None
            test.assertIsNone(old_owner())
    finally:
        if was_enabled:
            gc.enable()


def test_tiled_compiled_bounds(test, device):
    with wp.ScopedDevice(device):
        source = wp.array(np.arange(12, dtype=np.float32))
        destination = wp.zeros_like(source)
        wp.launch_tiled(copy_tile, dim=3, inputs=[source, destination], block_dim=32)
        with wp.ScopedCapture(record_launches=True) as capture:
            wp.launch_tiled(copy_tile, dim=3, inputs=[source, destination], block_dim=32)
        (record,) = wp.capture_get_launches(capture.graph)
        test.assertEqual(record.dim, (3,))
        bounds = type(launch_bounds_t((0,))).from_buffer_copy(record.parameter_bytes[0])
        test.assertEqual(bounds.coord_mult, 32)
        test.assertEqual(bounds.size, 96)
        test.assertEqual(record.block_dim, 32)
        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(destination.numpy(), source.numpy())


def test_joined_streams(test, device):
    with wp.ScopedDevice(device):
        values = wp.zeros(8, dtype=int)
        other = wp.zeros_like(values)
        wp.load_module(device=device)
        main, side = wp.Stream(device), wp.Stream(device)
        fork, join = wp.Event(device), wp.Event(device)
        with wp.ScopedStream(main, sync_enter=False, sync_exit=False):
            with wp.ScopedCapture(stream=main, record_launches=True) as capture:
                wp.launch(write_value, 8, [values, 4], stream=main)
                main.record_event(fork)
                side.wait_event(fork)
                wp.launch(write_value, 8, [other, 5], stream=side)
                side.record_event(join)
                main.wait_event(join)
        records = wp.capture_get_launches(capture.graph)
        test.assertEqual(len(records), 2)
        test.assertEqual({r.graph_handle for r in records}, {capture.graph.graph.value})
        test.assertNotEqual(records[0].node_handle, records[1].node_handle)
        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(values.numpy(), 4)
        np.testing.assert_allclose(other.numpy(), 5)


def test_conditional_body_records(test, device):
    if not wp.is_conditional_graph_supported():
        test.skipTest("Conditional graphs require CUDA 12.4+")
    with wp.ScopedDevice(device):
        values = wp.zeros(1, dtype=int)
        condition = wp.ones(1, dtype=int)
        wp.load_module(device=device)
        with wp.ScopedCapture(record_launches=True) as capture:
            wp.capture_if(condition, lambda: wp.launch(write_value, 1, [values, 13]))
            wp.capture_while(condition, lambda: wp.launch(stop_loop, 1, [condition, values]))
        records = wp.capture_get_launches(capture.graph)
        test.assertEqual(len(records), 2)
        test.assertTrue(all(r.graph_handle != capture.graph.graph.value for r in records))
        test.assertNotEqual(records[0].graph_handle, records[1].graph_handle)
        bodies = wp.capture_get_conditional_bodies(capture.graph)
        test.assertEqual(len(bodies), 2)
        test.assertEqual({body.body_graph_handle for body in bodies}, {record.graph_handle for record in records})
        test.assertTrue(all(body.graph_handle == capture.graph.graph.value and body.node_handle for body in bodies))
        test.assertIs(wp.capture_get_conditional_bodies(capture.graph)[0], bodies[0])
        with test.assertRaises(dataclasses.FrozenInstanceError):
            bodies[0].body_graph_handle = 0
        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(values.numpy(), 17)


def test_nested_conditional_containment(test, device):
    if not wp.is_conditional_graph_supported():
        test.skipTest("Conditional graphs require CUDA 12.4+")
    with wp.ScopedDevice(device):
        values = wp.zeros(1, dtype=int)
        outer, condition = wp.ones(1, dtype=int), wp.ones(1, dtype=int)
        wp.load_module(device=device)
        with wp.ScopedCapture(record_launches=True) as capture:
            wp.capture_if(
                outer,
                lambda: wp.capture_while(condition, lambda: wp.launch(stop_loop, 1, [condition, values])),
                lambda: wp.launch(write_value, 1, [values, 29]),
            )
        bodies = wp.capture_get_conditional_bodies(capture.graph)
        test.assertEqual(len(bodies), 3)
        test.assertEqual(len({body.body_graph_handle for body in bodies}), 3)
        roots = [body for body in bodies if body.graph_handle == capture.graph.graph.value]
        test.assertEqual(len(roots), 2)
        nested = next(body for body in bodies if body not in roots)
        test.assertIn(nested.graph_handle, {body.body_graph_handle for body in roots})
        loop_record = next(record for record in wp.capture_get_launches(capture.graph) if record.kernel is stop_loop)
        test.assertEqual(loop_record.graph_handle, nested.body_graph_handle)
        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(values.numpy(), 17)
        outer.zero_()
        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(values.numpy(), 29)


def test_caught_launch_failure(test, device):
    with wp.ScopedDevice(device):
        values = wp.zeros(1, dtype=int)
        wp.load_module(device=device)
        for failing in (
            lambda: wp.launch(write_value, -1, [values, 1]),
            lambda: wp.launch(write_value, 1, [values]),
            lambda: wp.launch(write_value, 1, [values, 1], adjoint=True),
            lambda: wp.launch_tiled(write_value, inputs=[values, 1], block_dim=32),
            lambda: wp.launch(write_value, 1, [values.__ctype__(), 1]),
        ):
            with test.subTest(operation=failing):
                with wp.ScopedCapture(record_launches=True) as capture:
                    wp.launch(write_value, 1, [values, 0])
                    with test.assertRaises((RuntimeError, ValueError)):
                        failing()
                with test.assertRaisesRegex(RuntimeError, "invalid|incomplete"):
                    wp.capture_get_launches(capture.graph)
                with test.assertRaisesRegex(RuntimeError, "invalid|incomplete"):
                    wp.capture_launch(capture.graph)


def test_post_emission_failure(test, device):
    with wp.ScopedDevice(device):
        values = wp.zeros(1, dtype=int)
        wp.load_module(device=device)
        with wp.ScopedCapture(record_launches=True) as capture:
            with mock.patch.object(context, "_record_captured_launch", side_effect=RuntimeError("snapshot failure")):
                with test.assertRaisesRegex(RuntimeError, "snapshot failure"):
                    wp.launch(write_value, 1, [values, 6])
        with test.assertRaisesRegex(RuntimeError, "incomplete"):
            wp.capture_launch(capture.graph)


def test_invalidation_after_instantiation(test, device):
    with wp.ScopedDevice(device):
        values = wp.zeros(1, dtype=int)
        wp.load_module(device=device)
        with wp.ScopedCapture(record_launches=True) as capture:
            wp.launch(write_value, 1, [values, 1])
        wp.capture_launch(capture.graph)
        test.assertIsNotNone(capture.graph.graph_exec)
        wp.capture_invalidate(capture.graph, "consumer lowering failed")
        wp.capture_invalidate(capture.graph, "later error")
        with test.assertRaisesRegex(RuntimeError, "consumer lowering failed"):
            wp.capture_launch(capture.graph)


def test_preparation_obligations(test, device):
    class Owner:
        def __eq__(self, other):
            raise AssertionError("Preparation must compare identity")

    with wp.ScopedDevice(device):
        values = wp.zeros(1, dtype=int)
        wp.load_module(device=device)
        first, second = Owner(), Owner()
        with wp.ScopedCapture(record_launches=True) as capture:
            active = device.captures[wp.get_stream(device)]
            wp.capture_require_preparation(active, first)
            wp.capture_require_preparation(active, first)
            wp.capture_require_preparation(active, second)
            wp.launch(write_value, 1, [values, 8])
            with test.assertRaisesRegex(RuntimeError, "Pause or finish graph capture"):
                wp.capture_get_launches(active)
        test.assertEqual(len(wp.capture_get_launches(capture.graph)), 1)
        with test.assertRaisesRegex(RuntimeError, "unresolved preparation"):
            wp.capture_launch(capture.graph)
        with test.assertRaises(ValueError):
            wp.capture_complete_preparation(capture.graph, Owner())
        wp.capture_complete_preparation(capture.graph, first)
        with test.assertRaisesRegex(RuntimeError, "unresolved preparation"):
            wp.capture_launch(capture.graph)
        wp.capture_complete_preparation(capture.graph, second)
        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(values.numpy(), 8)
        with test.assertRaisesRegex(RuntimeError, "instantiated"):
            wp.capture_require_preparation(capture.graph, first)


def test_child_insertion_guards(test, device):
    with wp.ScopedDevice(device):
        values = wp.zeros(1, dtype=int)
        condition = wp.ones(1, dtype=int)
        wp.load_module(device=device)
        with wp.ScopedCapture() as child:
            wp.launch(write_value, 1, [values, 1])
        for insert in (
            lambda: wp.capture_launch(child.graph),
            lambda: wp.capture_if(condition, child.graph),
            lambda: wp.capture_while(condition, child.graph),
        ):
            with test.subTest(operation=insert):
                with wp.ScopedCapture(record_launches=True) as parent:
                    with test.assertRaisesRegex(RuntimeError, "prebuilt"):
                        insert()
                with test.assertRaisesRegex(RuntimeError, "invalid"):
                    wp.capture_launch(parent.graph)
        wp.capture_invalidate(child.graph, "bad child")
        with wp.ScopedCapture() as parent:
            with test.assertRaisesRegex(RuntimeError, "bad child"):
                wp.capture_launch(child.graph)
        with test.assertRaisesRegex(RuntimeError, "bad child"):
            wp.capture_launch(parent.graph)


def test_command_descriptor_mutation(test, device):
    with wp.ScopedDevice(device):
        nested = NestedMember()
        nested.member.values = wp.zeros(2, dtype=int)
        command = wp.launch(write_member, 2, [nested], record_cmd=True)
        # Struct ctypes can alias the command packet; owner retention still has
        # to reflect the actual packet at launch, not command construction.
        nested.member.values = wp.zeros(2, dtype=int)
        with wp.ScopedCapture(record_launches=True) as capture:
            command.launch()
        (record,) = wp.capture_get_launches(capture.graph)
        test.assertTrue(any(owner is nested.member.values for owner in record.argument_owners))
        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(nested.member.values.numpy(), 11)


def test_overlapping_recording_fails_closed(test, device):
    with wp.ScopedDevice(device):
        with wp.ScopedCapture(record_launches=True) as capture:
            graph = context._begin_launch_recording(device, wp.get_stream(device))
            with test.assertRaisesRegex(RuntimeError, "overlapped"):
                context._begin_launch_recording(device, wp.get_stream(device))
            # The first operation's successful completion cannot unpoison a
            # graph after a second operation encountered that active launch.
            graph._launch_recording_incomplete = False
        with test.assertRaisesRegex(RuntimeError, "overlapped"):
            wp.capture_launch(capture.graph)


def test_deterministic_command_rejected(test, device):
    with wp.ScopedDevice(device):
        source = wp.ones(4, dtype=float)
        indices = wp.zeros(4, dtype=int)
        destination = wp.zeros(1, dtype=float)
        command = wp.launch(deterministic_scatter, 4, [source, indices, destination], record_cmd=True)
        test.assertEqual(type(command).__name__, "DeterministicLaunch")
        with wp.ScopedCapture(record_launches=True) as capture:
            with test.assertRaisesRegex(RuntimeError, "deterministic"):
                command.launch()
        with test.assertRaisesRegex(RuntimeError, "invalid"):
            wp.capture_launch(capture.graph)


def test_paused_capture_prefix(test, device):
    with wp.ScopedDevice(device):
        values = wp.zeros(1, dtype=int)
        wp.load_module(device=device)
        with wp.ScopedCapture(record_launches=True) as capture:
            wp.launch(write_value, 1, [values, 2])
            graph = context.capture_pause()
            try:
                prefix = wp.capture_get_launches(graph)
                test.assertEqual(len(prefix), 1)
            finally:
                context.capture_resume(graph)
            wp.launch(write_value, 1, [values, 3])
        finished = capture.graph
        records = wp.capture_get_launches(finished)
        test.assertIs(finished, graph)
        test.assertEqual(len(records), 2)
        test.assertEqual(len(prefix), 1)
        test.assertIs(prefix[0], records[0])
        wp.capture_launch(finished)
        np.testing.assert_allclose(values.numpy(), 3)


def test_capture_launch_count(test, device):
    with wp.ScopedDevice(device):
        values = wp.zeros(1, dtype=int)
        wp.load_module(device=device)
        with wp.ScopedCapture(record_launches=True) as capture:
            active = device.captures[wp.get_stream(device)]
            test.assertEqual(wp.capture_launch_count(active), 0)
            with test.assertRaisesRegex(RuntimeError, "finish graph capture"):
                wp.capture_get_launches(active)
            wp.launch(write_value, 0, [values, 1])
            test.assertEqual(wp.capture_launch_count(active), 0)
            wp.launch(write_value, 1, [values, 2])
            start = wp.capture_launch_count(active)
            test.assertEqual(start, 1)
            with mock.patch.object(active, "_launch_recording_incomplete", True):
                with test.assertRaisesRegex(RuntimeError, "incomplete"):
                    wp.capture_launch_count(active)
            wp.launch(write_value, 1, [values, 3])
            end = wp.capture_launch_count(active)
            test.assertEqual(end, 2)
        records = wp.capture_get_launches(capture.graph)
        test.assertEqual(wp.capture_launch_count(capture.graph), len(records))
        test.assertEqual(len(records[start:end]), 1)
        test.assertEqual(ctypes.c_int.from_buffer_copy(records[start].parameter_bytes[2]).value, 3)
        wp.capture_invalidate(capture.graph, "invalid count inspection")
        with test.assertRaisesRegex(RuntimeError, "invalid count inspection"):
            wp.capture_launch_count(capture.graph)


def test_default_capture_unchanged(test, device):
    with wp.ScopedDevice(device):
        values = wp.zeros(1, dtype=int)
        wp.load_module(device=device)
        with wp.ScopedCapture() as capture:
            wp.launch(write_value, 1, [values, 10])
        test.assertEqual(wp.capture_launch_count(capture.graph), 0)
        test.assertEqual(wp.capture_get_launches(capture.graph), ())
        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(values.numpy(), 10)
        with test.assertRaisesRegex(ValueError, "without APIC"):
            wp.capture_begin(apic=True, record_launches=True)


def test_captured_memory_values(test, device):
    with wp.ScopedDevice(device):
        byte_vector = wp.types.vector(length=3, dtype=wp.uint8)
        examples = (
            (wp.float32, -0.0),
            (wp.float64, -0.0),
            (wp.int16, -123),
            (wp.bool, True),
            (wp.vec3, (1.0, -0.0, 3.0)),
            (byte_vector, (1, 2, 3)),
        )
        for dtype, value in examples:
            with test.subTest(dtype=dtype):
                source = wp.empty((3, 4), dtype=dtype)
                destination = wp.empty_like(source)
                reference = wp.full((3, 4), value, dtype=dtype, device="cpu").numpy().tobytes()
                with wp.ScopedCapture(record_launches=True, record_memory_operations=True) as capture:
                    source.fill_(value)
                    wp.copy(destination, source)
                fill, copy = wp.capture_get_memory_operations(capture.graph)
                launches = wp.capture_get_launches(capture.graph)
                test.assertIsInstance(fill, wp.CapturedFill)
                test.assertIsInstance(copy, wp.CapturedCopy)
                test.assertIs(fill.launch, launches[0])
                test.assertIs(copy.launch, launches[1])
                test.assertEqual(fill.array_dtype, dtype)
                test.assertEqual(fill.array_descriptor, bytes(source.__ctype__()))
                test.assertEqual(copy.destination_descriptor, bytes(destination.__ctype__()))
                test.assertEqual(fill.value_bytes, reference[: wp.types.type_size_in_bytes(dtype)])
                test.assertEqual(fill.launch.dim[0], 3)
                with test.assertRaises(dataclasses.FrozenInstanceError):
                    fill.value_bytes = b""
                wp.capture_launch(capture.graph)
                test.assertEqual(destination.numpy().tobytes(), reference)


def test_captured_memory_strided_rows(test, device):
    with wp.ScopedDevice(device):
        # Packed fields have overlapping address spans but disjoint row payloads.
        storage = wp.full((4, 12), wp.uint8(211), dtype=wp.uint8)
        left = wp.array(ptr=storage.ptr, shape=(4, 3), strides=(12, 1), dtype=wp.uint8, device=device)
        right = wp.array(ptr=storage.ptr + 4, shape=(4, 3), strides=(12, 1), dtype=wp.uint8, device=device)
        with wp.ScopedCapture(record_launches=True, record_memory_operations=True) as capture:
            left.fill_(wp.uint8(37))
            wp.copy(right, left)
        fill, copy = wp.capture_get_memory_operations(capture.graph)
        test.assertEqual(fill.launch.dim, (4, 3))  # Byte path, including odd row widths.
        test.assertEqual(copy.launch.dim, (4, 3))
        wp.capture_launch(capture.graph)
        expected = np.full((4, 12), 211, np.uint8)
        expected[:, :3], expected[:, 4:7] = 37, 37
        np.testing.assert_array_equal(storage.numpy(), expected)


def test_captured_memory_zero_and_noops(test, device):
    with wp.ScopedDevice(device):
        values = wp.full((2, 5), 9.0, dtype=float)
        empty = wp.empty((0, 5), dtype=float)
        with wp.ScopedCapture(record_launches=True, record_memory_operations=True) as capture:
            values.zero_()
            wp.copy(values, values)
            empty.zero_()
            wp.copy(empty, empty)
            active = device.captures[wp.get_stream(device)]
            with test.assertRaisesRegex(RuntimeError, "Pause or finish"):
                wp.capture_get_memory_operations(active)
        (fill,) = wp.capture_get_memory_operations(capture.graph)
        test.assertEqual(fill.value_bytes, bytes(4))
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(values.numpy(), np.zeros((2, 5), np.float32))
        with wp.ScopedCapture(record_launches=True) as ordinary:
            values.fill_(3.0)
            wp.copy(values, values)
        test.assertEqual(wp.capture_get_memory_operations(ordinary.graph), ())
        wp.capture_launch(ordinary.graph)
        np.testing.assert_array_equal(values.numpy(), np.full((2, 5), 3.0, np.float32))


def test_captured_memory_owners_and_snapshots(test, device):
    with wp.ScopedDevice(device):
        source = wp.full((3, 2), 4, dtype=int)
        destination = wp.empty_like(source)
        source_ref, destination_ref = weakref.ref(source), weakref.ref(destination)
        with wp.ScopedCapture(record_launches=True, record_memory_operations=True) as capture:
            wp.copy(destination, source)
        (record,) = wp.capture_get_memory_operations(capture.graph)
        saved = record.destination_descriptor
        destination.shape = (6,)
        test.assertEqual(record.destination_descriptor, saved)
        destination.shape = (3, 2)
        del source, destination
        gc.collect()
        test.assertIsNotNone(source_ref())
        test.assertIsNotNone(destination_ref())
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(destination_ref().numpy(), np.full((3, 2), 4, np.int32))


def test_captured_memory_rejects_unsupported(test, device):
    with wp.ScopedDevice(device):
        source = wp.zeros((4, 4), dtype=float)
        destination = wp.empty_like(source)
        host = wp.empty(source.shape, dtype=float, device="cpu")
        inner_strided = source[:, ::2]
        overlap = wp.array(ptr=source.ptr + 4, shape=(4, 3), strides=(16, 4), dtype=float, device=device)
        left = wp.array(ptr=source.ptr, shape=(4, 3), strides=(16, 4), dtype=float, device=device)
        differentiated = wp.empty(source.shape, dtype=float, requires_grad=True)
        operations = (
            (lambda: wp.copy(destination, source, count=2), "full-array"),
            (lambda: wp.copy(destination, source, dest_offset=1), "full-array"),
            (lambda: wp.copy(host, source), "captured CUDA device"),
            (inner_strided.zero_, "contiguous inner"),
            (lambda: wp.copy(overlap, left), "overlapping copy"),
            (differentiated.zero_, "autodiff"),
            (lambda: destination.fill_(object()), "Failed to convert"),
        )
        for operation, message in operations:
            with test.subTest(message=message):
                with wp.ScopedCapture(record_launches=True, record_memory_operations=True) as capture:
                    with test.assertRaisesRegex((ValueError, NotImplementedError), message):
                        operation()
                with test.assertRaisesRegex(RuntimeError, "invalid"):
                    wp.capture_launch(capture.graph)


def test_captured_memory_streams_and_conditions(test, device):
    if not wp.is_conditional_graph_supported():
        test.skipTest("Conditional graphs require CUDA 12.4+")
    with wp.ScopedDevice(device):
        source = wp.zeros((3, 2), dtype=int)
        destination = wp.empty_like(source)
        condition = wp.ones(1, dtype=int)
        main, side = wp.Stream(device), wp.Stream(device)
        fork, join = wp.Event(device), wp.Event(device)
        with wp.ScopedStream(main, sync_enter=False, sync_exit=False):
            with wp.ScopedCapture(stream=main, record_launches=True, record_memory_operations=True) as capture:
                wp.capture_if(condition, lambda: source.fill_(27))
                main.record_event(fork)
                side.wait_event(fork)
                wp.copy(destination, source, stream=side)
                side.record_event(join)
                main.wait_event(join)
        fill, copy = wp.capture_get_memory_operations(capture.graph)
        test.assertNotEqual(fill.launch.graph_handle, capture.graph.graph.value)
        test.assertEqual(copy.launch.graph_handle, capture.graph.graph.value)
        test.assertEqual(len(wp.capture_get_launches(capture.graph)), 2)
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(destination.numpy(), np.full((3, 2), 27, np.int32))


def test_captured_memory_rejects_stale_array_abi(test, device):
    with wp.ScopedDevice(device):
        array = wp.zeros((4, 4), dtype=float)
        other = wp.empty_like(array)
        array.__ctype__()
        for name, value in (("ptr", other.ptr), ("shape", (2, 8)), ("strides", (32, 4)), ("size", 8)):
            with test.subTest(attribute=name):
                previous = getattr(array, name)
                try:
                    setattr(array, name, value)
                    with wp.ScopedCapture(record_launches=True, record_memory_operations=True) as capture:
                        with test.assertRaisesRegex(ValueError, "descriptor changed"):
                            array.fill_(3.0)
                    with test.assertRaisesRegex(RuntimeError, "invalid"):
                        wp.capture_launch(capture.graph)
                finally:
                    setattr(array, name, previous)


def test_count_parameter_occurrences_and_obligations(test, device):
    with wp.ScopedDevice(device):
        first, second = wp.CountParameter(8), wp.CountParameter(8)
        values = wp.zeros(8, dtype=int)
        condition = wp.ones(1, dtype=int)
        wp.load_module(device=device)
        with wp.ScopedCapture(record_launches=True) as capture:
            wp.launch(write_value, first, [values, second])
            wp.capture_if(condition, lambda: wp.launch(write_value, second, [values, first]))
        records = wp.capture_get_launches(capture.graph)
        test.assertEqual(len(records), 2)
        for record, extent, scalar in ((records[0], first, second), (records[1], second, first)):
            test.assertEqual(record.extent_parameters[0][0], 0)
            test.assertIs(record.extent_parameters[0][1], extent)
            test.assertEqual(record.scalar_parameters[0][0], 2)
            test.assertIs(record.scalar_parameters[0][1], scalar)
            test.assertEqual(ctypes.c_int32.from_buffer_copy(record.parameter_bytes[2]).value, 8)
        test.assertEqual(len(capture.graph._preparation_owners), 2)
        with test.assertRaisesRegex(RuntimeError, "preparation"):
            wp.capture_launch(capture.graph)
        wp.capture_complete_preparation(capture.graph, first)
        with test.assertRaisesRegex(RuntimeError, "preparation"):
            wp.capture_launch(capture.graph)
        # This test's consumer binds both operands to their captured maxima.
        wp.capture_complete_preparation(capture.graph, second)
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(values.numpy(), 8)


def test_count_parameter_tiled_and_prefix_memory(test, device):
    with wp.ScopedDevice(device):
        rows, value = wp.CountParameter(3), wp.CountParameter(7)
        source = wp.array(np.arange(20, dtype=np.float32))
        target = wp.full(20, -1.0, dtype=float)
        array = wp.full((8, 2), -1, dtype=int)
        copied = wp.full((10, 2), -2, dtype=int)
        optional = wp.empty((8, 0), dtype=float)
        wp.launch_tiled(copy_tile, dim=1, inputs=[source, target], block_dim=32)
        with wp.ScopedCapture(record_launches=True, record_memory_operations=True) as capture:
            wp.launch_tiled(copy_tile, dim=rows, inputs=[source, target], block_dim=32)
            array.fill_(value, extent=(rows, 2))
            wp.copy(copied, array, extent=(rows, 2))
            optional.zero_(extent=(wp.CountParameter(8), 0))
        records = wp.capture_get_launches(capture.graph)
        operations = wp.capture_get_memory_operations(capture.graph)
        test.assertEqual(len(records), 3)
        test.assertEqual(len(operations), 2)
        test.assertIs(records[0].extent_parameters[0][1], rows)
        bounds = type(launch_bounds_t((0,))).from_buffer_copy(records[0].parameter_bytes[0])
        test.assertEqual(bounds.coord_mult, 32)
        for operation in operations:
            test.assertIs(operation.extent[0], rows)
        test.assertIs(operations[0].value_parameter, value)
        test.assertIs(records[1].scalar_parameters[0][1], value)
        test.assertEqual(len(capture.graph._preparation_owners), 2)
        wp.capture_complete_preparation(capture.graph, rows)
        wp.capture_complete_preparation(capture.graph, value)
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(array.numpy()[:3], 7)
        np.testing.assert_array_equal(array.numpy()[3:], -1)
        np.testing.assert_array_equal(copied.numpy()[:3], 7)
        np.testing.assert_array_equal(copied.numpy()[3:], -2)
        np.testing.assert_array_equal(target.numpy()[:12], source.numpy()[:12])


def test_count_parameter_rejects_unrecorded_and_wrong_scalar(test, device):
    with wp.ScopedDevice(device):
        count = wp.CountParameter(4)
        values = wp.zeros(8, dtype=int)
        unaligned = wp.array(ptr=values.ptr + 1, shape=(4,), dtype=int, device=device)
        wp.load_module(device=device)
        for operation in (
            lambda: wp.launch(write_value, count, [values, 3]),
            lambda: values.fill_(count),
            lambda: values.zero_(extent=(count,)),
        ):
            with test.assertRaisesRegex(RuntimeError, "recorded"):
                operation()
        for operation in (
            lambda: wp.launch(write_value, count, [values, 3], record_cmd=True),
            lambda: wp.launch(write_value, (8, count), [values, 3]),
            lambda: values.zero_(extent=(wp.CountParameter(9),)),
            lambda: unaligned.fill_(count),
        ):
            with wp.ScopedCapture(record_launches=True, record_memory_operations=True) as capture:
                with test.assertRaises((RuntimeError, ValueError, NotImplementedError)):
                    operation()
            with test.assertRaises(RuntimeError):
                wp.capture_launch(capture.graph)


def _count_update_supported(test):
    if not wp.is_conditional_graph_supported():
        test.skipTest("Count updates require CUDA 12.4+")


def test_count_update_replay_zero_to_positive(test, device):
    _count_update_supported(test)
    with wp.ScopedDevice(device):
        count = wp.CountParameter(8)
        source = wp.zeros(1, dtype=wp.int32)
        values = wp.zeros(8, dtype=int)
        wp.load_module(device=device)
        with wp.ScopedCapture(record_launches=True) as capture:
            wp.capture_update_counts([count], [source])
            wp.launch(write_index, count, [values, count])
        graph = capture.graph
        records = wp.capture_get_launches(graph)
        test.assertEqual(len(records), 1)
        test.assertEqual(wp.capture_get_count_bound_launches(graph), records)
        test.assertEqual(wp.capture_launch_count(graph), 1)
        with test.assertRaises(ValueError):
            wp.capture_complete_preparation(graph, count)  # Warp discharged it
        test.assertIsNone(graph.graph_exec)
        other_stream = wp.Stream(device)
        exec_handle = None
        # The first replay reads zero, so the node is disabled before it ever ran.
        for live, stream in ((0, None), (4, None), (0, other_stream), (8, other_stream), (1, None)):
            values.zero_()
            source.fill_(live)
            wp.synchronize_device(device)
            wp.capture_launch(graph, stream=stream)
            wp.synchronize_device(device)
            if exec_handle is None:
                exec_handle = graph.graph_exec.value
            test.assertEqual(graph.graph_exec.value, exec_handle)
            expected = np.zeros(8, dtype=np.int32)
            expected[:live] = np.arange(1, live + 1) + live * 100
            np.testing.assert_array_equal(values.numpy(), expected)
            np.testing.assert_array_equal(wp.capture_get_count_update_errors(graph).numpy(), [0])


def test_count_update_matches_eager(test, device):
    _count_update_supported(test)
    with wp.ScopedDevice(device):
        count = wp.CountParameter(8)
        source = wp.zeros(1, dtype=wp.int32)
        inputs = wp.array(np.arange(8, dtype=np.float32) + 1.0)
        replayed = wp.full(8, -1.0, dtype=float)
        eager = wp.full(8, -1.0, dtype=float)
        wp.load_module(device=device)
        with wp.ScopedCapture(record_launches=True) as capture:
            wp.capture_update_counts([count], [source])
            wp.launch(write_scaled, count, [inputs, replayed, count])
        for live in (3, 0, 8, 5):
            replayed.fill_(-1.0)
            eager.fill_(-1.0)
            source.fill_(live)
            wp.capture_launch(capture.graph)
            wp.launch(write_scaled, live, [inputs, eager, live])
            np.testing.assert_array_equal(replayed.numpy(), eager.numpy())


def test_count_update_scalar_only_runs_at_zero(test, device):
    _count_update_supported(test)
    with wp.ScopedDevice(device):
        count = wp.CountParameter(8)
        source = wp.zeros(1, dtype=wp.int32)
        values = wp.full(4, -1, dtype=int)
        wp.load_module(device=device)
        with wp.ScopedCapture(record_launches=True) as capture:
            wp.capture_update_counts([count], [source])
            wp.launch(write_index, 4, [values, count])
        for live in (0, 5, 8):
            source.fill_(live)
            wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(values.numpy(), np.arange(1, 5) + live * 100)
        # The same count in two scalar arguments after an array argument binds both offsets.
        pair = wp.zeros(4, dtype=int)
        with wp.ScopedCapture(record_launches=True) as pair_capture:
            wp.capture_update_counts([count], [source])
            wp.launch(write_pair, 4, [pair, count, count])
        record = wp.capture_get_count_bound_launches(pair_capture.graph)[0]
        test.assertEqual([index for index, _ in record.scalar_parameters], [2, 3])
        for live in (3, 0, 8):
            source.fill_(live)
            wp.capture_launch(pair_capture.graph)
            np.testing.assert_array_equal(pair.numpy(), live * 11)


def test_count_update_two_axes_and_lean_grid(test, device):
    _count_update_supported(test)
    with wp.ScopedDevice(device):
        rows, cols = wp.CountParameter(4), wp.CountParameter(6)
        row_source, col_source = wp.zeros(1, dtype=wp.int32), wp.zeros(1, dtype=wp.int32)
        grid = wp.zeros((4, 6), dtype=int)
        # A lean kernel large enough that its grid spills past 2^24 threads into gridDim.y.
        lean_capacity = (1 << 24) + 300000
        lean_count = wp.CountParameter(lean_capacity)
        lean_source = wp.zeros(1, dtype=wp.int32)
        lean = wp.zeros(lean_capacity, dtype=int)
        wp.load_module(device=device)
        wp.load_module(module=write_lean.module, device=device)
        with wp.ScopedCapture(record_launches=True) as capture:
            wp.capture_update_counts([rows, cols, lean_count], [row_source, col_source, lean_source])
            wp.launch(write_grid, (rows, cols), [grid, cols])
            wp.launch(write_lean, lean_count, [lean, 5])
        test.assertEqual(len(wp.capture_get_count_bound_launches(capture.graph)), 2)
        for live_rows, live_cols, live_lean in (
            (2, 3, 1000),
            (0, 4, 1 << 24),
            (4, 6, (1 << 24) + 1),
            (1, 1, lean_capacity),
            (3, 0, 65536 * 256 + 1),
        ):
            grid.zero_()
            lean.zero_()
            row_source.fill_(live_rows)
            col_source.fill_(live_cols)
            lean_source.fill_(live_lean)
            wp.capture_launch(capture.graph)
            expected = np.zeros((4, 6), dtype=np.int32)
            expected[:live_rows, :live_cols] = live_cols
            np.testing.assert_array_equal(grid.numpy(), expected)
            written = lean.numpy()
            test.assertEqual(int(np.count_nonzero(written)), live_lean)
            test.assertTrue(np.all(written[:live_lean] == 6))
            np.testing.assert_array_equal(wp.capture_get_count_update_errors(capture.graph).numpy(), [0, 0])


def test_count_update_out_of_range_disables_launch(test, device):
    _count_update_supported(test)
    with wp.ScopedDevice(device):
        count = wp.CountParameter(8)
        source = wp.zeros(1, dtype=wp.int32)
        values = wp.zeros(8, dtype=int)
        wp.load_module(device=device)
        with wp.ScopedCapture(record_launches=True) as capture:
            wp.capture_update_counts([count], [source])
            wp.launch(write_index, count, [values, count])
        exec_handle = None
        for live, expected_error in ((9, -1), (-1, -1), (2, 0), (2**31 - 1, -1), (8, 0)):
            values.zero_()
            source.fill_(live)
            wp.capture_launch(capture.graph)
            if exec_handle is None:
                exec_handle = capture.graph.graph_exec.value
            test.assertEqual(capture.graph.graph_exec.value, exec_handle)
            np.testing.assert_array_equal(wp.capture_get_count_update_errors(capture.graph).numpy(), [expected_error])
            expected = np.zeros(8, dtype=np.int32)
            if expected_error == 0:
                expected[:live] = np.arange(1, live + 1) + live * 100
            np.testing.assert_array_equal(values.numpy(), expected)
        # A scalar-only binding is disabled by an out-of-range value too, and recovers.
        scalar_values = wp.full(4, -1, dtype=int)
        with wp.ScopedCapture(record_launches=True) as scalar_capture:
            wp.capture_update_counts([count], [source])
            wp.launch(write_index, 4, [scalar_values, count])
        for live, expected_error in ((9, -1), (3, 0)):
            source.fill_(live)
            wp.capture_launch(scalar_capture.graph)
            np.testing.assert_array_equal(
                wp.capture_get_count_update_errors(scalar_capture.graph).numpy(), [expected_error]
            )
            expected = np.full(4, -1, dtype=np.int32) if expected_error else np.arange(1, 5) + live * 100
            np.testing.assert_array_equal(scalar_values.numpy(), expected)


def test_count_update_leaves_unnamed_counts_pending(test, device):
    _count_update_supported(test)
    with wp.ScopedDevice(device):
        bound, pending = wp.CountParameter(8), wp.CountParameter(4)
        source = wp.zeros(1, dtype=wp.int32)
        first, second = wp.zeros(8, dtype=int), wp.zeros(4, dtype=int)
        wp.load_module(device=device)
        with wp.ScopedCapture(record_launches=True) as capture:
            wp.capture_update_counts([bound], [source])
            wp.launch(write_index, bound, [first, bound])
            wp.launch(write_index, pending, [second, 7])
        graph = capture.graph
        test.assertEqual(len(wp.capture_get_count_bound_launches(graph)), 1)
        with test.assertRaises(ValueError):
            wp.capture_complete_preparation(graph, bound)  # Warp discharged the bound count
        with test.assertRaisesRegex(RuntimeError, "preparation"):
            wp.capture_launch(graph)
        # An external binder keeps the pending count at its maximum here.
        wp.capture_complete_preparation(graph, pending)
        source.fill_(3)
        wp.capture_launch(graph)
        np.testing.assert_array_equal(first.numpy(), [301, 302, 303, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(second.numpy(), np.arange(1, 5) + 700)

        # A launch that mixes a bound count with one the update point does not name fails closed.
        mixed = wp.zeros(8, dtype=int)
        with test.assertRaisesRegex(RuntimeError, "does not name"):
            with wp.ScopedCapture(record_launches=True) as capture:
                active = device.captures[wp.get_stream(device)]
                wp.capture_update_counts([bound], [source])
                wp.launch(write_index, bound, [mixed, pending])
        with test.assertRaisesRegex(RuntimeError, "invalid"):
            wp.capture_launch(active)


def test_count_update_requires_recording_and_order(test, device):
    _count_update_supported(test)
    with wp.ScopedDevice(device):
        count = wp.CountParameter(8)
        source = wp.zeros(1, dtype=wp.int32)
        values = wp.zeros(8, dtype=int)
        wp.load_module(device=device)
        with test.assertRaisesRegex(RuntimeError, "record_launches"):
            wp.capture_update_counts([count], [source])
        with wp.ScopedCapture() as capture:
            with test.assertRaisesRegex(RuntimeError, "record_launches"):
                wp.capture_update_counts([count], [source])
        wp.capture_launch(capture.graph)
        # An older toolkit or driver is refused before anything is recorded; the capture stays usable.
        with wp.ScopedCapture(record_launches=True) as capture:
            with mock.patch.object(context, "is_conditional_graph_supported", return_value=False):
                with test.assertRaisesRegex(RuntimeError, "12.4"):
                    wp.capture_update_counts([count], [source])
            wp.launch(write_value, 1, [values, 5])
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(values.numpy()[0], 5)
        for bad_counts, bad_sources in (
            ([count], [wp.zeros(2, dtype=wp.int32)]),
            ([count], [wp.zeros(1, dtype=wp.int64)]),
            ([count, count], [source, source]),
            ([8], [source]),
            ([], []),
        ):
            with wp.ScopedCapture(record_launches=True) as capture:
                with test.assertRaises(ValueError):
                    wp.capture_update_counts(bad_counts, bad_sources)
            wp.capture_launch(capture.graph)
        # One update point per capture.
        with wp.ScopedCapture(record_launches=True) as capture:
            active = device.captures[wp.get_stream(device)]
            wp.capture_update_counts([count], [source])
            with test.assertRaisesRegex(RuntimeError, "one count update point"):
                wp.capture_update_counts([count], [source])
        with test.assertRaisesRegex(RuntimeError, "invalid"):
            wp.capture_launch(active)
        # A count launch recorded before the update point cannot be bound by it.
        with test.assertRaisesRegex(RuntimeError, "precedes"):
            with wp.ScopedCapture(record_launches=True) as capture:
                active = device.captures[wp.get_stream(device)]
                wp.launch(write_index, count, [values, count])
                wp.capture_update_counts([count], [source])
        with test.assertRaisesRegex(RuntimeError, "invalid"):
            wp.capture_launch(active)
        # Another capture open on the device at capture_end is refused, and the other capture survives.
        side = wp.Stream(device)
        wp.capture_begin(stream=side)
        try:
            with test.assertRaisesRegex(RuntimeError, "no other capture"):
                with wp.ScopedCapture(record_launches=True) as capture:
                    active = device.captures[wp.get_stream(device)]
                    wp.capture_update_counts([count], [source])
                    wp.launch(write_index, count, [values, count])
            with test.assertRaisesRegex(RuntimeError, "invalid"):
                wp.capture_launch(active)
            wp.launch(write_value, 1, [values, 9], stream=side)
        finally:
            side_graph = wp.capture_end(stream=side)
        wp.capture_launch(side_graph)
        wp.synchronize_device(device)
        test.assertEqual(values.numpy()[0], 9)
        # The stream is free again: a fresh recorded capture on it binds and replays.
        with wp.ScopedCapture(record_launches=True) as capture:
            wp.capture_update_counts([count], [source])
            wp.launch(write_index, count, [values, count])
        bound_graph = capture.graph
        source.fill_(2)
        wp.capture_launch(bound_graph)
        np.testing.assert_array_equal(values.numpy()[:2], [201, 202])
        # A count-bound graph cannot be inserted into another capture: child insertion clones nodes.
        condition = wp.ones(1, dtype=int)
        for insert in (
            lambda: wp.capture_launch(bound_graph),
            lambda: wp.capture_if(condition, bound_graph),
        ):
            with wp.ScopedCapture() as outer:
                with test.assertRaisesRegex(RuntimeError, "cannot be inserted"):
                    insert()
            with test.assertRaises(RuntimeError):
                wp.capture_launch(outer.graph)
        wp.capture_launch(bound_graph)


def test_count_update_joined_stream(test, device):
    _count_update_supported(test)
    with wp.ScopedDevice(device):
        count = wp.CountParameter(8)
        source = wp.zeros(1, dtype=wp.int32)
        values = wp.zeros(8, dtype=int)
        main_stream = wp.get_stream(device)
        side = wp.Stream(device)
        wp.load_module(device=device)
        with wp.ScopedCapture(record_launches=True) as capture:
            wp.capture_update_counts([count], [source])
            side.wait_stream(main_stream)
            wp.launch(write_index, count, [values, count], stream=side)
            main_stream.wait_stream(side)
        test.assertEqual(len(wp.capture_get_count_bound_launches(capture.graph)), 1)
        for live in (5, 0, 8):
            values.zero_()
            source.fill_(live)
            wp.capture_launch(capture.graph)
            expected = np.zeros(8, dtype=np.int32)
            expected[:live] = np.arange(1, live + 1) + live * 100
            np.testing.assert_array_equal(values.numpy(), expected)


def test_count_update_inside_conditional_bodies(test, device):
    _count_update_supported(test)
    with wp.ScopedDevice(device):
        count = wp.CountParameter(8)
        source = wp.zeros(1, dtype=wp.int32)
        outer, inner = wp.ones(1, dtype=int), wp.ones(1, dtype=int)
        values, nested = wp.zeros(8, dtype=int), wp.zeros(8, dtype=int)
        wp.load_module(device=device)
        with wp.ScopedCapture(record_launches=True) as capture:
            wp.capture_update_counts([count], [source])
            wp.capture_if(
                outer,
                lambda: (
                    wp.launch(write_index, count, [values, count]),
                    wp.capture_if(inner, lambda: wp.launch(write_index, count, [nested, count])),
                ),
            )
        graph = capture.graph
        test.assertEqual(len(wp.capture_get_count_bound_launches(graph)), 2)
        test.assertTrue(all(r.graph_handle != graph.graph.value for r in wp.capture_get_launches(graph)))
        for live, outer_on, inner_on in ((4, 1, 1), (0, 1, 1), (8, 1, 0), (3, 0, 1), (6, 1, 1)):
            values.zero_()
            nested.zero_()
            source.fill_(live)
            outer.fill_(outer_on)
            inner.fill_(inner_on)
            wp.capture_launch(graph)
            expected = np.zeros(8, dtype=np.int32)
            expected[:live] = np.arange(1, live + 1) + live * 100
            np.testing.assert_array_equal(values.numpy(), expected if outer_on else np.zeros(8, dtype=np.int32))
            np.testing.assert_array_equal(
                nested.numpy(), expected if outer_on and inner_on else np.zeros(8, dtype=np.int32)
            )
            np.testing.assert_array_equal(wp.capture_get_count_update_errors(graph).numpy(), [0, 0])
        # A while body: the updater before the loop serves the launch inside it.
        condition = wp.ones(1, dtype=int)
        stopped = wp.zeros(1, dtype=int)
        looped = wp.zeros(8, dtype=int)
        with wp.ScopedCapture(record_launches=True) as while_capture:
            wp.capture_update_counts([count], [source])
            wp.capture_while(
                condition,
                lambda: (
                    wp.launch(write_index, count, [looped, count]),
                    wp.launch(stop_loop, 1, [condition, stopped]),
                ),
            )
        for live in (3, 0, 8):
            looped.zero_()
            condition.fill_(1)
            source.fill_(live)
            wp.capture_launch(while_capture.graph)
            expected = np.zeros(8, dtype=np.int32)
            expected[:live] = np.arange(1, live + 1) + live * 100
            np.testing.assert_array_equal(looped.numpy(), expected)
            test.assertEqual(stopped.numpy()[0], 17)
        # An update point inside a body is refused: it would not run on replays that skip the body,
        # whether it serves a launch outside the body or inside it.
        for body in (
            lambda: wp.capture_update_counts([count], [source]),
            lambda: (wp.capture_update_counts([count], [source]), wp.launch(write_index, count, [values, count])),
        ):
            with test.assertRaisesRegex(RuntimeError, "root graph"):
                with wp.ScopedCapture(record_launches=True) as capture:
                    active = device.captures[wp.get_stream(device)]
                    wp.capture_if(outer, body)
                    wp.launch(write_index, count, [values, count])
            with test.assertRaisesRegex(RuntimeError, "invalid"):
                wp.capture_launch(active)


def test_count_update_newton_shaped_contacts(test, device):
    # One graph per solver step: a device kernel publishes the live contact count of the
    # current reset, the update point reads it, the narrowphase runs over exactly that many
    # contacts, and a fixed launch stays fixed. Resets change the count with no re-capture.
    _count_update_supported(test)
    with wp.ScopedDevice(device):
        capacity = 16
        n_contacts = wp.CountParameter(capacity)
        population_counts = wp.zeros(3, dtype=wp.int32)
        contact_count = population_counts[1:2]  # a view into the population's live counts, as Newton binds them
        reset_counts = wp.array([0, 5, 2, 16, 7, 20], dtype=int)  # the last reset exceeds the capacity
        reset_index = wp.zeros(1, dtype=int)
        step_condition = wp.ones(1, dtype=int)
        errors = wp.zeros(2, dtype=wp.int32)
        healthy = wp.zeros(1, dtype=int)
        contacts = wp.full(capacity, -1, dtype=int)
        touched = wp.zeros(capacity, dtype=int)
        fixed = wp.zeros(4, dtype=int)
        eager_contacts = wp.full(capacity, -1, dtype=int)
        eager_touched = wp.zeros(capacity, dtype=int)
        wp.load_module(device=device)
        with wp.ScopedCapture(record_launches=True) as capture:
            wp.launch(read_reset, 1, [reset_counts, reset_index, contact_count])
            wp.capture_update_counts([n_contacts], [contact_count], errors=errors)
            wp.capture_if(step_condition, lambda: wp.launch(narrowphase, n_contacts, [n_contacts, contacts, touched]))
            wp.launch(write_index, 4, [fixed, 2])
            wp.launch(latch_errors, 2, [errors, healthy])
        graph = capture.graph
        test.assertEqual(len(wp.capture_get_count_bound_launches(graph)), 1)
        with test.assertRaises(ValueError):
            wp.capture_complete_preparation(graph, n_contacts)
        exec_handle = None
        for reset, live in enumerate(reset_counts.numpy().tolist()):
            reset_index.fill_(reset)
            healthy.zero_()
            wp.capture_launch(graph)
            if live <= capacity:
                wp.launch(narrowphase, live, [live, eager_contacts, eager_touched])
            if exec_handle is None:
                exec_handle = graph.graph_exec.value
            test.assertEqual(graph.graph_exec.value, exec_handle)
            test.assertEqual(contact_count.numpy()[0], live)
            np.testing.assert_array_equal(contacts.numpy(), eager_contacts.numpy())
            np.testing.assert_array_equal(touched.numpy(), eager_touched.numpy())
            np.testing.assert_array_equal(fixed.numpy(), np.arange(1, 5) + 200)
            np.testing.assert_array_equal(errors.numpy(), [0 if live <= capacity else -1, 0])
            test.assertEqual(healthy.numpy()[0], 0 if live <= capacity else 1)
        test.assertEqual(int(touched.numpy().sum()), 30)


def test_count_update_shared_sources_and_repeated_counts(test, device):
    _count_update_supported(test)
    with wp.ScopedDevice(device):
        first, second, unused = wp.CountParameter(8), wp.CountParameter(8), wp.CountParameter(8)
        shared = wp.zeros(1, dtype=wp.int32)
        other = wp.zeros(1, dtype=wp.int32)
        shared_ref = weakref.ref(shared)
        # A non-owning view writes the live value after the owning name is gone.
        shared_view = wp.array(ptr=shared.ptr, dtype=wp.int32, shape=(1,), device=device)
        a, b, c = wp.zeros(8, dtype=int), wp.zeros(8, dtype=int), wp.zeros(8, dtype=int)
        wp.load_module(device=device)
        with wp.ScopedCapture(record_launches=True) as capture:
            # one count in two launches, one source serving two counts, equal maxima kept apart,
            # and a named count that no launch uses
            wp.capture_update_counts([first, second, unused], [shared, other, other])
            wp.launch(write_index, first, [a, 1])
            wp.launch(write_index, first, [b, 2])
            wp.launch(write_index, second, [c, 3])
        graph = capture.graph
        test.assertEqual(len(wp.capture_get_count_bound_launches(graph)), 3)
        test.assertEqual(wp.capture_get_count_update_errors(graph).shape, (3,))
        with wp.ScopedCapture(record_launches=True) as empty_capture:
            wp.capture_update_counts([unused], [other])
            wp.launch(write_value, 1, [a, 1])
        test.assertEqual(wp.capture_get_count_bound_launches(empty_capture.graph), ())
        test.assertEqual(wp.capture_get_count_update_errors(empty_capture.graph).shape, (0,))
        wp.capture_launch(empty_capture.graph)
        # Two graphs may bind one source; the second outlives the first.
        d = wp.zeros(8, dtype=int)
        with wp.ScopedCapture(record_launches=True) as second_capture:
            wp.capture_update_counts([second], [other])
            wp.launch(write_index, second, [d, 4])
        del shared
        gc.collect()
        test.assertIsNotNone(shared_ref())  # the Graph retains the source
        for live_first, live_second in ((3, 5), (0, 8), (8, 0)):
            a.zero_()
            b.zero_()
            c.zero_()
            shared_view.fill_(live_first)
            other.fill_(live_second)
            wp.capture_launch(graph)
            np.testing.assert_array_equal(a.numpy()[:live_first], np.arange(1, live_first + 1) + 100)
            np.testing.assert_array_equal(a.numpy()[live_first:], 0)
            np.testing.assert_array_equal(b.numpy()[:live_first], np.arange(1, live_first + 1) + 200)
            np.testing.assert_array_equal(b.numpy()[live_first:], 0)
            np.testing.assert_array_equal(c.numpy()[:live_second], np.arange(1, live_second + 1) + 300)
            np.testing.assert_array_equal(c.numpy()[live_second:], 0)
        del graph, capture
        gc.collect()
        test.assertIsNone(shared_ref())  # and releases it with the Graph
        other.fill_(6)
        wp.capture_launch(second_capture.graph)
        np.testing.assert_array_equal(d.numpy(), [401, 402, 403, 404, 405, 406, 0, 0])


def test_count_update_in_graph_guard(test, device):
    # Newton's shape: a kernel recorded after the bound launches reads the error words and
    # latches the batch unhealthy, so the words must exist during the capture.
    _count_update_supported(test)
    with wp.ScopedDevice(device):
        count = wp.CountParameter(8)
        source = wp.zeros(1, dtype=wp.int32)
        errors = wp.full(4, 7, dtype=wp.int32)
        latch = wp.zeros(1, dtype=int)
        values = wp.zeros(8, dtype=int)
        wp.load_module(device=device)
        with wp.ScopedCapture(record_launches=True) as capture:
            wp.capture_update_counts([count], [source], errors=errors)
            wp.launch(write_index, count, [values, count])
            wp.launch(latch_errors, 4, [errors, latch])
        graph = capture.graph
        test.assertIs(wp.capture_get_count_update_errors(graph), errors)
        np.testing.assert_array_equal(errors.numpy(), [0, 7, 7, 7])  # only the bound prefix is cleared
        errors[1:].zero_()
        for live, expected_latch in ((3, 0), (9, 1), (0, 0), (8, 0)):
            latch.zero_()
            source.fill_(live)
            wp.capture_launch(graph)
            test.assertEqual(latch.numpy()[0], expected_latch)
            np.testing.assert_array_equal(errors.numpy(), [-1 if expected_latch else 0, 0, 0, 0])
        # Too small an errors array fails closed at capture_end.
        with test.assertRaisesRegex(RuntimeError, "entries"):
            with wp.ScopedCapture(record_launches=True) as capture:
                active = device.captures[wp.get_stream(device)]
                wp.capture_update_counts([count], [source], errors=wp.zeros(1, dtype=wp.int32))
                wp.launch(write_index, count, [values, count])
                wp.launch(write_index, count, [values, count])
        with test.assertRaisesRegex(RuntimeError, "invalid"):
            wp.capture_launch(active)


def test_count_update_tiled_rank3_and_grid_caps(test, device):
    _count_update_supported(test)
    with wp.ScopedDevice(device):
        rows, depth, big = wp.CountParameter(5), wp.CountParameter(4), wp.CountParameter(4096)
        row_source, depth_source, big_source = (wp.zeros(1, dtype=wp.int32) for _ in range(3))
        tile_source = wp.array(np.arange(20, dtype=np.float32))
        tile_target = wp.full(20, -1.0, dtype=float)
        volume = wp.zeros((4, 2, 3), dtype=int)
        capped = wp.zeros(4096, dtype=int)
        eager = wp.zeros(4096, dtype=int)
        wp.load_module(device=device)
        wp.launch_tiled(copy_tile, dim=1, inputs=[tile_source, tile_target], block_dim=32)
        with wp.ScopedCapture(record_launches=True) as capture:
            wp.capture_update_counts([rows, depth, big], [row_source, depth_source, big_source])
            wp.launch_tiled(copy_tile, dim=rows, inputs=[tile_source, tile_target], block_dim=32)
            wp.launch(write_volume, (depth, 2, 3), [volume, depth])
            wp.launch(write_index, big, [capped, 0], max_blocks=4)
        records = wp.capture_get_count_bound_launches(capture.graph)
        test.assertEqual(len(records), 3)
        bounds = type(launch_bounds_t((0,))).from_buffer_copy(records[0].parameter_bytes[0])
        test.assertEqual(bounds.coord_mult, 32)
        test.assertEqual(records[2].max_blocks, 4)
        for live_rows, live_depth, live_big in ((3, 2, 1000), (0, 4, 4096), (5, 0, 1), (1, 1, 0)):
            tile_target.fill_(-1.0)
            volume.zero_()
            capped.zero_()
            eager.zero_()
            row_source.fill_(live_rows)
            depth_source.fill_(live_depth)
            big_source.fill_(live_big)
            wp.capture_launch(capture.graph)
            wp.launch(write_index, live_big, [eager, 0], max_blocks=4)
            np.testing.assert_array_equal(tile_target.numpy()[: live_rows * 4], tile_source.numpy()[: live_rows * 4])
            np.testing.assert_array_equal(tile_target.numpy()[live_rows * 4 :], -1.0)
            expected = np.zeros((4, 2, 3), dtype=np.int32)
            expected[:live_depth] = live_depth
            np.testing.assert_array_equal(volume.numpy(), expected)
            np.testing.assert_array_equal(capped.numpy(), eager.numpy())
            np.testing.assert_array_equal(wp.capture_get_count_update_errors(capture.graph).numpy(), [0, 0, 0])


def test_count_update_rejects_launch_forked_before_update_point(test, device):
    _count_update_supported(test)
    with wp.ScopedDevice(device):
        count = wp.CountParameter(8)
        source = wp.zeros(1, dtype=wp.int32)
        values = wp.zeros(8, dtype=int)
        main_stream = wp.get_stream(device)
        side = wp.Stream(device)
        wp.load_module(device=device)
        with test.assertRaisesRegex(RuntimeError, "does not depend"):
            with wp.ScopedCapture(record_launches=True) as capture:
                active = device.captures[main_stream]
                wp.launch(write_value, 1, [values, 1])
                side.wait_stream(main_stream)
                wp.capture_update_counts([count], [source])
                wp.launch(write_index, count, [values, count], stream=side)
                main_stream.wait_stream(side)
        with test.assertRaisesRegex(RuntimeError, "invalid"):
            wp.capture_launch(active)


class TestCountParameters(unittest.TestCase):
    def test_identity_and_host_protocol(self):
        value, other = wp.CountParameter(8), wp.CountParameter(8)
        self.assertIsNot(value, other)
        self.assertEqual(tuple(field.name for field in dataclasses.fields(value)), ("maximum",))
        self.assertEqual(wp.upper_bound(value), 8)
        self.assertEqual(wp.upper_bound(0), 0)
        self.assertEqual(wp.upper_bound(wp.CountParameter(2**40)), 2**40)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            value.maximum = 1
        for operation in (bool, int, float, complex, operator.index, abs, round, operator.neg):
            with self.subTest(operation=operation), self.assertRaises(TypeError):
                operation(value)
        for operation in (
            operator.eq,
            operator.ne,
            operator.lt,
            operator.add,
            operator.mul,
            operator.floordiv,
            operator.and_,
        ):
            with self.subTest(operation=operation), self.assertRaises(TypeError):
                operation(value, 1)
            with self.subTest(reverse=operation), self.assertRaises(TypeError):
                operation(1, value)
        for invalid in (-1, True, 1.0, np.int32(1)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                wp.CountParameter(invalid)
        with self.assertRaises(TypeError):
            wp.empty((value,), dtype=float, device="cpu")

    def test_eager_prefix_preserves_full_array_semantics(self):
        with wp.ScopedDevice("cpu"):
            source = wp.array(np.arange(24, dtype=np.int32).reshape(8, 3), dtype=int)
            target = wp.full((10, 3), -1, dtype=int)
            wp.copy(target, source, extent=(2, 3))
            target.fill_(7, extent=(1, 3))
            target.zero_(extent=(0, 3))
            np.testing.assert_array_equal(target.numpy()[0], 7)
            np.testing.assert_array_equal(target.numpy()[1], source.numpy()[1])
            np.testing.assert_array_equal(target.numpy()[2:], -1)
            target.fill_(4)
            np.testing.assert_array_equal(target.numpy(), 4)
            for extent in ((11, 3), (2, 2), (2,), (True, 3)):
                with self.subTest(extent=extent), self.assertRaises(ValueError):
                    target.fill_(0, extent=extent)
            with self.assertRaises(ValueError):
                wp.copy(target, source, count=1, extent=(2, 3))
            with self.assertRaisesRegex(RuntimeError, "recorded"):
                target.fill_(1, extent=(wp.CountParameter(8), 3))
            for shape in ((0,), (4, 0), (0, 3), (2, 0, 3)):
                with self.subTest(empty_shape=shape):
                    empty = wp.empty(shape, dtype=int)
                    empty.fill_(7, extent=shape)
                    empty.zero_(extent=shape)
                    wp.copy(empty, empty, extent=shape)
                    self.assertEqual(empty.size, 0)
            wp.copy(target, source, extent=(0, 3))
            np.testing.assert_array_equal(target.numpy(), 4)


def test_count_update_split_spelling(test, device):
    # The other spelling of the same binder: the sources are declared on the capture and the update
    # point names only the counts (or nothing, for all of them). Same replays, same executable.
    _count_update_supported(test)
    with wp.ScopedDevice(device):
        count = wp.CountParameter(8)
        other = wp.CountParameter(8)
        source = wp.zeros(1, dtype=wp.int32)
        conflicting = wp.zeros(1, dtype=wp.int32)
        values = wp.zeros(8, dtype=int)
        wp.load_module(device=device)

        def check(graph, live):
            values.zero_()
            source.fill_(live)
            wp.capture_launch(graph)
            expected = np.zeros(8, dtype=np.int32)
            expected[:live] = np.arange(1, live + 1) + live * 100
            np.testing.assert_array_equal(values.numpy(), expected)
            np.testing.assert_array_equal(wp.capture_get_count_update_errors(graph).numpy(), [0])

        with wp.ScopedCapture(record_launches=True, count_sources=((count, source),)) as capture:
            wp.capture_update_counts((count,))
            wp.launch(write_index, count, [values, count])
        graph = capture.graph
        test.assertEqual(len(wp.capture_get_count_bound_launches(graph)), 1)
        exec_handle = None
        for live in (4, 0, 8, 1):
            check(graph, live)
            if exec_handle is None:
                exec_handle = graph.graph_exec.value
            test.assertEqual(graph.graph_exec.value, exec_handle)
        # No counts named: every declared count is bound.
        with wp.ScopedCapture(record_launches=True, count_sources=[(count, source)]) as capture:
            wp.capture_update_counts()
            wp.launch(write_index, count, [values, count])
        check(capture.graph, 3)
        # The one-call spelling still works inside a capture with declarations, as long as it agrees.
        with wp.ScopedCapture(record_launches=True, count_sources=((count, source),)) as capture:
            with test.assertRaisesRegex(ValueError, "no source"):
                wp.capture_update_counts((other,))
            with test.assertRaisesRegex(ValueError, "Conflicting source"):
                wp.capture_update_counts((count,), (conflicting,))
            wp.capture_update_counts((count,), (source,))
            wp.launch(write_index, count, [values, count])
        check(capture.graph, 2)
        # Declarations are validated before the capture opens; an update point with nothing to bind is refused.
        with test.assertRaisesRegex(ValueError, "record_launches"):
            wp.capture_begin(count_sources=((count, source),))
        with test.assertRaisesRegex(ValueError, "CountParameter"):
            wp.capture_begin(record_launches=True, count_sources=((5, source),))
        with test.assertRaisesRegex(ValueError, "Count sources"):
            wp.capture_begin(record_launches=True, count_sources=((count, wp.zeros(2, dtype=wp.int32)),))
        with test.assertRaisesRegex(ValueError, "once"):
            wp.capture_begin(record_launches=True, count_sources=((count, source), (count, source)))
        with test.assertRaisesRegex(ValueError, "pairs"):
            wp.capture_begin(record_launches=True, count_sources=(count,))
        with test.assertRaisesRegex(ValueError, "external"):
            wp.capture_begin(record_launches=True, external=True, count_sources=((count, source),))
        with wp.ScopedCapture(record_launches=True) as capture:
            with test.assertRaisesRegex(ValueError, "count_sources"):
                wp.capture_update_counts()
            wp.launch(write_index, 4, [values, 4])
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(values.numpy()[:4], np.arange(1, 5) + 400)


class TestCaptureLaunches(unittest.TestCase):
    def test_argument_owner_release_without_cyclic_gc(self):
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            for kernel in (write_value, write_member, copy_tile):
                with self.subTest(kernel=kernel.key):
                    values = wp.zeros(2, dtype=float if kernel is copy_tile else int, device="cpu")
                    owner = weakref.ref(values)
                    if kernel is write_member:
                        argument = NestedMember()
                        argument.member.values = values
                        arguments = [argument]
                        del argument
                    elif kernel is copy_tile:
                        arguments = [values, values]
                    else:
                        arguments = [values, 3]
                    params = [launch_bounds_t((2,))] + [
                        context.pack_arg(kernel, arg.type, arg.label, value, wp.get_device("cpu"))
                        for arg, value in zip(kernel.adj.args, arguments, strict=True)
                    ]
                    packets, owners = context._capture_launch_arguments(kernel, params, arguments, wp.get_device("cpu"))
                    self.assertEqual(owners, (values, values) if kernel is copy_tile else (values,))
                    del values, arguments, params
                    self.assertIsNotNone(owner())
                    del owners, packets
                    self.assertIsNone(owner())
        finally:
            if was_enabled:
                gc.enable()

    def test_launch_count_rejects_non_graph(self):
        with self.assertRaisesRegex(TypeError, "Warp Graph"):
            wp.capture_launch_count(object())

    def test_cpu_recording_rejected(self):
        with self.assertRaisesRegex(ValueError, "CUDA capture"):
            wp.capture_begin(device="cpu", record_launches=True)

    def test_memory_recording_requires_launches(self):
        with self.assertRaisesRegex(ValueError, "record_launches=True"):
            wp.capture_begin(device="cpu", record_memory_operations=True)


for test_func in (
    test_capture_packets,
    test_reusable_snapshots,
    test_struct_resource_lifetime,
    test_tiled_compiled_bounds,
    test_joined_streams,
    test_conditional_body_records,
    test_nested_conditional_containment,
    test_caught_launch_failure,
    test_post_emission_failure,
    test_invalidation_after_instantiation,
    test_preparation_obligations,
    test_child_insertion_guards,
    test_command_descriptor_mutation,
    test_capture_launch_count,
    test_default_capture_unchanged,
    test_overlapping_recording_fails_closed,
    test_deterministic_command_rejected,
    test_paused_capture_prefix,
    test_captured_memory_values,
    test_captured_memory_strided_rows,
    test_captured_memory_zero_and_noops,
    test_captured_memory_owners_and_snapshots,
    test_captured_memory_rejects_unsupported,
    test_captured_memory_streams_and_conditions,
    test_captured_memory_rejects_stale_array_abi,
    test_count_parameter_occurrences_and_obligations,
    test_count_parameter_tiled_and_prefix_memory,
    test_count_parameter_rejects_unrecorded_and_wrong_scalar,
    test_count_update_replay_zero_to_positive,
    test_count_update_matches_eager,
    test_count_update_scalar_only_runs_at_zero,
    test_count_update_two_axes_and_lean_grid,
    test_count_update_out_of_range_disables_launch,
    test_count_update_leaves_unnamed_counts_pending,
    test_count_update_requires_recording_and_order,
    test_count_update_joined_stream,
    test_count_update_inside_conditional_bodies,
    test_count_update_newton_shaped_contacts,
    test_count_update_shared_sources_and_repeated_counts,
    test_count_update_in_graph_guard,
    test_count_update_tiled_rank3_and_grid_caps,
    test_count_update_split_spelling,
    test_count_update_rejects_launch_forked_before_update_point,
):
    add_function_test(TestCaptureLaunches, test_func.__name__, test_func, devices=get_selected_cuda_test_devices())


if __name__ == "__main__":
    wp.init()
    unittest.main(verbosity=2)
