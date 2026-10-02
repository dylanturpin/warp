# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact launch metadata, retained resources, and fail-closed graph preparation."""

import ctypes
import dataclasses
import gc
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
        gc.collect()
        test.assertIsNotNone(old_owner())
        test.assertEqual(record.parameter_bytes, packet)
        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(old_owner().numpy(), 11)
        np.testing.assert_allclose(nested.member.values.numpy(), 0)


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


class TestCaptureLaunches(unittest.TestCase):
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
):
    add_function_test(TestCaptureLaunches, test_func.__name__, test_func, devices=get_selected_cuda_test_devices())


if __name__ == "__main__":
    wp.init()
    unittest.main(verbosity=2)
