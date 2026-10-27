# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A reserved address range whose backing grows and shrinks while captured graphs keep their pointers."""

import gc
import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices


@wp.kernel
def write_rows(rows: wp.array2d[int], n: int):
    i, j = wp.tid()
    rows[i, j] = n * 1000 + i


@wp.kernel
def spin(ticks: int, out: wp.array[int]):
    acc = int(0)
    for k in range(ticks):
        acc = (acc * 1103515245 + 12345) & 0x7FFFFFFF
    out[0] = acc


def _supported(test, device):
    if not wp.Reservation.is_supported(device):
        test.skipTest("Virtual memory management is not supported on this device")


def test_reservation_reserve_map_unmap(test, device):
    _supported(test, device)
    with wp.ScopedDevice(device):
        reservation = wp.Reservation(1, device=device)
        granule = reservation.granularity
        test.assertGreater(granule, 0)
        test.assertEqual(reservation.size_bytes, granule)
        test.assertEqual(reservation.mapped_bytes, 0)
        test.assertTrue(reservation.address)
        reservation.release()
        with test.assertRaises(RuntimeError):
            reservation.map_prefix(1)

        # A large range is cheap to reserve and maps in granules at the front.
        reservation = wp.Reservation(64 * granule, device=device)
        address = reservation.address
        test.assertEqual(reservation.map_prefix(1), granule)
        test.assertEqual(reservation.map_prefix(granule), granule)
        test.assertEqual(reservation.map_prefix(2 * granule + 1), 3 * granule)
        test.assertEqual(reservation.address, address)
        with test.assertRaises(ValueError):
            reservation.map_prefix(65 * granule)
        rows_per_granule = granule // (8 * 4)
        rows = reservation.view((3 * rows_per_granule, 8), wp.int32)
        test.assertEqual(rows.ptr, address)
        wp.launch(write_rows, rows.shape, [rows, 1])
        values = rows.numpy()
        np.testing.assert_array_equal(values[:, 0], 1000 + np.arange(3 * rows_per_granule))
        # Shrinking below a live view is refused; after the view is gone it works once the events are done.
        with test.assertRaises(ValueError):
            reservation.unmap_prefix(granule)
        del rows, values
        gc.collect()
        event = wp.Event(device)
        wp.record_event(event)
        wp.synchronize_event(event)
        test.assertTrue(reservation.unmap_prefix(granule, after=[event]))
        test.assertEqual(reservation.mapped_bytes, granule)
        test.assertTrue(reservation.unmap_prefix(5 * granule))
        test.assertEqual(reservation.mapped_bytes, granule)
        # Regrow into released addresses and write again.
        test.assertEqual(reservation.map_prefix(2 * granule), 2 * granule)
        again = reservation.view((rows_per_granule, 8), wp.int32)
        wp.launch(write_rows, again.shape, [again, 2])
        np.testing.assert_array_equal(again.numpy()[:, 0], 2000 + np.arange(rows_per_granule))
        with test.assertRaises(RuntimeError):
            reservation.release()
        del again
        gc.collect()
        reservation.release()
        reservation.release()


def test_reservation_unmap_waits_for_events(test, device):
    _supported(test, device)
    with wp.ScopedDevice(device):
        granule = wp.Reservation.granularity_bytes(device)
        reservation = wp.Reservation(2 * granule, device=device)
        reservation.map_prefix(2 * granule)
        out = wp.zeros(1, dtype=int)
        wp.load_module(device=device)
        wp.launch(spin, 1, [1 << 26, out])
        pending = wp.Event(device)
        wp.record_event(pending)
        # The reader may still be running: the shrink is declined without waiting.
        declined = not reservation.unmap_prefix(granule, after=[pending])
        wp.synchronize_event(pending)
        test.assertTrue(reservation.unmap_prefix(granule, after=[pending]))
        test.assertEqual(reservation.mapped_bytes, granule)
        test.assertTrue(declined or pending.is_complete)
        reservation.release()


def test_reservation_capture_forbidden_and_grow_keeps_graph(test, device):
    # The point of the layer: a graph captured over a reservation-backed array replays after
    # the backing grows, with the same pointer and the same executable. The count binder trims
    # the rows the kernel touches to the live count, which the owner keeps within the mapped prefix.
    if not wp.is_conditional_graph_supported():
        test.skipTest("Count updates require CUDA 12.4+")
    _supported(test, device)
    with wp.ScopedDevice(device):
        granule = wp.Reservation.granularity_bytes(device)
        reservation = wp.Reservation(8 * granule, device=device)
        rows_per_granule = granule // (8 * 4)
        capacity = 8 * rows_per_granule
        rows = reservation.view((capacity, 8), wp.int32)
        reservation.map_prefix(granule)
        n_rows = wp.CountParameter(capacity)
        live = wp.zeros(1, dtype=wp.int32)
        wp.load_module(device=device)
        with wp.ScopedCapture(record_launches=True) as capture:
            with test.assertRaisesRegex(RuntimeError, "capture"):
                reservation.map_prefix(2 * granule)
            wp.capture_update_counts([n_rows], [live])
            wp.launch(write_rows, (n_rows, 8), [rows, n_rows])
        graph = capture.graph
        exec_handle = None
        mapped_rows = rows_per_granule
        for step, (grow_to, live_rows) in enumerate(
            ((1, rows_per_granule), (3, 2 * rows_per_granule + 5), (3, 0), (8, capacity), (8, 7))
        ):
            reservation.map_prefix(grow_to * granule)
            mapped_rows = grow_to * rows_per_granule
            test.assertLessEqual(live_rows, mapped_rows)
            live.fill_(live_rows)
            wp.capture_launch(graph)
            if exec_handle is None:
                exec_handle = graph.graph_exec.value
            test.assertEqual(graph.graph_exec.value, exec_handle)
            test.assertEqual(rows.ptr, reservation.address)
            if live_rows:
                written = wp.array(ptr=rows.ptr, dtype=wp.int32, shape=(live_rows, 8), device=device).numpy()
                np.testing.assert_array_equal(written[:, 0], live_rows * 1000 + np.arange(live_rows))
        test.assertEqual(reservation.mapped_bytes, 8 * granule)
        # The graph retains the view and the view retains the reservation: release is refused
        # until both are gone.
        del rows
        gc.collect()
        with test.assertRaisesRegex(RuntimeError, "live views"):
            reservation.release()
        del graph, capture
        gc.collect()
        reservation.release()


class TestReservation(unittest.TestCase):
    def test_cpu_rejected(self):
        with self.assertRaises(RuntimeError):
            wp.Reservation(1, device="cpu")
        self.assertFalse(wp.Reservation.is_supported("cpu"))


for test_func in (
    test_reservation_reserve_map_unmap,
    test_reservation_unmap_waits_for_events,
    test_reservation_capture_forbidden_and_grow_keeps_graph,
):
    add_function_test(TestReservation, test_func.__name__, test_func, devices=get_selected_cuda_test_devices())


if __name__ == "__main__":
    wp.init()
    unittest.main(verbosity=2)
