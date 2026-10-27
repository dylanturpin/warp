# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A reserved address range whose backing grows and shrinks while captured graphs keep their pointers.

Also: a shared budget refused cleanly and refunded on a driver failure, readiness published separately from
mapping, the two-step retirement with a reader still running over the retired rows, wp.empty(virtual=True),
wp.array(backing=) packed views and the guards on whole-array operations over a partially mapped array,
eagerly and under capture.
"""

import gc
import time
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
    for _k in range(ticks):
        acc = (acc * 1103515245 + 12345) & 0x7FFFFFFF
    out[0] = acc


@wp.kernel
def spin_then_read(ticks: int, rows: wp.array2d[int], row: int, out: wp.array[int]):
    # A reader that is still running when readiness is withdrawn: the row it reads at the end depends on
    # the loop result, so the load cannot be hoisted above the spin.
    acc = int(0)
    for _k in range(ticks):
        acc = (acc * 1103515245 + 12345) & 0x7FFFFFFF
    out[0] = rows[row - (acc & 1), 0]


@wp.kernel
def read_word(src: wp.array[int], out: wp.array[int]):
    out[0] = src[0]


@wp.kernel
def write_packed(q: wp.array2d[float], v: wp.array2d[float], n: int):
    i = wp.tid()
    if i < n:
        for j in range(3):
            q[i, j] = float(10 * i + j)
            v[i, j] = -float(10 * i + j)


def _supported(test, device):
    if not wp.Reservation.is_supported(device):
        test.skipTest("Virtual memory management is not supported on this device")


def _runtime():
    return wp._src.context.runtime


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
        with test.assertRaisesRegex(RuntimeError, "released"):
            reservation.map_prefix(1)

        # A large range is cheap to reserve and maps in granules at the front.
        reservation = wp.Reservation(64 * granule, device=device)
        address = reservation.address
        test.assertEqual(reservation.map_prefix(1), granule)
        test.assertEqual(reservation.map_prefix(granule), granule)
        test.assertEqual(reservation.map_prefix(2 * granule + 1), 3 * granule)
        test.assertEqual(reservation.address, address)
        with test.assertRaisesRegex(ValueError, "Cannot map"):
            reservation.map_prefix(65 * granule)
        rows_per_granule = granule // (8 * 4)
        rows = reservation.view((3 * rows_per_granule, 8), wp.int32)
        test.assertEqual(rows.ptr, address)
        test.assertIs(rows.backing, reservation)
        wp.launch(write_rows, rows.shape, [rows, 1])
        values = rows.numpy()
        np.testing.assert_array_equal(values[:, 0], 1000 + np.arange(3 * rows_per_granule))
        # A live view does not pin the backing (it covers the capacity; the owner bounds the rows it
        # touches). Published readiness does: shrinking below it is refused until it is withdrawn.
        ready = wp.zeros(1, dtype=wp.int32)
        reservation.publish_ready(ready, 3 * rows_per_granule, nbytes=3 * granule)
        test.assertEqual(ready.numpy()[0], 3 * rows_per_granule)
        test.assertEqual(reservation.ready_bytes, 3 * granule)
        with test.assertRaisesRegex(ValueError, "withdraw"):
            reservation.unmap_prefix(granule)
        with test.assertRaisesRegex(ValueError, "exceeds the mapped prefix"):
            reservation.publish_ready(ready, 4 * rows_per_granule, nbytes=4 * granule)
        with test.assertRaisesRegex(ValueError, "retire"):
            reservation.publish_ready(ready, 1, nbytes=granule)
        for bad in (
            wp.zeros(1, dtype=wp.int64),
            wp.zeros(2, dtype=wp.int32),
            wp.zeros(1, dtype=wp.int32, device="cpu"),
            wp.zeros(4, dtype=wp.int32)[::2][:1],
        ):
            with test.assertRaisesRegex(ValueError, "ready scalar"):
                reservation.publish_ready(bad, 1, nbytes=granule)
        retirement = reservation.retire(granule, ready=ready, count=rows_per_granule)
        test.assertEqual(reservation.ready_bytes, granule)
        test.assertEqual(reservation.mapped_bytes, 3 * granule)
        wp.synchronize_event(retirement.event)  # the write and the record have landed; nothing else was queued
        test.assertTrue(retirement.reclaim())
        test.assertFalse(retirement.pending)
        test.assertEqual(reservation.mapped_bytes, granule)
        test.assertEqual(ready.numpy()[0], rows_per_granule)
        # The view still covers three granules; reading it whole now raises instead of faulting,
        # and its first granule is intact.
        with test.assertRaisesRegex(RuntimeError, "partially mapped"):
            rows.numpy()
        np.testing.assert_array_equal(rows[:rows_per_granule].numpy()[:, 0], 1000 + np.arange(rows_per_granule))
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
        with test.assertRaisesRegex(RuntimeError, "live views"):
            reservation.release()
        del again
        gc.collect()
        reservation.release()
        reservation.release()
        # A reservation whose size is not a granule multiple rounds up, and the slack is usable.
        odd = wp.Reservation(granule + 1, device=device)
        test.assertEqual(odd.size_bytes, 2 * granule)
        tail = odd.view((4,), wp.int32, offset_bytes=granule)
        odd.map_prefix(granule + 1)
        tail.fill_(3)
        test.assertEqual(int(tail.numpy()[0]), 3)
        del tail
        gc.collect()
        odd.release()


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
        # The reader is still running: the shrink is declined without waiting and nothing changes.
        test.assertFalse(pending.is_complete)
        test.assertFalse(reservation.unmap_prefix(granule, after=[pending]))
        test.assertEqual(reservation.mapped_bytes, 2 * granule)
        wp.synchronize_event(pending)
        test.assertTrue(reservation.unmap_prefix(granule, after=[pending]))
        test.assertEqual(reservation.mapped_bytes, granule)
        # Unmapping during a capture is refused.
        with wp.ScopedCapture() as capture:
            with test.assertRaisesRegex(RuntimeError, "capture"):
                reservation.unmap_prefix(0)
        del capture
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
        for _step, (grow_to, live_rows) in enumerate(
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
                written = rows[:live_rows].numpy()  # a bounded read inside the mapped prefix
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


def test_reservation_budget_refused_cleanly(test, device):
    _supported(test, device)
    with wp.ScopedDevice(device):
        granule = wp.Reservation.granularity_bytes(device)
        budget = wp.BackingBudget(3 * granule)
        first = wp.Reservation(4 * granule, device=device, budget=budget)
        second = wp.Reservation(4 * granule, device=device, budget=budget)
        test.assertIs(first.budget, budget)
        first.map_prefix(2 * granule)
        second.map_prefix(granule)
        test.assertEqual(budget.used_bytes, 3 * granule)
        test.assertEqual(budget.available_bytes, 0)
        # The refusal is clean: nothing mapped, nothing charged, the reservation still usable.
        with test.assertRaisesRegex(MemoryError, "budget"):
            second.map_prefix(2 * granule)
        test.assertEqual(second.mapped_bytes, granule)
        test.assertEqual(budget.used_bytes, 3 * granule)
        rows = second.view((granule // 4,), wp.int32)
        rows.fill_(7)
        test.assertEqual(int(rows.numpy()[0]), 7)
        del rows
        gc.collect()
        # Unmapping returns bytes to the budget; then the refused mapping fits.
        test.assertTrue(first.unmap_prefix(granule))
        test.assertEqual(budget.used_bytes, 2 * granule)
        second.map_prefix(2 * granule)
        test.assertEqual(budget.used_bytes, 3 * granule)
        # A driver failure after the charge refunds it and changes nothing (the native is replaced for one call).
        core = _runtime().core
        real_map = core.wp_cuda_vmem_map
        core.wp_cuda_vmem_map = lambda *args: 0
        try:
            test.assertTrue(first.unmap_prefix(0))
            test.assertEqual(budget.used_bytes, 2 * granule)
            with test.assertRaisesRegex(MemoryError, "Could not map"):
                first.map_prefix(granule)
            test.assertEqual(first.mapped_bytes, 0)
            test.assertEqual(budget.used_bytes, 2 * granule)
        finally:
            core.wp_cuda_vmem_map = real_map
        first.map_prefix(granule)
        test.assertEqual(budget.used_bytes, 3 * granule)
        test.assertEqual(int(first.view((4,), wp.int32).numpy().shape[0]), 4)
        # Release returns everything; two budgets are independent.
        other = wp.BackingBudget(granule)
        third = wp.Reservation(granule, device=device, budget=other)
        third.map_prefix(granule)
        first.release()
        test.assertEqual(budget.used_bytes, 2 * granule)
        second.release()
        test.assertEqual(budget.used_bytes, 0)
        test.assertEqual(other.used_bytes, granule)
        third.release()
        test.assertEqual(other.used_bytes, 0)
        # A virtual array can be budgeted the same way; the first mapping beyond the limit is refused.
        small = wp.BackingBudget(granule)
        a = wp.empty((2 * granule // 4,), dtype=wp.int32, virtual=True, budget=small)
        a.backing.map_prefix(granule)
        with test.assertRaisesRegex(MemoryError, "budget"):
            a.backing.map_prefix(2 * granule)
        test.assertEqual(a.backing.mapped_bytes, granule)
        with test.assertRaisesRegex(TypeError, "acquire and release"):
            wp.Reservation(granule, device=device, budget=-1)
        with test.assertRaisesRegex(ValueError, "virtual=True"):
            wp.empty(4, dtype=int, budget=small)
        with test.assertRaisesRegex(ValueError, "non-negative"):
            wp.BackingBudget(-1)
        test.assertEqual(wp.BackingBudget(0).available_bytes, 0)


def test_reservation_two_step_retirement(test, device):
    _supported(test, device)
    with wp.ScopedDevice(device):
        granule = wp.Reservation.granularity_bytes(device)
        rows_per_granule = granule // (8 * 4)
        reservation = wp.Reservation(4 * granule, device=device)
        reservation.map_prefix(4 * granule)
        rows = reservation.view((4 * rows_per_granule, 8), wp.int32)
        wp.load_module(device=device)
        wp.launch(write_rows, rows.shape, [rows, 1])
        ready = wp.zeros(1, dtype=wp.int32)
        reservation.publish_ready(ready, 4 * rows_per_granule, nbytes=4 * granule)
        out = wp.zeros(1, dtype=int)
        reader = wp.Stream(device)
        # A reader on a side stream is still running over the rows about to be retired (the last row of the
        # fourth granule) when readiness is withdrawn. Launched with stream=, not ScopedStream, so nothing
        # orders it behind the current stream.
        last_row = 4 * rows_per_granule - 1
        wp.launch(spin_then_read, 1, [1 << 28, rows, last_row, out], stream=reader)
        retirement = reservation.retire(granule, ready=ready, count=rows_per_granule, streams=[reader])
        test.assertIs(reservation.retirement, retirement)
        test.assertTrue(retirement.pending)
        test.assertEqual(retirement.target_bytes, granule)
        test.assertEqual(reservation.ready_bytes, granule)
        test.assertEqual(reservation.mapped_bytes, 4 * granule)
        # The event trails the reader: the first poll declines without waiting and nothing changes.
        test.assertFalse(retirement.event.is_complete)
        test.assertFalse(retirement.reclaim())
        test.assertEqual(reservation.mapped_bytes, 4 * granule)
        # A second retirement, a mapping, a release and a readiness rise above the target are refused while pending.
        with test.assertRaisesRegex(RuntimeError, "pending"):
            reservation.retire(granule, ready=ready, count=0)
        with test.assertRaisesRegex(RuntimeError, "pending"):
            reservation.map_prefix(4 * granule)
        with test.assertRaisesRegex(RuntimeError, "pending"):
            reservation.release()
        with test.assertRaisesRegex(RuntimeError, "pending"):
            reservation.publish_ready(ready, 2 * rows_per_granule, nbytes=2 * granule)
        reservation.publish_ready(ready, rows_per_granule, nbytes=granule)  # at the target is fine
        while not retirement.reclaim():
            time.sleep(0.002)
        wp.synchronize_device(device)
        _runtime().verify_cuda_device(device)  # a premature unmap would have left a sticky error
        test.assertIn(int(out.numpy()[0]), (1000 + last_row, 1000 + last_row - 1))
        test.assertFalse(retirement.pending)
        test.assertIsNone(reservation.retirement)
        test.assertEqual(reservation.mapped_bytes, granule)
        test.assertEqual(ready.numpy()[0], rows_per_granule)
        test.assertTrue(retirement.reclaim())
        # Work the named stream submits after retire() sees the lower count even when the current stream is
        # busy with the write; an unnamed stream would not be ordered.
        reservation.map_prefix(2 * granule)
        reservation.publish_ready(ready, 100, nbytes=2 * granule)
        wp.synchronize_device(device)
        junk = wp.zeros(1, dtype=int)
        wp.launch(spin, 1, [1 << 27, junk])  # the current stream is busy, so the ready write is delayed
        later = reservation.retire(granule, ready=ready, count=7, streams=[reader])
        seen = wp.zeros(1, dtype=int)
        wp.launch(read_word, 1, [ready, seen], stream=reader)
        wp.synchronize_device(device)
        test.assertEqual(int(seen.numpy()[0]), 7)
        test.assertTrue(later.reclaim())
        # Retiring to more than is mapped, below readiness without a ready scalar, with count but no ready,
        # or with a stream that is not a stream on this device, is refused (a second GPU cannot be tested here).
        with test.assertRaisesRegex(ValueError, "Cannot retire"):
            reservation.retire(3 * granule)
        reservation.publish_ready(ready, rows_per_granule, nbytes=granule)
        with test.assertRaisesRegex(ValueError, "ready="):
            reservation.retire(0)
        with test.assertRaisesRegex(ValueError, "count= needs ready="):
            reservation.retire(granule, count=5)
        with test.assertRaisesRegex(ValueError, "Retirement streams"):
            reservation.retire(granule, streams=["cpu"])
        # Retiring without lowering readiness needs no ready scalar.
        same = reservation.retire(granule)
        wp.synchronize_event(same.event)
        test.assertTrue(same.reclaim())
        # A driver failure on unmap leaves the bookkeeping as it was (the native is replaced for one call).
        reservation.map_prefix(2 * granule)
        core = _runtime().core
        real_unmap = core.wp_cuda_vmem_unmap
        core.wp_cuda_vmem_unmap = lambda *args: 0
        try:
            with test.assertRaisesRegex(RuntimeError, "Unmap failed"):
                reservation.unmap_prefix(granule)
            test.assertEqual(reservation.mapped_bytes, 2 * granule)
            test.assertEqual(len(reservation._handles), 2)
        finally:
            core.wp_cuda_vmem_unmap = real_unmap
        test.assertTrue(reservation.unmap_prefix(granule))
        test.assertEqual(reservation.mapped_bytes, granule)
        # Nothing of this is legal during a capture.
        pending_in_capture = reservation.retire(granule)
        with wp.ScopedCapture() as capture:
            with test.assertRaisesRegex(RuntimeError, "capture"):
                reservation.publish_ready(ready, 1, nbytes=granule)
            with test.assertRaisesRegex(RuntimeError, "capture"):
                reservation.retire(0, ready=ready, count=0)
            with test.assertRaisesRegex(RuntimeError, "capture"):
                pending_in_capture.reclaim()
        del capture
        wp.synchronize_event(pending_in_capture.event)
        test.assertTrue(pending_in_capture.reclaim())
        del rows
        gc.collect()
        reservation.release()
        # A retirement whose reservation died first raises when reclaimed; the dying reservation
        # synchronized the device before it unmapped.
        gone = wp.Reservation(granule, device=device)
        gone.map_prefix(granule)
        orphan = gone.retire(0)
        del gone
        gc.collect()
        with test.assertRaisesRegex(RuntimeError, "released"):
            orphan.reclaim()


def test_virtual_empty_and_whole_array_guards(test, device):
    _supported(test, device)
    with wp.ScopedDevice(device):
        granule = wp.Reservation.granularity_bytes(device)
        rows_per_granule = granule // (8 * 4)
        capacity = 4 * rows_per_granule
        a = wp.empty((capacity, 8), dtype=wp.int32, virtual=True)
        test.assertIsInstance(a.backing, wp.Reservation)
        test.assertEqual(a.ptr, a.backing.address)
        test.assertEqual(a.shape, (capacity, 8))
        test.assertEqual(a.backing.mapped_bytes, 0)
        test.assertEqual(a.backing.size_bytes, 4 * granule)
        # Nothing is mapped: every whole-array operation raises instead of faulting.
        dense = wp.zeros((capacity, 8), dtype=wp.int32)
        for name, op in (
            ("numpy", a.numpy),
            ("zero_", a.zero_),
            ("fill_", lambda: a.fill_(1)),
            ("copy", lambda: wp.copy(dense, a)),
            ("copy", lambda: wp.copy(a, dense)),
            ("clone", lambda: wp.clone(a)),
            ("copy", lambda: a.assign(dense)),
            ("clone", lambda: a.to("cpu")),
            ("fill_", lambda: a.fill_(1, extent=(1, 8))),
            ("dlpack", a.__dlpack__),
            ("to_dlpack", lambda: wp.to_dlpack(a)),
            ("contiguous", lambda: a.transpose().contiguous()),
        ):
            with test.assertRaisesRegex(RuntimeError, f"{name} over a partially mapped array"):
                op()
        test.assertIn("Reservation", str(a))
        # Views keep the backing, so the guards follow them.
        test.assertIs(a[:1].backing, a.backing)
        test.assertIs(a.flatten().backing, a.backing)
        test.assertIs(a.transpose().backing, a.backing)
        with test.assertRaisesRegex(RuntimeError, "partially mapped"):
            a[:1].numpy()
        # Map one granule: operations bounded inside it work, anything past it raises.
        a.backing.map_prefix(granule)
        a[:rows_per_granule].fill_(5)
        a.fill_(6, extent=(rows_per_granule, 8))
        np.testing.assert_array_equal(a[:rows_per_granule].numpy(), 6)
        a.zero_(extent=(rows_per_granule // 2, 8))
        np.testing.assert_array_equal(a[: rows_per_granule // 2].numpy(), 0)
        with test.assertRaisesRegex(RuntimeError, "partially mapped"):
            a.fill_(6, extent=(rows_per_granule + 1, 8))
        with test.assertRaisesRegex(RuntimeError, "partially mapped"):
            a[: rows_per_granule + 1].numpy()
        with test.assertRaisesRegex(RuntimeError, "partially mapped"):
            a[rows_per_granule - 2 : rows_per_granule + 2].numpy()  # a slice whose ptr is inside the range
        # Copies: a whole copy of a smaller source into the mapped rows works; counts, offsets and int
        # extents are checked over the elements they address; the copied data is checked.
        small = wp.full((rows_per_granule, 8), 9, dtype=wp.int32)
        wp.copy(a, small)
        np.testing.assert_array_equal(a[:rows_per_granule].numpy(), 9)
        a.assign(np.full((rows_per_granule // 2, 8), 4, dtype=np.int32))
        np.testing.assert_array_equal(a[: rows_per_granule // 2].numpy(), 4)
        wp.copy(dense, a, count=rows_per_granule * 8)
        np.testing.assert_array_equal(dense[:rows_per_granule].numpy(), a[:rows_per_granule].numpy())
        with test.assertRaisesRegex(RuntimeError, "copy over a partially mapped"):
            wp.copy(dense, a, count=rows_per_granule * 8 + 1)
        wp.copy(a, dense, extent=(rows_per_granule, 8))
        with test.assertRaisesRegex(RuntimeError, "copy over a partially mapped"):
            wp.copy(a, dense, extent=(rows_per_granule + 1, 8))
        dense.fill_(2)
        wp.copy(a, dense, dest_offset=8, count=rows_per_granule * 8 - 8)
        np.testing.assert_array_equal(a[1:rows_per_granule].numpy(), 2)
        with test.assertRaisesRegex(RuntimeError, "copy over a partially mapped"):
            wp.copy(a, dense, dest_offset=8, count=rows_per_granule * 8)
        with test.assertRaisesRegex(RuntimeError, "copy over a partially mapped"):
            wp.copy(dense, a, src_offset=8, count=rows_per_granule * 8)
        # Argument errors keep their own messages and do not invalidate anything.
        with test.assertRaisesRegex(RuntimeError, "non-negative"):
            wp.copy(a, dense, dest_offset=-1)
        with test.assertRaisesRegex(ValueError, "exceeds"):
            wp.copy(a, dense, extent=(capacity + 1, 8))
        # A kernel bounded by the owner's count is fine over the capacity-sized array.
        wp.load_module(device=device)
        wp.launch(write_rows, (rows_per_granule, 8), [a, 3])
        np.testing.assert_array_equal(a[:rows_per_granule].numpy()[:, 0], 3000 + np.arange(rows_per_granule))
        # Under a capture the same whole-array operations raise and invalidate the graph, with the reason.
        for name, op in (
            ("zero_", a.zero_),
            ("fill_", lambda: a.fill_(1)),
            ("copy", lambda: wp.copy(a, dense)),
            ("copy", lambda: wp.copy(a, dense, dest_offset=8, count=rows_per_granule * 8)),
        ):
            with wp.ScopedCapture(record_launches=True, record_memory_operations=True) as capture:
                with test.assertRaisesRegex(RuntimeError, f"{name} over a partially mapped array"):
                    op()
            test.assertTrue(capture.graph._capture_error.startswith(f"{name} over a partially mapped array"))
            with test.assertRaisesRegex(RuntimeError, f"Graph is invalid: {name} over a partially mapped array"):
                wp.capture_launch(capture.graph)
            del capture
        # The capture is found through the stream the copy runs on, not only the device's current stream.
        side = wp.Stream(device)
        with wp.ScopedCapture(stream=side) as capture:
            with test.assertRaisesRegex(RuntimeError, "copy over a partially mapped array"):
                wp.copy(a, dense, stream=side)
        with test.assertRaisesRegex(RuntimeError, "Graph is invalid: copy over a partially mapped array"):
            wp.capture_launch(capture.graph)
        del capture
        # A count extent is the owner's to bound: it is accepted over the partially mapped array and
        # replays over the live rows only.
        if wp.is_conditional_graph_supported():
            n_rows = wp.CountParameter(capacity)
            live = wp.zeros(1, dtype=wp.int32)
            a[:rows_per_granule].fill_(1)
            with wp.ScopedCapture(record_launches=True, record_memory_operations=True) as capture:
                wp.capture_update_counts([n_rows], [live])
                a.fill_(7, extent=(n_rows, 8))
            live.fill_(rows_per_granule // 2)
            wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(a[: rows_per_granule // 2].numpy(), 7)
            np.testing.assert_array_equal(a[rows_per_granule // 2 : rows_per_granule].numpy(), 1)
            del capture
        # An indexed view over the array carries the backing and is refused until the array is fully mapped
        # (its index values live on the device).
        idx = wp.array([0, capacity - 1], dtype=int)
        indexed = a[idx]
        test.assertIs(indexed.backing if hasattr(indexed, "backing") else indexed._backing, a.backing)
        for name, op in (
            ("fill_", lambda: indexed.fill_(1)),
            ("fill_", indexed.zero_),
            ("contiguous", indexed.numpy),
            ("copy", lambda: wp.copy(dense[:2], indexed)),
            ("copy", lambda: wp.copy(indexed, dense[:2])),
        ):
            with test.assertRaisesRegex(RuntimeError, f"{name} over a partially mapped array"):
                op()
        # Whole-array natives in wp.utils are refused the same way; a count that stays inside the mapped prefix is fine.
        flat = wp.empty((2 * granule // 4,), dtype=float, virtual=True)
        flat.backing.map_prefix(granule)
        flat[: granule // 4].fill_(1.0)
        with test.assertRaisesRegex(RuntimeError, "array_sum over a partially mapped array"):
            wp.utils.array_sum(flat)
        with test.assertRaisesRegex(RuntimeError, "array_inner over a partially mapped array"):
            wp.utils.array_inner(flat, flat)
        test.assertEqual(float(wp.utils.array_sum(flat, value_count=granule // 4)), float(granule // 4))
        with test.assertRaisesRegex(RuntimeError, "array_sum over a partially mapped array"):
            wp.utils.array_sum(flat, value_count=granule // 4 + 1)
        del flat
        # A copy out of a negative-stride view is checked over the bytes its elements occupy, whatever the sign.
        rev_base = wp.Reservation(2 * granule, device=device)
        rev_base.map_prefix(granule)
        rev = wp.array(ptr=rev_base.address + granule, shape=(16,), strides=(-4,), dtype=wp.int32, backing=rev_base)
        line = wp.zeros(16, dtype=wp.int32)
        with test.assertRaisesRegex(RuntimeError, "copy over a partially mapped array"):
            wp.copy(line, rev, count=2)  # element 0 is the first unmapped word
        wp.copy(line, rev, src_offset=1, count=4)  # elements 1 to 4 lie below the boundary
        del rev
        gc.collect()
        rev_base.release()
        # Fully mapped, the array behaves like any other.
        a.backing.map_prefix(4 * granule)
        a.zero_()
        a.fill_(9)
        np.testing.assert_array_equal(a.numpy(), 9)
        wp.copy(dense, a)
        np.testing.assert_array_equal(dense.numpy(), 9)
        test.assertEqual(int(wp.clone(a).numpy().sum()), 9 * capacity * 8)
        test.assertEqual(a.to("cpu").numpy().shape, (capacity, 8))
        test.assertEqual(indexed.numpy().shape, (2, 8))
        test.assertTrue(wp.to_dlpack(a))
        # Gradients are refused unless the owner supplies them; zeros and full refuse virtual=True outright.
        with test.assertRaisesRegex(ValueError, "requires_grad"):
            wp.empty(4, dtype=float, virtual=True, requires_grad=True)
        b = wp.empty((granule // 4,), dtype=float, virtual=True)
        with test.assertRaisesRegex(ValueError, "grad"):
            b.requires_grad = True
        with test.assertRaisesRegex(ValueError, "virtual=True is only supported by wp.empty"):
            wp.zeros(4, dtype=int, virtual=True)
        with test.assertRaisesRegex(ValueError, "virtual=True is only supported by wp.empty"):
            wp.full(4, 1, dtype=int, virtual=True)
        with test.assertRaisesRegex(ValueError, "positive size"):
            wp.empty(0, dtype=int, virtual=True)
        with test.assertRaisesRegex(RuntimeError, "CUDA device"):
            wp.empty(4, dtype=int, device="cpu", virtual=True)
        # Release is refused while the array lives and allowed afterwards.
        backing = a.backing
        with test.assertRaisesRegex(RuntimeError, "live views"):
            backing.release()
        a = indexed = idx = (
            None  # drop the references (a plain del would make the closures above undefined for the linter)
        )
        gc.collect()
        backing.release()
        # An array that dies inside a reference cycle still releases its reservation and refunds the budget.
        budget = wp.BackingBudget(granule)
        c = wp.empty((granule // 4,), dtype=wp.int32, virtual=True, budget=budget)
        c.backing.map_prefix(granule)
        c.me = c
        del c
        gc.collect()
        test.assertEqual(budget.used_bytes, 0)


def test_array_backing_packed_views(test, device):
    _supported(test, device)
    with wp.ScopedDevice(device):
        granule = wp.Reservation.granularity_bytes(device)
        row_stride = 32  # q at 0 (3 floats), v at 12 (3 floats), 8 bytes of padding
        capacity = 2 * granule // row_stride
        arena = wp.Reservation(capacity * row_stride, device=device)
        q = wp.array(ptr=arena.address, shape=(capacity, 3), strides=(row_stride, 4), dtype=wp.float32, backing=arena)
        v = wp.array(
            ptr=arena.address + 12, shape=(capacity, 3), strides=(row_stride, 4), dtype=wp.float32, backing=arena
        )
        test.assertIs(q.backing, arena)
        test.assertEqual(q.ptr, arena.address)
        test.assertEqual(v.ptr - q.ptr, 12)
        test.assertFalse(q.is_contiguous)
        # Placement is checked against the reservation, not trusted.
        with test.assertRaisesRegex(ValueError, "fit"):
            wp.array(
                ptr=arena.address + 24, shape=(capacity, 3), strides=(row_stride, 4), dtype=wp.float32, backing=arena
            )
        with test.assertRaisesRegex(ValueError, "fit"):
            wp.array(ptr=arena.address - 4, shape=(4,), dtype=wp.float32, backing=arena)
        with test.assertRaisesRegex(ValueError, "deleter"):
            wp.array(ptr=arena.address, shape=(4,), dtype=wp.float32, backing=arena, deleter=lambda p, n: None)
        with test.assertRaisesRegex(RuntimeError, "not constructed from"):
            wp.array(data=[1.0, 2.0], dtype=wp.float32, backing=arena)
        with test.assertRaisesRegex(ValueError, "reservation's device"):
            wp.array(ptr=arena.address, shape=(4,), dtype=wp.float32, device="cpu", backing=arena)
        with test.assertRaisesRegex(TypeError, "Reservation"):
            wp.array(ptr=arena.address, shape=(4,), dtype=wp.float32, backing=object())
        # A negative-stride view is placed by its lowest byte; one that would run before the address is refused.
        reversed_view = wp.array(ptr=arena.address + 60, shape=(16,), strides=(-4,), dtype=wp.float32, backing=arena)
        test.assertEqual(reversed_view._byte_span(), (-60, 4))
        test.assertFalse(reversed_view._is_mapped())
        with test.assertRaisesRegex(ValueError, "fit"):
            wp.array(ptr=arena.address + 56, shape=(16,), strides=(-4,), dtype=wp.float32, backing=arena)
        del reversed_view
        # ptr defaults to the address; a plain view spelling and the explicit one agree.
        whole = wp.array(shape=(capacity * row_stride // 4,), dtype=wp.float32, backing=arena)
        test.assertEqual(whole.ptr, arena.address)
        test.assertEqual(arena.view((4,), wp.float32, offset_bytes=12).ptr, v.ptr)
        # One map_prefix maps all fields' live rows at once; a bounded kernel writes both.
        live = granule // row_stride
        arena.map_prefix(live * row_stride)
        wp.load_module(device=device)
        wp.launch(write_packed, capacity, [q, v, live])
        qn = q[:live].numpy()
        vn = v[:live].numpy()
        expected = 10 * np.arange(live)[:, None] + np.arange(3)[None, :]
        np.testing.assert_array_equal(qn, expected)
        np.testing.assert_array_equal(vn, -expected)
        # Whole-field operations over the capacity raise; bounded ones work.
        with test.assertRaisesRegex(RuntimeError, "partially mapped"):
            q.numpy()
        with test.assertRaisesRegex(RuntimeError, "partially mapped"):
            v.zero_()
        with test.assertRaisesRegex(RuntimeError, "contiguous over a partially mapped"):
            q.contiguous()
        q.zero_(extent=(live, 3))
        np.testing.assert_array_equal(q[:live].numpy(), 0)
        np.testing.assert_array_equal(v[:live].numpy(), -expected)
        # A gradient supplied by the owner as a view over a reservation is accepted, and a slice of the
        # primal carries the gradient's backing too, so the guards hold on the grad side.
        grads = wp.Reservation(capacity * row_stride, device=device)
        grads.map_prefix(granule)
        gq = wp.array(ptr=grads.address, shape=(capacity, 3), strides=(row_stride, 4), dtype=wp.float32, backing=grads)
        pq = wp.array(
            ptr=arena.address,
            shape=(capacity, 3),
            strides=(row_stride, 4),
            dtype=wp.float32,
            backing=arena,
            requires_grad=True,
            grad=gq,
        )
        test.assertTrue(pq.requires_grad)
        test.assertIs(pq.grad, gq)
        test.assertIs(pq[: live + 1].grad.backing, grads)
        with test.assertRaisesRegex(RuntimeError, "zero_ over a partially mapped array"):
            pq[: live + 1].grad.zero_()
        pq[:live].grad.zero_()
        del pq, gq
        gc.collect()
        grads.release()
        # Views over the arena keep it alive; release waits for all of them.
        with test.assertRaisesRegex(RuntimeError, "live views"):
            arena.release()
        del q, v, whole, qn, vn
        gc.collect()
        arena.release()


class TestReservation(unittest.TestCase):
    def test_cpu_rejected(self):
        with self.assertRaises(RuntimeError):
            wp.Reservation(1, device="cpu")
        self.assertFalse(wp.Reservation.is_supported("cpu"))


for test_func in (
    test_reservation_reserve_map_unmap,
    test_reservation_unmap_waits_for_events,
    test_reservation_capture_forbidden_and_grow_keeps_graph,
    test_reservation_budget_refused_cleanly,
    test_reservation_two_step_retirement,
    test_virtual_empty_and_whole_array_guards,
    test_array_backing_packed_views,
):
    add_function_test(TestReservation, test_func.__name__, test_func, devices=get_selected_cuda_test_devices())


if __name__ == "__main__":
    wp.init()
    unittest.main(verbosity=2)
