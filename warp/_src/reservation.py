# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A reserved CUDA virtual address range with a growable physical prefix.

The address of a :class:`Reservation` never moves. Physical memory is mapped
and unmapped in whole granules at the front of the range by host calls made
outside any graph capture, so a captured graph that refers to the address stays
valid while the backing changes. Three byte counts describe a reservation and
always satisfy ``size_bytes >= mapped_bytes >= ready_bytes``: the reserved
range, the mapped prefix, and the prefix whose readiness the owner has
published to the device. Arrays over the range (:meth:`Reservation.view`,
``wp.empty(..., virtual=True)``, ``wp.array(ptr=..., backing=...)``) cover the
capacity; what bounds the rows a kernel touches is the owner's count, and
whole-array operations over a partially mapped array raise instead of faulting.
Budgets are the owner's too: a :class:`BackingBudget` is handed to the
reservations that share it and a mapping that would exceed it is refused before
any driver call.

Unmapping is not stream ordered: :meth:`Reservation.unmap_prefix` trusts the
events the caller names, :meth:`Reservation.retire` records its own event behind
the named reader streams and :meth:`Retirement.reclaim` polls it, and
:meth:`Reservation.release` (or the last reference dying) synchronizes the device
before it unmaps, since nothing else orders the readers of a range that is going
away for good.
"""

from __future__ import annotations

import builtins
import ctypes
import threading
import weakref
from collections.abc import Sequence
from typing import Any, NamedTuple

import warp
import warp._src.context as _context


def _runtime():
    if _context.runtime is None:
        warp.init()
    return _context.runtime


class BackingBudget:
    """A shared limit on mapped physical bytes.

    The owner creates one and passes it to every :class:`Reservation` that
    shares it. :meth:`Reservation.map_prefix` acquires the new granules from
    the budget before any driver call and refuses with :class:`MemoryError`,
    changing nothing, when they do not fit; unmapping and releasing return
    them. A budget is the seam for a device-wide default: a reservation created
    without one is unlimited.
    """

    def __init__(self, limit_bytes: int):
        if type(limit_bytes) is not int or limit_bytes < 0:
            raise ValueError("Budget limit must be a non-negative int")
        self._limit = limit_bytes
        self._used = 0
        self._lock = threading.Lock()

    @property
    def limit_bytes(self) -> int:
        return self._limit

    @property
    def used_bytes(self) -> int:
        """The bytes currently mapped against this budget."""
        return self._used

    @property
    def available_bytes(self) -> int:
        return self._limit - self._used

    def acquire(self, nbytes: int) -> builtins.bool:
        """Take ``nbytes`` from the budget; ``False`` and no change when they do not fit."""
        with self._lock:
            if nbytes > self._limit - self._used:
                return False
            self._used += nbytes
            return True

    def release(self, nbytes: int) -> None:
        """Return ``nbytes`` to the budget."""
        with self._lock:
            self._used -= min(nbytes, self._used)

    def __repr__(self):
        return f"BackingBudget(limit_bytes={self._limit}, used_bytes={self._used})"


class Retirement:
    """The pending second step of a shrink started by :meth:`Reservation.retire`.

    Step one lowered the readiness the device sees and recorded an event after
    the last work that may read the released rows. :meth:`reclaim` polls that
    event without waiting and unmaps the rows once it has completed.
    """

    def __init__(self, reservation: Reservation, target_bytes: int, event: warp.Event):
        self._reservation = weakref.ref(reservation)
        self._target_bytes = target_bytes
        self._event = event
        self._pending = True

    @property
    def pending(self) -> builtins.bool:
        return self._pending

    @property
    def target_bytes(self) -> int:
        """The mapped bytes after reclamation (granule rounded)."""
        return self._target_bytes

    @property
    def event(self) -> warp.Event:
        """The event that marks the last reads of the released rows."""
        return self._event

    def reclaim(self) -> builtins.bool:
        """Unmap the retired rows if their readers are done; ``False`` without waiting otherwise.

        Raises:
            RuntimeError: If the reservation was released first, or a capture is active on any device
                (the event may not be queried then).
        """
        if not self._pending:
            return True
        reservation = self._reservation()
        if reservation is None or reservation._released:
            self._pending = False
            raise RuntimeError("The reservation was released before its retirement was reclaimed")
        runtime = _runtime()
        if reservation._device.is_capturing or runtime.captures:
            raise RuntimeError("Reservation backing cannot change during a graph capture")
        if reservation.mapped_bytes > self._target_bytes:
            if not self._event.is_complete:
                return False
            reservation.unmap_prefix(self._target_bytes)
        self._pending = False
        reservation._finish_retirement(self)
        return True

    def __repr__(self):
        return f"Retirement(target_bytes={self._target_bytes}, pending={self._pending})"


class _DeferredRelease(NamedTuple):
    """A reservation whose last reference died while a capture was open; released at the next safe call."""

    device: Any
    address: int
    size_bytes: int
    mapped_bytes: int
    granularity: int
    handles: tuple
    budget: Any


# Appended by finalizers and drained by the next safe call. No lock: every operation on it is a single list
# method (atomic under the GIL), so a finalizer that runs inside a drain cannot block the thread doing it.
_deferred_releases: list[_DeferredRelease] = []


def _release_range(runtime, device, address, size_bytes, mapped_bytes, granularity, handles, budget):
    """Synchronize the device, unmap what is mapped, free the range, refund the budget. Errors are reported, not raised."""
    with device.context_guard:
        runtime.core.wp_cuda_context_synchronize(device.context)
        if mapped_bytes:
            handle_array = (ctypes.c_uint64 * len(handles))(*handles)
            runtime.core.wp_cuda_vmem_unmap(
                device.context, ctypes.c_void_p(address), mapped_bytes, granularity, handle_array
            )
        runtime.core.wp_cuda_vmem_address_free(device.context, ctypes.c_void_p(address), size_bytes)
    if budget is not None and mapped_bytes:
        budget.release(mapped_bytes)


def _drain_deferred(runtime) -> None:
    """Release the reservations that died during a capture, on devices that are no longer capturing."""
    kept = []
    while True:
        try:
            item = _deferred_releases.pop()
        except IndexError:
            break
        if item.device.is_capturing or runtime.captures:
            kept.append(item)
        else:
            _release_range(runtime, *item)
    for item in kept:
        _deferred_releases.append(item)


class Reservation:
    """A virtual address range of ``size_bytes`` on one CUDA device, backed by a growable prefix.

    Args:
        size_bytes: The range to reserve. Rounded up to the device's granule.
        device: The CUDA device (the current device if ``None``).
        budget: An optional :class:`BackingBudget` shared with other reservations;
            mapping beyond it is refused cleanly.

    Raises:
        RuntimeError: If the device is not a CUDA device or does not support virtual memory management.
        MemoryError: If the driver refuses the reservation.
    """

    def __init__(self, size_bytes: int, device: Any = None, budget: BackingBudget | None = None):
        runtime = _runtime()
        device = runtime.get_device(device)
        if not device.is_cuda:
            raise RuntimeError("Reservations require a CUDA device")
        if not Reservation.is_supported(device):
            raise RuntimeError(f"Device '{device}' does not support virtual memory management")
        if type(size_bytes) is not int or size_bytes <= 0:
            raise ValueError("Reservation size must be a positive int")
        if budget is not None and not (
            callable(getattr(budget, "acquire", None)) and callable(getattr(budget, "release", None))
        ):
            raise TypeError("budget must be a BackingBudget (acquire and release)")
        _drain_deferred(runtime)

        granularity = ctypes.c_size_t()
        with device.context_guard:
            if not runtime.core.wp_cuda_vmem_granularity(device.context, ctypes.byref(granularity)):
                raise RuntimeError(runtime.get_error_string())
        granule = granularity.value
        rounded = ((size_bytes + granule - 1) // granule) * granule

        address = ctypes.c_void_p()
        with device.context_guard:
            if not runtime.core.wp_cuda_vmem_reserve(device.context, rounded, granule, ctypes.byref(address)):
                raise MemoryError(
                    f"Could not reserve {rounded} bytes of virtual address space: {runtime.get_error_string()}"
                )

        self._device = device
        self._address = address.value
        self._size_bytes = rounded
        self._granularity = granule
        self._mapped_bytes = 0
        self._ready_bytes = 0
        self._budget = budget
        self._handles: list[int] = []
        self._released = False
        self._lock = threading.Lock()
        # Arrays over the range keep it alive: token -> (first byte, last byte + 1) relative to the address.
        # The dict is only ever popped without the lock (from array finalizers) and read under it.
        self._views: dict[int, tuple[int, int]] = {}
        self._next_view = 0
        self._retirement: Retirement | None = None

    @staticmethod
    def granularity_bytes(device: Any = None) -> int:
        """The mapping granule of ``device`` in bytes; reservations and prefixes round up to it."""
        runtime = _runtime()
        device = runtime.get_device(device)
        if not device.is_cuda or not Reservation.is_supported(device):
            raise RuntimeError(f"Device '{device}' does not support virtual memory management")
        granularity = ctypes.c_size_t()
        with device.context_guard:
            if not runtime.core.wp_cuda_vmem_granularity(device.context, ctypes.byref(granularity)):
                raise RuntimeError(runtime.get_error_string())
        return granularity.value

    @staticmethod
    def is_supported(device: Any = None) -> builtins.bool:
        """Whether ``device`` supports the CUDA virtual memory management API."""
        runtime = _runtime()
        device = runtime.get_device(device)
        if not device.is_cuda:
            return False
        return bool(runtime.core.wp_cuda_vmem_supported(device.context))

    @property
    def device(self):
        return self._device

    @property
    def address(self) -> int:
        """The start of the range; it never changes."""
        return self._address

    @property
    def size_bytes(self) -> int:
        """The reserved size, rounded up to the granule."""
        return self._size_bytes

    @property
    def granularity(self) -> int:
        """The mapping granule in bytes."""
        return self._granularity

    @property
    def mapped_bytes(self) -> int:
        """The bytes with physical backing, always a prefix and a multiple of the granule."""
        return self._mapped_bytes

    @property
    def ready_bytes(self) -> int:
        """The prefix whose readiness the owner last published through this reservation; never above ``mapped_bytes``."""
        return self._ready_bytes

    @property
    def budget(self) -> BackingBudget | None:
        return self._budget

    @property
    def retirement(self) -> Retirement | None:
        """The pending retirement, if :meth:`retire` was called and :meth:`Retirement.reclaim` has not finished."""
        return self._retirement

    def _check(self):
        if self._released:
            raise RuntimeError("Reservation has been released")

    def _check_not_capturing(self, runtime):
        # Device.is_capturing also sees captures Warp did not start (cudaStreamIsCapturing on the current stream).
        if self._device.is_capturing or runtime.captures:
            raise RuntimeError("Reservation backing cannot change during a graph capture")

    def _pending_retirement(self):
        retirement = self._retirement
        return retirement if retirement is not None and retirement.pending else None

    def _round_up(self, nbytes: int) -> int:
        if type(nbytes) is not int or nbytes < 0:
            raise ValueError("Byte counts must be non-negative ints")
        return ((nbytes + self._granularity - 1) // self._granularity) * self._granularity

    def covers_bytes(self, ptr: int, nbytes: int) -> builtins.bool:
        """Whether ``nbytes`` from address ``ptr`` lie inside the mapped prefix (an empty range always does)."""
        if nbytes <= 0:
            return True
        return self._address <= ptr and ptr + nbytes <= self._address + self._mapped_bytes

    def map_prefix(self, nbytes: int) -> int:
        """Back the first ``nbytes`` (rounded up to the granule) with physical memory.

        Growing into never-mapped addresses needs no wait on readers: nothing valid can
        have reached them. Creates, maps and grants access to the new granules; a failure
        rolls the new granules back and leaves the mapped prefix as it was. The budget, if
        any, is charged before any driver call and refunded on failure. Not legal during a
        graph capture, nor while a retirement is pending.

        Returns:
            The mapped bytes after the call.

        Raises:
            ValueError: If ``nbytes`` exceeds ``size_bytes``.
            RuntimeError: If a capture is active or a retirement is pending.
            MemoryError: If the budget or the driver cannot provide the memory.
        """
        runtime = _runtime()
        target = self._round_up(nbytes)
        with self._lock:
            self._check()
            if target > self._size_bytes:
                raise ValueError(f"Cannot map {target} bytes of a {self._size_bytes}-byte reservation")
            self._check_not_capturing(runtime)
            if self._pending_retirement() is not None:
                raise RuntimeError("A retirement is pending; reclaim it before mapping more")
            _drain_deferred(runtime)
            if target <= self._mapped_bytes:
                return self._mapped_bytes
            delta = target - self._mapped_bytes
            if self._budget is not None and not self._budget.acquire(delta):
                raise MemoryError(
                    f"Mapping {delta} more bytes would exceed the backing budget "
                    f"({self._budget.used_bytes} of {self._budget.limit_bytes} bytes in use); nothing was mapped"
                )
            count = delta // self._granularity
            handles = (ctypes.c_uint64 * count)()
            with self._device.context_guard:
                ok = runtime.core.wp_cuda_vmem_map(
                    self._device.context,
                    ctypes.c_void_p(self._address + self._mapped_bytes),
                    delta,
                    self._granularity,
                    handles,
                )
            if not ok:
                if self._budget is not None:
                    self._budget.release(delta)
                raise MemoryError(f"Could not map {delta} bytes: {runtime.get_error_string()}")
            self._handles.extend(handles)
            self._mapped_bytes = target
            return self._mapped_bytes

    def unmap_prefix(self, nbytes: int, after: Sequence[warp.Event] = ()) -> builtins.bool:
        """Shrink the backing to the first ``nbytes`` (rounded up to the granule) once ``after`` is complete.

        The caller names the events that mark the last reads of the released rows; with
        none named the caller has ordered the readers itself. If any event is not yet
        complete the call returns ``False`` without waiting and changes nothing; call again
        later (or use :meth:`retire`, which keeps the record). Rows whose readiness is still
        published cannot be released: withdraw it first. Arrays over the range do not pin
        the backing; the owner bounds the rows they touch. Not legal during a graph capture.
        A driver failure leaves the bookkeeping as it was and raises.

        Returns:
            ``True`` once the backing is at or below ``nbytes``.

        Raises:
            ValueError: If ``nbytes`` is below the published readiness.
            RuntimeError: If a capture is active, or the driver refused the unmap.
        """
        runtime = _runtime()
        target = self._round_up(nbytes)
        with self._lock:
            self._check()
            self._check_not_capturing(runtime)
            if target >= self._mapped_bytes:
                return True
            if target < self._ready_bytes:
                raise ValueError(
                    f"Readiness is published over {self._ready_bytes} bytes; withdraw it (retire) before unmapping below"
                )
            for event in after:
                if not event.is_complete:
                    return False
            delta = self._mapped_bytes - target
            count = delta // self._granularity
            keep = len(self._handles) - count
            handles = (ctypes.c_uint64 * count)(*self._handles[keep:])
            with self._device.context_guard:
                ok = runtime.core.wp_cuda_vmem_unmap(
                    self._device.context,
                    ctypes.c_void_p(self._address + target),
                    delta,
                    self._granularity,
                    handles,
                )
            if not ok:
                raise RuntimeError(f"Unmap failed: {runtime.get_error_string()}")
            del self._handles[keep:]
            self._mapped_bytes = target
            if self._budget is not None:
                self._budget.release(delta)
            return True

    def _check_ready_array(self, ready):
        if (
            not isinstance(ready, warp.array)
            or ready.dtype is not warp.int32
            or ready.shape != (1,)
            or not ready.is_contiguous
            or ready.device != self._device
        ):
            raise ValueError("The ready scalar must be one contiguous int32 element on the reservation's device")

    def publish_ready(self, ready: warp.array, count: int, *, nbytes: int) -> None:
        """Publish ``count`` into the device scalar ``ready`` once the first ``nbytes`` are mapped.

        A device write on the current stream that the graph reads; separate from mapping,
        because mapping bytes does not initialize them. Readiness only rises here (it is
        withdrawn through :meth:`retire`) and never exceeds the mapped prefix, nor the target
        of a pending retirement. ``nbytes`` is the readiness in bytes (``count`` rows times
        the row stride, which the reservation does not know); it is kept as
        :attr:`ready_bytes`. Not legal during a graph capture.

        Raises:
            ValueError: If ``nbytes`` exceeds ``mapped_bytes`` or is below the published readiness.
            RuntimeError: If a capture is active, or a pending retirement's target is below ``nbytes``.
        """
        runtime = _runtime()
        self._check_ready_array(ready)
        if type(count) is not int or count < 0:
            raise ValueError("The ready count must be a non-negative int")
        if type(nbytes) is not int or nbytes < 0:
            raise ValueError("Byte counts must be non-negative ints")
        with self._lock:
            self._check()
            self._check_not_capturing(runtime)
            if nbytes > self._mapped_bytes:
                raise ValueError(
                    f"Readiness over {nbytes} bytes exceeds the mapped prefix of {self._mapped_bytes} bytes"
                )
            if nbytes < self._ready_bytes:
                raise ValueError("Readiness is withdrawn through retire(), not publish_ready()")
            retirement = self._pending_retirement()
            if retirement is not None and nbytes > retirement.target_bytes:
                raise RuntimeError(
                    "A retirement is pending; readiness cannot rise above its target until it is reclaimed"
                )
            with warp.ScopedDevice(self._device):
                ready.fill_(count)
            self._ready_bytes = nbytes

    def retire(
        self,
        nbytes: int,
        *,
        ready: warp.array | None = None,
        count: int = 0,
        streams: Sequence[warp.Stream] = (),
    ) -> Retirement:
        """Step one of a shrink: withdraw readiness to ``nbytes`` and mark the readers' last use.

        On the current stream, after every stream in ``streams`` has reached this point,
        writes ``count`` into ``ready`` and records an event; the named streams then wait for
        that publication, so work they submit later sees the lower readiness. The backing
        stays mapped until :meth:`Retirement.reclaim` finds the event complete and unmaps
        down to ``nbytes`` rounded up to the granule. The host never waits. ``ready`` may be
        omitted only when readiness is not being lowered (``nbytes`` at or above
        :attr:`ready_bytes`), because the device must see the lower count before rows go.
        Name every stream that may read the released rows; unnamed streams are the owner's
        to order. One retirement is pending at a time, and mapping, publishing above its
        target and releasing are refused until it is reclaimed. Not legal during a capture.

        Raises:
            ValueError: If ``nbytes`` exceeds the mapped prefix, ``ready`` is missing while readiness drops,
                ``count`` is given without ``ready``, or a stream is on another device.
            RuntimeError: If a capture is active or a retirement is already pending.
        """
        runtime = _runtime()
        if ready is not None:
            self._check_ready_array(ready)
        elif count:
            raise ValueError("count= needs ready=; nothing is written without a ready scalar")
        if type(count) is not int or count < 0:
            raise ValueError("The ready count must be a non-negative int")
        if type(nbytes) is not int or nbytes < 0:
            raise ValueError("Byte counts must be non-negative ints")
        streams = tuple(streams)
        for stream in streams:
            if not isinstance(stream, warp.Stream) or stream.device != self._device:
                raise ValueError("Retirement streams must be Warp streams on the reservation's device")
        with self._lock:
            self._check()
            self._check_not_capturing(runtime)
            if self._pending_retirement() is not None:
                raise RuntimeError("A retirement is already pending; reclaim it first")
            if nbytes > self._mapped_bytes:
                raise ValueError(f"Cannot retire to {nbytes} bytes: only {self._mapped_bytes} bytes are mapped")
            if ready is None and nbytes < self._ready_bytes:
                raise ValueError(
                    f"Retiring below the published readiness ({self._ready_bytes} bytes) needs ready= so the device sees the lower count"
                )
            with warp.ScopedDevice(self._device):
                current = warp.get_stream(self._device)
                for stream in streams:
                    if stream.cuda_stream != current.cuda_stream:
                        current.wait_stream(stream)
                if ready is not None:
                    ready.fill_(count)
                event = warp.Event(self._device)
                current.record_event(event)
                for stream in streams:
                    if stream.cuda_stream != current.cuda_stream:
                        stream.wait_event(event)
            if ready is not None:
                self._ready_bytes = min(self._ready_bytes, nbytes)
            retirement = Retirement(self, self._round_up(nbytes), event)
            self._retirement = retirement
            return retirement

    def _finish_retirement(self, retirement):
        with self._lock:
            if self._retirement is retirement:
                self._retirement = None

    def _attach(self, array) -> None:
        """Register ``array`` (already constructed over ``ptr``) as a view that retains this reservation."""
        if array.device != self._device:
            raise ValueError("An array over a reservation must be on the reservation's device")
        lo, hi = array._byte_span()
        begin = array.ptr + lo - self._address
        end = array.ptr + hi - self._address
        with self._lock:
            self._check()
            if array.ptr < self._address or begin < 0 or end > self._size_bytes:
                raise ValueError("The array does not fit inside the reservation")
            token = self._next_view
            self._next_view += 1
            self._views[token] = (begin, end)

        # The closure holds the reservation strongly (a weak reference would be cleared before a finalizer
        # runs when the array dies in a reference cycle) and takes no lock (a finalizer may run while this
        # thread holds it). The array already retains the reservation through its backing, so no cycle.
        def release_view(ptr, capacity, _token=token, _reservation=self):
            _reservation._views.pop(_token, None)

        array.deleter = release_view
        array._backing = self

    def view(
        self, shape: int | Sequence[int], dtype: Any, offset_bytes: int = 0, strides: Sequence[int] | None = None
    ) -> warp.array:
        """A Warp array over the range at ``offset_bytes``; it retains the reservation.

        The array may cover unmapped rows: the reservation is the capacity, the
        caller bounds the rows it touches (a count, a device guard). Its pointer
        never changes while the backing does. Whole-array operations raise while
        any addressed byte is unmapped. Same as ``wp.array(ptr=address + offset_bytes,
        shape=shape, dtype=dtype, strides=strides, backing=reservation)``.
        """
        self._check()
        if type(offset_bytes) is not int or offset_bytes < 0:
            raise ValueError("offset_bytes must be a non-negative int")
        return warp.array(
            ptr=self._address + offset_bytes,
            dtype=dtype,
            shape=shape,
            strides=None if strides is None else tuple(strides),
            device=self._device,
            backing=self,
        )

    def release(self) -> None:
        """Synchronize the device, unmap everything and free the address range.

        Refused while an array over the range is alive, while a retirement is pending, and
        during a graph capture. Dropping the last reference instead releases the same way
        (or defers it past an open capture).
        """
        runtime = _runtime()
        with self._lock:
            if self._released:
                return
            if self._pending_retirement() is not None:
                raise RuntimeError("A retirement is pending; reclaim it before releasing")
            if self._views:
                raise RuntimeError("Reservation still has live views")
            self._check_not_capturing(runtime)
            self._release_locked(runtime)
        _drain_deferred(runtime)

    def _release_locked(self, runtime):
        _release_range(
            runtime,
            self._device,
            self._address,
            self._size_bytes,
            self._mapped_bytes,
            self._granularity,
            tuple(self._handles),
            self._budget,
        )
        self._handles = []
        self._mapped_bytes = 0
        self._ready_bytes = 0
        self._retirement = None
        self._released = True

    def __del__(self):
        # Runs when the last reference dies, which can only be after every array over the range is gone
        # or dying in the same collection, so the view registry is not consulted here.
        try:
            if getattr(self, "_released", True):
                return
            runtime = _context.runtime
            if runtime is None or runtime.core is None:
                return
            if self._device.is_capturing or runtime.captures:
                # Unmapping (and the synchronize before it) is not legal inside a capture: finish later.
                _deferred_releases.append(
                    _DeferredRelease(
                        self._device,
                        self._address,
                        self._size_bytes,
                        self._mapped_bytes,
                        self._granularity,
                        tuple(self._handles),
                        self._budget,
                    )
                )
                self._released = True
                return
            with self._lock:
                self._release_locked(runtime)
        except Exception:
            pass

    def __repr__(self):
        return (
            f"Reservation(address=0x{self._address:x}, size_bytes={self._size_bytes}, "
            f"mapped_bytes={self._mapped_bytes}, ready_bytes={self._ready_bytes}, "
            f"granularity={self._granularity}, device='{self._device}')"
        )


def empty_virtual(shape, dtype, device=None, *, budget: BackingBudget | None = None, **kwargs) -> warp.array:
    """``wp.empty(shape, dtype, device, virtual=True)``: one array at capacity over its own reservation, unmapped.

    The array's ``backing`` is the :class:`Reservation`; map rows with
    ``backing.map_prefix`` and bound the rows kernels touch with a count. Gradients
    are refused (a fully backed gradient would defeat the point; supply ``grad=`` as a
    view over a reservation mapped in step through ``wp.array(backing=)``).
    """
    requires_grad = kwargs.pop("requires_grad", False)
    retain_grad = kwargs.pop("retain_grad", False)
    pinned = kwargs.pop("pinned", False)
    if kwargs:
        raise TypeError(f"Unexpected arguments for a virtual array: {sorted(kwargs)}")
    if requires_grad or retain_grad:
        raise ValueError("requires_grad is refused on a virtual array; pass grad= as a view over a reservation")
    if pinned:
        raise ValueError("Virtual arrays live on a CUDA device and cannot be pinned")
    runtime = _runtime()
    device = runtime.get_device(device)
    if not device.is_cuda:
        raise RuntimeError("Virtual arrays require a CUDA device")
    if getattr(runtime, "_apic_capture", None) is not None:
        raise NotImplementedError("Virtual arrays are not supported under APIC capture")
    if dtype is int:
        dtype = warp.int32
    elif dtype is float:
        dtype = warp.float32
    elif dtype is builtins.bool:
        dtype = warp.bool
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(int(s) for s in shape)
    nbytes = warp.types.type_size_in_bytes(dtype)
    for s in shape:
        nbytes *= s
    if nbytes <= 0:
        raise ValueError("A virtual array needs a positive size")
    reservation = Reservation(nbytes, device=device, budget=budget)
    return reservation.view(shape, dtype)
