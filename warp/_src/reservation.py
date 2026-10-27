# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A reserved CUDA virtual address range with a growable physical prefix.

The address of a :class:`Reservation` never moves. Physical memory is mapped
and unmapped in whole granules at the front of the range by host calls made
outside any graph capture, so a captured graph that refers to the address stays
valid while the backing changes. Which rows are mapped, and which rows are
ready to be read, are the owner's to track; this class knows ``mapped_bytes``
and nothing about readiness. It is the first, deliberately small, step of the
virtual-memory layer: no budget, no handle pool, no guards on whole-array
operations over a partially mapped array.
"""

from __future__ import annotations

import ctypes
import threading
import weakref
from collections.abc import Sequence
from typing import Any

import warp
import warp._src.context as _context


def _runtime():
    if _context.runtime is None:
        warp.init()
    return _context.runtime


class Reservation:
    """A virtual address range of ``size_bytes`` on one CUDA device, backed by a growable prefix.

    Args:
        size_bytes: The range to reserve. Rounded up to the device's granule.
        device: The CUDA device (the current device if ``None``).

    Raises:
        RuntimeError: If the device is not a CUDA device or does not support virtual memory management.
        MemoryError: If the driver refuses the reservation.
    """

    def __init__(self, size_bytes: int, device: Any = None):
        runtime = _runtime()
        device = runtime.get_device(device)
        if not device.is_cuda:
            raise RuntimeError("Reservations require a CUDA device")
        if not Reservation.is_supported(device):
            raise RuntimeError(f"Device '{device}' does not support virtual memory management")
        if type(size_bytes) is not int or size_bytes <= 0:
            raise ValueError("Reservation size must be a positive int")

        granularity = ctypes.c_size_t()
        with device.context_guard:
            if not runtime.core.wp_cuda_vmem_granularity(device.context, ctypes.byref(granularity)):
                raise RuntimeError(runtime.get_error_string())
        granule = granularity.value
        rounded = ((size_bytes + granule - 1) // granule) * granule

        address = ctypes.c_void_p()
        with device.context_guard:
            if not runtime.core.wp_cuda_vmem_reserve(device.context, rounded, granule, ctypes.byref(address)):
                raise MemoryError(f"Could not reserve {rounded} bytes of virtual address space: {runtime.get_error_string()}")

        self._device = device
        self._address = address.value
        self._size_bytes = rounded
        self._requested_bytes = size_bytes
        self._granularity = granule
        self._mapped_bytes = 0
        self._handles: list[int] = []
        self._released = False
        self._lock = threading.Lock()
        # Views over the range keep it alive; the largest byte extent any live view covers
        # is the floor for unmap_prefix.
        self._view_extents: dict[int, int] = {}
        self._next_view = 0

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
    def is_supported(device: Any = None) -> bool:
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

    def _check(self):
        if self._released:
            raise RuntimeError("Reservation has been released")

    def _round_up(self, nbytes: int) -> int:
        if type(nbytes) is not int or nbytes < 0:
            raise ValueError("Byte counts must be non-negative ints")
        return ((nbytes + self._granularity - 1) // self._granularity) * self._granularity

    def map_prefix(self, nbytes: int) -> int:
        """Back the first ``nbytes`` (rounded up to the granule) with physical memory.

        Growing into never-mapped addresses needs no wait on readers: nothing valid can
        have reached them. Creates, maps and grants access to the new granules; a failure
        rolls the new granules back and leaves the mapped prefix as it was. Not legal
        during a graph capture on this device.

        Returns:
            The mapped bytes after the call.

        Raises:
            ValueError: If ``nbytes`` exceeds ``size_bytes``.
            RuntimeError: If a capture is active on the device.
            MemoryError: If the driver cannot provide the memory.
        """
        runtime = _runtime()
        target = self._round_up(nbytes)
        with self._lock:
            self._check()
            if target > self._size_bytes:
                raise ValueError(f"Cannot map {target} bytes of a {self._size_bytes}-byte reservation")
            if self._device.captures:
                raise RuntimeError("Reservation backing cannot change during a graph capture")
            if target <= self._mapped_bytes:
                return self._mapped_bytes
            delta = target - self._mapped_bytes
            count = delta // self._granularity
            handles = (ctypes.c_uint64 * count)()
            with self._device.context_guard:
                if not runtime.core.wp_cuda_vmem_map(
                    self._device.context,
                    ctypes.c_void_p(self._address + self._mapped_bytes),
                    delta,
                    self._granularity,
                    handles,
                ):
                    raise MemoryError(f"Could not map {delta} bytes: {runtime.get_error_string()}")
            self._handles.extend(handles)
            self._mapped_bytes = target
            return self._mapped_bytes

    def unmap_prefix(self, nbytes: int, after: Sequence[warp.Event] = ()) -> bool:
        """Shrink the backing to the first ``nbytes`` (rounded up to the granule) once ``after`` is complete.

        The caller names the events that mark the last reads of the released rows.
        If any event is not yet complete the call returns ``False`` without waiting and
        changes nothing; call again later. Rows below the largest live view's extent
        cannot be released. Not legal during a graph capture on this device.

        Returns:
            ``True`` once the backing is at or below ``nbytes``.

        Raises:
            ValueError: If a live view still covers released rows.
            RuntimeError: If a capture is active on the device.
        """
        runtime = _runtime()
        target = self._round_up(nbytes)
        with self._lock:
            self._check()
            if self._device.captures:
                raise RuntimeError("Reservation backing cannot change during a graph capture")
            if target >= self._mapped_bytes:
                return True
            floor = max(self._view_extents.values(), default=0)
            if target < min(floor, self._mapped_bytes):
                raise ValueError(f"A view still covers {floor} bytes; cannot unmap below that")
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
            del self._handles[keep:]
            self._mapped_bytes = target
            if not ok:
                raise RuntimeError(f"Unmap failed: {runtime.get_error_string()}")
            return True

    def view(self, shape: int | Sequence[int], dtype: Any, offset_bytes: int = 0) -> warp.array:
        """A Warp array over the range at ``offset_bytes``; it retains the reservation.

        The array may cover unmapped rows: the reservation is the capacity, the
        caller bounds the rows it touches (a count, a device guard). Its pointer
        never changes while the backing does.
        """
        self._check()
        if isinstance(shape, int):
            shape = (shape,)
        shape = tuple(int(s) for s in shape)
        nbytes = warp.types.type_size_in_bytes(dtype)
        for s in shape:
            nbytes *= s
        if offset_bytes < 0 or offset_bytes + nbytes > self._size_bytes:
            raise ValueError("View does not fit inside the reservation")
        with self._lock:
            token = self._next_view
            self._next_view += 1
            self._view_extents[token] = offset_bytes + nbytes
        owner = weakref.ref(self)

        def release_view(ptr, capacity, _token=token, _owner=owner):
            reservation = _owner()
            if reservation is not None:
                with reservation._lock:
                    reservation._view_extents.pop(_token, None)

        array = warp.array(
            ptr=self._address + offset_bytes,
            dtype=dtype,
            shape=shape,
            device=self._device,
            deleter=release_view,
        )
        array._backing = self
        return array

    def release(self) -> None:
        """Unmap everything and free the address range. Refused while a view is alive."""
        runtime = _runtime()
        with self._lock:
            if self._released:
                return
            if self._view_extents:
                raise RuntimeError("Reservation still has live views")
            if self._device.captures:
                raise RuntimeError("Reservation cannot be released during a graph capture")
            self._release_locked(runtime)

    def _release_locked(self, runtime):
        with self._device.context_guard:
            if self._mapped_bytes:
                handles = (ctypes.c_uint64 * len(self._handles))(*self._handles)
                runtime.core.wp_cuda_vmem_unmap(
                    self._device.context,
                    ctypes.c_void_p(self._address),
                    self._mapped_bytes,
                    self._granularity,
                    handles,
                )
            runtime.core.wp_cuda_vmem_address_free(self._device.context, ctypes.c_void_p(self._address), self._size_bytes)
        self._handles = []
        self._mapped_bytes = 0
        self._released = True

    def __del__(self):
        try:
            if getattr(self, "_released", True) or self._view_extents:
                return
            runtime = _context.runtime
            if runtime is None or runtime.core is None:
                return
            with self._lock:
                self._release_locked(runtime)
        except Exception:
            pass

    def __repr__(self):
        return (
            f"Reservation(address=0x{self._address:x}, size_bytes={self._size_bytes}, "
            f"mapped_bytes={self._mapped_bytes}, granularity={self._granularity}, device='{self._device}')"
        )
