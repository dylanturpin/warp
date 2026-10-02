# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit capture lowering for whole-array memory operations.

Kernels preserve byte representation and expose the leading row axis in their
launch bounds. They know no storage owner, readable prefix, or execution count.
The consumer binds those relationships after capture from immutable records.
"""

import math
import operator

import warp as wp
from warp._src import context, types

wp.set_module_options({"enable_backward": False, "default_grid_stride": True})

_BytePattern = types.vector(length=128, dtype=wp.uint8)
_WordPattern = types.vector(length=32, dtype=wp.uint32)


@wp.kernel
def _fill_bytes(destination: wp.array2d[wp.uint8], pattern: _BytePattern, period: int):
    row, column = wp.tid()
    destination[row, column] = pattern[column % period]


@wp.kernel
def _fill_words(destination: wp.array2d[wp.uint32], pattern: _WordPattern, period: int):
    row, column = wp.tid()
    destination[row, column] = pattern[column % period]


@wp.kernel
def _fill_constant_byte(destination: wp.array2d[wp.uint8], value: wp.uint8):
    row, column = wp.tid()
    destination[row, column] = value


@wp.kernel
def _fill_constant_word(destination: wp.array2d[wp.uint32], value: wp.uint32):
    row, column = wp.tid()
    destination[row, column] = value


@wp.kernel
def _copy_bytes(destination: wp.array2d[wp.uint8], source: wp.array2d[wp.uint8]):
    row, column = wp.tid()
    destination[row, column] = source[row, column]


@wp.kernel
def _copy_words(destination: wp.array2d[wp.uint32], source: wp.array2d[wp.uint32]):
    row, column = wp.tid()
    destination[row, column] = source[row, column]


def _row_bytes(array, graph):
    if not isinstance(array, wp.array):
        raise NotImplementedError("Memory operation capture requires regular Warp arrays")
    if array.device != graph.device:
        raise NotImplementedError("Memory operation capture requires arrays on the captured CUDA device")
    if array.requires_grad or context.runtime.tape is not None:
        raise NotImplementedError("Memory operation capture does not support autodiff arrays or an active Tape")
    if not types.type_is_value(array.dtype):
        raise NotImplementedError("Memory operation capture supports scalar, vector and matrix element types")
    descriptor = array.__ctype__()
    if (
        descriptor.data != (array.ptr or 0)
        or descriptor.ndim != array.ndim
        or tuple(descriptor.shape[: array.ndim]) != array.shape
        or tuple(descriptor.strides[: array.ndim]) != array.strides
        or array.size != math.prod(array.shape)
    ):
        raise ValueError("Memory operation array descriptor changed after its Warp ABI was prepared")
    width = types.type_size_in_bytes(array.dtype)
    for axis in range(array.ndim - 1, 0, -1):
        if array.size and array.strides[axis] != width:
            raise NotImplementedError("Memory operation capture requires contiguous inner dimensions")
        width *= array.shape[axis]
    if array.size and (not array.ptr or array.strides[0] < width):
        raise ValueError("Memory operation capture requires a valid pointer and nonoverlapping positive row strides")
    return width


def _transfer_view(array, width, dtype):
    unit = types.type_size_in_bytes(dtype)
    view = wp.array(
        ptr=array.ptr,
        shape=(array.shape[0], width // unit),
        strides=(array.strides[0], unit),
        dtype=dtype,
        device=array.device,
    )
    view._ref = array
    return view


def _word_aligned(array, width):
    return array.ptr % 4 == 0 and array.strides[0] % 4 == 0 and width % 4 == 0


def _overlapping_rows(destination, source, width):
    left, right = sorted((destination, source), key=lambda array: array.ptr)
    count = left.shape[0]
    if left.ptr + (count - 1) * left.strides[0] + width <= right.ptr:
        return False
    if left.strides[0] != right.strides[0]:
        return True  # Distinct pitches require a stronger disjointness proof.
    quotient, remainder = divmod(right.ptr - left.ptr, left.strides[0])
    return (remainder < width and quotient < count) or (remainder + width > left.strides[0] and quotient + 1 < count)


def fill(graph, array, value_bytes):
    """Lower one converted fill value; preserve its complete typed byte pattern."""
    try:
        context._validate_capture(graph)
        width = _row_bytes(array, graph)
        element_bytes = types.type_size_in_bytes(array.dtype)
        if len(value_bytes) != element_bytes or element_bytes > 128:
            raise NotImplementedError("Memory operation capture supports fill elements of at most 128 bytes")
        if not array.size:
            return
        descriptor = bytes(array.__ctype__())
        words = _word_aligned(array, width)
        # Whole-word transfers preserve even subword/odd-sized element periods.
        pattern_bytes = value_bytes * (4 // math.gcd(element_bytes, 4)) if words else value_bytes
        if len(pattern_bytes) > 128:
            words, pattern_bytes = False, value_bytes
        if words:
            values = [int.from_bytes(pattern_bytes[i : i + 4], "little") for i in range(0, len(pattern_bytes), 4)]
            if len(set(values)) == 1:
                kernel, arguments = _fill_constant_word, [wp.uint32(values[0])]
            else:
                kernel, arguments = _fill_words, [_WordPattern(*values, *([0] * (32 - len(values)))), len(values)]
            dtype = wp.uint32
        else:
            if len(set(pattern_bytes)) == 1:
                kernel, arguments = _fill_constant_byte, [wp.uint8(pattern_bytes[0])]
            else:
                kernel = _fill_bytes
                arguments = [_BytePattern(*pattern_bytes, *([0] * (128 - len(pattern_bytes)))), len(pattern_bytes)]
            dtype = wp.uint8
        transfer = _transfer_view(array, width, dtype)
        before = len(graph._captured_launches)
        wp.launch(kernel, dim=transfer.shape, inputs=[transfer, *arguments], device=array.device)
        if len(graph._captured_launches) != before + 1:
            raise RuntimeError("Memory operation did not produce one canonical captured launch")
        graph._captured_memory_operations.append(
            context.CapturedFill(graph._captured_launches[-1], array.dtype, descriptor, value_bytes)
        )
    except BaseException as error:
        context.capture_invalidate(graph, f"Failed memory operation capture: {error}")
        raise


def copy(graph, destination, source, destination_offset, source_offset, count, stream):
    """Lower an exact full-array copy; reject partial or overlapping payloads."""
    try:
        context._validate_capture(graph)
        width = _row_bytes(destination, graph)
        source_width = _row_bytes(source, graph)
        destination_offset, source_offset, count = map(operator.index, (destination_offset, source_offset, count))
        if destination_offset or source_offset or count not in (0, source.size):
            raise NotImplementedError("Memory operation capture supports only full-array copies without offsets")
        if destination.shape != source.shape or width != source_width:
            raise NotImplementedError("Memory operation capture requires equal shapes and element byte sizes")
        if stream is not None and stream.device != graph.device:
            raise NotImplementedError("Memory operation capture requires a stream on the captured CUDA device")
        if not destination.size:
            return
        destination_descriptor, source_descriptor = bytes(destination.__ctype__()), bytes(source.__ctype__())
        if destination_descriptor == source_descriptor:
            return
        if _overlapping_rows(destination, source, width):
            raise ValueError("Memory operation capture rejects overlapping copy payloads")
        words = _word_aligned(destination, width) and _word_aligned(source, width)
        kernel, dtype = (_copy_words, wp.uint32) if words else (_copy_bytes, wp.uint8)
        dst, src = _transfer_view(destination, width, dtype), _transfer_view(source, width, dtype)
        before = len(graph._captured_launches)
        wp.launch(kernel, dim=dst.shape, inputs=[dst, src], stream=stream, device=destination.device)
        if len(graph._captured_launches) != before + 1:
            raise RuntimeError("Memory operation did not produce one canonical captured launch")
        graph._captured_memory_operations.append(
            context.CapturedCopy(
                graph._captured_launches[-1], destination.dtype, destination_descriptor, source.dtype, source_descriptor
            )
        )
    except BaseException as error:
        context.capture_invalidate(graph, f"Failed memory operation capture: {error}")
        raise
