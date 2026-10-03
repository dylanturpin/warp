# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from typing import Any

import numpy as np

import warp as wp
from warp._src import builtins
from warp._src.types import tile
from warp.tests.unittest_utils import *

wp.init()  # For wp._src.context.runtime.core.wp_is_mathdx_enabled()

TILE_M = wp.constant(8)
TILE_N = wp.constant(4)
TILE_K = wp.constant(8)

# num threads per-tile
TILE_DIM = 32

# Forward-only kernels skip adjoint codegen
wp.get_module("test_cholesky_fwd").options["enable_backward"] = False


@wp.kernel(module="test_cholesky_fwd")
def tile_math_cholesky(
    gA: wp.array2d[wp.float64],
    gD: wp.array1d[wp.float64],
    gL: wp.array2d[wp.float64],
    gy: wp.array1d[wp.float64],
    gx: wp.array1d[wp.float64],
):
    # Load A, D & y
    a = wp.tile_load(gA, shape=(TILE_M, TILE_M), storage="shared")
    d = wp.tile_load(gD, shape=TILE_M, storage="shared")
    y = wp.tile_load(gy, shape=TILE_M, storage="shared")
    # Ensure tile_diag_add() and tile_cholesky_solve() work with transposed matrices
    a_t = wp.tile_transpose(a)
    # Compute L st LL^T = A^T + diag(D)
    b = wp.tile_diag_add(a_t, d)
    l = wp.tile_cholesky(b)
    # Solve for y in LL^T x = y
    x = wp.tile_cholesky_solve(l, y)
    # Store L & y
    wp.tile_store(gL, l)
    wp.tile_store(gx, x)


def test_tile_cholesky_lower_factor_and_solve(test, device):
    """Compute a lower Cholesky factor and solve a vector right-hand side."""
    A_h = np.ones((TILE_M, TILE_M), dtype=np.float64)
    D_h = 8.0 * np.ones(TILE_M, dtype=np.float64)
    L_h = np.zeros_like(A_h)
    Y_h = np.arange(TILE_M, dtype=np.float64)
    X_h = np.zeros_like(Y_h)

    A_np = A_h.T + np.diag(D_h)
    L_np = np.linalg.cholesky(A_np)
    X_np = np.linalg.solve(A_np, Y_h)

    A_wp = wp.array(A_h, requires_grad=True, dtype=wp.float64, device=device)
    D_wp = wp.array(D_h, requires_grad=True, dtype=wp.float64, device=device)
    L_wp = wp.array(L_h, requires_grad=True, dtype=wp.float64, device=device)
    Y_wp = wp.array(Y_h, requires_grad=True, dtype=wp.float64, device=device)
    X_wp = wp.array(X_h, requires_grad=True, dtype=wp.float64, device=device)

    wp.launch_tiled(
        tile_math_cholesky, dim=[1, 1], inputs=[A_wp, D_wp, L_wp, Y_wp, X_wp], block_dim=TILE_DIM, device=device
    )

    np.testing.assert_allclose(X_wp.numpy(), X_np)
    np.testing.assert_allclose(L_wp.numpy(), L_np)

    # TODO: implement and test backward pass


@wp.kernel(module="test_cholesky_fwd")
def tile_math_cholesky_inplace(
    gA: wp.array2d[wp.float64],
    gy: wp.array1d[wp.float64],
):
    # Load A & y
    a = wp.tile_load(gA, shape=(TILE_M, TILE_M), storage="shared")
    y = wp.tile_load(gy, shape=TILE_M, storage="shared")
    # Compute L st LL^T = A inplace
    wp.tile_cholesky_inplace(a)
    # Solve for y in LL^T x = y inplace
    wp.tile_cholesky_solve_inplace(a, y)
    # Store L & y
    wp.tile_store(gA, a)
    wp.tile_store(gy, y)


def test_tile_cholesky_lower_factor_and_solve_inplace(test, device):
    """Compute a lower Cholesky factor and solve a vector right-hand side in place."""
    rng = np.random.default_rng(42)
    L_h = np.tril(rng.random((TILE_M, TILE_M)))  # Lower triangular matrix
    A_h = L_h @ L_h.T
    Y_h = np.arange(TILE_M, dtype=np.float64)

    Y_sol_np = np.linalg.solve(A_h, Y_h)

    A_wp = wp.array(A_h, dtype=wp.float64, device=device)
    Y_wp = wp.array(Y_h, dtype=wp.float64, device=device)

    wp.launch_tiled(tile_math_cholesky_inplace, dim=[1, 1], inputs=[A_wp, Y_wp], block_dim=TILE_DIM, device=device)

    np.testing.assert_allclose(Y_wp.numpy(), Y_sol_np)
    np.testing.assert_allclose(A_wp.numpy(), L_h)


@wp.kernel(module="test_cholesky_fwd")
def tile_math_cholesky_multiple_rhs(
    gA: wp.array2d[wp.float64],
    gD: wp.array1d[wp.float64],
    gL: wp.array2d[wp.float64],
    gy: wp.array2d[wp.float64],
    gx: wp.array2d[wp.float64],
    gz: wp.array2d[wp.float64],
):
    # Load A, D & y
    a = wp.tile_load(gA, shape=(TILE_M, TILE_M), storage="shared")
    d = wp.tile_load(gD, shape=TILE_M, storage="shared")
    y = wp.tile_load(gy, shape=(TILE_M, TILE_M), storage="shared")
    # Ensure tile_diag_add() and tile_cholesky_solve() work with transposed matrices
    a_t = wp.tile_transpose(a)
    # Compute L st LL^T = A.T + diag(D)
    b = wp.tile_diag_add(a_t, d)
    l = wp.tile_cholesky(b)
    # Solve for y in LL^T x = y.T
    y_t = wp.tile_transpose(y)
    x = wp.tile_cholesky_solve(l, y_t)
    # Ensure matmul receives correct layout information
    z = wp.tile_matmul(x, x)
    # Store L & y
    wp.tile_store(gL, l)
    wp.tile_store(gx, x)
    wp.tile_store(gz, z)


def test_tile_cholesky_lower_factor_and_solve_multiple_rhs(test, device):
    """Compute a lower Cholesky factor and solve multiple right-hand sides."""
    A_h = np.ones((TILE_M, TILE_M), dtype=np.float64)
    D_h = 8.0 * np.ones(TILE_M, dtype=np.float64)
    L_h = np.zeros_like(A_h)
    Y_h = np.arange(TILE_M * TILE_M, dtype=np.float64).reshape((TILE_M, TILE_M))
    X_h = np.zeros_like(Y_h)
    Z_h = np.zeros_like(Y_h)

    A_np = A_h.T + np.diag(D_h)
    L_np = np.linalg.cholesky(A_np)
    X_np = np.linalg.solve(A_np, Y_h.T)
    Z_np = X_np @ X_np

    A_wp = wp.array(A_h, requires_grad=True, dtype=wp.float64, device=device)
    D_wp = wp.array(D_h, requires_grad=True, dtype=wp.float64, device=device)
    L_wp = wp.array(L_h, requires_grad=True, dtype=wp.float64, device=device)
    Y_wp = wp.array(Y_h, requires_grad=True, dtype=wp.float64, device=device)
    X_wp = wp.array(X_h, requires_grad=True, dtype=wp.float64, device=device)
    Z_wp = wp.array(Z_h, requires_grad=True, dtype=wp.float64, device=device)

    wp.launch_tiled(
        tile_math_cholesky_multiple_rhs,
        dim=[1, 1],
        inputs=[A_wp, D_wp, L_wp, Y_wp, X_wp, Z_wp],
        block_dim=TILE_DIM,
        device=device,
    )

    np.testing.assert_allclose(L_wp.numpy(), L_np)
    np.testing.assert_allclose(X_wp.numpy(), X_np)
    np.testing.assert_allclose(Z_wp.numpy(), Z_np)

    # TODO: implement and test backward pass


@wp.kernel(module="test_cholesky_fwd")
def tile_math_cholesky_multiple_rhs_inplace(
    gA: wp.array2d[wp.float64],
    gy: wp.array2d[wp.float64],
    gz: wp.array2d[wp.float64],
):
    # Load A & y
    a = wp.tile_load(gA, shape=(TILE_M, TILE_M), storage="shared")
    y = wp.tile_load(gy, shape=(TILE_M, TILE_M), storage="shared")
    # Compute L st LL^T = A inplace
    wp.tile_cholesky_inplace(a)
    # Solve for y in LL^T x = y.T inplace
    y_t = wp.tile_transpose(y)
    wp.tile_cholesky_solve_inplace(a, y_t)
    y = wp.tile_transpose(y_t)
    # Ensure matmul receives correct layout information
    z = wp.tile_matmul(y, y)
    # Store L & y
    wp.tile_store(gA, a)
    wp.tile_store(gy, y)
    wp.tile_store(gz, z)


def test_tile_cholesky_lower_factor_and_solve_multiple_rhs_inplace(test, device):
    """Compute a lower Cholesky factor and solve multiple right-hand sides in place."""
    rng = np.random.default_rng(42)
    L_h = np.tril(rng.random((TILE_M, TILE_M)))  # Lower triangular matrix
    A_h = L_h @ L_h.T
    Y_h = np.arange(TILE_M * TILE_M, dtype=np.float64).reshape((TILE_M, TILE_M))
    Z_h = np.zeros_like(Y_h)

    Y_sol_np = np.linalg.solve(A_h, Y_h.T).T
    Z_np = Y_sol_np @ Y_sol_np

    A_wp = wp.array(A_h, dtype=wp.float64, device=device)
    Y_wp = wp.array(Y_h, dtype=wp.float64, device=device)
    Z_wp = wp.array(Z_h, dtype=wp.float64, device=device)

    wp.launch_tiled(
        tile_math_cholesky_multiple_rhs_inplace,
        dim=[1, 1],
        inputs=[A_wp, Y_wp, Z_wp],
        block_dim=TILE_DIM,
        device=device,
    )

    np.testing.assert_allclose(A_wp.numpy(), L_h)
    np.testing.assert_allclose(Y_wp.numpy(), Y_sol_np)
    np.testing.assert_allclose(Z_wp.numpy(), Z_np)


@wp.kernel()
def tile_cholesky_lower_backward_kernel(
    gA: wp.array2d[Any],
    gL: wp.array2d[Any],
):
    a = wp.tile_load(gA, shape=(TILE_M, TILE_M), storage="shared")
    l = wp.tile_cholesky(a)
    wp.tile_store(gL, l)


wp.overload(tile_cholesky_lower_backward_kernel, {"gA": wp.array2d[wp.float32], "gL": wp.array2d[wp.float32]})
wp.overload(tile_cholesky_lower_backward_kernel, {"gA": wp.array2d[wp.float64], "gL": wp.array2d[wp.float64]})


def cholesky_adjoint_numpy_lower(L, adj_L):
    """Analytic adjoint of Cholesky factorization (lower-triangle parameterization)."""
    P = L.T @ adj_L
    P = np.tril(P)
    P[np.diag_indices_from(P)] *= 0.5
    S = P + P.T
    X = np.linalg.solve(L.T, S)
    B = np.linalg.solve(L.T, X.T)
    grad_A = np.tril(B)
    grad_A[np.diag_indices_from(grad_A)] *= 0.5
    return grad_A


def _cholesky_lower_backward(A_np, adj_L, device, wp_dtype=wp.float64):
    """Run tile Cholesky forward+backward, return (L_wp, grad_A) as NumPy arrays."""
    np_dtype = wp.dtype_to_numpy(wp_dtype)
    A_wp = wp.array(A_np.astype(np_dtype), dtype=wp_dtype, requires_grad=True, device=device)
    L_wp = wp.zeros((TILE_M, TILE_M), dtype=wp_dtype, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_cholesky_lower_backward_kernel,
            dim=[1, 1],
            inputs=[A_wp, L_wp],
            block_dim=TILE_DIM,
            device=device,
        )

    tape.backward(grads={L_wp: wp.array(adj_L.astype(np_dtype), dtype=wp_dtype, device=device)})
    return L_wp.numpy(), A_wp.grad.numpy()


def test_tile_cholesky_lower_backward(dtype):
    def test(test, device):
        np_dtype = wp.dtype_to_numpy(dtype)
        fwd_atol = 1e-10 if dtype == wp.float64 else 1e-4
        bwd_atol = 1e-8 if dtype == wp.float64 else 1e-3
        zero_atol = 0.0 if dtype == wp.float64 else 1e-6

        def check(A_np, adj_L):
            L_np = np.linalg.cholesky(A_np)
            L_wp, grad_A = _cholesky_lower_backward(A_np, adj_L, device, wp_dtype=dtype)
            grad_A_ref = cholesky_adjoint_numpy_lower(L_np, adj_L)
            np.testing.assert_allclose(L_wp, L_np.astype(np_dtype), atol=fwd_atol)
            np.testing.assert_allclose(np.tril(grad_A), grad_A_ref.astype(np_dtype), atol=bwd_atol)
            np.testing.assert_allclose(np.triu(grad_A, k=1), 0.0, atol=zero_atol)

        # Random SPD
        rng = np.random.default_rng(42)
        M = rng.random((TILE_M, TILE_M))
        check(M.T @ M + 8.0 * np.eye(TILE_M), rng.standard_normal((TILE_M, TILE_M)))

        # Identity - closed-form adjoint (independent of Murray formula):
        # Linearize A = LL^T at L = I: dA = dL + dL^T, so adj_A = tril(adj_L) with halved diagonal.
        rng = np.random.default_rng(100)
        adj_L = rng.standard_normal((TILE_M, TILE_M))
        L_wp, grad_A = _cholesky_lower_backward(np.eye(TILE_M), adj_L, device, wp_dtype=dtype)
        grad_A_ref = np.tril(adj_L)
        grad_A_ref[np.diag_indices_from(grad_A_ref)] *= 0.5
        np.testing.assert_allclose(L_wp, np.eye(TILE_M, dtype=np_dtype), atol=fwd_atol)
        np.testing.assert_allclose(np.tril(grad_A), grad_A_ref.astype(np_dtype), atol=bwd_atol)
        np.testing.assert_allclose(np.triu(grad_A, k=1), 0.0, atol=zero_atol)

        # Diagonal - closed-form adjoint (independent of Murray formula):
        # Linearize A = LL^T at L = diag(s): dA = diag(s) dL^T + dL diag(s),
        # so adj_A = tril(adj_L) / s[newaxis,:] with halved diagonal.
        rng = np.random.default_rng(200)
        d = rng.uniform(1.0, 10.0, TILE_M)
        adj_L = rng.standard_normal((TILE_M, TILE_M))
        s = np.sqrt(d)
        L_wp, grad_A = _cholesky_lower_backward(np.diag(d), adj_L, device, wp_dtype=dtype)
        grad_A_ref = np.tril(adj_L) / s[np.newaxis, :]
        grad_A_ref[np.diag_indices_from(grad_A_ref)] *= 0.5
        np.testing.assert_allclose(L_wp, np.diag(s).astype(np_dtype), atol=fwd_atol)
        np.testing.assert_allclose(np.tril(grad_A), grad_A_ref.astype(np_dtype), atol=bwd_atol)
        np.testing.assert_allclose(np.triu(grad_A, k=1), 0.0, atol=zero_atol)

    return test


@wp.kernel()
def tile_cholesky_upper_backward_kernel(
    gA: wp.array2d[Any],
    gU: wp.array2d[Any],
):
    a = wp.tile_load(gA, shape=(TILE_M, TILE_M), storage="shared")
    u = wp.tile_cholesky(a, fill_mode="upper")
    wp.tile_store(gU, u)


wp.overload(tile_cholesky_upper_backward_kernel, {"gA": wp.array2d[wp.float32], "gU": wp.array2d[wp.float32]})
wp.overload(tile_cholesky_upper_backward_kernel, {"gA": wp.array2d[wp.float64], "gU": wp.array2d[wp.float64]})


def cholesky_adjoint_numpy_upper(U, adj_U):
    """Analytic adjoint of Cholesky factorization (upper-triangle parameterization)."""
    P = adj_U @ U.T
    P = np.triu(P)
    P[np.diag_indices_from(P)] *= 0.5
    S = P + P.T
    X = np.linalg.solve(U, S)
    B = np.linalg.solve(U, X.T)
    grad_A = np.triu(B)
    grad_A[np.diag_indices_from(grad_A)] *= 0.5
    return grad_A


def _cholesky_upper_backward(A_np, adj_U, device, wp_dtype=wp.float64):
    """Run tile Cholesky(upper) forward+backward, return (U_wp, grad_A) as NumPy arrays."""
    np_dtype = wp.dtype_to_numpy(wp_dtype)
    A_wp = wp.array(A_np.astype(np_dtype), dtype=wp_dtype, requires_grad=True, device=device)
    U_wp = wp.zeros((TILE_M, TILE_M), dtype=wp_dtype, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_cholesky_upper_backward_kernel,
            dim=[1, 1],
            inputs=[A_wp, U_wp],
            block_dim=TILE_DIM,
            device=device,
        )

    tape.backward(grads={U_wp: wp.array(adj_U.astype(np_dtype), dtype=wp_dtype, device=device)})
    return U_wp.numpy(), A_wp.grad.numpy()


def test_tile_cholesky_upper_backward(dtype):
    def test(test, device):
        np_dtype = wp.dtype_to_numpy(dtype)
        fwd_atol = 1e-10 if dtype == wp.float64 else 1e-4
        bwd_atol = 1e-8 if dtype == wp.float64 else 1e-3
        zero_atol = 0.0 if dtype == wp.float64 else 1e-6

        def check(A_np, adj_U):
            U_np = np.linalg.cholesky(A_np).T
            U_wp, grad_A = _cholesky_upper_backward(A_np, adj_U, device, wp_dtype=dtype)
            grad_A_ref = cholesky_adjoint_numpy_upper(U_np, adj_U)
            np.testing.assert_allclose(U_wp, U_np.astype(np_dtype), atol=fwd_atol)
            np.testing.assert_allclose(np.triu(grad_A), grad_A_ref.astype(np_dtype), atol=bwd_atol)
            np.testing.assert_allclose(np.tril(grad_A, k=-1), 0.0, atol=zero_atol)

        # Random SPD
        rng = np.random.default_rng(42)
        M = rng.random((TILE_M, TILE_M))
        check(M.T @ M + 8.0 * np.eye(TILE_M), rng.standard_normal((TILE_M, TILE_M)))

        # Identity - closed-form adjoint (independent of Murray formula):
        # Linearize A = U^TU at U = I: dA = dU^T + dU, so adj_A = triu(adj_U) with halved diagonal.
        rng = np.random.default_rng(100)
        adj_U = rng.standard_normal((TILE_M, TILE_M))
        U_wp, grad_A = _cholesky_upper_backward(np.eye(TILE_M), adj_U, device, wp_dtype=dtype)
        grad_A_ref = np.triu(adj_U)
        grad_A_ref[np.diag_indices_from(grad_A_ref)] *= 0.5
        np.testing.assert_allclose(U_wp, np.eye(TILE_M, dtype=np_dtype), atol=fwd_atol)
        np.testing.assert_allclose(np.triu(grad_A), grad_A_ref.astype(np_dtype), atol=bwd_atol)
        np.testing.assert_allclose(np.tril(grad_A, k=-1), 0.0, atol=zero_atol)

        # Diagonal - closed-form adjoint (independent of Murray formula):
        # Linearize A = U^TU at U = diag(s): dA = dU^T diag(s) + diag(s) dU,
        # so adj_A = triu(adj_U) / s[:,newaxis] with halved diagonal.
        rng = np.random.default_rng(200)
        d = rng.uniform(1.0, 10.0, TILE_M)
        adj_U = rng.standard_normal((TILE_M, TILE_M))
        s = np.sqrt(d)
        U_wp, grad_A = _cholesky_upper_backward(np.diag(d), adj_U, device, wp_dtype=dtype)
        grad_A_ref = np.triu(adj_U) / s[:, np.newaxis]
        grad_A_ref[np.diag_indices_from(grad_A_ref)] *= 0.5
        np.testing.assert_allclose(U_wp, np.diag(s).astype(np_dtype), atol=fwd_atol)
        np.testing.assert_allclose(np.triu(grad_A), grad_A_ref.astype(np_dtype), atol=bwd_atol)
        np.testing.assert_allclose(np.tril(grad_A, k=-1), 0.0, atol=zero_atol)

    return test


# tests a complex composition of most libmathdx calls
def test_tile_cholesky_block_cholesky(test, device):
    BLOCK_SIZE = wp.constant(TILE_M // 2)

    # enable_backward belongs in module_options, not as a kernel kwarg: the LTO
    # dispatch for tile_matmul/tile_cholesky/tile_lower_solve reads the module
    # builder's options, which kernel kwargs are not merged into. As a kwarg it
    # still suppresses the C++ backward, but libmathdx builds the unused
    # backward GEMM/TRSM LTOs anyway.
    @wp.kernel(module="unique", module_options={"enable_backward": False})
    def block_cholesky_kernel(
        A: wp.array2d[float],
        L: wp.array2d[float],
    ):
        """Compute the Cholesky factorization of a symmetric positive definite matrix ``A`` in blocks.

        It returns a lower-triangular matrix ``L`` such that ``A = L L^T``.
        """

        # Process the matrix in blocks along its leading dimension.
        for k in range(0, TILE_M, BLOCK_SIZE):
            end = k + BLOCK_SIZE

            # Load current diagonal block A[k:end, k:end]
            # and update with contributions from previously computed blocks.
            A_kk_tile = wp.tile_load(A, shape=(BLOCK_SIZE, BLOCK_SIZE), offset=(k, k), storage="shared")

            for j in range(0, k, BLOCK_SIZE):
                L_block = wp.tile_load(L, shape=(BLOCK_SIZE, BLOCK_SIZE), offset=(k, j))
                L_block_T = wp.tile_transpose(L_block)
                wp.tile_matmul(L_block, L_block_T, A_kk_tile, alpha=-1.0)

            # Compute the Cholesky factorization for the block
            # print(A_kk_tile)
            L_kk_tile = wp.tile_cholesky(A_kk_tile)
            wp.tile_store(L, L_kk_tile, offset=(k, k))

            # Process the blocks below the current block
            for i in range(end, TILE_M, BLOCK_SIZE):
                A_ik_tile = wp.tile_load(A, shape=(BLOCK_SIZE, BLOCK_SIZE), offset=(i, k), storage="shared")

                for j in range(0, k, BLOCK_SIZE):
                    L_tile = wp.tile_load(L, shape=(BLOCK_SIZE, BLOCK_SIZE), offset=(i, j))
                    L_2_tile = wp.tile_load(L, shape=(BLOCK_SIZE, BLOCK_SIZE), offset=(k, j))
                    L_T_tile = wp.tile_transpose(L_2_tile)
                    wp.tile_matmul(L_tile, L_T_tile, A_ik_tile, alpha=-1.0)

                A_ik_T_tile = wp.tile_transpose(A_ik_tile)
                sol_T_tile = wp.tile_lower_solve(L_kk_tile, A_ik_T_tile)
                sol_tile = wp.tile_transpose(sol_T_tile)

                wp.tile_store(L, sol_tile, offset=(i, k))

    @wp.kernel(module="unique", module_options={"enable_backward": False})
    def block_cholesky_solve_kernel(
        L: wp.array2d[float],
        b: wp.array2d[float],
        scratch: wp.array2d[float],
        x: wp.array2d[float],
    ):
        """Solve a Cholesky-factored system with blocked substitution.

        Apply forward and backward substitution to solve ``A x = b`` from the
        factorization ``A = L L^T``.
        """

        # Forward substitution: solve L y = b
        for i in range(0, TILE_M, BLOCK_SIZE):
            i_end = i + BLOCK_SIZE
            rhs_tile = wp.tile_load(b, shape=(BLOCK_SIZE, 1), offset=(i, 0))
            for j in range(0, i, BLOCK_SIZE):
                L_block = wp.tile_load(L, shape=(BLOCK_SIZE, BLOCK_SIZE), offset=(i, j))
                y_block = wp.tile_load(scratch, shape=(BLOCK_SIZE, 1), offset=(j, 0))
                wp.tile_matmul(L_block, y_block, rhs_tile, alpha=-1.0)
            L_tile = wp.tile_load(L, shape=(BLOCK_SIZE, BLOCK_SIZE), offset=(i, i))
            y_tile = wp.tile_lower_solve(L_tile, rhs_tile)
            wp.tile_store(scratch, y_tile, offset=(i, 0))

        # Backward substitution: solve L^T x = y
        for i in range(TILE_M - BLOCK_SIZE, -1, -BLOCK_SIZE):
            i_start = i
            i_end = i_start + BLOCK_SIZE
            rhs_tile = wp.tile_load(scratch, shape=(BLOCK_SIZE, 1), offset=(i_start, 0))
            for j in range(i_end, TILE_M, BLOCK_SIZE):
                L_tile = wp.tile_load(L, shape=(BLOCK_SIZE, BLOCK_SIZE), offset=(j, i_start))
                L_T_tile = wp.tile_transpose(L_tile)
                x_tile = wp.tile_load(x, shape=(BLOCK_SIZE, 1), offset=(j, 0))
                wp.tile_matmul(L_T_tile, x_tile, rhs_tile, alpha=-1.0)
            L_tile = wp.tile_load(L, shape=(BLOCK_SIZE, BLOCK_SIZE), offset=(i_start, i_start))
            x_tile = wp.tile_upper_solve(wp.tile_transpose(L_tile), rhs_tile)
            wp.tile_store(x, x_tile, offset=(i_start, 0))

    # check block cholesky decomposition

    rng = np.random.default_rng(42)

    M = np.array(rng.random((TILE_M, TILE_M)), dtype=float)

    A_np = M.T @ M + np.eye(TILE_M, TILE_M)
    L_np = np.linalg.cholesky(A_np)

    A_wp = wp.array(A_np, dtype=float, device=device)
    L_wp = wp.zeros_like(A_wp)

    wp.launch_tiled(block_cholesky_kernel, dim=1, inputs=[A_wp], outputs=[L_wp], block_dim=TILE_DIM, device=device)

    # check block cholesky solve

    assert_np_equal(L_wp.numpy(), L_np, tol=1e-6)

    b_np = np.array(rng.random((TILE_M, 1)), dtype=float)
    b_wp = wp.array(b_np, dtype=float, device=device)

    scratch = wp.zeros_like(b_wp)

    x_np = np.linalg.solve(L_np.T, np.linalg.solve(L_np, b_np))
    x_wp = wp.zeros_like(b_wp)

    wp.launch_tiled(
        block_cholesky_solve_kernel,
        dim=1,
        inputs=[L_wp, b_wp, scratch],
        outputs=[x_wp],
        block_dim=TILE_DIM,
        device=device,
    )

    assert_np_equal(x_wp.numpy(), x_np, tol=1e-6)


@wp.kernel(module="test_cholesky_fwd")
def tile_math_cholesky_upper(
    gA: wp.array2d[wp.float64],
    gD: wp.array1d[wp.float64],
    gU: wp.array2d[wp.float64],
    gy: wp.array1d[wp.float64],
    gx: wp.array1d[wp.float64],
):
    # Load A, D & y
    a = wp.tile_load(gA, shape=(TILE_M, TILE_M), storage="shared")
    d = wp.tile_load(gD, shape=TILE_M, storage="shared")
    y = wp.tile_load(gy, shape=TILE_M, storage="shared")
    # Ensure tile_diag_add() works with transposed matrices
    a_t = wp.tile_transpose(a)
    # Compute U st U^T U = A^T + diag(D)
    b = wp.tile_diag_add(a_t, d)
    u = wp.tile_cholesky(b, fill_mode="upper")
    # Solve for x in U^T U x = y
    x = wp.tile_cholesky_solve(u, y, fill_mode="upper")
    # Store U & x
    wp.tile_store(gU, u)
    wp.tile_store(gx, x)


def _test_tile_cholesky_upper_out_of_place(test, device, kernel, multiple_rhs):
    """Shared test logic for tile_cholesky(fill_mode="upper") (out-of-place) with vector or matrix RHS."""
    A_h = np.ones((TILE_M, TILE_M), dtype=np.float64)
    D_h = 8.0 * np.ones(TILE_M, dtype=np.float64)

    A_np = A_h.T + np.diag(D_h)
    U_np = np.linalg.cholesky(A_np).T

    if multiple_rhs:
        Y_h = np.arange(TILE_M * TILE_M, dtype=np.float64).reshape((TILE_M, TILE_M))
        X_np = np.linalg.solve(A_np, Y_h.T)
        Z_np = X_np @ X_np

        A_wp = wp.array(A_h, dtype=wp.float64, device=device)
        D_wp = wp.array(D_h, dtype=wp.float64, device=device)
        U_wp = wp.zeros((TILE_M, TILE_M), dtype=wp.float64, device=device)
        Y_wp = wp.array(Y_h, dtype=wp.float64, device=device)
        X_wp = wp.zeros_like(Y_wp)
        Z_wp = wp.zeros_like(Y_wp)

        wp.launch_tiled(
            kernel, dim=[1, 1], inputs=[A_wp, D_wp, U_wp, Y_wp, X_wp, Z_wp], block_dim=TILE_DIM, device=device
        )

        np.testing.assert_allclose(U_wp.numpy(), U_np)
        np.testing.assert_allclose(X_wp.numpy(), X_np)
        np.testing.assert_allclose(Z_wp.numpy(), Z_np)
    else:
        Y_h = np.arange(TILE_M, dtype=np.float64)
        X_np = np.linalg.solve(A_np, Y_h)

        A_wp = wp.array(A_h, dtype=wp.float64, device=device)
        D_wp = wp.array(D_h, dtype=wp.float64, device=device)
        U_wp = wp.zeros((TILE_M, TILE_M), dtype=wp.float64, device=device)
        Y_wp = wp.array(Y_h, dtype=wp.float64, device=device)
        X_wp = wp.zeros_like(Y_wp)

        wp.launch_tiled(kernel, dim=[1, 1], inputs=[A_wp, D_wp, U_wp, Y_wp, X_wp], block_dim=TILE_DIM, device=device)

        np.testing.assert_allclose(U_wp.numpy(), U_np)
        np.testing.assert_allclose(X_wp.numpy(), X_np)


def test_tile_cholesky_upper(test, device):
    _test_tile_cholesky_upper_out_of_place(test, device, tile_math_cholesky_upper, multiple_rhs=False)


@wp.kernel(module="test_cholesky_fwd")
def tile_math_cholesky_upper_inplace(
    gA: wp.array2d[wp.float64],
    gy: wp.array1d[wp.float64],
):
    # Load A & y
    a = wp.tile_load(gA, shape=(TILE_M, TILE_M), storage="shared")
    y = wp.tile_load(gy, shape=TILE_M, storage="shared")
    # Compute U st U^T U = A inplace
    wp.tile_cholesky_inplace(a, fill_mode="upper")
    # Solve for x in U^T U x = y inplace
    wp.tile_cholesky_solve_inplace(a, y, fill_mode="upper")
    # Store U & y
    wp.tile_store(gA, a)
    wp.tile_store(gy, y)


def _test_tile_cholesky_upper_inplace(test, device, kernel, multiple_rhs):
    """Shared test logic for tile_cholesky_inplace(fill_mode="upper") with vector or matrix RHS."""
    rng = np.random.default_rng(42)
    L_h = np.tril(rng.random((TILE_M, TILE_M)))
    U_h = L_h.T
    A_h = U_h.T @ U_h

    if multiple_rhs:
        Y_h = np.arange(TILE_M * TILE_M, dtype=np.float64).reshape((TILE_M, TILE_M))
        Y_sol_np = np.linalg.solve(A_h, Y_h.T)
        Z_np = Y_sol_np @ Y_sol_np

        A_wp = wp.array(A_h, dtype=wp.float64, device=device)
        Y_wp = wp.array(Y_h, dtype=wp.float64, device=device)
        Z_wp = wp.zeros_like(Y_wp)

        wp.launch_tiled(kernel, dim=[1, 1], inputs=[A_wp, Y_wp, Z_wp], block_dim=TILE_DIM, device=device)

        np.testing.assert_allclose(A_wp.numpy(), U_h)
        np.testing.assert_allclose(Y_wp.numpy(), Y_sol_np)
        np.testing.assert_allclose(Z_wp.numpy(), Z_np)
    else:
        Y_h = np.arange(TILE_M, dtype=np.float64)
        Y_sol_np = np.linalg.solve(A_h, Y_h)

        A_wp = wp.array(A_h, dtype=wp.float64, device=device)
        Y_wp = wp.array(Y_h, dtype=wp.float64, device=device)

        wp.launch_tiled(kernel, dim=[1, 1], inputs=[A_wp, Y_wp], block_dim=TILE_DIM, device=device)

        np.testing.assert_allclose(Y_wp.numpy(), Y_sol_np)
        np.testing.assert_allclose(A_wp.numpy(), U_h)


def test_tile_cholesky_upper_inplace(test, device):
    _test_tile_cholesky_upper_inplace(test, device, tile_math_cholesky_upper_inplace, multiple_rhs=False)


@wp.kernel(module="test_cholesky_fwd")
def tile_math_cholesky_upper_multiple_rhs(
    gA: wp.array2d[wp.float64],
    gD: wp.array1d[wp.float64],
    gU: wp.array2d[wp.float64],
    gy: wp.array2d[wp.float64],
    gx: wp.array2d[wp.float64],
    gz: wp.array2d[wp.float64],
):
    # Load A, D & y
    a = wp.tile_load(gA, shape=(TILE_M, TILE_M), storage="shared")
    d = wp.tile_load(gD, shape=TILE_M, storage="shared")
    y = wp.tile_load(gy, shape=(TILE_M, TILE_M), storage="shared")
    # Compute U st U^T U = A.T + diag(D)
    a_t = wp.tile_transpose(a)
    b = wp.tile_diag_add(a_t, d)
    u = wp.tile_cholesky(b, fill_mode="upper")
    # Solve for x in U^T U x = y.T
    y_t = wp.tile_transpose(y)
    x = wp.tile_cholesky_solve(u, y_t, fill_mode="upper")
    # Ensure matmul receives correct layout information
    z = wp.tile_matmul(x, x)
    # Store U, x & z
    wp.tile_store(gU, u)
    wp.tile_store(gx, x)
    wp.tile_store(gz, z)


def test_tile_cholesky_upper_multiple_rhs(test, device):
    _test_tile_cholesky_upper_out_of_place(test, device, tile_math_cholesky_upper_multiple_rhs, multiple_rhs=True)


@wp.kernel(module="test_cholesky_fwd")
def tile_math_cholesky_upper_multiple_rhs_inplace(
    gA: wp.array2d[wp.float64],
    gy: wp.array2d[wp.float64],
    gz: wp.array2d[wp.float64],
):
    # Load A & y
    a = wp.tile_load(gA, shape=(TILE_M, TILE_M), storage="shared")
    y = wp.tile_load(gy, shape=(TILE_M, TILE_M), storage="shared")
    # Compute U st U^T U = A inplace
    wp.tile_cholesky_inplace(a, fill_mode="upper")
    # Solve for x in U^T U x = y.T inplace
    y_t = wp.tile_transpose(y)
    wp.tile_cholesky_solve_inplace(a, y_t, fill_mode="upper")
    # Ensure matmul receives correct layout information
    z = wp.tile_matmul(y_t, y_t)
    # Store U, y & z
    wp.tile_store(gA, a)
    wp.tile_store(gy, y_t)
    wp.tile_store(gz, z)


def test_tile_cholesky_upper_multiple_rhs_inplace(test, device):
    _test_tile_cholesky_upper_inplace(test, device, tile_math_cholesky_upper_multiple_rhs_inplace, multiple_rhs=True)


@wp.kernel(module="test_cholesky_fwd")
def tile_math_cholesky_solve_upper(
    gU: wp.array2d[wp.float64],
    gy: wp.array1d[wp.float64],
    gx: wp.array1d[wp.float64],
):
    U = wp.tile_load(gU, shape=(TILE_M, TILE_M), storage="shared")
    y = wp.tile_load(gy, shape=TILE_M, storage="shared")
    x = wp.tile_cholesky_solve(U, y, fill_mode="upper")
    wp.tile_store(gx, x)


def test_tile_cholesky_solve_upper(test, device):
    A_h = np.ones((TILE_M, TILE_M), dtype=np.float64) + 8.0 * np.eye(TILE_M, dtype=np.float64)
    U_np = np.linalg.cholesky(A_h).T  # Upper triangular

    Y_h = np.arange(TILE_M, dtype=np.float64)
    X_np = np.linalg.solve(A_h, Y_h)

    U_wp = wp.array(U_np, dtype=wp.float64, device=device)
    Y_wp = wp.array(Y_h, dtype=wp.float64, device=device)
    X_wp = wp.zeros(TILE_M, dtype=wp.float64, device=device)

    wp.launch_tiled(
        tile_math_cholesky_solve_upper, dim=[1, 1], inputs=[U_wp, Y_wp, X_wp], block_dim=TILE_DIM, device=device
    )

    np.testing.assert_allclose(X_wp.numpy(), X_np)


@wp.kernel(module="test_cholesky_fwd")
def tile_math_cholesky_solve_upper_inplace(
    gU: wp.array2d[wp.float64],
    gy: wp.array1d[wp.float64],
):
    U = wp.tile_load(gU, shape=(TILE_M, TILE_M), storage="shared")
    y = wp.tile_load(gy, shape=TILE_M, storage="shared")
    wp.tile_cholesky_solve_inplace(U, y, fill_mode="upper")
    wp.tile_store(gy, y)


def test_tile_cholesky_solve_upper_inplace(test, device):
    A_h = np.ones((TILE_M, TILE_M), dtype=np.float64) + 8.0 * np.eye(TILE_M, dtype=np.float64)
    U_np = np.linalg.cholesky(A_h).T

    Y_h = np.arange(TILE_M, dtype=np.float64)
    X_np = np.linalg.solve(A_h, Y_h)

    U_wp = wp.array(U_np, dtype=wp.float64, device=device)
    Y_wp = wp.array(Y_h, dtype=wp.float64, device=device)

    wp.launch_tiled(
        tile_math_cholesky_solve_upper_inplace, dim=[1, 1], inputs=[U_wp, Y_wp], block_dim=TILE_DIM, device=device
    )

    np.testing.assert_allclose(Y_wp.numpy(), X_np)


@wp.kernel(module="test_cholesky_fwd")
def tile_math_cholesky_solve_upper_multiple_rhs(
    gU: wp.array2d[wp.float64],
    gY: wp.array2d[wp.float64],
    gX: wp.array2d[wp.float64],
):
    U = wp.tile_load(gU, shape=(TILE_M, TILE_M), storage="shared")
    Y = wp.tile_load(gY, shape=(TILE_M, TILE_M), storage="shared")
    X = wp.tile_cholesky_solve(U, Y, fill_mode="upper")
    wp.tile_store(gX, X)


def test_tile_cholesky_solve_upper_multiple_rhs(test, device):
    A_h = np.ones((TILE_M, TILE_M), dtype=np.float64) + 8.0 * np.eye(TILE_M, dtype=np.float64)
    U_np = np.linalg.cholesky(A_h).T

    Y_h = np.arange(TILE_M * TILE_M, dtype=np.float64).reshape((TILE_M, TILE_M))
    X_np = np.linalg.solve(A_h, Y_h)

    U_wp = wp.array(U_np, dtype=wp.float64, device=device)
    Y_wp = wp.array(Y_h, dtype=wp.float64, device=device)
    X_wp = wp.zeros((TILE_M, TILE_M), dtype=wp.float64, device=device)

    wp.launch_tiled(
        tile_math_cholesky_solve_upper_multiple_rhs,
        dim=[1, 1],
        inputs=[U_wp, Y_wp, X_wp],
        block_dim=TILE_DIM,
        device=device,
    )

    np.testing.assert_allclose(X_wp.numpy(), X_np)


@wp.kernel(module="test_cholesky_fwd")
def tile_math_cholesky_solve_upper_multiple_rhs_inplace(
    gU: wp.array2d[wp.float64],
    gY: wp.array2d[wp.float64],
):
    U = wp.tile_load(gU, shape=(TILE_M, TILE_M), storage="shared")
    Y = wp.tile_load(gY, shape=(TILE_M, TILE_M), storage="shared")
    wp.tile_cholesky_solve_inplace(U, Y, fill_mode="upper")
    wp.tile_store(gY, Y)


def test_tile_cholesky_solve_upper_multiple_rhs_inplace(test, device):
    A_h = np.ones((TILE_M, TILE_M), dtype=np.float64) + 8.0 * np.eye(TILE_M, dtype=np.float64)
    U_np = np.linalg.cholesky(A_h).T

    Y_h = np.arange(TILE_M * TILE_M, dtype=np.float64).reshape((TILE_M, TILE_M))
    X_np = np.linalg.solve(A_h, Y_h)

    U_wp = wp.array(U_np, dtype=wp.float64, device=device)
    Y_wp = wp.array(Y_h, dtype=wp.float64, device=device)

    wp.launch_tiled(
        tile_math_cholesky_solve_upper_multiple_rhs_inplace,
        dim=[1, 1],
        inputs=[U_wp, Y_wp],
        block_dim=TILE_DIM,
        device=device,
    )

    np.testing.assert_allclose(Y_wp.numpy(), X_np)


all_devices = get_test_devices()
cuda_devices = get_cuda_test_devices()


def test_tile_solver_strided_views(test, device):
    """Factor and solve offset views, retaining sentinels outside inplace operands."""
    rng = np.random.default_rng(122)
    m = rng.standard_normal((8, 8))
    matrix = m @ m.T + 8.0 * np.eye(8)
    rhs = rng.standard_normal((8, 3))
    factor = np.linalg.cholesky(matrix)
    for transposed in (False, True):
        for inplace in (False, True):
            for operation in ("cholesky", "cholesky_upper", "lower", "upper"):
                with test.subTest(transposed=transposed, inplace=inplace, operation=operation):

                    @wp.kernel(module="unique", module_options={"enable_backward": False})
                    def solve(a: wp.array2d[wp.float64], b: wp.array2d[wp.float64], out: wp.array2d[wp.float64]):
                        a_parent = wp.tile_load(a, shape=(16, 16), storage="shared")
                        b_parent = wp.tile_load(b, shape=(16, 12), storage="shared")
                        av = wp.tile_view(a_parent, offset=(2, 3), shape=(8, 8))
                        if wp.static(transposed):  # noqa: B023 - compiled before advancing the loop
                            av = wp.tile_transpose(av)
                        if wp.static(transposed):  # noqa: B023
                            bt = wp.tile_transpose(b_parent)
                            bv = wp.tile_view(bt, offset=(2, 3), shape=(8, 3))
                        else:
                            bv = wp.tile_view(b_parent, offset=(3, 2), shape=(8, 3))
                        if wp.static(operation.startswith("cholesky")):  # noqa: B023
                            if wp.static(inplace):  # noqa: B023 - compiled before advancing the loop
                                if wp.static(operation == "cholesky_upper"):  # noqa: B023
                                    wp.tile_cholesky_inplace(av, fill_mode="upper")
                                    wp.tile_cholesky_solve_inplace(av, bv, fill_mode="upper")
                                else:
                                    wp.tile_cholesky_inplace(av)
                                    wp.tile_cholesky_solve_inplace(av, bv)
                                wp.tile_store(out, bv)
                            else:
                                if wp.static(operation == "cholesky_upper"):  # noqa: B023
                                    l = wp.tile_cholesky(av, fill_mode="upper")
                                    x = wp.tile_cholesky_solve(l, bv, fill_mode="upper")
                                else:
                                    l = wp.tile_cholesky(av)
                                    x = wp.tile_cholesky_solve(l, bv)
                                wp.tile_store(out, x)
                        elif wp.static(operation == "lower"):  # noqa: B023
                            if wp.static(inplace):  # noqa: B023 - compiled before advancing the loop
                                wp.tile_lower_solve_inplace(av, bv)
                                wp.tile_store(out, bv)
                            else:
                                x = wp.tile_lower_solve(av, bv)
                                wp.tile_store(out, x)
                        else:
                            if wp.static(inplace):  # noqa: B023 - compiled before advancing the loop
                                wp.tile_upper_solve_inplace(av, bv)
                                wp.tile_store(out, bv)
                            else:
                                x = wp.tile_upper_solve(av, bv)
                                wp.tile_store(out, x)
                        wp.tile_store(a, a_parent)
                        wp.tile_store(b, b_parent)

                    actual_a = (
                        matrix if operation.startswith("cholesky") else factor if operation == "lower" else factor.T
                    )
                    ah = np.full((16, 16), 1234.0)
                    ah[2:10, 3:11] = actual_a.T if transposed else actual_a
                    bh = np.full((16, 12), 2345.0)
                    rhs_region = (slice(3, 6), slice(2, 10)) if transposed else (slice(3, 11), slice(2, 5))
                    bh[rhs_region] = rhs.T if transposed else rhs
                    a = wp.array(ah, dtype=wp.float64, device=device)
                    b = wp.array(bh, dtype=wp.float64, device=device)
                    out = wp.empty((8, 3), dtype=wp.float64, device=device)
                    wp.launch_tiled(solve, dim=1, inputs=[a, b, out], block_dim=TILE_DIM, device=device)
                    expected = np.linalg.solve(actual_a, rhs)
                    np.testing.assert_allclose(out.numpy(), expected, atol=1e-10)
                    if inplace:
                        bh[rhs_region] = expected.T if transposed else expected
                        if operation.startswith("cholesky"):
                            actual_factor = factor.T if operation == "cholesky_upper" else factor
                            ah[2:10, 3:11] = actual_factor.T if transposed else actual_factor
                    np.testing.assert_allclose(a.numpy(), ah, atol=1e-10)
                    np.testing.assert_allclose(b.numpy(), bh, atol=1e-10)


def test_tile_solver_strided_vector(test, device):
    @wp.kernel(module="unique", module_options={"enable_backward": False})
    def solve(a: wp.array2d[wp.float64], b: wp.array2d[wp.float64], out: wp.array[wp.float64]):
        ap = wp.tile_load(a, shape=(16, 16), storage="shared")
        bp = wp.tile_load(b, shape=(16, 3), storage="shared")
        av = wp.tile_view(ap, offset=(1, 2), shape=(8, 8))
        # A column of a row-major parent is a noncontiguous vector.
        bv = bp[2:10, 1]
        wp.tile_cholesky_solve_inplace(av, bv)
        wp.tile_store(out, bv)
        wp.tile_store(b, bp)

    factor = np.eye(8) * 2.0 + np.tril(np.ones((8, 8)), -1) * 0.1
    ah = np.full((16, 16), 1234.0)
    ah[1:9, 2:10] = factor
    bh = np.full((16, 3), 2345.0)
    bh[2:10, 1] = np.arange(8)
    a = wp.array(ah, dtype=wp.float64, device=device)
    b = wp.array(bh, dtype=wp.float64, device=device)
    out = wp.empty(8, dtype=wp.float64, device=device)
    wp.launch_tiled(solve, dim=1, inputs=[a, b, out], block_dim=TILE_DIM, device=device)
    expected = np.linalg.solve(factor @ factor.T, np.arange(8))
    np.testing.assert_allclose(out.numpy(), expected, atol=1e-10)
    bh[2:10, 1] = expected
    np.testing.assert_allclose(b.numpy(), bh, atol=1e-10)


def test_tile_cholesky_padded_backward(test, device):
    for transposed in (False, True):
        for upper in (False, True):

            @wp.kernel(module="unique")
            def factor(a: wp.array2d[wp.float64], out: wp.array2d[wp.float64]):
                ap = wp.tile_load(a, shape=(16, 16), storage="shared")
                av = wp.tile_view(ap, offset=(2, 3), shape=(8, 8))
                if wp.static(transposed):  # noqa: B023 - compiled before advancing the loop
                    av = wp.tile_transpose(av)
                if wp.static(upper):  # noqa: B023 - compiled before advancing the loop
                    l = wp.tile_cholesky(av, fill_mode="upper")
                else:
                    l = wp.tile_cholesky(av)
                wp.tile_store(out, l)

            rng = np.random.default_rng(123)
            m = rng.standard_normal((8, 8))
            matrix = m @ m.T + 8 * np.eye(8)
            adj = rng.standard_normal((8, 8))
            ah = np.zeros((16, 16))
            ah[2:10, 3:11] = matrix
            a = wp.array(ah, dtype=wp.float64, device=device, requires_grad=True)
            out = wp.empty((8, 8), dtype=wp.float64, device=device, requires_grad=True)
            with wp.Tape() as tape:
                wp.launch_tiled(factor, dim=1, inputs=[a, out], block_dim=TILE_DIM, device=device)
            tape.backward(grads={out: wp.array(adj, dtype=wp.float64, device=device)})
            l = np.linalg.cholesky(matrix)
            reference = cholesky_adjoint_numpy_upper(l.T, adj) if upper else cholesky_adjoint_numpy_lower(l, adj)
            expected = np.zeros_like(ah)
            expected[2:10, 3:11] = reference.T if transposed else reference
            np.testing.assert_allclose(a.grad.numpy(), expected, atol=1e-10)


def test_tile_lower_solve_padded_backward(test, device):
    for transposed in (False, True):

        @wp.kernel(module="unique")
        def solve(a: wp.array2d[wp.float64], b: wp.array2d[wp.float64], out: wp.array2d[wp.float64]):
            ap = wp.tile_load(a, shape=(16, 16), storage="shared")
            bp = wp.tile_load(b, shape=(16, 12), storage="shared")
            av = wp.tile_view(ap, offset=(2, 3), shape=(8, 8))
            if wp.static(transposed):  # noqa: B023 - compiled before advancing the loop
                av = wp.tile_transpose(av)
                bt = wp.tile_transpose(bp)
                bv = wp.tile_view(bt, offset=(2, 3), shape=(8, 3))
            else:
                bv = wp.tile_view(bp, offset=(3, 2), shape=(8, 3))
            x = wp.tile_lower_solve(av, bv)
            wp.tile_store(out, x)

        rng = np.random.default_rng(123)
        factor = np.tril(rng.standard_normal((8, 8))) + 8 * np.eye(8)
        rhs, adj = rng.standard_normal((2, 8, 3))
        ah, bh = np.zeros((16, 16)), np.zeros((16, 12))
        ah[2:10, 3:11] = factor.T if transposed else factor
        rhs_region = (slice(3, 6), slice(2, 10)) if transposed else (slice(3, 11), slice(2, 5))
        bh[rhs_region] = rhs.T if transposed else rhs
        a = wp.array(ah, dtype=wp.float64, device=device, requires_grad=True)
        b = wp.array(bh, dtype=wp.float64, device=device, requires_grad=True)
        out = wp.empty((8, 3), dtype=wp.float64, device=device, requires_grad=True)
        with wp.Tape() as tape:
            wp.launch_tiled(solve, dim=1, inputs=[a, b, out], block_dim=TILE_DIM, device=device)
        tape.backward(grads={out: wp.array(adj, dtype=wp.float64, device=device)})
        x = np.linalg.solve(factor, rhs)
        w = np.linalg.solve(factor.T, adj)
        factor_grad = -np.tril(w @ x.T)
        expected_a, expected_b = np.zeros_like(ah), np.zeros_like(bh)
        expected_a[2:10, 3:11] = factor_grad.T if transposed else factor_grad
        expected_b[rhs_region] = w.T if transposed else w
        np.testing.assert_allclose(out.numpy(), x, atol=1e-10)
        np.testing.assert_allclose(a.grad.numpy(), expected_a, atol=1e-10)
        np.testing.assert_allclose(b.grad.numpy(), expected_b, atol=1e-10)


class TestSolverOperandLayout(unittest.TestCase):
    def test_solver_results_own_dense_storage(self):
        a = tile(dtype=wp.float32, shape=(8, 8), strides=(16, 1), owner=False)
        b = tile(dtype=wp.float32, shape=(8, 3), strides=(12, 1), owner=False)
        d = tile(dtype=wp.float32, shape=(8,))
        for func, args, expected in (
            (builtins.tile_cholesky_generic_value_func, {"A": a}, (8, 1)),
            (builtins.tile_cholesky_solve_generic_value_func, {"L": a, "y": b}, (3, 1)),
            (builtins.tile_lower_solve_generic_value_func, {"L": a, "y": b}, (3, 1)),
            (builtins.tile_upper_solve_generic_value_func, {"U": a, "z": b}, (3, 1)),
            (builtins.tile_diag_add_value_func, {"a": a, "d": d}, (8, 1)),
        ):
            with self.subTest(operation=func.__name__):
                result = func(args, {})
                self.assertTrue(result.owner)
                self.assertEqual(tuple(result.strides), expected)
                self.assertEqual(result.size_in_bytes(), tile.round_up(result.size * 4))

    def test_solver_operand_layout_rejects_unrepresentable_strides(self):
        for shape, strides in (((8, 8), (16, 2)), ((8, 8), (0, 1)), ((8, 8), (4, 1)), ((8,), (0,))):
            with self.subTest(shape=shape, strides=strides):
                with self.assertRaisesRegex(ValueError, "unit inner stride"):
                    builtins._solver_operand_layout(tile(dtype=wp.float32, shape=shape, strides=strides))


@unittest.skipUnless(
    not wp._src.context.runtime.core.wp_is_mathdx_enabled()
    or (
        wp._src.context.runtime.core.wp_is_mathdx_enabled()
        and wp._src.context.runtime.core.wp_cuda_toolkit_version() >= 12060
    ),
    "MathDx is not enabled or is enabled but CUDA toolkit version is less than 12.6",
)
class TestTileCholesky(unittest.TestCase):
    pass


for name, func in (
    ("test_tile_solver_strided_views", test_tile_solver_strided_views),
    ("test_tile_solver_strided_vector", test_tile_solver_strided_vector),
    ("test_tile_cholesky_padded_backward", test_tile_cholesky_padded_backward),
    ("test_tile_lower_solve_padded_backward", test_tile_lower_solve_padded_backward),
):
    add_function_test(TestTileCholesky, name, func, devices=all_devices, check_output=False)


add_function_test(
    TestTileCholesky,
    "test_tile_cholesky_lower_factor_and_solve",
    test_tile_cholesky_lower_factor_and_solve,
    devices=all_devices,
    check_output=False,
)
add_function_test(
    TestTileCholesky,
    "test_tile_cholesky_lower_factor_and_solve_inplace",
    test_tile_cholesky_lower_factor_and_solve_inplace,
    devices=all_devices,
    check_output=False,
)

add_function_test(
    TestTileCholesky,
    "test_tile_cholesky_lower_factor_and_solve_multiple_rhs",
    test_tile_cholesky_lower_factor_and_solve_multiple_rhs,
    devices=all_devices,
    check_output=False,
)
add_function_test(
    TestTileCholesky,
    "test_tile_cholesky_lower_factor_and_solve_multiple_rhs_inplace",
    test_tile_cholesky_lower_factor_and_solve_multiple_rhs_inplace,
    devices=all_devices,
    check_output=False,
)


add_function_test(
    TestTileCholesky,
    "test_tile_cholesky_block_cholesky",
    test_tile_cholesky_block_cholesky,
    devices=cuda_devices,
    check_output=False,
)

add_function_test(
    TestTileCholesky,
    "test_tile_cholesky_upper",
    test_tile_cholesky_upper,
    devices=all_devices,
    check_output=False,
)
add_function_test(
    TestTileCholesky,
    "test_tile_cholesky_upper_inplace",
    test_tile_cholesky_upper_inplace,
    devices=all_devices,
    check_output=False,
)
add_function_test(
    TestTileCholesky,
    "test_tile_cholesky_upper_multiple_rhs",
    test_tile_cholesky_upper_multiple_rhs,
    devices=all_devices,
    check_output=False,
)
add_function_test(
    TestTileCholesky,
    "test_tile_cholesky_upper_multiple_rhs_inplace",
    test_tile_cholesky_upper_multiple_rhs_inplace,
    devices=all_devices,
    check_output=False,
)

add_function_test(
    TestTileCholesky, "test_tile_cholesky_solve_upper", test_tile_cholesky_solve_upper, devices=all_devices
)
add_function_test(
    TestTileCholesky,
    "test_tile_cholesky_solve_upper_inplace",
    test_tile_cholesky_solve_upper_inplace,
    devices=all_devices,
)
add_function_test(
    TestTileCholesky,
    "test_tile_cholesky_solve_upper_multiple_rhs",
    test_tile_cholesky_solve_upper_multiple_rhs,
    devices=all_devices,
)
add_function_test(
    TestTileCholesky,
    "test_tile_cholesky_solve_upper_multiple_rhs_inplace",
    test_tile_cholesky_solve_upper_multiple_rhs_inplace,
    devices=all_devices,
)


add_function_test(
    TestTileCholesky,
    "test_tile_cholesky_lower_backward_fp32",
    test_tile_cholesky_lower_backward(wp.float32),
    devices=all_devices,
    check_output=False,
)
add_function_test(
    TestTileCholesky,
    "test_tile_cholesky_lower_backward_fp64",
    test_tile_cholesky_lower_backward(wp.float64),
    devices=all_devices,
    check_output=False,
)

add_function_test(
    TestTileCholesky,
    "test_tile_cholesky_upper_backward_fp32",
    test_tile_cholesky_upper_backward(wp.float32),
    devices=all_devices,
    check_output=False,
)
add_function_test(
    TestTileCholesky,
    "test_tile_cholesky_upper_backward_fp64",
    test_tile_cholesky_upper_backward(wp.float64),
    devices=all_devices,
    check_output=False,
)

cpu_block_tests = (
    ("test_tile_cholesky_lower_factor_and_solve", test_tile_cholesky_lower_factor_and_solve),
    ("test_tile_cholesky_lower_factor_and_solve_inplace", test_tile_cholesky_lower_factor_and_solve_inplace),
    (
        "test_tile_cholesky_lower_factor_and_solve_multiple_rhs",
        test_tile_cholesky_lower_factor_and_solve_multiple_rhs,
    ),
    ("test_tile_cholesky_upper", test_tile_cholesky_upper),
    ("test_tile_cholesky_upper_inplace", test_tile_cholesky_upper_inplace),
    ("test_tile_cholesky_solve_upper", test_tile_cholesky_solve_upper),
    ("test_tile_cholesky_solve_upper_multiple_rhs", test_tile_cholesky_solve_upper_multiple_rhs),
    ("test_tile_cholesky_lower_backward_fp32", test_tile_cholesky_lower_backward(wp.float32)),
    ("test_tile_cholesky_upper_backward_fp32", test_tile_cholesky_upper_backward(wp.float32)),
)
for name, func in cpu_block_tests:
    add_function_test(
        TestTileCholesky,
        f"{name}_cpu_blocks",
        func,
        devices=get_cpu_test_devices(),
        check_output=False,
        enable_cpu_blocks=True,
    )


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
