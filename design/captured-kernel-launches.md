# Captured kernel launches

**Status**: Experimental kernel-capture implementation, validated on CUDA

## Motivation

Consumers that bind dynamic execution counts need the exact kernels and argument packets emitted
into a CUDA graph. Reconstructing those facts from mutable Python arrays, or replacing launch
functions globally, gives the consumer an unreliable second account of capture.

## Design

`capture_begin(record_launches=True)` and `ScopedCapture(record_launches=True)` opt into records.
The existing `Graph` is the composition root and sole ledger owner. `capture_get_launches(graph)`
returns an immutable tuple of frozen, identity-based `CapturedLaunch` records while capture is
inactive. A paused capture exposes a prefix that can grow on resume. A consumer preparing a
complete program must finish capture and must not resume it after binding. A record owns the
emitted kernel identity, compiled bounds shape, immediate CUDA graph and node handles, and a
byte snapshot of every packed parameter, including bounds. Argument owners are retained separately;
mutable arrays and structs are lifetime references, never the captured descriptor truth.

`context.py` owns the record type, graph ledger, launch recording and failure operations. `utils.py`
only forwards the capture option. A small native query in `warp.cu` identifies the single kernel
node immediately after emission and its immediate graph, including conditional bodies. Existing
stream capture IDs resolve joined streams to their owning Graph. The metadata adds no allocator,
execution scheduler, engine-specific count meanings or callbacks.

Ordinary `launch`, `launch_tiled`, and reusable `Launch.launch` calls keep their signatures. A
Graph-local incomplete-operation latch starts before launch validation and clears only after a
successful operation. Caught exceptions leave the graph unpublishable. `capture_invalidate(graph,
reason)` permanently invalidates capture or later consumer preparation; all replay paths reject it.
Successful metadata-only captures replay normally and require no downstream lowering.
External consumers may declare graph-owned identity obligations with
`capture_require_preparation(graph, owner)` and discharge them with
`capture_complete_preparation(graph, owner)`. Unresolved obligations block replay and child graph
embedding, but not capture completion or record inspection. Owners are strong references, not
callbacks; validation failures can be retried, while explicit invalidation remains permanent.

This prototype admits CUDA forward, unclustered, nondeterministic Warp launches. CPU/APIC,
adjoint and deterministic/clustered launch expansion are rejected for recorded captures. Captured
raw array ctypes, external array interfaces, textures and indexed/fabric arrays are rejected.
Regular Warp arrays and nested structs containing them retain their exact owners. Owner retention
does not establish physical backing readiness or make remapping safe; those remain the caller's
responsibility. Kernel metadata does not claim to record fills, copies, external CUDA launches
or allocations. Prebuilt graph insertion into a
recorded capture is rejected because CUDA clones nodes whose identities would otherwise be stale.
Callable conditional bodies are recorded normally. Recording on one graph is serialized at the
Python operation boundary; overlapping operations on that graph fail closed.

## Validation

Tests cover scalar/array packets, immutable snapshots after descriptor and reusable-command mutation,
retained nested struct arrays, tiled bounds, structural zero, joined streams and callable conditional
bodies. Unsupported/failed operations remain nonreplayable even when their exception is
caught inside capture, and explicit consumer invalidation also blocks an already-instantiated graph.

The separate GPU Components experiment adopts these records without changing ordinary calculation
signatures. GPU tests cover zero/shrink/grow replay counts, independent scalar and launch counts,
tiled reductions, joined streams, and two instances of the same kernel with independent populations.
The unchanged MJWarp collision-wake kernel also matches eager execution over eight changing counts.
These checks establish a kernel boundary; full engine adoption still needs explicit memory-operation
handling, physical-readiness admission, and adoption within conditional bodies.
