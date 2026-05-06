# tile_empty

**Status**: Proposed

**Issue**: TBD

## Motivation

Tile construction primitives currently force initialization. ``tile_zeros``
zero-fills every register (or shared-memory cell) before the user has a chance
to write the real values; ``tile_ones`` does the same with a one-fill. When a
kernel will fully overwrite the tile immediately afterward — e.g., the result
of a load, a generated lookup, the output of a custom reduction — that
initialization is wasted work.

For shared storage the cost is unconditional: the broadcast hits real
shared-memory writes plus a barrier. For register storage the situation is
worse than it appears at first glance: today, ``var = wp.tile_zeros<...>()``
emits three loop-fills' worth of work in C++ — the cinit declaration's default
constructor, the implicit converting constructor on the rvalue side, and the
copy assignment. The compiler may DCE some of those, but not reliably across
element types and storage paths.

``tile_empty`` exposes the same construction surface as ``tile_zeros`` but
skips initialization, letting users opt into "I will fill this myself"
semantics in the same way ``np.empty`` parallels ``np.zeros``. As a deliberate
side effect, the implementation also collapses ``tile_zeros`` to a single
loop-fill by making register tiles follow the same allocation/assignment model
that shared tiles already use today.

## Requirements

| ID  | Requirement                                                                  | Priority | Notes |
| --- | ---------------------------------------------------------------------------- | -------- | ----- |
| R1  | Provide ``wp.tile_empty(shape, dtype, storage)`` mirroring ``tile_zeros``    | Must     | Tuple- and scalar-shape overloads |
| R2  | Support both ``storage="register"`` and ``storage="shared"``                 | Must     |       |
| R3  | Default ``dtype=float`` and ``storage="register"`` to match ``tile_zeros``   | Must     |       |
| R4  | ``tile_empty`` must be at least as fast as ``tile_zeros``                    | Must     | Demonstrated by a benchmark |
| R5  | ``tile_zeros`` must remain at least as fast as it is today                   | Must     | The refactor must not regress existing code |
| R6  | Document the read-before-write contract: full overwrite required            | Must     | Docstring + user guide |
| R7  | Tests cover both storages, both shape overloads, and several dtypes          | Must     | New ``test_tile_empty.py`` |
| R8  | Scratch benchmark comparing ``tile_empty`` vs ``tile_zeros``                 | Should   | Under ``temp/`` |

**Non-goals**:

- Differentiability. Empty tiles have no meaningful adjoint; ``is_differentiable=False``.
- A NumPy-style host-side ``wp.empty`` for ``warp.array``. Scoped to the in-kernel tile primitive.
- Changing the existing ``tile_zeros``/``tile_ones`` Python-level API or defaults.

## User contract

The contract for ``tile_empty`` matches ``np.empty``: the contents are
undefined, and the user is responsible for overwriting every element before
any read. The C++ language level cannot enforce this — same as reading a
default-init ``int`` — so it is documented and tested via the safe patterns
only.

| Pattern | Safe? | Why |
| --- | --- | --- |
| ``a = wp.tile_empty(...)`` then ``a = wp.tile_load(...)``           | ✅ | ``tile_load`` overwrites every element |
| ``a = wp.tile_empty(...)`` then ``a = b`` (whole tile)              | ✅ | Whole-tile copy-assign overwrites |
| ``a = wp.tile_empty(...)`` then ``a = wp.tile_add(b, c)``           | ✅ | RHS does not read ``a`` |
| ``a = wp.tile_empty(...)`` then write every element in a loop      | ✅ | If every element is written |
| ``a = wp.tile_empty(...)`` then ``a += b``                          | ❌ UB | ``+=`` reads ``a`` before writing |
| ``a = wp.tile_empty(...)`` then ``c = a + b``                       | ❌ UB | Reads ``a`` as an operand |
| ``a = wp.tile_empty(...)``, write ``a[0,0]``, read ``a[1,1]``       | ❌ UB | Partial fill |

Recommended docstring guideline: "If you intend to accumulate into the tile
(``a += ...``), use ``tile_zeros`` instead. ``tile_empty`` is for the case
where the first operation is a full overwrite."

The existing ``FP_CHECK`` debug build option on ``tile_alloc_empty`` (which
NaN-fills shared-tile allocations) naturally extends a debug-mode safety net
to ``tile_empty`` for shared storage at no extra cost.

## Design

### Approach: register tiles mirror shared tiles

Today, shared tiles already follow an "allocate uninitialized, assign to
fill" model:

```cpp
auto var = wp::tile_alloc_empty<float, ...>();   // uninitialized
var = wp::tile_zeros<float, ...>();              // operator= broadcasts 0
```

Register tiles do not — their default constructor at ``tile.h:829``
zero-fills via a loop, and the rvalue side of an assignment goes through an
implicit converting constructor that fills again, producing redundant work.

This design extends the shared-tile model to register tiles:

1. ``tile_register_t``'s default constructor becomes a no-op.
2. A new ``tile_register_t::operator=(const T&)`` broadcasts a scalar to all
   elements, mirroring how ``tile_shared_t`` accepts scalar assignment.
3. A new tag type ``tile_no_init_t`` is added. Both ``tile_register_t`` and
   ``tile_shared_t`` get an ``operator=(tile_no_init_t)`` that is a literal
   no-op.
4. The new ``tile_empty<T, Shape...>()`` free function returns
   ``tile_no_init_t{}``.

Under this model, the generated C++ for both primitives is uniform across
storage:

```cpp
// tile_empty / register
wp::tile_register_t<float, ...> var = wp::tile_register_t<float, ...>{};
var = wp::tile_empty<float, 16, 16>();   // operator=(tile_no_init_t): no-op

// tile_zeros / register
wp::tile_register_t<float, ...> var = wp::tile_register_t<float, ...>{};
var = wp::tile_zeros<float, 16, 16>();   // operator=(float): broadcast

// tile_empty / shared
auto var = wp::tile_alloc_empty<float, ..., false>();   // uninitialized
var = wp::tile_empty<float, 16, 16>();   // no-op

// tile_zeros / shared
auto var = wp::tile_alloc_empty<float, ..., false>();
var = wp::tile_zeros<float, 16, 16>();   // broadcast
```

``tile_no_init_t`` is a one-byte tag that lives only in the rvalue position of
an assignment. The user-visible variable is always a real ``tile_register_t``
or ``tile_shared_t``; the tag never appears as a variable type. The compiler
sees an empty struct passed to an empty function and DCEs the call.

### Alternatives Considered

1. **Keep the existing default ctor; add a separate ``requires_init`` flag on
   the ``tile`` Python type that flows through to ``cinit``.** Rejected: it
   solves only ``tile_empty`` and leaves the redundant work in ``tile_zeros``
   intact. The symmetric design fixes both.

2. **``tile_empty`` returns ``tile_zeros()`` and trust the compiler to DCE
   the dead init.** Rejected: DCE is unreliable for non-trivial element types
   and never applies to shared-memory zero-fills. R4 would not hold.

3. **Use ``__builtin_assume_initialized`` style intrinsics.** Rejected: not
   portable across CUDA/Clang versions Warp targets; tag-type dispatch is the
   idiomatic C++ approach.

### Implementation Details

**Native** (``warp/native/tile.h``):

```cpp
struct tile_no_init_t {};
inline constexpr tile_no_init_t tile_no_init{};

template <typename T, typename L> struct tile_register_t {
    // existing data and helpers ...

    // default ctor is now a no-op (mirrors tile_alloc_empty for shared)
    inline CUDA_CALLABLE tile_register_t() {}

    // scalar broadcast assignment (new)
    inline CUDA_CALLABLE auto& operator=(const T& value) {
        for (int i = 0; i < Layout::NumRegs; ++i) data[i] = value;
        return *this;
    }

    // no-init assignment (new) — the empty case
    inline CUDA_CALLABLE auto& operator=(tile_no_init_t) { return *this; }

    // existing operator= for tile_global_t, tile_register_t, tile_shared_t are unchanged
};

// add the symmetric overload to tile_shared_t too
template <typename T, typename L, bool Owner> struct tile_shared_t {
    // ...
    inline CUDA_CALLABLE auto& operator=(tile_no_init_t) { return *this; }
};

template <typename T, unsigned... Shape>
inline CUDA_CALLABLE auto tile_empty() { return tile_no_init_t{}; }
```

The existing converting constructor ``tile_register_t(const tile_shared_t&)``
at ``tile.h:845`` is preserved.

**Native callers that depend on the old zero-init default ctor**: a recon
pass identified four call sites (one critical, three secondary) where a
``tile_register_t`` is default-constructed and then partially written via a
loop with an early ``break`` on ``!Layout::valid(linear)``. Trailing slots
remain uninitialized and are read by callers, which works today only because
of the implicit zero-fill.

| File:line | Function |
| --- | --- |
| ``tile.h:1677``        | ``tile_shared_t::grad_to_register()`` |
| ``tile_reduce.h:398``  | ``tile_reduce_axis_impl`` final copy |
| ``tile_reduce.h:530``  | ``tile_reduce_axis_impl`` variant |
| ``tile_reduce.h:577``  | ``tile_reduce_axis_impl`` variant |

All four have the same shape: a default-constructed ``tile_register_t out``
followed by a loop that writes ``out.data[i]`` only for valid linear indices
and breaks early otherwise. The fix at each site is to ensure all
``Layout::NumRegs`` slots are written. The simplest form is to drop the
``break`` and write a sentinel for the invalid path:

```cpp
out.data[i] = Layout::valid(linear) ? grad(linear) : T{};
```

This preserves today's externally observable behavior (the slots that were
zero before are still zero) and is a 1-2 line change per site.

**Codegen** (``warp/_src/codegen.py``):

- The reverse-pass adjoint reset at lines 2099 and 2039 currently emits
  ``var_adj = {};`` for non-owner tile types, which today zero-fills via the
  default ctor. Under the refactor that becomes a no-op. Replace with an
  explicit broadcast: ``var_adj = wp::tile_zeros<T, Shape...>();`` (a free
  function call, not a method — consistent with Warp's codegen convention of
  not emitting struct method calls).
- ``cinit`` for register storage at ``types.py:5194`` is unchanged. It still
  emits ``wp::tile_register_t<...>{}``; the now-no-op default ctor is the
  desired behavior — primal vars start uninitialized and are filled by their
  first assignment.

**Python** (``warp/_src/builtins.py``):

Add ``tile_empty`` as a normal builtin mirroring ``tile_zeros`` exactly:

- ``tile_empty_value_func`` is structurally identical to
  ``tile_zeros_value_func``, returning ``tile(dtype, shape, storage)``.
- ``tile_empty_dispatch_func`` returns ``([], [dtype, *shape])``.
- Two ``add_builtin`` registrations cover the tuple-shape and scalar-shape
  overloads.
- ``input_types``, ``defaults`` (``storage="register"``, ``dtype=float``),
  ``is_differentiable=False``, and ``export=False`` all match ``tile_zeros``.

## Testing Strategy

New file ``warp/tests/tile/test_tile_empty.py`` registered in
``default_suite``:

- Both shape overloads (tuple and scalar).
- Both storages (``"register"``, ``"shared"``).
- Multiple dtypes: ``float``, ``int``, ``wp.vec3``.
- Pattern: allocate via ``tile_empty``, write all elements deterministically
  (one of the safe patterns from the contract table), read back, assert
  equality with the written values. **Never assert on contents prior to a
  full write.**
- A shape/dtype metadata test that the returned tile object's type matches
  the requested signature.

Device coverage: run via ``add_function_test()`` to cover all available
devices where tile primitives are supported.

**Regression coverage** for the refactor: existing ``tile_zeros`` and
``tile_ones`` tests must continue to pass unchanged. Backward (adjoint) tests
are particularly important — exercise ``tile_zeros``-allocated accumulators
in reverse-mode AD to confirm the codegen change at lines 2099/2039 produces
zero-initialized adjoints.

### Scratch benchmark

``temp/benchmark_tile_empty.py`` (under the persistent scratch dir):

- Compare ``tile_empty`` vs ``tile_zeros`` across {register, shared} ×
  {float, vec3} × shapes {16×16, 64×64, 128×128}.
- Inside each kernel: many alloc-and-fully-write iterations to amplify the
  construction cost relative to launch overhead.
- CUDA-event timing, with warmup. Print a table of microseconds per launch
  and the speedup ratio.
- Report two ratios:
  - ``tile_empty`` vs new ``tile_zeros`` (expected: ``tile_empty`` ≤
    ``tile_zeros``, R4).
  - New ``tile_zeros`` vs old ``tile_zeros`` baseline (expected: new ≤ old,
    R5; the redundant loop-fills are gone).
- Assert a soft lower bound: ``tile_empty`` time ≤ ``tile_zeros`` time × 1.05
  (5% slack for noise).
