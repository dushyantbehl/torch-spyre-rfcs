# SpyreCode Launch Spec: a single input contract for launching compiled SpyreCode

**Authors:**
* Dushyant Behl
* Chander Govindarajan
* Ashok Pon Kumar

## **Summary**

Launching an already-compiled SpyreCode directory today with [spyre-cli](https://github.com/torch-spyre/torch-spyre/tree/main/extensions/spyre-cli)
requires the caller to retype tensor shapes and dtypes that the compiler already knew.
Current `spyre-cli` takes [inline strings](https://github.com/torch-spyre/torch-spyre/tree/main/extensions/spyre-cli#cli) such as `-i 10x512@fp16` and builds tensors positionally.
Three mechanisms overlap in this space, and were discussed on PR #4077 as ways
to provide input to `spyre-cli`:

* **A human-authored IOSpec JSON** was proposed in #4077 but closed in favour of
  compiler-generated iospecs, so that humans do not have to write them.
* **The OpSpec Lab** (`tests/op_specs/`, and its related PR #4060)
  captures a per-kernel replay script from `torch.compile`, carrying shapes,
  layouts, and pool size.
* **Reconstructing the launch plan from the compiled artifacts.** Rather than
  being told the tensor interface, a launcher can parse `bundle.mlir` and
  `sdsc_*.json` and infer it: operation order, tensor roles, dimensions,
  layouts, and allocation addresses, resolving symbolic addresses and external
  inputs and outputs from producer/consumer flow. This needs no PyTorch and no
  extra file, and it is a proven approach — but it re-derives what the compiler
  already knew, and a PyTorch-side implementation would mean a second
  MLIR/SDSC parser to keep in step with the emitters.

In PR #4290 we shipped the inline-string form as an interim step.
In the PR #4077 we converged on compiler-emitted I/O metadata,
across both emitters, as the right answer.
This RFC is the design that #4077's review explicitly deferred.

We propose a **Launch Spec**: a single declarative JSON contract,
emitted by the compiler at build time and consumed by `spyre-cli`, that
describes every argument of every kernel in a SpyreCode folder.
We plan to design the spec so that a KTIR-emitted folder is describable in the
same schema as an SDSC-emitted one.

The concrete problems it fixes, all verifiable in today's tree:

1. Shape and dtype mismatches are silently accepted and launched.
2. `spyre-cli` cannot use symbolic-address launches, multi-kernel folders,
   multiple outputs, in-place arguments, or non-`ones` input values.
3. There is no machine-readable description of what a SpyreCode folder expects,
   so every consumer re-derives it differently.

## **Motivation**

### Possible (silent) mismatch hole

Inside the `spyre-cli` README we had stated the problem which is,

> 1. You need to pass the right input and output list.
> 2. If the shapes don't match, there is no error - the code will silently work.

This is because `launch_from_cli` (`spyre_cli/core.py:74-104`) creates tensors purely from the user's strings,
appends them input-then-output into one flat list, and passes them to
`SpyreSDSCKernelRunner.run(*tensors)`.

Nowhere we check what the compiled kernel expects. As example, a transposed pair of dimensions
produces a launch, not an error, and the printed output tensor looks plausible.

The launch path also has no backstop. `SpyreStream::launch`
(`torch_spyre/csrc/spyre_stream.cpp:277-283`) validates exactly one property per argument:

```cpp
for (size_t i = 0; i < args.size(); ++i) {
  TORCH_CHECK(args[i].is_privateuseone(), "SpyreStream::launch: argument ", i,
              " must be on Spyre device, got ", args[i].device());
}
```

We do not check anything like count, order, dtype, shape which is a limitation of `spyre-cli` and more importantly a potential problem.
We should note here that **the compiled bundle is the source of truth for argument identity and order.**
Any second, human-maintained copy of that contract can only drift from it, and
requires extra work to create in the first place.
To make `spyre-cli` a useful tool for debugging compiled code, we need to fix
this.

### How spyre-cli uses kernel runner

`SpyreSDSCKernelRunner.__init__`
(`torch_spyre/execution/kernel_runner.py:60-119`) accepts four parameters:

```python
def __init__(self, name, code_dir, kernel_provenance=None, symbol_kinds=None):
```

Currently `spyre-cli` passes only two (`core.py:24`):

```python
runner = SpyreSDSCKernelRunner("spyre-cli", str(path))
runner.run(*tensors)
```

We need to enable the two additional parameters because:

* **`symbol_kinds`** is the canonical symbol order returned by
  `generate_bundle()`. When it is absent, `_symbolic_args` is set to `None`
  (`kernel_runner.py:118-119`), and `run` takes the two-argument
  `launch_jobplan(self.jobplan, args)` branch (`kernel_runner.py:147-150`).
  So every `spyre-cli` launch silently runs without the `SymbolicArg` address
  payload. Since `config.bundle_symbolic_args` defaults to **on**
  (`_inductor/config.py:245`, `BUNDLE_SYMBOLIC_ARGS=1`), `spyre-cli` is
  launching bundles in a mode the default compile path does not use. Any
  bundle whose symbol table needs `kAddress` binding — the pool parameter case
  at `kernel_runner.py:96-111`, the `kernel_slice` and `kernel_derived`
  variants documented at `codegen/compute_ops.py:37-53` — is not reachable
  from the CLI today.
* **`kernel_provenance`** is what populates `profiler_event_name` and drives
  `register_kernel_provenance`, so CLI launches cannot be joined to profiler
  traces.

### Multi kernel input

Currently `spyre-cli` treats the path as one kernel: it appends `/spyreCodeDir`
(`kernel_runner.py:130`) and calls `prepare_kernel` once. But a compiled output
directory for a real graph can hold several kernels, and the CLI has no way to name
one, list them, or launch them in sequence.

### Where to get this information

The OpSpec Lab's generated script proves the compiler can hand over everything
needed. `tests/op_specs/capture.py:418-435` emits, per kernel:

```python
KERNEL_NAME = "{rec.name}"
POOL_SIZE = {rec.pool_size}
BUNDLE_SYMBOLIC_ARGS = {rec.bundle_symbolic_args}
SHAPES = [ ... ]    # Host (shape, dtype) per kernel arg, in arg_index order.
LAYOUTS = [ ... ]   # Exact device layout each arg had when the real graph ran.
```

This potentially can be used as the missing contract — kernel name, per-arg host shape and dtype in
`arg_index` order, device layouts, pool size, and the symbolic-args flag.
The problem is that this is presented only by a separate script and
the contract right now is produced only by a capture run, not by an ordinary compile.

There is also a second emitter arriving: KTIR (`config.ktir_emitter`,
`TORCH_SPYRE_KTIR=1`), which writes `<kernel>.ktir` at
`execution/async_compile.py:456`. If the launch contract is not defined
emitter-neutrally before KTIR folders become common, we will end up with two
incompatible ad-hoc input paths instead of one.

## **Proposed Implementation**

### The Launch Spec

One `launch_spec.json` written at the root of a SpyreCode output directory,
beside the existing per-kernel `spyreCodeDir/`:

```json
{
  "version": 1,
  "emitter": "sdsc",
  "kernels": [
    {
      "name": "sdsc_fused_add_0",
      "code_dir": "sdsc_fused_add_0",
      "pool_size": 0,
      "bundle_symbolic_args": true,
      "symbol_kinds": [
        {"kind": "kernel", "arg_index": 0},
        {"kind": "kernel", "arg_index": 1},
        {"kind": "kernel", "arg_index": 2}
      ],
      "args": [
        {"name": "arg0", "arg_index": 0, "role": "input",
         "shape": [10, 512], "dtype": "float16",
         "layout": {"stick_size": [64], "padding": [0, 0]}},
        {"name": "arg1", "arg_index": 1, "role": "input",
         "shape": [10, 512], "dtype": "float16",
         "layout": {"stick_size": [64], "padding": [0, 0]}},
        {"name": "buf0", "arg_index": 2, "role": "output",
         "shape": [10, 512], "dtype": "float16",
         "layout": {"stick_size": [64], "padding": [0, 0]}}
      ]
    }
  ]
}
```

Field notes:

* `args` is ordered by `arg_index`, matching the MLIR `input_arg` slot order
  described at `kernel_runner.py:92-95`. The pool parameter, when
  `symbol_kinds[0].is_pool`, is **not** an entry in `args`; it is implied by
  `pool_size > 0` and prepended by the launcher exactly as `call_kernel` does.
  This keeps `arg_index` meaning the same thing it means in the compiler.
* `role` is one of `input`, `output`, `input_output`, closing the in-place gap.
* `dtype` uses full torch names (`float16`), not the CLI's short forms, so the
  schema is not limited to the three-entry `dtype_mapping` at `core.py:28-32`.
* `layout` is optional and advisory for launch, but is what lets a future
  validator check stick alignment before touching hardware.
* `symbol_kinds` is serialized from the same list `generate_bundle()` returns,
  so the CLI can finally construct the runner with its fourth parameter.

Symbolic dimensions get a `symbols` block per kernel, with `shape` entries
allowed to name a symbol instead of an integer:

```json
"symbols": {"s0": {"min": 1, "max": 256}},
"args": [{"name": "arg0", "arg_index": 0, "role": "input",
          "shape": ["s0", 512], "dtype": "float16"}]
```

Binding them at launch stays explicit: `spyre launch --bind s0=128 <folder>`.

### KTIR extension

The spec is emitter-neutral by construction. A KTIR folder sets
`"emitter": "ktir"` and points `code_dir` at the `<kernel>.ktir` artifact
written by `async_compile.ktir` (`execution/async_compile.py:456`) together with
its DBO-compiled output. The `args`, `role`, `shape`, `dtype`, and `symbols`
fields carry over unchanged, because they describe the kernel's interface rather
than the artifact format.

Two KTIR-specific facts must be recorded rather than inferred, both consequences
of constraints that already exist:

* `emitter` is not convertible after the fact. Emitter choice is fixed at
  capture time; KTIR requires per-buffer names under `config.ktir_emitter`, and
  KTIR artifacts may carry baked addresses the SDSC generator rejects. Writing
  `emitter` into the spec makes a mismatched launch a clear error instead of a
  confusing generator failure.
* KTIR device lowering needs `config.ktir_device_mlir`
  (`KTIR_DEVICE_MLIR`, `config.py:84`), checked by
  `_check_ktir_device_prerequisites` (`async_compile.py:63-71`). The spec
  records that the folder was built expecting it, so the CLI can fail with a
  precise message rather than a missing-dialect traceback.

A `"emitter": "ktir"` folder is therefore launchable through the same
`spyre launch` command, with the runner selection — not the input parsing —
being the only thing that branches.

### Changes to spyre-cli

`spyre launch` can be extended to support something like below,

* Spec-driven: no shapes retyped, validated, all kernels discoverable
```bash
spyre launch <folder>
spyre launch --kernel sdsc_fused_add_0 <folder>
spyre launch --list <folder>
```
* Values beyond torch.ones
```bash
spyre launch --input-file arg0=arg0.pt <folder>
```
* Inline strings stay for now, and can be deprecated once the spec is
  emitted by default.

Resolution order in `launch_from_cli`:

1. If `launch_spec.json` exists, load it. With no `-i/-o`, build every tensor
   from the spec — the common case becomes `spyre launch <folder>`.
2. If `-i/-o` are also given, parse them with the existing
   `create_tensor_info` and **compare against the spec**; on mismatch, exit
   non-zero naming the argument, the expected shape/dtype, and the supplied
   one. This closes the silent-mismatch hole described above.
3. If no spec exists, behave exactly as today, but print a warning that shapes
   are unvalidated. Folders compiled before this RFC keep working.

The runner construction becomes, with the spec supplying what is currently
dropped:

```python
runner = SpyreSDSCKernelRunner(
    kernel.name,
    str(kernel_dir),
    symbol_kinds=kernel.symbol_kinds,   # was always None
)
```

Multiple outputs are printed per output arg using `role`, retiring the
`core.py:99-102` TODO and its warning.

One related behaviour must be preserved: outputs are created with
`torch.empty`, not `torch.ones` (`core.py:88-95`). This was a review fix in
#4077 and still holds in `main`. Pre-filling outputs with `1.0` lets a launch
that never writes its output print plausible all-ones data; `empty` surfaces
that as obvious garbage. Any spec-driven allocation path must keep it.

### Where to emit the spec?

The spec should be emitted close to the bundle
path in `_inductor/codegen/bundle.py` for `generate_bundle()`, and
`execution/async_compile.py` for the KTIR path, both writing through one shared
serializer so the two emitters cannot drift.

* **Stage 0** — the argument-count check requested on #4077 and never landed.
  This needs no spec at all and should not wait for one: compare
  `len(tensors)` against the external-I/O argument count the prepared jobplan
  expects, and fail before launching. It requires exposing that count on
  `JobPlan`, which `torch_spyre/_C.pyi:395-415` does not yet do (it exposes
  `num_steps`, `job_allocation_size`, `get_step_type`, `get_step_name`), so the
  cost is one small C++ accessor. This catches the worst case — wrong number of
  tensors — independently of everything below, and is worth landing first even
  if the rest of this RFC is rejected.
* **Stage 1** — schema plus serializer, written behind a config flag; a spec is
  emitted but nothing consumes it. Unblocks review of the field set against
  real folders.
* **Stage 2** — `spyre-cli` reads the spec: `--list`, `--kernel`, tensor
  construction from the spec, mismatch validation, `symbol_kinds` wired
  through. The user-visible win lands here.
* **Stage 3** — symbolic dimension binding (`--bind`), `--input-file`, and
  `input_output` role handling.
* **Stage 4** — KTIR emitter support and `kernel_provenance` plumbing for
  profiler joins.

The OpSpec Lab can be updated to emit a `launch_spec.json` alongside its replay
script, from the same `SHAPES`/`LAYOUTS`/`POOL_SIZE` record it already builds
(`capture.py:418-435`). The declarative spec and the executable script serve
different jobs — launching versus compiler-boundary replay — and neither
replaces the other.

## **Metrics**

* Argument-count mismatches fail with a diagnostic instead of launching: today
  0%, target 100% (Stage 0, independent of the spec).
* Shape/dtype mismatches fail with a diagnostic instead of launching: today 0%,
  target 100% for folders carrying a spec.
* Shapes retyped by a user to launch a folder: from every invocation to zero.
* Fraction of `SpyreSDSCKernelRunner` capability reachable from the CLI:
  symbolic args, multi-kernel, multi-output, in-place, and provenance move from
  unreachable to reachable.
* Kernels in a multi-kernel folder launchable without writing Python: 1 → all.
* Same `spyre launch` invocation works against an SDSC and a KTIR folder.

## **Drawbacks**

* Maintaining another compiler artifact, a stale or wrong
  spec produces confident wrong validation, which is worse than none. Mitigated
  by generating it from the same structures `generate_bundle()` returns rather
  than a parallel derivation, and by treating a spec/kernel disagreement as a
  compiler bug with a test.
* Implementation to be done in inductor, while the CLI change is small; the real
  work is serializing `symbol_kinds`, layouts, and symbols faithfully across two
  emitters.
* The spec makes launches *well-formed*, not
  *correct*. `torch.ones` inputs still tell you nothing about numerics;
  `--input-file` and the OpSpec Lab's `--save-inputs` are the answer there.

## **Alternatives**

* **Query the compiled artifacts at launch.** Have the CLI parse
  `spyrecode.json` / `bundle.mlir` directly and infer the interface, the
  reconstruct-the-launch-plan approach described in the Summary, so no new file
  is needed. This is the most appealing alternative: single source of
  truth, nothing to keep in sync. Rejected for now because it puts a second
  MLIR/SDSC parser into the Python path and re-derives what the compiler
  already knew; the existing `spyrecode.json` carries `JobPreparationPlan` and
  `JobExecPlan` (`tests/test_prepare_kernel.py:158-171`), not host dtypes or
  argument roles. Reconsider if the emitted spec proves hard to keep truthful.
* **Extend the OpSpec Lab to cover launching.** It already has the data, but a
  capture run is a heavier prerequisite than an ordinary compile, and its
  artifact is executable Python — a fine debugging vehicle, a poor input
  contract.

## **How we teach this**

TBA

## **Unresolved questions**

To resolve through the RFC process:

* Should `launch_spec.json` be emitted always, or behind a config flag? Always
  is simpler to rely on; a flag avoids perturbing existing output directories.
* Is one spec per folder with a `kernels` array right, or one spec per
  `spyreCodeDir`? The array makes `--list` natural but adds a merge step when
  kernels are compiled independently and possibly concurrently.

To resolve during implementation:

* Exact serialization of the five `SymbolKind` variants (`kernel`,
  `kernel_slice`, `kernel_derived`, `kernel_derived_symbolic`, pool) described
  at `codegen/compute_ops.py:36-53`.
* Whether `input_output` aliasing needs an explicit `aliases` field or is
  adequately expressed by a shared `name`.
* How `--bind` interacts with `bundle_symbolic_args` and with symbolic dims
  already baked into the bundle.

Out of scope:

* Numerical validation and reference-result checking; the spec makes launches
  well-formed, not correct.
* Teaching a standalone (non-PyTorch) launcher to read the spec instead of
  reconstructing the interface itself. Worth doing — it would give every
  mechanism one shared vocabulary — but it is out of this repo's scope and
  should be its own RFC.
* Any change to how `torch.compile` lowers or schedules.

## Resolution

Not yet decided; this RFC is submitted for comment.

### Level of Support

Unset — pending review.

#### Additional Context

Status of the related work, verified against GitHub at the time of writing:

| PR | Title | State |
|---|---|---|
| #4077 | Spyre-cli: Add direct launcher for spyrecode (incl. IOSpec JSON) | **Closed, unmerged** |
| #4290 | add direct launcher for spyrecode (successor to #4077) | **Merged** 2026-09-08 (`d102b7b3`) |
| #4060 | Give the OpSpec Lab a `--stage ktir` alongside `--stage bundle` | **Open**, `REVIEW_REQUIRED` |
| #4058 | KTIR Emitter with OpSpec Lab (tracking issue for #4060) | Open |
| #4031 | Fix the OpSpec lab's HBM pool handling for the current sdsc ABI | Merged 2026-08-26 |

Two consequences for reviewers. First, the KTIR capture path this RFC builds on
is **not yet merged**, so Stage 4 depends on #4060 merge; the schema is
emitter-neutral specifically so that dependency does not block Stages 0-3.
Second, there is no open issue tracking compiler-derived IOSpec emission
(`gh search issues --repo torch-spyre/torch-spyre "iospec"` returns nothing), so
the work #4290 was explicitly waiting on has no home yet. This RFC is intended
to become it.

The silent-mismatch behavior, the device-residency-only check in
`SpyreStream::launch`, and the unused `symbol_kinds` parameter are all
verifiable in the current tree at the file and line references cited above.

