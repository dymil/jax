# Copyright 2024 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module for emitting custom GPU pipelines within a Pallas kernel."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import dataclasses
import functools
import itertools as it
import math
from typing import Any

import jax
from jax import lax
from jax._src import core
from jax._src import linear_util as lu
from jax._src import util
from jax._src.interpreters import partial_eval as pe
from jax._src.pallas import core as pallas_core
from jax._src.pallas.mosaic_gpu import core as gpu_core
from jax._src.pallas.mosaic_gpu import primitives as gpu_primitives
from jax.experimental import pallas as pl
import jax.numpy as jnp


map = util.safe_map
zip = util.safe_zip


def _uses_arguments(
    index_map: Callable[..., Any], grid: pallas_core.StaticGrid
) -> Sequence[bool]:
  jaxpr, _, _, () = pe.trace_to_jaxpr_dynamic(
      lu.wrap_init(index_map), (core.ShapedArray((), jnp.int32),) * len(grid)
  )
  _, used_inputs = pe.dce_jaxpr(jaxpr, used_outputs=[True] * len(jaxpr.outvars))
  return used_inputs


@dataclasses.dataclass(frozen=True)
class BufferedRef:
  spec: pallas_core.BlockSpec
  grid: pallas_core.StaticGrid
  gmem_ref: pallas_core.AbstractMemoryRef
  smem_ref: pallas_core.AbstractMemoryRef  # [num_slots, *spec.block_shape]

  def compute_gmem_slice(self, grid_indices) -> tuple[pl.Slice, ...]:
    index_map = self.spec.index_map
    assert index_map is not None
    return tuple(
        pl.Slice(idx * size, size)
        for idx, size in zip(
            index_map(*grid_indices), self.spec.block_shape  # type: ignore[arg-type]
        )
    )

  @functools.cached_property
  def is_index_invariant(self) -> bool:
    """Returns whether the ref is invariant to the grid indices."""
    return not any(_uses_arguments(self.spec.index_map, self.grid))

  def copy_in(self, slot, grid_indices, barrier_ref):
    gmem_slices = self.compute_gmem_slice(grid_indices)
    gpu_primitives.copy_gmem_to_smem(
        self.gmem_ref.at[gmem_slices],  # pytype: disable=unsupported-operands
        self.smem_ref.at[slot],
        barrier=barrier_ref.at[slot],
    )

  def copy_out(self, slot, grid_indices, predicate=None):
    gmem_slices = self.compute_gmem_slice(grid_indices)
    gpu_primitives.copy_smem_to_gmem(
        self.smem_ref.at[slot],
        self.gmem_ref.at[gmem_slices],  # pytype: disable=unsupported-operands
        predicate=predicate,
    )


jax.tree_util.register_dataclass(
    BufferedRef,
    data_fields=["gmem_ref", "smem_ref"],
    meta_fields=["spec", "grid"],
)


def _inc_grid(
    indices: tuple[jax.Array, ...],
    grid: Sequence[int],
    by: jax.typing.ArrayLike,
) -> tuple[jax.Array, ...]:
  """Returns the grid indices after ``by`` steps."""
  new_indices = []
  carry = by
  for idx, size in reversed(zip(indices, grid)):
    idx += carry
    # The branching avoids using expensive integer division in the common
    # case of ``size < by < 2*size``.
    new_idx, carry = jax.lax.cond(
        idx < 2 * size,
        lambda idx=idx, size=size: (
            jax.lax.cond(idx < size, lambda: (idx, 0), lambda: (idx - size, 1))
        ),
        lambda idx=idx, size=size: (lax.rem(idx, size), lax.div(idx, size)),
    )
    new_indices.append(new_idx)
  return tuple(reversed(new_indices))


# ``pl.Slice`` uses a different pytree encoding, depending on whether the
# start/size are static or dynamic. This leads to pytree structure mismatch
# in the pipeline body. So, we define a different ``Slice`` class below.


@dataclasses.dataclass(frozen=True)
class _Slice:
  start: int | jax.Array
  size: int | jax.Array

  def __eq__(self, other: _Slice) -> jax.Array:  # pytype: disable=signature-mismatch
    return lax.bitwise_and(self.start == other.start, self.size == other.size)


jax.tree_util.register_dataclass(
    _Slice, data_fields=["start", "size"], meta_fields=[]
)


def emit_pipeline(
    body,
    *,
    grid: pallas_core.StaticGrid,
    in_specs: Sequence[pallas_core.BlockSpec] = (),
    out_specs: Sequence[pallas_core.BlockSpec] = (),
    max_concurrent_steps: int = 1,
):
  """Creates a function to emit a manual pipeline within a Pallas kernel."""
  num_steps = math.prod(grid)

  # Shrink ``max_concurrent_steps`` if the total number of steps is lower to
  # reduce the size of the allocated buffers below.
  if max_concurrent_steps > num_steps:
    max_concurrent_steps = num_steps

  def pipeline(*gmem_refs: pallas_core.AbstractMemoryRef):
    in_gmem_refs, out_gmem_refs = util.split_list(gmem_refs, [len(in_specs)])
    in_smem_refs, out_smem_refs = util.split_list(
        map(
            lambda spec, ref: gpu_core.SMEM(
                (max_concurrent_steps, *spec.block_shape),  # type: ignore
                ref.dtype,
            ),
            it.chain(in_specs, out_specs),
            gmem_refs,
        ),
        [len(in_specs)],
    )
    return pl.run_scoped(
        functools.partial(
            scoped_pipeline,
            in_gmem_refs=in_gmem_refs,
            out_gmem_refs=out_gmem_refs,
        ),
        in_smem_refs=in_smem_refs,
        out_smem_refs=out_smem_refs,
        barrier_ref=gpu_core.Barrier(
            # TODO(slebedev): Change this to arrive only once.
            len(in_specs),
            num_barriers=max_concurrent_steps,
        ),
    )

  def scoped_pipeline(
      *, in_gmem_refs, out_gmem_refs, in_smem_refs, out_smem_refs, barrier_ref
  ):
    in_brefs: Sequence[BufferedRef] = [
        BufferedRef(spec, grid, gmem_ref, smem_ref)
        for spec, gmem_ref, smem_ref in zip(
            in_specs, in_gmem_refs, in_smem_refs
        )
    ]
    out_brefs: Sequence[BufferedRef] = [
        BufferedRef(spec, grid, gmem_ref, smem_ref)
        for spec, gmem_ref, smem_ref in zip(
            out_specs, out_gmem_refs, out_smem_refs
        )
    ]

    for step, indices in enumerate(
        it.islice(it.product(*map(range, grid)), max_concurrent_steps)
    ):
      map(lambda bref: bref.copy_in(step, indices, barrier_ref), in_brefs)

    def loop_body(step, carry):
      slot = step % max_concurrent_steps
      indices, last_store_slices = carry

      if in_specs:
        # Wait for the current GMEM->SMEM copy to complete.
        gpu_primitives.barrier_wait(barrier_ref.at[slot])
      # Wait for the previous output SMEM->GMEM copy to complete.
      gpu_primitives.wait_smem_to_gmem(max_concurrent_steps - 1)

      with pallas_core.grid_env(map(pallas_core.GridAxis, indices, grid)):
        body(
            *(bref.smem_ref.at[slot] for bref in it.chain(in_brefs, out_brefs))
        )

      if not all(bref.is_index_invariant for bref in out_brefs):
        gpu_primitives.commit_smem()

      # Copy the output from SMEM to GMEM.
      new_store_slices = last_store_slices[:]
      for idx, bref in enumerate(out_brefs):
        if bref.is_index_invariant:
          assert last_store_slices[idx] is None
          continue
        assert last_store_slices[idx] is not None
        new_store_slices[idx] = tuple(
            _Slice(s.start, s.size) for s in bref.compute_gmem_slice(indices)
        )
        are_same_slices = map(
            lambda old, new: old == new,
            last_store_slices[idx],
            new_store_slices[idx],
        )
        slices_changed = ~functools.reduce(lax.bitwise_and, are_same_slices)
        is_last_step = step == num_steps - 1
        # TODO(apaszke,slebedev): This still diverges significantly from the
        # TPU semantics in that it will move on to the next SMEM output slice
        # even if it's not storing the previous one.
        bref.copy_out(
            slot,
            indices,
            predicate=lax.bitwise_or(slices_changed, is_last_step),
        )

      fetch_step = step + max_concurrent_steps
      fetch_slot = slot  # (x + y) % y == x % y
      jax.lax.cond(
          fetch_step < num_steps,
          lambda: map(
              lambda bref: bref.copy_in(
                  fetch_slot,
                  _inc_grid(indices, grid, max_concurrent_steps),
                  barrier_ref,
              ),
              in_brefs,
          ),
          lambda: [None] * len(in_brefs),
      )

      return _inc_grid(indices, grid, 1), new_store_slices

    last_store_slices = [
        None
        if bref.is_index_invariant
        else (_Slice(-1, -1),) * len(bref.spec.block_shape)
        for bref in out_brefs
    ]
    lax.fori_loop(
        0, num_steps, loop_body, ((0,) * len(grid), last_store_slices)
    )

    # Outputs invariant to the sequential axis are never written from inside the
    # loop. This is the only place where we store them.
    if all(bref.is_index_invariant for bref in out_brefs):
      gpu_primitives.commit_smem()
    last_slot = (num_steps - 1) % max_concurrent_steps
    for bref in out_brefs:
      if bref.is_index_invariant:
        bref.copy_out(last_slot, (0,) * len(grid))

    # Finalize the pipeline.
    gpu_primitives.wait_smem_to_gmem(0)

  return pipeline
