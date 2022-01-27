# RUN: SUPPORT_LIB=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext \
# RUN:   %PYTHON %s | FileCheck %s

import ctypes
import errno
import itertools
import os
import sys
from typing import List, Callable

import numpy as np

import mlir.all_passes_registration

from mlir import ir
from mlir import runtime as rt
from mlir.execution_engine import ExecutionEngine
from mlir.passmanager import PassManager

from mlir.dialects import builtin
from mlir.dialects import std
from mlir.dialects import sparse_tensor as st

# ===----------------------------------------------------------------------=== #

# TODO: move this boilerplate to its own module, so it can be used by
# other tests and programs.
class TypeConverter:
  """Converter between NumPy types and MLIR types."""

  def __init__(self, context: ir.Context):
    # Note 1: these are numpy "scalar types" (i.e., the values of
    # np.sctypeDict) not numpy "dtypes" (i.e., the np.dtype class).
    #
    # Note 2: we must construct the MLIR types in the same context as the
    # types that'll be passed to irtype_to_sctype() or irtype_to_dtype();
    # otherwise, those methods will raise a KeyError.
    types_list = [
      (np.float64, ir.F64Type.get(context=context)),
      (np.float32, ir.F32Type.get(context=context)),
      (np.int64, ir.IntegerType.get_signless(64, context=context)),
      (np.int32, ir.IntegerType.get_signless(32, context=context)),
      (np.int16, ir.IntegerType.get_signless(16, context=context)),
      (np.int8, ir.IntegerType.get_signless(8, context=context)),
    ]
    self._sc2ir = dict(types_list)
    self._ir2sc = dict(( (ir,sc) for sc,ir in types_list ))

  def dtype_to_irtype(self, dtype: np.dtype) -> ir.Type:
    """Returns the MLIR equivalent of a NumPy dtype."""
    try:
      return self.sctype_to_irtype(dtype.type)
    except KeyError as e:
      raise KeyError(f'Unknown dtype: {dtype}') from e

  def sctype_to_irtype(self, sctype) -> ir.Type:
    """Returns the MLIR equivalent of a NumPy scalar type."""
    if sctype in self._sc2ir:
      return self._sc2ir[sctype]
    else:
      raise KeyError(f'Unknown sctype: {sctype}')

  def irtype_to_dtype(self, tp: ir.Type) -> np.dtype:
    """Returns the NumPy dtype equivalent of an MLIR type."""
    return np.dtype(self.irtype_to_sctype(tp))

  def irtype_to_sctype(self, tp: ir.Type):
    """Returns the NumPy scalar-type equivalent of an MLIR type."""
    if tp in self._ir2sc:
      return self._ir2sc[tp]
    else:
      raise KeyError(f'Unknown ir.Type: {tp}')

  def get_RankedTensorType_of_nparray(self, nparray: np.ndarray) -> ir.RankedTensorType:
    """Returns the ir.RankedTensorType of a NumPy array.  Note that NumPy
    arrays can only be converted to/from dense tensors, not sparse tensors."""
    # TODO: handle strides as well?
    return ir.RankedTensorType.get(nparray.shape,
                                   self.dtype_to_irtype(nparray.dtype))

# ===----------------------------------------------------------------------=== #

class StressTest:
  def __init__(self, tyconv: TypeConverter):
    self._tyconv = tyconv
    self._roundtripTp = None
    self._module = None
    self._engine = None

  def _assertEqualsRoundtripTp(self, tp: ir.RankedTensorType):
    assert self._roundtripTp is not None, \
        'StressTest: uninitialized roundtrip type'
    if tp != self._roundtripTp:
      raise AssertionError(
          f"Type is not equal to the roundtrip type.\n"
          f"\tExpected: {self._roundtripTp}\n"
          f"\tFound:    {tp}\n")

  def build(self, types: List[ir.Type]):
    """Builds the ir.Module.  The module has only the @main function,
    which will convert the input through the list of types and then back
    to the initial type.  The roundtrip type must be a dense tensor."""
    assert self._module is None, 'StressTest: must not call build() repeatedly'
    self._module = ir.Module.create()
    with ir.InsertionPoint(self._module.body):
      tp0 = types.pop(0)
      self._roundtripTp = tp0
      # TODO: assert dense? assert element type is recognised by the TypeConverter?
      types.append(tp0)
      funcTp = ir.FunctionType.get(inputs=[tp0], results=[tp0])
      funcOp = builtin.FuncOp(name='main', type=funcTp)
      funcOp.attributes['llvm.emit_c_interface'] = ir.UnitAttr.get()
      with ir.InsertionPoint(funcOp.add_entry_block()):
        arg0 = funcOp.entry_block.arguments[0]
        self._assertEqualsRoundtripTp(arg0.type)
        v = st.ConvertOp(types.pop(0), arg0)
        for tp in types:
          w = st.ConvertOp(tp, v)
          # Release intermediate tensors before they fall out of scope.
          st.ReleaseOp(v.result)
          v = w
        self._assertEqualsRoundtripTp(v.result.type)
        std.ReturnOp(v)
    return self

  def writeTo(self, filename):
    """Write the ir.Module to the given file.  If the file already exists,
    then raises an error.  If the filename is None, then is a no-op."""
    assert self._module is not None, \
        'StressTest: must call build() before writeTo()'
    if filename is None:
      # Silent no-op, for convenience.
      return self
    if os.path.exists(filename):
      raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), filename)
    with open(filename, 'w') as f:
      f.write(str(self._module))
    return self

  def compile(self, compiler: Callable[[ir.Module], ExecutionEngine]):
    """Compile the ir.Module."""
    assert self._module is not None, \
        'StressTest: must call build() before compile()'
    assert self._engine is None, \
        'StressTest: must not call compile() repeatedly'
    self._engine = compiler(self._module)
    return self

  def run(self, np_arg0: np.ndarray) -> np.ndarray:
    """Runs the test on the given numpy array, and returns the resulting
    numpy array."""
    assert self._engine is not None, \
        'StressTest: must call compile() before run()'
    self._assertEqualsRoundtripTp(
        self._tyconv.get_RankedTensorType_of_nparray(np_arg0))
    np_out = np.zeros(np_arg0.shape, dtype=np_arg0.dtype)
    self._assertEqualsRoundtripTp(
        self._tyconv.get_RankedTensorType_of_nparray(np_out))
    mem_arg0 = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(np_arg0)))
    mem_out = ctypes.pointer(ctypes.pointer(rt.get_ranked_memref_descriptor(np_out)))
    self._engine.invoke('main', mem_out, mem_arg0)
    return rt.ranked_memref_to_numpy(mem_out[0])

# ===----------------------------------------------------------------------=== #

# TODO: move this boilerplate to its own module, so it can be used by
# other tests and programs.
class SparseCompiler:
  """Sparse compiler passes."""

  def __init__(self, sparsification_options: str, support_lib: str):
    self._support_lib = support_lib
    self._pipeline = (
        f'builtin.func(linalg-generalize-named-ops,linalg-fuse-elementwise-ops),'
        f'sparsification{{{sparsification_options}}},'
        f'sparse-tensor-conversion,'
        f'builtin.func(linalg-bufferize,convert-linalg-to-loops,convert-vector-to-scf),'
        f'convert-scf-to-std,'
        f'func-bufferize,'
        f'tensor-constant-bufferize,'
        f'builtin.func(tensor-bufferize,std-bufferize,finalizing-bufferize),'
        f'convert-vector-to-llvm{{reassociate-fp-reductions=1 enable-index-optimizations=1}},'
        f'lower-affine,'
        f'convert-memref-to-llvm,'
        f'convert-std-to-llvm,'
        f'reconcile-unrealized-casts')
    # Must be in the scope of a `with ir.Context():`
    self._passmanager = PassManager.parse(self._pipeline)

  def __call__(self, module: ir.Module) -> ExecutionEngine:
    self._passmanager.run(module)
    return ExecutionEngine(module, opt_level=0, shared_libs=[self._support_lib])

# ===----------------------------------------------------------------------=== #

def main():
  """
  USAGE: python3 test_stress.py [raw_module.mlir [compiled_module.mlir]]

  The environment variable SUPPORT_LIB must be set to point to the
  libmlir_c_runner_utils shared library.  There are two optional
  arguments, for debugging purposes.  The first argument specifies where
  to write out the raw/generated ir.Module.  The second argument specifies
  where to write out the compiled version of that ir.Module.
  """
  support_lib = os.getenv('SUPPORT_LIB')
  assert support_lib is not None, 'SUPPORT_LIB is undefined'
  if not os.path.exists(support_lib):
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), support_lib)

  # CHECK-LABEL: TEST: test_stress
  print("\nTEST: test_stress")
  with ir.Context() as ctx, ir.Location.unknown():
    par = 0
    vec = 0
    vl = 1
    e = False
    sparsification_options = (
        f'parallelization-strategy={par} '
        f'vectorization-strategy={vec} '
        f'vl={vl} '
        f'enable-simd-index32={e}')
    compiler = SparseCompiler(sparsification_options, support_lib)
    f64 = ir.F64Type.get()
    # Be careful about increasing this because
    #     len(types) = 1 + 2^rank * rank! * len(bitwidths)^2
    shape = range(2, 6)
    rank = len(shape)
    # All combinations.
    levels = list(itertools.product(*itertools.repeat(
      [st.DimLevelType.dense, st.DimLevelType.compressed], rank)))
    # All permutations.
    orderings = list(map(ir.AffineMap.get_permutation,
      itertools.permutations(range(rank))))
    bitwidths = [0]
    # The first type must be a dense tensor for numpy conversion to work.
    types = [ir.RankedTensorType.get(shape, f64)]
    for level in levels:
      for ordering in orderings:
        for pwidth in bitwidths:
          for iwidth in bitwidths:
            attr = st.EncodingAttr.get(level, ordering, pwidth, iwidth)
            types.append(ir.RankedTensorType.get(shape, f64, attr))
    #
    # For exhaustiveness we should have one or more StressTest, such
    # that their paths cover all 2*n*(n-1) directed pairwise combinations
    # of the `types` set.  However, since n is already superexponential,
    # such exhaustiveness would be prohibitive for a test that runs on
    # every commit.  So for now we'll just pick one particular path that
    # at least hits all n elements of the `types` set.
    #
    tyconv = TypeConverter(ctx)
    size = 1
    for d in shape:
      size *= d
    np_arg0 = np.arange(size, dtype=tyconv.irtype_to_dtype(f64)).reshape(*shape)
    np_out = (
        StressTest(tyconv)
        .build(types)
        .writeTo(sys.argv[1] if len(sys.argv) > 1 else None)
        .compile(compiler)
        .writeTo(sys.argv[2] if len(sys.argv) > 2 else None)
        .run(np_arg0))
    # CHECK: Passed
    if np.allclose(np_out, np_arg0):
      print('Passed')
    else:
      sys.exit('FAILURE')

if __name__ == '__main__':
  main()
