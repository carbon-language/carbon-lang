# RUN: %PYTHON %s 2>&1 | FileCheck %s

import ctypes
import sys
from mlir.ir import *
from mlir.dialects import builtin
from mlir.dialects import linalg
from mlir.dialects import std
from mlir.passmanager import *
from mlir.execution_engine import *


# Log everything to stderr and flush so that we have a unified stream to match
# errors/info emitted by MLIR to stderr.
def log(*args):
  print(*args, file=sys.stderr)
  sys.stderr.flush()


matmul_boiler = """
func @main() -> f32 attributes {llvm.emit_c_interface} {
  %v0 = constant 0.0 : f32
  %v1 = constant 1.0 : f32
  %v2 = constant 2.0 : f32

  %A = memref.alloc() : memref<4x16xf32>
  %B = memref.alloc() : memref<16x8xf32>
  %C = memref.alloc() : memref<4x8xf32>
  linalg.fill(%v1, %A) : f32, memref<4x16xf32>
  linalg.fill(%v2, %B) : f32, memref<16x8xf32>
  linalg.fill(%v0, %C) : f32, memref<4x8xf32>

  call @matmul_on_buffers(%A, %B, %C) :
    (memref<4x16xf32>, memref<16x8xf32>, memref<4x8xf32>) -> ()

  %c0 = constant 0 : index
  %0 = memref.load %C[%c0, %c0] : memref<4x8xf32>

  // TODO: FFI-based solution to allow testing and printing with python code.
  return %0 : f32
}
"""

fill_boiler = """
func @main() -> i32 attributes {llvm.emit_c_interface} {
  %O = memref.alloc() : memref<4x16xi32>
  %min = constant -1000.0 : f64
  %max = constant 1000.0 : f64
  %seed = constant 42 : i32

  call @fill_on_buffers(%min, %max, %seed, %O) :
    (f64, f64, i32, memref<4x16xi32>) -> ()

  %c0 = constant 0 : index
  %0 = memref.load %O[%c0, %c0] : memref<4x16xi32>

  // TODO: FFI-based solution to allow testing and printing with python code.
  return %0 : i32
}
"""

conv_boiler = """
func @main() -> i32 attributes {llvm.emit_c_interface} {
  %v0 = constant 0 : i32
  %v1 = constant 1.0 : f64
  %v2 = constant 2.0 : f64

  %input = memref.alloc() : memref<1x4x16x1xf64>
  %filter = memref.alloc() : memref<2x2x1xf64>
  %output = memref.alloc() : memref<1x2x4x1xi32>
  linalg.fill(%v1, %input) : f64, memref<1x4x16x1xf64>
  linalg.fill(%v2, %filter) : f64, memref<2x2x1xf64>
  linalg.fill(%v0, %output) : i32, memref<1x2x4x1xi32>

  call @conv_on_buffers(%input, %filter, %output) :
    (memref<1x4x16x1xf64>, memref<2x2x1xf64>, memref<1x2x4x1xi32>) -> ()

  %c0 = constant 0 : index
  %0 = memref.load %output[%c0, %c0, %c0, %c0] : memref<1x2x4x1xi32>

  // TODO: FFI-based solution to allow testing and printing with python code.
  return %0 : i32
}
"""

pooling_boiler = """
func @main() -> i32 attributes {llvm.emit_c_interface} {
  %v0 = constant 0 : i32
  %v42 = constant 42.0 : f64
  %v77 = constant 77.0 : f64
  %v-13 = constant -13.0 : f64
  %v1 = constant 1.0 : f64

  %input = memref.alloc() : memref<1x4x16x1xf64>
  %shape = memref.alloc() : memref<2x2xf64>
  %output = memref.alloc() : memref<1x2x4x1xi32>
  linalg.fill(%v1, %input) : f64, memref<1x4x16x1xf64>
  linalg.fill(%v1, %shape) : f64, memref<2x2xf64>
  linalg.fill(%v0, %output) : i32, memref<1x2x4x1xi32>

  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  memref.store %v42, %input[%c0, %c0, %c0, %c0] : memref<1x4x16x1xf64>
  memref.store %v77, %input[%c0, %c0, %c1, %c0] : memref<1x4x16x1xf64>
  memref.store %v-13, %input[%c0, %c0, %c2, %c0] : memref<1x4x16x1xf64>

  call @pooling_on_buffers(%input, %shape, %output) :
    (memref<1x4x16x1xf64>, memref<2x2xf64>, memref<1x2x4x1xi32>) -> ()

  %0 = memref.load %output[%c0, %c0, %c0, %c0] : memref<1x2x4x1xi32>

  // TODO: FFI-based solution to allow testing and printing with python code.
  return %0 : i32
}
"""


def transform(module, boilerplate):
  import mlir.conversions
  import mlir.dialects.linalg.passes
  import mlir.transforms

  # TODO: Allow cloning functions from one module to another.
  # Atm we have to resort to string concatenation.
  mod = Module.parse(
      str(module.operation.regions[0].blocks[0].operations[0].operation) +
      boilerplate)
  pm = PassManager.parse(
      "builtin.func(convert-linalg-to-loops, lower-affine, " +
      "convert-scf-to-std), convert-vector-to-llvm," +
      "convert-memref-to-llvm,convert-std-to-llvm")
  pm.run(mod)
  return mod


def test_matmul_builtin():
  with Context() as ctx, Location.unknown():
    module = Module.create()
    f32 = F32Type.get()
    with InsertionPoint(module.body):

      @builtin.FuncOp.from_py_func(
          MemRefType.get((4, 16), f32), MemRefType.get((16, 8), f32),
          MemRefType.get((4, 8), f32))
      def matmul_on_buffers(lhs, rhs, out):
        linalg.matmul(lhs, rhs, outs=[out])

    execution_engine = ExecutionEngine(transform(module, matmul_boiler))

    # TODO: FFI-based solution to allow testing and printing with python code.
    # Prepare arguments: one result f32.
    # Arguments must be passed as pointers.
    c_float_p = ctypes.c_float * 1
    res = c_float_p(-1.)
    execution_engine.invoke("main", res)

    log("RESULT: ", res[0])
    # CHECK: RESULT: 32.0


test_matmul_builtin()


def test_matmul_generic():
  with Context() as ctx, Location.unknown():
    module = Module.create()
    f32 = F32Type.get()
    with InsertionPoint(module.body):

      @builtin.FuncOp.from_py_func(
          MemRefType.get((4, 16), f32), MemRefType.get((16, 8), f32),
          MemRefType.get((4, 8), f32))
      def matmul_on_buffers(lhs, rhs, out):
        linalg.matmul(lhs, rhs, outs=[out], emit_generic=True)

    execution_engine = ExecutionEngine(transform(module, matmul_boiler))

    # TODO: FFI-based solution to allow testing and printing with python code.
    # Prepare arguments: one result f32.
    # Arguments must be passed as pointers.
    c_float_p = ctypes.c_float * 1
    res = c_float_p(-1.)
    execution_engine.invoke("main", res)

    log("RESULT: ", res[0])
    # CHECK: RESULT: 32.0


test_matmul_generic()


def test_fill_builtin():
  with Context() as ctx, Location.unknown():
    module = Module.create()
    f64 = F64Type.get()
    i32 = IntegerType.get_signless(32)
    with InsertionPoint(module.body):

      @builtin.FuncOp.from_py_func(f64, f64, i32, MemRefType.get((4, 16), i32))
      def fill_on_buffers(min, max, seed, out):
        linalg.fill_rng_2d(min, max, seed, outs=[out])

    execution_engine = ExecutionEngine(transform(module, fill_boiler))

    # TODO: FFI-based solution to allow testing and printing with python code.
    # Prepare arguments: one result i32.
    # Arguments must be passed as pointers.
    c_int_p = ctypes.c_int * 1
    res = c_int_p(-1)
    execution_engine.invoke("main", res)

    log("RESULT: ", res[0])
    # CHECK: RESULT: -480


test_fill_builtin()


def test_fill_generic():
  with Context() as ctx, Location.unknown():
    module = Module.create()
    f64 = F64Type.get()
    i32 = IntegerType.get_signless(32)
    with InsertionPoint(module.body):

      @builtin.FuncOp.from_py_func(f64, f64, i32, MemRefType.get((4, 16), i32))
      def fill_on_buffers(min, max, seed, out):
        linalg.fill_rng_2d(min, max, seed, outs=[out], emit_generic=True)

    execution_engine = ExecutionEngine(transform(module, fill_boiler))

    # TODO: FFI-based solution to allow testing and printing with python code.
    # Prepare arguments: one result i32.
    # Arguments must be passed as pointers.
    c_int_p = ctypes.c_int * 1
    res = c_int_p(-1)
    execution_engine.invoke("main", res)

    log("RESULT: ", res[0])
    # CHECK: RESULT: -480


test_fill_generic()


def test_max_pooling_builtin():
  with Context() as ctx, Location.unknown():
    module = Module.create()
    f64 = F64Type.get()
    i32 = IntegerType.get_signless(32)
    with InsertionPoint(module.body):

      @builtin.FuncOp.from_py_func(
          MemRefType.get((1, 4, 16, 1), f64), MemRefType.get((2, 2), f64),
          MemRefType.get((1, 2, 4, 1), i32))
      def pooling_on_buffers(input, shape, output):
        linalg.pooling_nhwc_max(
            input, shape, outs=[output], strides=[2, 4], dilations=[1, 2])

    execution_engine = ExecutionEngine(transform(module, pooling_boiler))

    # TODO: FFI-based solution to allow testing and printing with python code.
    # Prepare arguments: one result i32.
    # Arguments must be passed as pointers.
    c_int_p = ctypes.c_int * 1
    res = c_int_p(-1)
    execution_engine.invoke("main", res)

    log("RESULT: ", res[0])
    # 77 is not selected due to the dilation 2 in the second dimension.
    # CHECK: RESULT: 42


test_max_pooling_builtin()


def test_max_pooling_generic():
  with Context() as ctx, Location.unknown():
    module = Module.create()
    f64 = F64Type.get()
    i32 = IntegerType.get_signless(32)
    with InsertionPoint(module.body):

      @builtin.FuncOp.from_py_func(
          MemRefType.get((1, 4, 16, 1), f64), MemRefType.get((2, 2), f64),
          MemRefType.get((1, 2, 4, 1), i32))
      def pooling_on_buffers(input, shape, output):
        linalg.pooling_nhwc_max(
            input,
            shape,
            outs=[output],
            strides=[2, 4],
            dilations=[1, 2],
            emit_generic=True)

    execution_engine = ExecutionEngine(transform(module, pooling_boiler))

    # TODO: FFI-based solution to allow testing and printing with python code.
    # Prepare arguments: one result i32.
    # Arguments must be passed as pointers.
    c_int_p = ctypes.c_int * 1
    res = c_int_p(-1)
    execution_engine.invoke("main", res)

    log("RESULT: ", res[0])
    # 77 is not selected due to the dilation 2 in the second dimension.
    # CHECK: RESULT: 42


test_max_pooling_generic()


def test_min_pooling_builtin():
  with Context() as ctx, Location.unknown():
    module = Module.create()
    f64 = F64Type.get()
    i32 = IntegerType.get_signless(32)
    with InsertionPoint(module.body):

      @builtin.FuncOp.from_py_func(
          MemRefType.get((1, 4, 16, 1), f64), MemRefType.get((2, 2), f64),
          MemRefType.get((1, 2, 4, 1), i32))
      def pooling_on_buffers(input, shape, output):
        linalg.pooling_nhwc_min(
            input, shape, outs=[output], strides=[2, 4], dilations=[1, 2])

    execution_engine = ExecutionEngine(transform(module, pooling_boiler))

    # TODO: FFI-based solution to allow testing and printing with python code.
    # Prepare arguments: one result i32.
    # Arguments must be passed as pointers.
    c_int_p = ctypes.c_int * 1
    res = c_int_p(-1)
    execution_engine.invoke("main", res)

    log("RESULT: ", res[0])
    # CHECK: RESULT: -13


test_min_pooling_builtin()


def test_min_pooling_generic():
  with Context() as ctx, Location.unknown():
    module = Module.create()
    f64 = F64Type.get()
    i32 = IntegerType.get_signless(32)
    with InsertionPoint(module.body):

      @builtin.FuncOp.from_py_func(
          MemRefType.get((1, 4, 16, 1), f64), MemRefType.get((2, 2), f64),
          MemRefType.get((1, 2, 4, 1), i32))
      def pooling_on_buffers(input, shape, output):
        linalg.pooling_nhwc_min(
            input,
            shape,
            outs=[output],
            strides=[2, 4],
            dilations=[1, 2],
            emit_generic=True)

    execution_engine = ExecutionEngine(transform(module, pooling_boiler))

    # TODO: FFI-based solution to allow testing and printing with python code.
    # Prepare arguments: one result i32.
    # Arguments must be passed as pointers.
    c_int_p = ctypes.c_int * 1
    res = c_int_p(-1)
    execution_engine.invoke("main", res)

    log("RESULT: ", res[0])
    # CHECK: RESULT: -13


test_min_pooling_generic()
