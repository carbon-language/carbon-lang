# RUN: %PYTHON %s 2>&1 | FileCheck %s

import gc, sys
from mlir.ir import *
from mlir.passmanager import *
from mlir.execution_engine import *
from mlir.runtime import *


# Log everything to stderr and flush so that we have a unified stream to match
# errors/info emitted by MLIR to stderr.
def log(*args):
  print(*args, file=sys.stderr)
  sys.stderr.flush()


def run(f):
  log("\nTEST:", f.__name__)
  f()
  gc.collect()
  assert Context._get_live_count() == 0


# Verify capsule interop.
# CHECK-LABEL: TEST: testCapsule
def testCapsule():
  with Context():
    module = Module.parse(r"""
llvm.func @none() {
  llvm.return
}
    """)
    execution_engine = ExecutionEngine(module)
    execution_engine_capsule = execution_engine._CAPIPtr
    # CHECK: mlir.execution_engine.ExecutionEngine._CAPIPtr
    log(repr(execution_engine_capsule))
    execution_engine._testing_release()
    execution_engine1 = ExecutionEngine._CAPICreate(execution_engine_capsule)
    # CHECK: _mlirExecutionEngine.ExecutionEngine
    log(repr(execution_engine1))


run(testCapsule)


# Test invalid ExecutionEngine creation
# CHECK-LABEL: TEST: testInvalidModule
def testInvalidModule():
  with Context():
    # Builtin function
    module = Module.parse(r"""
    func @foo() { return }
    """)
    # CHECK: Got RuntimeError:  Failure while creating the ExecutionEngine.
    try:
      execution_engine = ExecutionEngine(module)
    except RuntimeError as e:
      log("Got RuntimeError: ", e)


run(testInvalidModule)


def lowerToLLVM(module):
  import mlir.conversions
  pm = PassManager.parse(
      "convert-memref-to-llvm,convert-std-to-llvm,reconcile-unrealized-casts")
  pm.run(module)
  return module


# Test simple ExecutionEngine execution
# CHECK-LABEL: TEST: testInvokeVoid
def testInvokeVoid():
  with Context():
    module = Module.parse(r"""
func @void() attributes { llvm.emit_c_interface } {
  return
}
    """)
    execution_engine = ExecutionEngine(lowerToLLVM(module))
    # Nothing to check other than no exception thrown here.
    execution_engine.invoke("void")


run(testInvokeVoid)


# Test argument passing and result with a simple float addition.
# CHECK-LABEL: TEST: testInvokeFloatAdd
def testInvokeFloatAdd():
  with Context():
    module = Module.parse(r"""
func @add(%arg0: f32, %arg1: f32) -> f32 attributes { llvm.emit_c_interface } {
  %add = arith.addf %arg0, %arg1 : f32
  return %add : f32
}
    """)
    execution_engine = ExecutionEngine(lowerToLLVM(module))
    # Prepare arguments: two input floats and one result.
    # Arguments must be passed as pointers.
    c_float_p = ctypes.c_float * 1
    arg0 = c_float_p(42.)
    arg1 = c_float_p(2.)
    res = c_float_p(-1.)
    execution_engine.invoke("add", arg0, arg1, res)
    # CHECK: 42.0 + 2.0 = 44.0
    log("{0} + {1} = {2}".format(arg0[0], arg1[0], res[0]))


run(testInvokeFloatAdd)


# Test callback
# CHECK-LABEL: TEST: testBasicCallback
def testBasicCallback():
  # Define a callback function that takes a float and an integer and returns a float.
  @ctypes.CFUNCTYPE(ctypes.c_float, ctypes.c_float, ctypes.c_int)
  def callback(a, b):
    return a / 2 + b / 2

  with Context():
    # The module just forwards to a runtime function known as "some_callback_into_python".
    module = Module.parse(r"""
func @add(%arg0: f32, %arg1: i32) -> f32 attributes { llvm.emit_c_interface } {
  %resf = call @some_callback_into_python(%arg0, %arg1) : (f32, i32) -> (f32)
  return %resf : f32
}
func private @some_callback_into_python(f32, i32) -> f32 attributes { llvm.emit_c_interface }
    """)
    execution_engine = ExecutionEngine(lowerToLLVM(module))
    execution_engine.register_runtime("some_callback_into_python", callback)

    # Prepare arguments: two input floats and one result.
    # Arguments must be passed as pointers.
    c_float_p = ctypes.c_float * 1
    c_int_p = ctypes.c_int * 1
    arg0 = c_float_p(42.)
    arg1 = c_int_p(2)
    res = c_float_p(-1.)
    execution_engine.invoke("add", arg0, arg1, res)
    # CHECK: 42.0 + 2 = 44.0
    log("{0} + {1} = {2}".format(arg0[0], arg1[0], res[0] * 2))


run(testBasicCallback)


# Test callback with an unranked memref
# CHECK-LABEL: TEST: testUnrankedMemRefCallback
def testUnrankedMemRefCallback():
  # Define a callback function that takes an unranked memref, converts it to a numpy array and prints it.
  @ctypes.CFUNCTYPE(None, ctypes.POINTER(UnrankedMemRefDescriptor))
  def callback(a):
    arr = unranked_memref_to_numpy(a, np.float32)
    log("Inside callback: ")
    log(arr)

  with Context():
    # The module just forwards to a runtime function known as "some_callback_into_python".
    module = Module.parse(r"""
func @callback_memref(%arg0: memref<*xf32>) attributes { llvm.emit_c_interface } {
  call @some_callback_into_python(%arg0) : (memref<*xf32>) -> ()
  return
}
func private @some_callback_into_python(memref<*xf32>) -> () attributes { llvm.emit_c_interface }
""")
    execution_engine = ExecutionEngine(lowerToLLVM(module))
    execution_engine.register_runtime("some_callback_into_python", callback)
    inp_arr = np.array([[1.0, 2.0], [3.0, 4.0]], np.float32)
    # CHECK: Inside callback:
    # CHECK{LITERAL}: [[1. 2.]
    # CHECK{LITERAL}:  [3. 4.]]
    execution_engine.invoke(
        "callback_memref",
        ctypes.pointer(ctypes.pointer(get_unranked_memref_descriptor(inp_arr))),
    )
    inp_arr_1 = np.array([5, 6, 7], dtype=np.float32)
    strided_arr = np.lib.stride_tricks.as_strided(
        inp_arr_1, strides=(4, 0), shape=(3, 4))
    # CHECK: Inside callback:
    # CHECK{LITERAL}: [[5. 5. 5. 5.]
    # CHECK{LITERAL}:  [6. 6. 6. 6.]
    # CHECK{LITERAL}:  [7. 7. 7. 7.]]
    execution_engine.invoke(
        "callback_memref",
        ctypes.pointer(
            ctypes.pointer(get_unranked_memref_descriptor(strided_arr))),
    )


run(testUnrankedMemRefCallback)


# Test callback with a ranked memref.
# CHECK-LABEL: TEST: testRankedMemRefCallback
def testRankedMemRefCallback():
  # Define a callback function that takes a ranked memref, converts it to a numpy array and prints it.
  @ctypes.CFUNCTYPE(
      None,
      ctypes.POINTER(
          make_nd_memref_descriptor(2,
                                    np.ctypeslib.as_ctypes_type(np.float32))),
  )
  def callback(a):
    arr = ranked_memref_to_numpy(a)
    log("Inside Callback: ")
    log(arr)

  with Context():
    # The module just forwards to a runtime function known as "some_callback_into_python".
    module = Module.parse(r"""
func @callback_memref(%arg0: memref<2x2xf32>) attributes { llvm.emit_c_interface } {
  call @some_callback_into_python(%arg0) : (memref<2x2xf32>) -> ()
  return
}
func private @some_callback_into_python(memref<2x2xf32>) -> () attributes { llvm.emit_c_interface }
""")
    execution_engine = ExecutionEngine(lowerToLLVM(module))
    execution_engine.register_runtime("some_callback_into_python", callback)
    inp_arr = np.array([[1.0, 5.0], [6.0, 7.0]], np.float32)
    # CHECK: Inside Callback:
    # CHECK{LITERAL}: [[1. 5.]
    # CHECK{LITERAL}:  [6. 7.]]
    execution_engine.invoke(
        "callback_memref",
        ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(inp_arr))))


run(testRankedMemRefCallback)


#  Test addition of two memrefs.
# CHECK-LABEL: TEST: testMemrefAdd
def testMemrefAdd():
  with Context():
    module = Module.parse("""
      module  {
      func @main(%arg0: memref<1xf32>, %arg1: memref<f32>, %arg2: memref<1xf32>) attributes { llvm.emit_c_interface } {
        %0 = arith.constant 0 : index
        %1 = memref.load %arg0[%0] : memref<1xf32>
        %2 = memref.load %arg1[] : memref<f32>
        %3 = arith.addf %1, %2 : f32
        memref.store %3, %arg2[%0] : memref<1xf32>
        return
      }
     } """)
    arg1 = np.array([32.5]).astype(np.float32)
    arg2 = np.array(6).astype(np.float32)
    res = np.array([0]).astype(np.float32)

    arg1_memref_ptr = ctypes.pointer(
        ctypes.pointer(get_ranked_memref_descriptor(arg1)))
    arg2_memref_ptr = ctypes.pointer(
        ctypes.pointer(get_ranked_memref_descriptor(arg2)))
    res_memref_ptr = ctypes.pointer(
        ctypes.pointer(get_ranked_memref_descriptor(res)))

    execution_engine = ExecutionEngine(lowerToLLVM(module))
    execution_engine.invoke("main", arg1_memref_ptr, arg2_memref_ptr,
                            res_memref_ptr)
    # CHECK: [32.5] + 6.0 = [38.5]
    log("{0} + {1} = {2}".format(arg1, arg2, res))


run(testMemrefAdd)


#  Test addition of two 2d_memref
# CHECK-LABEL: TEST: testDynamicMemrefAdd2D
def testDynamicMemrefAdd2D():
  with Context():
    module = Module.parse("""
      module  {
        func @memref_add_2d(%arg0: memref<2x2xf32>, %arg1: memref<?x?xf32>, %arg2: memref<2x2xf32>) attributes {llvm.emit_c_interface} {
          %c0 = arith.constant 0 : index
          %c2 = arith.constant 2 : index
          %c1 = arith.constant 1 : index
          br ^bb1(%c0 : index)
        ^bb1(%0: index):  // 2 preds: ^bb0, ^bb5
          %1 = arith.cmpi slt, %0, %c2 : index
          cond_br %1, ^bb2, ^bb6
        ^bb2:  // pred: ^bb1
          %c0_0 = arith.constant 0 : index
          %c2_1 = arith.constant 2 : index
          %c1_2 = arith.constant 1 : index
          br ^bb3(%c0_0 : index)
        ^bb3(%2: index):  // 2 preds: ^bb2, ^bb4
          %3 = arith.cmpi slt, %2, %c2_1 : index
          cond_br %3, ^bb4, ^bb5
        ^bb4:  // pred: ^bb3
          %4 = memref.load %arg0[%0, %2] : memref<2x2xf32>
          %5 = memref.load %arg1[%0, %2] : memref<?x?xf32>
          %6 = arith.addf %4, %5 : f32
          memref.store %6, %arg2[%0, %2] : memref<2x2xf32>
          %7 = arith.addi %2, %c1_2 : index
          br ^bb3(%7 : index)
        ^bb5:  // pred: ^bb3
          %8 = arith.addi %0, %c1 : index
          br ^bb1(%8 : index)
        ^bb6:  // pred: ^bb1
          return
        }
      }
        """)
    arg1 = np.random.randn(2, 2).astype(np.float32)
    arg2 = np.random.randn(2, 2).astype(np.float32)
    res = np.random.randn(2, 2).astype(np.float32)

    arg1_memref_ptr = ctypes.pointer(
        ctypes.pointer(get_ranked_memref_descriptor(arg1)))
    arg2_memref_ptr = ctypes.pointer(
        ctypes.pointer(get_ranked_memref_descriptor(arg2)))
    res_memref_ptr = ctypes.pointer(
        ctypes.pointer(get_ranked_memref_descriptor(res)))

    execution_engine = ExecutionEngine(lowerToLLVM(module))
    execution_engine.invoke("memref_add_2d", arg1_memref_ptr, arg2_memref_ptr,
                            res_memref_ptr)
    # CHECK: True
    log(np.allclose(arg1 + arg2, res))


run(testDynamicMemrefAdd2D)


#  Test loading of shared libraries.
# CHECK-LABEL: TEST: testSharedLibLoad
def testSharedLibLoad():
  with Context():
    module = Module.parse("""
      module  {
      func @main(%arg0: memref<1xf32>) attributes { llvm.emit_c_interface } {
        %c0 = arith.constant 0 : index
        %cst42 = arith.constant 42.0 : f32
        memref.store %cst42, %arg0[%c0] : memref<1xf32>
        %u_memref = memref.cast %arg0 : memref<1xf32> to memref<*xf32>
        call @print_memref_f32(%u_memref) : (memref<*xf32>) -> ()
        return
      }
      func private @print_memref_f32(memref<*xf32>) attributes { llvm.emit_c_interface }
     } """)
    arg0 = np.array([0.0]).astype(np.float32)

    arg0_memref_ptr = ctypes.pointer(
        ctypes.pointer(get_ranked_memref_descriptor(arg0)))

    execution_engine = ExecutionEngine(
        lowerToLLVM(module),
        opt_level=3,
        shared_libs=[
            "../../../../lib/libmlir_runner_utils.so",
            "../../../../lib/libmlir_c_runner_utils.so"
        ])
    execution_engine.invoke("main", arg0_memref_ptr)
    # CHECK: Unranked Memref
    # CHECK-NEXT: [42]


run(testSharedLibLoad)


#  Test that nano time clock is available.
# CHECK-LABEL: TEST: testNanoTime
def testNanoTime():
  with Context():
    module = Module.parse("""
      module {
      func @main() attributes { llvm.emit_c_interface } {
        %now = call @nano_time() : () -> i64
        %memref = memref.alloca() : memref<1xi64>
        %c0 = arith.constant 0 : index
        memref.store %now, %memref[%c0] : memref<1xi64>
        %u_memref = memref.cast %memref : memref<1xi64> to memref<*xi64>
        call @print_memref_i64(%u_memref) : (memref<*xi64>) -> ()
        return
      }
      func private @nano_time() -> i64 attributes { llvm.emit_c_interface }
      func private @print_memref_i64(memref<*xi64>) attributes { llvm.emit_c_interface }
    }""")

    execution_engine = ExecutionEngine(
        lowerToLLVM(module),
        opt_level=3,
        shared_libs=[
            "../../../../lib/libmlir_runner_utils.so",
            "../../../../lib/libmlir_c_runner_utils.so"
        ])
    execution_engine.invoke("main")
    # CHECK: Unranked Memref
    # CHECK: [{{.*}}]


run(testNanoTime)
