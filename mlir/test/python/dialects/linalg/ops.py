# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import builtin
from mlir.dialects import linalg
from mlir.dialects import std

def run(f):
  print("\nTEST:", f.__name__)
  f()
  return f


# CHECK-LABEL: TEST: testInitTensor
@run
def testInitTensor():
  with Context() as ctx, Location.unknown():
    module = Module.create()
    f32 = F32Type.get()
    with InsertionPoint(module.body):
      # CHECK-LABEL: func @static_sizes
      # CHECK: %0 = linalg.init_tensor [3, 4] : tensor<3x4xf32>
      @builtin.FuncOp.from_py_func()
      def static_sizes():
        return linalg.InitTensorOp([3, 4], f32)

      # CHECK-LABEL: func @dynamic_sizes
      # CHECK: %0 = linalg.init_tensor [%arg0, %arg1] : tensor<?x?xf32>
      @builtin.FuncOp.from_py_func(IndexType.get(), IndexType.get())
      def dynamic_sizes(d0, d1):
        return linalg.InitTensorOp([d0, d1], f32)

      # CHECK-LABEL: func @zero_d
      # CHECK: %0 = linalg.init_tensor [] : tensor<f32>
      @builtin.FuncOp.from_py_func()
      def zero_d():
        return linalg.InitTensorOp([], f32)

  print(module)

# CHECK-LABEL: TEST: testInitTensorStaticSizesAttribute
@run
def testInitTensorStaticSizesAttribute():
  with Context() as ctx, Location.unknown():
    module = Module.create()
    f32 = F32Type.get()
    with InsertionPoint(module.body):
      op = linalg.InitTensorOp([3, 4], f32)
      # CHECK: [3, 4]
      print(op.attributes['static_sizes'])

# CHECK-LABEL: TEST: testFill
@run
def testFill():
  with Context() as ctx, Location.unknown():
    module = Module.create()
    f32 = F32Type.get()
    with InsertionPoint(module.body):
      # CHECK-LABEL: func @fill_tensor
      #  CHECK-SAME:   %[[OUT:[0-9a-z]+]]: tensor<12x?xf32>
      #  CHECK-NEXT: %[[CST:.*]] = constant 0.0{{.*}} : f32
      #  CHECK-NEXT: %[[RES:.*]] = linalg.fill(%[[CST]], %[[OUT]]) : f32, tensor<12x?xf32> -> tensor<12x?xf32>
      #  CHECK-NEXT: return %[[RES]] : tensor<12x?xf32>
      @builtin.FuncOp.from_py_func(
          RankedTensorType.get((12, -1), f32))
      def fill_tensor(out):
        zero = std.ConstantOp(value=FloatAttr.get(f32, 0.), result=f32).result
        # TODO: FillOp.result is None. When len(results) == 1 we expect it to
        # be results[0] as per _linalg_ops_gen.py. This seems like an
        # orthogonal bug in the generator of _linalg_ops_gen.py.
        return linalg.FillOp(output=out, value=zero).results[0]

      # CHECK-LABEL: func @fill_buffer
      #  CHECK-SAME:   %[[OUT:[0-9a-z]+]]: memref<12x?xf32>
      #  CHECK-NEXT: %[[CST:.*]] = constant 0.0{{.*}} : f32
      #  CHECK-NEXT: linalg.fill(%[[CST]], %[[OUT]]) : f32, memref<12x?xf32>
      #  CHECK-NEXT: return
      @builtin.FuncOp.from_py_func(
          MemRefType.get((12, -1), f32))
      def fill_buffer(out):
        zero = std.ConstantOp(value=FloatAttr.get(f32, 0.), result=f32).result
        linalg.FillOp(output=out, value=zero)

  print(module)


# CHECK-LABEL: TEST: testStructuredOpOnTensors
@run
def testStructuredOpOnTensors():
  with Context() as ctx, Location.unknown():
    module = Module.create()
    f32 = F32Type.get()
    tensor_type = RankedTensorType.get((2, 3, 4), f32)
    with InsertionPoint(module.body):
      func = builtin.FuncOp(name="matmul_test",
                            type=FunctionType.get(
                                inputs=[tensor_type, tensor_type],
                                results=[tensor_type]))
      with InsertionPoint(func.add_entry_block()):
        lhs, rhs = func.entry_block.arguments
        result = linalg.MatmulOp([lhs, rhs], results=[tensor_type]).result
        std.ReturnOp([result])

  # CHECK: %[[R:.*]] = linalg.matmul ins(%arg0, %arg1 : tensor<2x3x4xf32>, tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  print(module)


# CHECK-LABEL: TEST: testStructuredOpOnBuffers
@run
def testStructuredOpOnBuffers():
  with Context() as ctx, Location.unknown():
    module = Module.create()
    f32 = F32Type.get()
    memref_type = MemRefType.get((2, 3, 4), f32)
    with InsertionPoint(module.body):
      func = builtin.FuncOp(name="matmul_test",
                            type=FunctionType.get(
                                inputs=[memref_type, memref_type, memref_type],
                                results=[]))
      with InsertionPoint(func.add_entry_block()):
        lhs, rhs, result = func.entry_block.arguments
        # TODO: prperly hook up the region.
        linalg.MatmulOp([lhs, rhs], outputs=[result])
        std.ReturnOp([])

  # CHECK: linalg.matmul ins(%arg0, %arg1 : memref<2x3x4xf32>, memref<2x3x4xf32>) outs(%arg2 : memref<2x3x4xf32>)
  print(module)

# CHECK-LABEL: TEST: testNamedStructuredOpCustomForm
@run
def testNamedStructuredOpCustomForm():
  with Context() as ctx, Location.unknown():
    module = Module.create()
    f32 = F32Type.get()
    with InsertionPoint(module.body):
      @builtin.FuncOp.from_py_func(RankedTensorType.get((4, 16), f32),
                                   RankedTensorType.get((16, 8), f32))
      def named_form(lhs, rhs):
        init_result = linalg.InitTensorOp([4, 8], f32)
        # First check the named form with custom format
        #      CHECK: linalg.matmul
        #  CHECK-NOT: linalg.memoized_indexing_maps
        # CHECK-SAME:    ins(%{{.*}} : tensor<4x16xf32>, tensor<16x8xf32>)
        # CHECK-SAME:   outs(%{{.*}} : tensor<4x8xf32>)
        # CHECK-SAME:   -> tensor<4x8xf32>
        # CHECK-NEXT: return
        return linalg.matmul(lhs, rhs, outs=[init_result.result])

  print(module)

# CHECK-LABEL: TEST: testNamedStructuredOpGenericForm
@run
def testNamedStructuredOpGenericForm():
  with Context() as ctx, Location.unknown():
    module = Module.create()
    f32 = F32Type.get()
    with InsertionPoint(module.body):
      @builtin.FuncOp.from_py_func(RankedTensorType.get((4, 16), f32),
                                   RankedTensorType.get((16, 8), f32))
      def named_form(lhs, rhs):
        init_result = linalg.InitTensorOp([4, 8], f32)
        #      CHECK: "linalg.matmul"(%{{.*}})
        # CHECK-NEXT:  ^bb0(%{{.*}}: f32, %{{.*}}: f32, %{{.*}}: f32):
        # CHECK-NEXT:    std.mulf{{.*}} (f32, f32) -> f32
        # CHECK-NEXT:    std.addf{{.*}} (f32, f32) -> f32
        # CHECK-NEXT:    linalg.yield{{.*}} (f32) -> ()
        # CHECK-NEXT:    {linalg.memoized_indexing_maps{{.*}}operand_segment_sizes = dense<[2, 1]> : vector<2xi32>} :
        # CHECK-SAME: (tensor<4x16xf32>, tensor<16x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
        return linalg.matmul(lhs, rhs, outs=[init_result.result])

  module.operation.print(print_generic_op_form=True)

# CHECK-LABEL: TEST: testNamedStructuredAsGenericOp
@run
def testNamedStructuredAsGenericOp():
  with Context() as ctx, Location.unknown():
    module = Module.create()
    f32 = F32Type.get()
    with InsertionPoint(module.body):
      @builtin.FuncOp.from_py_func(RankedTensorType.get((4, 16), f32),
                                   RankedTensorType.get((16, 8), f32))
      def generic_form(lhs, rhs):
        init_result = linalg.InitTensorOp([4, 8], f32)
        # CHECK: linalg.generic
        return linalg.matmul(lhs, rhs, outs=[init_result.result], emit_generic=True)

  print(module)
