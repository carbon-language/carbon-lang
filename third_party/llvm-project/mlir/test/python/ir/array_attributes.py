# RUN: %PYTHON %s | FileCheck %s
# Note that this is separate from ir_attributes.py since it depends on numpy,
# and we may want to disable if not available.

import gc
from mlir.ir import *
import numpy as np

def run(f):
  print("\nTEST:", f.__name__)
  f()
  gc.collect()
  assert Context._get_live_count() == 0
  return f

################################################################################
# Tests of the array/buffer .get() factory method on unsupported dtype.
################################################################################

@run
def testGetDenseElementsUnsupported():
  with Context():
    array = np.array([["hello", "goodbye"]])
    try:
      attr = DenseElementsAttr.get(array)
    except ValueError as e:
      # CHECK: unimplemented array format conversion from format:
      print(e)

################################################################################
# Splats.
################################################################################

# CHECK-LABEL: TEST: testGetDenseElementsSplatInt
@run
def testGetDenseElementsSplatInt():
  with Context(), Location.unknown():
    t = IntegerType.get_signless(32)
    element = IntegerAttr.get(t, 555)
    shaped_type = RankedTensorType.get((2, 3, 4), t)
    attr = DenseElementsAttr.get_splat(shaped_type, element)
    # CHECK: dense<555> : tensor<2x3x4xi32>
    print(attr)
    # CHECK: is_splat: True
    print("is_splat:", attr.is_splat)


# CHECK-LABEL: TEST: testGetDenseElementsSplatFloat
@run
def testGetDenseElementsSplatFloat():
  with Context(), Location.unknown():
    t = F32Type.get()
    element = FloatAttr.get(t, 1.2)
    shaped_type = RankedTensorType.get((2, 3, 4), t)
    attr = DenseElementsAttr.get_splat(shaped_type, element)
    # CHECK: dense<1.200000e+00> : tensor<2x3x4xf32>
    print(attr)


# CHECK-LABEL: TEST: testGetDenseElementsSplatErrors
@run
def testGetDenseElementsSplatErrors():
  with Context(), Location.unknown():
    t = F32Type.get()
    other_t = F64Type.get()
    element = FloatAttr.get(t, 1.2)
    other_element = FloatAttr.get(other_t, 1.2)
    shaped_type = RankedTensorType.get((2, 3, 4), t)
    dynamic_shaped_type = UnrankedTensorType.get(t)
    non_shaped_type = t

    try:
      attr = DenseElementsAttr.get_splat(non_shaped_type, element)
    except ValueError as e:
      # CHECK: Expected a static ShapedType for the shaped_type parameter: Type(f32)
      print(e)

    try:
      attr = DenseElementsAttr.get_splat(dynamic_shaped_type, element)
    except ValueError as e:
      # CHECK: Expected a static ShapedType for the shaped_type parameter: Type(tensor<*xf32>)
      print(e)

    try:
      attr = DenseElementsAttr.get_splat(shaped_type, other_element)
    except ValueError as e:
      # CHECK: Shaped element type and attribute type must be equal: shaped=Type(tensor<2x3x4xf32>), element=Attribute(1.200000e+00 : f64)
      print(e)


# CHECK-LABEL: TEST: testRepeatedValuesSplat
@run
def testRepeatedValuesSplat():
  with Context():
    array = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    attr = DenseElementsAttr.get(array)
    # CHECK: dense<1.000000e+00> : tensor<2x3xf32>
    print(attr)
    # CHECK: is_splat: True
    print("is_splat:", attr.is_splat)
    # TODO: Re-enable this once a solution is found to raising an exception
    # from buffer protocol.
    # Reported as https://github.com/pybind/pybind11/issues/3336
    # print(np.array(attr))


# CHECK-LABEL: TEST: testNonSplat
@run
def testNonSplat():
  with Context():
    array = np.array([2.0, 1.0, 1.0], dtype=np.float32)
    attr = DenseElementsAttr.get(array)
    # CHECK: is_splat: False
    print("is_splat:", attr.is_splat)


################################################################################
# Tests of the array/buffer .get() factory method, in all of its permutations.
################################################################################

### explicitly provided types

@run
def testGetDenseElementsBF16():
  with Context():
    array = np.array([[2, 4, 8], [16, 32, 64]], dtype=np.uint16)
    attr = DenseElementsAttr.get(array, type=BF16Type.get())
    # Note: These values don't mean much since just bit-casting. But they
    # shouldn't change.
    # CHECK: dense<{{\[}}[1.836710e-40, 3.673420e-40, 7.346840e-40], [1.469370e-39, 2.938740e-39, 5.877470e-39]]> : tensor<2x3xbf16>
    print(attr)

@run
def testGetDenseElementsInteger4():
  with Context():
    array = np.array([[2, 4, 7], [-2, -4, -8]], dtype=np.uint8)
    attr = DenseElementsAttr.get(array, type=IntegerType.get_signless(4))
    # Note: These values don't mean much since just bit-casting. But they
    # shouldn't change.
    # CHECK: dense<{{\[}}[2, 4, 7], [-2, -4, -8]]> : tensor<2x3xi4>
    print(attr)


@run
def testGetDenseElementsBool():
  with Context():
    bool_array = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.bool_)
    array = np.packbits(bool_array, axis=None, bitorder="little")
    attr = DenseElementsAttr.get(
        array, type=IntegerType.get_signless(1), shape=bool_array.shape)
    # CHECK: dense<{{\[}}[true, false, true], [false, true, false]]> : tensor<2x3xi1>
    print(attr)


@run
def testGetDenseElementsBoolSplat():
  with Context():
    zero = np.array(0, dtype=np.uint8)
    one = np.array(255, dtype=np.uint8)
    print(one)
    # CHECK: dense<false> : tensor<4x2x5xi1>
    print(DenseElementsAttr.get(
        zero, type=IntegerType.get_signless(1), shape=(4, 2, 5)))
    # CHECK: dense<true> : tensor<4x2x5xi1>
    print(DenseElementsAttr.get(
        one, type=IntegerType.get_signless(1), shape=(4, 2, 5)))


### float and double arrays.

# CHECK-LABEL: TEST: testGetDenseElementsF16
@run
def testGetDenseElementsF16():
  with Context():
    array = np.array([[2.0, 4.0, 8.0], [16.0, 32.0, 64.0]], dtype=np.float16)
    attr = DenseElementsAttr.get(array)
    # CHECK: dense<{{\[}}[2.000000e+00, 4.000000e+00, 8.000000e+00], [1.600000e+01, 3.200000e+01, 6.400000e+01]]> : tensor<2x3xf16>
    print(attr)
    # CHECK: {{\[}}[ 2. 4. 8.]
    # CHECK: {{\[}}16. 32. 64.]]
    print(np.array(attr))


# CHECK-LABEL: TEST: testGetDenseElementsF32
@run
def testGetDenseElementsF32():
  with Context():
    array = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]], dtype=np.float32)
    attr = DenseElementsAttr.get(array)
    # CHECK: dense<{{\[}}[1.100000e+00, 2.200000e+00, 3.300000e+00], [4.400000e+00, 5.500000e+00, 6.600000e+00]]> : tensor<2x3xf32>
    print(attr)
    # CHECK: {{\[}}[1.1 2.2 3.3]
    # CHECK: {{\[}}4.4 5.5 6.6]]
    print(np.array(attr))


# CHECK-LABEL: TEST: testGetDenseElementsF64
@run
def testGetDenseElementsF64():
  with Context():
    array = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]], dtype=np.float64)
    attr = DenseElementsAttr.get(array)
    # CHECK: dense<{{\[}}[1.100000e+00, 2.200000e+00, 3.300000e+00], [4.400000e+00, 5.500000e+00, 6.600000e+00]]> : tensor<2x3xf64>
    print(attr)
    # CHECK: {{\[}}[1.1 2.2 3.3]
    # CHECK: {{\[}}4.4 5.5 6.6]]
    print(np.array(attr))


### 16 bit integer arrays
# CHECK-LABEL: TEST: testGetDenseElementsI16Signless
@run
def testGetDenseElementsI16Signless():
  with Context():
    array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)
    attr = DenseElementsAttr.get(array)
    # CHECK: dense<{{\[}}[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi16>
    print(attr)
    # CHECK: {{\[}}[1 2 3]
    # CHECK: {{\[}}4 5 6]]
    print(np.array(attr))


# CHECK-LABEL: TEST: testGetDenseElementsUI16Signless
@run
def testGetDenseElementsUI16Signless():
  with Context():
    array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)
    attr = DenseElementsAttr.get(array)
    # CHECK: dense<{{\[}}[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi16>
    print(attr)
    # CHECK: {{\[}}[1 2 3]
    # CHECK: {{\[}}4 5 6]]
    print(np.array(attr))


# CHECK-LABEL: TEST: testGetDenseElementsI16
@run
def testGetDenseElementsI16():
  with Context():
    array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)
    attr = DenseElementsAttr.get(array, signless=False)
    # CHECK: dense<{{\[}}[1, 2, 3], [4, 5, 6]]> : tensor<2x3xsi16>
    print(attr)
    # CHECK: {{\[}}[1 2 3]
    # CHECK: {{\[}}4 5 6]]
    print(np.array(attr))


# CHECK-LABEL: TEST: testGetDenseElementsUI16
@run
def testGetDenseElementsUI16():
  with Context():
    array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)
    attr = DenseElementsAttr.get(array, signless=False)
    # CHECK: dense<{{\[}}[1, 2, 3], [4, 5, 6]]> : tensor<2x3xui16>
    print(attr)
    # CHECK: {{\[}}[1 2 3]
    # CHECK: {{\[}}4 5 6]]
    print(np.array(attr))

### 32 bit integer arrays
# CHECK-LABEL: TEST: testGetDenseElementsI32Signless
@run
def testGetDenseElementsI32Signless():
  with Context():
    array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    attr = DenseElementsAttr.get(array)
    # CHECK: dense<{{\[}}[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
    print(attr)
    # CHECK: {{\[}}[1 2 3]
    # CHECK: {{\[}}4 5 6]]
    print(np.array(attr))


# CHECK-LABEL: TEST: testGetDenseElementsUI32Signless
@run
def testGetDenseElementsUI32Signless():
  with Context():
    array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint32)
    attr = DenseElementsAttr.get(array)
    # CHECK: dense<{{\[}}[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
    print(attr)
    # CHECK: {{\[}}[1 2 3]
    # CHECK: {{\[}}4 5 6]]
    print(np.array(attr))


# CHECK-LABEL: TEST: testGetDenseElementsI32
@run
def testGetDenseElementsI32():
  with Context():
    array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    attr = DenseElementsAttr.get(array, signless=False)
    # CHECK: dense<{{\[}}[1, 2, 3], [4, 5, 6]]> : tensor<2x3xsi32>
    print(attr)
    # CHECK: {{\[}}[1 2 3]
    # CHECK: {{\[}}4 5 6]]
    print(np.array(attr))


# CHECK-LABEL: TEST: testGetDenseElementsUI32
@run
def testGetDenseElementsUI32():
  with Context():
    array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint32)
    attr = DenseElementsAttr.get(array, signless=False)
    # CHECK: dense<{{\[}}[1, 2, 3], [4, 5, 6]]> : tensor<2x3xui32>
    print(attr)
    # CHECK: {{\[}}[1 2 3]
    # CHECK: {{\[}}4 5 6]]
    print(np.array(attr))


## 64bit integer arrays
# CHECK-LABEL: TEST: testGetDenseElementsI64Signless
@run
def testGetDenseElementsI64Signless():
  with Context():
    array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    attr = DenseElementsAttr.get(array)
    # CHECK: dense<{{\[}}[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi64>
    print(attr)
    # CHECK: {{\[}}[1 2 3]
    # CHECK: {{\[}}4 5 6]]
    print(np.array(attr))


# CHECK-LABEL: TEST: testGetDenseElementsUI64Signless
@run
def testGetDenseElementsUI64Signless():
  with Context():
    array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint64)
    attr = DenseElementsAttr.get(array)
    # CHECK: dense<{{\[}}[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi64>
    print(attr)
    # CHECK: {{\[}}[1 2 3]
    # CHECK: {{\[}}4 5 6]]
    print(np.array(attr))


# CHECK-LABEL: TEST: testGetDenseElementsI64
@run
def testGetDenseElementsI64():
  with Context():
    array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    attr = DenseElementsAttr.get(array, signless=False)
    # CHECK: dense<{{\[}}[1, 2, 3], [4, 5, 6]]> : tensor<2x3xsi64>
    print(attr)
    # CHECK: {{\[}}[1 2 3]
    # CHECK: {{\[}}4 5 6]]
    print(np.array(attr))


# CHECK-LABEL: TEST: testGetDenseElementsUI64
@run
def testGetDenseElementsUI64():
  with Context():
    array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint64)
    attr = DenseElementsAttr.get(array, signless=False)
    # CHECK: dense<{{\[}}[1, 2, 3], [4, 5, 6]]> : tensor<2x3xui64>
    print(attr)
    # CHECK: {{\[}}[1 2 3]
    # CHECK: {{\[}}4 5 6]]
    print(np.array(attr))

