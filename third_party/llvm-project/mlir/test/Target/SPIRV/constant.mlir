// RUN: mlir-translate -test-spirv-roundtrip %s | FileCheck %s

spv.module Logical GLSL450 requires #spv.vce<v1.0, [Shader], []> {
  // CHECK-LABEL: @bool_const
  spv.func @bool_const() -> () "None" {
    // CHECK: spv.Constant true
    %0 = spv.Constant true
    // CHECK: spv.Constant false
    %1 = spv.Constant false

    %2 = spv.Variable init(%0): !spv.ptr<i1, Function>
    %3 = spv.Variable init(%1): !spv.ptr<i1, Function>
    spv.Return
  }

  // CHECK-LABEL: @i32_const
  spv.func @i32_const() -> () "None" {
    // CHECK: spv.Constant 0 : i32
    %0 = spv.Constant  0 : i32
    // CHECK: spv.Constant 10 : i32
    %1 = spv.Constant 10 : i32
    // CHECK: spv.Constant -5 : i32
    %2 = spv.Constant -5 : i32

    %3 = spv.IAdd %0, %1 : i32
    %4 = spv.IAdd %2, %3 : i32
    spv.Return
  }

  // CHECK-LABEL: @si32_const
  spv.func @si32_const() -> () "None" {
    // CHECK: spv.Constant 0 : si32
    %0 = spv.Constant  0 : si32
    // CHECK: spv.Constant 10 : si32
    %1 = spv.Constant 10 : si32
    // CHECK: spv.Constant -5 : si32
    %2 = spv.Constant -5 : si32

    %3 = spv.IAdd %0, %1 : si32
    %4 = spv.IAdd %2, %3 : si32
    spv.Return
  }

  // CHECK-LABEL: @ui32_const
  // We cannot differentiate signless vs. unsigned integers in SPIR-V blob
  // because they all use 1 as the signedness bit. So we always treat them
  // as signless integers.
  spv.func @ui32_const() -> () "None" {
    // CHECK: spv.Constant 0 : i32
    %0 = spv.Constant  0 : ui32
    // CHECK: spv.Constant 10 : i32
    %1 = spv.Constant 10 : ui32
    // CHECK: spv.Constant -5 : i32
    %2 = spv.Constant 4294967291 : ui32

    %3 = spv.IAdd %0, %1 : ui32
    %4 = spv.IAdd %2, %3 : ui32
    spv.Return
  }

  // CHECK-LABEL: @i64_const
  spv.func @i64_const() -> () "None" {
    // CHECK: spv.Constant 4294967296 : i64
    %0 = spv.Constant           4294967296 : i64 //  2^32
    // CHECK: spv.Constant -4294967296 : i64
    %1 = spv.Constant          -4294967296 : i64 // -2^32
    // CHECK: spv.Constant 9223372036854775807 : i64
    %2 = spv.Constant  9223372036854775807 : i64 //  2^63 - 1
    // CHECK: spv.Constant -9223372036854775808 : i64
    %3 = spv.Constant -9223372036854775808 : i64 // -2^63

    %4 = spv.IAdd %0, %1 : i64
    %5 = spv.IAdd %2, %3 : i64
    spv.Return
  }

  // CHECK-LABEL: @i16_const
  spv.func @i16_const() -> () "None" {
    // CHECK: spv.Constant -32768 : i16
    %0 = spv.Constant -32768 : i16 // -2^15
    // CHECK: spv.Constant 32767 : i16
    %1 = spv.Constant 32767 : i16 //  2^15 - 1

    %2 = spv.IAdd %0, %1 : i16
    spv.Return
  }

  // CHECK-LABEL: @i8_const
  spv.func @i8_const() -> () "None" {
    // CHECK: spv.Constant 0 : i8
    %0 = spv.Constant 0 : i8
    // CHECK: spv.Constant -1 : i8
    %1 = spv.Constant 255 : i8

    // CHECK: spv.Constant 0 : si8
    %2 = spv.Constant 0 : si8
    // CHECK: spv.Constant 127 : si8
    %3 = spv.Constant 127 : si8
    // CHECK: spv.Constant -128 : si8
    %4 = spv.Constant -128 : si8

    // CHECK: spv.Constant 0 : i8
    %5 = spv.Constant 0 : ui8
    // CHECK: spv.Constant -1 : i8
    %6 = spv.Constant 255 : ui8

    %10 = spv.IAdd %0, %1: i8
    %11 = spv.IAdd %2, %3: si8
    %12 = spv.IAdd %3, %4: si8
    %13 = spv.IAdd %5, %6: ui8
    spv.Return
  }

  // CHECK-LABEL: @float_const
  spv.func @float_const() -> () "None" {
    // CHECK: spv.Constant 0.000000e+00 : f32
    %0 = spv.Constant 0. : f32
    // CHECK: spv.Constant 1.000000e+00 : f32
    %1 = spv.Constant 1. : f32
    // CHECK: spv.Constant -0.000000e+00 : f32
    %2 = spv.Constant -0. : f32
    // CHECK: spv.Constant -1.000000e+00 : f32
    %3 = spv.Constant -1. : f32
    // CHECK: spv.Constant 7.500000e-01 : f32
    %4 = spv.Constant 0.75 : f32
    // CHECK: spv.Constant -2.500000e-01 : f32
    %5 = spv.Constant -0.25 : f32

    %6 = spv.FAdd %0, %1 : f32
    %7 = spv.FAdd %2, %3 : f32
    %8 = spv.FAdd %4, %5 : f32
    spv.Return
  }

  // CHECK-LABEL: @double_const
  spv.func @double_const() -> () "None" {
    // TODO: test range boundary values
    // CHECK: spv.Constant 1.024000e+03 : f64
    %0 = spv.Constant 1024. : f64
    // CHECK: spv.Constant -1.024000e+03 : f64
    %1 = spv.Constant -1024. : f64

    %2 = spv.FAdd %0, %1 : f64
    spv.Return
  }

  // CHECK-LABEL: @half_const
  spv.func @half_const() -> () "None" {
    // CHECK: spv.Constant 5.120000e+02 : f16
    %0 = spv.Constant 512. : f16
    // CHECK: spv.Constant -5.120000e+02 : f16
    %1 = spv.Constant -512. : f16

    %2 = spv.FAdd %0, %1 : f16
    spv.Return
  }

  // CHECK-LABEL: @bool_vector_const
  spv.func @bool_vector_const() -> () "None" {
    // CHECK: spv.Constant dense<false> : vector<2xi1>
    %0 = spv.Constant dense<false> : vector<2xi1>
    // CHECK: spv.Constant dense<[true, true, true]> : vector<3xi1>
    %1 = spv.Constant dense<true> : vector<3xi1>
    // CHECK: spv.Constant dense<[false, true]> : vector<2xi1>
    %2 = spv.Constant dense<[false, true]> : vector<2xi1>

    %3 = spv.Variable init(%0): !spv.ptr<vector<2xi1>, Function>
    %4 = spv.Variable init(%1): !spv.ptr<vector<3xi1>, Function>
    %5 = spv.Variable init(%2): !spv.ptr<vector<2xi1>, Function>
    spv.Return
  }

  // CHECK-LABEL: @int_vector_const
  spv.func @int_vector_const() -> () "None" {
    // CHECK: spv.Constant dense<0> : vector<3xi32>
    %0 = spv.Constant dense<0> : vector<3xi32>
    // CHECK: spv.Constant dense<1> : vector<3xi32>
    %1 = spv.Constant dense<1> : vector<3xi32>
    // CHECK: spv.Constant dense<[2, -3, 4]> : vector<3xi32>
    %2 = spv.Constant dense<[2, -3, 4]> : vector<3xi32>

    %3 = spv.IAdd %0, %1 : vector<3xi32>
    %4 = spv.IAdd %2, %3 : vector<3xi32>
    spv.Return
  }

  // CHECK-LABEL: @fp_vector_const
  spv.func @fp_vector_const() -> () "None" {
    // CHECK: spv.Constant dense<0.000000e+00> : vector<4xf32>
    %0 = spv.Constant dense<0.> : vector<4xf32>
    // CHECK: spv.Constant dense<-1.500000e+01> : vector<4xf32>
    %1 = spv.Constant dense<-15.> : vector<4xf32>
    // CHECK: spv.Constant dense<[7.500000e-01, -2.500000e-01, 1.000000e+01, 4.200000e+01]> : vector<4xf32>
    %2 = spv.Constant dense<[0.75, -0.25, 10., 42.]> : vector<4xf32>

    %3 = spv.FAdd %0, %1 : vector<4xf32>
    %4 = spv.FAdd %2, %3 : vector<4xf32>
    spv.Return
  }

  // CHECK-LABEL: @ui64_array_const
  spv.func @ui64_array_const() -> (!spv.array<3xui64>) "None" {
    // CHECK: spv.Constant [5, 6, 7] : !spv.array<3 x i64>
    %0 = spv.Constant [5 : ui64, 6 : ui64, 7 : ui64] : !spv.array<3 x ui64>

    spv.ReturnValue %0: !spv.array<3xui64>
  }

  // CHECK-LABEL: @si32_array_const
  spv.func @si32_array_const() -> (!spv.array<3xsi32>) "None" {
    // CHECK: spv.Constant [5 : si32, 6 : si32, 7 : si32] : !spv.array<3 x si32>
    %0 = spv.Constant [5 : si32, 6 : si32, 7 : si32] : !spv.array<3 x si32>

    spv.ReturnValue %0 : !spv.array<3xsi32>
  }
  // CHECK-LABEL: @float_array_const
  spv.func @float_array_const() -> (!spv.array<2 x vector<2xf32>>) "None" {
    // CHECK: spv.Constant [dense<3.000000e+00> : vector<2xf32>, dense<[4.000000e+00, 5.000000e+00]> : vector<2xf32>] : !spv.array<2 x vector<2xf32>>
    %0 = spv.Constant [dense<3.0> : vector<2xf32>, dense<[4., 5.]> : vector<2xf32>] : !spv.array<2 x vector<2xf32>>

    spv.ReturnValue %0 : !spv.array<2 x vector<2xf32>>
  }

  // CHECK-LABEL: @ignore_not_used_const
  spv.func @ignore_not_used_const() -> () "None" {
    %0 = spv.Constant false
    // CHECK-NEXT: spv.Return
    spv.Return
  }

  // CHECK-LABEL: @materialize_const_at_each_use
  spv.func @materialize_const_at_each_use() -> (i32) "None" {
    // CHECK: %[[USE1:.*]] = spv.Constant 42 : i32
    // CHECK: %[[USE2:.*]] = spv.Constant 42 : i32
    // CHECK: spv.IAdd %[[USE1]], %[[USE2]]
    %0 = spv.Constant 42 : i32
    %1 = spv.IAdd %0, %0 : i32
    spv.ReturnValue %1 : i32
  }

  // CHECK-LABEL: @const_variable
  spv.func @const_variable(%arg0 : i32, %arg1 : i32) -> () "None" {
    // CHECK: %[[CONST:.*]] = spv.Constant 5 : i32
    // CHECK: spv.Variable init(%[[CONST]]) : !spv.ptr<i32, Function>
    // CHECK: spv.IAdd %arg0, %arg1
    %0 = spv.IAdd %arg0, %arg1 : i32
    %1 = spv.Constant 5 : i32
    %2 = spv.Variable init(%1) : !spv.ptr<i32, Function>
    %3 = spv.Load "Function" %2 : i32
    %4 = spv.IAdd %0, %3 : i32
    spv.Return
  }

  // CHECK-LABEL: @multi_dimensions_const
  spv.func @multi_dimensions_const() -> (!spv.array<2 x !spv.array<2 x !spv.array<3 x i32, stride=4>, stride=12>, stride=24>) "None" {
    // CHECK: spv.Constant {{\[}}{{\[}}[1 : i32, 2 : i32, 3 : i32], [4 : i32, 5 : i32, 6 : i32]], {{\[}}[7 : i32, 8 : i32, 9 : i32], [10 : i32, 11 : i32, 12 : i32]]] : !spv.array<2 x !spv.array<2 x !spv.array<3 x i32, stride=4>, stride=12>, stride=24>
    %0 = spv.Constant dense<[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]> : tensor<2x2x3xi32> : !spv.array<2 x !spv.array<2 x !spv.array<3 x i32, stride=4>, stride=12>, stride=24>
    spv.ReturnValue %0 : !spv.array<2 x !spv.array<2 x !spv.array<3 x i32, stride=4>, stride=12>, stride=24>
  }

  // CHECK-LABEL: @multi_dimensions_splat_const
  spv.func @multi_dimensions_splat_const() -> (!spv.array<2 x !spv.array<2 x !spv.array<3 x i32, stride=4>, stride=12>, stride=24>) "None" {
    // CHECK: spv.Constant {{\[}}{{\[}}[1 : i32, 1 : i32, 1 : i32], [1 : i32, 1 : i32, 1 : i32]], {{\[}}[1 : i32, 1 : i32, 1 : i32], [1 : i32, 1 : i32, 1 : i32]]] : !spv.array<2 x !spv.array<2 x !spv.array<3 x i32, stride=4>, stride=12>, stride=24>
    %0 = spv.Constant dense<1> : tensor<2x2x3xi32> : !spv.array<2 x !spv.array<2 x !spv.array<3 x i32, stride=4>, stride=12>, stride=24>
    spv.ReturnValue %0 : !spv.array<2 x !spv.array<2 x !spv.array<3 x i32, stride=4>, stride=12>, stride=24>
  }
}
