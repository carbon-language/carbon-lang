// RUN: mlir-opt --split-input-file --canonicalize %s | FileCheck %s

// CHECK-LABEL: @argmax_nofold
func @argmax_nofold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: "tosa.argmax"
  %0 = "tosa.argmax"(%arg0) {axis = 0 : i64}: (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// -----

// CHECK-LABEL: @add_zero_different_shape
func @add_zero_different_shape(%arg0: tensor<2x3xi32>) -> tensor<4x2x3xi32> {
  // CHECK: tosa.add
  %zeros = "tosa.const"() {value = dense<0> : tensor<4x2x3xi32>} : () -> tensor<4x2x3xi32>
  %1 = "tosa.add"(%arg0, %zeros) : (tensor<2x3xi32>, tensor<4x2x3xi32>) -> tensor<4x2x3xi32>
  return %1 : tensor<4x2x3xi32>
}


// -----

// CHECK-LABEL: @add_zero_int
func @add_zero_int(%arg0: tensor<2x3xi32>) -> tensor<2x3xi32> {
  // CHECK: return %arg0
  // CHECK-NOT: tosa.add
  %zeros = "tosa.const"() {value = dense<0> : tensor<2x3xi32>} : () -> tensor<2x3xi32>
  %1 = "tosa.add"(%arg0, %zeros) : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  return %1 : tensor<2x3xi32>
}

// -----

// CHECK-LABEL: @cast_fold
func @cast_fold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: return %arg0
  %0 = "tosa.cast"(%arg0) : (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// -----

// CHECK-LABEL: @cast_nofold
func @cast_nofold(%arg0: tensor<?x1xf32>) -> tensor<?x1xi32> {
  // CHECK: "tosa.cast"
  %0 = "tosa.cast"(%arg0) : (tensor<?x1xf32>) -> tensor<?x1xi32>
  return %0 : tensor<?x1xi32>
}

// -----

// CHECK-LABEL: @concat_fold
func @concat_fold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: return %arg0
  %0 = "tosa.concat"(%arg0) {axis = 0 : i64}: (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// -----

// CHECK-LABEL: @concat_fold_cast
func @concat_fold_cast(%arg0: tensor<?x1xf32>) -> tensor<?x?xf32> {
  // CHECK: %[[VAR0:.*]] = tensor.cast %arg0
  // CHECK: return %[[VAR0]]
  %0 = "tosa.concat"(%arg0) {axis = 0 : i64}: (tensor<?x1xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @conv2d_stride_2
func @conv2d_stride_2(%arg0: tensor<4x10x10x2xf32>) -> tensor<4x10x10x3xf32> {
  // CHECK: "tosa.conv2d"
  %weight = "tosa.const"() {value = dense<[[[[1.0, 1.0]]], [[[1.0, 1.0]]], [[[1.0, 1.0]]]]> : tensor<3x1x1x2xf32>} : ()-> tensor<3x1x1x2xf32>
  %bias = "tosa.const"() {value = dense<0.0> : tensor<3xf32>} : ()-> tensor<3xf32>
  %0 = "tosa.conv2d"(%arg0, %weight, %bias) {pad = [0, 0, 0, 0], stride = [2, 2], dilation = [1, 1]} : (tensor<4x10x10x2xf32>, tensor<3x1x1x2xf32>, tensor<3xf32>) -> tensor<4x10x10x3xf32>
  return %0 : tensor<4x10x10x3xf32>
}

// -----

// CHECK-LABEL: @conv2d_weight_2x2
func @conv2d_weight_2x2(%arg0: tensor<4x10x10x1xf32>) -> tensor<4x10x10x1xf32> {
  // CHECK: "tosa.conv2d"
  %weight = "tosa.const"() {value = dense<[[[[1.0], [1.0]], [[1.0], [1.0]]]]> : tensor<1x2x2x1xf32>} : ()-> tensor<1x2x2x1xf32>
  %bias = "tosa.const"() {value = dense<0.0> : tensor<1xf32>} : ()-> tensor<1xf32>
  %0 = "tosa.conv2d"(%arg0, %weight, %bias) {pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1]} : (tensor<4x10x10x1xf32>, tensor<1x2x2x1xf32>, tensor<1xf32>) -> tensor<4x10x10x1xf32>
  return %0 : tensor<4x10x10x1xf32>
}

// -----

// CHECK-LABEL: @depthwise_conv2d_stride_2
func @depthwise_conv2d_stride_2(%arg0: tensor<4x10x10x2xf32>, %arg1: tensor<1x1x2x3xf32>, %arg2: tensor<6xf32>) -> tensor<4x10x10x6xf32> {
  // CHECK: "tosa.depthwise_conv2d"
  %0 = "tosa.depthwise_conv2d"(%arg0, %arg1, %arg2) {pad = [0, 0, 0, 0], stride = [2, 2], dilation = [1, 1]} : (tensor<4x10x10x2xf32>, tensor<1x1x2x3xf32>, tensor<6xf32>) -> tensor<4x10x10x6xf32>
  return %0 : tensor<4x10x10x6xf32>
}

// -----

// CHECK-LABEL: @depthwise_conv2d_weight_2x2
func @depthwise_conv2d_weight_2x2(%arg0: tensor<4x10x10x2xf32>, %arg1: tensor<2x2x2x3xf32>, %arg2: tensor<6xf32>) -> tensor<4x10x10x6xf32> {
  // CHECK: "tosa.depthwise_conv2d"
  %0 = "tosa.depthwise_conv2d"(%arg0, %arg1, %arg2) {pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1]} : (tensor<4x10x10x2xf32>, tensor<2x2x2x3xf32>, tensor<6xf32>) -> tensor<4x10x10x6xf32>
  return %0 : tensor<4x10x10x6xf32>
}

// -----

// CHECK-LABEL: @max_pool2d_is_noop
func @max_pool2d_is_noop(%arg0: tensor<10x1x1x3xf32>) -> tensor<10x1x1x3xf32> {
  // CHECK-NOT: "tosa.max_pool2d"
  // CHECK: return %arg0
  %0 = "tosa.max_pool2d"(%arg0) {kernel = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1]} : (tensor<10x1x1x3xf32>) -> tensor<10x1x1x3xf32>
  return %0 : tensor<10x1x1x3xf32>
}

// -----

// CHECK-LABEL: @pad_noop
func @pad_noop(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK: return %arg0
  %0 = "tosa.const"() { value = dense<0> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %1 = "tosa.pad"(%arg0, %0) : (tensor<?x?xf32>, tensor<2x2xi32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @pad_determine_val_i32
func @pad_determine_val_i32(%arg0: tensor<?x?xi32>, %arg1 : tensor<2x2xi32>) -> tensor<?x?xi32> {
  // CHECK: %[[ZERO:.+]] = "tosa.const"() {value = dense<0> : tensor<i32>}
  // CHECK: "tosa.pad"(%arg0, %arg1, %[[ZERO]])
  %0 = "tosa.const"() { value = dense<[[1, 0], [0, 1]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %1 = "tosa.pad"(%arg0, %arg1) : (tensor<?x?xi32>, tensor<2x2xi32>) -> tensor<?x?xi32>
  return %1 : tensor<?x?xi32>
}

// -----

// CHECK-LABEL: @pad_determine_val_f32
func @pad_determine_val_f32(%arg0: tensor<?x?xf32>, %arg1 : tensor<2x2xi32>) -> tensor<?x?xf32> {
  // CHECK: %[[ZERO:.+]] = "tosa.const"() {value = dense<0.000000e+00> : tensor<f32>}
  // CHECK: "tosa.pad"(%arg0, %arg1, %[[ZERO]])
  %0 = "tosa.const"() { value = dense<[[1, 0], [0, 1]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %1 = "tosa.pad"(%arg0, %arg1) : (tensor<?x?xf32>, tensor<2x2xi32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @pad_determine_val_quant
func @pad_determine_val_quant(%arg0: tensor<?x?xi32>, %arg1 : tensor<2x2xi32>) -> tensor<?x?xi32> {
  // CHECK: %[[ZERO:.+]] = "tosa.const"() {value = dense<42> : tensor<i32>}
  // CHECK: "tosa.pad"(%arg0, %arg1, %[[ZERO]])
  %0 = "tosa.const"() { value = dense<[[1, 0], [0, 1]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %1 = "tosa.pad"(%arg0, %arg1) { quantization_info = {input_zp = 42:i32} } : (tensor<?x?xi32>, tensor<2x2xi32>) -> tensor<?x?xi32>
  return %1 : tensor<?x?xi32>
}

// -----

// CHECK-LABEL: @mul_one_different_shape
func @mul_one_different_shape(%arg0: tensor<2x3xf32>) -> tensor<4x2x3xf32> {
  // CHECK: tosa.mul
  %ones = "tosa.const"() {value = dense<1.0> : tensor<4x2x3xf32>} : () -> tensor<4x2x3xf32>
  %1 = "tosa.mul"(%arg0, %ones) {shift = 0 : i32} : (tensor<2x3xf32>, tensor<4x2x3xf32>) -> tensor<4x2x3xf32>
  return %1 : tensor<4x2x3xf32>
}

// -----

// CHECK-LABEL: @mul_one_float
func @mul_one_float(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK: return %arg0
  // CHECK-NOT: tosa.mul
  %ones = "tosa.const"() {value = dense<1.0> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
  %1 = "tosa.mul"(%arg0, %ones) {shift = 0 : i32} : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %1 : tensor<2x3xf32>
}

// -----

// CHECK-LABEL: @mul_one_int
func @mul_one_int(%arg0: tensor<2x3xi32>) -> tensor<2x3xi32> {
  // CHECK: return %arg0
  // CHECK-NOT: tosa.mul
  %ones = "tosa.const"() {value = dense<1> : tensor<2x3xi32>} : () -> tensor<2x3xi32>
  %1 = "tosa.mul"(%arg0, %ones) {shift = 0 : i32} : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  return %1 : tensor<2x3xi32>
}

// -----

// CHECK-LABEL: @reduce_all_fold
func @reduce_all_fold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: return %arg0
  %0 = "tosa.reduce_all"(%arg0) {axis = 1 : i64}: (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// -----

// CHECK-LABEL: @reduce_all_nofold
func @reduce_all_nofold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: "tosa.reduce_all"
  %0 = "tosa.reduce_all"(%arg0) {axis = 0 : i64}: (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// -----

// CHECK-LABEL: @reduce_any_fold
func @reduce_any_fold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: return %arg0
  %0 = "tosa.reduce_any"(%arg0) {axis = 1 : i64}: (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// -----

// CHECK-LABEL: @reduce_any_nofold
func @reduce_any_nofold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: "tosa.reduce_any"
  %0 = "tosa.reduce_any"(%arg0) {axis = 0 : i64}: (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// -----

// CHECK-LABEL: @reduce_max_fold
func @reduce_max_fold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: return %arg0
  %0 = "tosa.reduce_max"(%arg0) {axis = 1 : i64}: (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// -----

// CHECK-LABEL: @reduce_max_nofold
func @reduce_max_nofold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: "tosa.reduce_max"
  %0 = "tosa.reduce_max"(%arg0) {axis = 0 : i64}: (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// -----

// CHECK-LABEL: @reduce_min_fold
func @reduce_min_fold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: return %arg0
  %0 = "tosa.reduce_min"(%arg0) {axis = 1 : i64}: (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// -----

// CHECK-LABEL: @reduce_min_nofold
func @reduce_min_nofold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: "tosa.reduce_min"
  %0 = "tosa.reduce_min"(%arg0) {axis = 0 : i64}: (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// -----

// CHECK-LABEL: @reduce_prod_fold
func @reduce_prod_fold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: return %arg0
  %0 = "tosa.reduce_prod"(%arg0) {axis = 1 : i64}: (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// -----

// CHECK-LABEL: @reduce_prod_nofold
func @reduce_prod_nofold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: "tosa.reduce_prod"
  %0 = "tosa.reduce_prod"(%arg0) {axis = 0 : i64}: (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// -----

// CHECK-LABEL: @reduce_sum_fold
func @reduce_sum_fold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: return %arg0
  %0 = "tosa.reduce_sum"(%arg0) {axis = 1 : i64}: (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// -----

// CHECK-LABEL: @reduce_sum_nofold
func @reduce_sum_nofold(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  // CHECK: "tosa.reduce_sum"
  %0 = "tosa.reduce_sum"(%arg0) {axis = 0 : i64}: (tensor<?x1xf32>) -> tensor<?x1xf32>
  return %0 : tensor<?x1xf32>
}

// -----

// CHECK-LABEL: @reshape_canonicalize
func @reshape_canonicalize(%arg0: tensor<?x10xf32>) -> tensor<?x10xf32> {
  // CHECK: return %arg0
  %0 = "tosa.reshape"(%arg0) {new_shape = [-1, 10]}: (tensor<?x10xf32>) -> tensor<?x10xf32>
  return %0 : tensor<?x10xf32>
}

// -----

// CHECK-LABEL: @reshape_canonicalize_double
func @reshape_canonicalize_double(%arg0: tensor<?x10xf32>) -> tensor<?x5xf32> {
  // CHECK: %[[VAR0:.+]] = "tosa.reshape"(%arg0) {new_shape = [-1, 5]}
  // CHECK: return %[[VAR0]]
  %0 = "tosa.reshape"(%arg0) {new_shape = [5, -1]}: (tensor<?x10xf32>) -> tensor<5x?xf32>
  %1 = "tosa.reshape"(%0) {new_shape = [-1, 5]}: (tensor<5x?xf32>) -> tensor<?x5xf32>
  return %1 : tensor<?x5xf32>
}

// -----

// CHECK-LABEL: @reshape_canonicalize_const
func @reshape_canonicalize_const() -> tensor<1x10xi32> {
  // CHECK: %[[VAR0:.+]] = "tosa.const"() {value = dense<0> : tensor<1x10xi32>}
  // CHECK: return %[[VAR0]]
  %0 = "tosa.const"() {value = dense<0> : tensor<10xi32>} : () -> tensor<10xi32>
  %1 = "tosa.reshape"(%0) {new_shape = [1, 10]} : (tensor<10xi32>) -> tensor<1x10xi32>
  return %1 : tensor<1x10xi32>
}

// -----

// CHECK-LABEL: @reshape_canonicalize_const_spat
func @reshape_canonicalize_const_spat() -> (tensor<10xi32>, tensor<1x10xi32>) {
  // CHECK-DAG: %[[VAR0:.+]] = "tosa.const"() {value = dense<0> : tensor<10xi32>}
  // CHECK-DAG: %[[VAR1:.+]] = "tosa.const"() {value = dense<0> : tensor<1x10xi32>}
  // CHECK: return %[[VAR0]], %[[VAR1]]
  %0 = "tosa.const"() {value = dense<0> : tensor<10xi32>} : () -> tensor<10xi32>
  %1 = "tosa.reshape"(%0) {new_shape = [1, 10]} : (tensor<10xi32>) -> tensor<1x10xi32>
  return %0 , %1 : tensor<10xi32>, tensor<1x10xi32>
}

// -----

// CHECK-LABEL: @reshape_canonicalize_const_sparse
func @reshape_canonicalize_const_sparse() -> (tensor<3xi32>, tensor<1x3xi32>) {
  //CHECK: "tosa.reshape"
  %0 = "tosa.const"() {value = dense<[1, 2, 3]> : tensor<3xi32>} : ()-> tensor<3xi32>
  %1 = "tosa.reshape"(%0) {new_shape = [1, 3]} : (tensor<3xi32>) -> tensor<1x3xi32>
  return %0 , %1 : tensor<3xi32>, tensor<1x3xi32>
}

// -----

// CHECK-LABEL: @slice_fold
func @slice_fold(%arg0: tensor<3x4xf32>) -> tensor<3x4xf32> {
  // CHECK: return %arg0
  %0 = "tosa.slice"(%arg0) { size = [3, 4], start = [0, 0]}: (tensor<3x4xf32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

// -----

// CHECK-LABEL: @slice_nofold
func @slice_nofold(%arg0: tensor<?x4xf32>) -> tensor<?x4xf32> {
  // CHECK: "tosa.slice"
  %0 = "tosa.slice"(%arg0) { size = [3, 4], start = [0, 0]}: (tensor<?x4xf32>) -> tensor<?x4xf32>
  return %0 : tensor<?x4xf32>
}

// -----

// CHECK-LABEL: @tile_fold
func @tile_fold(%arg0: tensor<3x4xf32>) -> tensor<3x4xf32> {
  // CHECK: return %arg0
  %0 = "tosa.tile"(%arg0) { multiples = [1, 1] }: (tensor<3x4xf32>) -> tensor<3x4xf32>
  return %0 : tensor<3x4xf32>
}

// -----

// CHECK-LABEL: @tile_nofold
func @tile_nofold(%arg0: tensor<3x4xf32>) -> tensor<3x8xf32> {
  // CHECK: "tosa.tile"
  %0 = "tosa.tile"(%arg0) { multiples = [1, 2] }: (tensor<3x4xf32>) -> tensor<3x8xf32>
  return %0 : tensor<3x8xf32>
}

// -----

// CHECK-LABEL: @transpose_fold
func @transpose_fold(%arg0: tensor<3x4xf32>) -> tensor<3x4xf32> {
  // CHECK: return %arg0
  %0 = arith.constant dense<[0, 1]> : tensor<2xi32>
  %1 = "tosa.transpose"(%arg0, %0) { perms = [1, 0] }: (tensor<3x4xf32>, tensor<2xi32>) -> tensor<3x4xf32>
  return %1 : tensor<3x4xf32>
}

// -----

// CHECK-LABEL: @transpose_nofold
func @transpose_nofold(%arg0: tensor<3x3xf32>) -> tensor<3x3xf32> {
  // CHECK: "tosa.transpose"
  %0 = arith.constant dense<[1, 0]> : tensor<2xi32>
  %1 = "tosa.transpose"(%arg0, %0) { perms = [1, 0] }: (tensor<3x3xf32>, tensor<2xi32>) -> tensor<3x3xf32>
  return %1 : tensor<3x3xf32>
}

// -----

// CHECK-LABEL: @transpose_nofold_shape
func @transpose_nofold_shape(%arg0: tensor<3x4xf32>) -> tensor<?x?xf32> {
  // CHECK: "tosa.transpose"
  %0 = arith.constant dense<[1, 0]> : tensor<2xi32>
  %1 = "tosa.transpose"(%arg0, %0) { perms = [1, 0] }: (tensor<3x4xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %1 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @transpose_fold_splat
func @transpose_fold_splat() -> tensor<3x2xf32> {
  %input = "tosa.const"() {value = dense<4.0> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
  %perms = "tosa.const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
  //               CHECK: %[[CST:.+]] = "tosa.const"()
  // CHECK-SAME{LITERAL}: value = dense<4.000000e+00> : tensor<3x2xf32>
  %1 = "tosa.transpose"(%input, %perms) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
  // CHECK: return %[[CST]]
  return %1 : tensor<3x2xf32>
}

// -----

// CHECK-LABEL: @transpose_fold_2d_float
func @transpose_fold_2d_float() -> tensor<3x2xf32> {
  %input = "tosa.const"() {value = dense<[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
  %perms = "tosa.const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
  //               CHECK: %[[CST:.+]] = "tosa.const"()
  // CHECK-SAME{LITERAL}: value = dense<[[0.000000e+00, 3.000000e+00], [1.000000e+00, 4.000000e+00], [2.000000e+00, 5.000000e+00]]> : tensor<3x2xf32>
  %1 = "tosa.transpose"(%input, %perms) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
  // CHECK: return %[[CST]]
  return %1 : tensor<3x2xf32>
}

// -----

// CHECK-LABEL: @transpose_fold_4d_int
func @transpose_fold_4d_int() -> tensor<3x1x4x2xi32> {
  %input = "tosa.const"() {value = dense<[[
    [[ 0,  1,  2,  3], [ 4,  5,  6,  7], [ 8,  9, 10, 11]],
    [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]
  ]]> : tensor<1x2x3x4xi32>} : () -> tensor<1x2x3x4xi32>
  %perms = "tosa.const"() {value = dense<[2, 0, 3, 1]> : tensor<4xi64>} : () -> tensor<4xi64>
  //               CHECK: %[[CST:.+]] = "tosa.const"()
  // CHECK-SAME{LITERAL}: value = dense<[
  // CHECK-SAME{LITERAL}:   [[[0, 12], [1, 13], [2, 14], [3, 15]]],
  // CHECK-SAME{LITERAL}:   [[[4, 16], [5, 17], [6, 18], [7, 19]]],
  // CHECK-SAME{LITERAL}:   [[[8, 20], [9, 21], [10, 22], [11, 23]]]
  // CHECK-SAME{LITERAL}: ]>
  %1 = "tosa.transpose"(%input, %perms) : (tensor<1x2x3x4xi32>, tensor<4xi64>) -> tensor<3x1x4x2xi32>
  // CHECK: return %[[CST]]
  return %1 : tensor<3x1x4x2xi32>
}

// -----

// CHECK-LABEL: @transpose_nofold_non_cst_input
func @transpose_nofold_non_cst_input(%input: tensor<2x3xf32>) -> tensor<3x2xf32> {
  %perms = "tosa.const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
  // CHECK: tosa.transpose
  %1 = "tosa.transpose"(%input, %perms) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
  return %1 : tensor<3x2xf32>
}

// -----

// CHECK-LABEL: @transpose_nofold_non_cst_perms
func @transpose_nofold_non_cst_perms(%perms: tensor<2xi32>) -> tensor<3x2xf32> {
  %input = "tosa.const"() {value = dense<[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
  // CHECK: tosa.transpose
  %1 = "tosa.transpose"(%input, %perms) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
  return %1 : tensor<3x2xf32>
}

// -----

// CHECK-LABEL: @transpose_nofold_multi_users
func @transpose_nofold_multi_users() -> (tensor<3x2xf32>, tensor<2x3xf32>) {
  %input = "tosa.const"() {value = dense<[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
  %perms = "tosa.const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
  // CHECK: tosa.transpose
  %1 = "tosa.transpose"(%input, %perms) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
  return %1, %input : tensor<3x2xf32>, tensor<2x3xf32>
}

// -----

// CHECK-LABEL: @transpose_nofold_quantized_types
func @transpose_nofold_quantized_types() -> tensor<1x1x16x1x!quant.uniform<i8<-127:127>:f32:3, {1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,2.100000e+00,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01}>> {
  %perms = "tosa.const"() {value = dense<[1, 2, 3, 0]> : tensor<4xi32>} : () -> tensor<4xi32>
  %input = "tosa.const"() {value = dense<[[[[-127, 127, 127, -127, -127, -127, -127, -127, -127, 127, 127, 127, 127, 127, -127, 127]]]]> : tensor<1x1x1x16xi8>} : () -> tensor<1x1x1x16xi8>
  // CHECK: tosa.transpose
  %0 = "tosa.transpose"(%input, %perms) : (tensor<1x1x1x16xi8>, tensor<4xi32>) -> tensor<1x1x16x1x!quant.uniform<i8<-127:127>:f32:3, {1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,2.100000e+00,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01}>>
  return %0: tensor<1x1x16x1x!quant.uniform<i8<-127:127>:f32:3, {1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,2.100000e+00,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01,1.000000e-01}>>
}

// -----

// CHECK-LABEL: @transpose_no_op
func @transpose_no_op(%arg0: tensor<3x4x5x6xf32>) -> tensor<3x4x5x6xf32> {
  // CHECK: return %arg0
  // CHECK-NOT: tosa.transpose
  %perms = "tosa.const"() {value = dense<[0, 1, 2, 3]> : tensor<4xi32>} : () -> tensor<4xi32>
  %1 = "tosa.transpose"(%arg0, %perms) : (tensor<3x4x5x6xf32>, tensor<4xi32>) -> tensor<3x4x5x6xf32>
  return %1 : tensor<3x4x5x6xf32>
}
