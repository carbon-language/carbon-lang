// RUN: mlir-opt %s -pass-pipeline='builtin.func(canonicalize)' -split-input-file -allow-unregistered-dialect | FileCheck %s

// -----

// CHECK-LABEL: create_vector_mask_to_constant_mask
func @create_vector_mask_to_constant_mask() -> (vector<4x3xi1>) {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  // CHECK: vector.constant_mask [3, 2] : vector<4x3xi1>
  %0 = vector.create_mask %c3, %c2 : vector<4x3xi1>
  return %0 : vector<4x3xi1>
}

// -----

// CHECK-LABEL: create_vector_mask_to_constant_mask_truncation
func @create_vector_mask_to_constant_mask_truncation() -> (vector<4x3xi1>) {
  %c2 = arith.constant 2 : index
  %c5 = arith.constant 5 : index
  // CHECK: vector.constant_mask [4, 2] : vector<4x3xi1>
  %0 = vector.create_mask %c5, %c2 : vector<4x3xi1>
  return %0 : vector<4x3xi1>
}

// -----

// CHECK-LABEL: create_vector_mask_to_constant_mask_truncation_neg
func @create_vector_mask_to_constant_mask_truncation_neg() -> (vector<4x3xi1>) {
  %cneg2 = arith.constant -2 : index
  %c5 = arith.constant 5 : index
  // CHECK: vector.constant_mask [0, 0] : vector<4x3xi1>
  %0 = vector.create_mask %c5, %cneg2 : vector<4x3xi1>
  return %0 : vector<4x3xi1>
}

// -----

// CHECK-LABEL: create_vector_mask_to_constant_mask_truncation_zero
func @create_vector_mask_to_constant_mask_truncation_zero() -> (vector<4x3xi1>) {
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  // CHECK: vector.constant_mask [0, 0] : vector<4x3xi1>
  %0 = vector.create_mask %c0, %c2 : vector<4x3xi1>
  return %0 : vector<4x3xi1>
}

// -----

func @extract_strided_slice_of_constant_mask() -> (vector<2x2xi1>) {
  %0 = vector.constant_mask [2, 2] : vector<4x3xi1>
  %1 = vector.extract_strided_slice %0
    {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]}
      : vector<4x3xi1> to vector<2x2xi1>
  // CHECK: vector.constant_mask [2, 2] : vector<2x2xi1>
  return %1 : vector<2x2xi1>
}

// -----

func @extract_strided_slice_of_constant_mask() -> (vector<2x2xi1>) {
  %0 = vector.constant_mask [2, 2] : vector<4x3xi1>
  %1 = vector.extract_strided_slice %0
    {offsets = [1, 0], sizes = [2, 2], strides = [1, 1]}
      : vector<4x3xi1> to vector<2x2xi1>
  // CHECK: vector.constant_mask [1, 2] : vector<2x2xi1>
  return %1 : vector<2x2xi1>
}

// -----

func @extract_strided_slice_of_constant_mask() -> (vector<2x2xi1>) {
  %0 = vector.constant_mask [2, 2] : vector<4x3xi1>
  %1 = vector.extract_strided_slice %0
    {offsets = [0, 1], sizes = [2, 2], strides = [1, 1]}
      : vector<4x3xi1> to vector<2x2xi1>
  // CHECK: vector.constant_mask [2, 1] : vector<2x2xi1>
  return %1 : vector<2x2xi1>
}

// -----

func @extract_strided_slice_of_constant_mask() -> (vector<2x2xi1>) {
  %0 = vector.constant_mask [2, 2] : vector<4x3xi1>
  %1 = vector.extract_strided_slice %0
    {offsets = [2, 0], sizes = [2, 2], strides = [1, 1]}
      : vector<4x3xi1> to vector<2x2xi1>
  // CHECK: vector.constant_mask [0, 0] : vector<2x2xi1>
  return %1 : vector<2x2xi1>
}

// -----

func @extract_strided_slice_of_constant_mask() -> (vector<2x1xi1>) {
  %0 = vector.constant_mask [2, 2] : vector<4x3xi1>
  %1 = vector.extract_strided_slice %0
    {offsets = [0, 2], sizes = [2, 1], strides = [1, 1]}
      : vector<4x3xi1> to vector<2x1xi1>
  // CHECK: vector.constant_mask [0, 0] : vector<2x1xi1>
  return %1 : vector<2x1xi1>
}

// -----

func @extract_strided_slice_of_constant_mask() -> (vector<2x1xi1>) {
  %0 = vector.constant_mask [2, 2] : vector<4x3xi1>
  %1 = vector.extract_strided_slice %0
    {offsets = [0, 1], sizes = [2, 1], strides = [1, 1]}
      : vector<4x3xi1> to vector<2x1xi1>
  // CHECK: vector.constant_mask [2, 1] : vector<2x1xi1>
  return %1 : vector<2x1xi1>
}

// -----

func @extract_strided_slice_of_constant_mask() -> (vector<2x1xi1>) {
  %0 = vector.constant_mask [2, 2] : vector<4x3xi1>
  %1 = vector.extract_strided_slice %0
    {offsets = [1, 1], sizes = [2, 1], strides = [1, 1]}
      : vector<4x3xi1> to vector<2x1xi1>
  // CHECK: vector.constant_mask [1, 1] : vector<2x1xi1>
  return %1 : vector<2x1xi1>
}

// -----

// CHECK-LABEL: extract_strided_fold
//  CHECK-SAME: (%[[ARG:.*]]: vector<4x3xi1>)
//  CHECK-NEXT:   return %[[ARG]] : vector<4x3xi1>
func @extract_strided_fold(%arg : vector<4x3xi1>) -> (vector<4x3xi1>) {
  %0 = vector.extract_strided_slice %arg
    {offsets = [0, 0], sizes = [4, 3], strides = [1, 1]}
      : vector<4x3xi1> to vector<4x3xi1>
  return %0 : vector<4x3xi1>
}

// -----

// CHECK-LABEL: extract_strided_fold_insert
//  CHECK-SAME: (%[[ARG:.*]]: vector<4x4xf32>
//  CHECK-NEXT:   return %[[ARG]] : vector<4x4xf32>
func @extract_strided_fold_insert(%a: vector<4x4xf32>, %b: vector<8x16xf32>)
  -> (vector<4x4xf32>) {
  %0 = vector.insert_strided_slice %a, %b {offsets = [2, 2], strides = [1, 1]}
    : vector<4x4xf32> into vector<8x16xf32>
  %1 = vector.extract_strided_slice %0
    {offsets = [2, 2], sizes = [4, 4], strides = [1, 1]}
      : vector<8x16xf32> to vector<4x4xf32>
  return %1 : vector<4x4xf32>
}

// -----

// Case where the vector inserted is a subset of the vector extracted.
// CHECK-LABEL: extract_strided_fold_insert
//  CHECK-SAME: (%[[ARG0:.*]]: vector<6x4xf32>
//  CHECK-NEXT:   %[[EXT:.*]] = vector.extract_strided_slice %[[ARG0]]
//  CHECK-SAME:     {offsets = [0, 0], sizes = [4, 4], strides = [1, 1]}
//  CHECK-SAME:       : vector<6x4xf32> to vector<4x4xf32>
//  CHECK-NEXT:   return %[[EXT]] : vector<4x4xf32>
func @extract_strided_fold_insert(%a: vector<6x4xf32>, %b: vector<8x16xf32>)
  -> (vector<4x4xf32>) {
  %0 = vector.insert_strided_slice %a, %b {offsets = [2, 2], strides = [1, 1]}
    : vector<6x4xf32> into vector<8x16xf32>
  %1 = vector.extract_strided_slice %0
    {offsets = [2, 2], sizes = [4, 4], strides = [1, 1]}
      : vector<8x16xf32> to vector<4x4xf32>
  return %1 : vector<4x4xf32>
}

// -----

// Negative test where the extract is not a subset of the element inserted.
// CHECK-LABEL: extract_strided_fold_negative
//  CHECK-SAME: (%[[ARG0:.*]]: vector<4x4xf32>, %[[ARG1:.*]]: vector<8x16xf32>
//       CHECK:   %[[INS:.*]] = vector.insert_strided_slice %[[ARG0]], %[[ARG1]]
//  CHECK-SAME:     {offsets = [2, 2], strides = [1, 1]}
//  CHECK-SAME:       : vector<4x4xf32> into vector<8x16xf32>
//       CHECK:   %[[EXT:.*]] = vector.extract_strided_slice %[[INS]]
//  CHECK-SAME:     {offsets = [2, 2], sizes = [6, 4], strides = [1, 1]}
//  CHECK-SAME:       : vector<8x16xf32> to vector<6x4xf32>
//  CHECK-NEXT:   return %[[EXT]] : vector<6x4xf32>
func @extract_strided_fold_negative(%a: vector<4x4xf32>, %b: vector<8x16xf32>)
  -> (vector<6x4xf32>) {
  %0 = vector.insert_strided_slice %a, %b {offsets = [2, 2], strides = [1, 1]}
    : vector<4x4xf32> into vector<8x16xf32>
  %1 = vector.extract_strided_slice %0
    {offsets = [2, 2], sizes = [6, 4], strides = [1, 1]}
      : vector<8x16xf32> to vector<6x4xf32>
  return %1 : vector<6x4xf32>
}

// -----

// Case where we need to go through 2 level of insert element.
// CHECK-LABEL: extract_strided_fold_insert
//  CHECK-SAME: (%[[ARG0:.*]]: vector<2x8xf32>, %[[ARG1:.*]]: vector<1x4xf32>,
//  CHECK-NEXT:   %[[EXT:.*]] = vector.extract_strided_slice %[[ARG1]]
//  CHECK-SAME:     {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]}
//  CHECK-SAME:       : vector<1x4xf32> to vector<1x1xf32>
//  CHECK-NEXT:   return %[[EXT]] : vector<1x1xf32>
func @extract_strided_fold_insert(%a: vector<2x8xf32>, %b: vector<1x4xf32>,
                                  %c : vector<1x4xf32>) -> (vector<1x1xf32>) {
  %0 = vector.insert_strided_slice %b, %a {offsets = [0, 1], strides = [1, 1]}
    : vector<1x4xf32> into vector<2x8xf32>
  %1 = vector.insert_strided_slice %c, %0 {offsets = [1, 0], strides = [1, 1]}
    : vector<1x4xf32> into vector<2x8xf32>
  %2 = vector.extract_strided_slice %1
      {offsets = [0, 1], sizes = [1, 1], strides = [1, 1]}
        : vector<2x8xf32> to vector<1x1xf32>
  return %2 : vector<1x1xf32>
}

// -----

// CHECK-LABEL: transpose_1D_identity
// CHECK-SAME: ([[ARG:%.*]]: vector<4xf32>)
func @transpose_1D_identity(%arg : vector<4xf32>) -> vector<4xf32> {
  // CHECK-NOT: transpose
  %0 = vector.transpose %arg, [0] : vector<4xf32> to vector<4xf32>
  // CHECK-NEXT: return [[ARG]]
  return %0 : vector<4xf32>
}

// -----

// CHECK-LABEL: transpose_2D_identity
// CHECK-SAME: ([[ARG:%.*]]: vector<4x3xf32>)
func @transpose_2D_identity(%arg : vector<4x3xf32>) -> vector<4x3xf32> {
  // CHECK-NOT: transpose
  %0 = vector.transpose %arg, [0, 1] : vector<4x3xf32> to vector<4x3xf32>
  // CHECK-NEXT: return [[ARG]]
  return %0 : vector<4x3xf32>
}

// -----

// CHECK-LABEL: transpose_3D_identity
// CHECK-SAME: ([[ARG:%.*]]: vector<4x3x2xf32>)
func @transpose_3D_identity(%arg : vector<4x3x2xf32>) -> vector<4x3x2xf32> {
  // CHECK-NOT: transpose
  %0 = vector.transpose %arg, [0, 1, 2] : vector<4x3x2xf32> to vector<4x3x2xf32>
  // CHECK-NEXT: return [[ARG]]
  return %0 : vector<4x3x2xf32>
}

// -----

// CHECK-LABEL: transpose_2D_sequence
// CHECK-SAME: ([[ARG:%.*]]: vector<4x3xf32>)
func @transpose_2D_sequence(%arg : vector<4x3xf32>) -> vector<4x3xf32> {
  // CHECK-NOT: transpose
  %0 = vector.transpose %arg, [1, 0] : vector<4x3xf32> to vector<3x4xf32>
  %1 = vector.transpose %0, [0, 1] : vector<3x4xf32> to vector<3x4xf32>
  %2 = vector.transpose %1, [1, 0] : vector<3x4xf32> to vector<4x3xf32>
  %3 = vector.transpose %2, [0, 1] : vector<4x3xf32> to vector<4x3xf32>
  // CHECK: [[ADD:%.*]] = arith.addf [[ARG]], [[ARG]]
  %4 = arith.addf %2, %3 : vector<4x3xf32>
  // CHECK-NEXT: return [[ADD]]
  return %4 : vector<4x3xf32>
}

// -----

// CHECK-LABEL: transpose_3D_sequence
// CHECK-SAME: ([[ARG:%.*]]: vector<4x3x2xf32>)
func @transpose_3D_sequence(%arg : vector<4x3x2xf32>) -> vector<4x3x2xf32> {
  // CHECK: [[T0:%.*]] = vector.transpose [[ARG]], [2, 1, 0]
  %0 = vector.transpose %arg, [1, 2, 0] : vector<4x3x2xf32> to vector<3x2x4xf32>
  %1 = vector.transpose %0, [1, 0, 2] : vector<3x2x4xf32> to vector<2x3x4xf32>
  // CHECK: [[T1:%.*]] = vector.transpose %arg0, [2, 1, 0]
  %2 = vector.transpose %1, [2, 1, 0] : vector<2x3x4xf32> to vector<4x3x2xf32>
  %3 = vector.transpose %2, [2, 1, 0] : vector<4x3x2xf32> to vector<2x3x4xf32>
  // CHECK: [[MUL:%.*]] = arith.mulf [[T0]], [[T1]]
  %4 = arith.mulf %1, %3 : vector<2x3x4xf32>
  // CHECK: [[T5:%.*]] = vector.transpose [[MUL]], [2, 1, 0]
  %5 = vector.transpose %4, [2, 1, 0] : vector<2x3x4xf32> to vector<4x3x2xf32>
  // CHECK-NOT: transpose
  %6 = vector.transpose %3, [2, 1, 0] : vector<2x3x4xf32> to vector<4x3x2xf32>
  // CHECK: [[ADD:%.*]] = arith.addf [[T5]], [[ARG]]
  %7 = arith.addf %5, %6 : vector<4x3x2xf32>
  // CHECK-NEXT: return [[ADD]]
  return %7 : vector<4x3x2xf32>
}

// -----

// CHECK-LABEL: cast_transfers
func @cast_transfers(%A: memref<4x8xf32>) -> (vector<4x8xf32>) {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f32
  %0 = memref.cast %A : memref<4x8xf32> to memref<?x?xf32>

  // CHECK: vector.transfer_read %{{.*}} {in_bounds = [true, true]} : memref<4x8xf32>, vector<4x8xf32>
  %1 = vector.transfer_read %0[%c0, %c0], %f0 : memref<?x?xf32>, vector<4x8xf32>

  // CHECK: vector.transfer_write %{{.*}} {in_bounds = [true, true]} : vector<4x8xf32>, memref<4x8xf32>
  vector.transfer_write %1, %0[%c0, %c0] : vector<4x8xf32>, memref<?x?xf32>
  return %1 : vector<4x8xf32>
}

// -----

// CHECK-LABEL: cast_transfers
func @cast_transfers(%A: tensor<4x8xf32>) -> (vector<4x8xf32>) {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f32
  %0 = tensor.cast %A : tensor<4x8xf32> to tensor<?x?xf32>

  // CHECK: vector.transfer_read %{{.*}} {in_bounds = [true, true]} : tensor<4x8xf32>, vector<4x8xf32>
  %1 = vector.transfer_read %0[%c0, %c0], %f0 : tensor<?x?xf32>, vector<4x8xf32>

  return %1 : vector<4x8xf32>
}

// -----

// CHECK-LABEL: func @insert_extract_transpose_2d(
//  CHECK-SAME: %[[V:[a-zA-Z0-9]*]]: vector<2x3xf32>,
//  CHECK-SAME: %[[F0:[a-zA-Z0-9]*]]: f32,
//  CHECK-SAME: %[[F1:[a-zA-Z0-9]*]]: f32,
//  CHECK-SAME: %[[F2:[a-zA-Z0-9]*]]: f32,
//  CHECK-SAME: %[[F3:[a-zA-Z0-9]*]]: f32
func @insert_extract_transpose_2d(
    %v: vector<2x3xf32>, %f0: f32, %f1: f32, %f2: f32, %f3: f32)
-> (f32, f32, f32)
{
  %0 = vector.insert %f0, %v[0, 0] : f32 into vector<2x3xf32>
  %1 = vector.insert %f1, %0[0, 1] : f32 into vector<2x3xf32>
  %2 = vector.insert %f2, %1[1, 0] : f32 into vector<2x3xf32>
  %3 = vector.insert %f3, %2[1, 1] : f32 into vector<2x3xf32>
  %4 = vector.transpose %3, [1, 0] : vector<2x3xf32> to vector<3x2xf32>
  %5 = vector.insert %f3, %4[1, 0] : f32 into vector<3x2xf32>
  %6 = vector.transpose %5, [1, 0] : vector<3x2xf32> to vector<2x3xf32>

  // Expected %f2 from %2 = vector.insert %f2, %1[1, 0].
  %r1 = vector.extract %3[1, 0] : vector<2x3xf32>

  // Expected %f1 from %1 = vector.insert %f1, %0[0, 1] followed by
  // transpose [1, 0].
  %r2 = vector.extract %4[1, 0] : vector<3x2xf32>

  // Expected %f2 from %2 = vector.insert %f2, %1[1, 0] followed by double
  // transpose [1, 0].
  %r3 = vector.extract %6[1, 0] : vector<2x3xf32>

  // CHECK-NEXT: return %[[F2]], %[[F1]], %[[F2]] : f32, f32, f32
  return %r1, %r2, %r3 : f32, f32, f32
}

// -----

// CHECK-LABEL: insert_extract_chain
//  CHECK-SAME: %[[V234:[a-zA-Z0-9]*]]: vector<2x3x4xf32>
//  CHECK-SAME: %[[V34:[a-zA-Z0-9]*]]: vector<3x4xf32>
//  CHECK-SAME: %[[V4:[a-zA-Z0-9]*]]: vector<4xf32>
func @insert_extract_chain(%v234: vector<2x3x4xf32>, %v34: vector<3x4xf32>, %v4: vector<4xf32>)
    -> (vector<4xf32>, vector<4xf32>, vector<3x4xf32>, vector<3x4xf32>) {
  // CHECK-NEXT: %[[A34:.*]] = vector.insert
  %A34 = vector.insert %v34, %v234[0]: vector<3x4xf32> into vector<2x3x4xf32>
  // CHECK-NEXT: %[[B34:.*]] = vector.insert
  %B34 = vector.insert %v34, %A34[1]: vector<3x4xf32> into vector<2x3x4xf32>
  // CHECK-NEXT: %[[A4:.*]] = vector.insert
  %A4 = vector.insert %v4, %B34[1, 0]: vector<4xf32> into vector<2x3x4xf32>
  // CHECK-NEXT: %[[B4:.*]] = vector.insert
  %B4 = vector.insert %v4, %A4[1, 1]: vector<4xf32> into vector<2x3x4xf32>

  // Case 2.a. [1, 1] == insertpos ([1, 1])
  // Match %A4 insertionpos and fold to its source(i.e. %V4).
   %r0 = vector.extract %B4[1, 1]: vector<2x3x4xf32>

  // Case 3.a. insertpos ([1]) is a prefix of [1, 0].
  // Traverse %B34 to its source(i.e. %V34@[*0*]).
  // CHECK-NEXT: %[[R1:.*]] = vector.extract %[[V34]][0]
   %r1 = vector.extract %B34[1, 0]: vector<2x3x4xf32>

  // Case 4. [1] is a prefix of insertpos ([1, 1]).
  // Cannot traverse %B4.
  // CHECK-NEXT: %[[R2:.*]] = vector.extract %[[B4]][1]
   %r2 = vector.extract %B4[1]: vector<2x3x4xf32>

  // Case 5. [0] is disjoint from insertpos ([1, 1]).
  // Traverse %B4 to its dest(i.e. %A4@[0]).
  // Traverse %A4 to its dest(i.e. %B34@[0]).
  // Traverse %B34 to its dest(i.e. %A34@[0]).
  // Match %A34 insertionpos and fold to its source(i.e. %V34).
   %r3 = vector.extract %B4[0]: vector<2x3x4xf32>

  // CHECK: return %[[V4]], %[[R1]], %[[R2]], %[[V34]]
  return %r0, %r1, %r2, %r3:
    vector<4xf32>, vector<4xf32>, vector<3x4xf32>, vector<3x4xf32>
}

// -----

// CHECK-LABEL: func @insert_extract_transpose_3d(
//  CHECK-SAME: %[[V234:[a-zA-Z0-9]*]]: vector<2x3x4xf32>
func @insert_extract_transpose_3d(
  %v234: vector<2x3x4xf32>, %v43: vector<4x3xf32>, %f0: f32)
    -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<3x4xf32>) {

  %a432 = vector.transpose %v234, [2, 1, 0] : vector<2x3x4xf32> to vector<4x3x2xf32>
  %b432 = vector.insert %f0, %a432[0, 0, 1] : f32 into vector<4x3x2xf32>
  %c234 = vector.transpose %b432, [2, 1, 0] : vector<4x3x2xf32> to vector<2x3x4xf32>
  // Case 1. %c234 = transpose [2,1,0] posWithSentinels [1,2,-1] -> [-1,2,1]
  // Case 5. %b432 = insert [0,0,1] (inter([.,2,1], [.,0,1]) == 0) prop to %v432
  // Case 1. %a432 = transpose [2,1,0] posWithSentinels [-1,2,1] -> [1,2,-1]
  // can extract directly from %v234, the rest folds.
  // CHECK: %[[R0:.*]] = vector.extract %[[V234]][1, 2]
  %r0 = vector.extract %c234[1, 2] : vector<2x3x4xf32>

  // CHECK-NEXT: vector.transpose
  // CHECK-NEXT: vector.insert
  // CHECK-NEXT: %[[F234:.*]] = vector.transpose
  %d432 = vector.transpose %v234, [2, 1, 0] : vector<2x3x4xf32> to vector<4x3x2xf32>
  %e432 = vector.insert %f0, %d432[0, 2, 1] : f32 into vector<4x3x2xf32>
  %f234 = vector.transpose %e432, [2, 1, 0] : vector<4x3x2xf32> to vector<2x3x4xf32>
  // Case 1. %c234 = transpose [2,1,0] posWithSentinels [1,2,-1] -> [-1,2,1]
  // Case 4. %b432 = insert [0,0,1] (inter([.,2,1], [.,2,1]) != 0)
  // Bail, cannot do better than the current.
  // CHECK: %[[R1:.*]] = vector.extract %[[F234]]
  %r1 = vector.extract %f234[1, 2] : vector<2x3x4xf32>

  // CHECK-NEXT: vector.transpose
  // CHECK-NEXT: vector.insert
  // CHECK-NEXT: %[[H234:.*]] = vector.transpose
  %g243 = vector.transpose %v234, [0, 2, 1] : vector<2x3x4xf32> to vector<2x4x3xf32>
  %h243 = vector.insert %v43, %g243[0] : vector<4x3xf32> into vector<2x4x3xf32>
  %i234 = vector.transpose %h243, [0, 2, 1] : vector<2x4x3xf32> to vector<2x3x4xf32>
  // Case 1. %i234 = transpose [0,2,1] posWithSentinels [0,-1,-2] -> [0,-2,-1]
  // Case 3.b. %b432 = insert [0] is prefix of [0,.,.] but internal transpose.
  // Bail, cannot do better than the current.
  // CHECK: %[[R2:.*]] = vector.extract %[[H234]][0, 1]
  %r2 = vector.extract %i234[0, 1] : vector<2x3x4xf32>

  // CHECK-NEXT: vector.transpose
  // CHECK-NEXT: vector.insert
  // CHECK-NEXT: %[[K234:.*]] = vector.transpose
  %j243 = vector.transpose %v234, [0, 2, 1] : vector<2x3x4xf32> to vector<2x4x3xf32>
  %k243 = vector.insert %v43, %j243[0] : vector<4x3xf32> into vector<2x4x3xf32>
  %l234 = vector.transpose %k243, [0, 2, 1] : vector<2x4x3xf32> to vector<2x3x4xf32>
  // Case 1. %i234 = transpose [0,2,1] posWithSentinels [0,-1,-2] -> [0,-2,-1]
  // Case 2.b. %b432 = insert [0] == [0,.,.] but internal transpose.
  // Bail, cannot do better than the current.
  // CHECK: %[[R3:.*]] = vector.extract %[[K234]][0]
  %r3 = vector.extract %l234[0] : vector<2x3x4xf32>

  // CHECK-NEXT: return %[[R0]], %[[R1]], %[[R2]], %[[R3]]
  return %r0, %r1, %r2, %r3: vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<3x4xf32>
}

// -----

// CHECK-LABEL: fold_extracts
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]*]]: vector<3x4x5x6xf32>
func @fold_extracts(%a : vector<3x4x5x6xf32>) -> (f32, vector<4x5x6xf32>) {
  %b = vector.extract %a[0] : vector<3x4x5x6xf32>
  %c = vector.extract %b[1, 2] : vector<4x5x6xf32>
  //  CHECK-NEXT: vector.extract %[[A]][0, 1, 2, 3] : vector<3x4x5x6xf32>
  %d = vector.extract %c[3] : vector<6xf32>

  //  CHECK-NEXT: vector.extract %[[A]][0] : vector<3x4x5x6xf32>
  %e = vector.extract %a[0] : vector<3x4x5x6xf32>

  //  CHECK-NEXT: return
  return %d, %e : f32, vector<4x5x6xf32>
}

// -----

// CHECK-LABEL: fold_extract_transpose
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]*]]: vector<3x4x5x6xf32>
//  CHECK-SAME:   %[[B:[a-zA-Z0-9]*]]: vector<3x6x5x6xf32>
func @fold_extract_transpose(
    %a : vector<3x4x5x6xf32>, %b : vector<3x6x5x6xf32>) -> (
      vector<6xf32>, vector<6xf32>, vector<6xf32>) {
  // [3] is a proper most minor identity map in transpose.
  // Permutation is a self inverse and we have.
  // [0, 2, 1] ^ -1 o [0, 1, 2] = [0, 2, 1] o [0, 1, 2]
  //                            = [0, 2, 1]
  //  CHECK-NEXT: vector.extract %[[A]][0, 2, 1] : vector<3x4x5x6xf32>
  %0 = vector.transpose %a, [0, 2, 1, 3] : vector<3x4x5x6xf32> to vector<3x5x4x6xf32>
  %1 = vector.extract %0[0, 1, 2] : vector<3x5x4x6xf32>

  // [3] is a proper most minor identity map in transpose.
  // Permutation is a not self inverse and we have.
  // [1, 2, 0] ^ -1 o [0, 1, 2] = [2, 0, 1] o [0, 1, 2]
  //                            = [2, 0, 1]
  //  CHECK-NEXT: vector.extract %[[A]][2, 0, 1] : vector<3x4x5x6xf32>
  %2 = vector.transpose %a, [1, 2, 0, 3] : vector<3x4x5x6xf32> to vector<4x5x3x6xf32>
  %3 = vector.extract %2[0, 1, 2] : vector<4x5x3x6xf32>

  // Not a minor identity map so intra-vector level has been permuted
  //  CHECK-NEXT: vector.transpose %[[B]], [0, 2, 3, 1]
  //  CHECK-NEXT: vector.extract %{{.*}}[0, 1, 2]
  %4 = vector.transpose %b, [0, 2, 3, 1] : vector<3x6x5x6xf32> to vector<3x5x6x6xf32>
  %5 = vector.extract %4[0, 1, 2] : vector<3x5x6x6xf32>

  return %1, %3, %5 : vector<6xf32>, vector<6xf32>, vector<6xf32>
}

// -----

// CHECK-LABEL: fold_extract_broadcast
//  CHECK-SAME:   %[[A:.*]]: f32
//       CHECK:   return %[[A]] : f32
func @fold_extract_broadcast(%a : f32) -> f32 {
  %b = vector.broadcast %a : f32 to vector<1x2x4xf32>
  %r = vector.extract %b[0, 1, 2] : vector<1x2x4xf32>
  return %r : f32
}

// -----

// CHECK-LABEL: fold_extract_splat
//  CHECK-SAME:   %[[A:.*]]: f32
//       CHECK:   return %[[A]] : f32
func @fold_extract_splat(%a : f32) -> f32 {
  %b = splat %a : vector<1x2x4xf32>
  %r = vector.extract %b[0, 1, 2] : vector<1x2x4xf32>
  return %r : f32
}

// -----

// CHECK-LABEL: fold_extract_broadcast_vector
//  CHECK-SAME:   %[[A:.*]]: vector<4xf32>
//       CHECK:   return %[[A]] : vector<4xf32>
func @fold_extract_broadcast_vector(%a : vector<4xf32>) -> vector<4xf32> {
  %b = vector.broadcast %a : vector<4xf32> to vector<1x2x4xf32>
  %r = vector.extract %b[0, 1] : vector<1x2x4xf32>
  return %r : vector<4xf32>
}

// -----

// CHECK-LABEL: fold_extract_broadcast
//  CHECK-SAME:   %[[A:.*]]: vector<4xf32>
//       CHECK:   %[[R:.*]] = vector.extract %[[A]][2] : vector<4xf32>
//       CHECK:   return %[[R]] : f32
func @fold_extract_broadcast(%a : vector<4xf32>) -> f32 {
  %b = vector.broadcast %a : vector<4xf32> to vector<1x2x4xf32>
  %r = vector.extract %b[0, 1, 2] : vector<1x2x4xf32>
  return %r : f32
}

// -----

// CHECK-LABEL: fold_extract_broadcast
//       CHECK:   %[[B:.*]] = vector.broadcast %{{.*}} : f32 to vector<4xf32>
//       CHECK:   return %[[B]] : vector<4xf32>
func @fold_extract_broadcast(%a : f32) -> vector<4xf32> {
  %b = vector.broadcast %a : f32 to vector<1x2x4xf32>
  %r = vector.extract %b[0, 1] : vector<1x2x4xf32>
  return %r : vector<4xf32>
}

// -----

// CHECK-LABEL: func @fold_extract_shapecast
//  CHECK-SAME: (%[[A0:.*]]: vector<5x1x3x2xf32>, %[[A1:.*]]: vector<8x4x2xf32>
//       CHECK:   %[[R0:.*]] = vector.extract %[[A0]][1, 0, 1, 1] : vector<5x1x3x2xf32>
//       CHECK:   %[[R1:.*]] = vector.extract %[[A0]][1, 0, 2] : vector<5x1x3x2xf32>
//       CHECK:   %[[R2:.*]] = vector.extract %[[A1]][7] : vector<8x4x2xf32>
//       CHECK:   return %[[R0]], %[[R1]], %[[R2]], %[[A1]] : f32, vector<2xf32>, vector<4x2xf32>, vector<8x4x2xf32>
func @fold_extract_shapecast(%arg0 : vector<5x1x3x2xf32>,
                             %arg1 : vector<8x4x2xf32>)
  -> (f32, vector<2xf32>, vector<4x2xf32>, vector<8x4x2xf32>) {
  %0 = vector.shape_cast %arg0 : vector<5x1x3x2xf32> to vector<15x2xf32>
  %1 = vector.shape_cast %arg1 : vector<8x4x2xf32> to vector<4x2x4x2xf32>
  %2 = vector.shape_cast %arg1 : vector<8x4x2xf32> to vector<1x8x4x2xf32>
  %r1 = vector.extract %0[4, 1] : vector<15x2xf32>
  %r2 = vector.extract %0[5] : vector<15x2xf32>
  %r3 = vector.extract %1[3, 1] : vector<4x2x4x2xf32>
  %r4 = vector.extract %2[0] : vector<1x8x4x2xf32>
  return %r1, %r2, %r3, %r4 : f32, vector<2xf32>, vector<4x2xf32>, vector<8x4x2xf32>
}

// -----

// CHECK-LABEL: fold_extract_shapecast_negative
//       CHECK:   %[[V:.*]] = vector.shape_cast %{{.*}} : vector<16xf32> to vector<2x4x2xf32>
//       CHECK:   %[[R:.*]] = vector.extract %[[V]][1] : vector<2x4x2xf32>
//       CHECK:   return %[[R]] : vector<4x2xf32>
func @fold_extract_shapecast_negative(%arg0 : vector<16xf32>,
                             %arg1 : vector<8x4x2xf32>) -> vector<4x2xf32> {
  %0 = vector.shape_cast %arg0 : vector<16xf32> to vector<2x4x2xf32>
  %r = vector.extract %0[1] : vector<2x4x2xf32>
  return %r : vector<4x2xf32>
}

// -----

// CHECK-LABEL: dont_fold_expand_collapse
//       CHECK:   %[[A:.*]] = vector.shape_cast %{{.*}} : vector<1x1x64xf32> to vector<1x1x8x8xf32>
//       CHECK:   %[[B:.*]] = vector.shape_cast %{{.*}} : vector<1x1x8x8xf32> to vector<8x8xf32>
//       CHECK:   return %[[B]] : vector<8x8xf32>
func @dont_fold_expand_collapse(%arg0: vector<1x1x64xf32>) -> vector<8x8xf32> {
    %0 = vector.shape_cast %arg0 : vector<1x1x64xf32> to vector<1x1x8x8xf32>
    %1 = vector.shape_cast %0 : vector<1x1x8x8xf32> to vector<8x8xf32>
    return %1 : vector<8x8xf32>
}

// -----

// CHECK-LABEL: fold_vector_transfers
func @fold_vector_transfers(%A: memref<?x8xf32>) -> (vector<4x8xf32>, vector<4x9xf32>) {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f32

  // CHECK: vector.transfer_read %{{.*}} {in_bounds = [false, true]}
  %1 = vector.transfer_read %A[%c0, %c0], %f0 : memref<?x8xf32>, vector<4x8xf32>

  // CHECK: vector.transfer_write %{{.*}} {in_bounds = [false, true]}
  vector.transfer_write %1, %A[%c0, %c0] : vector<4x8xf32>, memref<?x8xf32>

  // Both dims may be out-of-bounds, attribute is elided.
  // CHECK: vector.transfer_read %{{.*}}
  // CHECK-NOT: in_bounds
  %2 = vector.transfer_read %A[%c0, %c0], %f0 : memref<?x8xf32>, vector<4x9xf32>

  // Both dims may be out-of-bounds, attribute is elided.
  // CHECK: vector.transfer_write %{{.*}}
  // CHECK-NOT: in_bounds
  vector.transfer_write %2, %A[%c0, %c0] : vector<4x9xf32>, memref<?x8xf32>

  // CHECK: return
  return %1, %2 : vector<4x8xf32>, vector<4x9xf32>
}

// -----

// CHECK-LABEL: bitcast_folding
//  CHECK-SAME:   %[[A:.*]]: vector<4x8xf32>
//  CHECK-SAME:   %[[B:.*]]: vector<2xi32>
//  CHECK:        return %[[A]], %[[B]] : vector<4x8xf32>, vector<2xi32>
func @bitcast_folding(%I1: vector<4x8xf32>, %I2: vector<2xi32>) -> (vector<4x8xf32>, vector<2xi32>) {
  %0 = vector.bitcast %I1 : vector<4x8xf32> to vector<4x8xf32>
  %1 = vector.bitcast %I2 : vector<2xi32> to vector<4xi16>
  %2 = vector.bitcast %1 : vector<4xi16> to vector<2xi32>
  return %0, %2 : vector<4x8xf32>, vector<2xi32>
}

// CHECK-LABEL: func @bitcast_f16_to_f32
//              bit pattern: 0x40004000
//       CHECK-DAG: %[[CST1:.+]] = arith.constant dense<2.00390625> : vector<4xf32>
//              bit pattern: 0x00000000
//       CHECK-DAG: %[[CST0:.+]] = arith.constant dense<0.000000e+00> : vector<4xf32>
//       CHECK: return %[[CST0]], %[[CST1]]
func @bitcast_f16_to_f32() -> (vector<4xf32>, vector<4xf32>) {
  %cst0 = arith.constant dense<0.0> : vector<8xf16> // bit pattern: 0x0000
  %cst1 = arith.constant dense<2.0> : vector<8xf16> // bit pattern: 0x4000
  %cast0 = vector.bitcast %cst0: vector<8xf16> to vector<4xf32>
  %cast1 = vector.bitcast %cst1: vector<8xf16> to vector<4xf32>
  return %cast0, %cast1: vector<4xf32>, vector<4xf32>
}

// -----

// CHECK-LABEL: broadcast_folding1
//       CHECK: %[[CST:.*]] = arith.constant dense<42> : vector<4xi32>
//   CHECK-NOT: vector.broadcast
//       CHECK: return %[[CST]]
func @broadcast_folding1() -> vector<4xi32> {
  %0 = arith.constant 42 : i32
  %1 = vector.broadcast %0 : i32 to vector<4xi32>
  return %1 : vector<4xi32>
}

// -----

// CHECK-LABEL: @broadcast_folding2
//       CHECK: %[[CST:.*]] = arith.constant dense<42> : vector<4x16xi32>
//   CHECK-NOT: vector.broadcast
//       CHECK: return %[[CST]]
func @broadcast_folding2() -> vector<4x16xi32> {
  %0 = arith.constant 42 : i32
  %1 = vector.broadcast %0 : i32 to vector<16xi32>
  %2 = vector.broadcast %1 : vector<16xi32> to vector<4x16xi32>
  return %2 : vector<4x16xi32>
}

// -----

// CHECK-LABEL: @fold_consecutive_broadcasts(
//  CHECK-SAME:                              %[[ARG0:.*]]: i32
//       CHECK: %[[RESULT:.*]] = vector.broadcast %[[ARG0]] : i32 to vector<4x16xi32>
//       CHECK: return %[[RESULT]]
func @fold_consecutive_broadcasts(%a : i32) -> vector<4x16xi32> {
  %1 = vector.broadcast %a : i32 to vector<16xi32>
  %2 = vector.broadcast %1 : vector<16xi32> to vector<4x16xi32>
  return %2 : vector<4x16xi32>
}

// -----

// CHECK-LABEL: shape_cast_constant
//       CHECK-DAG: %[[CST1:.*]] = arith.constant dense<1> : vector<3x4x2xi32>
//       CHECK-DAG: %[[CST0:.*]] = arith.constant dense<2.000000e+00> : vector<20x2xf32>
//       CHECK: return %[[CST0]], %[[CST1]] : vector<20x2xf32>, vector<3x4x2xi32>
func @shape_cast_constant() -> (vector<20x2xf32>, vector<3x4x2xi32>) {
  %cst = arith.constant dense<2.000000e+00> : vector<5x4x2xf32>
  %cst_1 = arith.constant dense<1> : vector<12x2xi32>
  %0 = vector.shape_cast %cst : vector<5x4x2xf32> to vector<20x2xf32>
  %1 = vector.shape_cast %cst_1 : vector<12x2xi32> to vector<3x4x2xi32>
  return %0, %1 : vector<20x2xf32>, vector<3x4x2xi32>
}

// -----

// CHECK-LABEL: extract_strided_constant
//       CHECK-DAG: %[[CST1:.*]] = arith.constant dense<1> : vector<2x13x3xi32>
//       CHECK-DAG: %[[CST0:.*]] = arith.constant dense<2.000000e+00> : vector<12x2xf32>
//       CHECK: return %[[CST0]], %[[CST1]] : vector<12x2xf32>, vector<2x13x3xi32>
func @extract_strided_constant() -> (vector<12x2xf32>, vector<2x13x3xi32>) {
  %cst = arith.constant dense<2.000000e+00> : vector<29x7xf32>
  %cst_1 = arith.constant dense<1> : vector<4x37x9xi32>
  %0 = vector.extract_strided_slice %cst
    {offsets = [2, 3], sizes = [12, 2], strides = [1, 1]}
      : vector<29x7xf32> to vector<12x2xf32>
  %1 = vector.extract_strided_slice %cst_1
    {offsets = [1, 2, 5], sizes = [2, 13, 3], strides = [1, 1, 1]}
      : vector<4x37x9xi32> to vector<2x13x3xi32>
  return %0, %1 : vector<12x2xf32>, vector<2x13x3xi32>
}

// -----

// CHECK-LABEL: extract_strided_broadcast
//       CHECK:   %[[B:.*]] = vector.broadcast %{{.*}} : vector<4xf16> to vector<2x4xf16>
//  CHECK-NEXT:   return %[[B]] : vector<2x4xf16>
func @extract_strided_broadcast(%arg0: vector<4xf16>) -> vector<2x4xf16> {
 %0 = vector.broadcast %arg0 : vector<4xf16> to vector<16x4xf16>
 %1 = vector.extract_strided_slice %0
  {offsets = [0, 0], sizes = [2, 4], strides = [1, 1]} :
  vector<16x4xf16> to vector<2x4xf16>
  return %1 : vector<2x4xf16>
}

// -----

// CHECK-LABEL: extract_strided_broadcast2
//       CHECK:   %[[E:.*]] = vector.extract_strided_slice %{{.*}} {offsets = [0], sizes = [2], strides = [1]} : vector<4xf16> to vector<2xf16>
//  CHECK-NEXT:   %[[B:.*]] = vector.broadcast %[[E]] : vector<2xf16> to vector<2x2xf16>
//  CHECK-NEXT:   return %[[B]] : vector<2x2xf16>
func @extract_strided_broadcast2(%arg0: vector<4xf16>) -> vector<2x2xf16> {
 %0 = vector.broadcast %arg0 : vector<4xf16> to vector<16x4xf16>
 %1 = vector.extract_strided_slice %0
  {offsets = [0, 0], sizes = [2, 2], strides = [1, 1]} :
  vector<16x4xf16> to vector<2x2xf16>
  return %1 : vector<2x2xf16>
}

// -----

// CHECK-LABEL: consecutive_shape_cast
//       CHECK:   %[[C:.*]] = vector.shape_cast %{{.*}} : vector<16xf16> to vector<4x4xf16>
//  CHECK-NEXT:   return %[[C]] : vector<4x4xf16>
func @consecutive_shape_cast(%arg0: vector<16xf16>) -> vector<4x4xf16> {
  %0 = vector.shape_cast %arg0 : vector<16xf16> to vector<2x8xf16>
  %1 = vector.shape_cast %0 : vector<2x8xf16> to vector<4x4xf16>
  return %1 : vector<4x4xf16>
}

// -----

// CHECK-LABEL: func @dead_transfer_op
//   CHECK-NOT:   vector.transfer_read
//   CHECK-NOT:   vector.transfer_write
//       CHECK:   return
func @dead_transfer_op(%arg0 : tensor<4x4xf32>, %arg1 : memref<4x4xf32>,
                       %v0 : vector<1x4xf32>) {
  %c0 = arith.constant 0 : index
  %cf0 = arith.constant 0.0 : f32
  %r = vector.transfer_read %arg1[%c0, %c0], %cf0 :
    memref<4x4xf32>, vector<1x4xf32>
  %w = vector.transfer_write %v0, %arg0[%c0, %c0] :
    vector<1x4xf32>, tensor<4x4xf32>
  return
}

// -----

// CHECK-LABEL: func @dead_load
//   CHECK-NOT:   vector.maskedload
//   CHECK-NOT:   vector.gather
//   CHECK-NOT:   vector.expandload
//       CHECK:   return
func @dead_load(%base: memref<?xf32>, %indices: vector<16xi32>,
                          %mask: vector<16xi1>, %passthru: vector<16xf32>) {
  %c0 = arith.constant 0 : index
  %0 = vector.maskedload %base[%c0], %mask, %passthru :
    memref<?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  %1 = vector.gather %base[%c0][%indices], %mask, %passthru :
    memref<?xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  %2 = vector.expandload %base[%c0], %mask, %passthru :
    memref<?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  return
}

// -----

#contraction_accesses0 = [
  affine_map<(i, j, k) -> (i, k)>,
  affine_map<(i, j, k) -> (k, j)>,
  affine_map<(i, j, k) -> (i, j)>
]
#contraction_trait0 = {
  indexing_maps = #contraction_accesses0,
  iterator_types = ["parallel", "parallel", "reduction"]
}

// CHECK-LABEL: func @contractions
//  CHECK-SAME:   %[[A:[0-9a-zA-Z]+]]: vector<2x3xf32>
//  CHECK-SAME:   %[[B:[0-9a-zA-Z]+]]: vector<3x4xf32>
//  CHECK-SAME:   %[[C:[0-9a-zA-Z]+]]: vector<2x4xf32>
//  CHECK-SAME:   %[[A_I8:[0-9a-zA-Z]+]]: vector<2x3xi8>
//  CHECK-SAME:   %[[B_I8:[0-9a-zA-Z]+]]: vector<3x4xi8>
//  CHECK-SAME:   %[[C_I8:[0-9a-zA-Z]+]]: vector<2x4xi8>
func @contractions(%a: vector<2x3xf32>, %b: vector<3x4xf32>, %c: vector<2x4xf32>,
                   %a_i8: vector<2x3xi8>, %b_i8: vector<3x4xi8>, %c_i8: vector<2x4xi8>)
  -> (vector<2x4xf32>, vector<2x4xi8>)
{
  // CHECK-NOT: arith.constant
  %vf_0 = arith.constant dense <0.0>: vector<2x4xf32>
  // CHECK-NOT: arith.addf
  //     CHECK: %[[D:.*]] = vector.contract {{.*}} %[[A]], %[[B]], %[[C]]
  %0 = vector.contract #contraction_trait0 %a, %b, %vf_0:
    vector<2x3xf32>, vector<3x4xf32> into vector<2x4xf32>
  // CHECK-NOT: arith.addf
  %1 = arith.addf %0, %c: vector<2x4xf32>

  // CHECK-NOT: arith.constant
  %vi8_0 = arith.constant dense <0>: vector<2x4xi8>
  // CHECK-NOT: arith.addi
  //     CHECK: %[[D_I8:.*]] = vector.contract {{.*}} %[[A_I8]], %[[B_I8]], %[[C_I8]]
  %i8_0 = vector.contract #contraction_trait0 %a_i8, %b_i8, %vi8_0:
    vector<2x3xi8>, vector<3x4xi8> into vector<2x4xi8>
  // CHECK-NOT: arith.addi
  %i8_1 = arith.addi %i8_0, %c_i8: vector<2x4xi8>

  // CHECK: return %[[D]], %[[D_I8]]
  return %1, %i8_1: vector<2x4xf32>, vector<2x4xi8>
}

// -----

// CHECK-LABEL: func @transfer_folding_1
//  CHECK-SAME:   %[[T0:[0-9a-zA-Z]+]]: tensor<2x3x4xf32>
//  CHECK-SAME:   %[[T1:[0-9a-zA-Z]+]]: tensor<2x3x4xf32>
func @transfer_folding_1(%t0: tensor<2x3x4xf32>, %t1: tensor<2x3x4xf32>)
  -> (tensor<2x3x4xf32>, tensor<2x3x4xf32>, tensor<2x3x4xf32>)
{
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  %v = vector.transfer_read %t0[%c0, %c0, %c0], %pad {in_bounds = [true, true, true]} :
    tensor<2x3x4xf32>, vector<2x3x4xf32>

  %r0 = vector.transfer_write %v, %t1[%c0, %c0, %c0] {in_bounds = [true, true, true]} :
    vector<2x3x4xf32>, tensor<2x3x4xf32>

  %t2 = "test.constant"() { value = dense<6.0> : tensor<2x3x4xf32>} : () -> (tensor<2x3x4xf32>)
  %r1 = vector.transfer_write %v, %t2[%c0, %c0, %c0] {in_bounds = [true, true, true]} :
    vector<2x3x4xf32>, tensor<2x3x4xf32>


  // CHECK-NEXT: some_op_that_may_have_side_effects
  %t3 = "some_op_that_may_have_side_effects"() : () -> (tensor<2x3x4xf32>)
  %r2 = vector.transfer_write %v, %t0[%c0, %c0, %c0] {in_bounds = [true, true, true]} :
    vector<2x3x4xf32>, tensor<2x3x4xf32>

  // CHECK-NEXT: return %[[T0]], %[[T0]], %[[T0]]
  return %r0, %r1, %r2: tensor<2x3x4xf32>, tensor<2x3x4xf32>, tensor<2x3x4xf32>
}

// -----

// CHECK-LABEL: func @store_after_load_tensor
//  CHECK-SAME: (%[[ARG:.*]]: tensor<4x4xf32>)
//   CHECK-NOT:   vector.transfer_read
//   CHECK-NOT:   vector.transfer_write
//       CHECK:   return %[[ARG]] : tensor<4x4xf32>
func @store_after_load_tensor(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %cf0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %arg0[%c1, %c0], %cf0 :
    tensor<4x4xf32>, vector<1x4xf32>
  %w0 = vector.transfer_write %0, %arg0[%c1, %c0] :
    vector<1x4xf32>, tensor<4x4xf32>
  return %w0 : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: func @store_after_load_tensor_negative
//       CHECK:   vector.transfer_read
//       CHECK:   vector.transfer_write
//       CHECK:   return
func @store_after_load_tensor_negative(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32> {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %cf0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %arg0[%c1, %c0], %cf0 :
    tensor<4x4xf32>, vector<1x4xf32>
  %w0 = vector.transfer_write %0, %arg0[%c0, %c0] :
    vector<1x4xf32>, tensor<4x4xf32>
  return %w0 : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: func @store_to_load_tensor
//  CHECK-SAME: (%[[ARG:.*]]: tensor<4x4xf32>, %[[V0:.*]]: vector<1x4xf32>, %[[V1:.*]]: vector<1x4xf32>)
//   CHECK-NOT:   vector.transfer_write
//   CHECK-NOT:   vector.transfer_read
//       CHECK:   return %[[V0]] : vector<1x4xf32>
func @store_to_load_tensor(%arg0 : tensor<4x4xf32>,
  %v0 : vector<1x4xf32>, %v1 : vector<1x4xf32>) -> vector<1x4xf32> {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %cf0 = arith.constant 0.0 : f32
  %w0 = vector.transfer_write %v0, %arg0[%c1, %c0] {in_bounds = [true, true]} :
    vector<1x4xf32>, tensor<4x4xf32>
  %w1 = vector.transfer_write %v1, %w0[%c2, %c0] {in_bounds = [true, true]} :
    vector<1x4xf32>, tensor<4x4xf32>
  %0 = vector.transfer_read %w1[%c1, %c0], %cf0 {in_bounds = [true, true]} :
    tensor<4x4xf32>, vector<1x4xf32>
  return %0 : vector<1x4xf32>
}

// -----

// CHECK-LABEL: func @store_to_load_negative_tensor
//       CHECK:   vector.transfer_write
//       CHECK:   vector.transfer_write
//       CHECK:   %[[V:.*]] = vector.transfer_read
//       CHECK:   return %[[V]] : vector<1x4xf32>
func @store_to_load_negative_tensor(%arg0 : tensor<4x4xf32>,
  %v0 : vector<1x4xf32>, %v1 : vector<1x4xf32>, %i : index) -> vector<1x4xf32> {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %cf0 = arith.constant 0.0 : f32
  %w0 = vector.transfer_write %v0, %arg0[%c1, %c0] {in_bounds = [true, true]} :
    vector<1x4xf32>, tensor<4x4xf32>
  %w1 = vector.transfer_write %v0, %w0[%i, %i] {in_bounds = [true, true]} :
    vector<1x4xf32>, tensor<4x4xf32>
  %0 = vector.transfer_read %w1[%c1, %c0], %cf0 {in_bounds = [true, true]} :
    tensor<4x4xf32>, vector<1x4xf32>
  return %0 : vector<1x4xf32>
}

// -----


// CHECK-LABEL: func @dead_store_tensor
//   CHECK-DAG:      %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:      %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG:      %[[C2:.*]] = arith.constant 2 : index
//   CHECK-NOT:   vector.transfer_write {{.*}}, {{.*}}[%[[C1]], %[[C0]]
//       CHECK:   vector.transfer_write {{.*}}, {{.*}}[%[[C2]], %[[C0]]
//       CHECK:   %[[VTW:.*]] = vector.transfer_write {{.*}}, {{.*}}[%[[C1]], %[[C0]]
//       CHECK:   return %[[VTW]] : tensor<4x4xf32>
func @dead_store_tensor(%arg0 : tensor<4x4xf32>,
  %v0 : vector<1x4xf32>, %v1 : vector<1x4xf32>, %i : index) -> tensor<4x4xf32> {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %cf0 = arith.constant 0.0 : f32
  %w0 = vector.transfer_write %v0, %arg0[%c1, %c0] {in_bounds = [true, true]} :
    vector<1x4xf32>, tensor<4x4xf32>
  %w1 = vector.transfer_write %v0, %w0[%c2, %c0] {in_bounds = [true, true]} :
    vector<1x4xf32>, tensor<4x4xf32>
  %w2 = vector.transfer_write %v1, %w1[%c1, %c0] {in_bounds = [true, true]} :
    vector<1x4xf32>, tensor<4x4xf32>
  return %w2 : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: func @dead_store_tensor_negative
//   CHECK-DAG:      %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:      %[[C1:.*]] = arith.constant 1 : index
//       CHECK:   vector.transfer_write
//       CHECK:   vector.transfer_write
//       CHECK:   vector.transfer_read
//       CHECK:   %[[VTW:.*]] = vector.transfer_write {{.*}}, {{.*}}[%[[C1]], %[[C0]]]
//       CHECK:   return %[[VTW]] : tensor<4x4xf32>
func @dead_store_tensor_negative(%arg0 : tensor<4x4xf32>,
  %v0 : vector<1x4xf32>, %v1 : vector<1x4xf32>, %i : index) -> tensor<4x4xf32> {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %cf0 = arith.constant 0.0 : f32
  %w0 = vector.transfer_write %v0, %arg0[%c1, %c0] {in_bounds = [true, true]} :
    vector<1x4xf32>, tensor<4x4xf32>
  %w1 = vector.transfer_write %v0, %w0[%c2, %c0] {in_bounds = [true, true]} :
    vector<1x4xf32>, tensor<4x4xf32>
  %0 = vector.transfer_read %w1[%i, %i], %cf0 {in_bounds = [true, true]} :
    tensor<4x4xf32>, vector<1x4xf32>
  %x = arith.addf %0, %0 : vector<1x4xf32>
  %w2 = vector.transfer_write %x, %w0[%c1, %c0] {in_bounds = [true, true]} :
    vector<1x4xf32>, tensor<4x4xf32>
  return %w2 : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: func @transfer_read_of_extract_slice(
//  CHECK-SAME:     %[[t:.*]]: tensor<?x?xf32>, %[[s1:.*]]: index, %[[s2:.*]]: index
//   CHECK-DAG:   %[[c4:.*]] = arith.constant 4 : index
//   CHECK-DAG:   %[[c8:.*]] = arith.constant 8 : index
//       CHECK:   %[[add:.*]] = arith.addi %[[s1]], %[[c4]]
//       CHECK:   %[[r:.*]] = vector.transfer_read %[[t]][%[[c8]], %[[add]]], %{{.*}} {in_bounds = [true, true]} : tensor<?x?xf32>, vector<5x6xf32>
//       CHECK:   return %[[r]]
func @transfer_read_of_extract_slice(%t : tensor<?x?xf32>, %s1 : index, %s2 : index) -> vector<5x6xf32> {
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant 0.0 : f32
  %0 = tensor.extract_slice %t[5, %s1] [10, %s2] [1, 1] : tensor<?x?xf32> to tensor<10x?xf32>
  %1 = vector.transfer_read %0[%c3, %c4], %cst {in_bounds = [true, true]} : tensor<10x?xf32>, vector<5x6xf32>
  return %1 : vector<5x6xf32>
}

// -----

// CHECK-LABEL: func @transfer_read_of_extract_slice_rank_reducing(
//  CHECK-SAME:     %[[t:.*]]: tensor<?x?x?xf32>, %[[s1:.*]]: index, %[[s2:.*]]: index
//   CHECK-DAG:   %[[c3:.*]] = arith.constant 3 : index
//   CHECK-DAG:   %[[c5:.*]] = arith.constant 5 : index
//   CHECK-DAG:   %[[c10:.*]] = arith.constant 10 : index
//       CHECK:   %[[add:.*]] = arith.addi %[[s1]], %[[c3]]
//       CHECK:   %[[r:.*]] = vector.transfer_read %[[t]][%[[c5]], %[[add]], %[[c10]]], %{{.*}} {in_bounds = [true, true]} : tensor<?x?x?xf32>, vector<5x6xf32>
//       CHECK:   return %[[r]]
func @transfer_read_of_extract_slice_rank_reducing(%t : tensor<?x?x?xf32>, %s1 : index, %s2 : index) -> vector<5x6xf32> {
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant 0.0 : f32
  %0 = tensor.extract_slice %t[5, %s1, 6] [1, %s2, 12] [1, 1, 1] : tensor<?x?x?xf32> to tensor<?x12xf32>
  %1 = vector.transfer_read %0[%c3, %c4], %cst {in_bounds = [true, true]} : tensor<?x12xf32>, vector<5x6xf32>
  return %1 : vector<5x6xf32>
}

// -----

// CHECK-LABEL: func @transfer_read_of_extract_slice_illegal_rank_reducing(
//       CHECK:   extract_slice
//       CHECK:   vector.transfer_read
func @transfer_read_of_extract_slice_illegal_rank_reducing(%t : tensor<?x?x?xf32>, %s1 : index, %s2 : index) -> vector<5x6xf32> {
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant 0.0 : f32
  %0 = tensor.extract_slice %t[5, %s1, 6] [%s2, 1, 12] [1, 1, 1] : tensor<?x?x?xf32> to tensor<?x12xf32>
  %1 = vector.transfer_read %0[%c3, %c4], %cst {in_bounds = [true, true]} : tensor<?x12xf32>, vector<5x6xf32>
  return %1 : vector<5x6xf32>
}

// -----

// CHECK-LABEL: func @insert_slice_of_transfer_write(
//  CHECK-SAME:     %[[t1:.*]]: tensor<?x12xf32>, %[[v:.*]]: vector<5x6xf32>, %[[s:.*]]: index
//       CHECK:   %[[c3:.*]] = arith.constant 3 : index
//       CHECK:   %[[r:.*]] = vector.transfer_write %[[v]], %[[t1]][%[[c3]], %[[s]]] {in_bounds = [true, true]} : vector<5x6xf32>, tensor<?x12xf32>
//       CHECK:   return %[[r]]
func @insert_slice_of_transfer_write(%t1 : tensor<?x12xf32>, %v : vector<5x6xf32>, %s : index, %t2 : tensor<5x6xf32>) -> tensor<?x12xf32> {
  %c0 = arith.constant 0 : index
  %0 = vector.transfer_write %v, %t2[%c0, %c0] {in_bounds = [true, true]} : vector<5x6xf32>, tensor<5x6xf32>
  %1 = tensor.insert_slice %0 into %t1[3, %s] [5, 6] [1, 1] : tensor<5x6xf32> into tensor<?x12xf32>
  return %1 : tensor<?x12xf32>
}

// -----

// CHECK-LABEL: func @insert_slice_of_transfer_write_illegal_rank_extending(
//       CHECK:   vector.transfer_write
//       CHECK:   insert_slice
func @insert_slice_of_transfer_write_illegal_rank_extending(%t1 : tensor<?x?x12xf32>, %v : vector<5x6xf32>, %s : index, %t2 : tensor<5x6xf32>) -> tensor<?x?x12xf32> {
  %c0 = arith.constant 0 : index
  %0 = vector.transfer_write %v, %t2[%c0, %c0] {in_bounds = [true, true]} : vector<5x6xf32>, tensor<5x6xf32>
  %1 = tensor.insert_slice %0 into %t1[4, 3, %s] [5, 1, 6] [1, 1, 1] : tensor<5x6xf32> into tensor<?x?x12xf32>
  return %1 : tensor<?x?x12xf32>
}

// -----

// CHECK-LABEL: func @insert_slice_of_transfer_write_rank_extending(
//  CHECK-SAME:     %[[t1:.*]]: tensor<?x?x12xf32>, %[[v:.*]]: vector<5x6xf32>, %[[s:.*]]: index
//   CHECK-DAG:   %[[c3:.*]] = arith.constant 3 : index
//   CHECK-DAG:   %[[c4:.*]] = arith.constant 4 : index
//       CHECK:   %[[r:.*]] = vector.transfer_write %[[v]], %[[t1]][%[[c4]], %[[c3]], %[[s]]] {in_bounds = [true, true]} : vector<5x6xf32>, tensor<?x?x12xf32>
//       CHECK:   return %[[r]]
func @insert_slice_of_transfer_write_rank_extending(%t1 : tensor<?x?x12xf32>, %v : vector<5x6xf32>, %s : index, %t2 : tensor<5x6xf32>) -> tensor<?x?x12xf32> {
  %c0 = arith.constant 0 : index
  %0 = vector.transfer_write %v, %t2[%c0, %c0] {in_bounds = [true, true]} : vector<5x6xf32>, tensor<5x6xf32>
  %1 = tensor.insert_slice %0 into %t1[4, 3, %s] [1, 5, 6] [1, 1, 1] : tensor<5x6xf32> into tensor<?x?x12xf32>
  return %1 : tensor<?x?x12xf32>
}

// -----

// CHECK-LABEL: func @vector_multi_reduction_single_parallel(
//  CHECK-SAME:     %[[v:.*]]: vector<2xf32>
func @vector_multi_reduction_single_parallel(%arg0: vector<2xf32>) -> vector<2xf32> {
    %0 = vector.multi_reduction <mul>, %arg0 [] : vector<2xf32> to vector<2xf32>

//       CHECK:     return %[[v]] : vector<2xf32>
    return %0 : vector<2xf32>
}

// -----

// CHECK-LABEL: func @insert_strided_slice_full_range
//  CHECK-SAME: %[[SOURCE:.+]]: vector<16x16xf16>, %{{.+}}: vector<16x16xf16>
func @insert_strided_slice_full_range(%source: vector<16x16xf16>, %dest: vector<16x16xf16>) -> vector<16x16xf16> {
  %0 = vector.insert_strided_slice %source, %dest {offsets = [0, 0], strides = [1, 1]} : vector<16x16xf16> into vector<16x16xf16>
  // CHECK: return %[[SOURCE]]
  return %0: vector<16x16xf16>
}

// -----

// CHECK-LABEL: extract_strided_splat
//       CHECK:   %[[B:.*]] = splat %{{.*}} : vector<2x4xf16>
//  CHECK-NEXT:   return %[[B]] : vector<2x4xf16>
func @extract_strided_splat(%arg0: f16) -> vector<2x4xf16> {
 %0 = splat %arg0 : vector<16x4xf16>
 %1 = vector.extract_strided_slice %0
  {offsets = [1, 0], sizes = [2, 4], strides = [1, 1]} :
  vector<16x4xf16> to vector<2x4xf16>
  return %1 : vector<2x4xf16>
}

// -----

// CHECK-LABEL: func @insert_extract_to_broadcast
//  CHECK-SAME: (%[[ARG0:.*]]: vector<1x1x4xf32>, %[[ARG1:.*]]: vector<4xf32>)
//       CHECK:   %[[V0:.*]] = vector.extract %[[ARG0]][0, 0] : vector<1x1x4xf32>
//       CHECK:   %[[V1:.*]] = vector.broadcast %[[ARG1]] : vector<4xf32> to vector<1x1x4xf32>
//       CHECK:   return %[[V0]], %[[V1]] : vector<4xf32>, vector<1x1x4xf32>
func @insert_extract_to_broadcast(%arg0 : vector<1x1x4xf32>,
  %arg1 : vector<4xf32>) -> (vector<4xf32>, vector<1x1x4xf32>) {
  %0 = vector.extract %arg0[0, 0] : vector<1x1x4xf32>
  %1 = vector.insert %arg1, %arg0 [0, 0] : vector<4xf32> into vector<1x1x4xf32>
  return %0, %1 : vector<4xf32>, vector<1x1x4xf32>
}

// -----

// CHECK-LABEL: extract_constant
//       CHECK-DAG: %[[CST1:.*]] = arith.constant 1 : i32
//       CHECK-DAG: %[[CST0:.*]] = arith.constant dense<2.000000e+00> : vector<7xf32>
//       CHECK: return %[[CST0]], %[[CST1]] : vector<7xf32>, i32
func @extract_constant() -> (vector<7xf32>, i32) {
  %cst = arith.constant dense<2.000000e+00> : vector<29x7xf32>
  %cst_1 = arith.constant dense<1> : vector<4x37x9xi32>
  %0 = vector.extract %cst[2] : vector<29x7xf32>
  %1 = vector.extract %cst_1[1, 4, 5] : vector<4x37x9xi32>
  return %0, %1 : vector<7xf32>, i32
}

// -----

// CHECK-LABEL: extract_extract_strided
//  CHECK-SAME: %[[A:.*]]: vector<32x16x4xf16>
//       CHECK: %[[V:.*]] = vector.extract %[[A]][9, 7] : vector<32x16x4xf16>
//       CHECK: return %[[V]] : vector<4xf16>
func @extract_extract_strided(%arg0: vector<32x16x4xf16>) -> vector<4xf16> {
 %1 = vector.extract_strided_slice %arg0
  {offsets = [7, 3], sizes = [10, 8], strides = [1, 1]} :
  vector<32x16x4xf16> to vector<10x8x4xf16>
  %2 = vector.extract %1[2, 4] : vector<10x8x4xf16>
  return %2 : vector<4xf16>
}

// -----

// CHECK-LABEL: extract_insert_strided
//  CHECK-SAME: %[[A:.*]]: vector<6x4xf32>
//       CHECK: %[[V:.*]] = vector.extract %[[A]][0, 2] : vector<6x4xf32>
//       CHECK: return %[[V]] : f32
func @extract_insert_strided(%a: vector<6x4xf32>, %b: vector<8x16xf32>)
  -> f32 {
  %0 = vector.insert_strided_slice %a, %b {offsets = [2, 2], strides = [1, 1]}
    : vector<6x4xf32> into vector<8x16xf32>
  %2 = vector.extract %0[2, 4] : vector<8x16xf32>
  return %2 : f32
}

// -----

// CHECK-LABEL: extract_insert_rank_reduce
//  CHECK-SAME: %[[A:.*]]: vector<4xf32>
//       CHECK: %[[V:.*]] = vector.extract %[[A]][2] : vector<4xf32>
//       CHECK: return %[[V]] : f32
func @extract_insert_rank_reduce(%a: vector<4xf32>, %b: vector<8x16xf32>)
  -> f32 {
  %0 = vector.insert_strided_slice %a, %b {offsets = [2, 2], strides = [1]}
    : vector<4xf32> into vector<8x16xf32>
  %2 = vector.extract %0[2, 4] : vector<8x16xf32>
  return %2 : f32
}

// -----

// CHECK-LABEL: extract_insert_negative
//       CHECK: vector.insert_strided_slice
//       CHECK: vector.extract
func @extract_insert_negative(%a: vector<2x15xf32>, %b: vector<12x8x16xf32>)
  -> vector<16xf32> {
  %0 = vector.insert_strided_slice %a, %b {offsets = [4, 2, 0], strides = [1, 1]}
    : vector<2x15xf32> into vector<12x8x16xf32>
  %2 = vector.extract %0[4, 2] : vector<12x8x16xf32>
  return %2 : vector<16xf32>
}

// -----

// CHECK-LABEL: extract_insert_chain
//  CHECK-SAME: (%[[A:.*]]: vector<2x16xf32>, %[[B:.*]]: vector<12x8x16xf32>, %[[C:.*]]: vector<2x16xf32>)
//       CHECK: %[[V:.*]] = vector.extract %[[C]][0] : vector<2x16xf32>
//       CHECK: return %[[V]] : vector<16xf32>
func @extract_insert_chain(%a: vector<2x16xf32>, %b: vector<12x8x16xf32>, %c: vector<2x16xf32>)
  -> vector<16xf32> {
  %0 = vector.insert_strided_slice %c, %b {offsets = [4, 2, 0], strides = [1, 1]}
    : vector<2x16xf32> into vector<12x8x16xf32>
  %1 = vector.insert_strided_slice %a, %0 {offsets = [0, 2, 0], strides = [1, 1]}
    : vector<2x16xf32> into vector<12x8x16xf32>
  %2 = vector.extract %1[4, 2] : vector<12x8x16xf32>
  return %2 : vector<16xf32>
}

// -----

// CHECK-LABEL: extract_extract_strided2
//  CHECK-SAME: %[[A:.*]]: vector<2x4xf32>
//       CHECK: %[[V:.*]] = vector.extract %[[A]][1] : vector<2x4xf32>
//       CHECK: return %[[V]] : vector<4xf32>
func @extract_extract_strided2(%A: vector<2x4xf32>)
  -> (vector<4xf32>) {
 %0 = vector.extract_strided_slice %A {offsets = [1, 0], sizes = [1, 4], strides = [1, 1]} : vector<2x4xf32> to vector<1x4xf32>
 %1 = vector.extract %0[0] : vector<1x4xf32>
 return %1 : vector<4xf32>
}
