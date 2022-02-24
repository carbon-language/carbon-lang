// RUN: mlir-opt <%s | mlir-opt | FileCheck %s

// CHECK-LABEL: func @cast(
func @cast(%arg0: tensor<*xf32>, %arg1 : tensor<4x4xf32>, %arg2: tensor<?x?xf32>) {
  // CHECK: tensor.cast %arg0 : tensor<*xf32> to tensor<?x?xf32>
  %0 = tensor.cast %arg0 : tensor<*xf32> to tensor<?x?xf32>
  // CHECK: tensor.cast %arg1 : tensor<4x4xf32> to tensor<*xf32>
  %1 = tensor.cast %arg1 : tensor<4x4xf32> to tensor<*xf32>
  // CHECK: tensor.cast %arg2 : tensor<?x?xf32> to tensor<4x?xf32>
  %2 = tensor.cast %arg2 : tensor<?x?xf32> to tensor<4x?xf32>
  // CHECK: tensor.cast %2 : tensor<4x?xf32> to tensor<?x?xf32>
  %3 = tensor.cast %2 : tensor<4x?xf32> to tensor<?x?xf32>
  return
}

// CHECK-LABEL:   func @extract(
// CHECK-SAME:                  %[[TENSOR:.*]]: tensor<?x?x?xf32>,
// CHECK-SAME:                  %[[INDEX:.*]]: index) {
func @extract(%arg0: tensor<?x?x?xf32>, %arg1: index) {
  // CHECK: tensor.extract %[[TENSOR]][%[[INDEX]], %[[INDEX]], %[[INDEX]]] : tensor<?x?x?xf32>
  %0 = tensor.extract %arg0[%arg1, %arg1, %arg1] : tensor<?x?x?xf32>
  return
}

// CHECK-LABEL:   func @insert(
// CHECK-SAME:                  %[[SCALAR:.*]]: f32
// CHECK-SAME:                  %[[INDEX:.*]]: index
// CHECK-SAME:                  %[[DEST1:.*]]: tensor<?x?x?xf32>
// CHECK-SAME:                  %[[DEST2:.*]]: tensor<*xf32>
func @insert(%arg0: f32, %arg1: index, %arg2: tensor<?x?x?xf32>, %arg3: tensor<*xf32>) {
  // CHECK: tensor.insert %[[SCALAR]] into %[[DEST1]][%[[INDEX]], %[[INDEX]], %[[INDEX]]] : tensor<?x?x?xf32>
  %0 = tensor.insert %arg0 into %arg2[%arg1, %arg1, %arg1] : tensor<?x?x?xf32>
  // CHECK: tensor.insert %[[SCALAR]] into %[[DEST2]][%[[INDEX]], %[[INDEX]], %[[INDEX]]] : tensor<*xf32>
  %1 = tensor.insert %arg0 into %arg3[%arg1, %arg1, %arg1] : tensor<*xf32>
  return
}

// CHECK-LABEL: func @tensor.from_elements() {
func @tensor.from_elements() {
  %c0 = "arith.constant"() {value = 0: index} : () -> index
  // CHECK: tensor.from_elements %c0 : tensor<1xindex>
  %0 = tensor.from_elements %c0 : tensor<1xindex>

  %c1 = "arith.constant"() {value = 1: index} : () -> index
  // CHECK: tensor.from_elements %c0, %c1 : tensor<2xindex>
  %1 = tensor.from_elements %c0, %c1 : tensor<2xindex>

  %c0_f32 = "arith.constant"() {value = 0.0: f32} : () -> f32
  // CHECK: [[C0_F32:%.*]] = arith.constant
  // CHECK: tensor.from_elements [[C0_F32]] : tensor<1xf32>
  %2 = tensor.from_elements %c0_f32 : tensor<1xf32>

  // CHECK: tensor.from_elements : tensor<0xindex>
  %3 = tensor.from_elements : tensor<0xindex>

  // CHECK: tensor.from_elements %c0, %c1, %c0, %c1, %c0, %c1 : tensor<2x3xindex>
  %4 = tensor.from_elements %c0, %c1, %c0, %c1, %c0, %c1 : tensor<2x3xindex>

  // CHECK: tensor.from_elements %c0 : tensor<index>
  %5 = tensor.from_elements %c0 : tensor<index>
  return
}

// CHECK-LABEL: @tensor.generate
func @tensor.generate(%m : index, %n : index)
    -> tensor<?x3x?xf32> {
  %tnsr = tensor.generate %m, %n {
    ^bb0(%i : index, %j : index, %k : index):
      %elem = arith.constant 8.0 : f32
      tensor.yield %elem : f32
  } : tensor<?x3x?xf32>
  return %tnsr : tensor<?x3x?xf32>
}

// CHECK-LABEL: func @tensor_reshape
func @tensor_reshape(%unranked: tensor<*xf32>, %shape1: tensor<1xi32>,
         %shape2: tensor<2xi32>, %shape3: tensor<?xi32>) -> tensor<*xf32> {
  %dyn_vec = tensor.reshape %unranked(%shape1)
               : (tensor<*xf32>, tensor<1xi32>) -> tensor<?xf32>
  %dyn_mat = tensor.reshape %dyn_vec(%shape2)
               : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  %new_unranked = tensor.reshape %dyn_mat(%shape3)
               : (tensor<?x?xf32>, tensor<?xi32>) -> tensor<*xf32>
  return %new_unranked : tensor<*xf32>
}

// CHECK-LABEL: func @slice({{.*}}) {
func @slice(%t: tensor<8x16x4xf32>, %idx : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK: tensor.extract_slice
  // CHECK-SAME: tensor<8x16x4xf32> to tensor<?x?x?xf32>
  %1 = tensor.extract_slice %t[%c0, %c0, %c0][%idx, %idx, %idx][%c1, %c1, %c1]
    : tensor<8x16x4xf32> to tensor<?x?x?xf32>

  // CHECK: tensor.extract_slice
  // CHECK-SAME: tensor<8x16x4xf32> to tensor<4x4x4xf32>
  %2 = tensor.extract_slice %t[0, 2, 0][4, 4, 4][1, 1, 1]
    : tensor<8x16x4xf32> to tensor<4x4x4xf32>

  // CHECK: tensor.extract_slice
  // CHECK-SAME: tensor<8x16x4xf32> to tensor<4x4xf32>
  %3 = tensor.extract_slice %t[0, 2, 0][4, 1, 4][1, 1, 1]
    : tensor<8x16x4xf32> to tensor<4x4xf32>

  return
}

// -----

// CHECK-LABEL: func @insert_slice({{.*}}) {
func @insert_slice(
    %t: tensor<8x16x4xf32>,
    %td: tensor<8x?x4xf32>,
    %t2: tensor<16x32x8xf32>,
    %t3: tensor<4x4xf32>,
    %idx : index,
    %sz : index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK: tensor.insert_slice
  // CHECK-SAME: tensor<8x16x4xf32> into tensor<16x32x8xf32>
  %1 = tensor.insert_slice %t into %t2[%c0, %c0, %c0][8, 16, 4][%c1, %c1, %c1]
    : tensor<8x16x4xf32> into tensor<16x32x8xf32>

  // CHECK: tensor.insert_slice
  // CHECK-SAME: tensor<8x16x4xf32> into tensor<16x32x8xf32>
  %2 = tensor.insert_slice %t into %t2[%c0, %idx, %c0][8, 16, 4][%c1, 1, %c1]
    : tensor<8x16x4xf32> into tensor<16x32x8xf32>

  // CHECK: tensor.insert_slice
  // CHECK-SAME: tensor<4x4xf32> into tensor<8x16x4xf32>
  %3 = tensor.insert_slice %t3 into %t[0, 2, 0][4, 1, 4][1, 1, 1]
    : tensor<4x4xf32> into tensor<8x16x4xf32>

  // CHECK: tensor.insert_slice
  // CHECK-SAME: tensor<8x?x4xf32> into tensor<8x16x4xf32>
  %4 = tensor.insert_slice %td into %t[0, %idx, 0][8, %sz, 4][1, 1, 1]
    : tensor<8x?x4xf32> into tensor<8x16x4xf32>

  return
}

// -----

func @tensor_reshape_zero_dim(%arg0 : tensor<1x1xf32>, %arg1 : tensor<f32>)
    -> (tensor<f32>, tensor<1x1xf32>) {
  %0 = tensor.collapse_shape %arg0 [] : tensor<1x1xf32> into tensor<f32>
  %1 = tensor.expand_shape %0 [] : tensor<f32> into tensor<1x1xf32>
  return %0, %1 : tensor<f32>, tensor<1x1xf32>
}
// CHECK-LABEL: func @tensor_reshape_zero_dim
//       CHECK:   tensor.collapse_shape %{{.*}} [] : tensor<1x1xf32> into tensor<f32>
//       CHECK:   tensor.expand_shape %{{.*}} [] : tensor<f32> into tensor<1x1xf32>

func @legal_collapsing_reshape_dynamic_tensor
  (%arg0: tensor<?x?x?x4x?xf32>) -> tensor<?x?x?xf32>
{
  %0 = tensor.collapse_shape %arg0 [[0], [1], [2, 3, 4]] :
    tensor<?x?x?x4x?xf32> into tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}
//      CHECK: func @legal_collapsing_reshape_dynamic_tensor
//      CHECK:   tensor.collapse_shape
// CHECK-SAME:    [0], [1], [2, 3, 4]

// -----

func @rank(%t : tensor<4x4x?xf32>) {
  // CHECK: %{{.*}} = tensor.rank %{{.*}} : tensor<4x4x?xf32>
  %0 = "tensor.rank"(%t) : (tensor<4x4x?xf32>) -> index

  // CHECK: %{{.*}} = tensor.rank %{{.*}} : tensor<4x4x?xf32>
  %1 = tensor.rank %t : tensor<4x4x?xf32>
  return
}
