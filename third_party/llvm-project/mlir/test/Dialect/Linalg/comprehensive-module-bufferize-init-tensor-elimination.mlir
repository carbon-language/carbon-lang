// RUN: mlir-opt %s -linalg-comprehensive-module-bufferize="allow-return-memref init-tensor-elimination" -canonicalize -split-input-file | FileCheck %s

// -----

//      CHECK: func @buffer_forwarding_conflict(
// CHECK-SAME:   %[[FUNC_ARG:[0-9a-zA-Z]*]]: memref<?xf32>
// CHECK-SAME:   %[[sz:[0-9a-zA-Z]*]]: index
func @buffer_forwarding_conflict(
  %t: tensor<?xf32> {linalg.buffer_layout = affine_map<(d0) -> (d0)>, linalg.inplaceable = true},
  %sz: index)
    -> (tensor<?xf32>, tensor<?xf32>)
{
  %f0 = arith.constant 0.0: f32
  // Alloc is needed for the **first** insert_slice (due to backward traversal during analysis).
  //     CHECK: %[[DIM:.*]] = memref.dim %[[FUNC_ARG]]
  // This allocs the whole dim to allow for a full clone of t.
  //     CHECK: %[[ALLOC:.*]] = memref.alloc(%[[DIM]])

  // init_tensor itself does not alloc but forwards to the **second**
  // insert_slice. InitTensorOp replaces the init_tensor with an out-of-place
  // extract_slice.
  //     CHECK: %[[EXTRACT_SLICE_ALLOC:.*]] = memref.alloc(%[[sz]])
  //     CHECK: %[[T_SUBVIEW:.*]] =  memref.subview %[[FUNC_ARG]][42] [%[[sz]]] [1]
  %a = linalg.init_tensor[%sz] : tensor<?xf32>

  //     CHECK: linalg.fill({{.*}}, %[[EXTRACT_SLICE_ALLOC]]) : f32, memref<?xf32>
  %f = linalg.fill(%f0, %a) : f32, tensor<?xf32> -> tensor<?xf32>

  //     CHECK: memref.copy %[[FUNC_ARG]], %[[ALLOC]] : memref<?xf32> to memref<?xf32>
  //     CHECK: %[[SV0_ALLOC:.*]] = memref.subview %[[ALLOC]][0] [%[[sz]]] [1] : memref<?xf32> to memref<?xf32>
  //     CHECK: memref.copy %[[EXTRACT_SLICE_ALLOC]], %[[SV0_ALLOC]] : memref<?xf32> to memref<?xf32>
  %r0 = tensor.insert_slice %f into %t[0][%sz][1]: tensor<?xf32> into tensor<?xf32>

  //     CHECK: memref.copy %[[EXTRACT_SLICE_ALLOC]], %[[T_SUBVIEW]]
  %r1 = tensor.insert_slice %f into %t[42][%sz][1]: tensor<?xf32> into tensor<?xf32>

  return %r0, %r1: tensor<?xf32>, tensor<?xf32>
}

// -----

//      CHECK: func @buffer_forwarding_no_conflict(
// CHECK-SAME:   %[[FUNC_ARG:[0-9a-zA-Z]*]]: memref<?xf32>
// CHECK-SAME:   %[[sz:[0-9a-zA-Z]*]]: index
func @buffer_forwarding_no_conflict(
  %t: tensor<?xf32> {linalg.buffer_layout = affine_map<(d0) -> (d0)>, linalg.inplaceable = true},
  %sz: index)
    -> (tensor<?xf32>)
{
  %f0 = arith.constant 0.0: f32

  // init_tensor itself does not alloc but forwards to the insert_slice.
  // InitTensorOp replaces the init_tensor with an inplace extract_slice.
  // CHECK: %[[T_SUBVIEW:.*]] =  memref.subview %[[FUNC_ARG]][42] [%[[sz]]] [1]
  %a = linalg.init_tensor[%sz] : tensor<?xf32>

  // CHECK: linalg.fill({{.*}}, %[[T_SUBVIEW]]) : f32, memref<?xf32
  %f = linalg.fill(%f0, %a) : f32, tensor<?xf32> -> tensor<?xf32>

  // Self-copy canonicalizes away later.
  %r1 = tensor.insert_slice %f into %t[42][%sz][1]: tensor<?xf32> into tensor<?xf32>

  return %r1: tensor<?xf32>
}

// -----

//      CHECK: func @insertion_point_inside_loop(
// CHECK-SAME:     %[[t:.*]]: memref<?xf32, #{{.*}}>, %[[sz:.*]]: index)
func @insertion_point_inside_loop(%t : tensor<?xf32>, %sz : index) -> (tensor<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index

  // CHECK-NOT: memref.alloc
  %blank = linalg.init_tensor [5] : tensor<5xf32>

  // CHECK: scf.for %[[iv:.*]] = %{{.*}} to %[[sz]] step %{{.*}} {
  %r = scf.for %iv = %c0 to %sz step %c5 iter_args(%bb = %t) -> (tensor<?xf32>) {
    // CHECK: %[[subview:.*]] = memref.subview %[[t]][%[[iv]]] [5] [1]
    %iv_i32 = arith.index_cast %iv : index to i32
    %f = arith.sitofp %iv_i32 : i32 to f32

    // CHECK: linalg.fill(%{{.*}}, %[[subview]])
    %filled = linalg.fill(%f, %blank) : f32, tensor<5xf32> -> tensor<5xf32>

    // CHECK-NOT: memref.copy
    %inserted = tensor.insert_slice %filled into %bb[%iv][5][1] : tensor<5xf32> into tensor<?xf32>
    scf.yield %inserted : tensor<?xf32>
  }

  return %r : tensor<?xf32>
}

// -----

//      CHECK: func @insertion_point_outside_loop(
// CHECK-SAME:     %[[t:.*]]: memref<?xf32, #{{.*}}>, %[[sz:.*]]: index, %[[idx:.*]]: index)
func @insertion_point_outside_loop(%t : tensor<?xf32>, %sz : index,
                                   %idx : index) -> (tensor<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c5 = arith.constant 5 : index

  // CHECK-NOT: memref.alloc
  // CHECK: %[[subview:.*]] = memref.subview %[[t]][%[[idx]]] [5] [1]
  %blank = linalg.init_tensor [5] : tensor<5xf32>

  // CHECK: scf.for %[[iv:.*]] = %{{.*}} to %[[sz]] step %{{.*}} {
  %r = scf.for %iv = %c0 to %sz step %c5 iter_args(%bb = %t) -> (tensor<?xf32>) {
    %iv_i32 = arith.index_cast %iv : index to i32
    %f = arith.sitofp %iv_i32 : i32 to f32

    // CHECK: linalg.fill(%{{.*}}, %[[subview]])
    %filled = linalg.fill(%f, %blank) : f32, tensor<5xf32> -> tensor<5xf32>

    // CHECK-NOT: memref.copy
    %inserted = tensor.insert_slice %filled into %bb[%idx][5][1] : tensor<5xf32> into tensor<?xf32>
    scf.yield %inserted : tensor<?xf32>
  }

  return %r : tensor<?xf32>
}
