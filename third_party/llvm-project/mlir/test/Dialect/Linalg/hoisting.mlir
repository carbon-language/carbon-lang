// RUN: mlir-opt %s -test-linalg-hoisting=test-hoist-redundant-transfers -allow-unregistered-dialect -split-input-file | FileCheck %s

// CHECK-LABEL: func @hoist_vector_transfer_pairs(
//  CHECK-SAME:   %[[MEMREF0:[a-zA-Z0-9]*]]: memref<?x?xf32>,
//  CHECK-SAME:   %[[MEMREF1:[a-zA-Z0-9]*]]: memref<?x?xf32>,
//  CHECK-SAME:   %[[MEMREF2:[a-zA-Z0-9]*]]: memref<?x?xf32>,
//  CHECK-SAME:   %[[MEMREF3:[a-zA-Z0-9]*]]: memref<?x?xf32>,
//  CHECK-SAME:   %[[MEMREF4:[a-zA-Z0-9]*]]: memref<?x?xf32>,
//  CHECK-SAME:   %[[MEMREF5:[a-zA-Z0-9]*]]: memref<?x?xf32>,
//  CHECK-SAME:   %[[VAL:[a-zA-Z0-9]*]]: index,
//  CHECK-SAME:   %[[LB:[a-zA-Z0-9]*]]: index,
//  CHECK-SAME:   %[[UB:[a-zA-Z0-9]*]]: index,
//  CHECK-SAME:   %[[STEP:[a-zA-Z0-9]*]]: index,
//  CHECK-SAME:   %[[CMP:[a-zA-Z0-9]*]]: i1
func @hoist_vector_transfer_pairs(
    %memref0: memref<?x?xf32>, %memref1: memref<?x?xf32>, %memref2: memref<?x?xf32>,
    %memref3: memref<?x?xf32>, %memref4: memref<?x?xf32>, %memref5: memref<?x?xf32>,
    %val: index, %lb : index, %ub : index, %step: index, %cmp: i1) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32

// CHECK: vector.transfer_read %{{.*}} : memref<?x?xf32>, vector<1xf32>
// CHECK: scf.for %[[I:.*]] = %[[LB]] to %[[UB]] step %[[STEP]] iter_args({{.*}}) -> (vector<1xf32>) {
// CHECK:   vector.transfer_read %{{.*}} : memref<?x?xf32>, vector<2xf32>
// CHECK:   scf.for %[[J:.*]] = %[[LB]] to %[[UB]] step %[[STEP]] iter_args({{.*}}) -> (vector<1xf32>, vector<2xf32>) {
// CHECK:     vector.transfer_read %{{.*}} : memref<?x?xf32>, vector<3xf32>
// CHECK:     vector.transfer_read %{{.*}} : memref<?x?xf32>, vector<4xf32>
// CHECK:     "some_crippling_use"(%[[MEMREF4]]) : (memref<?x?xf32>) -> ()
// CHECK:     vector.transfer_read %{{.*}} : memref<?x?xf32>, vector<5xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<1xf32>) -> vector<1xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<2xf32>) -> vector<2xf32>
// CHECK:     "some_use"(%[[MEMREF2]]) : (memref<?x?xf32>) -> vector<3xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<4xf32>) -> vector<4xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<5xf32>) -> vector<5xf32>
// CHECK:     vector.transfer_write %{{.*}} : vector<3xf32>, memref<?x?xf32>
// CHECK:     vector.transfer_write %{{.*}} : vector<4xf32>, memref<?x?xf32>
// CHECK:     vector.transfer_write %{{.*}} : vector<5xf32>, memref<?x?xf32>
// CHECK:     "some_crippling_use"(%[[MEMREF3]]) : (memref<?x?xf32>) -> ()
// CHECK:     scf.yield {{.*}} : vector<1xf32>, vector<2xf32>
// CHECK:   }
// CHECK:   vector.transfer_write %{{.*}} : vector<2xf32>, memref<?x?xf32>
// CHECK:   "unrelated_use"(%[[MEMREF0]]) : (memref<?x?xf32>) -> ()
// CHECK:   scf.yield {{.*}} : vector<1xf32>
// CHECK: }
// CHECK: vector.transfer_write %{{.*}} : vector<1xf32>, memref<?x?xf32>
// CHECK: "unrelated_use"(%[[MEMREF1]]) : (memref<?x?xf32>) -> ()
  scf.for %i = %lb to %ub step %step {
    scf.for %j = %lb to %ub step %step {
      %r0 = vector.transfer_read %memref1[%c0, %c0], %cst: memref<?x?xf32>, vector<1xf32>
      %r1 = vector.transfer_read %memref0[%i, %i], %cst: memref<?x?xf32>, vector<2xf32>
      %r2 = vector.transfer_read %memref2[%c0, %c0], %cst: memref<?x?xf32>, vector<3xf32>
      %r3 = vector.transfer_read %memref3[%c0, %c0], %cst: memref<?x?xf32>, vector<4xf32>
      "some_crippling_use"(%memref4) : (memref<?x?xf32>) -> ()
      %r4 = vector.transfer_read %memref4[%c0, %c0], %cst: memref<?x?xf32>, vector<5xf32>
      %r5 = vector.transfer_read %memref5[%c0, %c0], %cst: memref<?x?xf32>, vector<6xf32>
      "some_crippling_use"(%memref5) : (memref<?x?xf32>) -> ()
      %u0 = "some_use"(%r0) : (vector<1xf32>) -> vector<1xf32>
      %u1 = "some_use"(%r1) : (vector<2xf32>) -> vector<2xf32>
      %u2 = "some_use"(%memref2) : (memref<?x?xf32>) -> vector<3xf32>
      %u3 = "some_use"(%r3) : (vector<4xf32>) -> vector<4xf32>
      %u4 = "some_use"(%r4) : (vector<5xf32>) -> vector<5xf32>
      %u5 = "some_use"(%r5) : (vector<6xf32>) -> vector<6xf32>
      vector.transfer_write %u0, %memref1[%c0, %c0] : vector<1xf32>, memref<?x?xf32>
      vector.transfer_write %u1, %memref0[%i, %i] : vector<2xf32>, memref<?x?xf32>
      vector.transfer_write %u2, %memref2[%c0, %c0] : vector<3xf32>, memref<?x?xf32>
      vector.transfer_write %u3, %memref3[%c0, %c0] : vector<4xf32>, memref<?x?xf32>
      vector.transfer_write %u4, %memref4[%c0, %c0] : vector<5xf32>, memref<?x?xf32>
      vector.transfer_write %u5, %memref5[%c0, %c0] : vector<6xf32>, memref<?x?xf32>
      "some_crippling_use"(%memref3) : (memref<?x?xf32>) -> ()
    }
    "unrelated_use"(%memref0) : (memref<?x?xf32>) -> ()
  }
  "unrelated_use"(%memref1) : (memref<?x?xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @hoist_vector_transfer_pairs_disjoint(
//  CHECK-SAME:   %[[MEMREF0:[a-zA-Z0-9]*]]: memref<?x?xf32>,
//  CHECK-SAME:   %[[MEMREF1:[a-zA-Z0-9]*]]: memref<?x?xf32>,
//  CHECK-SAME:   %[[MEMREF2:[a-zA-Z0-9]*]]: memref<?x?xf32>,
//  CHECK-SAME:   %[[MEMREF3:[a-zA-Z0-9]*]]: memref<?x?xf32>,
//  CHECK-SAME:   %[[VAL:[a-zA-Z0-9]*]]: index,
//  CHECK-SAME:   %[[LB:[a-zA-Z0-9]*]]: index,
//  CHECK-SAME:   %[[UB:[a-zA-Z0-9]*]]: index,
//  CHECK-SAME:   %[[STEP:[a-zA-Z0-9]*]]: index,
//  CHECK-SAME:   %[[RANDOM:[a-zA-Z0-9]*]]: index,
//  CHECK-SAME:   %[[CMP:[a-zA-Z0-9]*]]: i1
func @hoist_vector_transfer_pairs_disjoint(
    %memref0: memref<?x?xf32>, %memref1: memref<?x?xf32>,
    %memref2: memref<?x?xf32>, %memref3: memref<?x?xf32>, %val: index, %lb : index, %ub : index,
    %step: index, %random_index : index, %cmp: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %cst = arith.constant 0.0 : f32

// CHECK: vector.transfer_read %[[MEMREF2]]{{.*}} : memref<?x?xf32>, vector<3xf32>
// CHECK: vector.transfer_read %[[MEMREF2]]{{.*}} : memref<?x?xf32>, vector<3xf32>
// CHECK: vector.transfer_read %[[MEMREF3]]{{.*}} : memref<?x?xf32>, vector<4xf32>
// CHECK: vector.transfer_read %[[MEMREF3]]{{.*}} : memref<?x?xf32>, vector<4xf32>
// CHECK: scf.for %[[I:.*]] = %[[LB]] to %[[UB]] step %[[STEP]] iter_args({{.*}}) ->
//  CHECK-SAME: (vector<3xf32>, vector<3xf32>, vector<4xf32>, vector<4xf32>) {
// CHECK:   scf.for %[[J:.*]] = %[[LB]] to %[[UB]] step %[[STEP]] iter_args({{.*}}) ->
//  CHECK-SAME: (vector<3xf32>, vector<3xf32>, vector<4xf32>, vector<4xf32>) {
// CHECK:     vector.transfer_read %[[MEMREF1]]{{.*}} : memref<?x?xf32>, vector<2xf32>
// CHECK:     vector.transfer_read %[[MEMREF1]]{{.*}} : memref<?x?xf32>, vector<2xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<2xf32>) -> vector<2xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<2xf32>) -> vector<2xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<3xf32>) -> vector<3xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<3xf32>) -> vector<3xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<4xf32>) -> vector<4xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<4xf32>) -> vector<4xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<2xf32>) -> vector<2xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<2xf32>) -> vector<2xf32>
// CHECK:     vector.transfer_write %{{.*}}, %[[MEMREF1]]{{.*}} : vector<2xf32>, memref<?x?xf32>
// CHECK:     vector.transfer_write %{{.*}}, %[[MEMREF1]]{{.*}} : vector<2xf32>, memref<?x?xf32>
// CHECK:     scf.yield {{.*}} : vector<3xf32>, vector<3xf32>, vector<4xf32>, vector<4xf32>
// CHECK:   }
// CHECK:   scf.yield {{.*}} : vector<3xf32>, vector<3xf32>, vector<4xf32>, vector<4xf32>
// CHECK: }
// CHECK: vector.transfer_write %{{.*}}, %[[MEMREF3]]{{.*}} : vector<4xf32>, memref<?x?xf32>
// CHECK: vector.transfer_write %{{.*}}, %[[MEMREF3]]{{.*}} : vector<4xf32>, memref<?x?xf32>
// CHECK: vector.transfer_write %{{.*}}, %[[MEMREF2]]{{.*}} : vector<3xf32>, memref<?x?xf32>
// CHECK: vector.transfer_write %{{.*}}, %[[MEMREF2]]{{.*}} : vector<3xf32>, memref<?x?xf32>
  scf.for %i = %lb to %ub step %step {
    scf.for %j = %lb to %ub step %step {
      %r00 = vector.transfer_read %memref1[%c0, %c0], %cst: memref<?x?xf32>, vector<2xf32>
      %r01 = vector.transfer_read %memref1[%c0, %c1], %cst: memref<?x?xf32>, vector<2xf32>
      %r20 = vector.transfer_read %memref2[%c0, %c0], %cst: memref<?x?xf32>, vector<3xf32>
      %r21 = vector.transfer_read %memref2[%c0, %c3], %cst: memref<?x?xf32>, vector<3xf32>
      %r30 = vector.transfer_read %memref3[%c0, %random_index], %cst: memref<?x?xf32>, vector<4xf32>
      %r31 = vector.transfer_read %memref3[%c1, %random_index], %cst: memref<?x?xf32>, vector<4xf32>
      %r10 = vector.transfer_read %memref0[%i, %i], %cst: memref<?x?xf32>, vector<2xf32>
      %r11 = vector.transfer_read %memref0[%random_index, %random_index], %cst: memref<?x?xf32>, vector<2xf32>
      %u00 = "some_use"(%r00) : (vector<2xf32>) -> vector<2xf32>
      %u01 = "some_use"(%r01) : (vector<2xf32>) -> vector<2xf32>
      %u20 = "some_use"(%r20) : (vector<3xf32>) -> vector<3xf32>
      %u21 = "some_use"(%r21) : (vector<3xf32>) -> vector<3xf32>
      %u30 = "some_use"(%r30) : (vector<4xf32>) -> vector<4xf32>
      %u31 = "some_use"(%r31) : (vector<4xf32>) -> vector<4xf32>
      %u10 = "some_use"(%r10) : (vector<2xf32>) -> vector<2xf32>
      %u11 = "some_use"(%r11) : (vector<2xf32>) -> vector<2xf32>
      vector.transfer_write %u00, %memref1[%c0, %c0] : vector<2xf32>, memref<?x?xf32>
      vector.transfer_write %u01, %memref1[%c0, %c1] : vector<2xf32>, memref<?x?xf32>
      vector.transfer_write %u20, %memref2[%c0, %c0] : vector<3xf32>, memref<?x?xf32>
      vector.transfer_write %u21, %memref2[%c0, %c3] : vector<3xf32>, memref<?x?xf32>
      vector.transfer_write %u30, %memref3[%c0, %random_index] : vector<4xf32>, memref<?x?xf32>
      vector.transfer_write %u31, %memref3[%c1, %random_index] : vector<4xf32>, memref<?x?xf32>
      vector.transfer_write %u10, %memref0[%i, %i] : vector<2xf32>, memref<?x?xf32>
      vector.transfer_write %u11, %memref0[%random_index, %random_index] : vector<2xf32>, memref<?x?xf32>
    }
  }
  return
}

// -----

// CHECK-LABEL: func @hoist_vector_transfer_pairs_tensor
func @hoist_vector_transfer_pairs_tensor(
    %tensor0: tensor<?x?xf32>, %tensor1: tensor<?x?xf32>, %tensor2: tensor<?x?xf32>,
    %tensor3: tensor<?x?xf32>, %tensor4: tensor<?x?xf32>, %tensor5: tensor<?x?xf32>,
    %val: index, %lb : index, %ub : index, %step: index) ->
    (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>,
     tensor<?x?xf32>, tensor<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32

// CHECK: vector.transfer_read %{{.*}} : tensor<?x?xf32>, vector<1xf32>
// CHECK: scf.for {{.*}} iter_args({{.*}}) ->
// CHECK-SAME: (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, vector<1xf32>) {
// CHECK:   vector.transfer_read %{{.*}} : tensor<?x?xf32>, vector<2xf32>
// CHECK:   scf.for {{.*}} iter_args({{.*}}) ->
// CHECK-SAME: (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, vector<2xf32>, vector<1xf32>) {
// CHECK:     vector.transfer_read %{{.*}} : tensor<?x?xf32>, vector<4xf32>
// CHECK:     "some_crippling_use"(%{{.*}}) : (tensor<?x?xf32>) -> ()
// CHECK:     vector.transfer_read %{{.*}} : tensor<?x?xf32>, vector<5xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<1xf32>) -> vector<1xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<2xf32>) -> vector<2xf32>
// CHECK:     "some_use"(%{{.*}}) : (tensor<?x?xf32>) -> vector<3xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<4xf32>) -> vector<4xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<5xf32>) -> vector<5xf32>
// CHECK:     vector.transfer_write %{{.*}} : vector<3xf32>, tensor<?x?xf32>
// CHECK:     vector.transfer_write %{{.*}} : vector<4xf32>, tensor<?x?xf32>
// CHECK:     vector.transfer_write %{{.*}} : vector<5xf32>, tensor<?x?xf32>
// CHECK:     "some_crippling_use"(%{{.*}}) : (tensor<?x?xf32>) -> ()
// CHECK:     scf.yield {{.*}} :
// CHECK-SAME: tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, vector<2xf32>, vector<1xf32>
// CHECK:   }
// CHECK:   vector.transfer_write %{{.*}} : vector<2xf32>, tensor<?x?xf32>
// CHECK:   scf.yield {{.*}} :
// CHECK-SAME: tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, vector<1xf32>
// CHECK: }
// CHECK: vector.transfer_write %{{.*}} : vector<1xf32>, tensor<?x?xf32>
  %0:6 = scf.for %i = %lb to %ub step %step
  iter_args(%arg0 = %tensor0, %arg1 = %tensor1, %arg2 = %tensor2,
            %arg3 = %tensor3,  %arg4 = %tensor4, %arg5 = %tensor5)
  -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>,
     tensor<?x?xf32>, tensor<?x?xf32>)  {
    %1:6 = scf.for %j = %lb to %ub step %step
    iter_args(%arg6 = %arg0, %arg7 = %arg1, %arg8 = %arg2,
              %arg9 = %arg3,  %arg10 = %arg4, %arg11 = %arg5)
    -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>,
       tensor<?x?xf32>, tensor<?x?xf32>)  {
      %r0 = vector.transfer_read %arg7[%c0, %c0], %cst: tensor<?x?xf32>, vector<1xf32>
      %r1 = vector.transfer_read %arg6[%i, %i], %cst: tensor<?x?xf32>, vector<2xf32>
      %r3 = vector.transfer_read %arg9[%c0, %c0], %cst: tensor<?x?xf32>, vector<4xf32>
      "some_crippling_use"(%arg10) : (tensor<?x?xf32>) -> ()
      %r4 = vector.transfer_read %arg10[%c0, %c0], %cst: tensor<?x?xf32>, vector<5xf32>
      %r5 = vector.transfer_read %arg11[%c0, %c0], %cst: tensor<?x?xf32>, vector<6xf32>
      "some_crippling_use"(%arg11) : (tensor<?x?xf32>) -> ()
      %u0 = "some_use"(%r0) : (vector<1xf32>) -> vector<1xf32>
      %u1 = "some_use"(%r1) : (vector<2xf32>) -> vector<2xf32>
      %u2 = "some_use"(%arg8) : (tensor<?x?xf32>) -> vector<3xf32>
      %u3 = "some_use"(%r3) : (vector<4xf32>) -> vector<4xf32>
      %u4 = "some_use"(%r4) : (vector<5xf32>) -> vector<5xf32>
      %u5 = "some_use"(%r5) : (vector<6xf32>) -> vector<6xf32>
      %w1 = vector.transfer_write %u0, %arg7[%c0, %c0] : vector<1xf32>, tensor<?x?xf32>
      %w0 = vector.transfer_write %u1, %arg6[%i, %i] : vector<2xf32>, tensor<?x?xf32>
      %w2 = vector.transfer_write %u2, %arg8[%c0, %c0] : vector<3xf32>, tensor<?x?xf32>
      %w3 = vector.transfer_write %u3, %arg9[%c0, %c0] : vector<4xf32>, tensor<?x?xf32>
      %w4 = vector.transfer_write %u4, %arg10[%c0, %c0] : vector<5xf32>, tensor<?x?xf32>
      %w5 = vector.transfer_write %u5, %arg11[%c0, %c0] : vector<6xf32>, tensor<?x?xf32>
      "some_crippling_use"(%w3) : (tensor<?x?xf32>) -> ()
      scf.yield %w0, %w1, %w2, %w3, %w4, %w5 :
        tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>,
        tensor<?x?xf32>, tensor<?x?xf32>
      }
      scf.yield %1#0,  %1#1, %1#2, %1#3, %1#4, %1#5 :
        tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>,
        tensor<?x?xf32>, tensor<?x?xf32>
  }
  return %0#0,  %0#1, %0#2, %0#3, %0#4,  %0#5 :
        tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>,
        tensor<?x?xf32>, tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @hoist_vector_transfer_pairs_disjoint_tensor(
//  CHECK-SAME:   %[[TENSOR0:[a-zA-Z0-9]*]]: tensor<?x?xf32>,
//  CHECK-SAME:   %[[TENSOR1:[a-zA-Z0-9]*]]: tensor<?x?xf32>,
//  CHECK-SAME:   %[[TENSOR2:[a-zA-Z0-9]*]]: tensor<?x?xf32>,
//  CHECK-SAME:   %[[TENSOR3:[a-zA-Z0-9]*]]: tensor<?x?xf32>,
func @hoist_vector_transfer_pairs_disjoint_tensor(
    %tensor0: tensor<?x?xf32>, %tensor1: tensor<?x?xf32>,
    %tensor2: tensor<?x?xf32>, %tensor3: tensor<?x?xf32>,
    %val: index, %lb : index, %ub : index, %step: index,
    %random_index : index) ->
    (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %cst = arith.constant 0.0 : f32

// CHECK: vector.transfer_read %[[TENSOR2]]{{.*}} : tensor<?x?xf32>, vector<3xf32>
// CHECK: vector.transfer_read %[[TENSOR2]]{{.*}} : tensor<?x?xf32>, vector<3xf32>
// CHECK: vector.transfer_read %[[TENSOR3]]{{.*}} : tensor<?x?xf32>, vector<4xf32>
// CHECK: vector.transfer_read %[[TENSOR3]]{{.*}} : tensor<?x?xf32>, vector<4xf32>
// CHECK: %[[R:.*]]:6 = scf.for {{.*}} iter_args({{.*}}) ->
// CHECK-SAME: (tensor<?x?xf32>, tensor<?x?xf32>, vector<3xf32>, vector<3xf32>, vector<4xf32>, vector<4xf32>) {
// CHECK:   scf.for {{.*}} iter_args({{.*}}) ->
// CHECK-SAME: (tensor<?x?xf32>, tensor<?x?xf32>, vector<3xf32>, vector<3xf32>, vector<4xf32>, vector<4xf32>) {
// CHECK:     vector.transfer_read %[[TENSOR1]]{{.*}} : tensor<?x?xf32>, vector<2xf32>
// CHECK:     vector.transfer_read %[[TENSOR1]]{{.*}} : tensor<?x?xf32>, vector<2xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<2xf32>) -> vector<2xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<2xf32>) -> vector<2xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<3xf32>) -> vector<3xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<3xf32>) -> vector<3xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<4xf32>) -> vector<4xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<4xf32>) -> vector<4xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<2xf32>) -> vector<2xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<2xf32>) -> vector<2xf32>
// CHECK:     vector.transfer_write %{{.*}}, %{{.*}}{{.*}} : vector<2xf32>, tensor<?x?xf32>
// CHECK:     vector.transfer_write %{{.*}}, %{{.*}}{{.*}} : vector<2xf32>, tensor<?x?xf32>
// CHECK:     scf.yield {{.*}} :
// CHECK-SAME: tensor<?x?xf32>, tensor<?x?xf32>, vector<3xf32>, vector<3xf32>, vector<4xf32>, vector<4xf32>
// CHECK:   }
// CHECK:   scf.yield {{.*}} :
// CHECK-SAME: tensor<?x?xf32>, tensor<?x?xf32>, vector<3xf32>, vector<3xf32>, vector<4xf32>, vector<4xf32>
// CHECK: }
// CHECK: %[[TENSOR4:.*]] = vector.transfer_write %[[R]]#5, %[[TENSOR3]]{{.*}} : vector<4xf32>, tensor<?x?xf32>
// CHECK:                   vector.transfer_write %[[R]]#4, %[[TENSOR4]]{{.*}} : vector<4xf32>, tensor<?x?xf32>
// CHECK: %[[TENSOR5:.*]] = vector.transfer_write %[[R]]#3, %[[TENSOR2]]{{.*}} : vector<3xf32>, tensor<?x?xf32>
// CHECK:                   vector.transfer_write %[[R]]#2, %[[TENSOR5]]{{.*}} : vector<3xf32>, tensor<?x?xf32>
  %0:4 = scf.for %i = %lb to %ub step %step
  iter_args(%arg0 = %tensor0, %arg1 = %tensor1, %arg2 = %tensor2,
            %arg3 = %tensor3)
  -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) {
    %1:4 = scf.for %j = %lb to %ub step %step
    iter_args(%arg4 = %arg0, %arg5 = %arg1, %arg6 = %arg2,
              %arg7 = %arg3)
    -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) {
      %r00 = vector.transfer_read %arg5[%c0, %c0], %cst: tensor<?x?xf32>, vector<2xf32>
      %r01 = vector.transfer_read %arg5[%c0, %c1], %cst: tensor<?x?xf32>, vector<2xf32>
      %r20 = vector.transfer_read %arg6[%c0, %c0], %cst: tensor<?x?xf32>, vector<3xf32>
      %r21 = vector.transfer_read %arg6[%c0, %c3], %cst: tensor<?x?xf32>, vector<3xf32>
      %r30 = vector.transfer_read %arg7[%c0, %random_index], %cst: tensor<?x?xf32>, vector<4xf32>
      %r31 = vector.transfer_read %arg7[%c1, %random_index], %cst: tensor<?x?xf32>, vector<4xf32>
      %r10 = vector.transfer_read %arg4[%i, %i], %cst: tensor<?x?xf32>, vector<2xf32>
      %r11 = vector.transfer_read %arg4[%random_index, %random_index], %cst: tensor<?x?xf32>, vector<2xf32>
      %u00 = "some_use"(%r00) : (vector<2xf32>) -> vector<2xf32>
      %u01 = "some_use"(%r01) : (vector<2xf32>) -> vector<2xf32>
      %u20 = "some_use"(%r20) : (vector<3xf32>) -> vector<3xf32>
      %u21 = "some_use"(%r21) : (vector<3xf32>) -> vector<3xf32>
      %u30 = "some_use"(%r30) : (vector<4xf32>) -> vector<4xf32>
      %u31 = "some_use"(%r31) : (vector<4xf32>) -> vector<4xf32>
      %u10 = "some_use"(%r10) : (vector<2xf32>) -> vector<2xf32>
      %u11 = "some_use"(%r11) : (vector<2xf32>) -> vector<2xf32>
      %w10 = vector.transfer_write %u00, %arg5[%c0, %c0] : vector<2xf32>, tensor<?x?xf32>
      %w11 = vector.transfer_write %u01, %w10[%c0, %c1] : vector<2xf32>, tensor<?x?xf32>
      %w20 = vector.transfer_write %u20, %arg6[%c0, %c0] : vector<3xf32>, tensor<?x?xf32>
      %w21 = vector.transfer_write %u21, %w20[%c0, %c3] : vector<3xf32>, tensor<?x?xf32>
      %w30 = vector.transfer_write %u30, %arg7[%c0, %random_index] : vector<4xf32>, tensor<?x?xf32>
      %w31 = vector.transfer_write %u31, %w30[%c1, %random_index] : vector<4xf32>, tensor<?x?xf32>
      %w00 = vector.transfer_write %u10, %arg4[%i, %i] : vector<2xf32>, tensor<?x?xf32>
      %w01 = vector.transfer_write %u11, %w00[%random_index, %random_index] : vector<2xf32>, tensor<?x?xf32>
      scf.yield %w01, %w11, %w21, %w31 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
    }
    scf.yield %1#0,  %1#1, %1#2, %1#3 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
  }
  return %0#0,  %0#1, %0#2, %0#3 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @hoist_vector_transfer_pairs_tensor_and_slices
//  CHECK-SAME:   %[[TENSOR0:[a-zA-Z0-9]*]]: tensor<?x?xf32>,
//  CHECK-SAME:   %[[TENSOR1:[a-zA-Z0-9]*]]: tensor<?x?xf32>,
//  CHECK-SAME:   %[[TENSOR2:[a-zA-Z0-9]*]]: tensor<?x?xf32>,
//  CHECK-SAME:   %[[TENSOR3:[a-zA-Z0-9]*]]: tensor<?x?xf32>,
//  CHECK-SAME:   %[[TENSOR4:[a-zA-Z0-9]*]]: tensor<?x?xf32>,
//  CHECK-SAME:   %[[TENSOR5:[a-zA-Z0-9]*]]: tensor<?x?xf32>
func @hoist_vector_transfer_pairs_tensor_and_slices(
    %tensor0: tensor<?x?xf32>, %tensor1: tensor<?x?xf32>, %tensor2: tensor<?x?xf32>,
    %tensor3: tensor<?x?xf32>, %tensor4: tensor<?x?xf32>, %tensor5: tensor<?x?xf32>,
    %val: index, %lb : index, %ub : index, %step: index) ->
    (
      tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>//, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
    ) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32

  //      CHECK: scf.for %[[I:.*]] = {{.*}} iter_args(
  // CHECK-SAME:   %[[TENSOR0_ARG:[0-9a-zA-Z]+]] = %[[TENSOR0]],
  // CHECK-SAME:   %[[TENSOR1_ARG:[0-9a-zA-Z]+]] = %[[TENSOR1]],
  // CHECK-SAME:   %[[TENSOR2_ARG:[0-9a-zA-Z]+]] = %[[TENSOR2]]
  // CHECK-SAME: ) ->
  // CHECK-SAME: (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
  %0:3 = scf.for %i = %lb to %ub step %step
  iter_args(%arg0 = %tensor0, %arg1 = %tensor1, %arg2 = %tensor2)
    -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>)  {

    // Hoisted
    // CHECK:   %[[ST0:.*]] = tensor.extract_slice %[[TENSOR0_ARG]][%[[I]], %[[I]]]{{.*}}: tensor<?x?xf32> to tensor<?x?xf32>
    // CHECK:   %[[V0:.*]] = vector.transfer_read %[[ST0]]{{.*}} : tensor<?x?xf32>, vector<1xf32>

    //      CHECK:   %[[R:.*]]:3 = scf.for %[[J:.*]] = {{.*}} iter_args(
    // CHECK-SAME:   %[[TENSOR1_ARG_L2:[0-9a-zA-Z]+]] = %[[TENSOR1_ARG]]
    // CHECK-SAME:   %[[TENSOR2_ARG_L2:[0-9a-zA-Z]+]] = %[[TENSOR2_ARG]]
    // CHECK-SAME:   %[[V0_ARG_L2:[0-9a-zA-Z]+]] = %[[V0]]
    // CHECK-SAME: ) ->
    // CHECK-SAME: (tensor<?x?xf32>, tensor<?x?xf32>, vector<1xf32>
    %1:3 = scf.for %j = %lb to %ub step %step
    iter_args(%arg6 = %arg0, %arg7 = %arg1, %arg8 = %arg2)
    -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>)  {
      // Hoists.
      %st0 = tensor.extract_slice %arg6[%i, %i][%step, %step][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
      %r0 = vector.transfer_read %st0[%c0, %c0], %cst: tensor<?x?xf32>, vector<1xf32>

      // CHECK:     %[[ST1:.*]] = tensor.extract_slice %[[TENSOR1_ARG_L2]][%[[J]],{{.*}}: tensor<?x?xf32> to tensor<?x?xf32>
      // CHECK:     %[[V1:.*]] = vector.transfer_read %[[ST1]]{{.*}} : tensor<?x?xf32>, vector<2xf32>
      // Does not hoist (slice depends on %j)
      %st1 = tensor.extract_slice %arg7[%j, %c0][%step, %step][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
      %r1 = vector.transfer_read %st1[%c0, %c0], %cst: tensor<?x?xf32>, vector<2xf32>

      // CHECK:     %[[ST2:.*]] = tensor.extract_slice %[[TENSOR2_ARG_L2]][%[[I]],{{.*}}: tensor<?x?xf32> to tensor<?x?xf32>
      // CHECK:     %[[V2:.*]] = vector.transfer_read %[[ST2]]{{.*}} : tensor<?x?xf32>, vector<3xf32>
      // Does not hoist, 2 slice %arg8.
      %st2 = tensor.extract_slice %arg8[%i, %c0][%step, %step][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
      %r2 = vector.transfer_read %st2[%c0, %c0], %cst: tensor<?x?xf32>, vector<3xf32>

      // CHECK:     %[[U0:.*]] = "some_use"(%[[V0_ARG_L2]]) : (vector<1xf32>) -> vector<1xf32>
      // CHECK:     %[[U1:.*]] = "some_use"(%[[V1]]) : (vector<2xf32>) -> vector<2xf32>
      // CHECK:     %[[U2:.*]] = "some_use"(%[[V2]]) : (vector<3xf32>) -> vector<3xf32>
      %u0 = "some_use"(%r0) : (vector<1xf32>) -> vector<1xf32>
      %u1 = "some_use"(%r1) : (vector<2xf32>) -> vector<2xf32>
      %u2 = "some_use"(%r2) : (vector<3xf32>) -> vector<3xf32>

      // Hoists
      %w0 = vector.transfer_write %u0, %st0[%c0, %c0] : vector<1xf32>, tensor<?x?xf32>

      // CHECK-DAG:     %[[STI1:.*]] = vector.transfer_write %[[U1]], %{{.*}} : vector<2xf32>, tensor<?x?xf32>
      // Does not hoist (associated slice depends on %j).
      %w1 = vector.transfer_write %u1, %st1[%i, %i] : vector<2xf32>, tensor<?x?xf32>

      // CHECK-DAG:     %[[STI2:.*]] = vector.transfer_write %[[U2]], %{{.*}} : vector<3xf32>, tensor<?x?xf32>
      // Does not hoist, 2 slice / insert_slice for %arg8.
      %w2 = vector.transfer_write %u2, %st2[%c0, %c0] : vector<3xf32>, tensor<?x?xf32>

      // Hoists.
      %sti0 = tensor.insert_slice %w0 into %arg6[%i, %i][%step, %step][1, 1] : tensor<?x?xf32> into tensor<?x?xf32>

      // CHECK-DAG:     tensor.insert_slice %[[STI1]] into %[[TENSOR1_ARG_L2]][%[[J]],{{.*}}: tensor<?x?xf32> into tensor<?x?xf32>
      // Does not hoist (depends on %j).
      %sti1 = tensor.insert_slice %w1 into %arg7[%j, %c0][%step, %step][1, 1] : tensor<?x?xf32> into tensor<?x?xf32>

      // CHECK-DAG:     tensor.insert_slice %[[STI2]] into %[[TENSOR2_ARG_L2]][%[[I]],{{.*}}: tensor<?x?xf32> into tensor<?x?xf32>
      // Does not hoist, 2 slice / insert_slice for %arg8.
      %sti2 = tensor.insert_slice %w2 into %arg8[%i, %c0][%step, %step][1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
      // Extract with a different stride to make sure we cannot fold this extract with the above insert.
      %st22 = tensor.extract_slice %sti2[%i, %c0][%step, %step][2, 1] : tensor<?x?xf32> to tensor<?x?xf32>
      %sti22 = tensor.insert_slice %st22 into %arg8[%i, %c0][%step, %step][1, 1] : tensor<?x?xf32> into tensor<?x?xf32>

      // CHECK:     scf.yield {{.*}} : tensor<?x?xf32>, tensor<?x?xf32>, vector<1xf32>
      // CHECK:   }
      scf.yield %sti0, %sti1, %sti22:
        tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
    }

    // Hoisted
    // CHECK:   %[[STI0:.*]] = vector.transfer_write %[[R]]#2, %[[ST0]]{{.*}} : vector<1xf32>, tensor<?x?xf32>
    // CHECK:   tensor.insert_slice %[[STI0]] into %[[TENSOR0_ARG]][%[[I]], %[[I]]]{{.*}} : tensor<?x?xf32> into tensor<?x?xf32>

    // CHECK:   scf.yield {{.*}} : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
    scf.yield %1#0, %1#1, %1#2 :
      tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>

    // CHECK: }
  }
  return %0#0, %0#1, %0#2 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
}
