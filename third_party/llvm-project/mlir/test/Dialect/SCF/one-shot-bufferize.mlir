// RUN: mlir-opt %s -allow-unregistered-dialect -one-shot-bufferize="allow-return-allocs bufferize-function-boundaries" -drop-equivalent-buffer-results -buffer-deallocation -split-input-file | FileCheck %s

// Run fuzzer with different seeds.
// RUN: mlir-opt %s -allow-unregistered-dialect -one-shot-bufferize="allow-return-allocs test-analysis-only analysis-fuzzer-seed=23 bufferize-function-boundaries" -split-input-file -o /dev/null
// RUN: mlir-opt %s -allow-unregistered-dialect -one-shot-bufferize="allow-return-allocs test-analysis-only analysis-fuzzer-seed=59 bufferize-function-boundaries" -split-input-file -o /dev/null
// RUN: mlir-opt %s -allow-unregistered-dialect -one-shot-bufferize="allow-return-allocs test-analysis-only analysis-fuzzer-seed=91 bufferize-function-boundaries" -split-input-file -o /dev/null

// Test bufferization using memref types that have no layout map.
// RUN: mlir-opt %s -allow-unregistered-dialect -one-shot-bufferize="allow-return-allocs unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map bufferize-function-boundaries" -buffer-deallocation -split-input-file -o /dev/null

// CHECK-DAG: #[[$map_1d_dyn:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>

// CHECK-LABEL: func @scf_for_yield_only(
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]*]]: memref<?xf32, #[[$map_1d_dyn]]>,
//  CHECK-SAME:   %[[t:[a-zA-Z0-9]*]]: memref<?xf32, #[[$map_1d_dyn]]>
//  CHECK-SAME:   ) -> memref<?xf32> {
func.func @scf_for_yield_only(
    %A : tensor<?xf32> {bufferization.writable = false},
    %B : tensor<?xf32> {bufferization.writable = true},
    %lb : index, %ub : index, %step : index)
  -> (tensor<?xf32>, tensor<?xf32>)
{
  //     CHECK:   %[[ALLOC_FOR_A:.*]] = memref.alloc
  //     CHECK:   memref.copy %[[A]], %[[ALLOC_FOR_A]]

  // The first scf.for remains but just turns into dead code.
  %r0 = scf.for %i = %lb to %ub step %step iter_args(%t = %A) -> (tensor<?xf32>) {
    scf.yield %t : tensor<?xf32>
  }

  // The second scf.for remains but just turns into dead code.
  %r1 = scf.for %i = %lb to %ub step %step iter_args(%t = %B) -> (tensor<?xf32>) {
    scf.yield %t : tensor<?xf32>
  }

  //     CHECK:   return %[[ALLOC_FOR_A]] : memref<?xf32>
  // CHECK-NOT:   dealloc
  return %r0, %r1: tensor<?xf32>, tensor<?xf32>
}

// -----

// Ensure that the function bufferizes without error. This tests pre-order
// traversal of scf.for loops during bufferization. No need to check the IR,
// just want to make sure that it does not crash.

// CHECK-LABEL: func @nested_scf_for
func.func @nested_scf_for(%A : tensor<?xf32> {bufferization.writable = true},
                          %v : vector<5xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %r1 = scf.for %i = %c0 to %c10 step %c1 iter_args(%B = %A) -> tensor<?xf32> {
    %r2 = scf.for %j = %c0 to %c10 step %c1 iter_args(%C = %B) -> tensor<?xf32> {
      %w = vector.transfer_write %v, %C[%c0] : vector<5xf32>, tensor<?xf32>
      scf.yield %w : tensor<?xf32>
    }
    scf.yield %r2 : tensor<?xf32>
  }
  return %r1 : tensor<?xf32>
}

// -----

// CHECK-DAG: #[[$map_1d_dyn:.*]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>

// CHECK-LABEL: func @scf_for_with_tensor.insert_slice
//  CHECK-SAME:   %[[A:[a-zA-Z0-9]*]]: memref<?xf32, #[[$map_1d_dyn]]>
//  CHECK-SAME:   %[[B:[a-zA-Z0-9]*]]: memref<?xf32, #[[$map_1d_dyn]]>
//  CHECK-SAME:   %[[C:[a-zA-Z0-9]*]]: memref<4xf32, #[[$map_1d_dyn]]>
func.func @scf_for_with_tensor.insert_slice(
    %A : tensor<?xf32> {bufferization.writable = false},
    %B : tensor<?xf32> {bufferization.writable = true},
    %C : tensor<4xf32> {bufferization.writable = false},
    %lb : index, %ub : index, %step : index)
  -> (tensor<?xf32>, tensor<?xf32>)
{
  //     CHECK:   %[[ALLOC_FOR_A:.*]] = memref.alloc
  //     CHECK:   memref.copy %[[A]], %[[ALLOC_FOR_A]]

  //     CHECK: %[[svA:.*]] = memref.subview %[[ALLOC_FOR_A]][0] [4] [1]
  //     CHECK: %[[svB:.*]] = memref.subview %[[B]][0] [4] [1]

  //     CHECK:   scf.for {{.*}}
  // CHECK-NOT: iter_args
  %r0:2 = scf.for %i = %lb to %ub step %step iter_args(%tA = %A, %tB = %B)
      -> (tensor<?xf32>, tensor<?xf32>)
  {
    // %ttA bufferizes to direct copy of %BUFFER_CAST_C into %svA
    //     CHECK: memref.copy %[[C]], %[[svA]]
    %ttA = tensor.insert_slice %C into %tA[0][4][1] : tensor<4xf32> into tensor<?xf32>

    // %ttB bufferizes to direct copy of %BUFFER_CAST_C into %BUFFER_CAST_B
    //     CHECK:   memref.copy %[[C]], %[[svB]]
    %ttB = tensor.insert_slice %C into %tB[0][4][1] : tensor<4xf32> into tensor<?xf32>

    // CHECK-NOT:   scf.yield
    scf.yield %ttA, %ttB : tensor<?xf32>, tensor<?xf32>
  }

  //     CHECK:  return %[[ALLOC_FOR_A]] : memref<?xf32>
  return %r0#0, %r0#1: tensor<?xf32>, tensor<?xf32>
}

// -----

// CHECK-LABEL: func @execute_region_with_conflict(
//  CHECK-SAME:     %[[m1:.*]]: memref<?xf32
func.func @execute_region_with_conflict(
    %t1 : tensor<?xf32> {bufferization.writable = "true"})
  -> (f32, tensor<?xf32>, f32)
{
  %f1 = arith.constant 0.0 : f32
  %idx = arith.constant 7 : index

  // scf.execute_region is canonicalized away after bufferization. So just the
  // memref.store is left over.

  // CHECK: %[[alloc:.*]] = memref.alloc
  // CHECK: memref.copy %[[m1]], %[[alloc]]
  // CHECK: memref.store %{{.*}}, %[[alloc]][%{{.*}}]
  %0, %1, %2 = scf.execute_region -> (f32, tensor<?xf32>, f32) {
    %t2 = tensor.insert %f1 into %t1[%idx] : tensor<?xf32>
    scf.yield %f1, %t2, %f1 : f32, tensor<?xf32>, f32
  }

  // CHECK: %[[casted:.*]] = memref.cast %[[alloc]]
  // CHECK: %[[load:.*]] = memref.load %[[m1]]
  %3 = tensor.extract %t1[%idx] : tensor<?xf32>

  // CHECK: return %{{.*}}, %[[casted]], %[[load]] : f32, memref<?xf32, #{{.*}}>, f32
  return %0, %1, %3 : f32, tensor<?xf32>, f32
}

// -----

// CHECK-LABEL: func @scf_if_inplace(
//  CHECK-SAME:     %[[cond:.*]]: i1, %[[t1:.*]]: memref<?xf32{{.*}}>, %[[v:.*]]: vector
func.func @scf_if_inplace(%cond: i1,
                          %t1: tensor<?xf32> {bufferization.writable = true},
                          %v: vector<5xf32>, %idx: index) -> tensor<?xf32> {

  //      CHECK: scf.if %[[cond]] {
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   vector.transfer_write %[[v]], %[[t1]]
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  %r = scf.if %cond -> (tensor<?xf32>) {
    scf.yield %t1 : tensor<?xf32>
  } else {
    %t2 = vector.transfer_write %v, %t1[%idx] : vector<5xf32>, tensor<?xf32>
    scf.yield %t2 : tensor<?xf32>
  }
  return %r : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @scf_if_inside_scf_for
//   CHECK-DAG:   %[[c0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[c1:.*]] = arith.constant 1 : index
//   CHECK-DAG:   %[[c10:.*]] = arith.constant 10 : index
//       CHECK:   scf.for %{{.*}} = %[[c0]] to %[[c10]] step %[[c1]] {
//       CHECK:     scf.if %{{.*}} {
//       CHECK:     } else {
//       CHECK:       vector.transfer_write
//       CHECK:     }
//       CHECK:   }
func.func @scf_if_inside_scf_for(
    %t1: tensor<?xf32> {bufferization.writable = true},
    %v: vector<5xf32>, %idx: index,
    %cond: i1)
  -> tensor<?xf32>
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %r = scf.for %iv = %c0 to %c10 step %c1 iter_args(%bb = %t1) -> (tensor<?xf32>) {
    %r2 = scf.if %cond -> (tensor<?xf32>) {
      scf.yield %bb : tensor<?xf32>
    } else {
      %t2 = vector.transfer_write %v, %bb[%idx] : vector<5xf32>, tensor<?xf32>
      scf.yield %t2 : tensor<?xf32>
    }
    scf.yield %r2 : tensor<?xf32>
  }
  return %r : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @scf_if_non_equiv_yields(
//  CHECK-SAME:     %[[cond:.*]]: i1, %[[A:.*]]: memref<{{.*}}>, %[[B:.*]]: memref<{{.*}}>) -> memref<{{.*}}>
func.func @scf_if_non_equiv_yields(
    %b : i1,
    %A : tensor<4xf32> {bufferization.writable = false},
    %B : tensor<4xf32> {bufferization.writable = false})
  -> tensor<4xf32>
{
  // CHECK: %[[r:.*]] = arith.select %[[cond]], %[[A]], %[[B]]
  %r = scf.if %b -> (tensor<4xf32>) {
    scf.yield %A : tensor<4xf32>
  } else {
    scf.yield %B : tensor<4xf32>
  }
  // CHECK: return %[[r]]
  return %r: tensor<4xf32>
}

// -----

// Note: This bufferization is inefficient, but it bufferizes correctly.

// CHECK-LABEL: func @scf_execute_region_yield_non_equivalent(
//       CHECK:   %[[alloc:.*]] = memref.alloc(%{{.*}})
//       CHECK:   %[[r:.*]] = memref.load %[[alloc]][%{{.*}}]
//       CHECK:   memref.dealloc %[[alloc]]
//       CHECK:   return %[[r]]
func.func @scf_execute_region_yield_non_equivalent(%i: index, %j: index) -> f32 {
  %r = scf.execute_region -> (tensor<?xf32>) {
    %t2 = bufferization.alloc_tensor(%i) : tensor<?xf32>
    scf.yield %t2 : tensor<?xf32>
  }
  %f = tensor.extract %r[%j] : tensor<?xf32>
  return %f : f32
}

// -----

// Note: This bufferizes to inefficient code, but bufferization should not see
// such IR in the first place. The iter_arg would canonicalize away. This test
// case is just to ensure that the bufferization generates correct code.

// CHECK-LABEL: func @scf_for_yield_non_equivalent(
//  CHECK-SAME:     %[[t:.*]]: memref<?xf32
//       CHECK:   %[[alloc:.*]] = memref.alloc(%{{.*}})
//       CHECK:   %[[cloned:.*]] = bufferization.clone %[[alloc]]
//       CHECK:   memref.dealloc %[[alloc]]
//       CHECK:   %[[for:.*]] = scf.for {{.*}} iter_args(%[[iter:.*]] = %[[cloned]])
//   CHECK-DAG:     memref.dealloc %[[iter]]
//   CHECK-DAG:     %[[alloc2:.*]] = memref.alloc(%{{.*}})
//       CHECK:     memref.copy %[[t]], %[[alloc2]]
//       CHECK:     %[[cloned2:.*]] = bufferization.clone %[[alloc2]]
//       CHECK:     memref.dealloc %[[alloc2]]
//       CHECK:     scf.yield %[[cloned2]]
//       CHECK:   return %[[for]]
func.func @scf_for_yield_non_equivalent(
    %t: tensor<?xf32>, %lb : index, %ub : index, %step : index) -> tensor<?xf32> {
  %r = scf.for %i = %lb to %ub step %step iter_args(%a = %t) -> tensor<?xf32> {
    scf.yield %t : tensor<?xf32>
  }

  return %r : tensor<?xf32>
}

// -----

// Note: This bufferizes to inefficient code, but bufferization should not see
// such IR in the first place. The iter_arg would canonicalize away. This test
// case is just to ensure that the bufferization generates correct code.

// CHECK-LABEL: func @scf_for_yield_allocation(
//  CHECK-SAME:     %[[t:.*]]: memref<?xf32
//       CHECK:   %[[cloned:.*]] = bufferization.clone %[[t]]
//       CHECK:   %[[for:.*]] = scf.for {{.*}} iter_args(%[[iter:.*]] = %[[cloned]])
// This alloc is for the bufferization.alloc_tensor.
//   CHECK-DAG:     %[[alloc2:.*]] = memref.alloc(%{{.*}})
//   CHECK-DAG:     memref.dealloc %[[iter]]
// This alloc is for the scf.yield.
//       CHECK:     %[[alloc3:.*]] = memref.alloc(%{{.*}})
//       CHECK:     memref.copy %[[alloc2]], %[[alloc3]]
//       CHECK:     memref.dealloc %[[alloc2]]
//       CHECK:     %[[casted3:.*]] = memref.cast %[[alloc3]]
//       CHECK:     %[[cloned3:.*]] = bufferization.clone %[[casted3]]
//       CHECK:     memref.dealloc %[[alloc3]]
//       CHECK:     scf.yield %[[cloned3]]
//       CHECK:   return %[[for]]
func.func @scf_for_yield_allocation(%t: tensor<?xf32>, %lb : index, %ub : index,
                               %step : index) -> tensor<?xf32> {
  %r = scf.for %i = %lb to %ub step %step iter_args(%a = %t) -> tensor<?xf32> {
    %t2 = bufferization.alloc_tensor(%i) : tensor<?xf32>
    scf.yield %t2 : tensor<?xf32>
  }

  return %r : tensor<?xf32>
}

// -----

// TODO: The scf.yield could bufferize to 1 alloc and 2 copies (instead of
// 2 allocs and 2 copies).

// CHECK-LABEL: func @scf_for_swapping_yields(
//  CHECK-SAME:     %[[A:.*]]: memref<?xf32, #{{.*}}>, %[[B:.*]]: memref<?xf32, #{{.*}}>
func.func @scf_for_swapping_yields(
    %A : tensor<?xf32>, %B : tensor<?xf32> {bufferization.writable = true},
    %C : tensor<4xf32>, %lb : index, %ub : index, %step : index)
  -> (f32, f32)
{
//   CHECK-DAG:   %[[clone1:.*]] = bufferization.clone %[[A]]
//   CHECK-DAG:   %[[clone2:.*]] = bufferization.clone %[[B]]
//       CHECK:   %[[for:.*]]:2 = scf.for {{.*}} iter_args(%[[iter1:.*]] = %[[clone1]], %[[iter2:.*]] = %[[clone2]])
  %r0:2 = scf.for %i = %lb to %ub step %step iter_args(%tA = %A, %tB = %B)
      -> (tensor<?xf32>, tensor<?xf32>)
  {
//       CHECK:     %[[sv1:.*]] = memref.subview %[[iter1]]
//       CHECK:     memref.copy %{{.*}}, %[[sv1]]
    %ttA = tensor.insert_slice %C into %tA[0][4][1] : tensor<4xf32> into tensor<?xf32>
//       CHECK:     %[[sv2:.*]] = memref.subview %[[iter2]]
//       CHECK:     memref.copy %{{.*}}, %[[sv2]]
    %ttB = tensor.insert_slice %C into %tB[0][4][1] : tensor<4xf32> into tensor<?xf32>

//       CHECK:     %[[alloc2:.*]] = memref.alloc(%{{.*}})
//       CHECK:     memref.copy %[[iter2]], %[[alloc2]]
//       CHECK:     memref.dealloc %[[iter2]]
//       CHECK:     %[[casted2:.*]] = memref.cast %[[alloc2]]
//       CHECK:     %[[alloc1:.*]] = memref.alloc(%{{.*}})
//       CHECK:     memref.copy %[[iter1]], %[[alloc1]]
//       CHECK:     memref.dealloc %[[iter1]]
//       CHECK:     %[[casted1:.*]] = memref.cast %[[alloc1]]
//       CHECK:     %[[cloned1:.*]] = bufferization.clone %[[casted1]]
//       CHECK:     memref.dealloc %[[alloc1]]
//       CHECK:     %[[cloned2:.*]] = bufferization.clone %[[casted2]]
//       CHECK:     memref.dealloc %[[alloc2]]
//       CHECK:     scf.yield %[[cloned2]], %[[cloned1]]
    // Yield tensors in different order.
    scf.yield %ttB, %ttA : tensor<?xf32>, tensor<?xf32>
  }

//       CHECK:     %[[r0:.*]] = memref.load %[[for]]#0
//       CHECK:     memref.dealloc %[[for]]#0
//       CHECK:     %[[r1:.*]] = memref.load %[[for]]#1
//       CHECK:     memref.dealloc %[[for]]#1
  %f0 = tensor.extract %r0#0[%step] : tensor<?xf32>
  %f1 = tensor.extract %r0#1[%step] : tensor<?xf32>
//       CHECK:     return %[[r0]], %[[r1]]
  return %f0, %f1: f32, f32
}

// -----

// CHECK-LABEL: func @scf_while(
//  CHECK-SAME:     %[[arg0:.*]]: memref<?xi1, #{{.*}}>
func.func @scf_while(%arg0: tensor<?xi1>, %idx: index) -> tensor<?xi1> {
  // CHECK: scf.while : () -> () {
  %res = scf.while (%arg1 = %arg0) : (tensor<?xi1>) -> tensor<?xi1> {
    // CHECK: %[[condition:.*]] = memref.load %[[arg0]]
    // CHECK: scf.condition(%[[condition]])
    %condition = tensor.extract %arg1[%idx] : tensor<?xi1>
    scf.condition(%condition) %arg1 : tensor<?xi1>
  } do {
  ^bb0(%arg2: tensor<?xi1>):
    // CHECK: } do {
    // CHECK: memref.store %{{.*}}, %[[arg0]]
    // CHECK: scf.yield
    // CHECK: }
    %pos = "dummy.some_op"() : () -> (index)
    %val = "dummy.another_op"() : () -> (i1)
    %1 = tensor.insert %val into %arg2[%pos] : tensor<?xi1>
    scf.yield %1 : tensor<?xi1>
  }

  // CHECK: return
  return %res : tensor<?xi1>
}

// -----

// The loop condition yields non-equivalent buffers.

// CHECK-LABEL: func @scf_while_non_equiv_condition(
//  CHECK-SAME:     %[[arg0:.*]]: memref<5xi1, #{{.*}}>, %[[arg1:.*]]: memref<5xi1, #{{.*}}>
func.func @scf_while_non_equiv_condition(%arg0: tensor<5xi1>,
                                         %arg1: tensor<5xi1>,
                                         %idx: index)
  -> (tensor<5xi1>, tensor<5xi1>)
{
  // CHECK: %[[clone1:.*]] = bufferization.clone %[[arg1]]
  // CHECK: %[[clone0:.*]] = bufferization.clone %[[arg0]]
  // CHECK: %[[loop:.*]]:2 = scf.while (%[[w0:.*]] = %[[clone0]], %[[w1:.*]] = %[[clone1]]) {{.*}} {
  %r0, %r1 = scf.while (%w0 = %arg0, %w1 = %arg1)
      : (tensor<5xi1>, tensor<5xi1>) -> (tensor<5xi1>, tensor<5xi1>) {
    // CHECK: %[[condition:.*]] = memref.load %[[w0]]
    // CHECK: %[[a1:.*]] = memref.alloc() {{.*}} : memref<5xi1>
    // CHECK: memref.copy %[[w1]], %[[a1]]
    // CHECK: memref.dealloc %[[w1]]
    // CHECK: %[[casted1:.*]] = memref.cast %[[a1]]
    // CHECK: %[[a0:.*]] = memref.alloc() {{.*}} : memref<5xi1>
    // CHECK: memref.copy %[[w0]], %[[a0]]
    // CHECK: memref.dealloc %[[w0]]
    // CHECK: %[[casted0:.*]] = memref.cast %[[a0]]
    // CHECK: %[[cloned0:.*]] = bufferization.clone %[[casted0]]
    // CHECK: memref.dealloc %[[a0]]
    // CHECK: %[[cloned1:.*]] = bufferization.clone %[[casted1]]
    // CHECK: memref.dealloc %[[a1]]
    // CHECK: scf.condition(%[[condition]]) %[[cloned1]], %[[cloned0]]
    %condition = tensor.extract %w0[%idx] : tensor<5xi1>
    scf.condition(%condition) %w1, %w0 : tensor<5xi1>, tensor<5xi1>
  } do {
  ^bb0(%b0: tensor<5xi1>, %b1: tensor<5xi1>):
    // CHECK: } do {
    // CHECK: ^bb0(%[[b0:.*]]: memref<5xi1, #{{.*}}>, %[[b1:.*]]: memref<5xi1, #{{.*}}):
    // CHECK: memref.store %{{.*}}, %[[b0]]
    // CHECK: %[[cloned2:.*]] = bufferization.clone %[[b1]]
    // CHECK: memref.dealloc %[[b1]]
    // CHECK: %[[cloned3:.*]] = bufferization.clone %[[b0]]
    // CHECK: memref.dealloc %[[b0]]
    // CHECK: scf.yield %[[cloned3]], %[[cloned2]]
    // CHECK: }
    %pos = "dummy.some_op"() : () -> (index)
    %val = "dummy.another_op"() : () -> (i1)
    %1 = tensor.insert %val into %b0[%pos] : tensor<5xi1>
    scf.yield %1, %b1 : tensor<5xi1>, tensor<5xi1>
  }

  // CHECK: return %[[loop]]#0, %[[loop]]#1
  return %r0, %r1 : tensor<5xi1>, tensor<5xi1>
}

// -----

// Both the loop condition and the loop buffer yield non-equivalent buffers.

// CHECK-LABEL: func @scf_while_non_equiv_condition_and_body(
//  CHECK-SAME:     %[[arg0:.*]]: memref<5xi1, #{{.*}}>, %[[arg1:.*]]: memref<5xi1, #{{.*}}>
func.func @scf_while_non_equiv_condition_and_body(%arg0: tensor<5xi1>,
                                                  %arg1: tensor<5xi1>,
                                                  %idx: index)
  -> (tensor<5xi1>, tensor<5xi1>)
{
  // CHECK: %[[clone1:.*]] = bufferization.clone %[[arg1]]
  // CHECK: %[[clone0:.*]] = bufferization.clone %[[arg0]]
  // CHECK: %[[loop:.*]]:2 = scf.while (%[[w0:.*]] = %[[clone0]], %[[w1:.*]] = %[[clone1]]) {{.*}} {
  %r0, %r1 = scf.while (%w0 = %arg0, %w1 = %arg1)
      : (tensor<5xi1>, tensor<5xi1>) -> (tensor<5xi1>, tensor<5xi1>) {
    // CHECK: %[[condition:.*]] = memref.load %[[w0]]
    // CHECK: %[[a1:.*]] = memref.alloc() {{.*}} : memref<5xi1>
    // CHECK: memref.copy %[[w1]], %[[a1]]
    // CHECK: memref.dealloc %[[w1]]
    // CHECK: %[[casted1:.*]] = memref.cast %[[a1]]
    // CHECK: %[[a0:.*]] = memref.alloc() {{.*}} : memref<5xi1>
    // CHECK: memref.copy %[[w0]], %[[a0]]
    // CHECK: memref.dealloc %[[w0]]
    // CHECK: %[[casted0:.*]] = memref.cast %[[a0]]
    // CHECK: %[[cloned0:.*]] = bufferization.clone %[[casted0]]
    // CHECK: memref.dealloc %[[a0]]
    // CHECK: %[[cloned1:.*]] = bufferization.clone %[[casted1]]
    // CHECK: memref.dealloc %[[a1]]
    // CHECK: scf.condition(%[[condition]]) %[[cloned1]], %[[cloned0]]
    %condition = tensor.extract %w0[%idx] : tensor<5xi1>
    scf.condition(%condition) %w1, %w0 : tensor<5xi1>, tensor<5xi1>
  } do {
  ^bb0(%b0: tensor<5xi1>, %b1: tensor<5xi1>):
    // CHECK: } do {
    // CHECK: ^bb0(%[[b0:.*]]: memref<5xi1, #{{.*}}>, %[[b1:.*]]: memref<5xi1, #{{.*}}):
    // CHECK: memref.store %{{.*}}, %[[b0]]
    // CHECK: %[[a3:.*]] = memref.alloc() {{.*}} : memref<5xi1>
    // CHECK: memref.copy %[[b1]], %[[a3]]
    // CHECK: memref.dealloc %[[b1]]
    // CHECK: %[[casted3:.*]] = memref.cast %[[a3]]
    // CHECK: %[[a2:.*]] = memref.alloc() {{.*}} : memref<5xi1>
    // CHECK: memref.copy %[[b0]], %[[a2]]
    // CHECK: %[[casted2:.*]] = memref.cast %[[a2]]
    // CHECK: %[[cloned2:.*]] = bufferization.clone %[[casted2]]
    // CHECK: memref.dealloc %[[a2]]
    // CHECK: %[[cloned3:.*]] = bufferization.clone %[[casted3]]
    // CHECK: memref.dealloc %[[a3]]
    // CHECK: scf.yield %[[cloned3]], %[[cloned2]]
    // CHECK: }
    %pos = "dummy.some_op"() : () -> (index)
    %val = "dummy.another_op"() : () -> (i1)
    %1 = tensor.insert %val into %b0[%pos] : tensor<5xi1>
    scf.yield %b1, %1 : tensor<5xi1>, tensor<5xi1>
  }

  // CHECK: return %[[loop]]#0, %[[loop]]#1
  return %r0, %r1 : tensor<5xi1>, tensor<5xi1>
}

// -----

// CHECK-LABEL: func @scf_while_iter_arg_result_mismatch(
//  CHECK-SAME:     %[[arg0:.*]]: memref<5xi1, #{{.*}}>, %[[arg1:.*]]: memref<5xi1, #{{.*}}>
//       CHECK:   %[[alloc2:.*]] = memref.alloc() {{.*}} : memref<5xi1>
//       CHECK:   %[[clone:.*]] = bufferization.clone %[[arg1]]
//       CHECK:   scf.while (%[[arg3:.*]] = %[[clone]]) : (memref<5xi1, #{{.*}}) -> () {
//   CHECK-DAG:     memref.dealloc %[[arg3]]
//   CHECK-DAG:     %[[load:.*]] = memref.load %[[arg0]]
//       CHECK:     scf.condition(%[[load]])
//       CHECK:   } do {
//       CHECK:     memref.copy %[[arg0]], %[[alloc2]]
//       CHECK:     memref.store %{{.*}}, %[[alloc2]]
//       CHECK:     %[[alloc1:.*]] = memref.alloc() {{.*}} : memref<5xi1>
//       CHECK:     memref.copy %[[alloc2]], %[[alloc1]]
//       CHECK:     %[[casted:.*]] = memref.cast %[[alloc1]] : memref<5xi1> to memref<5xi1, #{{.*}}>
//       CHECK:     %[[cloned:.*]] = bufferization.clone %[[casted]]
//       CHECK:     memref.dealloc %[[alloc1]]
//       CHECK:     scf.yield %[[cloned]]
//       CHECK:   }
//   CHECK-DAG:   memref.dealloc %[[alloc2]]
func.func @scf_while_iter_arg_result_mismatch(%arg0: tensor<5xi1>,
                                              %arg1: tensor<5xi1>,
                                              %arg2: index) {
  scf.while (%arg3 = %arg1) : (tensor<5xi1>) -> () {
    %0 = tensor.extract %arg0[%arg2] : tensor<5xi1>
    scf.condition(%0)
  } do {
    %0 = "dummy.some_op"() : () -> index
    %1 = "dummy.another_op"() : () -> i1
    %2 = tensor.insert %1 into %arg0[%0] : tensor<5xi1>
    scf.yield %2 : tensor<5xi1>
  }
  return
}

// -----

// CHECK-LABEL: func.func @parallel_insert_slice_no_conflict(
//  CHECK-SAME:     %[[idx:.*]]: index, %[[idx2:.*]]: index,
//  CHECK-SAME:     %[[arg1:.*]]: memref<?xf32, #{{.*}}>,
//  CHECK-SAME:     %[[arg2:.*]]: memref<?xf32, #{{.*}}>
func.func @parallel_insert_slice_no_conflict(
    %idx: index,
    %idx2: index,
    %arg1: tensor<?xf32> {bufferization.writable = true},
    %arg2: tensor<?xf32> {bufferization.writable = true}) -> (tensor<?xf32>, f32) {
  %cst = arith.constant 4.200000e+01 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK: scf.foreach_thread (%[[tidx:.*]]) in (%[[idx2]])  -> ()
  %2 = scf.foreach_thread (%arg3) in (%idx2)  -> (tensor<?xf32>) {
      // CHECK: %[[subview:.*]] = memref.subview %[[arg2]][5] [%[[idx]]] [1]
      %6 = tensor.extract_slice %arg2[5] [%idx] [%c1] : tensor<?xf32> to tensor<?xf32>
      // CHECK: linalg.fill ins(%{{.*}}) outs(%[[subview]] : memref<?xf32
      %8 = linalg.fill ins(%cst : f32) outs(%6 : tensor<?xf32>) -> tensor<?xf32>
      // Self-copy will DCE away later.
      // CHECK: memref.copy %[[subview]], %[[subview]]

      // Empty terminator is elided from pretty-printing.
      // CHECK-NOT: scf.foreach_thread.perform_concurrently
      // CHECK-NOT: parallel_insert_slice
      scf.foreach_thread.perform_concurrently {
        scf.foreach_thread.parallel_insert_slice %8 into %arg2[5] [%idx] [%c1] : 
          tensor<?xf32> into tensor<?xf32>
      }
  }

  // CHECK: %[[load:.*]] = memref.load %[[arg2]]
  %f = tensor.extract %2[%c0] : tensor<?xf32>

  // CHECK: return %[[load]] : f32
  return %2, %f : tensor<?xf32>, f32
}

// -----

// CHECK-LABEL: func.func @parallel_insert_slice_with_conflict(
//  CHECK-SAME:     %[[idx:.*]]: index, %[[idx2:.*]]: index,
//  CHECK-SAME:     %[[arg1:.*]]: memref<?xf32, #{{.*}}>,
//  CHECK-SAME:     %[[arg2:.*]]: memref<?xf32, #{{.*}}>
func.func @parallel_insert_slice_with_conflict(
    %idx: index, 
    %idx2: index, 
    %arg1: tensor<?xf32> {bufferization.writable = true},
    %arg2: tensor<?xf32> {bufferization.writable = true}) -> (f32, f32)
{
  %cst = arith.constant 4.200000e+01 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // The parallel_insert_slice_op bufferizes out-of-place due to a RAW conflict
  // on %arg2, so we need an allocation.
  // CHECK: %[[alloc1:.*]] = memref.alloc
  // CHECK: memref.copy %[[arg2]], %[[alloc1]]

  // CHECK: scf.foreach_thread (%[[tidx:.*]]) in (%[[idx2]])  -> ()
  %2 = scf.foreach_thread (%arg3) in (%idx2)  -> (tensor<?xf32>) {
      // Another alloc for the extract_slice op.
      // CHECK: %[[alloc2:.*]] = memref.alloc
      %6 = tensor.extract_slice %arg2[5] [%idx] [%c1] : tensor<?xf32> to tensor<?xf32>

      // CHECK: linalg.fill ins(%{{.*}}) outs(%[[alloc2]] : memref<?xf32
      %8 = linalg.fill ins(%cst : f32) outs(%6 : tensor<?xf32>) -> tensor<?xf32>

      // Now the copy of the actual insert_slice.
      // CHECK: %[[subview1:.*]] = memref.subview %[[alloc1]][5] [%[[idx]]] [1]
      //
      // CHECK: memref.copy %[[alloc2]], %[[subview1]]
      // CHECK: memref.dealloc %[[alloc2]]

      // Empty terminator is elided from pretty-printing.
      // CHECK-NOT: scf.foreach_thread.perform_concurrently
      // CHECK-NOT: parallel_insert_slice
      scf.foreach_thread.perform_concurrently {
        scf.foreach_thread.parallel_insert_slice %8 into %arg2[5] [%idx] [%c1] :
          tensor<?xf32> into tensor<?xf32>
      }
  }

  // CHECK: %[[load:.*]] = memref.load %[[arg2]]
  // CHECK: %[[load2:.*]] = memref.load %[[alloc1]]
  // CHECK: memref.dealloc %[[alloc1]]
  %f = tensor.extract %arg2[%c0] : tensor<?xf32>
  %f2 = tensor.extract %2[%c0] : tensor<?xf32>

  // CHECK: return %[[load2]], %[[load]] : f32, f32
  return %f2, %f : f32, f32
}

// -----

#map0 = affine_map<(d0) -> (d0 * 4)>
#map1 = affine_map<(d0) -> (d0 * 2)>

// CHECK: #[[$DYN_LAYOUT_MAP:.*]] = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>

// CHECK-LABEL: func.func @matmul
func.func @matmul(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>, %arg2: tensor<8x8xf32> {bufferization.writable = true}) -> tensor<8x8xf32> {
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index

  // CHECK: scf.foreach_thread {{.*}}  -> ()
  %0 = scf.foreach_thread (%arg3, %arg4) in (%c2, %c4) -> (tensor<8x8xf32>) {
    %1 = affine.apply #map0(%arg3)
    %3 = tensor.extract_slice %arg0[%1, 0] [4, 8] [1, 1] : tensor<8x8xf32> to tensor<4x8xf32>
    %4 = affine.apply #map1(%arg4)
    %6 = tensor.extract_slice %arg1[0, %4] [8, 4] [1, 1] : tensor<8x8xf32> to tensor<8x4xf32>
    %7 = tensor.extract_slice %arg2[%1, %4] [4, 4] [1, 1] : tensor<8x8xf32> to tensor<4x4xf32>
 
    //      CHECK: linalg.matmul ins({{.*}}memref<4x8xf32, #[[$DYN_LAYOUT_MAP]]>, memref<8x4xf32, #[[$DYN_LAYOUT_MAP]]>) outs({{.*}} : memref<4x4xf32, #[[$DYN_LAYOUT_MAP]]>)
    %8 = linalg.matmul ins(%3, %6 : tensor<4x8xf32>, tensor<8x4xf32>) outs(%7 : tensor<4x4xf32>) -> tensor<4x4xf32>
    scf.foreach_thread.perform_concurrently {
      scf.foreach_thread.parallel_insert_slice %8 into %arg2[%1, %4] [4, 4] [1, 1] : tensor<4x4xf32> into tensor<8x8xf32>
    }
  }
  return %0 : tensor<8x8xf32>
}
