// RUN: mlir-opt %s -allow-unregistered-dialect -test-linalg-transform-patterns=test-tiled-loop-peeling=2 -split-input-file | FileCheck %s -check-prefix=CHECK-TILE-2
// RUN: mlir-opt %s -allow-unregistered-dialect -test-linalg-transform-patterns=test-tiled-loop-peeling=0,1,2 -split-input-file | FileCheck %s -check-prefix=CHECK-TILE-012
// RUN: mlir-opt %s -allow-unregistered-dialect -test-linalg-transform-patterns="test-tiled-loop-peeling=0,1,2 skip-partial" -split-input-file | FileCheck %s -check-prefix=CHECK-TILE-012-SKIP-PARTIAL

// CHECK-TILE-2-LABEL: func @tiled_loop_3d_tensor(
//  CHECK-TILE-2-SAME:     %[[input:.*]]: tensor<?x?x?xf32>, %[[s0:.*]]: index, %[[s1:.*]]: index, %[[s2:.*]]: index
//   CHECK-TILE-2-DAG:   %[[c0:.*]] = arith.constant 0 : index
//   CHECK-TILE-2-DAG:   %[[c1:.*]] = arith.constant 1 : index
//   CHECK-TILE-2-DAG:   %[[c2:.*]] = arith.constant 2 : index
//       CHECK-TILE-2:   %[[dim0:.*]] = tensor.dim %[[input]], %[[c0]]
//       CHECK-TILE-2:   %[[dim1:.*]] = tensor.dim %[[input]], %[[c1]]
//       CHECK-TILE-2:   %[[dim2:.*]] = tensor.dim %[[input]], %[[c2]]
//       CHECK-TILE-2:   %[[init_tensor:.*]] = linalg.init_tensor
//       CHECK-TILE-2:   %[[split_bound:.*]] = affine.apply
//       CHECK-TILE-2:   %[[r1:.*]] = linalg.tiled_loop (%[[iv0:.*]], %[[iv1:.*]], %[[iv2:.*]]) = (%[[c0]], %[[c0]], %[[c0]])
//  CHECK-TILE-2-SAME:       to (%[[dim0]], %[[dim1]], %[[split_bound]])
//  CHECK-TILE-2-SAME:       step (%[[s0]], %[[s1]], %[[s2]])
//  CHECK-TILE-2-SAME:       ins (%[[loop_in1:.*]] = %[[input]]: tensor<?x?x?xf32>)
//  CHECK-TILE-2-SAME:       outs (%[[loop_out1:.*]] = %[[init_tensor]]: tensor<?x?x?xf32>) {
//       CHECK-TILE-2:     %[[min0_1:.*]] = affine.min
//       CHECK-TILE-2:     %[[min1_1:.*]] = affine.min
//       CHECK-TILE-2:     %[[in_slice1:.*]] = tensor.extract_slice %[[loop_in1]][%[[iv0]], %[[iv1]], %[[iv2]]] [%[[min0_1]], %[[min1_1]], %[[s2]]]
//       CHECK-TILE-2:     %[[out_slice1:.*]] = tensor.extract_slice %[[loop_out1]][%[[iv0]], %[[iv1]], %[[iv2]]] [%[[min0_1]], %[[min1_1]], %[[s2]]]
//       CHECK-TILE-2:     %[[mod_slice1:.*]] = tensor.insert_slice %{{.*}} into %[[loop_out1]][%[[iv0]], %[[iv1]], %[[iv2]]] [%[[min0_1]], %[[min1_1]], %[[s2]]]
//       CHECK-TILE-2:     linalg.yield %[[mod_slice1]]
//       CHECK-TILE-2:   %[[r2:.*]] = linalg.tiled_loop (%[[iv0:.*]], %[[iv1:.*]], %[[iv2:.*]]) = (%[[c0]], %[[c0]], %[[split_bound]])
//  CHECK-TILE-2-SAME:       to (%[[dim0]], %[[dim1]], %[[dim2]])
//  CHECK-TILE-2-SAME:       step (%[[s0]], %[[s1]], %[[s2]])
//  CHECK-TILE-2-SAME:       ins (%[[loop_in2:.*]] = %[[input]]: tensor<?x?x?xf32>)
//  CHECK-TILE-2-SAME:       outs (%[[loop_out2:.*]] = %[[r1]]: tensor<?x?x?xf32>) {
//       CHECK-TILE-2:     %[[min0_2:.*]] = affine.min
//       CHECK-TILE-2:     %[[min1_2:.*]] = affine.min
//       CHECK-TILE-2:     %[[apply2:.*]] = affine.apply
//       CHECK-TILE-2:     %[[in_slice2:.*]] = tensor.extract_slice %[[loop_in1]][%[[iv0]], %[[iv1]], %[[iv2]]] [%[[min0_2]], %[[min1_2]], %[[apply2]]]
//       CHECK-TILE-2:     %[[out_slice2:.*]] = tensor.extract_slice %[[loop_out1]][%[[iv0]], %[[iv1]], %[[iv2]]] [%[[min0_2]], %[[min1_2]], %[[apply2]]]
//       CHECK-TILE-2:     %[[mod_slice2:.*]] = tensor.insert_slice %{{.*}} into %[[loop_out1]][%[[iv0]], %[[iv1]], %[[iv2]]] [%[[min0_2]], %[[min1_2]], %[[apply2]]]
//       CHECK-TILE-2:     linalg.yield %[[mod_slice2]]
//       CHECK-TILE-2:   return %[[r2]]

// CHECK-TILE-012-LABEL: func @tiled_loop_3d_tensor
//       CHECK-TILE-012:   linalg.tiled_loop {{.*}} {
//       CHECK-TILE-012:     linalg.yield
//       CHECK-TILE-012:   }
//       CHECK-TILE-012:   linalg.tiled_loop {{.*}} {
//       CHECK-TILE-012:     linalg.yield
//       CHECK-TILE-012:   }
//       CHECK-TILE-012:   linalg.tiled_loop {{.*}} {
//       CHECK-TILE-012:     linalg.yield
//       CHECK-TILE-012:   }
//       CHECK-TILE-012:   linalg.tiled_loop {{.*}} {
//       CHECK-TILE-012:     linalg.yield
//       CHECK-TILE-012:   }
//       CHECK-TILE-012:   linalg.tiled_loop {{.*}} {
//       CHECK-TILE-012:     linalg.yield
//       CHECK-TILE-012:   }
//       CHECK-TILE-012:   linalg.tiled_loop {{.*}} {
//       CHECK-TILE-012:     linalg.yield
//       CHECK-TILE-012:   }
//       CHECK-TILE-012:   linalg.tiled_loop {{.*}} {
//       CHECK-TILE-012:     linalg.yield
//       CHECK-TILE-012:   }
//       CHECK-TILE-012:   linalg.tiled_loop {{.*}} {
//       CHECK-TILE-012:     linalg.yield
//       CHECK-TILE-012:   }
//   CHECK-TILE-012-NOT: linalg.tiled_loop

//      CHECK-TILE-012-SKIP-PARTIAL: func @tiled_loop_3d_tensor(
// CHECK-TILE-012-SKIP-PARTIAL-SAME:     %[[input:.*]]: tensor<?x?x?xf32>
//  CHECK-TILE-012-SKIP-PARTIAL-DAG:   %[[c0:.*]] = arith.constant 0 : index
//  CHECK-TILE-012-SKIP-PARTIAL-DAG:   %[[c1:.*]] = arith.constant 1 : index
//  CHECK-TILE-012-SKIP-PARTIAL-DAG:   %[[c2:.*]] = arith.constant 2 : index
//  CHECK-TILE-012-SKIP-PARTIAL-DAG:   %[[dim0:.*]] = tensor.dim %[[input]], %[[c0]]
//  CHECK-TILE-012-SKIP-PARTIAL-DAG:   %[[dim1:.*]] = tensor.dim %[[input]], %[[c1]]
//  CHECK-TILE-012-SKIP-PARTIAL-DAG:   %[[dim2:.*]] = tensor.dim %[[input]], %[[c2]]
//      CHECK-TILE-012-SKIP-PARTIAL:   %[[p0:.*]] = affine.apply #{{.*}}()[%[[dim0]]
//      CHECK-TILE-012-SKIP-PARTIAL:   %[[p1:.*]] = affine.apply #{{.*}}()[%[[dim1]]
//      CHECK-TILE-012-SKIP-PARTIAL:   %[[p2:.*]] = affine.apply #{{.*}}()[%[[dim2]]
//      CHECK-TILE-012-SKIP-PARTIAL:   linalg.tiled_loop {{.*}} = (%[[c0]], %[[c0]], %[[c0]]) to (%[[p0]], %[[p1]], %[[p2]])
//      CHECK-TILE-012-SKIP-PARTIAL:   linalg.tiled_loop {{.*}} = (%[[c0]], %[[c0]], %[[p2]]) to (%[[p0]], %[[p1]], %[[dim2]])
//      CHECK-TILE-012-SKIP-PARTIAL:   linalg.tiled_loop {{.*}} = (%[[c0]], %[[p1]], %[[c0]]) to (%[[p0]], %[[dim1]], %[[dim2]])
//      CHECK-TILE-012-SKIP-PARTIAL:   linalg.tiled_loop {{.*}} = (%[[p0]], %[[c0]], %[[c0]]) to (%[[dim0]], %[[dim1]], %[[dim2]])
func @tiled_loop_3d_tensor(%arg0: tensor<?x?x?xf32>, %s0: index, %s1: index,
                           %s2: index) -> tensor<?x?x?xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %dim0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %dim1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %dim2 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
  %output = linalg.init_tensor [%dim0, %dim1, %dim2] : tensor<?x?x?xf32>
  %result = linalg.tiled_loop
           (%arg1, %arg2, %arg3) = (%c0, %c0, %c0) to (%dim0, %dim1, %dim2)
           step (%s0, %s1, %s2) ins (%arg4 = %arg0: tensor<?x?x?xf32>)
           outs (%arg5 = %output: tensor<?x?x?xf32>) {
    %min0 = affine.min affine_map<(d0, d1)[s0] -> (d1, -d0 + s0)>(%arg1, %s0)[%dim0]
    %min1 = affine.min affine_map<(d0, d1)[s0] -> (d1, -d0 + s0)>(%arg2, %s1)[%dim1]
    %min2 = affine.min affine_map<(d0, d1)[s0] -> (d1, -d0 + s0)>(%arg3, %s2)[%dim2]
    %in_slice = tensor.extract_slice %arg4[%arg1, %arg2, %arg3] [%min0, %min1, %min2] [1, 1, 1]: tensor<?x?x?xf32> to tensor<?x?x?xf32>
    %out_slice = tensor.extract_slice %arg5[%arg1, %arg2, %arg3] [%min0, %min1, %min2] [1, 1, 1] : tensor<?x?x?xf32> to tensor<?x?x?xf32>
    %comp = "computation"(%in_slice, %out_slice) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %updated_slice = tensor.insert_slice %comp into %arg5[%arg1, %arg2, %arg3] [%min0, %min1, %min2] [1, 1, 1] : tensor<?x?x?xf32> into tensor<?x?x?xf32>
    linalg.yield %updated_slice : tensor<?x?x?xf32>
  }
  return %result : tensor<?x?x?xf32>
}

// -----

// CHECK-TILE-2-LABEL: func @tiled_loop_3d_memref(
//  CHECK-TILE-2-SAME:     %[[input:.*]]: memref<?x?x?xf32>, %[[output:.*]]: memref<?x?x?xf32>, %[[s0:.*]]: index, %[[s1:.*]]: index, %[[s2:.*]]: index
//   CHECK-TILE-2-DAG:   %[[c0:.*]] = arith.constant 0 : index
//   CHECK-TILE-2-DAG:   %[[c1:.*]] = arith.constant 1 : index
//   CHECK-TILE-2-DAG:   %[[c2:.*]] = arith.constant 2 : index
//       CHECK-TILE-2:   %[[dim0:.*]] = memref.dim %[[input]], %[[c0]]
//       CHECK-TILE-2:   %[[dim1:.*]] = memref.dim %[[input]], %[[c1]]
//       CHECK-TILE-2:   %[[dim2:.*]] = memref.dim %[[input]], %[[c2]]
//       CHECK-TILE-2:   %[[split_bound:.*]] = affine.apply
//       CHECK-TILE-2:   linalg.tiled_loop (%[[iv0:.*]], %[[iv1:.*]], %[[iv2:.*]]) = (%[[c0]], %[[c0]], %[[c0]])
//  CHECK-TILE-2-SAME:       to (%[[dim0]], %[[dim1]], %[[split_bound]])
//  CHECK-TILE-2-SAME:       step (%[[s0]], %[[s1]], %[[s2]])
//  CHECK-TILE-2-SAME:       ins (%[[loop_in1:.*]] = %[[input]]: memref<?x?x?xf32>)
//  CHECK-TILE-2-SAME:       outs (%[[loop_out1:.*]] = %[[output]]: memref<?x?x?xf32>) {
//       CHECK-TILE-2:     %[[min0_1:.*]] = affine.min
//       CHECK-TILE-2:     %[[min1_1:.*]] = affine.min
//       CHECK-TILE-2:     memref.subview %[[loop_in1]][%[[iv0]], %[[iv1]], %[[iv2]]] [%[[min0_1]], %[[min1_1]], %[[s2]]]
//       CHECK-TILE-2:     linalg.yield
//       CHECK-TILE-2:   linalg.tiled_loop (%[[iv0:.*]], %[[iv1:.*]], %[[iv2:.*]]) = (%[[c0]], %[[c0]], %[[split_bound]])
//  CHECK-TILE-2-SAME:       to (%[[dim0]], %[[dim1]], %[[dim2]])
//  CHECK-TILE-2-SAME:       step (%[[s0]], %[[s1]], %[[s2]])
//  CHECK-TILE-2-SAME:       ins (%[[loop_in2:.*]] = %[[input]]: memref<?x?x?xf32>)
//  CHECK-TILE-2-SAME:       outs (%[[loop_out2:.*]] = %[[output]]: memref<?x?x?xf32>) {
//       CHECK-TILE-2:     %[[min0_2:.*]] = affine.min
//       CHECK-TILE-2:     %[[min1_2:.*]] = affine.min
//       CHECK-TILE-2:     %[[apply2:.*]] = affine.apply
//       CHECK-TILE-2:     memref.subview %[[loop_in1]][%[[iv0]], %[[iv1]], %[[iv2]]] [%[[min0_2]], %[[min1_2]], %[[apply2]]]
//       CHECK-TILE-2:     linalg.yield
//       CHECK-TILE-2:   return

// CHECK-TILE-012-LABEL: func @tiled_loop_3d_memref

!memref_subview_type = type memref<?x?x?xf32, affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2 + d2)>>

func @tiled_loop_3d_memref(%arg0: memref<?x?x?xf32>, %output: memref<?x?x?xf32>,
                           %s0: index, %s1: index, %s2: index) {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %dim0 = memref.dim %arg0, %c0 : memref<?x?x?xf32>
  %dim1 = memref.dim %arg0, %c1 : memref<?x?x?xf32>
  %dim2 = memref.dim %arg0, %c2 : memref<?x?x?xf32>
  linalg.tiled_loop
           (%arg1, %arg2, %arg3) = (%c0, %c0, %c0) to (%dim0, %dim1, %dim2)
           step (%s0, %s1, %s2) ins (%arg4 = %arg0: memref<?x?x?xf32>)
           outs (%arg5 = %output : memref<?x?x?xf32>) {
    %min0 = affine.min affine_map<(d0, d1)[s0] -> (d1, -d0 + s0)>(%arg1, %s0)[%dim0]
    %min1 = affine.min affine_map<(d0, d1)[s0] -> (d1, -d0 + s0)>(%arg2, %s1)[%dim1]
    %min2 = affine.min affine_map<(d0, d1)[s0] -> (d1, -d0 + s0)>(%arg3, %s2)[%dim2]
    %in_slice = memref.subview %arg4[%arg1, %arg2, %arg3] [%min0, %min1, %min2] [1, 1, 1]: memref<?x?x?xf32> to !memref_subview_type
    "computation"(%in_slice) : (!memref_subview_type) -> memref<?x?x?xf32>
    linalg.yield
  }
  return
}

// -----

// CHECK-TILE-2-LABEL: func @step_1_do_not_peel
//       CHECK-TILE-2:   linalg.tiled_loop
//   CHECK-TILE-2-NOT:   linalg.tiled_loop

// CHECK-TILE-012-LABEL: func @step_1_do_not_peel

func @step_1_do_not_peel(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %dim0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %dim1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %dim2 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
  %output = linalg.init_tensor [%dim0, %dim1, %dim2] : tensor<?x?x?xf32>
  %result = linalg.tiled_loop
           (%arg1, %arg2, %arg3) = (%c0, %c0, %c0) to (%dim0, %dim1, %dim2)
           step (%c1, %c1, %c1) ins (%arg4 = %arg0: tensor<?x?x?xf32>)
           outs (%arg5 = %output: tensor<?x?x?xf32>) {
    %in_slice = tensor.extract_slice %arg4[%arg1, %arg2, %arg3] [%c1, %c1, %c1] [1, 1, 1]: tensor<?x?x?xf32> to tensor<?x?x?xf32>
    %out_slice = tensor.extract_slice %arg5[%arg1, %arg2, %arg3] [%c1, %c1, %c1] [1, 1, 1] : tensor<?x?x?xf32> to tensor<?x?x?xf32>
    %comp = "computation"(%in_slice, %out_slice) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %updated_slice = tensor.insert_slice %comp into %arg5[%arg1, %arg2, %arg3] [%c1, %c1, %c1] [1, 1, 1] : tensor<?x?x?xf32> into tensor<?x?x?xf32>
    linalg.yield %updated_slice : tensor<?x?x?xf32>
  }
  return %result : tensor<?x?x?xf32>
}

// -----

// CHECK-TILE-2-LABEL: func @divides_evenly_do_not_peel
//       CHECK-TILE-2:   linalg.tiled_loop
//   CHECK-TILE-2-NOT:   linalg.tiled_loop

// CHECK-TILE-012-LABEL: func @divides_evenly_do_not_peel

func @divides_evenly_do_not_peel(%arg0: tensor<?x?x?xf32>, %s: index)
    -> tensor<?x?x?xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  %c64 = arith.constant 64 : index
  %dim0 = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
  %dim1 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
  %dim2 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
  %output = linalg.init_tensor [%dim0, %dim1, %dim2] : tensor<?x?x?xf32>
  %result = linalg.tiled_loop
           (%arg1, %arg2, %arg3) = (%c0, %c0, %c0) to (%dim0, %dim1, %c64)
           step (%s, %s, %c8) ins (%arg4 = %arg0: tensor<?x?x?xf32>)
           outs (%arg5 = %output: tensor<?x?x?xf32>) {
    %in_slice = tensor.extract_slice %arg4[%arg1, %arg2, %arg3] [%c1, %c1, %c1] [1, 1, 1]: tensor<?x?x?xf32> to tensor<?x?x?xf32>
    %out_slice = tensor.extract_slice %arg5[%arg1, %arg2, %arg3] [%c1, %c1, %c1] [1, 1, 1] : tensor<?x?x?xf32> to tensor<?x?x?xf32>
    %comp = "computation"(%in_slice, %out_slice) : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    %updated_slice = tensor.insert_slice %comp into %arg5[%arg1, %arg2, %arg3] [%c1, %c1, %c1] [1, 1, 1] : tensor<?x?x?xf32> into tensor<?x?x?xf32>
    linalg.yield %updated_slice : tensor<?x?x?xf32>
  }
  return %result : tensor<?x?x?xf32>
}
