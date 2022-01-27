// RUN: mlir-opt %s -linalg-fuse-elementwise-ops -split-input-file | FileCheck %s

// CHECK-DAG: [[$MAP0:#[a-zA-Z0-9_]*]] = affine_map<(d0, d1) -> (d0, d1)>
#map0 = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @add_mul_fusion
func @add_mul_fusion(%arg0: tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %2 = linalg.init_tensor [%0, %1] : tensor<?x?xf32>
  %3 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%2 : tensor<?x?xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):       
      %4 = arith.addf %arg3, %arg4 : f32
      linalg.yield %4 : f32
  } -> tensor<?x?xf32>
  // CHECK: linalg.generic {
  // CHECK-SAME: indexing_maps = {{\[}}[[$MAP0]], [[$MAP0]], [[$MAP0]], [[$MAP0]]{{\]}}
  %4 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%3, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%2 : tensor<?x?xf32>) {
    // CHECK: ^{{[a-zA-Z0-9_]*}}
    // CHECK-SAME: [[ARG0:%[a-zA-Z0-9_]*]]
    // CHECK-SAME: [[ARG1:%[a-zA-Z0-9_]*]]
    // CHECK-SAME: [[ARG2:%[a-zA-Z0-9_]*]]
    ^bb0(%arg5: f32, %arg6: f32, %arg7: f32):       
      // CHECK: [[T1:%[a-zA-Z0-9_]*]] = arith.addf [[ARG0]], [[ARG1]]
      // CHECK-NOT: linalg.yield
      // CHECK: arith.mulf [[T1]], [[ARG2]]
      // CHECK: linalg.yield
      %5 = arith.mulf %arg5, %arg6 : f32
      linalg.yield %5 : f32
    } -> tensor<?x?xf32>
  return %4 : tensor<?x?xf32>
}

// -----

// CHECK-DAG: [[$MAP0:#[a-zA-Z0-9_]*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: [[$MAP1:#[a-zA-Z0-9_]*]] = affine_map<(d0, d1) -> ()>
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ()>

// CHECK-LABEL: @scalar_add_mul_fusion
func @scalar_add_mul_fusion(%arg0: tensor<?x?xf32>, %arg1 : f32, %arg2 : f32) -> tensor<?x?xf32>
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %2 = linalg.init_tensor [%0, %1] : tensor<?x?xf32>
  %3 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<?x?xf32>, f32)
      outs(%2 : tensor<?x?xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):       
      %4 = arith.addf %arg3, %arg4 : f32
      linalg.yield %4 : f32
  } -> tensor<?x?xf32>
  // CHECK: linalg.generic {
  // CHECK-SAME: indexing_maps = {{\[}}[[$MAP0]], [[$MAP1]], [[$MAP1]], [[$MAP0]]{{\]}}
  %4 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%3, %arg2 : tensor<?x?xf32>, f32)
      outs(%2 : tensor<?x?xf32>) {
    // CHECK: ^{{[a-zA-Z0-9_]*}}
    // CHECK-SAME: [[ARG3:%[a-zA-Z0-9_]*]]
    // CHECK-SAME: [[ARG4:%[a-zA-Z0-9_]*]]
    // CHECK-SAME: [[ARG5:%[a-zA-Z0-9_]*]]
    ^bb0(%arg5: f32, %arg6: f32, %arg7: f32):       
      // CHECK: [[T1:%[a-zA-Z0-9_]*]] = arith.addf [[ARG3]], [[ARG4]]
      // CHECK-NOT: linalg.yield
      // CHECK: arith.mulf [[T1]], [[ARG5]]
      // CHECK: linalg.yield
      %5 = arith.mulf %arg5, %arg6 : f32
      linalg.yield %5 : f32
    } -> tensor<?x?xf32>
  return %4 : tensor<?x?xf32>
}

// -----

// CHECK-DAG: [[$MAP0:#[a-zA-Z0-9_]*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: [[$MAP1:#[a-zA-Z0-9_]*]] = affine_map<(d0, d1) -> (d1, d0)>
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: @transpose_add_mul_fusion
func @transpose_add_mul_fusion(%arg0: tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %2 = linalg.init_tensor [%0, %1] : tensor<?x?xf32>
  %3 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%2 : tensor<?x?xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):       
      %4 = arith.addf %arg3, %arg4 : f32
      linalg.yield %4 : f32
  } -> tensor<?x?xf32>
  // CHECK: linalg.generic {
  // CHECK-SAME: indexing_maps = {{\[}}[[$MAP0]], [[$MAP1]], [[$MAP0]], [[$MAP0]]{{\]}}
  %4 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%3, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%2 : tensor<?x?xf32>) {
    ^bb0(%arg5: f32, %arg6: f32, %arg7: f32):       
      %5 = arith.mulf %arg5, %arg6 : f32
      linalg.yield %5 : f32
    } -> tensor<?x?xf32>
  return %4 : tensor<?x?xf32>
}

// -----

// CHECK-DAG: [[$MAP0:#[a-zA-Z0-9_]*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: [[$MAP1:#[a-zA-Z0-9_]*]] = affine_map<(d0, d1) -> (d1, d0)>
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: @add_transpose_mul_fusion
func @add_transpose_mul_fusion(%arg0: tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %2 = linalg.init_tensor [%0, %1] : tensor<?x?xf32>
  %3 = linalg.generic {indexing_maps = [#map0, #map1, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%2 : tensor<?x?xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):       
      %4 = arith.addf %arg3, %arg4 : f32
      linalg.yield %4 : f32
  } -> tensor<?x?xf32>
  // CHECK: linalg.generic {
  // CHECK-SAME: indexing_maps = {{\[}}[[$MAP1]], [[$MAP0]], [[$MAP0]], [[$MAP0]]{{\]}}
  %4 = linalg.generic {indexing_maps = [#map1, #map0, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%3, %arg2 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%2 : tensor<?x?xf32>){
    ^bb0(%arg5: f32, %arg6: f32, %arg7: f32):       
      %5 = arith.mulf %arg5, %arg6 : f32
      linalg.yield %5 : f32
    } -> tensor<?x?xf32>
  return %4 : tensor<?x?xf32>
}

// -----

// CHECK-DAG: [[$MAP0:#[a-zA-Z0-9_]*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: [[$MAP1:#[a-zA-Z0-9_]*]] = affine_map<(d0, d1) -> (d0)>
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: @add_broadcast_mul_fusion
func @add_broadcast_mul_fusion(%arg0: tensor<?xf32>, %arg1 : tensor<?xf32>, %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?xf32>
  %1 = linalg.init_tensor [%0] : tensor<?xf32>
  %2 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]}
      ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>)
      outs(%1 : tensor<?xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):       
      %3 = arith.addf %arg3, %arg4 : f32
      linalg.yield %3 : f32
  } -> tensor<?xf32>
  // CHECK: linalg.generic {
  // CHECK-SAME: indexing_maps = {{\[}}[[$MAP1]], [[$MAP1]], [[$MAP0]], [[$MAP0]]
  %3 = tensor.dim %arg2, %c1 : tensor<?x?xf32>
  %4 = linalg.init_tensor [%0, %3] : tensor<?x?xf32>
  %5 = linalg.generic {indexing_maps = [#map1, #map0, #map0], iterator_types = ["parallel", "parallel"]}
      ins(%2, %arg2 : tensor<?xf32>, tensor<?x?xf32>)
      outs(%4 : tensor<?x?xf32>){
    ^bb0(%arg5: f32, %arg6: f32, %arg7: f32):       
      %6 = arith.mulf %arg5, %arg6 : f32
      linalg.yield %6 : f32
    } -> tensor<?x?xf32>
  return %5 : tensor<?x?xf32>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<() -> ()>
#map0 = affine_map<() -> ()>

// CHECK-LABEL: @add_mul_scalar_fusion
func @add_mul_scalar_fusion(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32>
{
  %0 = linalg.init_tensor [] : tensor<f32>
  %1 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = []}
      ins(%arg0, %arg1 : tensor<f32>, tensor<f32>)
      outs(%0 : tensor<f32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):       
      %2 = arith.addf %arg3, %arg4 : f32
      linalg.yield %2 : f32
  } -> tensor<f32>
  // CHECK: linalg.generic {
  // CHECK: arith.addf
  // CHECK: arith.mulf
  %2 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = []}
      ins(%1, %arg2 : tensor<f32>, tensor<f32>)
      outs(%0 : tensor<f32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):       
      %3 = arith.mulf %arg3, %arg4 : f32
      linalg.yield %3 : f32
  } -> tensor<f32>

  return %2 : tensor<f32>
}

// -----

#map0 = affine_map<(d0, d1, d2) -> (d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func @generic_op_constant_fusion(%arg0 : tensor<5x?x?xf32>) -> tensor<5x?x?xf32>
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %cst = arith.constant dense<42.0> : tensor<5xf32>
  %0 = tensor.dim %arg0, %c1 : tensor<5x?x?xf32>
  %1 = tensor.dim %arg0, %c2 : tensor<5x?x?xf32>
  %2 = linalg.init_tensor [5, %0, %1] : tensor<5x?x?xf32>
  %3 = linalg.generic {
    indexing_maps = [#map0, #map1, #map1],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%cst, %arg0 : tensor<5xf32>, tensor<5x?x?xf32>)
    outs(%2 : tensor<5x?x?xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %4 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %4 : f32
    } -> tensor<5x?x?xf32>
  return %3 : tensor<5x?x?xf32>
}
//   CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: func @generic_op_constant_fusion
//       CHECK:   %[[CST:.*]] = arith.constant {{.*}} : f32
//       CHECK:   linalg.generic
//       CHECK:   ^{{.+}}(%[[ARG1:[a-zA-Z0-9_]+]]: f32, %{{.+}}: f32):
//       CHECK:     arith.mulf %[[CST]], %[[ARG1]]

// -----

#map0 = affine_map<(d0, d1, d2) -> ()>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func @generic_op_zero_dim_constant_fusion(%arg0 : tensor<5x?x?xf32>)
  -> tensor<5x?x?xf32>
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %cst = arith.constant dense<42.0> : tensor<f32>
  %0 = tensor.dim %arg0, %c1 : tensor<5x?x?xf32>
  %1 = tensor.dim %arg0, %c2 : tensor<5x?x?xf32>
  %2 = linalg.init_tensor [5, %0, %1] : tensor<5x?x?xf32>
  %3 = linalg.generic {
    indexing_maps = [#map0, #map1, #map1],
    iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%cst, %arg0 : tensor<f32>, tensor<5x?x?xf32>)
    outs(%2 : tensor<5x?x?xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %4 = arith.mulf %arg1, %arg2 : f32
      linalg.yield %4 : f32
    } -> tensor<5x?x?xf32>
  return %3 : tensor<5x?x?xf32>
}
//   CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: func @generic_op_zero_dim_constant_fusion
//       CHECK:   %[[CST:.*]] = arith.constant {{.*}} : f32
//       CHECK:   linalg.generic
//       CHECK:   ^{{.*}}(%[[ARG1:[a-zA-Z0-9_]*]]: f32, %{{.*}}: f32)
//       CHECK:     arith.mulf %[[CST]], %[[ARG1]]

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
func @producer_indexed_consumer_fusion(%arg0: tensor<?x?xi32>,
                                       %arg1: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xi32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xi32>
  %2 = linalg.init_tensor [%0, %1] : tensor<?x?xi32>
  %3 = linalg.generic {
    indexing_maps = [#map0, #map0, #map0],
    iterator_types = ["parallel", "parallel"] }
    ins(%arg0, %arg1  : tensor<?x?xi32>, tensor<?x?xi32>)
    outs(%2 : tensor<?x?xi32>) {
    ^bb0(%arg2: i32, %arg3: i32, %arg4: i32):       
      %10 = arith.addi %arg2, %arg3 : i32
      linalg.yield %10 : i32
    } -> tensor<?x?xi32>
  %4 = linalg.generic {
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel", "parallel"] }
    ins(%3 : tensor<?x?xi32>)
    outs(%2 : tensor<?x?xi32>) {
    ^bb0(%arg2: i32, %arg3: i32):       
      %idx0 = linalg.index 0 : index
      %idx1 = linalg.index 1 : index
      %5 = arith.index_cast %idx0 : index to i32
      %6 = arith.index_cast %idx1 : index to i32
      %7 = arith.addi %arg2, %5 : i32
      %8 = arith.subi %7, %6 : i32
      linalg.yield %8 : i32
    } -> tensor<?x?xi32>
  return %4 : tensor<?x?xi32>
}
//   CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @producer_indexed_consumer_fusion
//      CHECK: linalg.generic
// CHECK-SAME:    indexing_maps = [#[[$MAP0]], #[[$MAP0]], #[[$MAP0]]]
//      CHECK: ^{{[a-zA-Z0-9_]*}}
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: i32
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: i32
//      CHECK:   %[[VAL1:.+]] = arith.addi %[[ARG0]], %[[ARG1]] : i32
//      CHECK:   %[[IDX0:.+]] = linalg.index 0 : index
//      CHECK:   %[[IDX1:.+]] = linalg.index 1 : index
//      CHECK:   %[[ADD_OPERAND:.+]] = arith.index_cast %[[IDX0]] : index to i32
//      CHECK:   %[[SUB_OPERAND:.+]] = arith.index_cast %[[IDX1]] : index to i32
//      CHECK:   %[[VAL2:.+]] = arith.addi %[[VAL1]], %[[ADD_OPERAND]] : i32
//      CHECK:   %[[VAL3:.+]] = arith.subi %[[VAL2]], %[[SUB_OPERAND]] : i32
//      CHECK:   linalg.yield %[[VAL3]] : i32
//  CHECK-NOT: linalg.generic

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
func @indexed_producer_consumer_fusion(%arg0: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xi32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xi32>
  %2 = linalg.init_tensor [%0, %1] : tensor<?x?xi32>
  %3 = linalg.generic {
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel", "parallel"] }
    ins(%arg0 : tensor<?x?xi32>)
    outs(%2 : tensor<?x?xi32>) {
    ^bb0(%arg4: i32, %arg5: i32):       
      %idx0 = linalg.index 0 : index
      %idx1 = linalg.index 1 : index
      %4 = arith.index_cast %idx0 : index to i32
      %5 = arith.index_cast %idx1 : index to i32
      %6 = arith.addi %arg4, %4 : i32
      %7 = arith.subi %6, %5 : i32
      linalg.yield %7 : i32
    } -> tensor<?x?xi32>
  %4 = linalg.generic {
    indexing_maps = [#map0, #map0, #map0],
    iterator_types = ["parallel", "parallel"] }
    ins(%3, %arg0 : tensor<?x?xi32>, tensor<?x?xi32>)
    outs(%2 : tensor<?x?xi32>) {
    ^bb0(%arg2: i32, %arg3: i32, %arg4: i32):       
      %10 = arith.addi %arg2, %arg3 : i32
      linalg.yield %10 : i32
    } -> tensor<?x?xi32>
  return %4 : tensor<?x?xi32>
}
//   CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @indexed_producer_consumer_fusion
//       CHECK: linalg.generic
// CHECK-SAME:    indexing_maps = [#[[$MAP0]], #[[$MAP0]]]
//      CHECK: ^{{[a-zA-Z0-9_]*}}
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: i32
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: i32
//      CHECK:   %[[IDX0:.+]] = linalg.index 0 : index
//      CHECK:   %[[IDX1:.+]] = linalg.index 1 : index
//      CHECK:   %[[ADD_OPERAND:.+]] = arith.index_cast %[[IDX0]] : index to i32
//      CHECK:   %[[SUB_OPERAND:.+]] = arith.index_cast %[[IDX1]] : index to i32
//      CHECK:   %[[VAL1:.+]] = arith.addi %[[ARG0]], %[[ADD_OPERAND]] : i32
//      CHECK:   %[[VAL2:.+]] = arith.subi %[[VAL1]], %[[SUB_OPERAND]] : i32
//      CHECK:   %[[VAL3:.+]] = arith.addi %[[VAL2]], %[[ARG0]] : i32
//      CHECK:   linalg.yield %[[VAL3]] : i32
//   CHECK-NOT: linalg.generic

// -----

// The indices of the first generic op are swapped after fusion.
#map0 = affine_map<(d0, d1) -> (d1, d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
func @indexed_producer_indexed_consumer_fusion(%arg0: tensor<?x?xi32>)
                                               -> tensor<?x?xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xi32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xi32>
  %2 = linalg.init_tensor [%0, %1] : tensor<?x?xi32>
  %3 = linalg.generic {
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel", "parallel"] }
    ins(%arg0 : tensor<?x?xi32>)
    outs(%2 : tensor<?x?xi32>) {
    ^bb0(%arg2: i32, %arg3: i32):       
      %idx0 = linalg.index 0 : index
      %idx1 = linalg.index 1 : index
      %4 = arith.index_cast %idx0 : index to i32
      %5 = arith.index_cast %idx1 : index to i32
      %6 = arith.addi %arg2, %4 : i32
      %7 = arith.subi %5, %6 : i32
      linalg.yield %7 : i32
    } -> tensor<?x?xi32>
  %4= linalg.generic {
    indexing_maps = [#map1, #map1],
    iterator_types = ["parallel", "parallel"] }
    ins(%3 : tensor<?x?xi32>)
    outs(%2 : tensor<?x?xi32>) {
    ^bb0(%arg2: i32, %arg3: i32):       
      %idx0 = linalg.index 0 : index
      %idx1 = linalg.index 1 : index
      %5 = arith.index_cast %idx0 : index to i32
      %6 = arith.index_cast %idx1 : index to i32
      %7 = arith.addi %arg2, %5 : i32
      %8 = arith.subi %7, %6 : i32
      linalg.yield %8 : i32
    } -> tensor<?x?xi32>
  return %4 : tensor<?x?xi32>
}
//   CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @indexed_producer_indexed_consumer_fusion
//       CHECK: linalg.generic
// CHECK-SAME:    indexing_maps = [#[[$MAP0]], #[[$MAP0]]]
//      CHECK: ^{{[a-zA-Z0-9_]*}}
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: i32
//      CHECK:   %[[IDX0:.+]] = linalg.index 0 : index
//      CHECK:   %[[IDX1:.+]] = linalg.index 1 : index
//      CHECK:   %[[ADD_OPERAND1:.+]] = arith.index_cast %[[IDX1]] : index to i32
//      CHECK:   %[[SUB_OPERAND1:.+]] = arith.index_cast %[[IDX0]] : index to i32
//      CHECK:   %[[VAL1:.+]] = arith.addi %[[ARG0]], %[[ADD_OPERAND1]] : i32
//      CHECK:   %[[VAL2:.+]] = arith.subi %[[SUB_OPERAND1]], %[[VAL1]] : i32
//      CHECK:   %[[IDX2:.+]] = linalg.index 0 : index
//      CHECK:   %[[IDX3:.+]] = linalg.index 1 : index
//      CHECK:   %[[ADD_OPERAND2:.+]] = arith.index_cast %[[IDX2]] : index to i32
//      CHECK:   %[[SUB_OPERAND2:.+]] = arith.index_cast %[[IDX3]] : index to i32
//      CHECK:   %[[VAL3:.+]] = arith.addi %[[VAL2]], %[[ADD_OPERAND2]] : i32
//      CHECK:   %[[VAL4:.+]] = arith.subi %[[VAL3]], %[[SUB_OPERAND2]] : i32
//      CHECK:   linalg.yield %[[VAL4]] : i32
//   CHECK-NOT: linalg.generic

// -----

#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d1)>
func @one_dim_indexed_producer_consumer_fusion(%arg0 : tensor<?xi32>,
                                               %arg1 : tensor<?x?xi32>) -> tensor<?x?xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?xi32>
  %0 = linalg.init_tensor [%d0] : tensor<?xi32>
  %1 = linalg.generic
      {indexing_maps = [#map1, #map1],
       iterator_types = ["parallel"]}
      ins(%arg0 : tensor<?xi32>) outs(%0 : tensor<?xi32>) {
      ^bb0(%arg2 : i32, %arg3 : i32):
        %2 = linalg.index 0 : index
        %3 = arith.index_cast %2 : index to i32
        %4 = arith.addi %arg2, %3 : i32
        linalg.yield %4 : i32
      } -> tensor<?xi32>
  %2 = tensor.dim %arg1, %c0 : tensor<?x?xi32>
  %3 = tensor.dim %arg1, %c1 : tensor<?x?xi32>
  %4 = linalg.init_tensor [%2, %3] : tensor<?x?xi32>
  %5 = linalg.generic
      {indexing_maps = [#map2, #map3, #map2],
       iterator_types = ["parallel", "parallel"]}
      ins(%arg1, %1 : tensor<?x?xi32>, tensor<?xi32>)
      outs(%4 : tensor<?x?xi32>) {
      ^bb0(%arg2 : i32, %arg3 : i32, %arg4: i32):
        %6 = arith.addi %arg2, %arg3 : i32
        linalg.yield %6 : i32
     } -> tensor<?x?xi32>
  return %5 : tensor<?x?xi32>
}
//   CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
//   CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (d1)>
// CHECK-LABEL: func @one_dim_indexed_producer_consumer_fusion
//       CHECK: linalg.generic
// CHECK-SAME:    indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP0]]]
//      CHECK: ^{{[a-zA-Z0-9_]*}}
// CHECK-SAME: (%[[ARG0:[a-zA-Z0-9_]*]]: i32, %[[ARG1:[a-zA-Z0-9_]*]]: i32
//      CHECK:   %[[IDX1:.+]] = linalg.index 1 : index
//      CHECK:   %[[VAL1:.+]] = arith.index_cast %[[IDX1]] : index to i32
//      CHECK:   %[[VAL2:.+]] = arith.addi %[[ARG1]], %[[VAL1]] : i32
//      CHECK:   %[[VAL3:.+]] = arith.addi %[[ARG0]], %[[VAL2]] : i32
//      CHECK:   linalg.yield %[[VAL3]] : i32
//   CHECK-NOT: linalg.generic

// -----

func @scalar_generic_fusion
  (%arg0: tensor<5x1x1xf32>, %arg1 : tensor<i32>) -> tensor<10xf32>
{
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<1.000000e+00> : tensor<10xf32>
  %0 = linalg.init_tensor [] : tensor<f32>
  %1 = linalg.generic
    {indexing_maps = [affine_map<() -> ()>, affine_map<() -> ()>],
     iterator_types = []}
    ins(%arg1 : tensor<i32>) outs(%0 : tensor<f32>) {
    ^bb0(%arg2: i32, %arg3: f32):  
      %3 = arith.index_cast %arg2 : i32 to index
      %4 = tensor.extract %arg0[%3, %c0, %c0] : tensor<5x1x1xf32>
      linalg.yield %4 : f32
    } -> tensor<f32>
  %2 = linalg.init_tensor [10] : tensor<10xf32>
  %3 = linalg.generic
   {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>,
                     affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]}
    ins(%1, %cst : tensor<f32>, tensor<10xf32>) outs(%2 : tensor<10xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  
      %4 = arith.mulf %arg2, %arg3 : f32
      linalg.yield %4 : f32
    } -> tensor<10xf32>
  return %3 : tensor<10xf32>
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0) -> ()>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0) -> (d0)>
//       CHECK: func @scalar_generic_fusion
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9]+]]: tensor<5x1x1xf32>
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9]+]]: tensor<i32>
//       CHECK:   %[[T0:.+]] = linalg.generic
//  CHECK-SAME:     indexing_maps = [#[[MAP0]], #[[MAP1]]]
//  CHECK-SAME:     iterator_types = ["parallel"]
//  CHECK-SAME:     ins(%[[ARG1]] : tensor<i32>)
//       CHECK:     tensor.extract %[[ARG0]]
//       CHECK:     linalg.yield
//       CHECK   return %[[T0]]

// -----

func @constant_fusion(%arg0 : tensor<4xf32>) -> (tensor<4xf32>) {
  %cst = arith.constant dense<1.0> : tensor<4xf32>
  %1 = linalg.init_tensor [4] : tensor<4xf32>
  %2 = linalg.generic
    {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>,
                      affine_map<(d0) -> (d0)>],
     iterator_types = ["parallel"]}
    ins (%arg0, %cst : tensor<4xf32>, tensor<4xf32>)
    outs (%1 : tensor<4xf32>) {
    ^bb0(%arg1: f32, %arg2: f32, %arg3: f32):
      %3 = arith.addf %arg1, %arg2 : f32
      linalg.yield %3 : f32
    } -> tensor<4xf32>
  return %2 : tensor<4xf32>
}

//  CHECK-DAG: #[[MAP:.+]] = affine_map<(d0) -> (d0)>
//      CHECK: func @constant_fusion(%[[ARG0:.+]]: tensor<4xf32>)
//  CHECK-DAG:   %[[CST:.+]] = arith.constant 1.000000e+00 : f32
//  CHECK-DAG:   %[[T0:.+]] = linalg.init_tensor [4] : tensor<4xf32>
//      CHECK:   %[[T1:.+]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[MAP]], #[[MAP]]]
// CHECK-SAME:     ins(%[[ARG0]] : tensor<4xf32>)
// CHECK-SAME:     outs(%[[T0]] : tensor<4xf32>)
//      CHECK:   ^{{.+}}(
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9_]+]]: f32, %[[ARG2:[a-zA-Z0-9_]+]]: f32)
//      CHECK:     %[[T2:.+]] = arith.addf %[[ARG1]], %[[CST]]
//      CHECK:     linalg.yield %[[T2]]
//      CHECK:   return %[[T1]]

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (0, d0)>
#map2 = affine_map<(d0) -> (0)>
func @consumer_with_reduction(%arg0: tensor<1x10xf32>,
                              %arg1: tensor<1x10xf32>,
                              %arg2: tensor<1xf32>) -> tensor<1xf32> {
  %init = linalg.init_tensor [1, 10] : tensor<1x10xf32>
  %0 = linalg.generic
    {indexing_maps = [#map0, #map0, #map0],
     iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg1 : tensor<1x10xf32>, tensor<1x10xf32>)
    outs(%init : tensor<1x10xf32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  
    %2 = arith.addf %arg3, %arg4 : f32
    linalg.yield %2 : f32
  } -> tensor<1x10xf32>
  %1 = linalg.generic
    {indexing_maps = [#map1, #map2],
     iterator_types = ["reduction"]}
    ins(%0 : tensor<1x10xf32>)
    outs(%arg2 : tensor<1xf32>)  {
  ^bb0(%arg3: f32, %arg4: f32):  
    %2 = arith.addf %arg3, %arg4 : f32
    linalg.yield %2 : f32
  } -> tensor<1xf32>
  return %1 : tensor<1xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0) -> (0, d0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0) -> (0)>
//      CHECK: func @consumer_with_reduction(%[[ARG0:.+]]: tensor<1x10xf32>, %[[ARG1:.+]]: tensor<1x10xf32>, %[[ARG2:.+]]: tensor<1xf32>)
//      CHECK:   %[[RES:.+]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[MAP0]], #[[MAP0]], #[[MAP1]]]
// CHECK-SAME:     iterator_types = ["reduction"]
// CHECK-SAME:     ins(%[[ARG0]], %[[ARG1]] : tensor<1x10xf32>, tensor<1x10xf32>)
//      CHECK:   ^{{.+}}(%[[T0:.+]]: f32, %[[T1:.+]]: f32, %[[T2:.+]]: f32)
//      CHECK:     %[[T3:.+]] = arith.addf %[[T0]], %[[T1]] : f32
//      CHECK:     %[[T4:.+]] = arith.addf %[[T3]], %[[T2]] : f32
//      CHECK:     linalg.yield %[[T4]]
//      CHECK:   return %[[RES]]

// -----

// CHECK-LABEL: func @sigmoid_dynamic_dim(
//       CHECK:   %[[RES:.*]] = linalg.generic
//   CHECK-NOT:   linalg.generic
//       CHECK:   return %[[RES]]
func @sigmoid_dynamic_dim(%0: tensor<?x1xf32>) -> tensor<?x1xf32> {
  %cp5 = arith.constant 5.000000e-01 : f32
  %c0 = arith.constant 0 : index
  %shape = shape.shape_of %0 : tensor<?x1xf32> -> tensor<?xindex>
  %extend = shape.to_extent_tensor %shape : tensor<?xindex> -> tensor<2xindex>
  %extracted = tensor.extract %extend[%c0] : tensor<2xindex>
  %init0 = linalg.init_tensor [%extracted, 1] : tensor<?x1xf32>
  %1 = linalg.generic {indexing_maps = [
    affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  }
     outs(%init0 : tensor<?x1xf32>) {
    ^bb0(%a: f32):  
      linalg.yield %cp5 : f32
  } -> tensor<?x1xf32>
  %d0 = tensor.dim %0, %c0 : tensor<?x1xf32>
  %init1 = linalg.init_tensor [%d0, 1] : tensor<?x1xf32>
  %2 = linalg.generic {indexing_maps = [
    affine_map<(d0, d1) -> (d0, d1)>,
    affine_map<(d0, d1) -> (d0, d1)>,
    affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  }
      ins(%0, %1 : tensor<?x1xf32>, tensor<?x1xf32>)
     outs(%init1 : tensor<?x1xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):  
      %m = arith.mulf %a, %b : f32
      linalg.yield %m : f32
  } -> tensor<?x1xf32>
  return %2 : tensor<?x1xf32>
}

// -----

func private @compute1(%a: f64) -> f64
func private @compute2(%a: f64, %b: i32) -> i32

// CHECK-LABEL: func @generic_index_op2(
func @generic_index_op2(%arg0: tensor<1x8xf64>, %arg1: tensor<1x8xi32>) -> tensor<1x8xi32> {
  %0 = linalg.generic {
    indexing_maps = [affine_map<(i, j) -> (i, j)>],
    iterator_types = ["parallel", "parallel"]}
  outs(%arg0 : tensor<1x8xf64>) {
  ^bb0(%a: f64):
    %r = call @compute1(%a) : (f64) -> f64
    linalg.yield %r : f64
  } -> tensor<1x8xf64>

  // CHECK-NEXT:   %[[R:.*]] = linalg.generic
  //      CHECK:     bb0(%[[BBA:[0-9a-z]*]]: f64, %[[BBB:[0-9a-z]*]]: i32):
  // CHECK-NEXT:       %[[A:.*]] = call @compute1(%[[BBA]]) : (f64) -> f64
  // CHECK-NEXT:       %[[B:.*]] = call @compute2(%[[A]], %[[BBB]]) : (f64, i32) -> i32
  // CHECK-NEXT:       linalg.yield %[[B]] : i32
  // CHECK-NEXT:   } -> tensor<1x8xi32>
  %1 = linalg.generic {
    indexing_maps = [affine_map<(i, j) -> (i, j)>, affine_map<(i, j) -> (i, j)>],
    iterator_types = ["parallel", "parallel"]}
  ins(%0 : tensor<1x8xf64>)
  outs(%arg1 : tensor<1x8xi32>) {
  ^bb0(%a: f64, %b: i32):
    %r = call @compute2(%a, %b) : (f64, i32) -> i32
    linalg.yield %r : i32
  } -> tensor<1x8xi32>

  // CHECK-NEXT:   return %[[R]] : tensor<1x8xi32>
  return %1 : tensor<1x8xi32>
}

// -----

// CHECK-LABEL: func @no_fuse_constant_with_reduction
func @no_fuse_constant_with_reduction() -> tensor<3xf32>
{
  //      CHECK: %[[CONST:.+]] = arith.constant {{.+}} : tensor<3x2xf32>
  //      CHECK: %[[RESULT:.+]] = linalg.generic
  // CHECK-SAME:   ins(%[[CONST]] : tensor<3x2xf32>)
  //      CHECK: return %[[RESULT]]
  %three = arith.constant dense<3.0> : tensor<3x2xf32>
  %init = linalg.init_tensor [3] : tensor<3xf32>
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]}
     ins(%three : tensor<3x2xf32>) outs(%init : tensor<3xf32>) {
     ^bb0(%arg0 : f32, %arg1 : f32):
        %0 = arith.addf %arg0, %arg1 : f32
        linalg.yield %0 : f32
  } -> tensor<3xf32>
  return %result : tensor<3xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#trait = {
  indexing_maps = [#map, #map],
  iterator_types = ["parallel", "parallel"]
}
func @break_outs_dependency(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32>
{
  %0 = linalg.generic #trait ins(%arg0 : tensor<?x?xf32>) outs(%arg0 : tensor<?x?xf32>) {
       ^bb0(%arg1 : f32, %arg2 : f32) :
         %1 = arith.addf %arg1, %arg1 : f32
         linalg.yield %1 : f32
       } -> tensor<?x?xf32>
  %2 = linalg.generic #trait ins(%0 : tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) {
       ^bb0(%arg1 : f32, %arg2 : f32) :
         %3 = arith.mulf %arg1, %arg1 : f32
         linalg.yield %3 : f32
       } -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}
//      CHECK: func @break_outs_dependency(
// CHECK-SAME:   %[[ARG0:.+]]: tensor<?x?xf32>)
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:   %[[INIT:.+]] = linalg.init_tensor [%[[D0]], %[[D1]]]
//      CHECK:   %[[GENERIC1:.+]] = linalg.generic
// CHECK-SAME:     outs(%[[INIT]] : tensor<?x?xf32>)
//  CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[GENERIC1]], %[[C0]]
//  CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[GENERIC1]], %[[C1]]
//  CHECK-DAG:   %[[INIT:.+]] = linalg.init_tensor [%[[D0]], %[[D1]]]
//      CHECK:   %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:     outs(%[[INIT]] : tensor<?x?xf32>)

// -----

func @fuse_scalar_constant(%arg0 : tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xi32>) {
  %cst = arith.constant 4.0 : f32
  %c42 = arith.constant 42 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %0 = linalg.init_tensor[%d0, %d1] : tensor<?x?xf32>
  %1 = linalg.init_tensor[%d0, %d1] : tensor<?x?xi32>
  %2:2 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> ()>,
                       affine_map<(d0, d1) -> ()>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0, %cst, %c42 : tensor<?x?xf32>, f32, i32)
      outs(%0, %1 : tensor<?x?xf32>, tensor<?x?xi32>) {
      ^bb0(%arg1 : f32, %arg2 : f32, %arg3 : i32, %arg4 : f32, %arg5 : i32) :
        %3 = arith.addf %arg1, %arg2 : f32
        linalg.yield %3, %arg3 : f32, i32
      } -> (tensor<?x?xf32>, tensor<?x?xi32>)
  return %2#0, %2#1 : tensor<?x?xf32>, tensor<?x?xi32>
}
// CHECK-LABEL: func @fuse_scalar_constant
//   CHECK-DAG:   %[[CST:.+]] = arith.constant 4.000000e+00 : f32
//   CHECK-DAG:   %[[C42:.+]] = arith.constant 42 : i32
//       CHECK:   linalg.generic
//  CHECK-SAME:       ins(%{{.+}} : tensor<?x?xf32>)
//       CHECK:     %[[YIELD:.+]] = arith.addf %{{.+}}, %[[CST]] : f32
//       CHECK:     linalg.yield %[[YIELD]], %[[C42]] : f32, i32

// -----

// CHECK-LABEL: @transpose_fold_2d_fp32
func @transpose_fold_2d_fp32(%init: tensor<3x2xf32>) -> tensor<3x2xf32> {
  %input = arith.constant dense<[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]> : tensor<2x3xf32>
  //               CHECK: %[[CST:.+]] = arith.constant
  // CHECK-SAME{LITERAL}:   dense<[[0.000000e+00, 3.000000e+00], [1.000000e+00, 4.000000e+00], [2.000000e+00, 5.000000e+00]]> : tensor<3x2xf32>
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%input : tensor<2x3xf32>) outs(%init : tensor<3x2xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    linalg.yield %arg1 : f32
  } -> tensor<3x2xf32>
  // CHECK: return %[[CST]]
  return %1 : tensor<3x2xf32>
}

// -----

// CHECK-LABEL: @transpose_fold_2d_fp64
func @transpose_fold_2d_fp64(%init: tensor<3x2xf64>) -> tensor<3x2xf64> {
  %input = arith.constant dense<[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]> : tensor<2x3xf64>
  //               CHECK: %[[CST:.+]] = arith.constant
  // CHECK-SAME{LITERAL}:   dense<[[0.000000e+00, 3.000000e+00], [1.000000e+00, 4.000000e+00], [2.000000e+00, 5.000000e+00]]> : tensor<3x2xf64>
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%input : tensor<2x3xf64>) outs(%init : tensor<3x2xf64>) {
  ^bb0(%arg1: f64, %arg2: f64):
    linalg.yield %arg1 : f64
  } -> tensor<3x2xf64>
  // CHECK: return %[[CST]]
  return %1 : tensor<3x2xf64>
}

// -----

// CHECK-LABEL: @transpose_fold_4d_i32
func @transpose_fold_4d_i32(%init: tensor<3x1x4x2xi32>) -> tensor<3x1x4x2xi32> {
  %input = arith.constant dense<[[
    [[ 0,  1,  2,  3], [ 4,  5,  6,  7], [ 8,  9, 10, 11]],
    [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]
  ]]> : tensor<1x2x3x4xi32>
  //               CHECK: %[[CST:.+]] = arith.constant dense<[
  // CHECK-SAME{LITERAL}:   [[[0, 12], [1, 13], [2, 14], [3, 15]]],
  // CHECK-SAME{LITERAL}:   [[[4, 16], [5, 17], [6, 18], [7, 19]]],
  // CHECK-SAME{LITERAL}:   [[[8, 20], [9, 21], [10, 22], [11, 23]]]
  // CHECK-SAME{LITERAL}: ]>
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d0, d3, d1)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%input : tensor<1x2x3x4xi32>) outs(%init : tensor<3x1x4x2xi32>) {
  ^bb0(%arg1: i32, %arg2: i32):
    linalg.yield %arg1 : i32
  } -> tensor<3x1x4x2xi32>
  // CHECK: return %[[CST]]
  return %1 : tensor<3x1x4x2xi32>
}

// -----

// CHECK-LABEL: @transpose_fold_4d_i16
func @transpose_fold_4d_i16(%init: tensor<3x1x4x2xi16>) -> tensor<3x1x4x2xi16> {
  %input = arith.constant dense<[[
    [[ 0,  1,  2,  3], [ 4,  5,  6,  7], [ 8,  9, 10, 11]],
    [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]
  ]]> : tensor<1x2x3x4xi16>
  //               CHECK: %[[CST:.+]] = arith.constant dense<[
  // CHECK-SAME{LITERAL}:   [[[0, 12], [1, 13], [2, 14], [3, 15]]],
  // CHECK-SAME{LITERAL}:   [[[4, 16], [5, 17], [6, 18], [7, 19]]],
  // CHECK-SAME{LITERAL}:   [[[8, 20], [9, 21], [10, 22], [11, 23]]]
  // CHECK-SAME{LITERAL}: ]>
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d0, d3, d1)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%input : tensor<1x2x3x4xi16>) outs(%init : tensor<3x1x4x2xi16>) {
  ^bb0(%arg1: i16, %arg2: i16):
    linalg.yield %arg1 : i16
  } -> tensor<3x1x4x2xi16>
  // CHECK: return %[[CST]]
  return %1 : tensor<3x1x4x2xi16>
}

// -----

// CHECK-LABEL: @transpose_nofold_non_cst_input
func @transpose_nofold_non_cst_input(%input: tensor<2x3xf32>, %init: tensor<3x2xf32>) -> tensor<3x2xf32> {
  // CHECK: linalg.generic
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%input : tensor<2x3xf32>) outs(%init : tensor<3x2xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    linalg.yield %arg1 : f32
  } -> tensor<3x2xf32>
  return %1 : tensor<3x2xf32>
}

// -----

// CHECK-LABEL: @transpose_nofold_yield_const
func @transpose_nofold_yield_const(%init: tensor<3x2xf32>) -> tensor<3x2xf32> {
  %input = arith.constant dense<[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]> : tensor<2x3xf32>
  %cst = arith.constant 8.0 : f32
  // CHECK: linalg.generic
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%input : tensor<2x3xf32>) outs(%init : tensor<3x2xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    linalg.yield %cst : f32
  } -> tensor<3x2xf32>
  return %1 : tensor<3x2xf32>
}

// -----

// CHECK-LABEL: @transpose_nofold_multi_ops_in_region
func @transpose_nofold_multi_ops_in_region(%init: tensor<3x2xf32>) -> tensor<3x2xf32> {
  %input = arith.constant dense<[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]> : tensor<2x3xf32>
  // CHECK: linalg.generic
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%input : tensor<2x3xf32>) outs(%init : tensor<3x2xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    %add = arith.addf %arg1, %arg1 : f32
    linalg.yield %add : f32
  } -> tensor<3x2xf32>
  return %1 : tensor<3x2xf32>
}

// -----

// Fusing the broadcast into a reduction would require to insert extra knowledge
// about the size of the reduction dimension. As long, as this is not
// implemented, we check that two linalg operations remain.
// TODO: Support this case in element-wise fusion.

#map0 = affine_map<(d0, d1) -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d1, d0)>
#map3 = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: @no_fusion_missing_reduction_shape
// CHECK: linalg.generic
// CHECK: linalg.generic
func @no_fusion_missing_reduction_shape(%arg0: tensor<f32>, %arg1: index) -> tensor<?xf32> {
  %cst = arith.constant 0xFF800000 : f32
  %4 = linalg.init_tensor [%arg1, %arg1] : tensor<?x?xf32>
  %5 = linalg.generic {
    indexing_maps = [#map0, #map1],
    iterator_types = ["parallel", "parallel"]
  } ins(%arg0 : tensor<f32>) outs(%4 : tensor<?x?xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):  
    linalg.yield %arg2 : f32
  } -> tensor<?x?xf32>
  %6 = linalg.init_tensor [%arg1] : tensor<?xf32>
  %7 = linalg.fill(%cst, %6) : f32, tensor<?xf32> -> tensor<?xf32>
  %8 = linalg.generic {
    indexing_maps = [#map2, #map3],
    iterator_types = ["parallel", "reduction"]
  } ins(%5 : tensor<?x?xf32>) outs(%7 : tensor<?xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):  
    %9 = arith.maxf %arg2, %arg3 : f32
    linalg.yield %9 : f32
  } -> tensor<?xf32>
  return %8 : tensor<?xf32>
}

// -----

func @illegal_fusion(%arg0 : tensor<5000xi64>, %arg1 : tensor<5000xi32>) -> tensor<5000xi32> {
  %c1_i32 = arith.constant 1 : i32
  %0 = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]}
        outs(%arg0 : tensor<5000xi64>) {
        ^bb0(%arg3: i64):  // no predecessors
          %22 = linalg.index 0 : index
          %23 = arith.index_cast %22 : index to i64
          linalg.yield %23 : i64
        } -> tensor<5000xi64>
  %1 = linalg.init_tensor [5000] : tensor<5000xi32>
  %2 = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d1)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%0 : tensor<5000xi64>) outs(%1 : tensor<5000xi32>) {
        ^bb0(%arg3: i64, %arg5: i32):  // no predecessors
          %22 = arith.index_cast %arg3 : i64 to index
          %23 = tensor.extract %arg1[%22] : tensor<5000xi32>
          linalg.yield %23 : i32
        } -> tensor<5000xi32>
  return %2 : tensor<5000xi32>
}
// CHECK-LABEL: func @illegal_fusion(
//       CHECK:   %[[PRODUCER:.+]] = linalg.generic
//       CHECK:    linalg.generic
//   CHECK-SAME:       ins(%[[PRODUCER]]
