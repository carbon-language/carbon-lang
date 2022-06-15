// RUN: mlir-opt %s -test-linalg-codegen-strategy="anchor-op=linalg.matmul pad padding-values=0.:f32,0.:f32,0.:f32 padding-dimensions=0,1,2 pack-paddings=1,1,0 run-enable-pass=false" -cse -split-input-file | FileCheck %s --check-prefix=MATMUL
// RUN: mlir-opt %s -test-linalg-codegen-strategy="anchor-op=linalg.fill pad padding-values=0.:f32,1.:f32 pack-paddings=0,1 padding-dimensions=0,1,2 run-enable-pass=false" -cse -split-input-file | FileCheck %s --check-prefix=FILL
// RUN: mlir-opt %s -test-linalg-codegen-strategy="anchor-op=linalg.fill pad padding-values=0.:f32,0.:f32 pack-paddings=0,1 padding-dimensions=0,1,2 run-enable-pass=false" -test-linalg-codegen-strategy="anchor-op=linalg.matmul pad padding-values=0.:f32,0.:f32,0.:f32 padding-dimensions=0,1,2 pack-paddings=0,1 run-enable-pass=false" -cse -split-input-file | FileCheck %s --check-prefix=FILL-MATMUL
// RUN: mlir-opt %s -test-linalg-codegen-strategy="anchor-op=linalg.matmul pad padding-values=0.:f32,0.:f32 pack-paddings=1,1,0 padding-dimensions=0,1,2 run-enable-pass=false" -cse -split-input-file | FileCheck %s --check-prefix=INPUTS-ONLY
// RUN: mlir-opt %s -test-linalg-codegen-strategy="anchor-op=linalg.matmul pad padding-values=0.:f32,0.:f32,0.:f32 padding-dimensions=0,1 pack-paddings=1,1,1 run-enable-pass=false" -cse -split-input-file | FileCheck %s --check-prefix=PARTIAL
// RUN: mlir-opt %s -test-linalg-codegen-strategy="anchor-op=linalg.depthwise_conv_2d_nhwc_hwc pad padding-values=0.:f32,0.:f32,0.:f32 padding-dimensions=1,2 pack-paddings=1,0,1 run-enable-pass=false" -cse -split-input-file | FileCheck %s --check-prefix=DEPTHWISE_CONV_2D

// MATMUL-DAG: #[[MAP0:[0-9a-z]+]] = affine_map<()[s0] -> (-s0 + 12, 7)>
// MATMUL-DAG: #[[MAP1:[0-9a-z]+]] = affine_map<()[s0] -> (-s0 + 7)>
#map = affine_map<()[s0] -> (-s0 + 12, 7)>

//      MATMUL:  static_sizes_output_divisible
// MATMUL-SAME:    %[[ARG0:[0-9a-zA-Z]*]]: tensor<24x12xf32>
// MATMUL-SAME:    %[[ARG1:[0-9a-zA-Z]*]]: tensor<12x25xf32>
// MATMUL-SAME:    %[[ARG2:[0-9a-zA-Z]*]]: tensor<24x25xf32>
// MATMUL-SAME:    %[[IV0:[0-9a-zA-Z]*]]: index
// MATMUL-SAME:    %[[IV1:[0-9a-zA-Z]*]]: index
// MATMUL-SAME:    %[[IV2:[0-9a-zA-Z]*]]: index
func.func @static_sizes_output_divisible(%arg0: tensor<24x12xf32>,
                                         %arg1: tensor<12x25xf32>,
                                         %arg2: tensor<24x25xf32>,
                                         %iv0 : index, %iv1 : index, %iv2 : index) -> tensor<24x25xf32> {
  //  MATMUL-DAG: %[[CST:.*]] = arith.constant 0.
  //  MATMUL-DAG: %[[C0:.*]] = arith.constant 0 : index

  //      MATMUL:   %[[TS2:.*]] = affine.min #[[MAP0]]()[%[[IV2]]]
  %0 = affine.min #map()[%iv2]

  //      MATMUL:   %[[T0:.*]] = tensor.extract_slice %[[ARG0]]
  //      MATMUL:   %[[T1:.*]] = tensor.extract_slice %[[ARG1]]
  //      MATMUL:   %[[T2:.*]] = tensor.extract_slice %[[ARG2]]
  %1 = tensor.extract_slice %arg0[%iv0, %iv2] [4, %0] [1, 1] : tensor<24x12xf32> to tensor<4x?xf32>
  %2 = tensor.extract_slice %arg1[%iv2, %iv1] [%0, 5] [1, 1] : tensor<12x25xf32> to tensor<?x5xf32>
  %3 = tensor.extract_slice %arg2[%iv0, %iv1] [4, 5] [1, 1] : tensor<24x25xf32> to tensor<4x5xf32>

  // Check statically sized matmul inputs with partially divisible sizes are padded.
  //      MATMUL:   %[[V0:.*]] = affine.apply #[[MAP1]]()[%[[TS2]]]
  //      MATMUL:   %[[T3:.*]] = tensor.pad %[[T0]] nofold
  // MATMUL-SAME:                  [%[[C0]], %[[C0]]]
  // MATMUL-SAME:                  [%[[C0]], %[[V0]]
  //      MATMUL:   tensor.yield %[[CST]]
  //      MATMUL:   %[[T4:.*]] = tensor.pad %[[T1]] nofold

  // Check the statically sized matmul output with fully divisible sizes is not padded.
  //      MATMUL:   %[[T5:.*]] = linalg.matmul
  // MATMUL-SAME:                  ins(%[[T3]], %[[T4]] : tensor<4x7xf32>, tensor<7x5xf32>)
  // MATMUL-SAME:                  outs(%[[T2]] : tensor<4x5xf32>)
  //      MATMUL:   %[[T6:.*]] = tensor.insert_slice %[[T5]]
  %4 = linalg.matmul ins(%1, %2 : tensor<4x?xf32>, tensor<?x5xf32>) outs(%3 : tensor<4x5xf32>) -> tensor<4x5xf32>
  %5 = tensor.insert_slice %4 into %arg2[%iv0, %iv1] [4, 5] [1, 1] : tensor<4x5xf32> into tensor<24x25xf32>
  func.return %5 : tensor<24x25xf32>
}

// -----

// MATMUL-DAG: #[[MAP0:[0-9a-z]+]] = affine_map<()[s0] -> (-s0 + 25, 7)>
// MATMUL-DAG: #[[MAP1:[0-9a-z]+]] = affine_map<()[s0] -> (-s0 + 7)>
#map = affine_map<()[s0] -> (-s0 + 25, 7)>

//      MATMUL:  static_sizes_input_divisible
// MATMUL-SAME:    %[[ARG2:[0-9a-zA-Z]*]]: tensor<24x25xf32>
// MATMUL-SAME:    %[[IV0:[0-9a-zA-Z]*]]: index
// MATMUL-SAME:    %[[IV1:[0-9a-zA-Z]*]]: index
// MATMUL-SAME:    %[[IV2:[0-9a-zA-Z]*]]: index
func.func @static_sizes_input_divisible(%arg0: tensor<24x12xf32>,
                                        %arg1: tensor<12x25xf32>,
                                        %arg2: tensor<24x25xf32>,
                                        %iv0 : index, %iv1 : index, %iv2 : index) ->  tensor<24x25xf32> {
  //  MATMUL-DAG: %[[CST:.*]] = arith.constant 0.
  //  MATMUL-DAG: %[[C0:.*]] = arith.constant 0 : index

  %3 = tensor.extract_slice %arg0[%iv0, %iv2] [4, 6] [1, 1] : tensor<24x12xf32> to tensor<4x6xf32>

  //      MATMUL:   %[[TS1:.*]] = affine.min #[[MAP0]]()[%[[IV1]]]
  %4 = affine.min #map()[%iv1]
  %5 = tensor.extract_slice %arg1[%iv2, %iv1] [6, %4] [1, 1] : tensor<12x25xf32> to tensor<6x?xf32>

  //      MATMUL:   %[[T0:.*]] = tensor.extract_slice %[[ARG2]]
  %6 = tensor.extract_slice %arg2[%iv0, %iv1] [4, %4] [1, 1] : tensor<24x25xf32> to tensor<4x?xf32>

  // Check the statically sized matmul output with partially divisible sizes is padded.
  //      MATMUL:   %[[V0:.*]] = affine.apply #[[MAP1]]()[%[[TS1]]]
  //      MATMUL:   %[[T1:.*]] = tensor.pad %[[T0]] low
  // MATMUL-SAME:                  [%[[C0]], %[[C0]]]
  // MATMUL-SAME:                  [%[[C0]], %[[V0]]
  //      MATMUL:   tensor.yield %[[CST]]

  //      MATMUL:   %[[T2:.*]] = linalg.matmul
  // MATMUL-SAME:                  outs(%[[T1]] : tensor<4x7xf32>)
  //      MATMUL:   %[[T3:.*]] = tensor.extract_slice %[[T2]]
  //      MATMUL:   %[[T4:.*]] = tensor.insert_slice %[[T3]]
  %7 = linalg.matmul ins(%3, %5 : tensor<4x6xf32>, tensor<6x?xf32>) outs(%6 : tensor<4x?xf32>) -> tensor<4x?xf32>
  %8 = tensor.insert_slice %7 into %arg2[%iv0, %iv1] [4, %4] [1, 1] : tensor<4x?xf32> into tensor<24x25xf32>

   //      MATMUL:   return %[[T4]]
  func.return %8 : tensor<24x25xf32>
}

// -----

// MATMUL-DAG: #[[MAP0:[0-9a-z]+]] = affine_map<()[s0, s1] -> (-s0 + s1, 5)>
// MATMUL-DAG: #[[MAP1:[0-9a-z]+]] = affine_map<()[s0, s1] -> (-s0 + s1, 7)>
// MATMUL-DAG: #[[MAP2:[0-9a-z]+]] = affine_map<()[s0, s1] -> (-s0 + s1, 6)>
// MATMUL-DAG: #[[MAP3:[0-9a-z]+]] = affine_map<()[s0] -> (-s0 + 5)>
// MATMUL-DAG: #[[MAP4:[0-9a-z]+]] = affine_map<()[s0] -> (-s0 + 6)>

#map0 = affine_map<()[s0, s1] -> (-s0 + s1, 5)>
#map1 = affine_map<()[s0, s1] -> (-s0 + s1, 6)>
#map2 = affine_map<()[s0, s1] -> (-s0 + s1, 7)>

//      MATMUL:  dynamic_sizes
// MATMUL-SAME:    %[[ARG0:[0-9a-zA-Z]*]]: tensor<?x?xf32>
// MATMUL-SAME:    %[[ARG1:[0-9a-zA-Z]*]]: tensor<?x?xf32>
// MATMUL-SAME:    %[[ARG2:[0-9a-zA-Z]*]]: tensor<?x?xf32>
// MATMUL-SAME:    %[[IV0:[0-9a-zA-Z]*]]: index
// MATMUL-SAME:    %[[IV1:[0-9a-zA-Z]*]]: index
// MATMUL-SAME:    %[[IV2:[0-9a-zA-Z]*]]: index
func.func @dynamic_sizes(%arg0: tensor<?x?xf32>,
                         %arg1: tensor<?x?xf32>,
                         %arg2: tensor<?x?xf32>,
                         %iv0 : index, %iv1 : index, %iv2 : index) -> tensor<?x?xf32> {
  //  MATMUL-DAG: %[[C0:.*]] = arith.constant 0 : index
  //  MATMUL-DAG: %[[C1:.*]] = arith.constant 1
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index

  //  MATMUL-DAG: %[[D0:.*]] = tensor.dim %[[ARG0]], %[[C0]]
  //  MATMUL-DAG: %[[D2:.*]] = tensor.dim %[[ARG0]], %[[C1]]
  //  MATMUL-DAG: %[[D1:.*]] = tensor.dim %[[ARG1]], %[[C1]]
  %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %2 = tensor.dim %arg1, %c1 : tensor<?x?xf32>

  //      MATMUL:   %[[TS0:.*]] = affine.min #[[MAP0]]()[%[[IV0]], %[[D0]]]
  //      MATMUL:   %[[TS2:.*]] = affine.min #[[MAP2]]()[%[[IV2]], %[[D2]]]
  //      MATMUL:   %[[TS1:.*]] = affine.min #[[MAP1]]()[%[[IV1]], %[[D1]]]
  %6 = affine.min #map0()[%iv0, %0]
  %7 = affine.min #map1()[%iv2, %1]
  %8 = tensor.extract_slice %arg0[%iv0, %iv2] [%6, %7] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  %9 = affine.min #map2()[%iv1, %2]
  %10 = tensor.extract_slice %arg1[%iv2, %iv1] [%7, %9] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
  %11 = tensor.extract_slice %arg2[%iv0, %iv1] [%6, %9] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>

  // Check all matmul operands are padded.
  //      MATMUL:   %[[V0:.*]] = affine.apply #[[MAP3]]()[%[[TS0]]]
  //      MATMUL:   %[[V1:.*]] = affine.apply #[[MAP4]]()[%[[TS2]]]
  //      MATMUL:   %[[T3:.*]] = tensor.pad %{{.*}} nofold
  // MATMUL-SAME:                  [%[[C0]], %[[C0]]]
  // MATMUL-SAME:                  [%[[V0]], %[[V1]]
  //      MATMUL:   %[[T4:.*]] = tensor.pad %{{.*}} nofold
  //      MATMUL:   %[[T5:.*]] = tensor.pad %{{.*}} low

  // Check the dynamic matmul has been erased.
  //  MATMUL-NOT:   = linalg.matmul {{.*}} tensor<?x?xf32>

  // Check all padded matmul operands are statically sized.
  //      MATMUL:   %[[T6:.*]] = linalg.matmul
  // MATMUL-SAME:                  ins(%[[T3]], %[[T4]] : tensor<5x6xf32>, tensor<6x7xf32>)
  // MATMUL-SAME:                  outs(%[[T5]] : tensor<5x7xf32>)
  //      MATMUL:   %[[T7:.*]] = tensor.extract_slice %[[T6]][0, 0] [%[[TS0]], %[[TS1]]]
  //      MATMUL:   %[[T8:.*]] = tensor.insert_slice %[[T7]]
  %12 = linalg.matmul ins(%8, %10 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%11 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %13 = tensor.insert_slice %12 into %arg2[%iv0, %iv1] [%6, %9] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>

  //      MATMUL:   return %[[T8]]
  func.return %13 : tensor<?x?xf32>
}

// -----

#map0 = affine_map<()[s0] -> (64, s0)>

//      FILL-MATMUL:  pad_multiple
// FILL-MATMUL-SAME:    %[[ARG0:[0-9a-zA-Z]*]]: tensor<64x64xf32>
func.func @pad_multiple(%arg0: tensor<64x64xf32>,
                        %iv0 : index) -> tensor<?x?xf32> {
  %cst = arith.constant 0.0 : f32
  %size = affine.min #map0()[%iv0]

  //      FILL-MATMUL:  %[[T0:.*]] = tensor.extract_slice
  %0 = tensor.extract_slice %arg0[0, 0] [%size, %size] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>

  // Check the two operations are padded by the same pad tensor operation.
  //      FILL-MATMUL:  %[[T1:.*]] = tensor.pad %[[T0]]
  //      FILL-MATMUL:  %[[T2:.*]] = linalg.fill {{.*}} outs(%[[T1]]
  //      FILL-MATMUL:  %[[T3:.*]] = linalg.matmul {{.*}} outs(%[[T2]]
  //      FILL-MATMUL:  = tensor.extract_slice %[[T3]]
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = linalg.matmul ins(%0, %0 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%1 : tensor<?x?xf32>) -> tensor<?x?xf32>
  func.return %2 : tensor<?x?xf32>
}

// -----

#map0 = affine_map<()[s0] -> (64, s0)>

//      MATMUL:  pad_chain
// MATMUL-SAME:    %[[ARG0:[0-9a-zA-Z]*]]: tensor<64x64xf32>
func.func @pad_chain(%arg0: tensor<64x64xf32>,
                     %iv0 : index) -> tensor<?x?xf32> {
  %cst = arith.constant 0.0 : f32
  %size = affine.min #map0()[%iv0]
  %0 = tensor.extract_slice %arg0[0, 0] [%size, %size] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>

  // Check the matmul at the end of the use-def chain is padded.
  //      MATMUL:  %[[T0:.*]] = linalg.fill
  //      MATMUL:  %[[T1:.*]] = tensor.pad %[[T0]]
  //      MATMUL:  %[[T2:.*]] = linalg.matmul {{.*}} outs(%[[T1]]
  //      MATMUL:  = tensor.extract_slice %[[T2]]
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = linalg.matmul ins(%0, %0 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%1 : tensor<?x?xf32>) -> tensor<?x?xf32>
  func.return %2 : tensor<?x?xf32>
}

// -----

#map0 = affine_map<()[s0] -> (64, s0)>

//      MATMUL:  compose_padding
// MATMUL-SAME:    %[[ARG0:[0-9a-zA-Z]*]]: tensor<64x64xf32>
func.func @compose_padding(%arg0: tensor<64x64xf32>,
                           %iv0 : index) -> tensor<?x?xf32> {
  %cst = arith.constant 0.0 : f32

  //      MATMUL:  %[[SIZE:.*]] = affine.min
  %size = affine.min #map0()[%iv0]

  //      MATMUL:  %[[T0:.*]] = tensor.extract_slice %[[ARG0]]
  // MATMUL-SAME:                                     [0, 0]
  // MATMUL-SAME:                                     [%[[SIZE]], %[[SIZE]]]
  //      MATMUL:  %[[T1:.*]] = tensor.pad %[[T0]]
  //      MATMUL:  %[[T2:.*]] = linalg.fill ins(%{{.*}}{{.*}}outs(%[[T1]]
  //      MATMUL:  %[[T3:.*]] = linalg.fill ins(%{{.*}}{{.*}}outs(%[[T2]]
  %0 = tensor.extract_slice %arg0[0, 0] [%size, %size] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
  %1 = tensor.pad %0 low[0, 0] high[%iv0, %iv0]  {
    ^bb0(%arg3: index, %arg4: index):
      tensor.yield %cst : f32
  } : tensor<?x?xf32> to tensor<64x64xf32>
  %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %4 = tensor.extract_slice %3[0, 0] [%size, %size] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>

  // Check there are no additional pad tensor operations.
  //  MATMUL-NOT:  tensor.pad

  // Check the matmul directly uses the result of the fill operation.
  //      MATMUL:  %[[T4:.*]] = linalg.matmul ins(%[[T3]]
  //      MATMUL:  %[[T5:.*]] = tensor.extract_slice %[[T4]]
  // MATMUL-SAME:                                     [0, 0]
  // MATMUL-SAME:                                     [%[[SIZE]], %[[SIZE]]]
  %5 = linalg.matmul ins(%4, %4 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%4 : tensor<?x?xf32>) -> tensor<?x?xf32>

  //      MATMUL:  return %[[T5]]
  func.return %5 : tensor<?x?xf32>
}

// -----

#map0 = affine_map<()[s0] -> (64, s0)>

//      MATMUL:  different_padding_values
func.func @different_padding_values(%arg0: tensor<64x64xf32>,
                                    %iv0 : index) -> tensor<?x?xf32> {
  %cst = arith.constant 42.0 : f32
  %size = affine.min #map0()[%iv0]
  %0 = tensor.extract_slice %arg0[0, 0] [%size, %size] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
  %1 = tensor.pad %0 low[0, 0] high[%iv0, %iv0]  {
    ^bb0(%arg3: index, %arg4: index):
      tensor.yield %cst : f32
  } : tensor<?x?xf32> to tensor<64x64xf32>
  %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %4 = tensor.extract_slice %2[0, 0] [%size, %size] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>

  // Different padding values prevent composing the paddings (42.0 vs. 0.0).
  //      MATMUL:  = linalg.fill
  //      MATMUL:  = tensor.pad
  //      MATMUL:  = linalg.matmul
  %5 = linalg.matmul ins(%4, %4 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%4 : tensor<?x?xf32>) -> tensor<?x?xf32>
  func.return %5 : tensor<?x?xf32>
}

// -----

#map0 = affine_map<()[s0] -> (64, s0)>

//      MATMUL:  different_padding_dynamic_sizes
func.func @different_padding_dynamic_sizes(%arg0: tensor<64x64xf32>,
                                           %iv0 : index) -> tensor<?x?xf32> {
  %cst = arith.constant 0.0 : f32
  %size = affine.min #map0()[%iv0]
  %0 = tensor.extract_slice %arg0[0, 0] [%iv0, %iv0] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>
  %1 = tensor.pad %0 low[0, 0] high[%iv0, %iv0]  {
    ^bb0(%arg3: index, %arg4: index):
      tensor.yield %cst : f32
  } : tensor<?x?xf32> to tensor<64x64xf32>
  %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %4 = tensor.extract_slice %2[0, 0] [%size, %size] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>

  // Different dynamic sizes prevent composing the paddings (%iv0 vs %size).
  //      MATMUL:  = linalg.fill
  //      MATMUL:  = tensor.pad
  //      MATMUL:  = linalg.matmul
  %5 = linalg.matmul ins(%4, %4 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%4 : tensor<?x?xf32>) -> tensor<?x?xf32>
  func.return %5 : tensor<?x?xf32>
}

// -----

#map0 = affine_map<()[s0] -> (64, s0)>

//      MATMUL:  different_padding_dynamic_rank
func.func @different_padding_dynamic_rank(%arg0: tensor<64x64x1xf32>,
                                          %iv0 : index) -> tensor<?x?xf32> {
  %cst = arith.constant 0.0 : f32
  %size = affine.min #map0()[%iv0]
  %0 = tensor.extract_slice %arg0[0, 0, 0] [%size, %size, 1] [1, 1, 1] : tensor<64x64x1xf32> to tensor<?x?xf32>
  %1 = tensor.pad %0 low[0, 0] high[%iv0, %iv0]  {
    ^bb0(%arg3: index, %arg4: index):
      tensor.yield %cst : f32
  } : tensor<?x?xf32> to tensor<64x64xf32>
  %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<64x64xf32>) -> tensor<64x64xf32>
  %3 = tensor.extract_slice %2[0, 0] [%size, %size] [1, 1] : tensor<64x64xf32> to tensor<?x?xf32>

  // Different dynamic ranks prevent composing the paddings ([%size, %size, 1] vs [%size, %size]).
  //      MATMUL:  = linalg.fill
  //      MATMUL:  = tensor.pad
  //      MATMUL:  = linalg.matmul
  %4 = linalg.matmul ins(%3, %3 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%3 : tensor<?x?xf32>) -> tensor<?x?xf32>
  func.return %4 : tensor<?x?xf32>
}

// -----

#map0 = affine_map<()[s0] -> (64, s0)>

//      MATMUL:  different_padding_static_sizes
func.func @different_padding_static_sizes(%arg0: tensor<62x62xf32>,
                                          %iv0 : index) -> tensor<?x?xf32> {
  %cst = arith.constant 0.0 : f32
  %size = affine.min #map0()[%iv0]
  %0 = tensor.extract_slice %arg0[0, 0] [%size, %size] [1, 1] : tensor<62x62xf32> to tensor<?x?xf32>
  %1 = tensor.pad %0 low[0, 0] high[%iv0, %iv0]  {
    ^bb0(%arg3: index, %arg4: index):
      tensor.yield %cst : f32
  } : tensor<?x?xf32> to tensor<62x62xf32>
  %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<62x62xf32>) -> tensor<62x62xf32>
  %4 = tensor.extract_slice %2[0, 0] [%size, %size] [1, 1] : tensor<62x62xf32> to tensor<?x?xf32>

  // Different static sizes prevent composing the paddings (62 vs 64 derived from #map0).
  //      MATMUL:  = linalg.fill
  //      MATMUL:  = tensor.pad
  //      MATMUL:  = linalg.matmul
  %5 = linalg.matmul ins(%4, %4 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%4 : tensor<?x?xf32>) -> tensor<?x?xf32>
  func.return %5 : tensor<?x?xf32>
}

// -----

#map0 = affine_map<()[s0] -> (7, s0)>

//      FILL:  scalar_operand
// FILL-SAME:    %[[ARG0:[0-9a-zA-Z]*]]: f32
// FILL-SAME:    %[[ARG1:[0-9a-zA-Z]*]]: tensor<24x12xf32>
func.func @scalar_operand(%arg0: f32,
                          %arg1: tensor<24x12xf32>,
                          %iv0 : index) -> tensor<24x12xf32> {
  %0 = affine.min #map0()[%iv0]

  //      FILL:   %[[T0:.*]] = tensor.extract_slice %[[ARG1]]
  //      FILL:   %[[T1:.*]] = tensor.pad %[[T0]] nofold
  %1 = tensor.extract_slice %arg1[0, 0] [4, %0] [1, 1] : tensor<24x12xf32> to tensor<4x?xf32>

  // Check only the fill output operand is padded.
  //      FILL:   %[[T6:.*]] = linalg.fill ins(%[[ARG0]]{{.*}}outs(%[[T1]]
  %2 = linalg.fill ins(%arg0 : f32) outs(%1 : tensor<4x?xf32>) -> tensor<4x?xf32>
  %3 = tensor.insert_slice %2 into %arg1[0, 0] [4, %0] [1, 1] : tensor<4x?xf32> into tensor<24x12xf32>
  func.return %3 : tensor<24x12xf32>
}

// -----

#map0 = affine_map<()[s0] -> (7, s0)>

//      MATMUL:  static_extract_slice_missing
// MATMUL-SAME:    %[[ARG2:[0-9a-zA-Z]*]]: tensor<4x5xf32>,
func.func @static_extract_slice_missing(%arg0: tensor<24x12xf32>,
                                        %arg1: tensor<12x25xf32>,
                                        %arg2: tensor<4x5xf32>,
                                        %iv0 : index, %iv1 : index, %iv2 : index) -> tensor<4x5xf32> {
  %0 = affine.min #map0()[%iv2]
  %1 = tensor.extract_slice %arg0[%iv0, %iv2] [4, %0] [1, 1] : tensor<24x12xf32> to tensor<4x?xf32>
  %2 = tensor.extract_slice %arg1[%iv2, %iv1] [%0, 5] [1, 1] : tensor<12x25xf32> to tensor<?x5xf32>

  // Check the matmul inputs are padded despite the missing slice for the static output.
  //      MATMUL:  %[[T0:.*]] = tensor.pad
  //      MATMUL:  %[[T1:.*]] = tensor.pad
  //      MATMUL:  = linalg.matmul ins(%[[T0]], %[[T1]]
  // MATMUL-SAME:                 outs(%[[ARG2]]
  %3 = linalg.matmul ins(%1, %2 : tensor<4x?xf32>, tensor<?x5xf32>) outs(%arg2 : tensor<4x5xf32>) -> tensor<4x5xf32>
  func.return %3 : tensor<4x5xf32>
}

// -----

#map0 = affine_map<()[s0] -> (7, s0)>

//      MATMUL:  dynamic_extract_slice_missing
// MATMUL-SAME:    %[[ARG0:[0-9a-zA-Z]*]]: tensor<4x?xf32>,
// MATMUL-SAME:    %[[ARG1:[0-9a-zA-Z]*]]: tensor<12x25xf32>,
// MATMUL-SAME:    %[[ARG2:[0-9a-zA-Z]*]]: tensor<24x25xf32>,
func.func @dynamic_extract_slice_missing(%arg0: tensor<4x?xf32>,
                                         %arg1: tensor<12x25xf32>,
                                         %arg2: tensor<24x25xf32>,
                                         %iv0 : index, %iv1 : index, %iv2 : index) -> tensor<24x25xf32> {
  %0 = affine.min #map0()[%iv2]

  //      MATMUL:  %[[T0:.*]] = tensor.extract_slice %[[ARG1]]
  //      MATMUL:  %[[T1:.*]] = tensor.extract_slice %[[ARG2]]
  %2 = tensor.extract_slice %arg1[%iv2, %iv1] [%0, 5] [1, 1] : tensor<12x25xf32> to tensor<?x5xf32>
  %3 = tensor.extract_slice %arg2[%iv0, %iv1] [4, 5] [1, 1] : tensor<24x25xf32> to tensor<4x5xf32>

  // Check the matmul is not padded due to the missing slice for the dynamic input.
  //      MATMUL:  = linalg.matmul ins(%[[ARG0]], %[[T0]]
  // MATMUL-SAME:                 outs(%[[T1]]
  %4 = linalg.matmul ins(%arg0, %2 : tensor<4x?xf32>, tensor<?x5xf32>) outs(%3 : tensor<4x5xf32>) -> tensor<4x5xf32>
  %5 = tensor.insert_slice %4 into %arg2[%iv0, %iv1] [4, 5] [1, 1] : tensor<4x5xf32> into tensor<24x25xf32>
  func.return %5 : tensor<24x25xf32>
}

// -----

#map0 = affine_map<()[s0] -> (7, s0)>

//      INPUTS-ONLY:  static_input_padding_only
// INPUTS-ONLY-SAME:    %[[ARG2:[0-9a-zA-Z]*]]: tensor<24x25xf32>,
func.func @static_input_padding_only(%arg0: tensor<24x12xf32>,
                                     %arg1: tensor<12x25xf32>,
                                     %arg2: tensor<24x25xf32>,
                                     %iv0 : index, %iv1 : index, %iv2 : index) -> tensor<24x25xf32> {
  %0 = affine.min #map0()[%iv2]
  %1 = tensor.extract_slice %arg0[%iv0, %iv2] [4, %0] [1, 1] : tensor<24x12xf32> to tensor<4x?xf32>
  %2 = tensor.extract_slice %arg1[%iv2, %iv1] [%0, 5] [1, 1] : tensor<12x25xf32> to tensor<?x5xf32>

  // INPUTS-ONLY:  %[[T0:.*]] = tensor.extract_slice %[[ARG2]]
  %3 = tensor.extract_slice %arg2[%iv0, %iv1] [4, 5] [1, 1] : tensor<24x25xf32> to tensor<4x5xf32>

  // Check the matmul inputs are padded despite the failure to compute a padding value for the static output.
  // INPUTS-ONLY:  %[[T1:.*]] = tensor.pad
  // INPUTS-ONLY:  %[[T2:.*]] = tensor.pad
  // INPUTS-ONLY:  = linalg.matmul ins(%[[T1]], %[[T2]]
  // INPUTS-ONLY-SAME:             outs(%[[T0]]
  %4 = linalg.matmul ins(%1, %2 : tensor<4x?xf32>, tensor<?x5xf32>) outs(%3 : tensor<4x5xf32>) -> tensor<4x5xf32>
  %5 = tensor.insert_slice %4 into %arg2[%iv0, %iv1] [4, 5] [1, 1] : tensor<4x5xf32> into tensor<24x25xf32>
  func.return %5 : tensor<24x25xf32>
}

// -----

#map0 = affine_map<()[s0] -> (7, s0)>

//      INPUTS-ONLY:  dynamic_input_padding_only
// INPUTS-ONLY-SAME:    %[[ARG0:[0-9a-zA-Z]*]]: tensor<24x12xf32>,
// INPUTS-ONLY-SAME:    %[[ARG1:[0-9a-zA-Z]*]]: tensor<12x25xf32>,
// INPUTS-ONLY-SAME:    %[[ARG2:[0-9a-zA-Z]*]]: tensor<24x25xf32>,
func.func @dynamic_input_padding_only(%arg0: tensor<24x12xf32>,
                                      %arg1: tensor<12x25xf32>,
                                      %arg2: tensor<24x25xf32>,
                                      %iv0 : index, %iv1 : index, %iv2 : index) -> tensor<24x25xf32> {
  %0 = affine.min #map0()[%iv2]

  // INPUTS-ONLY:  %[[T0:.*]] = tensor.extract_slice %[[ARG0]]
  // INPUTS-ONLY:  %[[T1:.*]] = tensor.extract_slice %[[ARG1]]
  // INPUTS-ONLY:  %[[T2:.*]] = tensor.extract_slice %[[ARG2]]
  %1 = tensor.extract_slice %arg0[%iv0, %iv2] [4, %0] [1, 1] : tensor<24x12xf32> to tensor<4x?xf32>
  %2 = tensor.extract_slice %arg1[%iv2, %iv1] [%0, %0] [1, 1] : tensor<12x25xf32> to tensor<?x?xf32>
  %3 = tensor.extract_slice %arg2[%iv0, %iv1] [4, %0] [1, 1] : tensor<24x25xf32> to tensor<4x?xf32>

  // Check the matmul is not padded due to the failure to compute a padding value for the dynamic output.
  // INPUTS-ONLY:  = linalg.matmul ins(%[[T0]], %[[T1]]
  // INPUTS-ONLY-SAME:             outs(%[[T2]]
  %4 = linalg.matmul ins(%1, %2 : tensor<4x?xf32>, tensor<?x?xf32>) outs(%3 : tensor<4x?xf32>) -> tensor<4x?xf32>
  %5 = tensor.insert_slice %4 into %arg2[%iv0, %iv1] [4, %0] [1, 1] : tensor<4x?xf32> into tensor<24x25xf32>
  func.return %5 : tensor<24x25xf32>
}

// -----

#map0 = affine_map<()[s0] -> (64, s0)>

//      FILL:  rank_reducing
// FILL-SAME:    %[[ARG0:[0-9a-zA-Z]*]]: tensor<1x64x1x64xf32>
func.func @rank_reducing(%arg0: tensor<1x64x1x64xf32>,
                         %iv0 : index) -> tensor<1x?x?xf32> {
  //      FILL:  %[[CST:.*]] = arith.constant 1.
  %cst = arith.constant 0.0 : f32
  %size = affine.min #map0()[%iv0]
  %0 = tensor.extract_slice %arg0[0, 0, 0, 0] [1, %size, 1, %size] [1, 1, 1, 1] : tensor<1x64x1x64xf32> to tensor<1x?x?xf32>

  // Check the fill is padded despite the rank-reducing slice operation.
  //      FILL:  %[[T0:.*]] = tensor.pad
  //      FILL:  tensor.yield %[[CST]]
  //      FILL:  %[[T1:.*]] = linalg.fill ins(%{{.*}}{{.*}}outs(%[[T0]]
  // FILL-SAME:    tensor<1x64x64xf32>
  //      FILL:  = tensor.extract_slice %[[T1]]
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x?x?xf32>) -> tensor<1x?x?xf32>
  func.return %1 : tensor<1x?x?xf32>
}

// -----

#map0 = affine_map<()[s0] -> (7, s0)>

//      PARTIAL:  padding_the_output_dims_only
func.func @padding_the_output_dims_only(%arg0: tensor<24x12xf32>,
                                        %arg1: tensor<12x25xf32>,
                                        %arg2: tensor<24x25xf32>,
                                        %iv0 : index, %iv1 : index, %iv2 : index) -> tensor<24x25xf32> {
  //  PARTIAL-DAG:  %[[C0:.*]] = arith.constant 0 : index
  //  PARTIAL-DAG:  %[[TS:.*]] = affine.apply
  %0 = affine.min #map0()[%iv2]

  // Check only the output dimensions of the matmul are padded.
  //      PARTIAL:  %[[T0:.*]] = tensor.pad
  // PARTIAL-SAME:                 [%[[TS]], %[[C0]]
  //      PARTIAL:  %[[T1:.*]] = tensor.pad
  // PARTIAL-SAME:                 [%[[C0]], %[[TS]]
  //      PARTIAL:  %[[T2:.*]] = tensor.pad
  // PARTIAL-SAME:                 [%[[TS]], %[[TS]]
  %1 = tensor.extract_slice %arg0[%iv0, %iv2] [%0, %0] [1, 1] : tensor<24x12xf32> to tensor<?x?xf32>
  %2 = tensor.extract_slice %arg1[%iv2, %iv1] [%0, %0] [1, 1] : tensor<12x25xf32> to tensor<?x?xf32>
  %3 = tensor.extract_slice %arg2[%iv0, %iv1] [%0, %0] [1, 1] : tensor<24x25xf32> to tensor<?x?xf32>

  //      PARTIAL:  = linalg.matmul ins(%[[T0]], %[[T1]]
  // PARTIAL-SAME:             outs(%[[T2]]
  %4 = linalg.matmul ins(%1, %2 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%3 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %5 = tensor.insert_slice %4 into %arg2[%iv0, %iv1] [%0, %0] [1, 1] : tensor<?x?xf32> into tensor<24x25xf32>
  func.return %5 : tensor<24x25xf32>
}

// -----

//  DEPTHWISE_CONV_2D-DAG: #[[MAP0:[0-9a-z]+]] = affine_map<()[s0] -> (4, -s0 + 11)>
//  DEPTHWISE_CONV_2D-DAG: #[[MAP1:[0-9a-z]+]] = affine_map<()[s0] -> (s0 * 2)>
//  DEPTHWISE_CONV_2D-DAG: #[[MAP2:[0-9a-z]+]] = affine_map<()[s0] -> (s0 * 2 + 1)>
//  DEPTHWISE_CONV_2D-DAG: #[[MAP3:[0-9a-z]+]] = affine_map<()[s0] -> (s0 * -2 + 8)>
//  DEPTHWISE_CONV_2D-DAG: #[[MAP4:[0-9a-z]+]] = affine_map<()[s0] -> (-s0 + 4)>

#map0 = affine_map<()[s0] -> (4, -s0 + 11)>
#map1 = affine_map<()[s0] -> (s0 * 2)>
#map2 = affine_map<()[s0] -> (s0 * 2 + 1)>

//      DEPTHWISE_CONV_2D: depthwise_conv_2d_padding
// DEPTHWISE_CONV_2D-SAME: %[[ARG0:[0-9a-zA-Z]*]]: tensor<1x23x3x16xf32>
// DEPTHWISE_CONV_2D-SAME: %[[ARG1:[0-9a-zA-Z]*]]: tensor<3x3x16xf32>
// DEPTHWISE_CONV_2D-SAME: %[[ARG2:[0-9a-zA-Z]*]]: tensor<1x13x1x16xf32>
// DEPTHWISE_CONV_2D-SAME: %[[IV0:[0-9a-zA-Z]*]]: index
func.func @depthwise_conv_2d_padding(%arg0: tensor<1x23x3x16xf32>,
                                     %arg1: tensor<3x3x16xf32>,
                                     %arg2: tensor<1x13x1x16xf32>,
                                     %iv0: index) -> tensor<1x?x1x16xf32> {
  //  DEPTHWISE_CONV_2D-DAG: %[[CST:.*]] = arith.constant 0.
  //  DEPTHWISE_CONV_2D-DAG: %[[C0:.*]] = arith.constant 0 : index
  //  DEPTHWISE_CONV_2D-DAG: %[[T0:.*]] = affine.min #[[MAP0]]()[%[[IV0]]]
  %0 = affine.min #map0()[%iv0]
  %1 = affine.apply #map1()[%iv0]
  %2 = affine.apply #map2()[%0]

  //      DEPTHWISE_CONV_2D: %[[T3:.*]] = tensor.extract_slice %[[ARG0]]
  //      DEPTHWISE_CONV_2D: %[[T4:.*]] = tensor.extract_slice %[[ARG2]]
  %3 = tensor.extract_slice %arg0[0, %1, 0, 0] [1, %2, 3, 16] [1, 1, 1, 1] : tensor<1x23x3x16xf32> to tensor<1x?x3x16xf32>
  %4 = tensor.extract_slice %arg2[0, %iv0, 0, 0] [1, %0, 1, 16] [1, 1, 1, 1] : tensor<1x13x1x16xf32> to tensor<1x?x1x16xf32>

  // Check the padding on the input.
  //      DEPTHWISE_CONV_2D: %[[T5:.*]] = affine.apply #[[MAP3]]()[%[[T0]]]
  //      DEPTHWISE_CONV_2D: %[[T6:.*]] = tensor.pad %[[T3]]
  // DEPTHWISE_CONV_2D-SAME:                low[%[[C0]], %[[C0]], %[[C0]], %[[C0]]]
  // DEPTHWISE_CONV_2D-SAME:                high[%[[C0]], %[[T5]], %[[C0]], %[[C0]]]
  //      DEPTHWISE_CONV_2D: tensor.yield %[[CST]] : f32

  // Check the padding on the output.
  //      DEPTHWISE_CONV_2D: %[[T7:.*]] = affine.apply #[[MAP4]]()[%[[T0]]]
  //      DEPTHWISE_CONV_2D: %[[T8:.*]] = tensor.pad %[[T4]]
  // DEPTHWISE_CONV_2D-SAME:                low[%[[C0]], %[[C0]], %[[C0]], %[[C0]]]
  // DEPTHWISE_CONV_2D-SAME:                high[%[[C0]], %[[T7]], %[[C0]], %[[C0]]]
  //      DEPTHWISE_CONV_2D: tensor.yield %[[CST]] : f32

  //      DEPTHWISE_CONV_2D: %[[T9:.*]] = linalg.depthwise_conv_2d_nhwc_hwc
  // DEPTHWISE_CONV_2D-SAME: ins(%[[T6]], %[[ARG1]] : tensor<1x9x3x16xf32>, tensor<3x3x16xf32>)
  // DEPTHWISE_CONV_2D-SAME: outs(%[[T8]] : tensor<1x4x1x16xf32>)
  %5 = linalg.depthwise_conv_2d_nhwc_hwc
      {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%3, %arg1 : tensor<1x?x3x16xf32>, tensor<3x3x16xf32>)
      outs(%4 : tensor<1x?x1x16xf32>) -> tensor<1x?x1x16xf32>

  // Check the extract_slice to crop the padded output before return.
  //      DEPTHWISE_CONV_2D: %[[T10:.*]] = tensor.extract_slice %[[T9]][0, 0, 0, 0]
  // DEPTHWISE_CONV_2D-SAME:                 [1, %[[T0]], 1, 16]
  //      DEPTHWISE_CONV_2D: return %[[T10]] : tensor<1x?x1x16xf32>
  return %5 : tensor<1x?x1x16xf32>
}
