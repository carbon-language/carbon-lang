// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK-LABEL: func @vector_transfer_ops_0d(
func.func @vector_transfer_ops_0d(%arg0: tensor<f32>, %arg1: memref<f32>)
  -> tensor<f32> {
    %f0 = arith.constant 0.0 : f32
    %0 = vector.transfer_read %arg0[], %f0 {permutation_map = affine_map<()->()>} :
      tensor<f32>, vector<f32>
    %1 = vector.transfer_write %0, %arg0[] {permutation_map = affine_map<()->()>} :
      vector<f32>, tensor<f32>
    %2 = vector.transfer_read %arg1[], %f0 {permutation_map = affine_map<()->()>} :
      memref<f32>, vector<f32>
    vector.transfer_write %2, %arg1[] {permutation_map = affine_map<()->()>} :
      vector<f32>, memref<f32>
    return %1: tensor<f32>
}

// CHECK-LABEL: func @vector_transfer_ops_0d_from_higher_d(
func.func @vector_transfer_ops_0d_from_higher_d(%arg0: tensor<?xf32>, %arg1: memref<?x?xf32>)
  -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %f0 = arith.constant 0.0 : f32
    %0 = vector.transfer_read %arg0[%c0], %f0 {permutation_map = affine_map<(d0)->()>} :
      tensor<?xf32>, vector<f32>
    %1 = vector.transfer_write %0, %arg0[%c0] {permutation_map = affine_map<(d0)->()>} :
      vector<f32>, tensor<?xf32>
    %2 = vector.transfer_read %arg1[%c0, %c0], %f0 {permutation_map = affine_map<(d0, d1)->()>} :
      memref<?x?xf32>, vector<f32>
    vector.transfer_write %2, %arg1[%c0, %c0] {permutation_map = affine_map<(d0, d1)->()>} :
      vector<f32>, memref<?x?xf32>
    return %1: tensor<?xf32>
}

// CHECK-LABEL: func @vector_transfer_ops(
func.func @vector_transfer_ops(%arg0: memref<?x?xf32>,
                          %arg1 : memref<?x?xvector<4x3xf32>>,
                          %arg2 : memref<?x?xvector<4x3xi32>>,
                          %arg3 : memref<?x?xvector<4x3xindex>>,
                          %arg4 : memref<?x?x?xf32>) {
  // CHECK: %[[C3:.*]] = arith.constant 3 : index
  %c3 = arith.constant 3 : index
  %cst = arith.constant 3.0 : f32
  %f0 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : i32
  %i0 = arith.constant 0 : index
  %i1 = arith.constant 1 : i1

  %vf0 = vector.splat %f0 : vector<4x3xf32>
  %v0 = vector.splat %c0 : vector<4x3xi32>
  %vi0 = vector.splat %i0 : vector<4x3xindex>
  %m = arith.constant dense<[0, 0, 1, 0, 1]> : vector<5xi1>
  %m2 = vector.splat %i1 : vector<5x4xi1>
  //
  // CHECK: vector.transfer_read
  %0 = vector.transfer_read %arg0[%c3, %c3], %f0 {permutation_map = affine_map<(d0, d1)->(d0)>} : memref<?x?xf32>, vector<128xf32>
  // CHECK: vector.transfer_read
  %1 = vector.transfer_read %arg0[%c3, %c3], %f0 {permutation_map = affine_map<(d0, d1)->(d1, d0)>} : memref<?x?xf32>, vector<3x7xf32>
  // CHECK: vector.transfer_read
  %2 = vector.transfer_read %arg0[%c3, %c3], %cst {permutation_map = affine_map<(d0, d1)->(d0)>} : memref<?x?xf32>,  vector<128xf32>
  // CHECK: vector.transfer_read
  %3 = vector.transfer_read %arg0[%c3, %c3], %cst {permutation_map = affine_map<(d0, d1)->(d1)>} : memref<?x?xf32>,  vector<128xf32>
  // CHECK: vector.transfer_read %{{.*}}[%[[C3]], %[[C3]]], %{{.*}} : memref<?x?xvector<4x3xf32>>, vector<1x1x4x3xf32>
  %4 = vector.transfer_read %arg1[%c3, %c3], %vf0 {permutation_map = affine_map<(d0, d1)->(d0, d1)>} : memref<?x?xvector<4x3xf32>>, vector<1x1x4x3xf32>
  // CHECK: vector.transfer_read %{{.*}}[%[[C3]], %[[C3]]], %{{.*}} {in_bounds = [false, true]} : memref<?x?xvector<4x3xf32>>, vector<1x1x4x3xf32>
  %5 = vector.transfer_read %arg1[%c3, %c3], %vf0 {in_bounds = [false, true]} : memref<?x?xvector<4x3xf32>>, vector<1x1x4x3xf32>
  // CHECK: vector.transfer_read %{{.*}}[%[[C3]], %[[C3]]], %{{.*}} : memref<?x?xvector<4x3xi32>>, vector<5x24xi8>
  %6 = vector.transfer_read %arg2[%c3, %c3], %v0 : memref<?x?xvector<4x3xi32>>, vector<5x24xi8>
  // CHECK: vector.transfer_read %{{.*}}[%[[C3]], %[[C3]]], %{{.*}} : memref<?x?xvector<4x3xindex>>, vector<5x48xi8>
  %7 = vector.transfer_read %arg3[%c3, %c3], %vi0 : memref<?x?xvector<4x3xindex>>, vector<5x48xi8>
  // CHECK: vector.transfer_read %{{.*}}[%[[C3]], %[[C3]]], %{{.*}}, %{{.*}} : memref<?x?xf32>, vector<5xf32>
  %8 = vector.transfer_read %arg0[%c3, %c3], %f0, %m : memref<?x?xf32>, vector<5xf32>
  // CHECK: vector.transfer_read %{{.*}}[%[[C3]], %[[C3]], %[[C3]]], %{{.*}}, %{{.*}} : memref<?x?x?xf32>, vector<5x4x8xf32>
  %9 = vector.transfer_read %arg4[%c3, %c3, %c3], %f0, %m2 {permutation_map = affine_map<(d0, d1, d2)->(d1, d0, 0)>} : memref<?x?x?xf32>, vector<5x4x8xf32>

  // CHECK: vector.transfer_write
  vector.transfer_write %0, %arg0[%c3, %c3] {permutation_map = affine_map<(d0, d1)->(d0)>} : vector<128xf32>, memref<?x?xf32>
  // CHECK: vector.transfer_write
  vector.transfer_write %1, %arg0[%c3, %c3] {permutation_map = affine_map<(d0, d1)->(d1, d0)>} : vector<3x7xf32>, memref<?x?xf32>
  // CHECK: vector.transfer_write %{{.*}}, %{{.*}}[%[[C3]], %[[C3]]] : vector<1x1x4x3xf32>, memref<?x?xvector<4x3xf32>>
  vector.transfer_write %4, %arg1[%c3, %c3] {permutation_map = affine_map<(d0, d1)->(d0, d1)>} : vector<1x1x4x3xf32>, memref<?x?xvector<4x3xf32>>
  // CHECK: vector.transfer_write %{{.*}}, %{{.*}}[%[[C3]], %[[C3]]] : vector<1x1x4x3xf32>, memref<?x?xvector<4x3xf32>>
  vector.transfer_write %5, %arg1[%c3, %c3] {in_bounds = [false, false]} : vector<1x1x4x3xf32>, memref<?x?xvector<4x3xf32>>
  // CHECK: vector.transfer_write %{{.*}}, %{{.*}}[%[[C3]], %[[C3]]] : vector<5x24xi8>, memref<?x?xvector<4x3xi32>>
  vector.transfer_write %6, %arg2[%c3, %c3] : vector<5x24xi8>, memref<?x?xvector<4x3xi32>>
  // CHECK: vector.transfer_write %{{.*}}, %{{.*}}[%[[C3]], %[[C3]]] : vector<5x48xi8>, memref<?x?xvector<4x3xindex>>
  vector.transfer_write %7, %arg3[%c3, %c3] : vector<5x48xi8>, memref<?x?xvector<4x3xindex>>
  // CHECK: vector.transfer_write %{{.*}}, %{{.*}}[%[[C3]], %[[C3]]], %{{.*}} : vector<5xf32>, memref<?x?xf32>
  vector.transfer_write %8, %arg0[%c3, %c3], %m : vector<5xf32>, memref<?x?xf32>

  return
}


// CHECK-LABEL: func @vector_transfer_ops_tensor(
func.func @vector_transfer_ops_tensor(%arg0: tensor<?x?xf32>,
                          %arg1 : tensor<?x?xvector<4x3xf32>>,
                          %arg2 : tensor<?x?xvector<4x3xi32>>,
                          %arg3 : tensor<?x?xvector<4x3xindex>>) ->
  (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xvector<4x3xf32>>,
   tensor<?x?xvector<4x3xf32>>, tensor<?x?xvector<4x3xi32>>,
   tensor<?x?xvector<4x3xindex>>){
  // CHECK: %[[C3:.*]] = arith.constant 3 : index
  %c3 = arith.constant 3 : index
  %cst = arith.constant 3.0 : f32
  %f0 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : i32
  %i0 = arith.constant 0 : index

  %vf0 = vector.splat %f0 : vector<4x3xf32>
  %v0 = vector.splat %c0 : vector<4x3xi32>
  %vi0 = vector.splat %i0 : vector<4x3xindex>

  //
  // CHECK: vector.transfer_read
  %0 = vector.transfer_read %arg0[%c3, %c3], %f0 {permutation_map = affine_map<(d0, d1)->(d0)>} : tensor<?x?xf32>, vector<128xf32>
  // CHECK: vector.transfer_read
  %1 = vector.transfer_read %arg0[%c3, %c3], %f0 {permutation_map = affine_map<(d0, d1)->(d1, d0)>} : tensor<?x?xf32>, vector<3x7xf32>
  // CHECK: vector.transfer_read
  %2 = vector.transfer_read %arg0[%c3, %c3], %cst {permutation_map = affine_map<(d0, d1)->(d0)>} : tensor<?x?xf32>,  vector<128xf32>
  // CHECK: vector.transfer_read
  %3 = vector.transfer_read %arg0[%c3, %c3], %cst {permutation_map = affine_map<(d0, d1)->(d1)>} : tensor<?x?xf32>,  vector<128xf32>
  // CHECK: vector.transfer_read %{{.*}}[%[[C3]], %[[C3]]], %{{.*}} : tensor<?x?xvector<4x3xf32>>, vector<1x1x4x3xf32>
  %4 = vector.transfer_read %arg1[%c3, %c3], %vf0 {permutation_map = affine_map<(d0, d1)->(d0, d1)>} : tensor<?x?xvector<4x3xf32>>, vector<1x1x4x3xf32>
  // CHECK: vector.transfer_read %{{.*}}[%[[C3]], %[[C3]]], %{{.*}} {in_bounds = [false, true]} : tensor<?x?xvector<4x3xf32>>, vector<1x1x4x3xf32>
  %5 = vector.transfer_read %arg1[%c3, %c3], %vf0 {in_bounds = [false, true]} : tensor<?x?xvector<4x3xf32>>, vector<1x1x4x3xf32>
  // CHECK: vector.transfer_read %{{.*}}[%[[C3]], %[[C3]]], %{{.*}} : tensor<?x?xvector<4x3xi32>>, vector<5x24xi8>
  %6 = vector.transfer_read %arg2[%c3, %c3], %v0 : tensor<?x?xvector<4x3xi32>>, vector<5x24xi8>
  // CHECK: vector.transfer_read %{{.*}}[%[[C3]], %[[C3]]], %{{.*}} : tensor<?x?xvector<4x3xindex>>, vector<5x48xi8>
  %7 = vector.transfer_read %arg3[%c3, %c3], %vi0 : tensor<?x?xvector<4x3xindex>>, vector<5x48xi8>


  // CHECK: vector.transfer_write
  %8 = vector.transfer_write %0, %arg0[%c3, %c3] {permutation_map = affine_map<(d0, d1)->(d0)>} : vector<128xf32>, tensor<?x?xf32>
  // CHECK: vector.transfer_write
  %9 = vector.transfer_write %1, %arg0[%c3, %c3] {permutation_map = affine_map<(d0, d1)->(d1, d0)>} : vector<3x7xf32>, tensor<?x?xf32>
  // CHECK: vector.transfer_write %{{.*}}, %{{.*}}[%[[C3]], %[[C3]]] : vector<1x1x4x3xf32>, tensor<?x?xvector<4x3xf32>>
  %10 = vector.transfer_write %4, %arg1[%c3, %c3] {permutation_map = affine_map<(d0, d1)->(d0, d1)>} : vector<1x1x4x3xf32>, tensor<?x?xvector<4x3xf32>>
  // CHECK: vector.transfer_write %{{.*}}, %{{.*}}[%[[C3]], %[[C3]]] : vector<1x1x4x3xf32>, tensor<?x?xvector<4x3xf32>>
  %11 = vector.transfer_write %5, %arg1[%c3, %c3] {in_bounds = [false, false]} : vector<1x1x4x3xf32>, tensor<?x?xvector<4x3xf32>>
  // CHECK: vector.transfer_write %{{.*}}, %{{.*}}[%[[C3]], %[[C3]]] : vector<5x24xi8>, tensor<?x?xvector<4x3xi32>>
  %12 = vector.transfer_write %6, %arg2[%c3, %c3] : vector<5x24xi8>, tensor<?x?xvector<4x3xi32>>
  // CHECK: vector.transfer_write %{{.*}}, %{{.*}}[%[[C3]], %[[C3]]] : vector<5x48xi8>, tensor<?x?xvector<4x3xindex>>
  %13 = vector.transfer_write %7, %arg3[%c3, %c3] : vector<5x48xi8>, tensor<?x?xvector<4x3xindex>>

  return %8, %9, %10, %11, %12, %13 :
    tensor<?x?xf32>, tensor<?x?xf32>,  tensor<?x?xvector<4x3xf32>>,
    tensor<?x?xvector<4x3xf32>>, tensor<?x?xvector<4x3xi32>>,
    tensor<?x?xvector<4x3xindex>>
}

// CHECK-LABEL: @vector_broadcast
func.func @vector_broadcast(%a: f32, %b: vector<f32>, %c: vector<16xf32>, %d: vector<1x16xf32>, %e: vector<8x1xf32>) -> vector<8x16xf32> {
  // CHECK: vector.broadcast %{{.*}} : f32 to vector<f32>
  %0 = vector.broadcast %a : f32 to vector<f32>
  // CHECK: vector.broadcast %{{.*}} : vector<f32> to vector<4xf32>
  %1 = vector.broadcast %b : vector<f32> to vector<4xf32>
  // CHECK: vector.broadcast %{{.*}} : f32 to vector<16xf32>
  %2 = vector.broadcast %a : f32 to vector<16xf32>
  // CHECK-NEXT: vector.broadcast %{{.*}} : vector<16xf32> to vector<8x16xf32>
  %3 = vector.broadcast %c : vector<16xf32> to vector<8x16xf32>
  // CHECK-NEXT: vector.broadcast %{{.*}} : vector<1x16xf32> to vector<8x16xf32>
  %4 = vector.broadcast %d : vector<1x16xf32> to vector<8x16xf32>
  // CHECK-NEXT: vector.broadcast %{{.*}} : vector<8x1xf32> to vector<8x16xf32>
  %5 = vector.broadcast %e : vector<8x1xf32> to vector<8x16xf32>
  return %4 : vector<8x16xf32>
}

// CHECK-LABEL: @shuffle1D
func.func @shuffle1D(%a: vector<2xf32>, %b: vector<4xf32>) -> vector<2xf32> {
  // CHECK: vector.shuffle %{{.*}}, %{{.*}}[0, 1, 2, 3] : vector<2xf32>, vector<2xf32>
  %1 = vector.shuffle %a, %a[0, 1, 2, 3] : vector<2xf32>, vector<2xf32>
  // CHECK-NEXT: vector.shuffle %{{.*}}, %{{.*}}[0, 1, 2] : vector<4xf32>, vector<4xf32>
  %2 = vector.shuffle %1, %b[0, 1, 2] : vector<4xf32>, vector<4xf32>
  // CHECK-NEXT: vector.shuffle %{{.*}}, %{{.*}}[0, 6] : vector<3xf32>, vector<4xf32>
  %3 = vector.shuffle %2, %b[0, 6] : vector<3xf32>, vector<4xf32>
  return %3 : vector<2xf32>
}

// CHECK-LABEL: @shuffle2D
func.func @shuffle2D(%a: vector<1x4xf32>, %b: vector<2x4xf32>) -> vector<3x4xf32> {
  // CHECK: vector.shuffle %{{.*}}, %{{.*}}[0, 1, 2] : vector<1x4xf32>, vector<2x4xf32>
  %1 = vector.shuffle %a, %b[0, 1, 2] : vector<1x4xf32>, vector<2x4xf32>
  return %1 : vector<3x4xf32>
}

// CHECK-LABEL: @extract_element_0d
func.func @extract_element_0d(%a: vector<f32>) -> f32 {
  // CHECK-NEXT: vector.extractelement %{{.*}}[] : vector<f32>
  %1 = vector.extractelement %a[] : vector<f32>
  return %1 : f32
}

// CHECK-LABEL: @extract_element
func.func @extract_element(%a: vector<16xf32>) -> f32 {
  // CHECK:      %[[C15:.*]] = arith.constant 15 : i32
  %c = arith.constant 15 : i32
  // CHECK-NEXT: vector.extractelement %{{.*}}[%[[C15]] : i32] : vector<16xf32>
  %1 = vector.extractelement %a[%c : i32] : vector<16xf32>
  return %1 : f32
}

// CHECK-LABEL: @extract
func.func @extract(%arg0: vector<4x8x16xf32>) -> (vector<4x8x16xf32>, vector<8x16xf32>, vector<16xf32>, f32) {
  // CHECK: vector.extract {{.*}}[] : vector<4x8x16xf32>
  %0 = vector.extract %arg0[] : vector<4x8x16xf32>
  // CHECK: vector.extract {{.*}}[3] : vector<4x8x16xf32>
  %1 = vector.extract %arg0[3] : vector<4x8x16xf32>
  // CHECK-NEXT: vector.extract {{.*}}[3, 3] : vector<4x8x16xf32>
  %2 = vector.extract %arg0[3, 3] : vector<4x8x16xf32>
  // CHECK-NEXT: vector.extract {{.*}}[3, 3, 3] : vector<4x8x16xf32>
  %3 = vector.extract %arg0[3, 3, 3] : vector<4x8x16xf32>
  return %0, %1, %2, %3 : vector<4x8x16xf32>, vector<8x16xf32>, vector<16xf32>, f32
}

// CHECK-LABEL: @insert_element_0d
func.func @insert_element_0d(%a: f32, %b: vector<f32>) -> vector<f32> {
  // CHECK-NEXT: vector.insertelement %{{.*}}, %{{.*}}[] : vector<f32>
  %1 = vector.insertelement %a, %b[] : vector<f32>
  return %1 : vector<f32>
}

// CHECK-LABEL: @insert_element
func.func @insert_element(%a: f32, %b: vector<16xf32>) -> vector<16xf32> {
  // CHECK:      %[[C15:.*]] = arith.constant 15 : i32
  %c = arith.constant 15 : i32
  // CHECK-NEXT: vector.insertelement %{{.*}}, %{{.*}}[%[[C15]] : i32] : vector<16xf32>
  %1 = vector.insertelement %a, %b[%c : i32] : vector<16xf32>
  return %1 : vector<16xf32>
}

// CHECK-LABEL: @insert
func.func @insert(%a: f32, %b: vector<16xf32>, %c: vector<8x16xf32>, %res: vector<4x8x16xf32>) -> vector<4x8x16xf32> {
  // CHECK: vector.insert %{{.*}}, %{{.*}}[3] : vector<8x16xf32> into vector<4x8x16xf32>
  %1 = vector.insert %c, %res[3] : vector<8x16xf32> into vector<4x8x16xf32>
  // CHECK: vector.insert %{{.*}}, %{{.*}}[3, 3] : vector<16xf32> into vector<4x8x16xf32>
  %2 = vector.insert %b, %res[3, 3] : vector<16xf32> into vector<4x8x16xf32>
  // CHECK: vector.insert %{{.*}}, %{{.*}}[3, 3, 3] : f32 into vector<4x8x16xf32>
  %3 = vector.insert %a, %res[3, 3, 3] : f32 into vector<4x8x16xf32>
  // CHECK: vector.insert %{{.*}}, %{{.*}}[] : vector<4x8x16xf32> into vector<4x8x16xf32>
  %4 = vector.insert %3, %3[] : vector<4x8x16xf32> into vector<4x8x16xf32>
  return %4 : vector<4x8x16xf32>
}

// CHECK-LABEL: @outerproduct
func.func @outerproduct(%arg0: vector<4xf32>, %arg1: vector<8xf32>, %arg2: vector<4x8xf32>) -> vector<4x8xf32> {
  // CHECK: vector.outerproduct {{.*}} : vector<4xf32>, vector<8xf32>
  %0 = vector.outerproduct %arg0, %arg1 : vector<4xf32>, vector<8xf32>
  // CHECK: vector.outerproduct {{.*}}, {{.*}}, {{.*}} : vector<4xf32>, vector<8xf32>
  %1 = vector.outerproduct %arg0, %arg1, %arg2 : vector<4xf32>, vector<8xf32>
  return %1 : vector<4x8xf32>
}

// CHECK-LABEL: @insert_strided_slice
func.func @insert_strided_slice(%a: vector<4x4xf32>, %b: vector<4x8x16xf32>) {
  // CHECK: vector.insert_strided_slice %{{.*}}, %{{.*}} {offsets = [2, 2, 2], strides = [1, 1]} : vector<4x4xf32> into vector<4x8x16xf32>
  %1 = vector.insert_strided_slice %a, %b {offsets = [2, 2, 2], strides = [1, 1]} : vector<4x4xf32> into vector<4x8x16xf32>
  return
}

// CHECK-LABEL: @extract_strided_slice
func.func @extract_strided_slice(%arg0: vector<4x8x16xf32>) -> vector<2x2x16xf32> {
  // CHECK: vector.extract_strided_slice %{{.*}} {offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x8x16xf32>
  %1 = vector.extract_strided_slice %arg0 {offsets = [2, 2], sizes = [2, 2], strides = [1, 1]} : vector<4x8x16xf32> to vector<2x2x16xf32>
  return %1: vector<2x2x16xf32>
}

#contraction_to_scalar_accesses = [
  affine_map<(i) -> (i)>,
  affine_map<(i) -> (i)>,
  affine_map<(i) -> ()>
]
#contraction_to_scalar_trait = {
  indexing_maps = #contraction_to_scalar_accesses,
  iterator_types = ["reduction"]
}
// CHECK-LABEL: @contraction_to_scalar
func.func @contraction_to_scalar(%arg0: vector<10xf32>, %arg1: vector<10xf32>) -> f32 {
  // CHECK:      %[[C0:.*]] = arith.constant 0.000000e+00 : f32
  %f0 = arith.constant 0.0: f32
  // CHECK:      %[[X:.*]] = vector.contract {indexing_maps = [#{{.*}}, #{{.*}}, #{{.*}}], iterator_types = ["reduction"], kind = #vector.kind<add>} %{{.*}}, %{{.*}}, %[[C0]] : vector<10xf32>, vector<10xf32> into f32
  %0 = vector.contract #contraction_to_scalar_trait %arg0, %arg1, %f0
    : vector<10xf32>, vector<10xf32> into f32
  // CHECK:      return %[[X]] : f32
  return %0 : f32
}

#contraction_to_scalar_max_accesses = [
  affine_map<(i) -> (i)>,
  affine_map<(i) -> (i)>,
  affine_map<(i) -> ()>
]
#contraction_to_scalar_max_trait = {
  indexing_maps = #contraction_to_scalar_max_accesses,
  iterator_types = ["reduction"],
  kind = #vector.kind<maxf>
}
// CHECK-LABEL: @contraction_to_scalar_with_max
func.func @contraction_to_scalar_with_max(%arg0: vector<10xf32>, %arg1: vector<10xf32>) -> f32 {
  // CHECK:      %[[C0:.*]] = arith.constant 0.000000e+00 : f32
  %f0 = arith.constant 0.0: f32
  // CHECK:      %[[X:.*]] = vector.contract {indexing_maps = [#{{.*}}, #{{.*}}, #{{.*}}], iterator_types = ["reduction"], kind = #vector.kind<maxf>} %{{.*}}, %{{.*}}, %[[C0]] : vector<10xf32>, vector<10xf32> into f32
  %0 = vector.contract #contraction_to_scalar_max_trait %arg0, %arg1, %f0
    : vector<10xf32>, vector<10xf32> into f32
  // CHECK:      return %[[X]] : f32
  return %0 : f32
}

#contraction_accesses0 = [
  affine_map<(b0, f0, f1, c0, c1) -> (c0, b0, c1, f0)>,
  affine_map<(b0, f0, f1, c0, c1) -> (b0, c1, c0, f1)>,
  affine_map<(b0, f0, f1, c0, c1) -> (b0, f0, f1)>
]
#contraction_trait0 = {
  indexing_maps = #contraction_accesses0,
  iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]
}
#contraction_accesses1 = [              // 7,  8, 16, 15
  affine_map<(f0, f1, f2, f3, c0, c1) -> (c0, f0, c1, f2)>,
                                        // 8, 16,  7,  5
  affine_map<(f0, f1, f2, f3, c0, c1) -> (f1, c1, c0, f3)>,
                                        // 8,  8, 15,  5
  affine_map<(f0, f1, f2, f3, c0, c1) -> (f0, f1, f2, f3)>
]
#iterator_types1 = ["parallel", "parallel", "parallel", "parallel", "reduction",
                    "reduction"]
#contraction_trait1 = {
  indexing_maps = #contraction_accesses1,
  iterator_types = #iterator_types1
}
#contraction_trait2 = {
  indexing_maps = #contraction_accesses1,
  iterator_types = #iterator_types1,
  kind = #vector.kind<maxf>
}
// CHECK-LABEL: @contraction
func.func @contraction(%arg0 : vector<7x8x16x15xf32>, %arg1 : vector<8x16x7x5xf32>,
                  %arg2 : vector<8x15x5xf32>, %arg3 : vector<8x8x15x5xf32>,
                  %arg4 : vector<7x8x16x15xf16>, %arg5 : vector<8x16x7x5xf16>) {
  // Test contraction with batch and contracting dims.
  // CHECK: vector.contract {indexing_maps = [#{{.*}}, #{{.*}}, #{{.*}}], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"], kind = #vector.kind<add>} {{.*}}, {{.*}}, {{.*}} : vector<7x8x16x15xf32>, vector<8x16x7x5xf32> into vector<8x15x5xf32>
  %0 = vector.contract #contraction_trait0 %arg0, %arg1, %arg2
      : vector<7x8x16x15xf32>, vector<8x16x7x5xf32> into vector<8x15x5xf32>
  // Test contraction with only contracting dims. In this case the lhs/rhs
  // dimension of size 8 will be considered a parallel dim for lhs/rhs and will
  // appear twice in the output.
  // CHECK: vector.contract {indexing_maps = [#{{.*}}, #{{.*}}, #{{.*}}], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"], kind = #vector.kind<add>} {{.*}}, {{.*}}, {{.*}} : vector<7x8x16x15xf32>, vector<8x16x7x5xf32> into vector<8x8x15x5xf32>
  %1 = vector.contract #contraction_trait1 %arg0, %arg1, %arg3
      : vector<7x8x16x15xf32>, vector<8x16x7x5xf32> into vector<8x8x15x5xf32>
  // Test contraction with optional vector mask arguments.
  %lhs_mask = vector.constant_mask [7, 8, 16, 15] : vector<7x8x16x15xi1>
  %rhs_mask = vector.constant_mask [8, 16, 7, 5] : vector<8x16x7x5xi1>
  // CHECK: vector.contract {indexing_maps = [#{{.*}}, #{{.*}}, #{{.*}}], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"], kind = #vector.kind<add>} {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}} : vector<7x8x16x15xf32>, vector<8x16x7x5xf32> into vector<8x8x15x5xf32>
  %2 = vector.contract #contraction_trait1 %arg0, %arg1, %arg3, %lhs_mask,
                                           %rhs_mask
      : vector<7x8x16x15xf32>, vector<8x16x7x5xf32> into vector<8x8x15x5xf32>
  // Test contraction with mixed type.
  // CHECK: vector.contract {indexing_maps = [#{{.*}}, #{{.*}}, #{{.*}}], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"], kind = #vector.kind<add>} {{.*}}, {{.*}}, {{.*}} : vector<7x8x16x15xf16>, vector<8x16x7x5xf16> into vector<8x8x15x5xf32>
  %3 = vector.contract #contraction_trait1 %arg4, %arg5, %arg3
      : vector<7x8x16x15xf16>, vector<8x16x7x5xf16> into vector<8x8x15x5xf32>
  // Test contraction with "max" instead of "add".
  // CHECK: vector.contract {indexing_maps = [#{{.*}}, #{{.*}}, #{{.*}}], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"], kind = #vector.kind<maxf>} {{.*}}, {{.*}}, {{.*}} : vector<7x8x16x15xf32>, vector<8x16x7x5xf32> into vector<8x8x15x5xf32>
  %4 = vector.contract #contraction_trait2 %arg0, %arg1, %arg3
      : vector<7x8x16x15xf32>, vector<8x16x7x5xf32> into vector<8x8x15x5xf32>
  return
}

// CHECK-LABEL: @create_vector_mask
func.func @create_vector_mask() {
  // CHECK:      %[[C2:.*]] = arith.constant 2 : index
  %c2 = arith.constant 2 : index
  // CHECK-NEXT: %[[C3:.*]] = arith.constant 3 : index
  %c3 = arith.constant 3 : index
  // CHECK-NEXT: vector.create_mask %[[C3]], %[[C2]] : vector<4x3xi1>
  %0 = vector.create_mask %c3, %c2 : vector<4x3xi1>

  return
}

// CHECK-LABEL: @constant_vector_mask_0d
func.func @constant_vector_mask_0d() {
  // CHECK: vector.constant_mask [0] : vector<i1>
  %0 = vector.constant_mask [0] : vector<i1>
  // CHECK: vector.constant_mask [1] : vector<i1>
  %1 = vector.constant_mask [1] : vector<i1>
  return
}

// CHECK-LABEL: @constant_vector_mask
func.func @constant_vector_mask() {
  // CHECK: vector.constant_mask [3, 2] : vector<4x3xi1>
  %0 = vector.constant_mask [3, 2] : vector<4x3xi1>
  // CHECK: vector.constant_mask [0] : vector<[4]xi1>
  %1 = vector.constant_mask [0] : vector<[4]xi1>
  return
}

// CHECK-LABEL: @vector_print
func.func @vector_print(%arg0: vector<8x4xf32>) {
  // CHECK: vector.print %{{.*}} : vector<8x4xf32>
  vector.print %arg0 : vector<8x4xf32>
  return
}

// CHECK-LABEL: @reshape
func.func @reshape(%arg0 : vector<3x2x4xf32>) -> (vector<2x3x4xf32>) {
  // CHECK:      %[[C2:.*]] = arith.constant 2 : index
  %c2 = arith.constant 2 : index
  // CHECK:      %[[C3:.*]] = arith.constant 3 : index
  %c3 = arith.constant 3 : index
  // CHECK:      %[[C6:.*]] = arith.constant 6 : index
  %c6 = arith.constant 6 : index
  // CHECK:      %[[C9:.*]] = arith.constant 9 : index
  %c9 = arith.constant 9 : index
  // CHECK: vector.reshape %{{.*}}, [%[[C3]], %[[C6]]], [%[[C2]], %[[C9]]], [4] : vector<3x2x4xf32> to vector<2x3x4xf32>
  %1 = vector.reshape %arg0, [%c3, %c6], [%c2, %c9], [4]
    : vector<3x2x4xf32> to vector<2x3x4xf32>

  return %1 : vector<2x3x4xf32>
}

// CHECK-LABEL: @shape_cast
func.func @shape_cast(%arg0 : vector<5x1x3x2xf32>,
                 %arg1 : vector<8x1xf32>,
                 %arg2 : vector<16x1x1xf32>)
  -> (vector<15x2xf32>, vector<8xf32>, vector<16xf32>, vector<16x1xf32>) {

  // CHECK: vector.shape_cast %{{.*}} : vector<5x1x3x2xf32> to vector<15x2xf32>
  %0 = vector.shape_cast %arg0 : vector<5x1x3x2xf32> to vector<15x2xf32>

  // CHECK-NEXT: vector.shape_cast %{{.*}} : vector<8x1xf32> to vector<8xf32>
  %1 = vector.shape_cast %arg1 : vector<8x1xf32> to vector<8xf32>

  // CHECK-NEXT: vector.shape_cast %{{.*}} : vector<16x1x1xf32> to vector<16xf32>
  %2 = vector.shape_cast %arg2 : vector<16x1x1xf32> to vector<16xf32>

  // CHECK-NEXT: vector.shape_cast %{{.*}} : vector<16x1x1xf32> to vector<16x1xf32>
  %3 = vector.shape_cast %arg2 : vector<16x1x1xf32> to vector<16x1xf32>

  return %0, %1, %2, %3 : vector<15x2xf32>, vector<8xf32>, vector<16xf32>, vector<16x1xf32>
}

// CHECK-LABEL: @bitcast
func.func @bitcast(%arg0 : vector<5x1x3x2xf32>,
                 %arg1 : vector<8x1xi32>,
                 %arg2 : vector<16x1x8xi8>,
                 %arg3 : vector<8x2x1xindex>,
                 %arg4 : vector<f32>)
  -> (vector<5x1x3x4xf16>, vector<5x1x3x8xi8>, vector<8x4xi8>, vector<8x1xf32>, vector<16x1x2xi32>, vector<16x1x4xi16>, vector<16x1x1xindex>, vector<8x2x2xf32>, vector<i32>) {

  // CHECK: vector.bitcast %{{.*}} : vector<5x1x3x2xf32> to vector<5x1x3x4xf16>
  %0 = vector.bitcast %arg0 : vector<5x1x3x2xf32> to vector<5x1x3x4xf16>

  // CHECK-NEXT: vector.bitcast %{{.*}} : vector<5x1x3x2xf32> to vector<5x1x3x8xi8>
  %1 = vector.bitcast %arg0 : vector<5x1x3x2xf32> to vector<5x1x3x8xi8>

  // CHECK-NEXT: vector.bitcast %{{.*}} : vector<8x1xi32> to vector<8x4xi8>
  %2 = vector.bitcast %arg1 : vector<8x1xi32> to vector<8x4xi8>

  // CHECK-NEXT: vector.bitcast %{{.*}} : vector<8x1xi32> to vector<8x1xf32>
  %3 = vector.bitcast %arg1 : vector<8x1xi32> to vector<8x1xf32>

  // CHECK-NEXT: vector.bitcast %{{.*}} : vector<16x1x8xi8> to vector<16x1x2xi32>
  %4 = vector.bitcast %arg2 : vector<16x1x8xi8> to vector<16x1x2xi32>

  // CHECK-NEXT: vector.bitcast %{{.*}} : vector<16x1x8xi8> to vector<16x1x4xi16>
  %5 = vector.bitcast %arg2 : vector<16x1x8xi8> to vector<16x1x4xi16>

  // CHECK-NEXT: vector.bitcast %{{.*}} : vector<16x1x8xi8> to vector<16x1x1xindex>
  %6 = vector.bitcast %arg2 : vector<16x1x8xi8> to vector<16x1x1xindex>

  // CHECK-NEXT: vector.bitcast %{{.*}} : vector<8x2x1xindex> to vector<8x2x2xf32>
  %7 = vector.bitcast %arg3 : vector<8x2x1xindex> to vector<8x2x2xf32>

  // CHECK: vector.bitcast %{{.*}} : vector<f32> to vector<i32>
  %8 = vector.bitcast %arg4 : vector<f32> to vector<i32>

  return %0, %1, %2, %3, %4, %5, %6, %7, %8 : vector<5x1x3x4xf16>, vector<5x1x3x8xi8>, vector<8x4xi8>, vector<8x1xf32>, vector<16x1x2xi32>, vector<16x1x4xi16>, vector<16x1x1xindex>, vector<8x2x2xf32>, vector<i32>
}

// CHECK-LABEL: @vector_fma
func.func @vector_fma(%a: vector<8xf32>, %b: vector<8x4xf32>) {
  // CHECK: vector.fma %{{.*}} : vector<8xf32>
  vector.fma %a, %a, %a : vector<8xf32>
  // CHECK: vector.fma %{{.*}} : vector<8x4xf32>
  vector.fma %b, %b, %b : vector<8x4xf32>
  return
}

// CHECK-LABEL: @reduce_fp
func.func @reduce_fp(%arg0: vector<16xf32>, %arg1: f32) -> f32 {
  // CHECK:    vector.reduction <add>, %{{.*}} : vector<16xf32> into f32
  vector.reduction <add>, %arg0 : vector<16xf32> into f32
  // CHECK:    vector.reduction <add>, %{{.*}}, %{{.*}} : vector<16xf32> into f32
  vector.reduction <add>, %arg0, %arg1 : vector<16xf32> into f32
  // CHECK:    vector.reduction <mul>, %{{.*}} : vector<16xf32> into f32
  vector.reduction <mul>, %arg0 : vector<16xf32> into f32
  // CHECK:    vector.reduction <mul>, %{{.*}}, %{{.*}} : vector<16xf32> into f32
  vector.reduction <mul>, %arg0, %arg1 : vector<16xf32> into f32
  // CHECK:    vector.reduction <minf>, %{{.*}} : vector<16xf32> into f32
  vector.reduction <minf>, %arg0 : vector<16xf32> into f32
  // CHECK:    %[[X:.*]] = vector.reduction <maxf>, %{{.*}} : vector<16xf32> into f32
  %0 = vector.reduction <maxf>, %arg0 : vector<16xf32> into f32
  // CHECK:    return %[[X]] : f32
  return %0 : f32
}

// CHECK-LABEL: @reduce_int
func.func @reduce_int(%arg0: vector<16xi32>) -> i32 {
  // CHECK:    vector.reduction <add>, %{{.*}} : vector<16xi32> into i32
  vector.reduction <add>, %arg0 : vector<16xi32> into i32
  // CHECK:    vector.reduction <mul>, %{{.*}} : vector<16xi32> into i32
  vector.reduction <mul>, %arg0 : vector<16xi32> into i32
  // CHECK:    vector.reduction <minui>, %{{.*}} : vector<16xi32> into i32
  vector.reduction <minui>, %arg0 : vector<16xi32> into i32
  // CHECK:    vector.reduction <minsi>, %{{.*}} : vector<16xi32> into i32
  vector.reduction <minsi>, %arg0 : vector<16xi32> into i32
  // CHECK:    vector.reduction <maxui>, %{{.*}} : vector<16xi32> into i32
  vector.reduction <maxui>, %arg0 : vector<16xi32> into i32
  // CHECK:    vector.reduction <maxsi>, %{{.*}} : vector<16xi32> into i32
  vector.reduction <maxsi>, %arg0 : vector<16xi32> into i32
  // CHECK:    vector.reduction <and>, %{{.*}} : vector<16xi32> into i32
  vector.reduction <and>, %arg0 : vector<16xi32> into i32
  // CHECK:    vector.reduction <or>, %{{.*}} : vector<16xi32> into i32
  vector.reduction <or>, %arg0 : vector<16xi32> into i32
  // CHECK:    %[[X:.*]] = vector.reduction <xor>, %{{.*}} : vector<16xi32> into i32
  %0 = vector.reduction <xor>, %arg0 : vector<16xi32> into i32
  // CHECK:    return %[[X]] : i32
  return %0 : i32
}

// CHECK-LABEL: @transpose_fp
func.func @transpose_fp(%arg0: vector<3x7xf32>) -> vector<7x3xf32> {
  // CHECK: %[[X:.*]] = vector.transpose %{{.*}}, [1, 0] : vector<3x7xf32> to vector<7x3xf32>
  %0 = vector.transpose %arg0, [1, 0] : vector<3x7xf32> to vector<7x3xf32>
  // CHECK: return %[[X]] : vector<7x3xf32>
  return %0 : vector<7x3xf32>
}

// CHECK-LABEL: @transpose_int
func.func @transpose_int(%arg0: vector<11x7x3x2xi32>) -> vector<2x11x7x3xi32> {
  // CHECK: %[[X:.*]] = vector.transpose %{{.*}}, [3, 0, 1, 2] : vector<11x7x3x2xi32> to vector<2x11x7x3xi32>
  %0 = vector.transpose %arg0, [3, 0, 1, 2] : vector<11x7x3x2xi32> to vector<2x11x7x3xi32>
  // CHECK: return %[[X]] : vector<2x11x7x3xi32>
  return %0 : vector<2x11x7x3xi32>
}

// CHECK-LABEL: @flat_transpose_fp
func.func @flat_transpose_fp(%arg0: vector<16xf32>) -> vector<16xf32> {
  // CHECK: %[[X:.*]] = vector.flat_transpose %{{.*}} {columns = 4 : i32, rows = 4 : i32} : vector<16xf32> -> vector<16xf32>
  %0 = vector.flat_transpose %arg0 { rows = 4: i32, columns = 4: i32 } : vector<16xf32> -> vector<16xf32>
  // CHECK: return %[[X]] : vector<16xf32>
  return %0 : vector<16xf32>
}

// CHECK-LABEL: @flat_transpose_int
func.func @flat_transpose_int(%arg0: vector<16xi32>) -> vector<16xi32> {
  // CHECK: %[[X:.*]] = vector.flat_transpose %{{.*}} {columns = 8 : i32, rows = 2 : i32} : vector<16xi32> -> vector<16xi32>
  %0 = vector.flat_transpose %arg0 { rows = 2: i32, columns = 8: i32 } : vector<16xi32> -> vector<16xi32>
  // CHECK: return %[[X]] : vector<16xi32>
  return %0 : vector<16xi32>
}

// CHECK-LABEL: @vector_load_and_store_1d_scalar_memref
func.func @vector_load_and_store_1d_scalar_memref(%memref : memref<200x100xf32>,
                                             %i : index, %j : index) {
  // CHECK: %[[ld:.*]] = vector.load %{{.*}}[%{{.*}}] : memref<200x100xf32>, vector<8xf32>
  %0 = vector.load %memref[%i, %j] : memref<200x100xf32>, vector<8xf32>
  // CHECK: vector.store %[[ld]], %{{.*}}[%{{.*}}] : memref<200x100xf32>, vector<8xf32>
  vector.store %0, %memref[%i, %j] : memref<200x100xf32>, vector<8xf32>
  return
}

// CHECK-LABEL: @vector_load_and_store_1d_vector_memref
func.func @vector_load_and_store_1d_vector_memref(%memref : memref<200x100xvector<8xf32>>,
                                             %i : index, %j : index) {
  // CHECK: %[[ld:.*]] = vector.load %{{.*}}[%{{.*}}] : memref<200x100xvector<8xf32>>, vector<8xf32>
  %0 = vector.load %memref[%i, %j] : memref<200x100xvector<8xf32>>, vector<8xf32>
  // CHECK: vector.store %[[ld]], %{{.*}}[%{{.*}}] : memref<200x100xvector<8xf32>>, vector<8xf32>
  vector.store %0, %memref[%i, %j] : memref<200x100xvector<8xf32>>, vector<8xf32>
  return
}

// CHECK-LABEL: @vector_load_and_store_scalable_vector_memref
func.func @vector_load_and_store_scalable_vector_memref(%v: vector<[4]xi32>, %m: memref<?xi32>) -> vector<[4]xi32> {
  %c0 = arith.constant 0 : index
  // CHECK: vector.load {{.*}}: memref<?xi32>, vector<[4]xi32>
  %0 = vector.load %m[%c0] : memref<?xi32>, vector<[4]xi32>
  // CHECK: vector.store {{.*}}: memref<?xi32>, vector<[4]xi32>
  vector.store %v, %m[%c0] : memref<?xi32>, vector<[4]xi32>
  return %0 : vector<[4]xi32>
}

func.func @vector_load_and_store_1d_scalable_vector_memref(%memref : memref<200x100xvector<8xf32>>,
                                                      %i : index, %j : index) {
  // CHECK: %[[ld:.*]] = vector.load %{{.*}}[%{{.*}}] : memref<200x100xvector<8xf32>>, vector<8xf32>
  %0 = vector.load %memref[%i, %j] : memref<200x100xvector<8xf32>>, vector<8xf32>
  // CHECK: vector.store %[[ld]], %{{.*}}[%{{.*}}] : memref<200x100xvector<8xf32>>, vector<8xf32>
  vector.store %0, %memref[%i, %j] : memref<200x100xvector<8xf32>>, vector<8xf32>
  return
}

// CHECK-LABEL: @vector_load_and_store_out_of_bounds
func.func @vector_load_and_store_out_of_bounds(%memref : memref<7xf32>) {
  %c0 = arith.constant 0 : index
  // CHECK: %[[ld:.*]] = vector.load %{{.*}}[%{{.*}}] : memref<7xf32>, vector<8xf32>
  %0 = vector.load %memref[%c0] : memref<7xf32>, vector<8xf32>
  // CHECK: vector.store %[[ld]], %{{.*}}[%{{.*}}] : memref<7xf32>, vector<8xf32>
  vector.store %0, %memref[%c0] : memref<7xf32>, vector<8xf32>
  return
}

// CHECK-LABEL: @vector_load_and_store_2d_scalar_memref
func.func @vector_load_and_store_2d_scalar_memref(%memref : memref<200x100xf32>,
                                             %i : index, %j : index) {
  // CHECK: %[[ld:.*]] = vector.load %{{.*}}[%{{.*}}] : memref<200x100xf32>, vector<4x8xf32>
  %0 = vector.load %memref[%i, %j] : memref<200x100xf32>, vector<4x8xf32>
  // CHECK: vector.store %[[ld]], %{{.*}}[%{{.*}}] : memref<200x100xf32>, vector<4x8xf32>
  vector.store %0, %memref[%i, %j] : memref<200x100xf32>, vector<4x8xf32>
  return
}

// CHECK-LABEL: @vector_load_and_store_2d_vector_memref
func.func @vector_load_and_store_2d_vector_memref(%memref : memref<200x100xvector<4x8xf32>>,
                                             %i : index, %j : index) {
  // CHECK: %[[ld:.*]] = vector.load %{{.*}}[%{{.*}}] : memref<200x100xvector<4x8xf32>>, vector<4x8xf32>
  %0 = vector.load %memref[%i, %j] : memref<200x100xvector<4x8xf32>>, vector<4x8xf32>
  // CHECK: vector.store %[[ld]], %{{.*}}[%{{.*}}] : memref<200x100xvector<4x8xf32>>, vector<4x8xf32>
  vector.store %0, %memref[%i, %j] : memref<200x100xvector<4x8xf32>>, vector<4x8xf32>
  return
}

// CHECK-LABEL: @masked_load_and_store
func.func @masked_load_and_store(%base: memref<?xf32>, %mask: vector<16xi1>, %passthru: vector<16xf32>) {
  %c0 = arith.constant 0 : index
  // CHECK: %[[X:.*]] = vector.maskedload %{{.*}}[%{{.*}}], %{{.*}}, %{{.*}} : memref<?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  %0 = vector.maskedload %base[%c0], %mask, %passthru : memref<?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  // CHECK: vector.maskedstore %{{.*}}[%{{.*}}], %{{.*}}, %[[X]] : memref<?xf32>, vector<16xi1>, vector<16xf32>
  vector.maskedstore %base[%c0], %mask, %0 : memref<?xf32>, vector<16xi1>, vector<16xf32>
  return
}

// CHECK-LABEL: @masked_load_and_store2d
func.func @masked_load_and_store2d(%base: memref<?x?xf32>, %mask: vector<16xi1>, %passthru: vector<16xf32>) {
  %c0 = arith.constant 0 : index
  // CHECK: %[[X:.*]] = vector.maskedload %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}}, %{{.*}} : memref<?x?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  %0 = vector.maskedload %base[%c0, %c0], %mask, %passthru : memref<?x?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  // CHECK: vector.maskedstore %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}}, %[[X]] : memref<?x?xf32>, vector<16xi1>, vector<16xf32>
  vector.maskedstore %base[%c0, %c0], %mask, %0 : memref<?x?xf32>, vector<16xi1>, vector<16xf32>
  return
}

// CHECK-LABEL: @gather_and_scatter
func.func @gather_and_scatter(%base: memref<?xf32>, %v: vector<16xi32>, %mask: vector<16xi1>, %pass_thru: vector<16xf32>) {
  %c0 = arith.constant 0 : index
  // CHECK: %[[X:.*]] = vector.gather %{{.*}}[%{{.*}}] [%{{.*}}], %{{.*}}, %{{.*}} : memref<?xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  %0 = vector.gather %base[%c0][%v], %mask, %pass_thru : memref<?xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  // CHECK: vector.scatter %{{.*}}[%{{.*}}] [%{{.*}}], %{{.*}}, %[[X]] : memref<?xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32>
  vector.scatter %base[%c0][%v], %mask, %0 : memref<?xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32>
  return
}

// CHECK-LABEL: @gather_and_scatter2d
func.func @gather_and_scatter2d(%base: memref<?x?xf32>, %v: vector<16xi32>, %mask: vector<16xi1>, %pass_thru: vector<16xf32>) {
  %c0 = arith.constant 0 : index
  // CHECK: %[[X:.*]] = vector.gather %{{.*}}[%{{.*}}, %{{.*}}] [%{{.*}}], %{{.*}}, %{{.*}} : memref<?x?xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  %0 = vector.gather %base[%c0, %c0][%v], %mask, %pass_thru : memref<?x?xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  // CHECK: vector.scatter %{{.*}}[%{{.*}}] [%{{.*}}], %{{.*}}, %[[X]] : memref<?x?xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32>
  vector.scatter %base[%c0, %c0][%v], %mask, %0 : memref<?x?xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32>
  return
}

// CHECK-LABEL: @expand_and_compress
func.func @expand_and_compress(%base: memref<?xf32>, %mask: vector<16xi1>, %pass_thru: vector<16xf32>) {
  %c0 = arith.constant 0 : index
  // CHECK: %[[X:.*]] = vector.expandload %{{.*}}[%{{.*}}], %{{.*}}, %{{.*}} : memref<?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  %0 = vector.expandload %base[%c0], %mask, %pass_thru : memref<?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  // CHECK: vector.compressstore %{{.*}}[%{{.*}}], %{{.*}}, %[[X]] : memref<?xf32>, vector<16xi1>, vector<16xf32>
  vector.compressstore %base[%c0], %mask, %0 : memref<?xf32>, vector<16xi1>, vector<16xf32>
  return
}

// CHECK-LABEL: @expand_and_compress2d
func.func @expand_and_compress2d(%base: memref<?x?xf32>, %mask: vector<16xi1>, %pass_thru: vector<16xf32>) {
  %c0 = arith.constant 0 : index
  // CHECK: %[[X:.*]] = vector.expandload %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}}, %{{.*}} : memref<?x?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  %0 = vector.expandload %base[%c0, %c0], %mask, %pass_thru : memref<?x?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
  // CHECK: vector.compressstore %{{.*}}[%{{.*}}, %{{.*}}], %{{.*}}, %[[X]] : memref<?x?xf32>, vector<16xi1>, vector<16xf32>
  vector.compressstore %base[%c0, %c0], %mask, %0 : memref<?x?xf32>, vector<16xi1>, vector<16xf32>
  return
}

// CHECK-LABEL: @extract_insert_map
func.func @extract_insert_map(%v: vector<32xf32>, %v2: vector<16x32xf32>,
  %id0 : index, %id1 : index) -> (vector<32xf32>, vector<16x32xf32>) {
  // CHECK: %[[V:.*]] = vector.extract_map %{{.*}}[%{{.*}}] : vector<32xf32> to vector<2xf32>
  %vd = vector.extract_map %v[%id0] : vector<32xf32> to vector<2xf32>
  // CHECK: %[[V1:.*]] = vector.extract_map %{{.*}}[%{{.*}}, %{{.*}}] : vector<16x32xf32> to vector<4x2xf32>
  %vd2 = vector.extract_map %v2[%id0, %id1] : vector<16x32xf32> to vector<4x2xf32>
  // CHECK: %[[R:.*]] = vector.insert_map %[[V]], %{{.*}}[%{{.*}}] : vector<2xf32> into vector<32xf32>
  %r = vector.insert_map %vd, %v[%id0] : vector<2xf32> into vector<32xf32>
  // CHECK: %[[R1:.*]] = vector.insert_map %[[V1]], %{{.*}}[%{{.*}}, %{{.*}}] : vector<4x2xf32> into vector<16x32xf32>
  %r2 = vector.insert_map %vd2, %v2[%id0, %id1] : vector<4x2xf32> into vector<16x32xf32>
  // CHECK: return %[[R]], %[[R1]] : vector<32xf32>, vector<16x32xf32>
  return %r, %r2 : vector<32xf32>, vector<16x32xf32>
}

// CHECK-LABEL: @multi_reduction
func.func @multi_reduction(%0: vector<4x8x16x32xf32>) -> f32 {
  %1 = vector.multi_reduction <add>, %0 [1, 3] :
    vector<4x8x16x32xf32> to vector<4x16xf32>
  %2 = vector.multi_reduction <add>, %1 [0, 1] :
    vector<4x16xf32> to f32
  return %2 : f32
}

// CHECK-LABEL: @get_vector_scale
func.func @get_vector_scale() -> index {
  // CHECK: vector.vscale
  %0 = vector.vscale
  return %0 : index
}

// CHECK-LABEL: @vector_scan
func.func @vector_scan(%0: vector<4x8x16x32xf32>) -> vector<4x8x16x32xf32> {
  %1 = arith.constant dense<0.0> : vector<4x16x32xf32>
  %2:2 = vector.scan <add>, %0, %1 {reduction_dim = 1 : i64, inclusive = true} :
    vector<4x8x16x32xf32>, vector<4x16x32xf32>
  return %2#0 : vector<4x8x16x32xf32>
}

// CHECK-LABEL: func @test_splat_op
// CHECK-SAME: [[S:%arg[0-9]+]]: f32
func.func @test_splat_op(%s : f32) {
  // CHECK: vector.splat [[S]] : vector<8xf32>
  %v = vector.splat %s : vector<8xf32>
  
  // CHECK: vector.splat [[S]] : vector<4xf32>
  %u = "vector.splat"(%s) : (f32) -> vector<4xf32>
  return
}

// CHECK-LABEL: func @vector_splat_0d(
func.func @vector_splat_0d(%a: f32) -> vector<f32> {
  // CHECK: vector.splat %{{.*}} : vector<f32>
  %0 = vector.splat %a : vector<f32>
  return %0 : vector<f32>
}

// CHECK-LABEL:   func @warp_execute_on_lane_0(
func.func @warp_execute_on_lane_0(%laneid: index) {
//  CHECK-NEXT:     vector.warp_execute_on_lane_0(%{{.*}})[32] {
  vector.warp_execute_on_lane_0(%laneid)[32] {
//  CHECK-NEXT:     }
  }
//  CHECK-NEXT:     return
  return
}

// CHECK-LABEL:   func @warp_operand_result(
func.func @warp_operand_result(%laneid: index, %v0 : vector<4xi32>) -> (vector<4xi32>) {
//  CHECK-NEXT:     %{{.*}} = vector.warp_execute_on_lane_0(%{{.*}})[32] args(%{{.*}} : vector<4xi32>) -> (vector<4xi32>) {
  %2 = vector.warp_execute_on_lane_0(%laneid)[32]
  args(%v0 : vector<4xi32>) -> (vector<4xi32>) {
   ^bb0(%arg0 : vector<128xi32>) :
    %0 = arith.constant dense<2>: vector<128xi32>
    %1 = arith.addi %arg0, %0 : vector<128xi32>
//       CHECK:       vector.yield %{{.*}} : vector<128xi32>
    vector.yield %1 : vector<128xi32>
//  CHECK-NEXT:     }
  }
  return %2 : vector<4xi32>
}


