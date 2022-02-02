// RUN: mlir-opt %s -canonicalize --split-input-file -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func @subview_of_size_memcast
//  CHECK-SAME:   %[[ARG0:.[a-z0-9A-Z_]+]]: memref<4x6x16x32xi8>
//       CHECK:   %[[S:.+]] = memref.subview %[[ARG0]][0, 1, 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] : memref<4x6x16x32xi8> to memref<16x32xi8, #{{.*}}>
//       CHECK:   %[[M:.+]] = memref.cast %[[S]] : memref<16x32xi8, #{{.*}}> to memref<16x32xi8, #{{.*}}>
//       CHECK:   return %[[M]] : memref<16x32xi8, #{{.*}}>
func @subview_of_size_memcast(%arg : memref<4x6x16x32xi8>) ->
  memref<16x32xi8, affine_map<(d0, d1)[s0] -> (d0 * 32 + d1 + s0)>>{
  %0 = memref.cast %arg : memref<4x6x16x32xi8> to memref<?x?x16x32xi8>
  %1 = memref.subview %0[0, 1, 0, 0] [1, 1, 16, 32] [1, 1, 1, 1] :
    memref<?x?x16x32xi8> to
    memref<16x32xi8, affine_map<(d0, d1)[s0] -> (d0 * 32 + d1 + s0)>>
  return %1 : memref<16x32xi8, affine_map<(d0, d1)[s0] -> (d0 * 32 + d1 + s0)>>
}

// -----

//   CHECK-DAG: #[[MAP0:[0-9a-z]+]] = affine_map<(d0, d1)[s0] -> (d0 * 7 + s0 + d1)>
//   CHECK-DAG: #[[MAP1:[0-9a-z]+]] = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>
#map0 = affine_map<(d0, d1, d2)[s0] -> (d0 * 35 + s0 + d1 * 7 + d2)>
#map1 = affine_map<(d0, d1, d2)[s0, s1, s2, s3] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3)>
#map2 = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>

//       CHECK: func @subview_of_strides_memcast
//  CHECK-SAME:   %[[ARG0:.[a-z0-9A-Z_]+]]: memref<1x1x?xf32, #{{.*}}>
//       CHECK:   %[[S:.+]] = memref.subview %[[ARG0]][0, 0, 0] [1, 1, 4]
//  CHECK-SAME:                    to memref<1x4xf32, #[[MAP0]]>
//       CHECK:   %[[M:.+]] = memref.cast %[[S]]
//  CHECK-SAME:                    to memref<1x4xf32, #[[MAP1]]>
//       CHECK:   return %[[M]]
func @subview_of_strides_memcast(%arg : memref<1x1x?xf32, #map0>) -> memref<1x4xf32, #map2> {
  %0 = memref.cast %arg : memref<1x1x?xf32, #map0> to memref<1x1x?xf32, #map1>
  %1 = memref.subview %0[0, 0, 0] [1, 1, 4] [1, 1, 1] : memref<1x1x?xf32, #map1> to memref<1x4xf32, #map2>
  return %1 : memref<1x4xf32, #map2>
}

// -----

// CHECK-LABEL: func @subview_of_static_full_size
// CHECK-SAME: %[[ARG0:.+]]: memref<4x6x16x32xi8>
// CHECK-NOT: memref.subview
// CHECK: return %[[ARG0]] : memref<4x6x16x32xi8>
func @subview_of_static_full_size(%arg0 : memref<4x6x16x32xi8>) -> memref<4x6x16x32xi8> {
  %0 = memref.subview %arg0[0, 0, 0, 0] [4, 6, 16, 32] [1, 1, 1, 1] : memref<4x6x16x32xi8> to memref<4x6x16x32xi8>
  return %0 : memref<4x6x16x32xi8>
}

// -----

#map0 = affine_map<(d0, d1, d2)[s0, s1, s2, s3] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3)>
func @subview_canonicalize(%arg0 : memref<?x?x?xf32>, %arg1 : index,
    %arg2 : index) -> memref<?x?x?xf32, #map0>
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %0 = memref.subview %arg0[%c0, %arg1, %c1] [%c4, %c1, %arg2] [%c1, %c1, %c1] : memref<?x?x?xf32> to memref<?x?x?xf32, #map0>
  return %0 : memref<?x?x?xf32, #map0>
}
// CHECK-LABEL: func @subview_canonicalize
//  CHECK-SAME:   %[[ARG0:.+]]: memref<?x?x?xf32>
//       CHECK:   %[[SUBVIEW:.+]] = memref.subview %[[ARG0]][0, %{{[a-zA-Z0-9_]+}}, 1]
//  CHECK-SAME:      [4, 1, %{{[a-zA-Z0-9_]+}}] [1, 1, 1]
//  CHECK-SAME:      : memref<?x?x?xf32> to memref<4x1x?xf32
//       CHECK:   %[[RESULT:.+]] = memref.cast %[[SUBVIEW]]
//       CHEKC:   return %[[RESULT]]

// -----

#map0 = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
func @rank_reducing_subview_canonicalize(%arg0 : memref<?x?x?xf32>, %arg1 : index,
    %arg2 : index) -> memref<?x?xf32, #map0>
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %0 = memref.subview %arg0[%c0, %arg1, %c1] [%c4, 1, %arg2] [%c1, %c1, %c1] : memref<?x?x?xf32> to memref<?x?xf32, #map0>
  return %0 : memref<?x?xf32, #map0>
}
// CHECK-LABEL: func @rank_reducing_subview_canonicalize
//  CHECK-SAME:   %[[ARG0:.+]]: memref<?x?x?xf32>
//       CHECK:   %[[SUBVIEW:.+]] = memref.subview %[[ARG0]][0, %{{[a-zA-Z0-9_]+}}, 1]
//  CHECK-SAME:      [4, 1, %{{[a-zA-Z0-9_]+}}] [1, 1, 1]
//  CHECK-SAME:      : memref<?x?x?xf32> to memref<4x?xf32
//       CHECK:   %[[RESULT:.+]] = memref.cast %[[SUBVIEW]]
//       CHECK:   return %[[RESULT]]

// -----

func @multiple_reducing_dims(%arg0 : memref<1x384x384xf32>,
    %arg1 : index, %arg2 : index, %arg3 : index) -> memref<?xf32, offset: ?, strides: [1]>
{
  %c1 = arith.constant 1 : index
  %0 = memref.subview %arg0[0, %arg1, %arg2] [1, %c1, %arg3] [1, 1, 1] : memref<1x384x384xf32> to memref<?x?xf32, offset: ?, strides: [384, 1]>
  %1 = memref.subview %0[0, 0] [1, %arg3] [1, 1] : memref<?x?xf32, offset: ?, strides: [384, 1]> to memref<?xf32, offset: ?, strides: [1]>
  return %1 : memref<?xf32, offset: ?, strides: [1]>
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1)[s0] -> (d0 * 384 + s0 + d1)>
//       CHECK: func @multiple_reducing_dims
//       CHECK:   %[[REDUCED1:.+]] = memref.subview %{{.+}}[0, %{{.+}}, %{{.+}}] [1, 1, %{{.+}}] [1, 1, 1]
//  CHECK-SAME:       : memref<1x384x384xf32> to memref<1x?xf32, #[[MAP1]]>
//       CHECK:   %[[REDUCED2:.+]] = memref.subview %[[REDUCED1]][0, 0] [1, %{{.+}}] [1, 1]
//  CHECK-SAME:       : memref<1x?xf32, #[[MAP1]]> to memref<?xf32, #[[MAP0]]>

// -----

func @multiple_reducing_dims_dynamic(%arg0 : memref<?x?x?xf32>,
    %arg1 : index, %arg2 : index, %arg3 : index) -> memref<?xf32, offset: ?, strides: [1]>
{
  %c1 = arith.constant 1 : index
  %0 = memref.subview %arg0[0, %arg1, %arg2] [1, %c1, %arg3] [1, 1, 1] : memref<?x?x?xf32> to memref<?x?xf32, offset: ?, strides: [?, 1]>
  %1 = memref.subview %0[0, 0] [1, %arg3] [1, 1] : memref<?x?xf32, offset: ?, strides: [?, 1]> to memref<?xf32, offset: ?, strides: [1]>
  return %1 : memref<?xf32, offset: ?, strides: [1]>
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
//       CHECK: func @multiple_reducing_dims_dynamic
//       CHECK:   %[[REDUCED1:.+]] = memref.subview %{{.+}}[0, %{{.+}}, %{{.+}}] [1, 1, %{{.+}}] [1, 1, 1]
//  CHECK-SAME:       : memref<?x?x?xf32> to memref<1x?xf32, #[[MAP1]]>
//       CHECK:   %[[REDUCED2:.+]] = memref.subview %[[REDUCED1]][0, 0] [1, %{{.+}}] [1, 1]
//  CHECK-SAME:       : memref<1x?xf32, #[[MAP1]]> to memref<?xf32, #[[MAP0]]>

// -----

func @multiple_reducing_dims_all_dynamic(%arg0 : memref<?x?x?xf32, offset: ?, strides: [?, ?, ?]>,
    %arg1 : index, %arg2 : index, %arg3 : index) -> memref<?xf32, offset: ?, strides: [?]>
{
  %c1 = arith.constant 1 : index
  %0 = memref.subview %arg0[0, %arg1, %arg2] [1, %c1, %arg3] [1, 1, 1]
      : memref<?x?x?xf32, offset: ?, strides: [?, ?, ?]> to memref<?x?xf32, offset: ?, strides: [?, ?]>
  %1 = memref.subview %0[0, 0] [1, %arg3] [1, 1] : memref<?x?xf32, offset: ?, strides: [?, ?]> to memref<?xf32, offset: ?, strides: [?]>
  return %1 : memref<?xf32, offset: ?, strides: [?]>
}
//   CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>
//   CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1)[s0, s1, s2] -> (d0 * s1 + s0 + d1 * s2)>
//   CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2)[s0, s1, s2, s3] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3)>
//       CHECK: func @multiple_reducing_dims_all_dynamic
//       CHECK:   %[[REDUCED1:.+]] = memref.subview %{{.+}}[0, %{{.+}}, %{{.+}}] [1, 1, %{{.+}}] [1, 1, 1]
//  CHECK-SAME:       : memref<?x?x?xf32, #[[MAP2]]> to memref<1x?xf32, #[[MAP1]]>
//       CHECK:   %[[REDUCED2:.+]] = memref.subview %[[REDUCED1]][0, 0] [1, %{{.+}}] [1, 1]
//  CHECK-SAME:       : memref<1x?xf32, #[[MAP1]]> to memref<?xf32, #[[MAP0]]>


// -----

// CHECK-LABEL: func @dim_of_sized_view
//  CHECK-SAME:   %{{[a-z0-9A-Z_]+}}: memref<?xi8>
//  CHECK-SAME:   %[[SIZE:.[a-z0-9A-Z_]+]]: index
//       CHECK:   return %[[SIZE]] : index
func @dim_of_sized_view(%arg : memref<?xi8>, %size: index) -> index {
  %c0 = arith.constant 0 : index
  %0 = memref.reinterpret_cast %arg to offset: [0], sizes: [%size], strides: [1] : memref<?xi8> to memref<?xi8>
  %1 = memref.dim %0, %c0 : memref<?xi8>
  return %1 : index
}

// -----

// CHECK-LABEL: func @no_fold_of_store
//  CHECK:   %[[cst:.+]] = memref.cast %arg
//  CHECK:   memref.store %[[cst]]
func @no_fold_of_store(%arg : memref<32xi8>, %holder: memref<memref<?xi8>>) {
  %0 = memref.cast %arg : memref<32xi8> to memref<?xi8>
  memref.store %0, %holder[] : memref<memref<?xi8>>
  return
}

// -----

// Test case: Folding of memref.dim(memref.alloca(%size), %idx) -> %size
// CHECK-LABEL: func @dim_of_alloca(
//  CHECK-SAME:     %[[SIZE:[0-9a-z]+]]: index
//  CHECK-NEXT:   return %[[SIZE]] : index
func @dim_of_alloca(%size: index) -> index {
  %0 = memref.alloca(%size) : memref<?xindex>
  %c0 = arith.constant 0 : index
  %1 = memref.dim %0, %c0 : memref<?xindex>
  return %1 : index
}

// -----

// Test case: Folding of memref.dim(memref.alloca(rank(%v)), %idx) -> rank(%v)
// CHECK-LABEL: func @dim_of_alloca_with_dynamic_size(
//  CHECK-SAME:     %[[MEM:[0-9a-z]+]]: memref<*xf32>
//  CHECK-NEXT:   %[[RANK:.*]] = memref.rank %[[MEM]] : memref<*xf32>
//  CHECK-NEXT:   return %[[RANK]] : index
func @dim_of_alloca_with_dynamic_size(%arg0: memref<*xf32>) -> index {
  %0 = memref.rank %arg0 : memref<*xf32>
  %1 = memref.alloca(%0) : memref<?xindex>
  %c0 = arith.constant 0 : index
  %2 = memref.dim %1, %c0 : memref<?xindex>
  return %2 : index
}

// -----

// Test case: Folding of memref.dim(memref.reshape %v %shp, %idx) -> memref.load %shp[%idx]
// CHECK-LABEL: func @dim_of_memref_reshape(
//  CHECK-SAME:     %[[MEM:[0-9a-z]+]]: memref<*xf32>,
//  CHECK-SAME:     %[[SHP:[0-9a-z]+]]: memref<?xindex>
//  CHECK-NEXT:   %[[IDX:.*]] = arith.constant 3
//  CHECK-NEXT:   %[[DIM:.*]] = memref.load %[[SHP]][%[[IDX]]]
//  CHECK-NEXT:   memref.store
//   CHECK-NOT:   memref.dim
//       CHECK:   return %[[DIM]] : index
func @dim_of_memref_reshape(%arg0: memref<*xf32>, %arg1: memref<?xindex>)
    -> index {
  %c3 = arith.constant 3 : index
  %0 = memref.reshape %arg0(%arg1)
      : (memref<*xf32>, memref<?xindex>) -> memref<*xf32>
  // Update the shape to test that he load ends up in the right place.
  memref.store %c3, %arg1[%c3] : memref<?xindex>
  %1 = memref.dim %0, %c3 : memref<*xf32>
  return %1 : index
}

// -----

// Test case: Folding of memref.dim(memref.reshape %v %shp, %idx) -> memref.load %shp[%idx]
// CHECK-LABEL: func @dim_of_memref_reshape_i32(
//  CHECK-SAME:     %[[MEM:[0-9a-z]+]]: memref<*xf32>,
//  CHECK-SAME:     %[[SHP:[0-9a-z]+]]: memref<?xi32>
//  CHECK-NEXT:   %[[IDX:.*]] = arith.constant 3
//  CHECK-NEXT:   %[[DIM:.*]] = memref.load %[[SHP]][%[[IDX]]]
//  CHECK-NEXT:   %[[CAST:.*]] = arith.index_cast %[[DIM]]
//   CHECK-NOT:   memref.dim
//       CHECK:   return %[[CAST]] : index
func @dim_of_memref_reshape_i32(%arg0: memref<*xf32>, %arg1: memref<?xi32>)
    -> index {
  %c3 = arith.constant 3 : index
  %0 = memref.reshape %arg0(%arg1)
      : (memref<*xf32>, memref<?xi32>) -> memref<*xf32>
  %1 = memref.dim %0, %c3 : memref<*xf32>
  return %1 : index
}

// -----

// CHECK-LABEL: func @alloc_const_fold
func @alloc_const_fold() -> memref<?xf32> {
  // CHECK-NEXT: %0 = memref.alloc() : memref<4xf32>
  %c4 = arith.constant 4 : index
  %a = memref.alloc(%c4) : memref<?xf32>

  // CHECK-NEXT: %1 = memref.cast %0 : memref<4xf32> to memref<?xf32>
  // CHECK-NEXT: return %1 : memref<?xf32>
  return %a : memref<?xf32>
}

// -----

// CHECK-LABEL: func @alloc_alignment_const_fold
func @alloc_alignment_const_fold() -> memref<?xf32> {
  // CHECK-NEXT: %0 = memref.alloc() {alignment = 4096 : i64} : memref<4xf32>
  %c4 = arith.constant 4 : index
  %a = memref.alloc(%c4) {alignment = 4096 : i64} : memref<?xf32>

  // CHECK-NEXT: %1 = memref.cast %0 : memref<4xf32> to memref<?xf32>
  // CHECK-NEXT: return %1 : memref<?xf32>
  return %a : memref<?xf32>
}

// -----

// CHECK-LABEL: func @alloc_const_fold_with_symbols1(
//  CHECK: %[[c1:.+]] = arith.constant 1 : index
//  CHECK: %[[mem1:.+]] = memref.alloc({{.*}})[%[[c1]], %[[c1]]] : memref<?xi32, #map>
//  CHECK: return %[[mem1]] : memref<?xi32, #map>
#map0 = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>
func @alloc_const_fold_with_symbols1(%arg0 : index) -> memref<?xi32, #map0> {
  %c1 = arith.constant 1 : index
  %0 = memref.alloc(%arg0)[%c1, %c1] : memref<?xi32, #map0>
  return %0 : memref<?xi32, #map0>
}

// -----

// CHECK-LABEL: func @alloc_const_fold_with_symbols2(
//  CHECK: %[[c1:.+]] = arith.constant 1 : index
//  CHECK: %[[mem1:.+]] = memref.alloc()[%[[c1]], %[[c1]]] : memref<1xi32, #map>
//  CHECK: %[[mem2:.+]] = memref.cast %[[mem1]] : memref<1xi32, #map> to memref<?xi32, #map>
//  CHECK: return %[[mem2]] : memref<?xi32, #map>
#map0 = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>
func @alloc_const_fold_with_symbols2() -> memref<?xi32, #map0> {
  %c1 = arith.constant 1 : index
  %0 = memref.alloc(%c1)[%c1, %c1] : memref<?xi32, #map0>
  return %0 : memref<?xi32, #map0>
}

// -----
// CHECK-LABEL: func @allocator
// CHECK:   %[[alloc:.+]] = memref.alloc
// CHECK:   memref.store %[[alloc:.+]], %arg0
func @allocator(%arg0 : memref<memref<?xi32>>, %arg1 : index)  {
  %0 = memref.alloc(%arg1) : memref<?xi32>
  memref.store %0, %arg0[] : memref<memref<?xi32>>
  return
}

// -----

func @collapsing_memref_reshapes_to_zero_dim(%arg0 : memref<1x1x1xf32>)
                                             -> memref<f32> {
  %0 = memref.collapse_shape %arg0 [[0, 1, 2]]
      : memref<1x1x1xf32> into memref<1xf32>
  %1 = memref.collapse_shape %0 [] : memref<1xf32> into memref<f32>
  return %1 : memref<f32>
}
// CHECK-LABEL: collapsing_memref_reshapes_to_zero
//       CHECK:   memref.collapse_shape %{{.*}} []
//  CHECK-SAME:     memref<1x1x1xf32> into memref<f32>

// -----

func @collapsing_memref_reshapes(%arg0 : memref<?x?x?x?x?xf32>)
    -> memref<?x?xf32> {
  %0 = memref.collapse_shape %arg0 [[0, 1], [2], [3, 4]]
      : memref<?x?x?x?x?xf32> into memref<?x?x?xf32>
  %1 = memref.collapse_shape %0 [[0, 1], [2]]
      : memref<?x?x?xf32> into memref<?x?xf32>
  return %1 : memref<?x?xf32>
}
// CHECK-LABEL: collapsing_memref_reshapes
//       CHECK:   memref.collapse_shape %{{.*}} {{\[}}[0, 1, 2], [3, 4]]
//   CHECK-NOT:   memref.collapse_shape

// -----

func @expanding_memref_reshapes(%arg0 : memref<?x?xf32>)
    -> memref<?x6x4x5x?xf32> {
  %0 = memref.expand_shape %arg0 [[0, 1], [2]]
      : memref<?x?xf32> into memref<?x4x?xf32>
  %1 = memref.expand_shape %0 [[0, 1], [2], [3, 4]]
      : memref<?x4x?xf32> into memref<?x6x4x5x?xf32>
  return %1 : memref<?x6x4x5x?xf32>
}
// CHECK-LABEL: expanding_memref_reshapes
//       CHECK:   memref.expand_shape %{{.*}} {{\[}}[0, 1, 2], [3, 4]]
//   CHECK-NOT:   memref.expand_shape

// -----

func @expanding_memref_reshapes_to_zero_dim(%arg0 : memref<f32>)
                                             -> memref<1x1x1xf32> {
  %0 = memref.expand_shape %arg0 [] : memref<f32> into memref<1xf32>
  %1 = memref.expand_shape %0 [[0, 1, 2]]
      : memref<1xf32> into memref<1x1x1xf32>
  return %1 : memref<1x1x1xf32>
}
// CHECK-LABEL: expanding_memref_reshapes_to_zero
//       CHECK:   memref.expand_shape %{{.*}} []
//  CHECK-SAME:     memref<f32> into memref<1x1x1xf32>

// -----

func @fold_memref_reshape(%arg0 : memref<12x4xf32>) -> memref<12x4xf32> {
  %0 = memref.expand_shape %arg0 [[0, 1], [2]]
      : memref<12x4xf32> into memref<3x4x4xf32>
  %1 = memref.collapse_shape %0 [[0, 1], [2]]
      : memref<3x4x4xf32> into memref<12x4xf32>
  return %1 : memref<12x4xf32>
}
// CHECK-LABEL: @fold_memref_reshape
//   CHECK-NOT:   linalg.{{.*}}_shape

// -----

func @fold_memref_reshape_dynamic(%arg0 : memref<?x?xf32>) -> memref<?x?xf32> {
  %0 = memref.expand_shape %arg0 [[0, 1], [2]]
      : memref<?x?xf32> into memref<?x4x?xf32>
  %1 = memref.collapse_shape %0 [[0, 1], [2]]
      : memref<?x4x?xf32> into memref<?x?xf32>
  return %1 : memref<?x?xf32>
}
// CHECK-LABEL: @fold_memref_reshape_dynamic
//   CHECK-NOT:   linalg.{{.*}}_shape

// -----

func @fold_memref_expand_cast(%arg0 : memref<?x?xf32>) -> memref<2x4x4xf32> {
  %0 = memref.cast %arg0 : memref<?x?xf32> to memref<8x4xf32>
  %1 = memref.expand_shape %0 [[0, 1], [2]]
      : memref<8x4xf32> into memref<2x4x4xf32>
  return %1 : memref<2x4x4xf32>
}

// CHECK-LABEL: @fold_memref_expand_cast
// CHECK: memref.expand_shape

// -----

// CHECK-LABEL:   func @collapse_after_memref_cast_type_change(
// CHECK-SAME:      %[[INPUT:.*]]: memref<?x512x1x1xf32>) -> memref<?x?xf32> {
// CHECK:           %[[COLLAPSED:.*]] = memref.collapse_shape %[[INPUT]]
// CHECK-SAME:         {{\[\[}}0], [1, 2, 3]] : memref<?x512x1x1xf32> into memref<?x512xf32>
// CHECK:           %[[DYNAMIC:.*]] = memref.cast %[[COLLAPSED]] :
// CHECK-SAME:         memref<?x512xf32> to memref<?x?xf32>
// CHECK:           return %[[DYNAMIC]] : memref<?x?xf32>
// CHECK:         }
func @collapse_after_memref_cast_type_change(%arg0 : memref<?x512x1x1xf32>) -> memref<?x?xf32> {
  %dynamic = memref.cast %arg0: memref<?x512x1x1xf32> to memref<?x?x?x?xf32>
  %collapsed = memref.collapse_shape %dynamic [[0], [1, 2, 3]] : memref<?x?x?x?xf32> into memref<?x?xf32>
  return %collapsed : memref<?x?xf32>
}

// CHECK-LABEL:   func @collapse_after_memref_cast(
// CHECK-SAME:      %[[INPUT:.*]]: memref<?x512x1x?xf32>) -> memref<?x?xf32> {
// CHECK:           %[[COLLAPSED:.*]] = memref.collapse_shape %[[INPUT]]
// CHECK_SAME:        {{\[\[}}0], [1, 2, 3]] : memref<?x512x1x?xf32> into memref<?x?xf32>
// CHECK:           return %[[COLLAPSED]] : memref<?x?xf32>
func @collapse_after_memref_cast(%arg0 : memref<?x512x1x?xf32>) -> memref<?x?xf32> {
  %dynamic = memref.cast %arg0: memref<?x512x1x?xf32> to memref<?x?x?x?xf32>
  %collapsed = memref.collapse_shape %dynamic [[0], [1, 2, 3]] : memref<?x?x?x?xf32> into memref<?x?xf32>
  return %collapsed : memref<?x?xf32>
}

// -----

func @reduced_memref(%arg0: memref<2x5x7x1xf32>, %arg1 :index)
    -> memref<1x4x1xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 35 + s0 + d1 * 7 + d2)>> {
  %c0 = arith.constant 0 : index
  %c5 = arith.constant 5 : index
  %c4 = arith.constant 4 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %0 = memref.subview %arg0[%arg1, %arg1, %arg1, 0] [%c1, %c4, %c1, 1] [1, 1, 1, 1]
      : memref<2x5x7x1xf32> to memref<?x?x?xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 35 + s0 + d1 * 7 + d2)>>
  %1 = memref.cast %0
      : memref<?x?x?xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 35 + s0 + d1 * 7 + d2)>> to
        memref<1x4x1xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 35 + s0 + d1 * 7 + d2)>>
  return %1 : memref<1x4x1xf32, affine_map<(d0, d1, d2)[s0] -> (d0 * 35 + s0 + d1 * 7 + d2)>>
}

// CHECK-LABEL: func @reduced_memref
//       CHECK:   %[[RESULT:.+]] = memref.subview
//  CHECK-SAME:       memref<2x5x7x1xf32> to memref<1x4x1xf32, #{{.+}}>
//       CHECK:   return %[[RESULT]]

// -----

// CHECK-LABEL: func @fold_rank_memref
func @fold_rank_memref(%arg0 : memref<?x?xf32>) -> (index) {
  // Fold a rank into a constant
  // CHECK-NEXT: [[C2:%.+]] = arith.constant 2 : index
  %rank_0 = memref.rank %arg0 : memref<?x?xf32>

  // CHECK-NEXT: return [[C2]]
  return %rank_0 : index
}

// -----

#map = affine_map<(d0, d1) -> (d0 * 42 + d1)>
func @fold_no_op_subview(%arg0 : memref<20x42xf32>) -> memref<20x42xf32, #map> {
  %0 = memref.subview %arg0[0, 0] [20, 42] [1, 1] : memref<20x42xf32> to memref<20x42xf32, #map>
  return %0 : memref<20x42xf32, #map>
}
// CHECK-LABEL: func @fold_no_op_subview(
//       CHECK:   %[[ARG0:.+]]: memref<20x42xf32>)
//       CHECK:   %[[CAST:.+]] = memref.cast %[[ARG0]]
//       CHECK:   return %[[CAST]]

// -----

#map = affine_map<(d0, d1) -> (d0 * 42 + d1 + 1)>
func @no_fold_subview_with_non_zero_offset(%arg0 : memref<20x42xf32>) -> memref<20x42xf32, #map> {
  %0 = memref.subview %arg0[0, 1] [20, 42] [1, 1] : memref<20x42xf32> to memref<20x42xf32, #map>
  return %0 : memref<20x42xf32, #map>
}
// CHECK-LABEL: func @no_fold_subview_with_non_zero_offset(
//       CHECK:   %[[SUBVIEW:.+]] = memref.subview
//       CHECK:    return %[[SUBVIEW]]

// -----

#map = affine_map<(d0, d1) -> (d0 * 42 + d1 * 2)>
func @no_fold_subview_with_non_unit_stride(%arg0 : memref<20x42xf32>) -> memref<20x42xf32, #map> {
  %0 = memref.subview %arg0[0, 0] [20, 42] [1, 2] : memref<20x42xf32> to memref<20x42xf32, #map>
  return %0 : memref<20x42xf32, #map>
}
// CHECK-LABEL: func @no_fold_subview_with_non_unit_stride(
//       CHECK:   %[[SUBVIEW:.+]] = memref.subview
//       CHECK:    return %[[SUBVIEW]]

// -----

#map = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + d1 + s0)>
func @no_fold_dynamic_no_op_subview(%arg0 : memref<?x?xf32>) -> memref<?x?xf32, #map> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.dim %arg0, %c0 : memref<?x?xf32>
  %1 = memref.dim %arg0, %c1 : memref<?x?xf32>
  %2 = memref.subview %arg0[0, 0] [%0, %1] [1, 1] : memref<?x?xf32> to memref<?x?xf32, #map>
  return %2 : memref<?x?xf32, #map>
}
// CHECK-LABEL: func @no_fold_dynamic_no_op_subview(
//       CHECK:   %[[SUBVIEW:.+]] = memref.subview
//       CHECK:    return %[[SUBVIEW]]

// -----

func @atomicrmw_cast_fold(%arg0 : f32, %arg1 : memref<4xf32>, %c : index) {
  %v = memref.cast %arg1 : memref<4xf32> to memref<?xf32>
  %a = memref.atomic_rmw addf %arg0, %v[%c] : (f32, memref<?xf32>) -> f32
  return
}

// CHECK-LABEL: func @atomicrmw_cast_fold
// CHECK-NEXT: memref.atomic_rmw addf %arg0, %arg1[%arg2] : (f32, memref<4xf32>) -> f32
