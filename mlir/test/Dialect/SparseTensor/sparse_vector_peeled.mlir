// RUN: mlir-opt %s -sparsification="vectorization-strategy=2 vl=16" -for-loop-peeling -canonicalize -split-input-file | \
// RUN:   FileCheck %s

#SparseVector = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed" ],
  pointerBitWidth = 32,
  indexBitWidth = 32
}>

#trait_mul_s = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a
    affine_map<(i) -> (i)>,  // b
    affine_map<(i) -> (i)>   // x (out)
  ],
  iterator_types = ["parallel"],
  doc = "x(i) = a(i) * b(i)"
}

// CHECK-DAG:   #[[$map0:.*]] = affine_map<()[s0, s1] -> (s0 + ((-s0 + s1) floordiv 16) * 16)>
// CHECK-DAG:   #[[$map1:.*]] = affine_map<()[s0, s1] -> ((s0 - s1) mod 16)>
// CHECK-LABEL: func @mul_s
// CHECK-DAG:   %[[c0:.*]] = constant 0 : index
// CHECK-DAG:   %[[c1:.*]] = constant 1 : index
// CHECK-DAG:   %[[c16:.*]] = constant 16 : index
// CHECK:       %[[p:.*]] = memref.load %{{.*}}[%[[c0]]] : memref<?xi32>
// CHECK:       %[[a:.*]] = zexti %[[p]] : i32 to i64
// CHECK:       %[[q:.*]] = index_cast %[[a]] : i64 to index
// CHECK:       %[[r:.*]] = memref.load %{{.*}}[%[[c1]]] : memref<?xi32>
// CHECK:       %[[b:.*]] = zexti %[[r]] : i32 to i64
// CHECK:       %[[s:.*]] = index_cast %[[b]] : i64 to index
// CHECK:       %[[boundary:.*]] = affine.apply #[[$map0]]()[%[[q]], %[[s]]]
// CHECK:       scf.for %[[i:.*]] = %[[q]] to %[[boundary]] step %[[c16]] {
// CHECK:         %[[mask:.*]] = vector.constant_mask [16] : vector<16xi1>
// CHECK:         %[[li:.*]] = vector.load %{{.*}}[%[[i]]] : memref<?xi32>, vector<16xi32>
// CHECK:         %[[zi:.*]] = zexti %[[li]] : vector<16xi32> to vector<16xi64>
// CHECK:         %[[la:.*]] = vector.load %{{.*}}[%[[i]]] : memref<?xf32>, vector<16xf32>
// CHECK:         %[[lb:.*]] = vector.gather %{{.*}}[%[[c0]]] [%[[zi]]], %[[mask]], %{{.*}} : memref<1024xf32>, vector<16xi64>, vector<16xi1>, vector<16xf32> into vector<16xf32>
// CHECK:         %[[m:.*]] = mulf %[[la]], %[[lb]] : vector<16xf32>
// CHECK:         vector.scatter %{{.*}}[%[[c0]]] [%[[zi]]], %[[mask]], %[[m]] : memref<1024xf32>, vector<16xi64>, vector<16xi1>, vector<16xf32>
// CHECK:       }
// CHECK:       %[[has_more:.*]] = cmpi slt, %[[boundary]], %[[s]] : index
// CHECK:       scf.if %[[has_more]] {
// CHECK:         %[[sub:.*]] = affine.apply #[[$map1]]()[%[[s]], %[[q]]]
// CHECK:         %[[mask2:.*]] = vector.create_mask %[[sub]] : vector<16xi1>
// CHECK:         %[[li2:.*]] = vector.maskedload %{{.*}}[%[[boundary]]], %[[mask2]], %{{.*}} : memref<?xi32>, vector<16xi1>, vector<16xi32> into vector<16xi32>
// CHECK:         %[[zi2:.*]] = zexti %[[li2]] : vector<16xi32> to vector<16xi64>
// CHECK:         %[[la2:.*]] = vector.maskedload %{{.*}}[%[[boundary]]], %[[mask2]], %{{.*}} : memref<?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
// CHECK:         %[[lb2:.*]] = vector.gather %{{.*}}[%[[c0]]] [%[[zi2]]], %[[mask2]], %{{.*}} : memref<1024xf32>, vector<16xi64>, vector<16xi1>, vector<16xf32> into vector<16xf32>
// CHECK:         %[[m2:.*]] = mulf %[[la2]], %[[lb2]] : vector<16xf32>
// CHECK:         vector.scatter %{{.*}}[%[[c0]]] [%[[zi2]]], %[[mask2]], %[[m2]] : memref<1024xf32>, vector<16xi64>, vector<16xi1>, vector<16xf32>
// CHECK:       }
// CHECK:       return
//
func @mul_s(%arga: tensor<1024xf32, #SparseVector>, %argb: tensor<1024xf32>, %argx: tensor<1024xf32>) -> tensor<1024xf32> {
  %0 = linalg.generic #trait_mul_s
    ins(%arga, %argb: tensor<1024xf32, #SparseVector>, tensor<1024xf32>)
    outs(%argx: tensor<1024xf32>) {
      ^bb(%a: f32, %b: f32, %x: f32):
        %0 = mulf %a, %b : f32
        linalg.yield %0 : f32
  } -> tensor<1024xf32>
  return %0 : tensor<1024xf32>
}
