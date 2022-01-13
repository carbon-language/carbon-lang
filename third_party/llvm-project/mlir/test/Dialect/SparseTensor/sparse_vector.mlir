// RUN: mlir-opt %s -sparsification="vectorization-strategy=0 vl=16" -split-input-file | \
// RUN:   FileCheck %s --check-prefix=CHECK-VEC0
// RUN: mlir-opt %s -sparsification="vectorization-strategy=1 vl=16" -split-input-file | \
// RUN:   FileCheck %s --check-prefix=CHECK-VEC1
// RUN: mlir-opt %s -sparsification="vectorization-strategy=2 vl=16" -split-input-file | \
// RUN:   FileCheck %s --check-prefix=CHECK-VEC2
// RUN: mlir-opt %s -sparsification="vectorization-strategy=2 vl=16 enable-simd-index32=true" -split-input-file | \
// RUN:   FileCheck %s --check-prefix=CHECK-VEC3

#DenseVector = #sparse_tensor.encoding<{ dimLevelType = [ "dense" ] }>

#trait_scale_d = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a
    affine_map<(i) -> (i)>   // x (out)
  ],
  iterator_types = ["parallel"],
  doc = "x(i) = a(i) * b"
}

//
// CHECK-VEC0-LABEL: func @scale_d
// CHECK-VEC0-DAG:   %[[c0:.*]] = constant 0 : index
// CHECK-VEC0-DAG:   %[[c1:.*]] = constant 1 : index
// CHECK-VEC0-DAG:   %[[c1024:.*]] = constant 1024 : index
// CHECK-VEC0:       scf.for %[[i:.*]] = %[[c0]] to %[[c1024]] step %[[c1]] {
// CHECK-VEC0:         %[[l:.*]] = memref.load %{{.*}}[%[[i]]] : memref<?xf32>
// CHECK-VEC0:         %[[m:.*]] = mulf %[[l]], %{{.*}} : f32
// CHECK-VEC0:         store %[[m]], %{{.*}}[%[[i]]] : memref<1024xf32>
// CHECK-VEC0:       }
// CHECK-VEC0:       return
//
// CHECK-VEC1-LABEL: func @scale_d
// CHECK-VEC1-DAG:   %[[c0:.*]] = constant 0 : index
// CHECK-VEC1-DAG:   %[[c16:.*]] = constant 16 : index
// CHECK-VEC1-DAG:   %[[c1024:.*]] = constant 1024 : index
// CHECK-VEC1:       scf.for %[[i:.*]] = %[[c0]] to %[[c1024]] step %[[c16]] {
// CHECK-VEC1:         %[[r:.*]] = vector.load %{{.*}}[%[[i]]] : memref<?xf32>, vector<16xf32>
// CHECK-VEC1:         %[[b:.*]] = vector.broadcast %{{.*}} : f32 to vector<16xf32>
// CHECK-VEC1:         %[[m:.*]] = mulf %[[r]], %[[b]] : vector<16xf32>
// CHECK-VEC1:         vector.store %[[m]], %{{.*}}[%[[i]]] : memref<1024xf32>, vector<16xf32>
// CHECK-VEC1:       }
// CHECK-VEC1:       return
//
// CHECK-VEC2-LABEL: func @scale_d
// CHECK-VEC2-DAG:   %[[c0:.*]] = constant 0 : index
// CHECK-VEC2-DAG:   %[[c16:.*]] = constant 16 : index
// CHECK-VEC2-DAG:   %[[c1024:.*]] = constant 1024 : index
// CHECK-VEC2:       scf.for %[[i:.*]] = %[[c0]] to %[[c1024]] step %[[c16]] {
// CHECK-VEC2:         %[[r:.*]] = vector.load %{{.*}}[%[[i]]] : memref<?xf32>, vector<16xf32>
// CHECK-VEC2:         %[[b:.*]] = vector.broadcast %{{.*}} : f32 to vector<16xf32>
// CHECK-VEC2:         %[[m:.*]] = mulf %[[r]], %[[b]] : vector<16xf32>
// CHECK-VEC2:         vector.store %[[m]], %{{.*}}[%[[i]]] : memref<1024xf32>, vector<16xf32>
// CHECK-VEC2:       }
// CHECK-VEC2:       return
//
func @scale_d(%arga: tensor<1024xf32, #DenseVector>, %b: f32, %argx: tensor<1024xf32>) -> tensor<1024xf32> {
  %0 = linalg.generic #trait_scale_d
    ins(%arga: tensor<1024xf32, #DenseVector>)
    outs(%argx: tensor<1024xf32>) {
      ^bb(%a: f32, %x: f32):
        %0 = mulf %a, %b : f32
        linalg.yield %0 : f32
  } -> tensor<1024xf32>
  return %0 : tensor<1024xf32>
}

// -----

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

//
// CHECK-VEC0-LABEL: func @mul_s
// CHECK-VEC0-DAG:   %[[c0:.*]] = constant 0 : index
// CHECK-VEC0-DAG:   %[[c1:.*]] = constant 1 : index
// CHECK-VEC0:       %[[p:.*]] = memref.load %{{.*}}[%[[c0]]] : memref<?xi32>
// CHECK-VEC0:       %[[a:.*]] = zexti %[[p]] : i32 to i64
// CHECK-VEC0:       %[[q:.*]] = index_cast %[[a]] : i64 to index
// CHECK-VEC0:       %[[r:.*]] = memref.load %{{.*}}[%[[c1]]] : memref<?xi32>
// CHECK-VEC0:       %[[b:.*]] = zexti %[[r]] : i32 to i64
// CHECK-VEC0:       %[[s:.*]] = index_cast %[[b]] : i64 to index
// CHECK-VEC0:       scf.for %[[i:.*]] = %[[q]] to %[[s]] step %[[c1]] {
// CHECK-VEC0:         %[[li:.*]] = memref.load %{{.*}}[%[[i]]] : memref<?xi32>
// CHECK-VEC0:         %[[zi:.*]] = zexti %[[li]] : i32 to i64
// CHECK-VEC0:         %[[ci:.*]] = index_cast %[[zi]] : i64 to index
// CHECK-VEC0:         %[[la:.*]] = memref.load %{{.*}}[%[[i]]] : memref<?xf32>
// CHECK-VEC0:         %[[lb:.*]] = memref.load %{{.*}}[%[[ci]]] : memref<1024xf32>
// CHECK-VEC0:         %[[m:.*]] = mulf %[[la]], %[[lb]] : f32
// CHECK-VEC0:         store %[[m]], %{{.*}}[%[[ci]]] : memref<1024xf32>
// CHECK-VEC0:       }
// CHECK-VEC0:       return
//
// CHECK-VEC1-LABEL: func @mul_s
// CHECK-VEC1-DAG:   %[[c0:.*]] = constant 0 : index
// CHECK-VEC1-DAG:   %[[c1:.*]] = constant 1 : index
// CHECK-VEC1:       %[[p:.*]] = memref.load %{{.*}}[%[[c0]]] : memref<?xi32>
// CHECK-VEC1:       %[[a:.*]] = zexti %[[p]] : i32 to i64
// CHECK-VEC1:       %[[q:.*]] = index_cast %[[a]] : i64 to index
// CHECK-VEC1:       %[[r:.*]] = memref.load %{{.*}}[%[[c1]]] : memref<?xi32>
// CHECK-VEC1:       %[[b:.*]] = zexti %[[r]] : i32 to i64
// CHECK-VEC1:       %[[s:.*]] = index_cast %[[b]] : i64 to index
// CHECK-VEC1:       scf.for %[[i:.*]] = %[[q]] to %[[s]] step %[[c1]] {
// CHECK-VEC1:         %[[li:.*]] = memref.load %{{.*}}[%[[i]]] : memref<?xi32>
// CHECK-VEC1:         %[[zi:.*]] = zexti %[[li]] : i32 to i64
// CHECK-VEC1:         %[[ci:.*]] = index_cast %[[zi]] : i64 to index
// CHECK-VEC1:         %[[la:.*]] = memref.load %{{.*}}[%[[i]]] : memref<?xf32>
// CHECK-VEC1:         %[[lb:.*]] = memref.load %{{.*}}[%[[ci]]] : memref<1024xf32>
// CHECK-VEC1:         %[[m:.*]] = mulf %[[la]], %[[lb]] : f32
// CHECK-VEC1:         store %[[m]], %{{.*}}[%[[ci]]] : memref<1024xf32>
// CHECK-VEC1:       }
// CHECK-VEC1:       return
//
// CHECK-VEC2:       #[[$map:.*]] = affine_map<(d0, d1)[s0] -> (16, d0 - d1)
// CHECK-VEC2-LABEL: func @mul_s
// CHECK-VEC2-DAG:   %[[c0:.*]] = constant 0 : index
// CHECK-VEC2-DAG:   %[[c1:.*]] = constant 1 : index
// CHECK-VEC2-DAG:   %[[c16:.*]] = constant 16 : index
// CHECK-VEC2:       %[[p:.*]] = memref.load %{{.*}}[%[[c0]]] : memref<?xi32>
// CHECK-VEC2:       %[[a:.*]] = zexti %[[p]] : i32 to i64
// CHECK-VEC2:       %[[q:.*]] = index_cast %[[a]] : i64 to index
// CHECK-VEC2:       %[[r:.*]] = memref.load %{{.*}}[%[[c1]]] : memref<?xi32>
// CHECK-VEC2:       %[[b:.*]] = zexti %[[r]] : i32 to i64
// CHECK-VEC2:       %[[s:.*]] = index_cast %[[b]] : i64 to index
// CHECK-VEC2:       scf.for %[[i:.*]] = %[[q]] to %[[s]] step %[[c16]] {
// CHECK-VEC2:         %[[sub:.*]] = affine.min #[[$map]](%[[s]], %[[i]])[%[[c16]]]
// CHECK-VEC2:         %[[mask:.*]] = vector.create_mask %[[sub]] : vector<16xi1>
// CHECK-VEC2:         %[[li:.*]] = vector.maskedload %{{.*}}[%[[i]]], %[[mask]], %{{.*}} : memref<?xi32>, vector<16xi1>, vector<16xi32> into vector<16xi32>
// CHECK-VEC2:         %[[zi:.*]] = zexti %[[li]] : vector<16xi32> to vector<16xi64>
// CHECK-VEC2:         %[[la:.*]] = vector.maskedload %{{.*}}[%[[i]]], %[[mask]], %{{.*}} : memref<?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
// CHECK-VEC2:         %[[lb:.*]] = vector.gather %{{.*}}[%[[c0]]] [%[[zi]]], %[[mask]], %{{.*}} : memref<1024xf32>, vector<16xi64>, vector<16xi1>, vector<16xf32> into vector<16xf32>
// CHECK-VEC2:         %[[m:.*]] = mulf %[[la]], %[[lb]] : vector<16xf32>
// CHECK-VEC2:         vector.scatter %{{.*}}[%[[c0]]] [%[[zi]]], %[[mask]], %[[m]] : memref<1024xf32>, vector<16xi64>, vector<16xi1>, vector<16xf32>
// CHECK-VEC2:       }
// CHECK-VEC2:       return
//
// CHECK-VEC3:       #[[$map:.*]] = affine_map<(d0, d1)[s0] -> (16, d0 - d1)
// CHECK-VEC3-LABEL: func @mul_s
// CHECK-VEC3-DAG:   %[[c0:.*]] = constant 0 : index
// CHECK-VEC3-DAG:   %[[c1:.*]] = constant 1 : index
// CHECK-VEC3-DAG:   %[[c16:.*]] = constant 16 : index
// CHECK-VEC3:       %[[p:.*]] = memref.load %{{.*}}[%[[c0]]] : memref<?xi32>
// CHECK-VEC3:       %[[a:.*]] = zexti %[[p]] : i32 to i64
// CHECK-VEC3:       %[[q:.*]] = index_cast %[[a]] : i64 to index
// CHECK-VEC3:       %[[r:.*]] = memref.load %{{.*}}[%[[c1]]] : memref<?xi32>
// CHECK-VEC3:       %[[b:.*]] = zexti %[[r]] : i32 to i64
// CHECK-VEC3:       %[[s:.*]] = index_cast %[[b]] : i64 to index
// CHECK-VEC3:       scf.for %[[i:.*]] = %[[q]] to %[[s]] step %[[c16]] {
// CHECK-VEC3:         %[[sub:.*]] = affine.min #[[$map]](%[[s]], %[[i]])[%[[c16]]]
// CHECK-VEC3:         %[[mask:.*]] = vector.create_mask %[[sub]] : vector<16xi1>
// CHECK-VEC3:         %[[li:.*]] = vector.maskedload %{{.*}}[%[[i]]], %[[mask]], %{{.*}} : memref<?xi32>, vector<16xi1>, vector<16xi32> into vector<16xi32>
// CHECK-VEC3:         %[[la:.*]] = vector.maskedload %{{.*}}[%[[i]]], %[[mask]], %{{.*}} : memref<?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
// CHECK-VEC3:         %[[lb:.*]] = vector.gather %{{.*}}[%[[c0]]] [%[[li]]], %[[mask]], %{{.*}} : memref<1024xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
// CHECK-VEC3:         %[[m:.*]] = mulf %[[la]], %[[lb]] : vector<16xf32>
// CHECK-VEC3:         vector.scatter %{{.*}}[%[[c0]]] [%[[li]]], %[[mask]], %[[m]] : memref<1024xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32>
// CHECK-VEC3:       }
// CHECK-VEC3:       return
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

// -----

#DenseVector = #sparse_tensor.encoding<{ dimLevelType = [ "dense" ] }>

#trait_reduction_d = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a
    affine_map<(i) -> (i)>,  // b
    affine_map<(i) -> ()>    // x (out)
  ],
  iterator_types = ["reduction"],
  doc = "x += a(i) * b(i)"
}

//
// CHECK-VEC0-LABEL: func @reduction_d
// CHECK-VEC0-DAG:   %[[c0:.*]] = constant 0 : index
// CHECK-VEC0-DAG:   %[[c1:.*]] = constant 1 : index
// CHECK-VEC0-DAG:   %[[c1024:.*]] = constant 1024 : index
// CHECK-VEC0:       %[[red:.*]] = scf.for %[[i:.*]] = %[[c0]] to %[[c1024]] step %[[c1]] iter_args(%[[red_in:.*]] = %{{.*}}) -> (f32) {
// CHECK-VEC0:         %[[la:.*]] = memref.load %{{.*}}[%[[i]]] : memref<?xf32>
// CHECK-VEC0:         %[[lb:.*]] = memref.load %{{.*}}[%[[i]]] : memref<1024xf32>
// CHECK-VEC0:         %[[m:.*]] = mulf %[[la]], %[[lb]] : f32
// CHECK-VEC0:         %[[a:.*]] = addf %[[red_in]], %[[m]] : f32
// CHECK-VEC0:         scf.yield %[[a]] : f32
// CHECK-VEC0:       }
// CHECK-VEC0:       return
//
// CHECK-VEC1-LABEL: func @reduction_d
// CHECK-VEC1-DAG:   %[[c0:.*]] = constant 0 : index
// CHECK-VEC1-DAG:   %[[c16:.*]] = constant 16 : index
// CHECK-VEC1-DAG:   %[[c1024:.*]] = constant 1024 : index
// CHECK-VEC1-DAG:   %[[v0:.*]] = constant dense<0.000000e+00> : vector<16xf32>
// CHECK-VEC1:       %[[red:.*]] = scf.for %[[i:.*]] = %[[c0]] to %[[c1024]] step %[[c16]] iter_args(%[[red_in:.*]] = %[[v0]]) -> (vector<16xf32>) {
// CHECK-VEC1:         %[[la:.*]] = vector.load %{{.*}}[%[[i]]] : memref<?xf32>, vector<16xf32>
// CHECK-VEC1:         %[[lb:.*]] = vector.load %{{.*}}[%[[i]]] : memref<1024xf32>, vector<16xf32>
// CHECK-VEC1:         %[[m:.*]] = mulf %[[la]], %[[lb]] : vector<16xf32>
// CHECK-VEC1:         %[[a:.*]] = addf %[[red_in]], %[[m]] : vector<16xf32>
// CHECK-VEC1:         scf.yield %[[a]] : vector<16xf32>
// CHECK-VEC1:       }
// CHECK-VEC1:       %{{.*}} = vector.reduction "add", %[[red]], %{{.*}} : vector<16xf32> into f32
// CHECK-VEC1:       return
//
// CHECK-VEC2-LABEL: func @reduction_d
// CHECK-VEC2-DAG:   %[[c0:.*]] = constant 0 : index
// CHECK-VEC2-DAG:   %[[c16:.*]] = constant 16 : index
// CHECK-VEC2-DAG:   %[[c1024:.*]] = constant 1024 : index
// CHECK-VEC2-DAG:   %[[v0:.*]] = constant dense<0.000000e+00> : vector<16xf32>
// CHECK-VEC2:       %[[red:.*]] = scf.for %[[i:.*]] = %[[c0]] to %[[c1024]] step %[[c16]] iter_args(%[[red_in:.*]] = %[[v0]]) -> (vector<16xf32>) {
// CHECK-VEC2:         %[[la:.*]] = vector.load %{{.*}}[%[[i]]] : memref<?xf32>, vector<16xf32>
// CHECK-VEC2:         %[[lb:.*]] = vector.load %{{.*}}[%[[i]]] : memref<1024xf32>, vector<16xf32>
// CHECK-VEC2:         %[[m:.*]] = mulf %[[la]], %[[lb]] : vector<16xf32>
// CHECK-VEC2:         %[[a:.*]] = addf %[[red_in]], %[[m]] : vector<16xf32>
// CHECK-VEC2:         scf.yield %[[a]] : vector<16xf32>
// CHECK-VEC2:       }
// CHECK-VEC2:       %{{.*}} = vector.reduction "add", %[[red]], %{{.*}} : vector<16xf32> into f32
// CHECK-VEC2:       return
//
func @reduction_d(%arga: tensor<1024xf32, #DenseVector>, %argb: tensor<1024xf32>, %argx: tensor<f32>) -> tensor<f32> {
  %0 = linalg.generic #trait_reduction_d
    ins(%arga, %argb: tensor<1024xf32, #DenseVector>, tensor<1024xf32>)
    outs(%argx: tensor<f32>) {
      ^bb(%a: f32, %b: f32, %x: f32):
        %0 = mulf %a, %b : f32
        %1 = addf %x, %0 : f32
        linalg.yield %1 : f32
  } -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

#SparseMatrix = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  pointerBitWidth = 32,
  indexBitWidth = 32
}>

#trait_mul_ds = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // A
    affine_map<(i,j) -> (i,j)>,  // B
    affine_map<(i,j) -> (i,j)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) = A(i,j) * B(i,j)"
}

//
// CHECK-VEC0-LABEL: func @mul_ds
// CHECK-VEC0-DAG:   %[[c0:.*]] = constant 0 : index
// CHECK-VEC0-DAG:   %[[c1:.*]] = constant 1 : index
// CHECK-VEC0-DAG:   %[[c512:.*]] = constant 512 : index
// CHECK-VEC0:       scf.for %[[i:.*]] = %[[c0]] to %[[c512]] step %[[c1]] {
// CHECK-VEC0:         %[[p:.*]] = memref.load %{{.*}}[%[[i]]] : memref<?xi32>
// CHECK-VEC0:         %[[a:.*]] = zexti %[[p]] : i32 to i64
// CHECK-VEC0:         %[[q:.*]] = index_cast %[[a]] : i64 to index
// CHECK-VEC0:         %[[a:.*]] = addi %[[i]], %[[c1]] : index
// CHECK-VEC0:         %[[r:.*]] = memref.load %{{.*}}[%[[a]]] : memref<?xi32>
// CHECK-VEC0:         %[[b:.*]] = zexti %[[r]] : i32 to i64
// CHECK-VEC0:         %[[s:.*]] = index_cast %[[b]] : i64 to index
// CHECK-VEC0:         scf.for %[[j:.*]] = %[[q]] to %[[s]] step %[[c1]] {
// CHECK-VEC0:           %[[lj:.*]] = memref.load %{{.*}}[%[[j]]] : memref<?xi32>
// CHECK-VEC0:           %[[zj:.*]] = zexti %[[lj]] : i32 to i64
// CHECK-VEC0:           %[[cj:.*]] = index_cast %[[zj]] : i64 to index
// CHECK-VEC0:           %[[la:.*]] = memref.load %{{.*}}[%[[j]]] : memref<?xf32>
// CHECK-VEC0:           %[[lb:.*]] = memref.load %{{.*}}[%[[i]], %[[cj]]] : memref<512x1024xf32>
// CHECK-VEC0:           %[[m:.*]] = mulf %[[la]], %[[lb]] : f32
// CHECK-VEC0:           store %[[m]], %{{.*}}[%[[i]], %[[cj]]] : memref<512x1024xf32>
// CHECK-VEC0:         }
// CHECK-VEC0:       }
// CHECK-VEC0:       return
//
// CHECK-VEC1-LABEL: func @mul_ds
// CHECK-VEC1-DAG:   %[[c0:.*]] = constant 0 : index
// CHECK-VEC1-DAG:   %[[c1:.*]] = constant 1 : index
// CHECK-VEC1-DAG:   %[[c512:.*]] = constant 512 : index
// CHECK-VEC1:       scf.for %[[i:.*]] = %[[c0]] to %[[c512]] step %[[c1]] {
// CHECK-VEC1:         %[[p:.*]] = memref.load %{{.*}}[%[[i]]] : memref<?xi32>
// CHECK-VEC1:         %[[a:.*]] = zexti %[[p]] : i32 to i64
// CHECK-VEC1:         %[[q:.*]] = index_cast %[[a]] : i64 to index
// CHECK-VEC1:         %[[a:.*]] = addi %[[i]], %[[c1]] : index
// CHECK-VEC1:         %[[r:.*]] = memref.load %{{.*}}[%[[a]]] : memref<?xi32>
// CHECK-VEC1:         %[[b:.*]] = zexti %[[r]] : i32 to i64
// CHECK-VEC1:         %[[s:.*]] = index_cast %[[b]] : i64 to index
// CHECK-VEC1:         scf.for %[[j:.*]] = %[[q]] to %[[s]] step %[[c1]] {
// CHECK-VEC1:           %[[lj:.*]] = memref.load %{{.*}}[%[[j]]] : memref<?xi32>
// CHECK-VEC1:           %[[zj:.*]] = zexti %[[lj]] : i32 to i64
// CHECK-VEC1:           %[[cj:.*]] = index_cast %[[zj]] : i64 to index
// CHECK-VEC1:           %[[la:.*]] = memref.load %{{.*}}[%[[j]]] : memref<?xf32>
// CHECK-VEC1:           %[[lb:.*]] = memref.load %{{.*}}[%[[i]], %[[cj]]] : memref<512x1024xf32>
// CHECK-VEC1:           %[[m:.*]] = mulf %[[la]], %[[lb]] : f32
// CHECK-VEC1:           store %[[m]], %{{.*}}[%[[i]], %[[cj]]] : memref<512x1024xf32>
// CHECK-VEC1:         }
// CHECK-VEC1:       }
// CHECK-VEC1:       return
//
// CHECK-VEC2:       #[[$map:.*]] = affine_map<(d0, d1)[s0] -> (16, d0 - d1)
// CHECK-VEC2-LABEL: func @mul_ds
// CHECK-VEC2-DAG:   %[[c0:.*]] = constant 0 : index
// CHECK-VEC2-DAG:   %[[c1:.*]] = constant 1 : index
// CHECK-VEC2-DAG:   %[[c16:.*]] = constant 16 : index
// CHECK-VEC2-DAG:   %[[c512:.*]] = constant 512 : index
// CHECK-VEC2:       scf.for %[[i:.*]] = %[[c0]] to %[[c512]] step %[[c1]] {
// CHECK-VEC2:         %[[p:.*]] = memref.load %{{.*}}[%[[i]]] : memref<?xi32>
// CHECK-VEC2:         %[[a:.*]] = zexti %[[p]] : i32 to i64
// CHECK-VEC2:         %[[q:.*]] = index_cast %[[a]] : i64 to index
// CHECK-VEC2:         %[[a:.*]] = addi %[[i]], %[[c1]] : index
// CHECK-VEC2:         %[[r:.*]] = memref.load %{{.*}}[%[[a]]] : memref<?xi32>
// CHECK-VEC2:         %[[b:.*]] = zexti %[[r]] : i32 to i64
// CHECK-VEC2:         %[[s:.*]] = index_cast %[[b]] : i64 to index
// CHECK-VEC2:         scf.for %[[j:.*]] = %[[q]] to %[[s]] step %[[c16]] {
// CHECK-VEC2:           %[[sub:.*]] = affine.min #[[$map]](%[[s]], %[[j]])[%[[c16]]]
// CHECK-VEC2:           %[[mask:.*]] = vector.create_mask %[[sub]] : vector<16xi1>
// CHECK-VEC2:           %[[lj:.*]] = vector.maskedload %{{.*}}[%[[j]]], %[[mask]], %{{.*}} : memref<?xi32>, vector<16xi1>, vector<16xi32> into vector<16xi32>
// CHECK-VEC2:           %[[zj:.*]] = zexti %[[lj]] : vector<16xi32> to vector<16xi64>
// CHECK-VEC2:           %[[la:.*]] = vector.maskedload %{{.*}}[%[[j]]], %[[mask]], %{{.*}} : memref<?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
// CHECK-VEC2:           %[[lb:.*]] = vector.gather %{{.*}}[%[[i]], %[[c0]]] [%[[zj]]], %[[mask]], %{{.*}} : memref<512x1024xf32>, vector<16xi64>, vector<16xi1>, vector<16xf32> into vector<16xf32>
// CHECK-VEC2:           %[[m:.*]] = mulf %[[la]], %[[lb]] : vector<16xf32>
// CHECK-VEC2:           vector.scatter %{{.*}}[%[[i]], %[[c0]]] [%[[zj]]], %[[mask]], %[[m]] : memref<512x1024xf32>, vector<16xi64>, vector<16xi1>, vector<16xf32>
// CHECK-VEC2:         }
// CHECK-VEC2:       }
// CHECK-VEC2:       return
//
// CHECK-VEC3:       #[[$map:.*]] = affine_map<(d0, d1)[s0] -> (16, d0 - d1)
// CHECK-VEC3-LABEL: func @mul_ds
// CHECK-VEC3-DAG:   %[[c0:.*]] = constant 0 : index
// CHECK-VEC3-DAG:   %[[c1:.*]] = constant 1 : index
// CHECK-VEC3-DAG:   %[[c16:.*]] = constant 16 : index
// CHECK-VEC3-DAG:   %[[c512:.*]] = constant 512 : index
// CHECK-VEC3:       scf.for %[[i:.*]] = %[[c0]] to %[[c512]] step %[[c1]] {
// CHECK-VEC3:         %[[p:.*]] = memref.load %{{.*}}[%[[i]]] : memref<?xi32>
// CHECK-VEC3:         %[[a:.*]] = zexti %[[p]] : i32 to i64
// CHECK-VEC3:         %[[q:.*]] = index_cast %[[a]] : i64 to index
// CHECK-VEC3:         %[[a:.*]] = addi %[[i]], %[[c1]] : index
// CHECK-VEC3:         %[[r:.*]] = memref.load %{{.*}}[%[[a]]] : memref<?xi32>
// CHECK-VEC3:         %[[b:.*]] = zexti %[[r]] : i32 to i64
// CHECK-VEC3:         %[[s:.*]] = index_cast %[[b]] : i64 to index
// CHECK-VEC3:         scf.for %[[j:.*]] = %[[q]] to %[[s]] step %[[c16]] {
// CHECK-VEC3:           %[[sub:.*]] = affine.min #[[$map]](%[[s]], %[[j]])[%[[c16]]]
// CHECK-VEC3:           %[[mask:.*]] = vector.create_mask %[[sub]] : vector<16xi1>
// CHECK-VEC3:           %[[lj:.*]] = vector.maskedload %{{.*}}[%[[j]]], %[[mask]], %{{.*}} : memref<?xi32>, vector<16xi1>, vector<16xi32> into vector<16xi32>
// CHECK-VEC3:           %[[la:.*]] = vector.maskedload %{{.*}}[%[[j]]], %[[mask]], %{{.*}} : memref<?xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
// CHECK-VEC3:           %[[lb:.*]] = vector.gather %{{.*}}[%[[i]], %[[c0]]] [%[[lj]]], %[[mask]], %{{.*}} : memref<512x1024xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
// CHECK-VEC3:           %[[m:.*]] = mulf %[[la]], %[[lb]] : vector<16xf32>
// CHECK-VEC3:           vector.scatter %{{.*}}[%[[i]], %[[c0]]] [%[[lj]]], %[[mask]], %[[m]] : memref<512x1024xf32>, vector<16xi32>, vector<16xi1>, vector<16xf32>
// CHECK-VEC3:         }
// CHECK-VEC3:       }
// CHECK-VEC3:       return
//
func @mul_ds(%arga: tensor<512x1024xf32, #SparseMatrix>, %argb: tensor<512x1024xf32>, %argx: tensor<512x1024xf32>) -> tensor<512x1024xf32> {
  %0 = linalg.generic #trait_mul_ds
    ins(%arga, %argb: tensor<512x1024xf32, #SparseMatrix>, tensor<512x1024xf32>)
    outs(%argx: tensor<512x1024xf32>) {
      ^bb(%a: f32, %b: f32, %x: f32):
        %0 = mulf %a, %b : f32
        linalg.yield %0 : f32
  } -> tensor<512x1024xf32>
  return %0 : tensor<512x1024xf32>
}
