// RUN: mlir-opt %s --convert-vector-to-llvm='force-32bit-vector-indices=1' | FileCheck %s --check-prefix=CMP32
// RUN: mlir-opt %s --convert-vector-to-llvm='force-32bit-vector-indices=0' | FileCheck %s --check-prefix=CMP64

// CMP32-LABEL: @genbool_var_1d(
// CMP32-SAME: %[[ARG:.*]]: index)
// CMP32: %[[T0:.*]] = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]> : vector<11xi32>
// CMP32: %[[T1:.*]] = arith.index_cast %[[ARG]] : index to i32
// CMP32: %[[T2:.*]] = llvm.insertelement %[[T1]], %{{.*}}[%{{.*}} : i32] : vector<11xi32>
// CMP32: %[[T3:.*]] = llvm.shufflevector %[[T2]], %{{.*}} [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32] : vector<11xi32>, vector<11xi32>
// CMP32: %[[T4:.*]] = arith.cmpi slt, %[[T0]], %[[T3]] : vector<11xi32>
// CMP32: return %[[T4]] : vector<11xi1>

// CMP64-LABEL: @genbool_var_1d(
// CMP64-SAME: %[[ARG:.*]]: index)
// CMP64: %[[T0:.*]] = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]> : vector<11xi64>
// CMP64: %[[T1:.*]] = arith.index_cast %[[ARG]] : index to i64
// CMP64: %[[T2:.*]] = llvm.insertelement %[[T1]], %{{.*}}[%{{.*}} : i32] : vector<11xi64>
// CMP64: %[[T3:.*]] = llvm.shufflevector %[[T2]], %{{.*}} [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32] : vector<11xi64>, vector<11xi64>
// CMP64: %[[T4:.*]] = arith.cmpi slt, %[[T0]], %[[T3]] : vector<11xi64>
// CMP64: return %[[T4]] : vector<11xi1>

func.func @genbool_var_1d(%arg0: index) -> vector<11xi1> {
  %0 = vector.create_mask %arg0 : vector<11xi1>
  return %0 : vector<11xi1>
}

// CMP32-LABEL: @genbool_var_1d_scalable(
// CMP32-SAME: %[[ARG:.*]]: index)
// CMP32: %[[T0:.*]] = llvm.intr.experimental.stepvector : vector<[11]xi32>
// CMP32: %[[T1:.*]] = arith.index_cast %[[ARG]] : index to i32
// CMP32: %[[T2:.*]] = llvm.insertelement %[[T1]], %{{.*}}[%{{.*}} : i32] : vector<[11]xi32>
// CMP32: %[[T3:.*]] = llvm.shufflevector %[[T2]], %{{.*}} [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32] : vector<[11]xi32>, vector<[11]xi32>
// CMP32: %[[T4:.*]] = arith.cmpi slt, %[[T0]], %[[T3]] : vector<[11]xi32>
// CMP32: return %[[T4]] : vector<[11]xi1>

// CMP64-LABEL: @genbool_var_1d_scalable(
// CMP64-SAME: %[[ARG:.*]]: index)
// CMP64: %[[T0:.*]] = llvm.intr.experimental.stepvector : vector<[11]xi64>
// CMP64: %[[T1:.*]] = arith.index_cast %[[ARG]] : index to i64
// CMP64: %[[T2:.*]] = llvm.insertelement %[[T1]], %{{.*}}[%{{.*}} : i32] : vector<[11]xi64>
// CMP64: %[[T3:.*]] = llvm.shufflevector %[[T2]], %{{.*}} [0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32] : vector<[11]xi64>, vector<[11]xi64>
// CMP64: %[[T4:.*]] = arith.cmpi slt, %[[T0]], %[[T3]] : vector<[11]xi64>
// CMP64: return %[[T4]] : vector<[11]xi1>

func.func @genbool_var_1d_scalable(%arg0: index) -> vector<[11]xi1> {
  %0 = vector.create_mask %arg0 : vector<[11]xi1>
  return %0 : vector<[11]xi1>
}

// CMP32-LABEL: @transfer_read_1d
// CMP32: %[[MEM:.*]]: memref<?xf32>, %[[OFF:.*]]: index) -> vector<16xf32> {
// CMP32: %[[D:.*]] = memref.dim %[[MEM]], %{{.*}} : memref<?xf32>
// CMP32: %[[S:.*]] = arith.subi %[[D]], %[[OFF]] : index
// CMP32: %[[C:.*]] = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xi32>
// CMP32: %[[B:.*]] = arith.index_cast %[[S]] : index to i32
// CMP32: %[[B0:.*]] = llvm.insertelement %[[B]], %{{.*}} : vector<16xi32>
// CMP32: %[[BV:.*]] = llvm.shufflevector %[[B0]], {{.*}} : vector<16xi32>, vector<16xi32>
// CMP32: %[[M:.*]] = arith.cmpi slt, %[[C]], %[[BV]] : vector<16xi32>
// CMP32: %[[L:.*]] = llvm.intr.masked.load %{{.*}}, %[[M]], %{{.*}}
// CMP32: return %[[L]] : vector<16xf32>

// CMP64-LABEL: @transfer_read_1d(
// CMP64: %[[MEM:.*]]: memref<?xf32>, %[[OFF:.*]]: index) -> vector<16xf32> {
// CMP64: %[[D:.*]] = memref.dim %[[MEM]], %{{.*}} : memref<?xf32>
// CMP64: %[[S:.*]] = arith.subi %[[D]], %[[OFF]] : index
// CMP64: %[[C:.*]] = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xi64>
// CMP64: %[[B:.*]] = arith.index_cast %[[S]] : index to i64
// CMP64: %[[B0:.*]] = llvm.insertelement %[[B]], %{{.*}} : vector<16xi64>
// CMP64: %[[BV:.*]] = llvm.shufflevector %[[B0]], {{.*}} : vector<16xi64>, vector<16xi64>
// CMP64: %[[M:.*]] = arith.cmpi slt, %[[C]], %[[BV]] : vector<16xi64>
// CMP64: %[[L:.*]] = llvm.intr.masked.load %{{.*}}, %[[M]], %{{.*}}
// CMP64: return %[[L]] : vector<16xf32>

func.func @transfer_read_1d(%A : memref<?xf32>, %i: index) -> vector<16xf32> {
  %d = arith.constant -1.0: f32
  %f = vector.transfer_read %A[%i], %d {permutation_map = affine_map<(d0) -> (d0)>} : memref<?xf32>, vector<16xf32>
  return %f : vector<16xf32>
}
