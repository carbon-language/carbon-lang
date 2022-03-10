// RUN: mlir-opt %s -split-input-file | mlir-opt | FileCheck %s

#SparseVector = #sparse_tensor.encoding<{dimLevelType = ["compressed"]}>

// CHECK-LABEL: func @sparse_new(
// CHECK-SAME: %[[A:.*]]: !llvm.ptr<i8>)
//       CHECK: %[[T:.*]] = sparse_tensor.new %[[A]] : !llvm.ptr<i8> to tensor<128xf64, #{{.*}}>
//       CHECK: return %[[T]] : tensor<128xf64, #{{.*}}>
func @sparse_new(%arg0: !llvm.ptr<i8>) -> tensor<128xf64, #SparseVector> {
  %0 = sparse_tensor.new %arg0 : !llvm.ptr<i8> to tensor<128xf64, #SparseVector>
  return %0 : tensor<128xf64, #SparseVector>
}

// -----

#SparseMatrix = #sparse_tensor.encoding<{dimLevelType = ["compressed", "compressed"]}>

// CHECK-LABEL: func @sparse_init()
//   CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
//   CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
//       CHECK: %[[T:.*]] = sparse_tensor.init[%[[C16]], %[[C32]]] : tensor<?x32xf64, #{{.*}}>
//       CHECK: return %[[T]] : tensor<?x32xf64, #{{.*}}>
func @sparse_init() -> tensor<?x32xf64, #SparseMatrix> {
  %d1 = arith.constant 16 : index
  %d2 = arith.constant 32 : index
  %0 = sparse_tensor.init [%d1, %d2] : tensor<?x32xf64, #SparseMatrix>
  return %0 : tensor<?x32xf64, #SparseMatrix>
}

// -----

#SparseVector = #sparse_tensor.encoding<{dimLevelType = ["compressed"]}>

// CHECK-LABEL: func @sparse_release(
// CHECK-SAME: %[[A:.*]]: tensor<128xf64, #{{.*}}>
//       CHECK: sparse_tensor.release %[[A]] : tensor<128xf64, #{{.*}}>
//       CHECK: return
func @sparse_release(%arg0: tensor<128xf64, #SparseVector>) {
  sparse_tensor.release %arg0 : tensor<128xf64, #SparseVector>
  return
}

// -----

#SparseVector = #sparse_tensor.encoding<{dimLevelType = ["compressed"]}>

// CHECK-LABEL: func @sparse_convert_1d_to_sparse(
// CHECK-SAME: %[[A:.*]]: tensor<64xf32>)
//       CHECK: %[[T:.*]] = sparse_tensor.convert %[[A]] : tensor<64xf32> to tensor<64xf32, #{{.*}}>
//       CHECK: return %[[T]] : tensor<64xf32, #{{.*}}>
func @sparse_convert_1d_to_sparse(%arg0: tensor<64xf32>) -> tensor<64xf32, #SparseVector> {
  %0 = sparse_tensor.convert %arg0 : tensor<64xf32> to tensor<64xf32, #SparseVector>
  return %0 : tensor<64xf32, #SparseVector>
}

// -----

#SparseTensor = #sparse_tensor.encoding<{ dimLevelType = [ "dense", "dense", "compressed" ] }>

// CHECK-LABEL: func @sparse_convert_3d_from_sparse(
// CHECK-SAME: %[[A:.*]]: tensor<8x8x8xf64, #{{.*}}>)
//       CHECK: %[[T:.*]] = sparse_tensor.convert %[[A]] : tensor<8x8x8xf64, #{{.*}}> to tensor<8x8x8xf64>
//       CHECK: return %[[T]] : tensor<8x8x8xf64>
func @sparse_convert_3d_from_sparse(%arg0: tensor<8x8x8xf64, #SparseTensor>) -> tensor<8x8x8xf64> {
  %0 = sparse_tensor.convert %arg0 : tensor<8x8x8xf64, #SparseTensor> to tensor<8x8x8xf64>
  return %0 : tensor<8x8x8xf64>
}

// -----

#SparseVector = #sparse_tensor.encoding<{dimLevelType = ["compressed"]}>

// CHECK-LABEL: func @sparse_pointers(
//  CHECK-SAME: %[[A:.*]]: tensor<128xf64, #{{.*}}>)
//       CHECK: %[[C:.*]] = arith.constant 0 : index
//       CHECK: %[[T:.*]] = sparse_tensor.pointers %[[A]], %[[C]] : tensor<128xf64, #{{.*}}> to memref<?xindex>
//       CHECK: return %[[T]] : memref<?xindex>
func @sparse_pointers(%arg0: tensor<128xf64, #SparseVector>) -> memref<?xindex> {
  %c = arith.constant 0 : index
  %0 = sparse_tensor.pointers %arg0, %c : tensor<128xf64, #SparseVector> to memref<?xindex>
  return %0 : memref<?xindex>
}

// -----

#SparseVector = #sparse_tensor.encoding<{dimLevelType = ["compressed"]}>

// CHECK-LABEL: func @sparse_indices(
//  CHECK-SAME: %[[A:.*]]: tensor<128xf64, #{{.*}}>)
//       CHECK: %[[C:.*]] = arith.constant 0 : index
//       CHECK: %[[T:.*]] = sparse_tensor.indices %[[A]], %[[C]] : tensor<128xf64, #{{.*}}> to memref<?xindex>
//       CHECK: return %[[T]] : memref<?xindex>
func @sparse_indices(%arg0: tensor<128xf64, #SparseVector>) -> memref<?xindex> {
  %c = arith.constant 0 : index
  %0 = sparse_tensor.indices %arg0, %c : tensor<128xf64, #SparseVector> to memref<?xindex>
  return %0 : memref<?xindex>
}

// -----

#SparseVector = #sparse_tensor.encoding<{dimLevelType = ["compressed"]}>

// CHECK-LABEL: func @sparse_values(
//  CHECK-SAME: %[[A:.*]]: tensor<128xf64, #{{.*}}>)
//       CHECK: %[[T:.*]] = sparse_tensor.values %[[A]] : tensor<128xf64, #{{.*}}> to memref<?xf64>
//       CHECK: return %[[T]] : memref<?xf64>
func @sparse_values(%arg0: tensor<128xf64, #SparseVector>) -> memref<?xf64> {
  %0 = sparse_tensor.values %arg0 : tensor<128xf64, #SparseVector> to memref<?xf64>
  return %0 : memref<?xf64>
}

// -----

#DenseMatrix = #sparse_tensor.encoding<{dimLevelType = ["dense","dense"]}>

// CHECK-LABEL: func @sparse_load(
//  CHECK-SAME: %[[A:.*]]: tensor<16x32xf64, #{{.*}}>)
//       CHECK: %[[T:.*]] = sparse_tensor.load %[[A]] : tensor<16x32xf64, #{{.*}}>
//       CHECK: return %[[T]] : tensor<16x32xf64, #{{.*}}>
func @sparse_load(%arg0: tensor<16x32xf64, #DenseMatrix>) -> tensor<16x32xf64, #DenseMatrix> {
  %0 = sparse_tensor.load %arg0 : tensor<16x32xf64, #DenseMatrix>
  return %0 : tensor<16x32xf64, #DenseMatrix>
}

// -----

#DenseMatrix = #sparse_tensor.encoding<{dimLevelType = ["dense","dense"]}>

// CHECK-LABEL: func @sparse_load_ins(
//  CHECK-SAME: %[[A:.*]]: tensor<16x32xf64, #{{.*}}>)
//       CHECK: %[[T:.*]] = sparse_tensor.load %[[A]] hasInserts : tensor<16x32xf64, #{{.*}}>
//       CHECK: return %[[T]] : tensor<16x32xf64, #{{.*}}>
func @sparse_load_ins(%arg0: tensor<16x32xf64, #DenseMatrix>) -> tensor<16x32xf64, #DenseMatrix> {
  %0 = sparse_tensor.load %arg0 hasInserts : tensor<16x32xf64, #DenseMatrix>
  return %0 : tensor<16x32xf64, #DenseMatrix>
}

// -----

#SparseVector = #sparse_tensor.encoding<{dimLevelType = ["compressed"]}>

// CHECK-LABEL: func @sparse_insert(
//  CHECK-SAME: %[[A:.*]]: tensor<128xf64, #sparse_tensor.encoding<{{.*}}>>,
//  CHECK-SAME: %[[B:.*]]: memref<?xindex>,
//  CHECK-SAME: %[[C:.*]]: f64) {
//       CHECK: sparse_tensor.lex_insert %[[A]], %[[B]], %[[C]] : tensor<128xf64, #{{.*}}>, memref<?xindex>, f64
//       CHECK: return
func @sparse_insert(%arg0: tensor<128xf64, #SparseVector>, %arg1: memref<?xindex>, %arg2: f64) {
  sparse_tensor.lex_insert %arg0, %arg1, %arg2 : tensor<128xf64, #SparseVector>, memref<?xindex>, f64
  return
}

// -----

#SparseMatrix = #sparse_tensor.encoding<{dimLevelType = ["compressed", "compressed"]}>

// CHECK-LABEL: func @sparse_expansion(
//  CHECK-SAME: %[[A:.*]]: tensor<8x8xf64, #sparse_tensor.encoding<{{.*}}>>)
//       CHECK: sparse_tensor.expand %[[A]]
//       CHECK: return
func @sparse_expansion(%arg0: tensor<8x8xf64, #SparseMatrix>) {
  %values, %filled, %added, %count = sparse_tensor.expand %arg0
    : tensor<8x8xf64, #SparseMatrix> to memref<?xf64>, memref<?xi1>, memref<?xindex>, index
  return
}

// -----

#SparseMatrix = #sparse_tensor.encoding<{dimLevelType = ["compressed", "compressed"]}>

// CHECK-LABEL: func @sparse_compression(
//  CHECK-SAME: %[[A:.*]]: tensor<8x8xf64, #sparse_tensor.encoding<{{.*}}>>,
//       CHECK: sparse_tensor.compress %[[A]]
//       CHECK: return
func @sparse_compression(%arg0: tensor<8x8xf64, #SparseMatrix>,
                         %arg1: memref<?xindex>, %arg2: memref<?xf64>, %arg3: memref<?xi1>,
                         %arg4: memref<?xindex>, %arg5: index) {
  sparse_tensor.compress %arg0, %arg1, %arg2, %arg3, %arg4, %arg5
    : tensor<8x8xf64, #SparseMatrix>, memref<?xindex>, memref<?xf64>, memref<?xi1>, memref<?xindex>, index
  return
}

// -----

#SparseMatrix = #sparse_tensor.encoding<{dimLevelType = ["compressed", "compressed"]}>

// CHECK-LABEL: func @sparse_out(
//  CHECK-SAME: %[[A:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{{.*}}>>,
//  CHECK-SAME: %[[B:.*]]: !llvm.ptr<i8>)
//       CHECK: sparse_tensor.out %[[A]], %[[B]] : tensor<?x?xf64, #sparse_tensor.encoding<{{.*}}>>, !llvm.ptr<i8>
//       CHECK: return
func @sparse_out(%arg0: tensor<?x?xf64, #SparseMatrix>, %arg1: !llvm.ptr<i8>) {
  sparse_tensor.out %arg0, %arg1 : tensor<?x?xf64, #SparseMatrix>, !llvm.ptr<i8>
  return
}
