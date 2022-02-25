// RUN: mlir-opt %s -sparsification | FileCheck %s --check-prefix=CHECK-HIR
//
// RUN: mlir-opt %s -sparsification --sparse-tensor-conversion | \
// RUN: FileCheck %s --check-prefix=CHECK-MIR
//
// RUN: mlir-opt %s -sparsification --sparse-tensor-conversion \
// RUN: --func-bufferize --tensor-constant-bufferize           \
// RUN: --tensor-bufferize --finalizing-bufferize |            \
// RUN: FileCheck %s --check-prefix=CHECK-LIR

#CSR = #sparse_tensor.encoding<{dimLevelType = [ "dense", "compressed" ]}>

#trait_matvec = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // A
    affine_map<(i,j) -> (j)>,    // b
    affine_map<(i,j) -> (i)>     // x (out)
  ],
  iterator_types = ["parallel","reduction"],
  doc = "x(i) += A(i,j) * b(j)"
}

// CHECK-HIR-LABEL:   func @matvec(
// CHECK-HIR-SAME:                 %[[VAL_0:.*]]: tensor<32x64xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 0, indexBitWidth = 0 }>>,
// CHECK-HIR-SAME:                 %[[VAL_1:.*]]: tensor<64xf64>,
// CHECK-HIR-SAME:                 %[[VAL_2:.*]]: tensor<32xf64> {linalg.inplaceable = true}) -> tensor<32xf64> {
// CHECK-HIR:           %[[VAL_3:.*]] = constant 32 : index
// CHECK-HIR:           %[[VAL_4:.*]] = constant 0 : index
// CHECK-HIR:           %[[VAL_5:.*]] = constant 1 : index
// CHECK-HIR:           %[[VAL_6:.*]] = sparse_tensor.pointers %[[VAL_0]], %[[VAL_5]] : tensor<32x64xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 0, indexBitWidth = 0 }>> to memref<?xindex>
// CHECK-HIR:           %[[VAL_7:.*]] = sparse_tensor.indices %[[VAL_0]], %[[VAL_5]] : tensor<32x64xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 0, indexBitWidth = 0 }>> to memref<?xindex>
// CHECK-HIR:           %[[VAL_8:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<32x64xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], pointerBitWidth = 0, indexBitWidth = 0 }>> to memref<?xf64>
// CHECK-HIR:           %[[VAL_9:.*]] = memref.buffer_cast %[[VAL_1]] : memref<64xf64>
// CHECK-HIR:           %[[VAL_10:.*]] = memref.buffer_cast %[[VAL_2]] : memref<32xf64>
// CHECK-HIR:           scf.for %[[VAL_11:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_5]] {
// CHECK-HIR:             %[[VAL_12:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_11]]] : memref<?xindex>
// CHECK-HIR:             %[[VAL_13:.*]] = addi %[[VAL_11]], %[[VAL_5]] : index
// CHECK-HIR:             %[[VAL_14:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_13]]] : memref<?xindex>
// CHECK-HIR:             %[[VAL_15:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_11]]] : memref<32xf64>
// CHECK-HIR:             %[[VAL_16:.*]] = scf.for %[[VAL_17:.*]] = %[[VAL_12]] to %[[VAL_14]] step %[[VAL_5]] iter_args(%[[VAL_18:.*]] = %[[VAL_15]]) -> (f64) {
// CHECK-HIR:               %[[VAL_19:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_17]]] : memref<?xindex>
// CHECK-HIR:               %[[VAL_20:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_17]]] : memref<?xf64>
// CHECK-HIR:               %[[VAL_21:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_19]]] : memref<64xf64>
// CHECK-HIR:               %[[VAL_22:.*]] = mulf %[[VAL_20]], %[[VAL_21]] : f64
// CHECK-HIR:               %[[VAL_23:.*]] = addf %[[VAL_18]], %[[VAL_22]] : f64
// CHECK-HIR:               scf.yield %[[VAL_23]] : f64
// CHECK-HIR:             }
// CHECK-HIR:             memref.store %[[VAL_24:.*]], %[[VAL_10]]{{\[}}%[[VAL_11]]] : memref<32xf64>
// CHECK-HIR:           }
// CHECK-HIR:           %[[VAL_25:.*]] = memref.tensor_load %[[VAL_10]] : memref<32xf64>
// CHECK-HIR:           return %[[VAL_25]] : tensor<32xf64>
// CHECK-HIR:         }

// CHECK-MIR-LABEL:   func @matvec(
// CHECK-MIR-SAME:                 %[[VAL_0:.*]]: !llvm.ptr<i8>,
// CHECK-MIR-SAME:                 %[[VAL_1:.*]]: tensor<64xf64>,
// CHECK-MIR-SAME:                 %[[VAL_2:.*]]: tensor<32xf64> {linalg.inplaceable = true}) -> tensor<32xf64> {
// CHECK-MIR:           %[[VAL_3:.*]] = constant 32 : index
// CHECK-MIR:           %[[VAL_4:.*]] = constant 0 : index
// CHECK-MIR:           %[[VAL_5:.*]] = constant 1 : index
// CHECK-MIR:           %[[VAL_6:.*]] = call @sparsePointers(%[[VAL_0]], %[[VAL_5]]) : (!llvm.ptr<i8>, index) -> memref<?xindex>
// CHECK-MIR:           %[[VAL_7:.*]] = call @sparseIndices(%[[VAL_0]], %[[VAL_5]]) : (!llvm.ptr<i8>, index) -> memref<?xindex>
// CHECK-MIR:           %[[VAL_8:.*]] = call @sparseValuesF64(%[[VAL_0]]) : (!llvm.ptr<i8>) -> memref<?xf64>
// CHECK-MIR:           %[[VAL_9:.*]] = memref.buffer_cast %[[VAL_1]] : memref<64xf64>
// CHECK-MIR:           %[[VAL_10:.*]] = memref.buffer_cast %[[VAL_2]] : memref<32xf64>
// CHECK-MIR:           scf.for %[[VAL_11:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_5]] {
// CHECK-MIR:             %[[VAL_12:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_11]]] : memref<?xindex>
// CHECK-MIR:             %[[VAL_13:.*]] = addi %[[VAL_11]], %[[VAL_5]] : index
// CHECK-MIR:             %[[VAL_14:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_13]]] : memref<?xindex>
// CHECK-MIR:             %[[VAL_15:.*]] = memref.load %[[VAL_10]]{{\[}}%[[VAL_11]]] : memref<32xf64>
// CHECK-MIR:             %[[VAL_16:.*]] = scf.for %[[VAL_17:.*]] = %[[VAL_12]] to %[[VAL_14]] step %[[VAL_5]] iter_args(%[[VAL_18:.*]] = %[[VAL_15]]) -> (f64) {
// CHECK-MIR:               %[[VAL_19:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_17]]] : memref<?xindex>
// CHECK-MIR:               %[[VAL_20:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_17]]] : memref<?xf64>
// CHECK-MIR:               %[[VAL_21:.*]] = memref.load %[[VAL_9]]{{\[}}%[[VAL_19]]] : memref<64xf64>
// CHECK-MIR:               %[[VAL_22:.*]] = mulf %[[VAL_20]], %[[VAL_21]] : f64
// CHECK-MIR:               %[[VAL_23:.*]] = addf %[[VAL_18]], %[[VAL_22]] : f64
// CHECK-MIR:               scf.yield %[[VAL_23]] : f64
// CHECK-MIR:             }
// CHECK-MIR:             memref.store %[[VAL_24:.*]], %[[VAL_10]]{{\[}}%[[VAL_11]]] : memref<32xf64>
// CHECK-MIR:           }
// CHECK-MIR:           %[[VAL_25:.*]] = memref.tensor_load %[[VAL_10]] : memref<32xf64>
// CHECK-MIR:           return %[[VAL_25]] : tensor<32xf64>
// CHECK-MIR:         }

// CHECK-LIR-LABEL:   func @matvec(
// CHECK-LIR-SAME:                 %[[VAL_0:.*]]: !llvm.ptr<i8>,
// CHECK-LIR-SAME:                 %[[VAL_1:.*]]: memref<64xf64>,
// CHECK-LIR-SAME:                 %[[VAL_2:.*]]: memref<32xf64> {linalg.inplaceable = true}) -> memref<32xf64> {
// CHECK-LIR:           %[[VAL_3:.*]] = constant 32 : index
// CHECK-LIR:           %[[VAL_4:.*]] = constant 0 : index
// CHECK-LIR:           %[[VAL_5:.*]] = constant 1 : index
// CHECK-LIR:           %[[VAL_6:.*]] = call @sparsePointers(%[[VAL_0]], %[[VAL_5]]) : (!llvm.ptr<i8>, index) -> memref<?xindex>
// CHECK-LIR:           %[[VAL_7:.*]] = call @sparseIndices(%[[VAL_0]], %[[VAL_5]]) : (!llvm.ptr<i8>, index) -> memref<?xindex>
// CHECK-LIR:           %[[VAL_8:.*]] = call @sparseValuesF64(%[[VAL_0]]) : (!llvm.ptr<i8>) -> memref<?xf64>
// CHECK-LIR:           scf.for %[[VAL_9:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_5]] {
// CHECK-LIR:             %[[VAL_10:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_9]]] : memref<?xindex>
// CHECK-LIR:             %[[VAL_11:.*]] = addi %[[VAL_9]], %[[VAL_5]] : index
// CHECK-LIR:             %[[VAL_12:.*]] = memref.load %[[VAL_6]]{{\[}}%[[VAL_11]]] : memref<?xindex>
// CHECK-LIR:             %[[VAL_13:.*]] = memref.load %[[VAL_2]]{{\[}}%[[VAL_9]]] : memref<32xf64>
// CHECK-LIR:             %[[VAL_14:.*]] = scf.for %[[VAL_15:.*]] = %[[VAL_10]] to %[[VAL_12]] step %[[VAL_5]] iter_args(%[[VAL_16:.*]] = %[[VAL_13]]) -> (f64) {
// CHECK-LIR:               %[[VAL_17:.*]] = memref.load %[[VAL_7]]{{\[}}%[[VAL_15]]] : memref<?xindex>
// CHECK-LIR:               %[[VAL_18:.*]] = memref.load %[[VAL_8]]{{\[}}%[[VAL_15]]] : memref<?xf64>
// CHECK-LIR:               %[[VAL_19:.*]] = memref.load %[[VAL_1]]{{\[}}%[[VAL_17]]] : memref<64xf64>
// CHECK-LIR:               %[[VAL_20:.*]] = mulf %[[VAL_18]], %[[VAL_19]] : f64
// CHECK-LIR:               %[[VAL_21:.*]] = addf %[[VAL_16]], %[[VAL_20]] : f64
// CHECK-LIR:               scf.yield %[[VAL_21]] : f64
// CHECK-LIR:             }
// CHECK-LIR:             memref.store %[[VAL_22:.*]], %[[VAL_2]]{{\[}}%[[VAL_9]]] : memref<32xf64>
// CHECK-LIR:           }
// CHECK-LIR:           return %[[VAL_2]] : memref<32xf64>
// CHECK-LIR:         }

func @matvec(%arga: tensor<32x64xf64, #CSR>,
             %argb: tensor<64xf64>,
	     %argx: tensor<32xf64> {linalg.inplaceable = true}) -> tensor<32xf64> {
  %0 = linalg.generic #trait_matvec
      ins(%arga, %argb : tensor<32x64xf64, #CSR>, tensor<64xf64>)
      outs(%argx: tensor<32xf64>) {
    ^bb(%A: f64, %b: f64, %x: f64):
      %0 = mulf %A, %b : f64
      %1 = addf %x, %0 : f64
      linalg.yield %1 : f64
  } -> tensor<32xf64>
  return %0 : tensor<32xf64>
}
