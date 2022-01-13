// RUN: mlir-opt %s -canonicalize --split-input-file -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: @memcpy_after_cast
func @memcpy_after_cast(%arg0: memref<10xf32>, %arg1: memref<10xf32>) {
  // CHECK-NOT: memref.cast
  // CHECK: gpu.memcpy
  %0 = memref.cast %arg0 : memref<10xf32> to memref<?xf32>
  %1 = memref.cast %arg1 : memref<10xf32> to memref<?xf32>
  gpu.memcpy %0, %1 : memref<?xf32>, memref<?xf32>
  return
}

// CHECK-LABEL: @memset_after_cast
func @memset_after_cast(%arg0: memref<10xf32>, %arg1: f32) {
  // CHECK-NOT: memref.cast
  // CHECK: gpu.memset
  %0 = memref.cast %arg0 : memref<10xf32> to memref<?xf32>
  gpu.memset %0, %arg1 : memref<?xf32>, f32
  return
}

// -----

// Test case: Folding of memref.dim(gpu.alloc(%size), %idx) -> %size
// CHECK-LABEL: func @gpu_dim_of_alloc(
//  CHECK-SAME:     %[[SIZE:[0-9a-z]+]]: index
//  CHECK-NEXT:   return %[[SIZE]] : index
func @gpu_dim_of_alloc(%size: index) -> index {
  %0 = gpu.alloc(%size) : memref<?xindex>
  %c0 = arith.constant 0 : index
  %1 = memref.dim %0, %c0 : memref<?xindex>
  return %1 : index
}

// -----

// CHECK-LABEL: func @simplify_gpu_launch
func @simplify_gpu_launch() attributes {llvm.emit_c_interface} {
  %cst = arith.constant 0.000000e+00 : f32
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %c16 = arith.constant 16 : index
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %0 = memref.alloc() : memref<2x16x16xf32>
  scf.for %arg0 = %c0 to %c2 step %c1 {
    scf.for %arg1 = %c0 to %c16 step %c1 {
      scf.for %arg2 = %c0 to %c16 step %c1 {
        memref.store %cst, %0[%arg0, %arg1, %arg2] : memref<2x16x16xf32>
      }
    }
  }
  %1 = gpu.wait async
  %memref, %asyncToken = gpu.alloc async [%1] () : memref<2x16x16xf32>
  %2 = gpu.memcpy async [%1] %memref, %0 : memref<2x16x16xf32>, memref<2x16x16xf32>
  gpu.wait [%1]
  gpu.launch blocks(%arg0, %arg1, %arg2) in (%arg6 = %c1, %arg7 = %c1, %arg8 = %c1)
    threads(%arg3, %arg4, %arg5) in (%arg9 = %c32, %arg10 = %c1, %arg11 = %c1) {
    %3 = arith.muli %arg5, %c32 : index
    %4 = arith.muli %arg4, %c32 : index
    %5 = arith.addi %3, %4 : index
    %6 = arith.addi %5, %arg3 : index
    %7 = arith.divui %6, %c32 : index
    %8 = arith.muli %arg0, %c16 : index
    %9 = arith.muli %arg1, %c2 : index
    %10 = arith.muli %7, %c2 : index
    %11 = arith.addi %9, %10 : index
    %12 = memref.load %memref[%11, %c0, %8] : memref<2x16x16xf32>
    %13 = arith.addi %11, %c1 : index
    %14 = memref.load %memref[%13, %c0, %8] : memref<2x16x16xf32>
    memref.store %12, %memref[%11, %c0, %8] : memref<2x16x16xf32>
    memref.store %14, %memref[%13, %c0, %8] : memref<2x16x16xf32>
    gpu.terminator
  }
  return
}

// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK: gpu.launch blocks(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}} = %[[C1]], %{{.*}} = %[[C1]], %{{.*}} = %[[C1]]) threads(%[[TIDX:.*]], %{{.*}}, %{{.*}}) in (%{{.*}} = %c32, %{{.*}} = %[[C1]], %{{.*}} = %[[C1]]) {
// CHECK-NEXT:  	arith.divui %[[TIDX]], %c32 : index
// CHECK-NEXT:  	arith.muli %{{.*}}, %c2 : index
// CHECK-NEXT:    memref.load %memref[%{{.*}}, %[[C0]], %[[C0]]] : memref<2x16x16xf32>
// CHECK-NEXT:    arith.addi %{{.*}}, %[[C1]] : index
// CHECK-NEXT:    memref.load %memref[%{{.*}}, %[[C0]], %[[C0]]] : memref<2x16x16xf32>
// CHECK-NEXT:    memref.store %{{.*}}, %memref[%{{.*}}, %[[C0]], %[[C0]]] : memref<2x16x16xf32>
// CHECK-NEXT:    memref.store %{{.*}}, %memref[%{{.*}}, %[[C0]], %[[C0]]] : memref<2x16x16xf32>
// CHECK-NEXT:    gpu.terminator
// CHECK-NEXT:  }
