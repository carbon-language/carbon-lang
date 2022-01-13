// RUN: mlir-opt -convert-affine-for-to-gpu="gpu-block-dims=1 gpu-thread-dims=1" %s | FileCheck %s

// CHECK-LABEL: @step_var
func @step_var(%A : memref<?x?xf32>, %B : memref<?x?xf32>) {
  // Check that we divide by step.
  // CHECK:  %[[range_i:.*]] = divi_signed {{.*}}, %{{.*}}
  // CHECK:  %[[range_j:.*]] = divi_signed {{.*}}, %{{.*}}

  // CHECK: gpu.launch
  // CHECK-SAME: blocks(%{{[^)]*}}, %{{[^)]*}}, %{{[^)]*}}) in (%{{[^)]*}} = %[[range_i]], %{{[^)]*}} = %{{[^)]*}}, %{{[^)]*}} = %{{[^)]*}})
  // CHECK-SAME: threads(%{{[^)]*}}, %{{[^)]*}}, %{{[^)]*}}) in (%{{[^)]*}} = %[[range_j]], %{{[^)]*}} = %{{[^)]*}}, %{{[^)]*}} = %{{[^)]*}})
  affine.for %i = 5 to 15 step 4 {
    affine.for %j = 3 to 19 step 7 {
      // Loop induction variable remapping:
      //     iv = thread(block)_id * step + lower_bound
      // CHECK:      %[[prod_i:.*]] = muli %{{.*}}, %{{.*}} : index
      // CHECK-NEXT: %[[i:.*]] = addi %{{.*}}, %[[prod_i]] : index
      // CHECK-NEXT: %[[prod_j:.*]] = muli %{{.*}}, %{{.*}} : index
      // CHECK-NEXT: %[[j:.*]] = addi %{{.*}}, %[[prod_j]] : index

      // CHECK:     {{.*}} = memref.load %{{.*}}[%[[i]], %[[j]]] : memref<?x?xf32>
      %0 = memref.load %A[%i, %j] : memref<?x?xf32>
      // CHECK:     memref.store {{.*}}, %{{.*}}[%[[i]], %[[j]]] : memref<?x?xf32>
      memref.store %0, %B[%i, %j] : memref<?x?xf32>
    }
  }
  return
}
