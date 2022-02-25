// RUN: mlir-opt %s -convert-linalg-to-loops -convert-scf-to-cf -convert-arith-to-llvm -convert-linalg-to-llvm -convert-memref-to-llvm -convert-std-to-llvm -reconcile-unrealized-casts | mlir-cpu-runner -e print_0d -entry-point-result=void -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext | FileCheck %s --check-prefix=PRINT-0D
// RUN: mlir-opt %s -convert-linalg-to-loops -convert-scf-to-cf -convert-arith-to-llvm -convert-linalg-to-llvm -convert-memref-to-llvm -convert-std-to-llvm -reconcile-unrealized-casts | mlir-cpu-runner -e print_1d -entry-point-result=void -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext | FileCheck %s --check-prefix=PRINT-1D
// RUN: mlir-opt %s -convert-linalg-to-loops -convert-scf-to-cf -convert-arith-to-llvm -convert-linalg-to-llvm -convert-memref-to-llvm -convert-std-to-llvm -reconcile-unrealized-casts | mlir-cpu-runner -e print_3d -entry-point-result=void -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext | FileCheck %s --check-prefix=PRINT-3D
// RUN: mlir-opt %s -convert-linalg-to-loops -convert-scf-to-cf -convert-arith-to-llvm -convert-linalg-to-llvm -convert-memref-to-llvm -convert-std-to-llvm -reconcile-unrealized-casts | mlir-cpu-runner -e vector_splat_2d -entry-point-result=void -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext | FileCheck %s --check-prefix=PRINT-VECTOR-SPLAT-2D

func @print_0d() {
  %f = arith.constant 2.00000e+00 : f32
  %A = memref.alloc() : memref<f32>
  memref.store %f, %A[]: memref<f32>
  %U = memref.cast %A :  memref<f32> to memref<*xf32>
  call @print_memref_f32(%U): (memref<*xf32>) -> ()
  memref.dealloc %A : memref<f32>
  return
}
// PRINT-0D: Unranked Memref base@ = {{.*}} rank = 0 offset = 0 sizes = [] strides = [] data =
// PRINT-0D: [2]

func @print_1d() {
  %f = arith.constant 2.00000e+00 : f32
  %A = memref.alloc() : memref<16xf32>
  %B = memref.cast %A: memref<16xf32> to memref<?xf32>
  linalg.fill(%f, %B) : f32, memref<?xf32>
  %U = memref.cast %B :  memref<?xf32> to memref<*xf32>
  call @print_memref_f32(%U): (memref<*xf32>) -> ()
  memref.dealloc %A : memref<16xf32>
  return
}
// PRINT-1D: Unranked Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [16] strides = [1] data =
// PRINT-1D-NEXT: [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

func @print_3d() {
  %f = arith.constant 2.00000e+00 : f32
  %f4 = arith.constant 4.00000e+00 : f32
  %A = memref.alloc() : memref<3x4x5xf32>
  %B = memref.cast %A: memref<3x4x5xf32> to memref<?x?x?xf32>
  linalg.fill(%f, %B) : f32, memref<?x?x?xf32>

  %c2 = arith.constant 2 : index
  memref.store %f4, %B[%c2, %c2, %c2]: memref<?x?x?xf32>
  %U = memref.cast %B : memref<?x?x?xf32> to memref<*xf32>
  call @print_memref_f32(%U): (memref<*xf32>) -> ()
  memref.dealloc %A : memref<3x4x5xf32>
  return
}
// PRINT-3D: Unranked Memref base@ = {{.*}} rank = 3 offset = 0 sizes = [3, 4, 5] strides = [20, 5, 1] data =
// PRINT-3D-COUNT-4: {{.*[[:space:]].*}}2,    2,    2,    2,    2
// PRINT-3D-COUNT-4: {{.*[[:space:]].*}}2,    2,    2,    2,    2
// PRINT-3D-COUNT-2: {{.*[[:space:]].*}}2,    2,    2,    2,    2
//    PRINT-3D-NEXT: 2,    2,    4,    2,    2
//    PRINT-3D-NEXT: 2,    2,    2,    2,    2

func private @print_memref_f32(memref<*xf32>) attributes { llvm.emit_c_interface }

!vector_type_C = type vector<4x4xf32>
!matrix_type_CC = type memref<1x1x!vector_type_C>
func @vector_splat_2d() {
  %c0 = arith.constant 0 : index
  %f10 = arith.constant 10.0 : f32
  %vf10 = vector.splat %f10: !vector_type_C
  %C = memref.alloc() : !matrix_type_CC
  memref.store %vf10, %C[%c0, %c0]: !matrix_type_CC

  %CC = memref.cast %C: !matrix_type_CC to memref<?x?x!vector_type_C>
  call @print_memref_vector_4x4xf32(%CC): (memref<?x?x!vector_type_C>) -> ()

  memref.dealloc %C : !matrix_type_CC
  return
}

// PRINT-VECTOR-SPLAT-2D: Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [1, 1] strides = [1, 1] data =
// PRINT-VECTOR-SPLAT-2D-NEXT: [((10, 10, 10, 10),   (10, 10, 10, 10),   (10, 10, 10, 10),   (10, 10, 10, 10))]

func private @print_memref_vector_4x4xf32(memref<?x?x!vector_type_C>) attributes { llvm.emit_c_interface }
