// RUN: mlir-opt %s -convert-scf-to-std -convert-vector-to-llvm -convert-memref-to-llvm -convert-std-to-llvm | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

func @maskedstore16(%base: memref<?xf32>,
                    %mask: vector<16xi1>, %value: vector<16xf32>) {
  %c0 = constant 0: index
  vector.maskedstore %base[%c0], %mask, %value
    : memref<?xf32>, vector<16xi1>, vector<16xf32>
  return
}

func @maskedstore16_at8(%base: memref<?xf32>,
                        %mask: vector<16xi1>, %value: vector<16xf32>) {
  %c8 = constant 8: index
  vector.maskedstore %base[%c8], %mask, %value
    : memref<?xf32>, vector<16xi1>, vector<16xf32>
  return
}

func @printmem16(%A: memref<?xf32>) {
  %c0 = constant 0: index
  %c1 = constant 1: index
  %c16 = constant 16: index
  %z = constant 0.0: f32
  %m = vector.broadcast %z : f32 to vector<16xf32>
  %mem = scf.for %i = %c0 to %c16 step %c1
    iter_args(%m_iter = %m) -> (vector<16xf32>) {
    %c = memref.load %A[%i] : memref<?xf32>
    %i32 = index_cast %i : index to i32
    %m_new = vector.insertelement %c, %m_iter[%i32 : i32] : vector<16xf32>
    scf.yield %m_new : vector<16xf32>
  }
  vector.print %mem : vector<16xf32>
  return
}

func @entry() {
  // Set up memory.
  %f0 = constant 0.0: f32
  %c0 = constant 0: index
  %c1 = constant 1: index
  %c16 = constant 16: index
  %A = memref.alloc(%c16) : memref<?xf32>
  scf.for %i = %c0 to %c16 step %c1 {
    memref.store %f0, %A[%i] : memref<?xf32>
  }

  // Set up value vector.
  %v = vector.broadcast %f0 : f32 to vector<16xf32>
  %val = scf.for %i = %c0 to %c16 step %c1
    iter_args(%v_iter = %v) -> (vector<16xf32>) {
    %i32 = index_cast %i : index to i32
    %fi = sitofp %i32 : i32 to f32
    %v_new = vector.insertelement %fi, %v_iter[%i32 : i32] : vector<16xf32>
    scf.yield %v_new : vector<16xf32>
  }

  // Set up masks.
  %t = constant 1: i1
  %none = vector.constant_mask [0] : vector<16xi1>
  %some = vector.constant_mask [8] : vector<16xi1>
  %more = vector.insert %t, %some[13] : i1 into vector<16xi1>
  %all = vector.constant_mask [16] : vector<16xi1>

  //
  // Masked store tests.
  //

  vector.print %val : vector<16xf32>
  // CHECK: ( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 )

  call @printmem16(%A) : (memref<?xf32>) -> ()
  // CHECK: ( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )

  call @maskedstore16(%A, %none, %val)
    : (memref<?xf32>, vector<16xi1>, vector<16xf32>) -> ()
  call @printmem16(%A) : (memref<?xf32>) -> ()
  // CHECK: ( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 )

  call @maskedstore16(%A, %some, %val)
    : (memref<?xf32>, vector<16xi1>, vector<16xf32>) -> ()
  call @printmem16(%A) : (memref<?xf32>) -> ()
  // CHECK: ( 0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0 )

  call @maskedstore16(%A, %more, %val)
    : (memref<?xf32>, vector<16xi1>, vector<16xf32>) -> ()
  call @printmem16(%A) : (memref<?xf32>) -> ()
  // CHECK: ( 0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 13, 0, 0 )

  call @maskedstore16(%A, %all, %val)
    : (memref<?xf32>, vector<16xi1>, vector<16xf32>) -> ()
  call @printmem16(%A) : (memref<?xf32>) -> ()
  // CHECK: ( 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 )

  call @maskedstore16_at8(%A, %some, %val)
    : (memref<?xf32>, vector<16xi1>, vector<16xf32>) -> ()
  call @printmem16(%A) : (memref<?xf32>) -> ()
  // CHECK: ( 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7 )

  return
}
