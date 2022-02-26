// RUN: mlir-opt %s --sparse-compiler | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

#DCSR  = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed" ],
  pointerBitWidth = 8,
  indexBitWidth = 8
}>

#DCSC  = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (j,i)>,
  pointerBitWidth = 64,
  indexBitWidth = 64
}>

#CSC  = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (j,i)>,
  pointerBitWidth = 16,
  indexBitWidth = 32
}>

//
// Integration test that tests conversions between sparse tensors,
// where the pointer and index sizes in the overhead storage change
// in addition to layout.
//
module {

  //
  // Helper method to print values and indices arrays. The transfer actually
  // reads more than required to verify size of buffer as well.
  //
  func @dumpf64(%arg0: memref<?xf64>) {
    %c = arith.constant 0 : index
    %d = arith.constant -1.0 : f64
    %0 = vector.transfer_read %arg0[%c], %d: memref<?xf64>, vector<8xf64>
    vector.print %0 : vector<8xf64>
    return
  }
  func @dumpi08(%arg0: memref<?xi8>) {
    %c = arith.constant 0 : index
    %d = arith.constant -1 : i8
    %0 = vector.transfer_read %arg0[%c], %d: memref<?xi8>, vector<8xi8>
    vector.print %0 : vector<8xi8>
    return
  }
  func @dumpi32(%arg0: memref<?xi32>) {
    %c = arith.constant 0 : index
    %d = arith.constant -1 : i32
    %0 = vector.transfer_read %arg0[%c], %d: memref<?xi32>, vector<8xi32>
    vector.print %0 : vector<8xi32>
    return
  }
  func @dumpi64(%arg0: memref<?xi64>) {
    %c = arith.constant 0 : index
    %d = arith.constant -1 : i64
    %0 = vector.transfer_read %arg0[%c], %d: memref<?xi64>, vector<8xi64>
    vector.print %0 : vector<8xi64>
    return
  }

  func @entry() {
    %c1 = arith.constant 1 : index
    %t1 = arith.constant sparse<
      [ [0,0], [0,1], [0,63], [1,0], [1,1], [31,0], [31,63] ],
        [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 ]> : tensor<32x64xf64>
    %t2 = tensor.cast %t1 : tensor<32x64xf64> to tensor<?x?xf64>

    // Dense to sparse.
    %1 = sparse_tensor.convert %t1 : tensor<32x64xf64> to tensor<32x64xf64, #DCSR>
    %2 = sparse_tensor.convert %t1 : tensor<32x64xf64> to tensor<32x64xf64, #DCSC>
    %3 = sparse_tensor.convert %t1 : tensor<32x64xf64> to tensor<32x64xf64, #CSC>

    // Sparse to sparse.
    %4 = sparse_tensor.convert %1 : tensor<32x64xf64, #DCSR> to tensor<32x64xf64, #DCSC>
    %5 = sparse_tensor.convert %2 : tensor<32x64xf64, #DCSC> to tensor<32x64xf64, #DCSR>
    %6 = sparse_tensor.convert %3 : tensor<32x64xf64, #CSC>  to tensor<32x64xf64, #DCSR>

    //
    // All proper row-/column-wise?
    //
    // CHECK:      ( 1, 2, 3, 4, 5, 6, 7, -1 )
    // CHECK-NEXT: ( 1, 4, 6, 2, 5, 3, 7, -1 )
    // CHECK-NEXT: ( 1, 4, 6, 2, 5, 3, 7, -1 )
    // CHECK-NEXT: ( 1, 4, 6, 2, 5, 3, 7, -1 )
    // CHECK-NEXT: ( 1, 2, 3, 4, 5, 6, 7, -1 )
    // CHECK-NEXT: ( 1, 2, 3, 4, 5, 6, 7, -1 )
    //
    %m1 = sparse_tensor.values %1 : tensor<32x64xf64, #DCSR> to memref<?xf64>
    %m2 = sparse_tensor.values %2 : tensor<32x64xf64, #DCSC> to memref<?xf64>
    %m3 = sparse_tensor.values %3 : tensor<32x64xf64, #CSC>  to memref<?xf64>
    %m4 = sparse_tensor.values %4 : tensor<32x64xf64, #DCSC> to memref<?xf64>
    %m5 = sparse_tensor.values %5 : tensor<32x64xf64, #DCSR> to memref<?xf64>
    %m6 = sparse_tensor.values %6 : tensor<32x64xf64, #DCSR> to memref<?xf64>
    call @dumpf64(%m1) : (memref<?xf64>) -> ()
    call @dumpf64(%m2) : (memref<?xf64>) -> ()
    call @dumpf64(%m3) : (memref<?xf64>) -> ()
    call @dumpf64(%m4) : (memref<?xf64>) -> ()
    call @dumpf64(%m5) : (memref<?xf64>) -> ()
    call @dumpf64(%m6) : (memref<?xf64>) -> ()

    //
    // Sanity check on indices.
    //
    // CHECK-NEXT: ( 0, 1, 63, 0, 1, 0, 63, -1 )
    // CHECK-NEXT: ( 0, 1, 31, 0, 1, 0, 31, -1 )
    // CHECK-NEXT: ( 0, 1, 31, 0, 1, 0, 31, -1 )
    // CHECK-NEXT: ( 0, 1, 31, 0, 1, 0, 31, -1 )
    // CHECK-NEXT: ( 0, 1, 63, 0, 1, 0, 63, -1 )
    // CHECK-NEXT: ( 0, 1, 63, 0, 1, 0, 63, -1 )
    //
    %i1 = sparse_tensor.indices %1, %c1 : tensor<32x64xf64, #DCSR> to memref<?xi8>
    %i2 = sparse_tensor.indices %2, %c1 : tensor<32x64xf64, #DCSC> to memref<?xi64>
    %i3 = sparse_tensor.indices %3, %c1 : tensor<32x64xf64, #CSC>  to memref<?xi32>
    %i4 = sparse_tensor.indices %4, %c1 : tensor<32x64xf64, #DCSC> to memref<?xi64>
    %i5 = sparse_tensor.indices %5, %c1 : tensor<32x64xf64, #DCSR> to memref<?xi8>
    %i6 = sparse_tensor.indices %6, %c1 : tensor<32x64xf64, #DCSR> to memref<?xi8>
    call @dumpi08(%i1) : (memref<?xi8>)  -> ()
    call @dumpi64(%i2) : (memref<?xi64>) -> ()
    call @dumpi32(%i3) : (memref<?xi32>) -> ()
    call @dumpi64(%i4) : (memref<?xi64>) -> ()
    call @dumpi08(%i5) : (memref<?xi08>) -> ()
    call @dumpi08(%i6) : (memref<?xi08>) -> ()

    // Release the resources.
    sparse_tensor.release %1 : tensor<32x64xf64, #DCSR>
    sparse_tensor.release %2 : tensor<32x64xf64, #DCSC>
    sparse_tensor.release %3 : tensor<32x64xf64, #CSC>
    sparse_tensor.release %4 : tensor<32x64xf64, #DCSC>
    sparse_tensor.release %5 : tensor<32x64xf64, #DCSR>
    sparse_tensor.release %6 : tensor<32x64xf64, #DCSR>

    return
  }
}
