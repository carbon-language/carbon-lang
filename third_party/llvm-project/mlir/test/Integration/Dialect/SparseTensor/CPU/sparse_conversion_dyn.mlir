// RUN: mlir-opt %s \
// RUN:   --sparsification --sparse-tensor-conversion \
// RUN:   --linalg-bufferize --convert-linalg-to-loops \
// RUN:   --convert-vector-to-scf --convert-scf-to-std \
// RUN:   --func-bufferize --tensor-constant-bufferize --tensor-bufferize \
// RUN:   --std-bufferize --finalizing-bufferize --lower-affine \
// RUN:   --convert-vector-to-llvm --convert-memref-to-llvm --convert-math-to-llvm \
// RUN:   --convert-std-to-llvm --reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

#DCSR  = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed" ]
}>

#DCSC  = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (j,i)>
}>

//
// Integration test that tests conversions between sparse tensors,
// where the dynamic sizes of the shape of the enveloping tensor
// may change (the actual underlying sizes obviously never change).
//
module {

  //
  // Helper method to print values array. The transfer actually
  // reads more than required to verify size of buffer as well.
  //
  func @dump(%arg0: memref<?xf64>) {
    %c = arith.constant 0 : index
    %d = arith.constant -1.0 : f64
    %0 = vector.transfer_read %arg0[%c], %d: memref<?xf64>, vector<8xf64>
    vector.print %0 : vector<8xf64>
    return
  }

  func @entry() {
    %t1 = arith.constant sparse<
      [ [0,0], [0,1], [0,63], [1,0], [1,1], [31,0], [31,63] ],
        [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 ]> : tensor<32x64xf64>
    %t2 = tensor.cast %t1 : tensor<32x64xf64> to tensor<?x?xf64>

    // Four dense to sparse conversions.
    %1 = sparse_tensor.convert %t1 : tensor<32x64xf64> to tensor<?x?xf64, #DCSR>
    %2 = sparse_tensor.convert %t1 : tensor<32x64xf64> to tensor<?x?xf64, #DCSC>
    %3 = sparse_tensor.convert %t2 : tensor<?x?xf64> to tensor<?x?xf64, #DCSR>
    %4 = sparse_tensor.convert %t2 : tensor<?x?xf64> to tensor<?x?xf64, #DCSC>

    // Two cross conversions.
    %5 = sparse_tensor.convert %3 : tensor<?x?xf64, #DCSR> to tensor<?x?xf64, #DCSC>
    %6 = sparse_tensor.convert %4 : tensor<?x?xf64, #DCSC> to tensor<?x?xf64, #DCSR>

    //
    // All proper row-/column-wise?
    //
    // CHECK: ( 1, 2, 3, 4, 5, 6, 7, -1 )
    // CHECK: ( 1, 4, 6, 2, 5, 3, 7, -1 )
    // CHECK: ( 1, 2, 3, 4, 5, 6, 7, -1 )
    // CHECK: ( 1, 4, 6, 2, 5, 3, 7, -1 )
    // CHECK: ( 1, 4, 6, 2, 5, 3, 7, -1 )
    // CHECK: ( 1, 2, 3, 4, 5, 6, 7, -1 )
    //
    %m1 = sparse_tensor.values %1 : tensor<?x?xf64, #DCSR> to memref<?xf64>
    %m2 = sparse_tensor.values %2 : tensor<?x?xf64, #DCSC> to memref<?xf64>
    %m3 = sparse_tensor.values %3 : tensor<?x?xf64, #DCSR> to memref<?xf64>
    %m4 = sparse_tensor.values %4 : tensor<?x?xf64, #DCSC> to memref<?xf64>
    %m5 = sparse_tensor.values %5 : tensor<?x?xf64, #DCSC> to memref<?xf64>
    %m6 = sparse_tensor.values %6 : tensor<?x?xf64, #DCSR> to memref<?xf64>
    call @dump(%m1) : (memref<?xf64>) -> ()
    call @dump(%m2) : (memref<?xf64>) -> ()
    call @dump(%m3) : (memref<?xf64>) -> ()
    call @dump(%m4) : (memref<?xf64>) -> ()
    call @dump(%m5) : (memref<?xf64>) -> ()
    call @dump(%m6) : (memref<?xf64>) -> ()

    // Release the resources.
    sparse_tensor.release %1 : tensor<?x?xf64, #DCSR>
    sparse_tensor.release %2 : tensor<?x?xf64, #DCSC>
    sparse_tensor.release %3 : tensor<?x?xf64, #DCSR>
    sparse_tensor.release %4 : tensor<?x?xf64, #DCSC>
    sparse_tensor.release %5 : tensor<?x?xf64, #DCSC>
    sparse_tensor.release %6 : tensor<?x?xf64, #DCSR>

    return
  }
}
