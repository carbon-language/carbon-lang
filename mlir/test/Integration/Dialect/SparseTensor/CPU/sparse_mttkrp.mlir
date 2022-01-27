// RUN: mlir-opt %s \
// RUN:   --sparsification --sparse-tensor-conversion \
// RUN:   --convert-vector-to-scf --convert-scf-to-std \
// RUN:   --func-bufferize --tensor-constant-bufferize --tensor-bufferize \
// RUN:   --std-bufferize --finalizing-bufferize --lower-affine \
// RUN:   --convert-vector-to-llvm --convert-memref-to-llvm --convert-std-to-llvm --reconcile-unrealized-casts | \
// RUN: TENSOR0="%mlir_integration_test_dir/data/mttkrp_b.tns" \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s
//
// Do the same run, but now with SIMDization as well. This should not change the outcome.
//
// RUN: mlir-opt %s \
// RUN:   --sparsification="vectorization-strategy=2 vl=4" --sparse-tensor-conversion \
// RUN:   --convert-vector-to-scf --convert-scf-to-std \
// RUN:   --func-bufferize --tensor-constant-bufferize --tensor-bufferize \
// RUN:   --std-bufferize --finalizing-bufferize --lower-affine \
// RUN:   --convert-vector-to-llvm --convert-memref-to-llvm --convert-std-to-llvm --reconcile-unrealized-casts | \
// RUN: TENSOR0="%mlir_integration_test_dir/data/mttkrp_b.tns" \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

!Filename = type !llvm.ptr<i8>

#SparseTensor = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed", "compressed" ]
}>

#mttkrp = {
  indexing_maps = [
    affine_map<(i,j,k,l) -> (i,k,l)>, // B
    affine_map<(i,j,k,l) -> (k,j)>,   // C
    affine_map<(i,j,k,l) -> (l,j)>,   // D
    affine_map<(i,j,k,l) -> (i,j)>    // A (out)
  ],
  iterator_types = ["parallel", "parallel", "reduction", "reduction"],
  doc = "A(i,j) += B(i,k,l) * D(l,j) * C(k,j)"
}

//
// Integration test that lowers a kernel annotated as sparse to
// actual sparse code, initializes a matching sparse storage scheme
// from file, and runs the resulting code with the JIT compiler.
//
module {
  //
  // Computes Matricized Tensor Times Khatri-Rao Product (MTTKRP) kernel. See
  // http://tensor-compiler.org/docs/data_analytics/index.html.
  //
  func @kernel_mttkrp(%argb: tensor<?x?x?xf64, #SparseTensor>,
                      %argc: tensor<?x?xf64>,
                      %argd: tensor<?x?xf64>,
                      %arga: tensor<?x?xf64> {linalg.inplaceable = true})
		      -> tensor<?x?xf64> {
    %0 = linalg.generic #mttkrp
      ins(%argb, %argc, %argd:
            tensor<?x?x?xf64, #SparseTensor>, tensor<?x?xf64>, tensor<?x?xf64>)
      outs(%arga: tensor<?x?xf64>) {
      ^bb(%b: f64, %c: f64, %d: f64, %a: f64):
        %0 = arith.mulf %b, %c : f64
        %1 = arith.mulf %d, %0 : f64
        %2 = arith.addf %a, %1 : f64
        linalg.yield %2 : f64
    } -> tensor<?x?xf64>
    return %0 : tensor<?x?xf64>
  }

  func private @getTensorFilename(index) -> (!Filename)

  //
  // Main driver that reads matrix from file and calls the sparse kernel.
  //
  func @entry() {
    %f0 = arith.constant 0.0 : f64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    // Read the sparse input tensor B from a file.
    %fileName = call @getTensorFilename(%c0) : (index) -> (!Filename)
    %b = sparse_tensor.new %fileName
          : !Filename to tensor<?x?x?xf64, #SparseTensor>

    // Get sizes from B, pick a fixed size for dim-2 of A.
    %isz = tensor.dim %b, %c0 : tensor<?x?x?xf64, #SparseTensor>
    %jsz = arith.constant 5 : index
    %ksz = tensor.dim %b, %c1 : tensor<?x?x?xf64, #SparseTensor>
    %lsz = tensor.dim %b, %c2 : tensor<?x?x?xf64, #SparseTensor>

    // Initialize dense input matrix C.
    %cdata = memref.alloc(%ksz, %jsz) : memref<?x?xf64>
    scf.for %k = %c0 to %ksz step %c1 {
      scf.for %j = %c0 to %jsz step %c1 {
        %k0 = arith.muli %k, %jsz : index
        %k1 = arith.addi %k0, %j : index
        %k2 = arith.index_cast %k1 : index to i32
        %kf = arith.sitofp %k2 : i32 to f64
        memref.store %kf, %cdata[%k, %j] : memref<?x?xf64>
      }
    }
    %c = bufferization.to_tensor %cdata : memref<?x?xf64>

    // Initialize dense input matrix D.
    %ddata = memref.alloc(%lsz, %jsz) : memref<?x?xf64>
    scf.for %l = %c0 to %lsz step %c1 {
      scf.for %j = %c0 to %jsz step %c1 {
        %k0 = arith.muli %l, %jsz : index
        %k1 = arith.addi %k0, %j : index
        %k2 = arith.index_cast %k1 : index to i32
        %kf = arith.sitofp %k2 : i32 to f64
        memref.store %kf, %ddata[%l, %j] : memref<?x?xf64>
      }
    }
    %d = bufferization.to_tensor %ddata : memref<?x?xf64>

    // Initialize dense output matrix A.
    %adata = memref.alloc(%isz, %jsz) : memref<?x?xf64>
    scf.for %i = %c0 to %isz step %c1 {
      scf.for %j = %c0 to %jsz step %c1 {
        memref.store %f0, %adata[%i, %j] : memref<?x?xf64>
      }
    }
    %a = bufferization.to_tensor %adata : memref<?x?xf64>

    // Call kernel.
    %0 = call @kernel_mttkrp(%b, %c, %d, %a)
      : (tensor<?x?x?xf64, #SparseTensor>,
        tensor<?x?xf64>, tensor<?x?xf64>, tensor<?x?xf64>) -> tensor<?x?xf64>

    // Print the result for verification.
    //
    // CHECK: ( ( 16075, 21930, 28505, 35800, 43815 ),
    // CHECK:   ( 10000, 14225, 19180, 24865, 31280 ) )
    //
    %m = bufferization.to_memref %0 : memref<?x?xf64>
    %v = vector.transfer_read %m[%c0, %c0], %f0
          : memref<?x?xf64>, vector<2x5xf64>
    vector.print %v : vector<2x5xf64>

    // Release the resources.
    memref.dealloc %adata : memref<?x?xf64>
    memref.dealloc %cdata : memref<?x?xf64>
    memref.dealloc %ddata : memref<?x?xf64>
    sparse_tensor.release %b : tensor<?x?x?xf64, #SparseTensor>

    return
  }
}
