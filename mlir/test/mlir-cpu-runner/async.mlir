// RUN:   mlir-opt %s -async-to-async-runtime                                  \
// RUN:               -async-runtime-ref-counting                              \
// RUN:               -async-runtime-ref-counting-opt                          \
// RUN:               -convert-async-to-llvm                                   \
// RUN:               -convert-linalg-to-loops                                 \
// RUN:               -convert-scf-to-std                                      \
// RUN:               -convert-linalg-to-llvm                                  \
// RUN:               -convert-memref-to-llvm                                  \
// RUN:               -convert-arith-to-llvm                                   \
// RUN:               -convert-std-to-llvm                                     \
// RUN:               -reconcile-unrealized-casts                              \
// RUN: | mlir-cpu-runner                                                      \
// RUN:     -e main -entry-point-result=void -O0                               \
// RUN:     -shared-libs=%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext  \
// RUN:     -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext    \
// RUN:     -shared-libs=%linalg_test_lib_dir/libmlir_async_runtime%shlibext   \
// RUN: | FileCheck %s

func @main() {
  %i0 = arith.constant 0 : index
  %i1 = arith.constant 1 : index
  %i2 = arith.constant 2 : index
  %i3 = arith.constant 3 : index

  %c0 = arith.constant 0.0 : f32
  %c1 = arith.constant 1.0 : f32
  %c2 = arith.constant 2.0 : f32
  %c3 = arith.constant 3.0 : f32
  %c4 = arith.constant 4.0 : f32

  %A = memref.alloc() : memref<4xf32>
  linalg.fill(%c0, %A) : f32, memref<4xf32>

  // CHECK: [0, 0, 0, 0]
  %U = memref.cast %A :  memref<4xf32> to memref<*xf32>
  call @print_memref_f32(%U): (memref<*xf32>) -> ()

  // CHECK: Current thread id: [[MAIN:.*]]
  // CHECK: [1, 0, 0, 0]
  memref.store %c1, %A[%i0]: memref<4xf32>
  call @mlirAsyncRuntimePrintCurrentThreadId(): () -> ()
  call @print_memref_f32(%U): (memref<*xf32>) -> ()

  %outer = async.execute {
    // CHECK: Current thread id: [[THREAD0:.*]]
    // CHECK: [1, 2, 0, 0]
    memref.store %c2, %A[%i1]: memref<4xf32>
    call @mlirAsyncRuntimePrintCurrentThreadId(): () -> ()
    call @print_memref_f32(%U): (memref<*xf32>) -> ()

    // No op async region to create a token for testing async dependency.
    %noop = async.execute {
      // CHECK: Current thread id: [[THREAD1:.*]]
      call @mlirAsyncRuntimePrintCurrentThreadId(): () -> ()
      async.yield
    }

    %inner = async.execute [%noop] {
      // CHECK: Current thread id: [[THREAD2:.*]]
      // CHECK: [1, 2, 3, 0]
      memref.store %c3, %A[%i2]: memref<4xf32>
      call @mlirAsyncRuntimePrintCurrentThreadId(): () -> ()
      call @print_memref_f32(%U): (memref<*xf32>) -> ()

      async.yield
    }
    async.await %inner : !async.token

    // CHECK: Current thread id: [[THREAD3:.*]]
    // CHECK: [1, 2, 3, 4]
    memref.store %c4, %A[%i3]: memref<4xf32>
    call @mlirAsyncRuntimePrintCurrentThreadId(): () -> ()
    call @print_memref_f32(%U): (memref<*xf32>) -> ()

    async.yield
  }
  async.await %outer : !async.token

  // CHECK: Current thread id: [[MAIN]]
  // CHECK: [1, 2, 3, 4]
  call @mlirAsyncRuntimePrintCurrentThreadId(): () -> ()
  call @print_memref_f32(%U): (memref<*xf32>) -> ()

  memref.dealloc %A : memref<4xf32>

  return
}

func private @mlirAsyncRuntimePrintCurrentThreadId() -> ()

func private @print_memref_f32(memref<*xf32>) attributes { llvm.emit_c_interface }
