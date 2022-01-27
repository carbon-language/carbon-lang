// RUN: mlir-opt -allow-unregistered-dialect --convert-gpu-to-nvvm --split-input-file %s | FileCheck --check-prefix=NVVM %s
// RUN: mlir-opt -allow-unregistered-dialect --convert-gpu-to-rocdl --split-input-file %s | FileCheck --check-prefix=ROCDL %s

gpu.module @kernel {
  // NVVM-LABEL:  llvm.func @private
  gpu.func @private(%arg0: f32) private(%arg1: memref<4xf32, 5>) {
    // Allocate private memory inside the function.
    // NVVM: %[[size:.*]] = llvm.mlir.constant(4 : i64) : i64
    // NVVM: %[[raw:.*]] = llvm.alloca %[[size]] x f32 : (i64) -> !llvm.ptr<f32>

    // ROCDL: %[[size:.*]] = llvm.mlir.constant(4 : i64) : i64
    // ROCDL: %[[raw:.*]] = llvm.alloca %[[size]] x f32 : (i64) -> !llvm.ptr<f32, 5>

    // Populate the memref descriptor.
    // NVVM: %[[descr1:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    // NVVM: %[[descr2:.*]] = llvm.insertvalue %[[raw]], %[[descr1]][0]
    // NVVM: %[[descr3:.*]] = llvm.insertvalue %[[raw]], %[[descr2]][1]
    // NVVM: %[[c0:.*]] = llvm.mlir.constant(0 : index) : i64
    // NVVM: %[[descr4:.*]] = llvm.insertvalue %[[c0]], %[[descr3]][2]
    // NVVM: %[[c4:.*]] = llvm.mlir.constant(4 : index) : i64
    // NVVM: %[[descr5:.*]] = llvm.insertvalue %[[c4]], %[[descr4]][3, 0]
    // NVVM: %[[c1:.*]] = llvm.mlir.constant(1 : index) : i64
    // NVVM: %[[descr6:.*]] = llvm.insertvalue %[[c1]], %[[descr5]][4, 0]

    // ROCDL: %[[descr1:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f32, 5>, ptr<f32, 5>, i64, array<1 x i64>, array<1 x i64>)>
    // ROCDL: %[[descr2:.*]] = llvm.insertvalue %[[raw]], %[[descr1]][0]
    // ROCDL: %[[descr3:.*]] = llvm.insertvalue %[[raw]], %[[descr2]][1]
    // ROCDL: %[[c0:.*]] = llvm.mlir.constant(0 : index) : i64
    // ROCDL: %[[descr4:.*]] = llvm.insertvalue %[[c0]], %[[descr3]][2]
    // ROCDL: %[[c4:.*]] = llvm.mlir.constant(4 : index) : i64
    // ROCDL: %[[descr5:.*]] = llvm.insertvalue %[[c4]], %[[descr4]][3, 0]
    // ROCDL: %[[c1:.*]] = llvm.mlir.constant(1 : index) : i64
    // ROCDL: %[[descr6:.*]] = llvm.insertvalue %[[c1]], %[[descr5]][4, 0]

    // "Store" lowering should work just as any other memref, only check that
    // we emit some core instructions.
    // NVVM: llvm.extractvalue %[[descr6:.*]]
    // NVVM: llvm.getelementptr
    // NVVM: llvm.store

    // ROCDL: llvm.extractvalue %[[descr6:.*]]
    // ROCDL: llvm.getelementptr
    // ROCDL: llvm.store
    %c0 = arith.constant 0 : index
    memref.store %arg0, %arg1[%c0] : memref<4xf32, 5>

    "terminator"() : () -> ()
  }
}

// -----

gpu.module @kernel {
  // Workgroup buffers are allocated as globals.
  // NVVM: llvm.mlir.global internal @[[$buffer:.*]]()
  // NVVM-SAME:  addr_space = 3
  // NVVM-SAME:  !llvm.array<4 x f32>

  // ROCDL: llvm.mlir.global internal @[[$buffer:.*]]()
  // ROCDL-SAME:  addr_space = 3
  // ROCDL-SAME:  !llvm.array<4 x f32>

  // NVVM-LABEL: llvm.func @workgroup
  // NVVM-SAME: {

  // ROCDL-LABEL: llvm.func @workgroup
  // ROCDL-SAME: {
  gpu.func @workgroup(%arg0: f32) workgroup(%arg1: memref<4xf32, 3>) {
    // Get the address of the first element in the global array.
    // NVVM: %[[c0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // NVVM: %[[addr:.*]] = llvm.mlir.addressof @[[$buffer]] : !llvm.ptr<array<4 x f32>, 3>
    // NVVM: %[[raw:.*]] = llvm.getelementptr %[[addr]][%[[c0]], %[[c0]]]
    // NVVM-SAME: !llvm.ptr<f32, 3>

    // ROCDL: %[[c0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // ROCDL: %[[addr:.*]] = llvm.mlir.addressof @[[$buffer]] : !llvm.ptr<array<4 x f32>, 3>
    // ROCDL: %[[raw:.*]] = llvm.getelementptr %[[addr]][%[[c0]], %[[c0]]]
    // ROCDL-SAME: !llvm.ptr<f32, 3>

    // Populate the memref descriptor.
    // NVVM: %[[descr1:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i64, array<1 x i64>, array<1 x i64>)>
    // NVVM: %[[descr2:.*]] = llvm.insertvalue %[[raw]], %[[descr1]][0]
    // NVVM: %[[descr3:.*]] = llvm.insertvalue %[[raw]], %[[descr2]][1]
    // NVVM: %[[c0:.*]] = llvm.mlir.constant(0 : index) : i64
    // NVVM: %[[descr4:.*]] = llvm.insertvalue %[[c0]], %[[descr3]][2]
    // NVVM: %[[c4:.*]] = llvm.mlir.constant(4 : index) : i64
    // NVVM: %[[descr5:.*]] = llvm.insertvalue %[[c4]], %[[descr4]][3, 0]
    // NVVM: %[[c1:.*]] = llvm.mlir.constant(1 : index) : i64
    // NVVM: %[[descr6:.*]] = llvm.insertvalue %[[c1]], %[[descr5]][4, 0]

    // ROCDL: %[[descr1:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i64, array<1 x i64>, array<1 x i64>)>
    // ROCDL: %[[descr2:.*]] = llvm.insertvalue %[[raw]], %[[descr1]][0]
    // ROCDL: %[[descr3:.*]] = llvm.insertvalue %[[raw]], %[[descr2]][1]
    // ROCDL: %[[c0:.*]] = llvm.mlir.constant(0 : index) : i64
    // ROCDL: %[[descr4:.*]] = llvm.insertvalue %[[c0]], %[[descr3]][2]
    // ROCDL: %[[c4:.*]] = llvm.mlir.constant(4 : index) : i64
    // ROCDL: %[[descr5:.*]] = llvm.insertvalue %[[c4]], %[[descr4]][3, 0]
    // ROCDL: %[[c1:.*]] = llvm.mlir.constant(1 : index) : i64
    // ROCDL: %[[descr6:.*]] = llvm.insertvalue %[[c1]], %[[descr5]][4, 0]

    // "Store" lowering should work just as any other memref, only check that
    // we emit some core instructions.
    // NVVM: llvm.extractvalue %[[descr6:.*]]
    // NVVM: llvm.getelementptr
    // NVVM: llvm.store

    // ROCDL: llvm.extractvalue %[[descr6:.*]]
    // ROCDL: llvm.getelementptr
    // ROCDL: llvm.store
    %c0 = arith.constant 0 : index
    memref.store %arg0, %arg1[%c0] : memref<4xf32, 3>

    "terminator"() : () -> ()
  }
}

// -----

gpu.module @kernel {
  // Check that the total size was computed correctly.
  // NVVM: llvm.mlir.global internal @[[$buffer:.*]]()
  // NVVM-SAME:  addr_space = 3
  // NVVM-SAME:  !llvm.array<48 x f32>

  // ROCDL: llvm.mlir.global internal @[[$buffer:.*]]()
  // ROCDL-SAME:  addr_space = 3
  // ROCDL-SAME:  !llvm.array<48 x f32>

  // NVVM-LABEL: llvm.func @workgroup3d
  // ROCDL-LABEL: llvm.func @workgroup3d
  gpu.func @workgroup3d(%arg0: f32) workgroup(%arg1: memref<4x2x6xf32, 3>) {
    // Get the address of the first element in the global array.
    // NVVM: %[[c0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // NVVM: %[[addr:.*]] = llvm.mlir.addressof @[[$buffer]] : !llvm.ptr<array<48 x f32>, 3>
    // NVVM: %[[raw:.*]] = llvm.getelementptr %[[addr]][%[[c0]], %[[c0]]]
    // NVVM-SAME: !llvm.ptr<f32, 3>

    // ROCDL: %[[c0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // ROCDL: %[[addr:.*]] = llvm.mlir.addressof @[[$buffer]] : !llvm.ptr<array<48 x f32>, 3>
    // ROCDL: %[[raw:.*]] = llvm.getelementptr %[[addr]][%[[c0]], %[[c0]]]
    // ROCDL-SAME: !llvm.ptr<f32, 3>

    // Populate the memref descriptor.
    // NVVM: %[[descr1:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i64, array<3 x i64>, array<3 x i64>)>
    // NVVM: %[[descr2:.*]] = llvm.insertvalue %[[raw]], %[[descr1]][0]
    // NVVM: %[[descr3:.*]] = llvm.insertvalue %[[raw]], %[[descr2]][1]
    // NVVM: %[[c0:.*]] = llvm.mlir.constant(0 : index) : i64
    // NVVM: %[[descr4:.*]] = llvm.insertvalue %[[c0]], %[[descr3]][2]
    // NVVM: %[[c4:.*]] = llvm.mlir.constant(4 : index) : i64
    // NVVM: %[[descr5:.*]] = llvm.insertvalue %[[c4]], %[[descr4]][3, 0]
    // NVVM: %[[c12:.*]] = llvm.mlir.constant(12 : index) : i64
    // NVVM: %[[descr6:.*]] = llvm.insertvalue %[[c12]], %[[descr5]][4, 0]
    // NVVM: %[[c2:.*]] = llvm.mlir.constant(2 : index) : i64
    // NVVM: %[[descr7:.*]] = llvm.insertvalue %[[c2]], %[[descr6]][3, 1]
    // NVVM: %[[c6:.*]] = llvm.mlir.constant(6 : index) : i64
    // NVVM: %[[descr8:.*]] = llvm.insertvalue %[[c6]], %[[descr7]][4, 1]
    // NVVM: %[[c6:.*]] = llvm.mlir.constant(6 : index) : i64
    // NVVM: %[[descr9:.*]] = llvm.insertvalue %[[c6]], %[[descr8]][3, 2]
    // NVVM: %[[c1:.*]] = llvm.mlir.constant(1 : index) : i64
    // NVVM: %[[descr10:.*]] = llvm.insertvalue %[[c1]], %[[descr9]][4, 2]

    // ROCDL: %[[descr1:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<f32, 3>, ptr<f32, 3>, i64, array<3 x i64>, array<3 x i64>)>
    // ROCDL: %[[descr2:.*]] = llvm.insertvalue %[[raw]], %[[descr1]][0]
    // ROCDL: %[[descr3:.*]] = llvm.insertvalue %[[raw]], %[[descr2]][1]
    // ROCDL: %[[c0:.*]] = llvm.mlir.constant(0 : index) : i64
    // ROCDL: %[[descr4:.*]] = llvm.insertvalue %[[c0]], %[[descr3]][2]
    // ROCDL: %[[c4:.*]] = llvm.mlir.constant(4 : index) : i64
    // ROCDL: %[[descr5:.*]] = llvm.insertvalue %[[c4]], %[[descr4]][3, 0]
    // ROCDL: %[[c12:.*]] = llvm.mlir.constant(12 : index) : i64
    // ROCDL: %[[descr6:.*]] = llvm.insertvalue %[[c12]], %[[descr5]][4, 0]
    // ROCDL: %[[c2:.*]] = llvm.mlir.constant(2 : index) : i64
    // ROCDL: %[[descr7:.*]] = llvm.insertvalue %[[c2]], %[[descr6]][3, 1]
    // ROCDL: %[[c6:.*]] = llvm.mlir.constant(6 : index) : i64
    // ROCDL: %[[descr8:.*]] = llvm.insertvalue %[[c6]], %[[descr7]][4, 1]
    // ROCDL: %[[c6:.*]] = llvm.mlir.constant(6 : index) : i64
    // ROCDL: %[[descr9:.*]] = llvm.insertvalue %[[c6]], %[[descr8]][3, 2]
    // ROCDL: %[[c1:.*]] = llvm.mlir.constant(1 : index) : i64
    // ROCDL: %[[descr10:.*]] = llvm.insertvalue %[[c1]], %[[descr9]][4, 2]

    %c0 = arith.constant 0 : index
    memref.store %arg0, %arg1[%c0,%c0,%c0] : memref<4x2x6xf32, 3>
    "terminator"() : () -> ()
  }
}

// -----

gpu.module @kernel {
  // Check that several buffers are defined.
  // NVVM: llvm.mlir.global internal @[[$buffer1:.*]]()
  // NVVM-SAME:  !llvm.array<1 x f32>
  // NVVM: llvm.mlir.global internal @[[$buffer2:.*]]()
  // NVVM-SAME:  !llvm.array<2 x f32>

  // ROCDL: llvm.mlir.global internal @[[$buffer1:.*]]()
  // ROCDL-SAME:  !llvm.array<1 x f32>
  // ROCDL: llvm.mlir.global internal @[[$buffer2:.*]]()
  // ROCDL-SAME:  !llvm.array<2 x f32>

  // NVVM-LABEL: llvm.func @multiple
  // ROCDL-LABEL: llvm.func @multiple
  gpu.func @multiple(%arg0: f32)
      workgroup(%arg1: memref<1xf32, 3>, %arg2: memref<2xf32, 3>)
      private(%arg3: memref<3xf32, 5>, %arg4: memref<4xf32, 5>) {

    // Workgroup buffers.
    // NVVM: llvm.mlir.addressof @[[$buffer1]]
    // NVVM: llvm.mlir.addressof @[[$buffer2]]

    // ROCDL: llvm.mlir.addressof @[[$buffer1]]
    // ROCDL: llvm.mlir.addressof @[[$buffer2]]

    // Private buffers.
    // NVVM: %[[c3:.*]] = llvm.mlir.constant(3 : i64)
    // NVVM: llvm.alloca %[[c3]] x f32 : (i64) -> !llvm.ptr<f32>
    // NVVM: %[[c4:.*]] = llvm.mlir.constant(4 : i64)
    // NVVM: llvm.alloca %[[c4]] x f32 : (i64) -> !llvm.ptr<f32>

    // ROCDL: %[[c3:.*]] = llvm.mlir.constant(3 : i64)
    // ROCDL: llvm.alloca %[[c3]] x f32 : (i64) -> !llvm.ptr<f32, 5>
    // ROCDL: %[[c4:.*]] = llvm.mlir.constant(4 : i64)
    // ROCDL: llvm.alloca %[[c4]] x f32 : (i64) -> !llvm.ptr<f32, 5>

    %c0 = arith.constant 0 : index
    memref.store %arg0, %arg1[%c0] : memref<1xf32, 3>
    memref.store %arg0, %arg2[%c0] : memref<2xf32, 3>
    memref.store %arg0, %arg3[%c0] : memref<3xf32, 5>
    memref.store %arg0, %arg4[%c0] : memref<4xf32, 5>
    "terminator"() : () -> ()
  }
}
