// RUN: mlir-opt -convert-scf-to-spirv %s -o - | FileCheck %s

module attributes {
  spv.target_env = #spv.target_env<
    #spv.vce<v1.0, [Shader], [SPV_KHR_storage_buffer_storage_class]>, {}>
} {

// CHECK-LABEL: @kernel_simple_selection
func @kernel_simple_selection(%arg2 : memref<10xf32>, %arg3 : i1) {
  %value = arith.constant 0.0 : f32
  %i = arith.constant 0 : index

  // CHECK:       spv.mlir.selection {
  // CHECK-NEXT:    spv.BranchConditional {{%.*}}, [[TRUE:\^.*]], [[MERGE:\^.*]]
  // CHECK-NEXT:  [[TRUE]]:
  // CHECK:         spv.Branch [[MERGE]]
  // CHECK-NEXT:  [[MERGE]]:
  // CHECK-NEXT:    spv.mlir.merge
  // CHECK-NEXT:  }
  // CHECK-NEXT:  spv.Return

  scf.if %arg3 {
    memref.store %value, %arg2[%i] : memref<10xf32>
  }
  return
}

// CHECK-LABEL: @kernel_nested_selection
func @kernel_nested_selection(%arg3 : memref<10xf32>, %arg4 : memref<10xf32>, %arg5 : i1, %arg6 : i1) {
  %i = arith.constant 0 : index
  %j = arith.constant 9 : index

  // CHECK:       spv.mlir.selection {
  // CHECK-NEXT:    spv.BranchConditional {{%.*}}, [[TRUE_TOP:\^.*]], [[FALSE_TOP:\^.*]]
  // CHECK-NEXT:  [[TRUE_TOP]]:
  // CHECK-NEXT:    spv.mlir.selection {
  // CHECK-NEXT:      spv.BranchConditional {{%.*}}, [[TRUE_NESTED_TRUE_PATH:\^.*]], [[FALSE_NESTED_TRUE_PATH:\^.*]]
  // CHECK-NEXT:    [[TRUE_NESTED_TRUE_PATH]]:
  // CHECK:           spv.Branch [[MERGE_NESTED_TRUE_PATH:\^.*]]
  // CHECK-NEXT:    [[FALSE_NESTED_TRUE_PATH]]:
  // CHECK:           spv.Branch [[MERGE_NESTED_TRUE_PATH]]
  // CHECK-NEXT:    [[MERGE_NESTED_TRUE_PATH]]:
  // CHECK-NEXT:      spv.mlir.merge
  // CHECK-NEXT:    }
  // CHECK-NEXT:    spv.Branch [[MERGE_TOP:\^.*]]
  // CHECK-NEXT:  [[FALSE_TOP]]:
  // CHECK-NEXT:    spv.mlir.selection {
  // CHECK-NEXT:      spv.BranchConditional {{%.*}}, [[TRUE_NESTED_FALSE_PATH:\^.*]], [[FALSE_NESTED_FALSE_PATH:\^.*]]
  // CHECK-NEXT:    [[TRUE_NESTED_FALSE_PATH]]:
  // CHECK:           spv.Branch [[MERGE_NESTED_FALSE_PATH:\^.*]]
  // CHECK-NEXT:    [[FALSE_NESTED_FALSE_PATH]]:
  // CHECK:           spv.Branch [[MERGE_NESTED_FALSE_PATH]]
  // CHECK:         [[MERGE_NESTED_FALSE_PATH]]:
  // CHECK-NEXT:      spv.mlir.merge
  // CHECK-NEXT:    }
  // CHECK-NEXT:    spv.Branch [[MERGE_TOP]]
  // CHECK-NEXT:  [[MERGE_TOP]]:
  // CHECK-NEXT:    spv.mlir.merge
  // CHECK-NEXT:  }
  // CHECK-NEXT:  spv.Return

  scf.if %arg5 {
    scf.if %arg6 {
      %value = memref.load %arg3[%i] : memref<10xf32>
      memref.store %value, %arg4[%i] : memref<10xf32>
    } else {
      %value = memref.load %arg4[%i] : memref<10xf32>
      memref.store %value, %arg3[%i] : memref<10xf32>
    }
  } else {
    scf.if %arg6 {
      %value = memref.load %arg3[%j] : memref<10xf32>
      memref.store %value, %arg4[%j] : memref<10xf32>
    } else {
      %value = memref.load %arg4[%j] : memref<10xf32>
      memref.store %value, %arg3[%j] : memref<10xf32>
    }
  }
  return
}

// CHECK-LABEL: @simple_if_yield
func @simple_if_yield(%arg2 : memref<10xf32>, %arg3 : i1) {
  // CHECK: %[[VAR1:.*]] = spv.Variable : !spv.ptr<f32, Function>
  // CHECK: %[[VAR2:.*]] = spv.Variable : !spv.ptr<f32, Function>
  // CHECK:       spv.mlir.selection {
  // CHECK-NEXT:    spv.BranchConditional {{%.*}}, [[TRUE:\^.*]], [[FALSE:\^.*]]
  // CHECK-NEXT:  [[TRUE]]:
  // CHECK:         %[[RET1TRUE:.*]] = spv.Constant 0.000000e+00 : f32
  // CHECK:         %[[RET2TRUE:.*]] = spv.Constant 1.000000e+00 : f32
  // CHECK-DAG:     spv.Store "Function" %[[VAR1]], %[[RET1TRUE]] : f32
  // CHECK-DAG:     spv.Store "Function" %[[VAR2]], %[[RET2TRUE]] : f32
  // CHECK:         spv.Branch ^[[MERGE:.*]]
  // CHECK-NEXT:  [[FALSE]]:
  // CHECK:         %[[RET2FALSE:.*]] = spv.Constant 2.000000e+00 : f32
  // CHECK:         %[[RET1FALSE:.*]] = spv.Constant 3.000000e+00 : f32
  // CHECK-DAG:     spv.Store "Function" %[[VAR1]], %[[RET1FALSE]] : f32
  // CHECK-DAG:     spv.Store "Function" %[[VAR2]], %[[RET2FALSE]] : f32
  // CHECK:         spv.Branch ^[[MERGE]]
  // CHECK-NEXT:  ^[[MERGE]]:
  // CHECK:         spv.mlir.merge
  // CHECK-NEXT:  }
  // CHECK-DAG:   %[[OUT1:.*]] = spv.Load "Function" %[[VAR1]] : f32
  // CHECK-DAG:   %[[OUT2:.*]] = spv.Load "Function" %[[VAR2]] : f32
  // CHECK:       spv.Store "StorageBuffer" {{%.*}}, %[[OUT1]] : f32
  // CHECK:       spv.Store "StorageBuffer" {{%.*}}, %[[OUT2]] : f32
  // CHECK:       spv.Return
  %0:2 = scf.if %arg3 -> (f32, f32) {
    %c0 = arith.constant 0.0 : f32
    %c1 = arith.constant 1.0 : f32
    scf.yield %c0, %c1 : f32, f32
  } else {
    %c0 = arith.constant 2.0 : f32
    %c1 = arith.constant 3.0 : f32
    scf.yield %c1, %c0 : f32, f32
  }
  %i = arith.constant 0 : index
  %j = arith.constant 1 : index
  memref.store %0#0, %arg2[%i] : memref<10xf32>
  memref.store %0#1, %arg2[%j] : memref<10xf32>
  return
}

// TODO: The transformation should only be legal if VariablePointer capability
// is supported. This test is still useful to make sure we can handle scf op
// result with type change.
func @simple_if_yield_type_change(%arg2 : memref<10xf32>, %arg3 : memref<10xf32>, %arg4 : i1) {
  // CHECK-LABEL: @simple_if_yield_type_change
  // CHECK:       %[[VAR:.*]] = spv.Variable : !spv.ptr<!spv.ptr<!spv.struct<(!spv.array<10 x f32, stride=4> [0])>, StorageBuffer>, Function>
  // CHECK:       spv.mlir.selection {
  // CHECK-NEXT:    spv.BranchConditional {{%.*}}, [[TRUE:\^.*]], [[FALSE:\^.*]]
  // CHECK-NEXT:  [[TRUE]]:
  // CHECK:         spv.Store "Function" %[[VAR]], {{%.*}} : !spv.ptr<!spv.struct<(!spv.array<10 x f32, stride=4> [0])>, StorageBuffer>
  // CHECK:         spv.Branch ^[[MERGE:.*]]
  // CHECK-NEXT:  [[FALSE]]:
  // CHECK:         spv.Store "Function" %[[VAR]], {{%.*}} : !spv.ptr<!spv.struct<(!spv.array<10 x f32, stride=4> [0])>, StorageBuffer>
  // CHECK:         spv.Branch ^[[MERGE]]
  // CHECK-NEXT:  ^[[MERGE]]:
  // CHECK:         spv.mlir.merge
  // CHECK-NEXT:  }
  // CHECK:       %[[OUT:.*]] = spv.Load "Function" %[[VAR]] : !spv.ptr<!spv.struct<(!spv.array<10 x f32, stride=4> [0])>, StorageBuffer>
  // CHECK:       %[[ADD:.*]] = spv.AccessChain %[[OUT]][{{%.*}}, {{%.*}}] : !spv.ptr<!spv.struct<(!spv.array<10 x f32, stride=4> [0])>, StorageBuffer>
  // CHECK:       spv.Store "StorageBuffer" %[[ADD]], {{%.*}} : f32
  // CHECK:       spv.Return
  %i = arith.constant 0 : index
  %value = arith.constant 0.0 : f32
  %0 = scf.if %arg4 -> (memref<10xf32>) {
    scf.yield %arg2 : memref<10xf32>
  } else {
    scf.yield %arg3 : memref<10xf32>
  }
  memref.store %value, %0[%i] : memref<10xf32>
  return
}

} // end module
