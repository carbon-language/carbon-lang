// RUN: mlir-opt -split-input-file -convert-cf-to-spirv -verify-diagnostics %s | FileCheck %s

//===----------------------------------------------------------------------===//
// cf.br, cf.cond_br
//===----------------------------------------------------------------------===//

module attributes {
  spv.target_env = #spv.target_env<#spv.vce<v1.0, [], []>, #spv.resource_limits<>>
} {

// CHECK-LABEL: func @simple_loop
func.func @simple_loop(%begin: i32, %end: i32, %step: i32) {
// CHECK-NEXT:  spv.Branch ^bb1
  cf.br ^bb1

// CHECK-NEXT: ^bb1:    // pred: ^bb0
// CHECK-NEXT:  spv.Branch ^bb2({{.*}} : i32)
^bb1:   // pred: ^bb0
  cf.br ^bb2(%begin : i32)

// CHECK:      ^bb2({{.*}}: i32):       // 2 preds: ^bb1, ^bb3
// CHECK:        spv.BranchConditional {{.*}}, ^bb3, ^bb4
^bb2(%0: i32):        // 2 preds: ^bb1, ^bb3
  %1 = arith.cmpi slt, %0, %end : i32
  cf.cond_br %1, ^bb3, ^bb4

// CHECK:      ^bb3:    // pred: ^bb2
// CHECK:        spv.Branch ^bb2({{.*}} : i32)
^bb3:   // pred: ^bb2
  %2 = arith.addi %0, %step : i32
  cf.br ^bb2(%2 : i32)

// CHECK:      ^bb4:    // pred: ^bb2
^bb4:   // pred: ^bb2
  return
}

}
