// RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline='func.func(canonicalize)' -split-input-file | FileCheck %s

// Check the simple case of single operation blocks with a return.

// CHECK-LABEL: func @return_blocks(
func.func @return_blocks() {
  // CHECK: "foo.cond_br"()[^bb1, ^bb1]
  // CHECK: ^bb1:
  // CHECK-NEXT: return
  // CHECK-NOT: ^bb2

  "foo.cond_br"() [^bb1, ^bb2] : () -> ()

^bb1:
  return
^bb2:
  return
}

// Check the case of identical blocks with matching arguments.

// CHECK-LABEL: func @matching_arguments(
func.func @matching_arguments() -> i32 {
  // CHECK: "foo.cond_br"()[^bb1, ^bb1]
  // CHECK: ^bb1(%{{.*}}: i32):
  // CHECK-NEXT: return
  // CHECK-NOT: ^bb2

  "foo.cond_br"() [^bb1, ^bb2] : () -> ()

^bb1(%arg0 : i32):
  return %arg0 : i32
^bb2(%arg1 : i32):
  return %arg1 : i32
}

// Check that no merging occurs if there is an operand mismatch and we can't
// update th predecessor.

// CHECK-LABEL: func @mismatch_unknown_terminator
func.func @mismatch_unknown_terminator(%arg0 : i32, %arg1 : i32) -> i32 {
  // CHECK: "foo.cond_br"()[^bb1, ^bb2]

  "foo.cond_br"() [^bb1, ^bb2] : () -> ()

^bb1:
  return %arg0 : i32
^bb2:
  return %arg1 : i32
}

// Check that merging does occurs if there is an operand mismatch and we can
// update th predecessor.

// CHECK-LABEL: func @mismatch_operands
// CHECK-SAME: %[[COND:.*]]: i1, %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32
func.func @mismatch_operands(%cond : i1, %arg0 : i32, %arg1 : i32) -> i32 {
  // CHECK: %[[RES:.*]] = arith.select %[[COND]], %[[ARG0]], %[[ARG1]]
  // CHECK: return %[[RES]]

  cf.cond_br %cond, ^bb1, ^bb2

^bb1:
  return %arg0 : i32
^bb2:
  return %arg1 : i32
}

// Check the same as above, but with pre-existing arguments.

// CHECK-LABEL: func @mismatch_operands_matching_arguments(
// CHECK-SAME: %[[COND:.*]]: i1, %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32
func.func @mismatch_operands_matching_arguments(%cond : i1, %arg0 : i32, %arg1 : i32) -> (i32, i32) {
  // CHECK: %[[RES0:.*]] = arith.select %[[COND]], %[[ARG1]], %[[ARG0]]
  // CHECK: %[[RES1:.*]] = arith.select %[[COND]], %[[ARG0]], %[[ARG1]]
  // CHECK: return %[[RES1]], %[[RES0]]

  cf.cond_br %cond, ^bb1(%arg1 : i32), ^bb2(%arg0 : i32)

^bb1(%arg2 : i32):
  return %arg0, %arg2 : i32, i32
^bb2(%arg3 : i32):
  return %arg1, %arg3 : i32, i32
}

// Check that merging does not occur if the uses of the arguments differ.

// CHECK-LABEL: func @mismatch_argument_uses(
func.func @mismatch_argument_uses(%cond : i1, %arg0 : i32, %arg1 : i32) -> (i32, i32) {
  // CHECK: cf.cond_br %{{.*}}, ^bb1(%{{.*}}), ^bb2

  cf.cond_br %cond, ^bb1(%arg1 : i32), ^bb2(%arg0 : i32)

^bb1(%arg2 : i32):
  return %arg0, %arg2 : i32, i32
^bb2(%arg3 : i32):
  return %arg3, %arg1 : i32, i32
}

// Check that merging does not occur if the types of the arguments differ.

// CHECK-LABEL: func @mismatch_argument_types(
func.func @mismatch_argument_types(%cond : i1, %arg0 : i32, %arg1 : i16) {
  // CHECK: cf.cond_br %{{.*}}, ^bb1(%{{.*}}), ^bb2

  cf.cond_br %cond, ^bb1(%arg0 : i32), ^bb2(%arg1 : i16)

^bb1(%arg2 : i32):
  "foo.return"(%arg2) : (i32) -> ()
^bb2(%arg3 : i16):
  "foo.return"(%arg3) : (i16) -> ()
}

// Check that merging does not occur if the number of the arguments differ.

// CHECK-LABEL: func @mismatch_argument_count(
func.func @mismatch_argument_count(%cond : i1, %arg0 : i32) {
  // CHECK: cf.cond_br %{{.*}}, ^bb1(%{{.*}}), ^bb2

  cf.cond_br %cond, ^bb1(%arg0 : i32), ^bb2

^bb1(%arg2 : i32):
  "foo.return"(%arg2) : (i32) -> ()
^bb2:
  "foo.return"() : () -> ()
}

// Check that merging does not occur if the operations differ.

// CHECK-LABEL: func @mismatch_operations(
func.func @mismatch_operations(%cond : i1) {
  // CHECK: cf.cond_br %{{.*}}, ^bb1, ^bb2

  cf.cond_br %cond, ^bb1, ^bb2

^bb1:
  "foo.return"() : () -> ()
^bb2:
  return
}

// Check that merging does not occur if the number of operations differ.

// CHECK-LABEL: func @mismatch_operation_count(
func.func @mismatch_operation_count(%cond : i1) {
  // CHECK: cf.cond_br %{{.*}}, ^bb1, ^bb2

  cf.cond_br %cond, ^bb1, ^bb2

^bb1:
  "foo.op"() : () -> ()
  return
^bb2:
  return
}

// Check that merging does not occur if the blocks contain regions.

// CHECK-LABEL: func @contains_regions(
func.func @contains_regions(%cond : i1) {
  // CHECK: cf.cond_br %{{.*}}, ^bb1, ^bb2

  cf.cond_br %cond, ^bb1, ^bb2

^bb1:
  scf.if %cond {
    "foo.op"() : () -> ()
  }
  return
^bb2:
  scf.if %cond {
    "foo.op"() : () -> ()
  }
  return
}

// Check that properly handles back edges.

// CHECK-LABEL: func @mismatch_loop(
// CHECK-SAME: %[[ARG:.*]]: i1, %[[ARG2:.*]]: i1
func.func @mismatch_loop(%cond : i1, %cond2 : i1) {
  // CHECK-NEXT: %[[LOOP_CARRY:.*]] = "foo.op"
  // CHECK: cf.cond_br %{{.*}}, ^bb1(%[[ARG2]] : i1), ^bb2

  %cond3 = "foo.op"() : () -> (i1)
  cf.cond_br %cond, ^bb2, ^bb3

^bb1:
  // CHECK: ^bb1(%[[ARG3:.*]]: i1):
  // CHECK-NEXT: cf.cond_br %[[ARG3]], ^bb1(%[[LOOP_CARRY]] : i1), ^bb2

  cf.cond_br %cond3, ^bb1, ^bb3

^bb2:
  cf.cond_br %cond2, ^bb1, ^bb3

^bb3:
  // CHECK: ^bb2:
  // CHECK-NEXT: return

  return
}

// Check that blocks are not merged if the types of the operands differ.

// CHECK-LABEL: func @mismatch_operand_types(
func.func @mismatch_operand_types(%arg0 : i1, %arg1 : memref<i32>, %arg2 : memref<i1>) {
  %c0_i32 = arith.constant 0 : i32
  %true = arith.constant true
  cf.br ^bb1

^bb1:
  cf.cond_br %arg0, ^bb2, ^bb3

^bb2:
  // CHECK: memref.store %{{.*}}, %{{.*}} : memref<i32>
  memref.store %c0_i32, %arg1[] : memref<i32>
  cf.br ^bb1

^bb3:
  // CHECK: memref.store %{{.*}}, %{{.*}} : memref<i1>
  memref.store %true, %arg2[] : memref<i1>
  cf.br ^bb1
}

// Check that it is illegal to merge blocks containing an operand
// with an external user. Incorrectly performing the optimization
// anyways will result in print(merged, merged) rather than
// distinct operands.
func.func private @print(%arg0: i32, %arg1: i32)
// CHECK-LABEL: @nomerge
func.func @nomerge(%arg0: i32, %i: i32) {
  %c1_i32 = arith.constant 1 : i32
  %icmp = arith.cmpi slt, %i, %arg0 : i32
  cf.cond_br %icmp, ^bb2, ^bb3

^bb2:  // pred: ^bb1
  %ip1 = arith.addi %i, %c1_i32 : i32
  cf.br ^bb4(%ip1 : i32)

^bb7:  // pred: ^bb5
  %jp1 = arith.addi %j, %c1_i32 : i32
  cf.br ^bb4(%jp1 : i32)

^bb4(%j: i32):  // 2 preds: ^bb2, ^bb7
  %jcmp = arith.cmpi slt, %j, %arg0 : i32
// CHECK-NOT:  call @print(%[[arg1:.+]], %[[arg1]])
  call @print(%j, %ip1) : (i32, i32) -> ()
  cf.cond_br %jcmp, ^bb7, ^bb3

^bb3:  // pred: ^bb1
  return
}


// CHECK-LABEL: func @mismatch_dominance(
func.func @mismatch_dominance() -> i32 {
  // CHECK: %[[RES:.*]] = "test.producing_br"()
  %0 = "test.producing_br"()[^bb1, ^bb2] {
        operand_segment_sizes = dense<0> : vector<2 x i32>
	} : () -> i32

^bb1:
  // CHECK: "test.br"(%[[RES]])[^[[MERGE_BLOCK:.*]]]
  "test.br"(%0)[^bb4] : (i32) -> ()

^bb2:
  %1 = "foo.def"() : () -> i32
  "test.br"()[^bb3] : () -> ()

^bb3:
  // CHECK: "test.br"(%{{.*}})[^[[MERGE_BLOCK]]]
  "test.br"(%1)[^bb4] : (i32) -> ()

^bb4(%3: i32):
  return %3 : i32
}
