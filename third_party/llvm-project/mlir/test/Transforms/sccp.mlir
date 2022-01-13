// RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline="builtin.func(sccp)" -split-input-file | FileCheck %s

/// Check simple forward constant propagation without any control flow.

// CHECK-LABEL: func @no_control_flow
func @no_control_flow(%arg0: i32) -> i32 {
  // CHECK: %[[CST:.*]] = arith.constant 1 : i32
  // CHECK: return %[[CST]] : i32

  %cond = arith.constant true
  %cst_1 = arith.constant 1 : i32
  %select = select %cond, %cst_1, %arg0 : i32
  return %select : i32
}

/// Check that a constant is properly propagated when only one edge of a branch
/// is taken.

// CHECK-LABEL: func @simple_control_flow
func @simple_control_flow(%arg0 : i32) -> i32 {
  // CHECK: %[[CST:.*]] = arith.constant 1 : i32

  %cond = arith.constant true
  %1 = arith.constant 1 : i32
  cond_br %cond, ^bb1, ^bb2(%arg0 : i32)

^bb1:
  br ^bb2(%1 : i32)

^bb2(%arg : i32):
  // CHECK: ^bb2(%{{.*}}: i32):
  // CHECK: return %[[CST]] : i32

  return %arg : i32
}

/// Check that the arguments go to overdefined if the branch cannot detect when
/// a specific successor is taken.

// CHECK-LABEL: func @simple_control_flow_overdefined
func @simple_control_flow_overdefined(%arg0 : i32, %arg1 : i1) -> i32 {
  %1 = arith.constant 1 : i32
  cond_br %arg1, ^bb1, ^bb2(%arg0 : i32)

^bb1:
  br ^bb2(%1 : i32)

^bb2(%arg : i32):
  // CHECK: ^bb2(%[[ARG:.*]]: i32):
  // CHECK: return %[[ARG]] : i32

  return %arg : i32
}

/// Check that the arguments go to overdefined if there are conflicting
/// constants.

// CHECK-LABEL: func @simple_control_flow_constant_overdefined
func @simple_control_flow_constant_overdefined(%arg0 : i32, %arg1 : i1) -> i32 {
  %1 = arith.constant 1 : i32
  %2 = arith.constant 2 : i32
  cond_br %arg1, ^bb1, ^bb2(%arg0 : i32)

^bb1:
  br ^bb2(%2 : i32)

^bb2(%arg : i32):
  // CHECK: ^bb2(%[[ARG:.*]]: i32):
  // CHECK: return %[[ARG]] : i32

  return %arg : i32
}

/// Check that the arguments go to overdefined if the branch is unknown.

// CHECK-LABEL: func @unknown_terminator
func @unknown_terminator(%arg0 : i32, %arg1 : i1) -> i32 {
  %1 = arith.constant 1 : i32
  "foo.cond_br"() [^bb1, ^bb2] : () -> ()

^bb1:
  br ^bb2(%1 : i32)

^bb2(%arg : i32):
  // CHECK: ^bb2(%[[ARG:.*]]: i32):
  // CHECK: return %[[ARG]] : i32

  return %arg : i32
}

/// Check that arguments are properly merged across loop-like control flow.

func private @ext_cond_fn() -> i1

// CHECK-LABEL: func @simple_loop
func @simple_loop(%arg0 : i32, %cond1 : i1) -> i32 {
  // CHECK: %[[CST:.*]] = arith.constant 1 : i32

  %cst_1 = arith.constant 1 : i32
  cond_br %cond1, ^bb1(%cst_1 : i32), ^bb2(%cst_1 : i32)

^bb1(%iv: i32):
  // CHECK: ^bb1(%{{.*}}: i32):
  // CHECK-NEXT: %[[COND:.*]] = call @ext_cond_fn()
  // CHECK-NEXT: cond_br %[[COND]], ^bb1(%[[CST]] : i32), ^bb2(%[[CST]] : i32)

  %cst_0 = arith.constant 0 : i32
  %res = arith.addi %iv, %cst_0 : i32
  %cond2 = call @ext_cond_fn() : () -> i1
  cond_br %cond2, ^bb1(%res : i32), ^bb2(%res : i32)

^bb2(%arg : i32):
  // CHECK: ^bb2(%{{.*}}: i32):
  // CHECK: return %[[CST]] : i32

  return %arg : i32
}

/// Test that we can properly propagate within inner control, and in situations
/// where the executable edges within the CFG are sensitive to the current state
/// of the analysis.

// CHECK-LABEL: func @simple_loop_inner_control_flow
func @simple_loop_inner_control_flow(%arg0 : i32) -> i32 {
  // CHECK-DAG: %[[CST:.*]] = arith.constant 1 : i32
  // CHECK-DAG: %[[TRUE:.*]] = arith.constant true

  %cst_1 = arith.constant 1 : i32
  br ^bb1(%cst_1 : i32)

^bb1(%iv: i32):
  %cond2 = call @ext_cond_fn() : () -> i1
  cond_br %cond2, ^bb5(%iv : i32), ^bb2

^bb2:
  // CHECK: ^bb2:
  // CHECK: cond_br %[[TRUE]], ^bb3, ^bb4

  %cst_20 = arith.constant 20 : i32
  %cond = arith.cmpi ult, %iv, %cst_20 : i32
  cond_br %cond, ^bb3, ^bb4

^bb3:
  // CHECK: ^bb3:
  // CHECK: br ^bb1(%[[CST]] : i32)

  %cst_1_2 = arith.constant 1 : i32
  br ^bb1(%cst_1_2 : i32)

^bb4:
  %iv_inc = arith.addi %iv, %cst_1 : i32
  br ^bb1(%iv_inc : i32)

^bb5(%result: i32):
  // CHECK: ^bb5(%{{.*}}: i32):
  // CHECK: return %[[CST]] : i32

  return %result : i32
}

/// Check that arguments go to overdefined when loop backedges produce a
/// conflicting value.

func private @ext_cond_and_value_fn() -> (i1, i32)

// CHECK-LABEL: func @simple_loop_overdefined
func @simple_loop_overdefined(%arg0 : i32, %cond1 : i1) -> i32 {
  %cst_1 = arith.constant 1 : i32
  cond_br %cond1, ^bb1(%cst_1 : i32), ^bb2(%cst_1 : i32)

^bb1(%iv: i32):
  %cond2, %res = call @ext_cond_and_value_fn() : () -> (i1, i32)
  cond_br %cond2, ^bb1(%res : i32), ^bb2(%res : i32)

^bb2(%arg : i32):
  // CHECK: ^bb2(%[[ARG:.*]]: i32):
  // CHECK: return %[[ARG]] : i32

  return %arg : i32
}

// Check that we reprocess executable edges when information changes.

// CHECK-LABEL: func @recheck_executable_edge
func @recheck_executable_edge(%cond0: i1) -> (i1, i1) {
  %true = arith.constant true
  %false = arith.constant false
  cond_br %cond0, ^bb_1a, ^bb2(%false : i1)
^bb_1a:
  br ^bb2(%true : i1)

^bb2(%x: i1):
  // CHECK: ^bb2(%[[X:.*]]: i1):
  br ^bb3(%x : i1)

^bb3(%y: i1):
  // CHECK: ^bb3(%[[Y:.*]]: i1):
  // CHECK: return %[[X]], %[[Y]]
  return %x, %y : i1, i1
}
