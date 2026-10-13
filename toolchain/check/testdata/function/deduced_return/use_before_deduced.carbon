// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/deduced_return/use_before_deduced.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/deduced_return/use_before_deduced.carbon

// --- fail_recursion.carbon
library "[[@TEST_NAME]]";

fn F(b: bool) -> auto {
  if (b) {
    // CHECK:STDERR: fail_recursion.carbon:[[@LINE+3]]:12-19: error: call to `F` before its return type is known [CallBeforeReturnTypeDeduced]
    // CHECK:STDERR: fail_recursion.carbon:[[@LINE-3]]:1-23: note: return type of `F` is deduced from its definition [ReturnTypeDeducedFromDefinition]
    // CHECK:STDERR:
    return F(false);
  }
  return 0 as i32;
}

// --- fail_member_order.carbon
library "[[@TEST_NAME]]";

class C {
  // The body of `F` is checked after the body of `G`.
  // CHECK:STDERR: fail_member_order.carbon:[[@LINE+1]]:26-28: error: call to `F` before its return type is known [CallBeforeReturnTypeDeduced]
  fn G() -> i32 { return F(); }
  // CHECK:STDERR: fail_member_order.carbon:[[@LINE+2]]:3-18: note: return type of `F` is deduced from its definition [ReturnTypeDeducedFromDefinition]
  // CHECK:STDERR:
  fn F() -> auto { return 0 as i32; }
}

// --- recursion_after_returned_var.carbon
library "[[@TEST_NAME]]";

fn F(b: bool) -> auto {
  returned var r: i32 = 0;
  if (b) {
    let unused n: i32 = F(false);
  }
  return var;
}
