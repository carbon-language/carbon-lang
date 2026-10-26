// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/deduced_return/fail_return.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/deduced_return/fail_return.carbon

// --- fail_mismatch.carbon
library "[[@TEST_NAME]]";

fn F(b: bool) -> auto {
  if (b) {
    return 1 as i32;
  }
  // CHECK:STDERR: fail_mismatch.carbon:[[@LINE+3]]:10-13: error: deduced return type `bool` does not match return type deduced earlier [DeducedReturnTypeMismatch]
  // CHECK:STDERR: fail_mismatch.carbon:[[@LINE-3]]:12-19: note: return type previously deduced as `i32` here [DeducedReturnTypeMismatchPrevious]
  // CHECK:STDERR:
  return true;
}

// --- fail_todo_mismatch_int_vs_int_literal.carbon
library "[[@TEST_NAME]]";

fn F(b: bool) -> auto {
  if (b) {
    return 1 as i32;
  }
  // TODO: This should work once we use CommonType in return type deduction.
  // CHECK:STDERR: fail_todo_mismatch_int_vs_int_literal.carbon:[[@LINE+3]]:10: error: deduced return type `Core.IntLiteral` does not match return type deduced earlier [DeducedReturnTypeMismatch]
  // CHECK:STDERR: fail_todo_mismatch_int_vs_int_literal.carbon:[[@LINE-4]]:12-19: note: return type previously deduced as `i32` here [DeducedReturnTypeMismatchPrevious]
  // CHECK:STDERR:
  return 0;
}

// --- fail_error_return_ignored.carbon
library "[[@TEST_NAME]]";

// The erroneous `return` doesn't affect the deduced return type.
fn F(b: bool) -> auto {
  if (b) {
    // CHECK:STDERR: fail_error_return_ignored.carbon:[[@LINE+2]]:12-21: error: name `Undeclared` not found [NameNotFound]
    // CHECK:STDERR:
    return Undeclared;
  }
  return 1 as i32;
}

fn G() -> i32 {
  return F(true);
}

// --- fail_return_no_value.carbon
library "[[@TEST_NAME]]";

fn F(b: bool) -> auto {
  if (b) {
    // CHECK:STDERR: fail_return_no_value.carbon:[[@LINE+3]]:5-11: error: missing return value [ReturnStatementMissingExpr]
    // CHECK:STDERR: fail_return_no_value.carbon:[[@LINE-3]]:1-23: note: return type of function is deduced [ReturnTypeIsDeducedNote]
    // CHECK:STDERR:
    return;
  }
  return 1 as i32;
}

// --- fail_return_no_value_only.carbon
library "[[@TEST_NAME]]";

fn F() -> auto {
  // CHECK:STDERR: fail_return_no_value_only.carbon:[[@LINE+3]]:3-9: error: missing return value [ReturnStatementMissingExpr]
  // CHECK:STDERR: fail_return_no_value_only.carbon:[[@LINE-2]]:1-16: note: return type of function is deduced [ReturnTypeIsDeducedNote]
  // CHECK:STDERR:
  return;
}

// --- fail_reachable_end.carbon
library "[[@TEST_NAME]]";

fn F(b: bool) -> auto {
  if (b) {
    return 1 as i32;
  }
// CHECK:STDERR: fail_reachable_end.carbon:[[@LINE+2]]:1: error: missing `return` at end of function with declared return type [MissingReturnStatement]
// CHECK:STDERR:
}

// --- fail_no_return.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_no_return.carbon:[[@LINE+2]]:1-16: error: no `return` in function with deduced return type [DeducedReturnTypeWithoutReturn]
// CHECK:STDERR:
fn F() -> auto {}

// --- fail_return_var_no_returned_var.carbon
library "[[@TEST_NAME]]";

fn F() -> auto {
  // CHECK:STDERR: fail_return_var_no_returned_var.carbon:[[@LINE+2]]:3-13: error: `return var;` with no `returned var` in scope [ReturnVarWithNoReturnedVar]
  // CHECK:STDERR:
  return var;
}

// --- fail_incomplete.carbon
library "[[@TEST_NAME]]";

class C;

fn Make() -> C*;

fn F() -> auto {
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE+3]]:10-16: error: function returns incomplete type `C` [IncompleteTypeInFunctionReturnType]
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE-6]]:1-8: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR:
  return *Make();
}
