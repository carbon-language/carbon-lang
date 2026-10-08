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
  // CHECK:STDERR: fail_mismatch.carbon:[[@LINE+7]]:10: error: `return` of type `bool` does not match earlier `return` of type `i32` [DeducedReturnTypeMismatch]
  // CHECK:STDERR:   return true;
  // CHECK:STDERR:          ^~~~
  // CHECK:STDERR: fail_mismatch.carbon:[[@LINE-5]]:12: note: return type deduced as `i32` here [ReturnTypeDeducedHere]
  // CHECK:STDERR:     return 1 as i32;
  // CHECK:STDERR:            ^~~~~~~~
  // CHECK:STDERR:
  return true;
}

// --- fail_mismatch_literal.carbon
library "[[@TEST_NAME]]";

fn F(b: bool) -> auto {
  if (b) {
    return 1 as i32;
  }
  // CHECK:STDERR: fail_mismatch_literal.carbon:[[@LINE+7]]:10: error: `return` of type `Core.IntLiteral` does not match earlier `return` of type `i32` [DeducedReturnTypeMismatch]
  // CHECK:STDERR:   return 0;
  // CHECK:STDERR:          ^
  // CHECK:STDERR: fail_mismatch_literal.carbon:[[@LINE-5]]:12: note: return type deduced as `i32` here [ReturnTypeDeducedHere]
  // CHECK:STDERR:     return 1 as i32;
  // CHECK:STDERR:            ^~~~~~~~
  // CHECK:STDERR:
  return 0;
}

// --- fail_error_return_ignored.carbon
library "[[@TEST_NAME]]";

// The erroneous `return` doesn't affect the deduced return type.
fn F(b: bool) -> auto {
  if (b) {
    // CHECK:STDERR: fail_error_return_ignored.carbon:[[@LINE+4]]:12: error: name `Undeclared` not found [NameNotFound]
    // CHECK:STDERR:     return Undeclared;
    // CHECK:STDERR:            ^~~~~~~~~~
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
    // CHECK:STDERR: fail_return_no_value.carbon:[[@LINE+7]]:5: error: missing return value [ReturnStatementMissingExpr]
    // CHECK:STDERR:     return;
    // CHECK:STDERR:     ^~~~~~~
    // CHECK:STDERR: fail_return_no_value.carbon:[[@LINE-5]]:1: note: return type of function is deduced [ReturnTypeIsDeducedNote]
    // CHECK:STDERR: fn F(b: bool) -> auto {
    // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    return;
  }
  return 1 as i32;
}

// --- fail_return_no_value_only.carbon
library "[[@TEST_NAME]]";

fn F() -> auto {
  // CHECK:STDERR: fail_return_no_value_only.carbon:[[@LINE+7]]:3: error: missing return value [ReturnStatementMissingExpr]
  // CHECK:STDERR:   return;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_return_no_value_only.carbon:[[@LINE-4]]:1: note: return type of function is deduced [ReturnTypeIsDeducedNote]
  // CHECK:STDERR: fn F() -> auto {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return;
}

// --- fail_reachable_end.carbon
library "[[@TEST_NAME]]";

fn F(b: bool) -> auto {
  if (b) {
    return 1 as i32;
  }
// CHECK:STDERR: fail_reachable_end.carbon:[[@LINE+4]]:1: error: missing `return` at end of function with declared return type [MissingReturnStatement]
// CHECK:STDERR: }
// CHECK:STDERR: ^
// CHECK:STDERR:
}

// --- fail_no_return.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_no_return.carbon:[[@LINE+4]]:1: error: no `return` in function with deduced return type [DeducedReturnTypeWithoutReturn]
// CHECK:STDERR: fn F() -> auto {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F() -> auto {}

// --- fail_return_var_no_returned_var.carbon
library "[[@TEST_NAME]]";

fn F() -> auto {
  // CHECK:STDERR: fail_return_var_no_returned_var.carbon:[[@LINE+4]]:3: error: `return var;` with no `returned var` in scope [ReturnVarWithNoReturnedVar]
  // CHECK:STDERR:   return var;
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR:
  return var;
}

// --- fail_incomplete.carbon
library "[[@TEST_NAME]]";

class C;

fn Make() -> C*;

fn F() -> auto {
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE+7]]:10: error: function returns incomplete type `C` [IncompleteTypeInFunctionReturnType]
  // CHECK:STDERR:   return *Make();
  // CHECK:STDERR:          ^~~~~~~
  // CHECK:STDERR: fail_incomplete.carbon:[[@LINE-8]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR: class C;
  // CHECK:STDERR: ^~~~~~~~
  // CHECK:STDERR:
  return *Make();
}
