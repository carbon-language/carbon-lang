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
  // CHECK:STDERR:        ╭─┤ fail_mismatch.carbon:[[@LINE+13]]:10
  // CHECK:STDERR:        │     return 1 as i32;
  // CHECK:STDERR:        ·            ────┬───
  // CHECK:STDERR:        ·                ╰──┤ return type previously deduced as `i32` here
  // CHECK:STDERR:        ·                   │ [DeducedReturnTypeMismatchPrevious]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   return true;
  // CHECK:STDERR:        ·          ┳━━━
  // CHECK:STDERR:        ·          ┗━━┫ error: deduced return type `bool` does not match return type
  // CHECK:STDERR:        ·             ┃        deduced earlier
  // CHECK:STDERR:        ·             ┃        [DeducedReturnTypeMismatch]
  // CHECK:STDERR:        ╰────
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
  // CHECK:STDERR:        ╭─┤ fail_todo_mismatch_int_vs_int_literal.carbon:[[@LINE+13]]:10
  // CHECK:STDERR:        │     return 1 as i32;
  // CHECK:STDERR:        ·            ────┬───
  // CHECK:STDERR:        ·                ╰──┤ return type previously deduced as `i32` here
  // CHECK:STDERR:        ·                   │ [DeducedReturnTypeMismatchPrevious]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   return 0;
  // CHECK:STDERR:        ·          ┳
  // CHECK:STDERR:        ·          ┗━━┫ error: deduced return type `Core.IntLiteral` does not match return
  // CHECK:STDERR:        ·             ┃        type deduced earlier
  // CHECK:STDERR:        ·             ┃        [DeducedReturnTypeMismatch]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  return 0;
}

// --- fail_error_return_ignored.carbon
library "[[@TEST_NAME]]";

// The erroneous `return` doesn't affect the deduced return type.
fn F(b: bool) -> auto {
  if (b) {
    // CHECK:STDERR:        ╭─┤ fail_error_return_ignored.carbon:[[@LINE+6]]:12
    // CHECK:STDERR: ->     │     return Undeclared;
    // CHECK:STDERR:        ·            ━━━━━┳━━━━
    // CHECK:STDERR:        ·                 ┗━━┫ error: name `Undeclared` not found [NameNotFound]
    // CHECK:STDERR:        ╰────
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
    // CHECK:STDERR:        ╭─┤ fail_return_no_value.carbon:[[@LINE+10]]:5
    // CHECK:STDERR:        │ fn F(b: bool) -> auto {
    // CHECK:STDERR:        · ───────────┬───────────
    // CHECK:STDERR:        ·            ╰──┤ return type of function is deduced [ReturnTypeIsDeducedNote]
    // CHECK:STDERR:        ┆
    // CHECK:STDERR: ->     │     return;
    // CHECK:STDERR:        ·     ━━━┳━━━
    // CHECK:STDERR:        ·        ┗━━┫ error: missing return value [ReturnStatementMissingExpr]
    // CHECK:STDERR:        ╰────
    // CHECK:STDERR:
    return;
  }
  return 1 as i32;
}

// --- fail_return_no_value_only.carbon
library "[[@TEST_NAME]]";

fn F() -> auto {
  // CHECK:STDERR:        ╭─┤ fail_return_no_value_only.carbon:[[@LINE+10]]:3
  // CHECK:STDERR:        │ fn F() -> auto {
  // CHECK:STDERR:        · ────────┬───────
  // CHECK:STDERR:        ·         ╰──┤ return type of function is deduced [ReturnTypeIsDeducedNote]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   return;
  // CHECK:STDERR:        ·   ━━━┳━━━
  // CHECK:STDERR:        ·      ┗━━┫ error: missing return value [ReturnStatementMissingExpr]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  return;
}

// --- fail_reachable_end.carbon
library "[[@TEST_NAME]]";

fn F(b: bool) -> auto {
  if (b) {
    return 1 as i32;
  }
// CHECK:STDERR:        ╭─┤ fail_reachable_end.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ }
// CHECK:STDERR:        · ┳
// CHECK:STDERR:        · ┗━━┫ error: missing `return` at end of function with declared return type
// CHECK:STDERR:        ·    ┃        [MissingReturnStatement]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
}

// --- fail_no_return.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR:        ╭─┤ fail_no_return.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ fn F() -> auto {}
// CHECK:STDERR:        · ━━━━━━━━┳━━━━━━━
// CHECK:STDERR:        ·         ┗━━┫ error: no `return` in function with deduced return type
// CHECK:STDERR:        ·            ┃        [DeducedReturnTypeWithoutReturn]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn F() -> auto {}

// --- fail_return_var_no_returned_var.carbon
library "[[@TEST_NAME]]";

fn F() -> auto {
  // CHECK:STDERR:        ╭─┤ fail_return_var_no_returned_var.carbon:[[@LINE+7]]:3
  // CHECK:STDERR: ->     │   return var;
  // CHECK:STDERR:        ·   ━━━━━┳━━━━━
  // CHECK:STDERR:        ·        ┗━━┫ error: `return var;` with no `returned var` in scope
  // CHECK:STDERR:        ·           ┃        [ReturnVarWithNoReturnedVar]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  return var;
}

// --- fail_incomplete.carbon
library "[[@TEST_NAME]]";

class C;

fn Make() -> C*;

fn F() -> auto {
  // CHECK:STDERR:        ╭─┤ fail_incomplete.carbon:[[@LINE+11]]:10
  // CHECK:STDERR:        │ class C;
  // CHECK:STDERR:        · ────┬───
  // CHECK:STDERR:        ·     ╰──┤ class was forward declared here [ClassForwardDeclaredHere]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   return *Make();
  // CHECK:STDERR:        ·          ━━━┳━━━
  // CHECK:STDERR:        ·             ┗━━┫ error: function returns incomplete type `C`
  // CHECK:STDERR:        ·                ┃        [IncompleteTypeInFunctionReturnType]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  return *Make();
}
