// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/alias/preserve_in_type_printing.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/alias/preserve_in_type_printing.carbon

// --- fail_alias_nested_in_type.carbon

library "[[@TEST_NAME]]";

class C {}
alias A = C;

// The diagnostics below should refer to `A`, not `C`.

// CHECK:STDERR: fail_alias_nested_in_type.carbon:[[@LINE+7]]:15: error: missing return value [ReturnStatementMissingExpr]
// CHECK:STDERR: fn F() -> A { return; }
// CHECK:STDERR:               ^~~~~~~
// CHECK:STDERR: fail_alias_nested_in_type.carbon:[[@LINE+4]]:11: note: return type of function is `A` [ReturnTypeHereNote]
// CHECK:STDERR: fn F() -> A { return; }
// CHECK:STDERR:           ^
// CHECK:STDERR:
fn F() -> A { return; }
// CHECK:STDERR: fail_alias_nested_in_type.carbon:[[@LINE+7]]:16: error: missing return value [ReturnStatementMissingExpr]
// CHECK:STDERR: fn G() -> A* { return; }
// CHECK:STDERR:                ^~~~~~~
// CHECK:STDERR: fail_alias_nested_in_type.carbon:[[@LINE+4]]:11: note: return type of function is `A*` [ReturnTypeHereNote]
// CHECK:STDERR: fn G() -> A* { return; }
// CHECK:STDERR:           ^~
// CHECK:STDERR:
fn G() -> A* { return; }
// CHECK:STDERR: fail_alias_nested_in_type.carbon:[[@LINE+7]]:22: error: missing return value [ReturnStatementMissingExpr]
// CHECK:STDERR: fn H() -> const A* { return; }
// CHECK:STDERR:                      ^~~~~~~
// CHECK:STDERR: fail_alias_nested_in_type.carbon:[[@LINE+4]]:11: note: return type of function is `const A*` [ReturnTypeHereNote]
// CHECK:STDERR: fn H() -> const A* { return; }
// CHECK:STDERR:           ^~~~~~~~
// CHECK:STDERR:
fn H() -> const A* { return; }
// CHECK:STDERR: fail_alias_nested_in_type.carbon:[[@LINE+7]]:26: error: missing return value [ReturnStatementMissingExpr]
// CHECK:STDERR: fn I() -> array(A, 42) { return; }
// CHECK:STDERR:                          ^~~~~~~
// CHECK:STDERR: fail_alias_nested_in_type.carbon:[[@LINE+4]]:11: note: return type of function is `array(A, 42)` [ReturnTypeHereNote]
// CHECK:STDERR: fn I() -> array(A, 42) { return; }
// CHECK:STDERR:           ^~~~~~~~~~~~
// CHECK:STDERR:
fn I() -> array(A, 42) { return; }

// --- fail_alias_in_type_of_expr.carbon

library "[[@TEST_NAME]]";

class C {}
alias A = C;

fn MakePtr() -> A*;

// The diagnostics below should describe the types of the expressions using the
// spelling `A` from the declaration of the callee, not the canonical `C`.

fn F() {
  // The type of a call is the declared return type of the callee.
  // CHECK:STDERR: fail_alias_in_type_of_expr.carbon:[[@LINE+4]]:3: error: type `A*` does not support qualified expressions [QualifiedExprUnsupported]
  // CHECK:STDERR:   MakePtr().missing;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  MakePtr().missing;

  // The type of a dereference is the pointee type of the pointer's type.
  // CHECK:STDERR: fail_alias_in_type_of_expr.carbon:[[@LINE+4]]:3: error: value of type `A` is not callable [CallToNonCallable]
  // CHECK:STDERR:   (*MakePtr())();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  (*MakePtr())();
}
