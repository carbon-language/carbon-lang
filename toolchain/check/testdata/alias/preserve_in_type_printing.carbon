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

// CHECK:STDERR: fail_alias_nested_in_type.carbon:[[@LINE+3]]:15-21: error: missing return value [ReturnStatementMissingExpr]
// CHECK:STDERR: fail_alias_nested_in_type.carbon:[[@LINE+2]]:11: note: return type of function is `A` [ReturnTypeHereNote]
// CHECK:STDERR:
fn F() -> A { return; }
// CHECK:STDERR: fail_alias_nested_in_type.carbon:[[@LINE+3]]:16-22: error: missing return value [ReturnStatementMissingExpr]
// CHECK:STDERR: fail_alias_nested_in_type.carbon:[[@LINE+2]]:11-12: note: return type of function is `A*` [ReturnTypeHereNote]
// CHECK:STDERR:
fn G() -> A* { return; }
// CHECK:STDERR: fail_alias_nested_in_type.carbon:[[@LINE+3]]:22-28: error: missing return value [ReturnStatementMissingExpr]
// CHECK:STDERR: fail_alias_nested_in_type.carbon:[[@LINE+2]]:11-18: note: return type of function is `const A*` [ReturnTypeHereNote]
// CHECK:STDERR:
fn H() -> const A* { return; }
// CHECK:STDERR: fail_alias_nested_in_type.carbon:[[@LINE+3]]:26-32: error: missing return value [ReturnStatementMissingExpr]
// CHECK:STDERR: fail_alias_nested_in_type.carbon:[[@LINE+2]]:11-22: note: return type of function is `array(A, 42)` [ReturnTypeHereNote]
// CHECK:STDERR:
fn I() -> array(A, 42) { return; }
