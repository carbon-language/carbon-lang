// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/declaration/fail_todo_no_params.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/declaration/fail_todo_no_params.carbon

// --- fail_no_body.carbon

library "[[@TEST_NAME]]";
// CHECK:STDERR: fail_no_body.carbon:[[@LINE+4]]:1: error: semantics TODO: `function with positional parameters` [SemanticsTodo]
// CHECK:STDERR: fn A;
// CHECK:STDERR: ^~~~~
// CHECK:STDERR:
fn A;

// --- fail_todo_brace_body.carbon

library "[[@TEST_NAME]]";
// CHECK:STDERR: fail_todo_brace_body.carbon:[[@LINE+4]]:1: error: semantics TODO: `function with positional parameters` [SemanticsTodo]
// CHECK:STDERR: fn A {
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
fn A {
}

// --- fail_todo_return_type.carbon

library "[[@TEST_NAME]]";
// CHECK:STDERR: fail_todo_return_type.carbon:[[@LINE+4]]:1: error: semantics TODO: `function with positional parameters` [SemanticsTodo]
// CHECK:STDERR: fn A -> ();
// CHECK:STDERR: ^~~~~~~~~~~
// CHECK:STDERR:
fn A -> ();

// --- fail_todo_implicit_only.carbon

library "[[@TEST_NAME]]";
// CHECK:STDERR: fail_todo_implicit_only.carbon:[[@LINE+4]]:1: error: semantics TODO: `function with positional parameters` [SemanticsTodo]
// CHECK:STDERR: fn A[] -> ();
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
fn A[] -> ();

// --- fail_todo_arrow_body.carbon

library "[[@TEST_NAME]]";
// CHECK:STDERR: fail_todo_arrow_body.carbon:[[@LINE+4]]:1: error: semantics TODO: `function with positional parameters` [SemanticsTodo]
// CHECK:STDERR: fn A => 0;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
fn A => 0;

// --- fail_invalid_file_generic_regression_test.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_invalid_file_generic_regression_test.carbon:[[@LINE+8]]:13: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
// CHECK:STDERR: var generic x: () = ();
// CHECK:STDERR:             ^~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_invalid_file_generic_regression_test.carbon:[[@LINE+4]]:19: error: found generic binding inside `var` pattern [NonRegularBindingInVarDecl]
// CHECK:STDERR: var generic x: () = ();
// CHECK:STDERR:                   ^
// CHECK:STDERR:
var generic x: () = ();

fn A {
  A();
}
