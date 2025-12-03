// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/named_constraint/invalid_members.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/named_constraint/invalid_members.carbon

// --- todo_fail_invalid_fn.carbon
library "[[@TEST_NAME]]";

constraint E {
  // TODO: Any fn in a non-template constraint is an error.
  fn DeclaredFunction();
  fn DeclaredMethod[self: Self](b: Self);
}

constraint F {
  // TODO: Any fn in a non-template constraint is an error.
  fn DefinedFunction() {}
  fn DefinedMethod[self: Self](b: Self) {}
}

// --- todo_fail_invalid_var.carbon
library "[[@TEST_NAME]]";

constraint E {
  // TODO: Any var in a constraint is an error.
  var c: ();
}

// --- fail_todo_invalid_var_template.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_todo_invalid_var_template.carbon:[[@LINE+8]]:1: error: unrecognized declaration introducer [UnrecognizedDecl]
// CHECK:STDERR: template constraint E {
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_invalid_var_template.carbon:[[@LINE+4]]:1: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
// CHECK:STDERR: template constraint E {
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
template constraint E {
  // TODO: Any var in a constraint is an error.
  var c: ();
}

// --- todo_fail_invalid_let.carbon
library "[[@TEST_NAME]]";

constraint E {
  // TODO: Any let in a constraint is an error.
  var c: ();
}

// --- fail_todo_invalid_let_template.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_todo_invalid_let_template.carbon:[[@LINE+8]]:1: error: unrecognized declaration introducer [UnrecognizedDecl]
// CHECK:STDERR: template constraint E {
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_invalid_let_template.carbon:[[@LINE+4]]:1: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
// CHECK:STDERR: template constraint E {
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
template constraint E {
  // TODO: Any let in a constraint is an error.
  var c: ();
}
