// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/dataflow/unused.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/dataflow/unused.carbon

// --- todo_ensure_no_warning.carbon

library "no_unused_warning";

fn ReturnVar() -> i32 {
  var x: i32 = 0;
  x = 1;
  return x;
   // TODO: ensure no warning.
 }

fn ReturnLet() -> i32 {
  let y: i32 = 1;
  return y;
  // TODO: ensure no warning.
}

fn RefParam(ref x: i32) {
  x = 1;
  // TODO: ensure no warning.
}

fn ReturnedVar() -> i32 {
  returned var x: i32 = 0;
  return var;
  // TODO: ensure no warning.
}

fn ReturnVarExplicit() -> i32 {
  var x: i32 = 0;
  return x;
  // TODO: ensure no warning.
}

fn FieldAccess() {
  var x: {.a: i32} = {.a = 0};
  // Accessing a field should count as a use of x.
  var y: i32 = x.a;
  // TODO: ensure no warning.
}

fn ParametersUsed(x: i32) -> i32 {
  // TODO: ensure no warning.
  return x;
}

// --- fail_todo_unused_ref_param.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_todo_ensure_no_warning.carbon:[[@LINE+4]]:19: error: semantics TODO: `unused` [SemanticsTodo]
// CHECK:STDERR: fn RefParamUnused(unused ref y: i32) {
// CHECK:STDERR:                   ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn RefParamUnused(unused ref y: i32) {
  // TODO: ensure no warning
}

// --- todo_ensure_warning.carbon

library "unused_warning";

fn LetUnused() {
  let x: i32 = 0;
  // TODO: warning, x is unused.
}

fn VarUnused() {
  var a: i32 = 0;
  // TODO: warning, a is unused.
}

fn Shadowing() {
  var x: i32 = 0;
  if (true) {
    var x: i32 = 1;
    // TODO: warning, x is unused.
  }
  x = 2;
  // TODO: warning, x is unused.
}

fn Parameters(p: i32) {
  // TODO: warning, p is unused.
}

// --- todo_fail_param_ref_named_not_used.carbon
library "[[@TEST_NAME]]";

fn F(ref y: i32) {
  // TODO: warning, y is unused.
}

// --- fail_todo_read_from_param_unused_ref.carbon
library "[[@TEST_NAME]]";
// CHECK:STDERR: fail_todo_ensure_error.carbon:[[@LINE+4]]:19: error: semantics TODO: `unused` [SemanticsTodo]
// CHECK:STDERR: fn RefParamUnused(unused ref y: i32) {
// CHECK:STDERR:                   ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn RefParamUnused(unused ref y: i32) {
  var _: i32 = y;
  // TODO: ensure error, marked "unused" but used.
}

// --- todo_fail_unused_returned_var.carbon
library "[[@TEST_NAME]]";

fn F() -> i32 {
  returned var unused x: i32 = 0;
  // TODO: ensure error, marked "unused" but used.
  return var;
}

fn Underscore(x: i32) {
  var _: i32 = x;
  var unused y: i32 = x;
  // TODO: error, marked "unused", but used
  var unused z: i32 = y;
}
