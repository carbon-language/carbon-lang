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

// --- no_unused_error.carbon

library "no_unused_error";

fn SimpleUsed() {
  var x: i32 = 0;
  x = 1;
  var _: i32 = x;
}

fn RefParam(ref x: i32) {
  x = 1;
}

fn ReturnedVar() -> i32 {
  returned var x: i32 = 0;
  return var;
}

fn ReturnVarExplicit() -> i32 {
  var x: i32 = 0;
  return x;
}

fn FieldAccess() {
  var x: {.a: i32} = {.a = 0};
  // Accessing a field should count as a use of x.
  var y: i32 = x.a;
}

fn ParametersUsed(x: i32) -> i32 {
  return x;
}

// --- fail_unused.carbon

fn LetUnused() {
  let x: i32 = 0;
}

fn VarUnused() {
  var a: i32 = 0;
}

fn Shadowing() {
  var x: i32 = 0;
  if (true) {
    var x: i32 = 1;
  }
  x = 2;
}

fn Parameters(p: i32) {
}

fn Underscore(x: i32) {
  var _: i32 = x;
  // CHECK:STDERR: fail_unused.carbon:[[@LINE+4]]:7: error: semantics TODO: `unused` [SemanticsTodo]
  // CHECK:STDERR:   var unused y: i32 = x;
  // CHECK:STDERR:       ^~~~~~~~~~~~~
  // CHECK:STDERR:
  var unused y: i32 = x;
}

fn RefParamUnused(ref y: i32) {
}
