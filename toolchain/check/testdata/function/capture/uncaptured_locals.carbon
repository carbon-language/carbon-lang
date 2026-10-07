// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/capture/uncaptured_locals.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/capture/uncaptured_locals.carbon

// --- fail_todo_reference_uncaptured.carbon
library "[[@TEST_NAME]]";

class X {}

fn Test() {
  let x: X = {};
  fn Inner() {
    // TODO: This should fail. We can't reference `x` from here.
    let unused y: X = x;
  }
}

// Intentionally include an error here so we skip dominance checking, which CHECK-fails for the above example.
// CHECK:STDERR: fail_todo_reference_uncaptured.carbon:[[@LINE+4]]:14: error: name `error` not found [NameNotFound]
// CHECK:STDERR: fn Error() { error; }
// CHECK:STDERR:              ^~~~~
// CHECK:STDERR:
fn Error() { error; }

// --- fail_todo_reference_uncaptured_lambda.carbon
library "[[@TEST_NAME]]";

class X {}

fn Test() {
  let x: X = {};
  (fn() {
    // TODO: This should fail. We can't reference `x` from here.
    let unused y: X = x;
  });
}

// Intentionally include an error here so we skip dominance checking, which CHECK-fails for the above example.
// CHECK:STDERR: fail_todo_reference_uncaptured_lambda.carbon:[[@LINE+4]]:14: error: name `error` not found [NameNotFound]
// CHECK:STDERR: fn Error() { error; }
// CHECK:STDERR:              ^~~~~
// CHECK:STDERR:
fn Error() { error; }
