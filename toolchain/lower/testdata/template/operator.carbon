// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// EXTRA-ARGS: --dump-sem-ir
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/template/operator.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/template/operator.carbon

fn Add[template T: type](a: T, b: T) -> T {
  // CHECK:STDERR: operator.carbon:[[@LINE+4]]:10: error: name `Core.AddWith` implicitly referenced here, but not found [CoreNameNotFound]
  // CHECK:STDERR:   return a + b;
  // CHECK:STDERR:          ^~~~~
  // CHECK:STDERR:
  return a + b;
}

// CHECK:STDERR: operator.carbon:[[@LINE+7]]:30: error: no return expression should be provided in this context [ReturnStatementDisallowExpr]
// CHECK:STDERR: fn CallAdd(a: i32, b: i32) { return Add(a, b); }
// CHECK:STDERR:                              ^~~~~~~~~~~~~~~~~
// CHECK:STDERR: operator.carbon:[[@LINE+4]]:1: note: there was no return type provided [ReturnTypeOmittedNote]
// CHECK:STDERR: fn CallAdd(a: i32, b: i32) { return Add(a, b); }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn CallAdd(a: i32, b: i32) -> i32 { return Add(a, b); }

