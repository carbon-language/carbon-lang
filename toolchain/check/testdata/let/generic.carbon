// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/let/generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/let/generic.carbon

// --- fail_todo_type.carbon
library "[[@TEST_NAME]]";

fn F() {
  // CHECK:STDERR: fail_todo_type.carbon:[[@LINE+4]]:7: error: semantics TODO: `local `let :!` bindings are currently unsupported` [SemanticsTodo]
  // CHECK:STDERR:   let T:! type = i32;
  // CHECK:STDERR:       ^~~~~~~~
  // CHECK:STDERR:
  let T:! type = i32;
  var p: T*;
  let a: T = *p;
}

// --- fail_todo_assignment.carbon
library "[[@TEST_NAME]]";

fn F(a: i32) -> i32 {
  // CHECK:STDERR: fail_todo_assignment.carbon:[[@LINE+4]]:7: error: semantics TODO: `local `let :!` bindings are currently unsupported` [SemanticsTodo]
  // CHECK:STDERR:   let T:! type = i32;
  // CHECK:STDERR:       ^~~~~~~~
  // CHECK:STDERR:
  let T:! type = i32;
  let x: T = 5;
  return x;
}

// CHECK:STDOUT: --- fail_todo_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_assignment.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a.param: %i32) -> %i32;
// CHECK:STDOUT:
