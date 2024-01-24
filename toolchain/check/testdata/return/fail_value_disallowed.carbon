// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Main() {
  // CHECK:STDERR: fail_value_disallowed.carbon:[[@LINE+6]]:3: ERROR: No return expression should be provided in this context.
  // CHECK:STDERR:   return 0;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_value_disallowed.carbon:[[@LINE-4]]:1: There was no return type provided.
  // CHECK:STDERR: fn Main() {
  // CHECK:STDERR: ^~~~~~~~~~~
  return 0;
}

// CHECK:STDOUT: --- fail_value_disallowed.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Main = %Main} [template]
// CHECK:STDOUT:   %Main: <function> = fn_decl @Main [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14: i32 = int_literal 0 [template = constants.%.1]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
