// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Main() -> i32 {
  // CHECK:STDERR: fail_value_missing.carbon:[[@LINE+6]]:3: ERROR: Missing return value.
  // CHECK:STDERR:   return;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_value_missing.carbon:[[@LINE-4]]:1: Return type of function is `i32`.
  // CHECK:STDERR: fn Main() -> i32 {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  return;
}

// CHECK:STDOUT: --- fail_value_missing.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Main = %Main} [template]
// CHECK:STDOUT:   %Main: <function> = fn_decl @Main [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
