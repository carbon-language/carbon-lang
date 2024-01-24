// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_bad_run_2.carbon:[[@LINE+3]]:1: ERROR: Invalid signature for `Main.Run` function. Expected `fn ()` or `fn () -> i32`.
// CHECK:STDERR: fn Run(n: i32) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~
fn Run(n: i32) {}

// CHECK:STDOUT: --- fail_bad_run_2.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Run = %Run} [template]
// CHECK:STDOUT:   %Run: <function> = fn_decl @Run [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run(%n: i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
