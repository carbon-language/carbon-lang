// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_bad_run.carbon:[[@LINE+6]]:1: ERROR: Invalid signature for `Main.Run` function. Expected `fn ()` or `fn () -> i32`.
// CHECK:STDERR: fn Run() -> String {}
// CHECK:STDERR: ^
// CHECK:STDERR: fail_bad_run.carbon:[[@LINE+3]]:21: ERROR: Missing `return` at end of function with declared return type.
// CHECK:STDERR: fn Run() -> String {}
// CHECK:STDERR:                     ^
fn Run() -> String {}

// CHECK:STDOUT: file "fail_bad_run.carbon" {
// CHECK:STDOUT:   %.1: type = ptr_type String
// CHECK:STDOUT:   %Run: <function> = fn_decl @Run
// CHECK:STDOUT:   %.loc13: type = tuple_type ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() -> %return: String {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT: }
