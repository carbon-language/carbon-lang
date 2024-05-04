// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_bad_run.carbon:[[@LINE+7]]:1: ERROR: Invalid signature for `Main.Run` function. Expected `fn ()` or `fn () -> i32`.
// CHECK:STDERR: fn Run() -> String {}
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_bad_run.carbon:[[@LINE+3]]:21: ERROR: Missing `return` at end of function with declared return type.
// CHECK:STDERR: fn Run() -> String {}
// CHECK:STDERR:                     ^
fn Run() -> String {}

// CHECK:STDOUT: --- fail_bad_run.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Run: type = fn_type @Run [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Run = struct_value () [template]
// CHECK:STDOUT:   %.2: type = ptr_type String [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Run.decl: Run = fn_decl @Run [template = constants.%struct] {
// CHECK:STDOUT:     @Run.%return: ref String = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() -> %return: String {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT: }
// CHECK:STDOUT:
