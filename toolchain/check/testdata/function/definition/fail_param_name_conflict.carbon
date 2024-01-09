// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_param_name_conflict.carbon:[[@LINE+6]]:16: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: fn Bar(a: i32, a: i32) {}
// CHECK:STDERR:                ^
// CHECK:STDERR: fail_param_name_conflict.carbon:[[@LINE+3]]:8: Name is previously declared here.
// CHECK:STDERR: fn Bar(a: i32, a: i32) {}
// CHECK:STDERR:        ^
fn Bar(a: i32, a: i32) {}

// CHECK:STDOUT: --- fail_param_name_conflict.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Bar = %Bar}
// CHECK:STDOUT:   %Bar: <function> = fn_decl @Bar, const
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bar(%a.loc13_8: i32, %a.loc13_16: i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
