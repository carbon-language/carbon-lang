// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_param_in_type.carbon:[[@LINE+3]]:23: ERROR: Array bound is not a constant.
// CHECK:STDERR: fn F(n: i32, a: [i32; n]*);
// CHECK:STDERR:                       ^
fn F(n: i32, a: [i32; n]*);

// CHECK:STDOUT: --- fail_param_in_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.F = %F} [template]
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%n: i32, %a: <error>);
// CHECK:STDOUT:
