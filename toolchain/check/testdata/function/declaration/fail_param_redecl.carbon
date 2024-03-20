// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_param_redecl.carbon:[[@LINE+6]]:14: ERROR: Duplicate name being declared in the same scope.
// CHECK:STDERR: fn F(n: i32, n: i32);
// CHECK:STDERR:              ^
// CHECK:STDERR: fail_param_redecl.carbon:[[@LINE+3]]:6: Name is previously declared here.
// CHECK:STDERR: fn F(n: i32, n: i32);
// CHECK:STDERR:      ^
fn F(n: i32, n: i32);

// CHECK:STDOUT: --- fail_param_redecl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {
// CHECK:STDOUT:     %n.loc13_6.1: i32 = param n
// CHECK:STDOUT:     @F.%n.loc13_6: i32 = bind_name n, %n.loc13_6.1
// CHECK:STDOUT:     %n.loc13_14.1: i32 = param n
// CHECK:STDOUT:     @F.%n.loc13_14: i32 = bind_name n, %n.loc13_14.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%n.loc13_6: i32, %n.loc13_14: i32);
// CHECK:STDOUT:
