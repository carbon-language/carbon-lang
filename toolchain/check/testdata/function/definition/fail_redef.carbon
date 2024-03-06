// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() {}
// CHECK:STDERR: fail_redef.carbon:[[@LINE+6]]:1: ERROR: Redefinition of function F.
// CHECK:STDERR: fn F() {}
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_redef.carbon:[[@LINE-4]]:1: Previously defined here.
// CHECK:STDERR: fn F() {}
// CHECK:STDERR: ^~~~~~~~
fn F() {}

// CHECK:STDOUT: --- fail_redef.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F.loc7
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.loc7: <function> = fn_decl @F [template] {}
// CHECK:STDOUT:   %F.loc14: <function> = fn_decl @F [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc14:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
