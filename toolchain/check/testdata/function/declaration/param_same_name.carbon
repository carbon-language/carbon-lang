// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F(a: i32);

fn G(a: i32);

// CHECK:STDOUT: --- param_same_name.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {
// CHECK:STDOUT:     .F = %F
// CHECK:STDOUT:     .G = %G
// CHECK:STDOUT:   } [template]
// CHECK:STDOUT:   %F: <function> = fn_decl @F {
// CHECK:STDOUT:     %a.loc7_6.1: i32 = param a
// CHECK:STDOUT:     @F.%a: i32 = bind_name a, %a.loc7_6.1
// CHECK:STDOUT:   } [template]
// CHECK:STDOUT:   %G: <function> = fn_decl @G {
// CHECK:STDOUT:     %a.loc9_6.1: i32 = param a
// CHECK:STDOUT:     @G.%a: i32 = bind_name a, %a.loc9_6.1
// CHECK:STDOUT:   } [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%a: i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%a: i32);
// CHECK:STDOUT:
