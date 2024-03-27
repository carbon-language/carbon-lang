// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn A(n: i32, m: i32) -> i32 = "int.add";
// CHECK:STDERR: fail_redefined.carbon:[[@LINE+7]]:1: ERROR: Redefinition of function A.
// CHECK:STDERR: fn A(n: i32, m: i32) -> i32 { return n; }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_redefined.carbon:[[@LINE-4]]:1: Previously defined here.
// CHECK:STDERR: fn A(n: i32, m: i32) -> i32 = "int.add";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn A(n: i32, m: i32) -> i32 { return n; }

fn B(n: i32, m: i32) -> i32 { return n; }
// CHECK:STDERR: fail_redefined.carbon:[[@LINE+7]]:1: ERROR: Redefinition of function B.
// CHECK:STDERR: fn B(n: i32, m: i32) -> i32 = "int.add";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_redefined.carbon:[[@LINE-4]]:1: Previously defined here.
// CHECK:STDERR: fn B(n: i32, m: i32) -> i32 { return n; }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn B(n: i32, m: i32) -> i32 = "int.add";

fn C(n: i32, m: i32) -> i32 = "int.add";
// CHECK:STDERR: fail_redefined.carbon:[[@LINE+6]]:1: ERROR: Redefinition of function C.
// CHECK:STDERR: fn C(n: i32, m: i32) -> i32 = "int.add";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_redefined.carbon:[[@LINE-4]]:1: Previously defined here.
// CHECK:STDERR: fn C(n: i32, m: i32) -> i32 = "int.add";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fn C(n: i32, m: i32) -> i32 = "int.add";

// CHECK:STDOUT: --- fail_redefined.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %A.loc7
// CHECK:STDOUT:     .B = %B.loc17
// CHECK:STDOUT:     .C = %C.loc27
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.loc7: <function> = fn_decl @A [template] {
// CHECK:STDOUT:     %n.loc7_6.1: i32 = param n
// CHECK:STDOUT:     %n.loc7_6.2: i32 = bind_name n, %n.loc7_6.1
// CHECK:STDOUT:     %m.loc7_14.1: i32 = param m
// CHECK:STDOUT:     %m.loc7_14.2: i32 = bind_name m, %m.loc7_14.1
// CHECK:STDOUT:     %return.var.loc7: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.loc15: <function> = fn_decl @A [template] {
// CHECK:STDOUT:     %n.loc15_6.1: i32 = param n
// CHECK:STDOUT:     @A.%n: i32 = bind_name n, %n.loc15_6.1
// CHECK:STDOUT:     %m.loc15_14.1: i32 = param m
// CHECK:STDOUT:     @A.%m: i32 = bind_name m, %m.loc15_14.1
// CHECK:STDOUT:     %return.var.loc15: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.loc17: <function> = fn_decl @B [template] {
// CHECK:STDOUT:     %n.loc17_6.1: i32 = param n
// CHECK:STDOUT:     %n.loc17_6.2: i32 = bind_name n, %n.loc17_6.1
// CHECK:STDOUT:     %m.loc17_14.1: i32 = param m
// CHECK:STDOUT:     %m.loc17_14.2: i32 = bind_name m, %m.loc17_14.1
// CHECK:STDOUT:     %return.var.loc17: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.loc25: <function> = fn_decl @B [template] {
// CHECK:STDOUT:     %n.loc25_6.1: i32 = param n
// CHECK:STDOUT:     @B.%n: i32 = bind_name n, %n.loc25_6.1
// CHECK:STDOUT:     %m.loc25_14.1: i32 = param m
// CHECK:STDOUT:     @B.%m: i32 = bind_name m, %m.loc25_14.1
// CHECK:STDOUT:     %return.var.loc25: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.loc27: <function> = fn_decl @C [template] {
// CHECK:STDOUT:     %n.loc27_6.1: i32 = param n
// CHECK:STDOUT:     %n.loc27_6.2: i32 = bind_name n, %n.loc27_6.1
// CHECK:STDOUT:     %m.loc27_14.1: i32 = param m
// CHECK:STDOUT:     %m.loc27_14.2: i32 = bind_name m, %m.loc27_14.1
// CHECK:STDOUT:     %return.var.loc27: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.loc34: <function> = fn_decl @C [template] {
// CHECK:STDOUT:     %n.loc34_6.1: i32 = param n
// CHECK:STDOUT:     @C.%n: i32 = bind_name n, %n.loc34_6.1
// CHECK:STDOUT:     %m.loc34_14.1: i32 = param m
// CHECK:STDOUT:     @C.%m: i32 = bind_name m, %m.loc34_14.1
// CHECK:STDOUT:     %return.var.loc34: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A(%n: i32, %m: i32) -> i32 = "int.add" {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: i32 = name_ref n, %n
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%n: i32, %m: i32) -> i32 = "int.add" {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: i32 = name_ref n, file.%n.loc17_6.2
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C(%n: i32, %m: i32) -> i32 = "int.add";
// CHECK:STDOUT:
