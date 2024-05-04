// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn A(n: i32, m: i32) -> i32 = "int.sadd";
// CHECK:STDERR: fail_redefined.carbon:[[@LINE+7]]:1: ERROR: Redefinition of `fn A`.
// CHECK:STDERR: fn A(n: i32, m: i32) -> i32 { return n; }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_redefined.carbon:[[@LINE-4]]:1: Previously defined here.
// CHECK:STDERR: fn A(n: i32, m: i32) -> i32 = "int.sadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn A(n: i32, m: i32) -> i32 { return n; }

fn B(n: i32, m: i32) -> i32 { return n; }
// CHECK:STDERR: fail_redefined.carbon:[[@LINE+7]]:1: ERROR: Redefinition of `fn B`.
// CHECK:STDERR: fn B(n: i32, m: i32) -> i32 = "int.sadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_redefined.carbon:[[@LINE-4]]:1: Previously defined here.
// CHECK:STDERR: fn B(n: i32, m: i32) -> i32 { return n; }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn B(n: i32, m: i32) -> i32 = "int.sadd";

fn C(n: i32, m: i32) -> i32 = "int.sadd";
// CHECK:STDERR: fail_redefined.carbon:[[@LINE+6]]:1: ERROR: Redefinition of `fn C`.
// CHECK:STDERR: fn C(n: i32, m: i32) -> i32 = "int.sadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_redefined.carbon:[[@LINE-4]]:1: Previously defined here.
// CHECK:STDERR: fn C(n: i32, m: i32) -> i32 = "int.sadd";
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fn C(n: i32, m: i32) -> i32 = "int.sadd";

// CHECK:STDOUT: --- fail_redefined.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = fn_type @A [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: A = struct_value () [template]
// CHECK:STDOUT:   %B: type = fn_type @B [template]
// CHECK:STDOUT:   %struct.2: B = struct_value () [template]
// CHECK:STDOUT:   %C: type = fn_type @C [template]
// CHECK:STDOUT:   %struct.3: C = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .A = %A.decl.loc7
// CHECK:STDOUT:     .B = %B.decl.loc17
// CHECK:STDOUT:     .C = %C.decl.loc27
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %A.decl.loc7: A = fn_decl @A [template = constants.%struct.1] {
// CHECK:STDOUT:     %n.loc7_6.1: i32 = param n
// CHECK:STDOUT:     %n.loc7_6.2: i32 = bind_name n, %n.loc7_6.1
// CHECK:STDOUT:     %m.loc7_14.1: i32 = param m
// CHECK:STDOUT:     %m.loc7_14.2: i32 = bind_name m, %m.loc7_14.1
// CHECK:STDOUT:     %return.var.loc7: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl.loc15: A = fn_decl @A [template = constants.%struct.1] {
// CHECK:STDOUT:     %n.loc15_6.1: i32 = param n
// CHECK:STDOUT:     @A.%n: i32 = bind_name n, %n.loc15_6.1
// CHECK:STDOUT:     %m.loc15_14.1: i32 = param m
// CHECK:STDOUT:     @A.%m: i32 = bind_name m, %m.loc15_14.1
// CHECK:STDOUT:     @A.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl.loc17: B = fn_decl @B [template = constants.%struct.2] {
// CHECK:STDOUT:     %n.loc17_6.1: i32 = param n
// CHECK:STDOUT:     %n.loc17_6.2: i32 = bind_name n, %n.loc17_6.1
// CHECK:STDOUT:     %m.loc17_14.1: i32 = param m
// CHECK:STDOUT:     %m.loc17_14.2: i32 = bind_name m, %m.loc17_14.1
// CHECK:STDOUT:     %return.var.loc17: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl.loc25: B = fn_decl @B [template = constants.%struct.2] {
// CHECK:STDOUT:     %n.loc25_6.1: i32 = param n
// CHECK:STDOUT:     @B.%n: i32 = bind_name n, %n.loc25_6.1
// CHECK:STDOUT:     %m.loc25_14.1: i32 = param m
// CHECK:STDOUT:     @B.%m: i32 = bind_name m, %m.loc25_14.1
// CHECK:STDOUT:     @B.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc27: C = fn_decl @C [template = constants.%struct.3] {
// CHECK:STDOUT:     %n.loc27_6.1: i32 = param n
// CHECK:STDOUT:     %n.loc27_6.2: i32 = bind_name n, %n.loc27_6.1
// CHECK:STDOUT:     %m.loc27_14.1: i32 = param m
// CHECK:STDOUT:     %m.loc27_14.2: i32 = bind_name m, %m.loc27_14.1
// CHECK:STDOUT:     %return.var.loc27: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl.loc34: C = fn_decl @C [template = constants.%struct.3] {
// CHECK:STDOUT:     %n.loc34_6.1: i32 = param n
// CHECK:STDOUT:     @C.%n: i32 = bind_name n, %n.loc34_6.1
// CHECK:STDOUT:     %m.loc34_14.1: i32 = param m
// CHECK:STDOUT:     @C.%m: i32 = bind_name m, %m.loc34_14.1
// CHECK:STDOUT:     @C.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A(%n: i32, %m: i32) -> i32 = "int.sadd" {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: i32 = name_ref n, %n
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%n: i32, %m: i32) -> i32 = "int.sadd" {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: i32 = name_ref n, file.%n.loc17_6.2
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C(%n: i32, %m: i32) -> i32 = "int.sadd";
// CHECK:STDOUT:
