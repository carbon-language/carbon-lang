// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn A();
// CHECK:STDERR: fail_redecl.carbon:[[@LINE+7]]:1: ERROR: Redundant redeclaration of function A.
// CHECK:STDERR: fn A();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR: fail_redecl.carbon:[[@LINE-4]]:1: Previously declared here.
// CHECK:STDERR: fn A();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
fn A();

fn B(x: i32);
// CHECK:STDERR: fail_redecl.carbon:[[@LINE+7]]:1: ERROR: Redundant redeclaration of function B.
// CHECK:STDERR: fn B(x: i32);
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR: fail_redecl.carbon:[[@LINE-4]]:1: Previously declared here.
// CHECK:STDERR: fn B(x: i32);
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
fn B(x: i32);

fn C();
// CHECK:STDERR: fail_redecl.carbon:[[@LINE+7]]:1: ERROR: Function redeclaration differs because of parameter count of 1.
// CHECK:STDERR: fn C(x: i32);
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR: fail_redecl.carbon:[[@LINE-4]]:1: Previously declared with parameter count of 0.
// CHECK:STDERR: fn C();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
fn C(x: i32);

fn D() {}
// CHECK:STDERR: fail_redecl.carbon:[[@LINE+7]]:1: ERROR: Redundant redeclaration of function D.
// CHECK:STDERR: fn D();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR: fail_redecl.carbon:[[@LINE-4]]:1: Previously declared here.
// CHECK:STDERR: fn D() {}
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
fn D();

fn E() {}
// CHECK:STDERR: fail_redecl.carbon:[[@LINE+6]]:1: ERROR: Redefinition of function E.
// CHECK:STDERR: fn E() {}
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_redecl.carbon:[[@LINE-4]]:1: Previously defined here.
// CHECK:STDERR: fn E() {}
// CHECK:STDERR: ^~~~~~~~
fn E() {}

// CHECK:STDOUT: --- fail_redecl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %A.loc7
// CHECK:STDOUT:     .B = %B.loc17
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:     .D = %D.loc37
// CHECK:STDOUT:     .E = %E.loc47
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.loc7: <function> = fn_decl @A [template] {}
// CHECK:STDOUT:   %A.loc15: <function> = fn_decl @A [template] {}
// CHECK:STDOUT:   %B.loc17: <function> = fn_decl @B [template] {
// CHECK:STDOUT:     %x.loc17_6.1: i32 = param x
// CHECK:STDOUT:     @B.%x: i32 = bind_name x, %x.loc17_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.loc25: <function> = fn_decl @B [template] {
// CHECK:STDOUT:     %x.loc25_6.1: i32 = param x
// CHECK:STDOUT:     %x.loc25_6.2: i32 = bind_name x, %x.loc25_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C: <function> = fn_decl @C [template] {}
// CHECK:STDOUT:   %.loc35: <function> = fn_decl @.1 [template] {
// CHECK:STDOUT:     %x.loc35_6.1: i32 = param x
// CHECK:STDOUT:     @.1.%x: i32 = bind_name x, %x.loc35_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.loc37: <function> = fn_decl @D [template] {}
// CHECK:STDOUT:   %D.loc45: <function> = fn_decl @D [template] {}
// CHECK:STDOUT:   %E.loc47: <function> = fn_decl @E [template] {}
// CHECK:STDOUT:   %E.loc54: <function> = fn_decl @E [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%x: i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1(%x: i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @D() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @E() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT:
// CHECK:STDOUT: !.loc54:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
