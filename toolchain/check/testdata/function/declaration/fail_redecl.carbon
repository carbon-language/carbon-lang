// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn A();
// CHECK:STDERR: fail_redecl.carbon:[[@LINE+6]]:1: ERROR: Redeclaration of function A.
// CHECK:STDERR: fn A();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR: fail_redecl.carbon:[[@LINE-4]]:1: Previous declaration was here.
// CHECK:STDERR: fn A();
// CHECK:STDERR: ^~~~~~~
fn A();

fn B(x: i32);
// CHECK:STDERR: fail_redecl.carbon:[[@LINE+6]]:1: ERROR: Redeclaration of function B.
// CHECK:STDERR: fn B(x: i32);
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR: fail_redecl.carbon:[[@LINE-4]]:1: Previous declaration was here.
// CHECK:STDERR: fn B(x: i32);
// CHECK:STDERR: ^~~~~~~~~~~~~
fn B(x: i32);

fn C();
// CHECK:STDERR: fail_redecl.carbon:[[@LINE+6]]:1: ERROR: Function declared with 1 parameter(s).
// CHECK:STDERR: fn C(x: i32);
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR: fail_redecl.carbon:[[@LINE-4]]:1: Function previously declared with 0 parameter(s).
// CHECK:STDERR: fn C();
// CHECK:STDERR: ^~~~~~~
fn C(x: i32);

fn D() {}
// CHECK:STDERR: fail_redecl.carbon:[[@LINE+6]]:1: ERROR: Redeclaration of function D.
// CHECK:STDERR: fn D();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR: fail_redecl.carbon:[[@LINE-4]]:1: Previous declaration was here.
// CHECK:STDERR: fn D() {}
// CHECK:STDERR: ^~~~~~~~
fn D();

fn E() {}
// CHECK:STDERR: fail_redecl.carbon:[[@LINE+6]]:1: ERROR: Redefinition of function E.
// CHECK:STDERR: fn E() {}
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_redecl.carbon:[[@LINE-4]]:1: Previous definition was here.
// CHECK:STDERR: fn E() {}
// CHECK:STDERR: ^~~~~~~~
fn E() {}

// CHECK:STDOUT: --- fail_redecl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %A.loc7
// CHECK:STDOUT:     .B = %B.loc16
// CHECK:STDOUT:     .C = %C
// CHECK:STDOUT:     .D = %D.loc34
// CHECK:STDOUT:     .E = %E.loc43
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.loc7: <function> = fn_decl @A [template] {}
// CHECK:STDOUT:   %A.loc14: <function> = fn_decl @A [template] {}
// CHECK:STDOUT:   %B.loc16: <function> = fn_decl @B [template] {
// CHECK:STDOUT:     %x.loc16_6.1: i32 = param x
// CHECK:STDOUT:     @B.%x: i32 = bind_name x, %x.loc16_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.loc23: <function> = fn_decl @B [template] {
// CHECK:STDOUT:     %x.loc23_6.1: i32 = param x
// CHECK:STDOUT:     %x.loc23_6.2: i32 = bind_name x, %x.loc23_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C: <function> = fn_decl @C [template] {}
// CHECK:STDOUT:   %.loc32: <function> = fn_decl @.1 [template] {
// CHECK:STDOUT:     %x.loc32_6.1: i32 = param x
// CHECK:STDOUT:     @.1.%x: i32 = bind_name x, %x.loc32_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.loc34: <function> = fn_decl @D [template] {}
// CHECK:STDOUT:   %D.loc41: <function> = fn_decl @D [template] {}
// CHECK:STDOUT:   %E.loc43: <function> = fn_decl @E [template] {}
// CHECK:STDOUT:   %E.loc50: <function> = fn_decl @E [template] {}
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
// CHECK:STDOUT: !.loc50:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
