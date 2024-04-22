// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn A();
// CHECK:STDERR: fail_redecl.carbon:[[@LINE+7]]:1: ERROR: Redeclaration of `fn A` is redundant.
// CHECK:STDERR: fn A();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR: fail_redecl.carbon:[[@LINE-4]]:1: Previously declared here.
// CHECK:STDERR: fn A();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
fn A();

fn B(x: ());
// CHECK:STDERR: fail_redecl.carbon:[[@LINE+7]]:1: ERROR: Redeclaration of `fn B` is redundant.
// CHECK:STDERR: fn B(x: ());
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR: fail_redecl.carbon:[[@LINE-4]]:1: Previously declared here.
// CHECK:STDERR: fn B(x: ());
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
fn B(x: ());

fn C();
// CHECK:STDERR: fail_redecl.carbon:[[@LINE+7]]:1: ERROR: Function redeclaration differs because of parameter count of 1.
// CHECK:STDERR: fn C(x: ());
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR: fail_redecl.carbon:[[@LINE-4]]:1: Previously declared with parameter count of 0.
// CHECK:STDERR: fn C();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
fn C(x: ());

fn D() {}
// CHECK:STDERR: fail_redecl.carbon:[[@LINE+7]]:1: ERROR: Redeclaration of `fn D` is redundant.
// CHECK:STDERR: fn D();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR: fail_redecl.carbon:[[@LINE-4]]:1: Previously declared here.
// CHECK:STDERR: fn D() {}
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
fn D();

fn E() {}
// CHECK:STDERR: fail_redecl.carbon:[[@LINE+6]]:1: ERROR: Redefinition of `fn E`.
// CHECK:STDERR: fn E() {}
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_redecl.carbon:[[@LINE-4]]:1: Previously defined here.
// CHECK:STDERR: fn E() {}
// CHECK:STDERR: ^~~~~~~~
fn E() {}

// CHECK:STDOUT: --- fail_redecl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT: }
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
// CHECK:STDOUT:     %.loc17_10.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc17_10.2: type = converted %.loc17_10.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %x.loc17_6.1: () = param x
// CHECK:STDOUT:     @B.%x: () = bind_name x, %x.loc17_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.loc25: <function> = fn_decl @B [template] {
// CHECK:STDOUT:     %.loc25_10.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc25_10.2: type = converted %.loc25_10.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %x.loc25_6.1: () = param x
// CHECK:STDOUT:     %x.loc25_6.2: () = bind_name x, %x.loc25_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C: <function> = fn_decl @C [template] {}
// CHECK:STDOUT:   %.loc35_12: <function> = fn_decl @.1 [template] {
// CHECK:STDOUT:     %.loc35_10.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc35_10.2: type = converted %.loc35_10.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %x.loc35_6.1: () = param x
// CHECK:STDOUT:     @.1.%x: () = bind_name x, %x.loc35_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.loc37: <function> = fn_decl @D [template] {}
// CHECK:STDOUT:   %D.loc45: <function> = fn_decl @D [template] {}
// CHECK:STDOUT:   %E.loc47: <function> = fn_decl @E [template] {}
// CHECK:STDOUT:   %E.loc54: <function> = fn_decl @E [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%x: ());
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1(%x: ());
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
