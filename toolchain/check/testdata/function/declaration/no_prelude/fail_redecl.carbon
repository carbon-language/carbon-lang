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
// CHECK:STDERR: fail_redecl.carbon:[[@LINE+7]]:1: ERROR: Redeclaration differs because of parameter count of 1.
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
// CHECK:STDOUT:   %A: type = fn_type @A [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: A = struct_value () [template]
// CHECK:STDOUT:   %B: type = fn_type @B [template]
// CHECK:STDOUT:   %struct.2: B = struct_value () [template]
// CHECK:STDOUT:   %C: type = fn_type @C [template]
// CHECK:STDOUT:   %struct.3: C = struct_value () [template]
// CHECK:STDOUT:   %.2: type = fn_type @.1 [template]
// CHECK:STDOUT:   %struct.4: <invalid> = struct_value () [template]
// CHECK:STDOUT:   %D: type = fn_type @D [template]
// CHECK:STDOUT:   %struct.5: D = struct_value () [template]
// CHECK:STDOUT:   %E: type = fn_type @E [template]
// CHECK:STDOUT:   %struct.6: E = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %A.decl.loc7
// CHECK:STDOUT:     .B = %B.decl.loc17
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl.loc37
// CHECK:STDOUT:     .E = %E.decl.loc47
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A.decl.loc7: A = fn_decl @A [template = constants.%struct.1] {}
// CHECK:STDOUT:   %A.decl.loc15: A = fn_decl @A [template = constants.%struct.1] {}
// CHECK:STDOUT:   %B.decl.loc17: B = fn_decl @B [template = constants.%struct.2] {
// CHECK:STDOUT:     %.loc17_10.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc17_10.2: type = converted %.loc17_10.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %x.loc17_6.1: () = param x
// CHECK:STDOUT:     @B.%x: () = bind_name x, %x.loc17_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl.loc25: B = fn_decl @B [template = constants.%struct.2] {
// CHECK:STDOUT:     %.loc25_10.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc25_10.2: type = converted %.loc25_10.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %x.loc25_6.1: () = param x
// CHECK:STDOUT:     %x.loc25_6.2: () = bind_name x, %x.loc25_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: C = fn_decl @C [template = constants.%struct.3] {}
// CHECK:STDOUT:   %.decl: <invalid> = fn_decl @.1 [template = constants.%struct.4] {
// CHECK:STDOUT:     %.loc35_10.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc35_10.2: type = converted %.loc35_10.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %x.loc35_6.1: () = param x
// CHECK:STDOUT:     @.1.%x: () = bind_name x, %x.loc35_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl.loc37: D = fn_decl @D [template = constants.%struct.5] {}
// CHECK:STDOUT:   %D.decl.loc45: D = fn_decl @D [template = constants.%struct.5] {}
// CHECK:STDOUT:   %E.decl.loc47: E = fn_decl @E [template = constants.%struct.6] {}
// CHECK:STDOUT:   %E.decl.loc54: E = fn_decl @E [template = constants.%struct.6] {}
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
