// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F();
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE+7]]:1: ERROR: Function redeclaration differs because of parameter count of 1.
// CHECK:STDERR: fn F(x: ()) {}
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE-4]]:1: Previously declared with parameter count of 0.
// CHECK:STDERR: fn F();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
fn F(x: ()) {}

fn G(x: ());
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE+7]]:1: ERROR: Function redeclaration differs because of parameter count of 0.
// CHECK:STDERR: fn G() {}
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE-4]]:1: Previously declared with parameter count of 1.
// CHECK:STDERR: fn G(x: ());
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
fn G() {}

fn H(x: ());
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE+7]]:6: ERROR: Function redeclaration differs at parameter 1.
// CHECK:STDERR: fn H(x: bool) {}
// CHECK:STDERR:      ^
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE-4]]:6: Previous declaration's corresponding parameter here.
// CHECK:STDERR: fn H(x: ());
// CHECK:STDERR:      ^
// CHECK:STDERR:
fn H(x: bool) {}

fn I();
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE+7]]:1: ERROR: Function redeclaration differs because return type is `()`.
// CHECK:STDERR: fn I() -> () { return (); }
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE-4]]:1: Previously declared with no return type.
// CHECK:STDERR: fn I();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
fn I() -> () { return (); }

fn J() -> ();
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE+7]]:1: ERROR: Function redeclaration differs because no return type is provided.
// CHECK:STDERR: fn J() {}
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE-4]]:1: Previously declared with return type `()`.
// CHECK:STDERR: fn J() -> ();
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
fn J() {}

fn K() -> ();
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE+6]]:1: ERROR: Function redeclaration differs because return type is `{}`.
// CHECK:STDERR: fn K() -> {} { return {}; }
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE-4]]:1: Previously declared with return type `()`.
// CHECK:STDERR: fn K() -> ();
// CHECK:STDERR: ^~~~~~~~~~~~~
fn K() -> {} { return {}; }

// CHECK:STDOUT: --- fail_decl_param_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %tuple: () = tuple_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %struct: {} = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F
// CHECK:STDOUT:     .G = %G
// CHECK:STDOUT:     .H = %H
// CHECK:STDOUT:     .I = %I
// CHECK:STDOUT:     .J = %J
// CHECK:STDOUT:     .K = %K
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {}
// CHECK:STDOUT:   %.loc15_13: <function> = fn_decl @.1 [template] {
// CHECK:STDOUT:     %.loc15_10.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc15_10.2: type = converted %.loc15_10.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %x.loc15_6.1: () = param x
// CHECK:STDOUT:     @.1.%x: () = bind_name x, %x.loc15_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G: <function> = fn_decl @G [template] {
// CHECK:STDOUT:     %.loc17_10.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc17_10.2: type = converted %.loc17_10.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %x.loc17_6.1: () = param x
// CHECK:STDOUT:     @G.%x: () = bind_name x, %x.loc17_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc25: <function> = fn_decl @.2 [template] {}
// CHECK:STDOUT:   %H: <function> = fn_decl @H [template] {
// CHECK:STDOUT:     %.loc27_10.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc27_10.2: type = converted %.loc27_10.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %x.loc27_6.1: () = param x
// CHECK:STDOUT:     @H.%x: () = bind_name x, %x.loc27_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc35: <function> = fn_decl @.3 [template] {
// CHECK:STDOUT:     %x.loc35_6.1: bool = param x
// CHECK:STDOUT:     @.3.%x: bool = bind_name x, %x.loc35_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I: <function> = fn_decl @I [template] {}
// CHECK:STDOUT:   %.loc45_14: <function> = fn_decl @.4 [template] {
// CHECK:STDOUT:     %.loc45_12.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc45_12.2: type = converted %.loc45_12.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     @.4.%return: ref () = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J: <function> = fn_decl @J [template] {
// CHECK:STDOUT:     %.loc47_12.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc47_12.2: type = converted %.loc47_12.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     @J.%return: ref () = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc55: <function> = fn_decl @.5 [template] {}
// CHECK:STDOUT:   %K: <function> = fn_decl @K [template] {
// CHECK:STDOUT:     %.loc57_12.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc57_12.2: type = converted %.loc57_12.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     @K.%return: ref () = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc64_14: <function> = fn_decl @.6 [template] {
// CHECK:STDOUT:     %.loc64_12.1: {} = struct_literal ()
// CHECK:STDOUT:     %.loc64_12.2: type = converted %.loc64_12.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:     @.6.%return: ref {} = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1(%x: ()) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%x: ());
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H(%x: ());
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.3(%x: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @I();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.4() -> () {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc45_24: () = tuple_literal ()
// CHECK:STDOUT:   %tuple: () = tuple_value () [template = constants.%tuple]
// CHECK:STDOUT:   %.loc45_25: () = converted %.loc45_24, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   return %.loc45_25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @J() -> ();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.5() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @K() -> ();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.6() -> {} {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc64_24: {} = struct_literal ()
// CHECK:STDOUT:   %struct: {} = struct_value () [template = constants.%struct]
// CHECK:STDOUT:   %.loc64_25: {} = converted %.loc64_24, %struct [template = constants.%struct]
// CHECK:STDOUT:   return %.loc64_25
// CHECK:STDOUT: }
// CHECK:STDOUT:
