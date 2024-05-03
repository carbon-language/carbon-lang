// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F();
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE+7]]:1: ERROR: Redeclaration differs because of parameter count of 1.
// CHECK:STDERR: fn F(x: ()) {}
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE-4]]:1: Previously declared with parameter count of 0.
// CHECK:STDERR: fn F();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
fn F(x: ()) {}

fn G(x: ());
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE+7]]:1: ERROR: Redeclaration differs because of parameter count of 0.
// CHECK:STDERR: fn G() {}
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE-4]]:1: Previously declared with parameter count of 1.
// CHECK:STDERR: fn G(x: ());
// CHECK:STDERR: ^~~~~~~~~~~~
// CHECK:STDERR:
fn G() {}

fn H(x: ());
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE+7]]:6: ERROR: Redeclaration differs at parameter 1.
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
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: F = struct_value () [template]
// CHECK:STDOUT:   %.2: type = fn_type @.1 [template]
// CHECK:STDOUT:   %struct.2: <invalid> = struct_value () [template]
// CHECK:STDOUT:   %G: type = fn_type @G [template]
// CHECK:STDOUT:   %struct.3: G = struct_value () [template]
// CHECK:STDOUT:   %.3: type = fn_type @.2 [template]
// CHECK:STDOUT:   %struct.4: <invalid> = struct_value () [template]
// CHECK:STDOUT:   %H: type = fn_type @H [template]
// CHECK:STDOUT:   %struct.5: H = struct_value () [template]
// CHECK:STDOUT:   %.4: type = fn_type @.3 [template]
// CHECK:STDOUT:   %struct.6: <invalid> = struct_value () [template]
// CHECK:STDOUT:   %I: type = fn_type @I [template]
// CHECK:STDOUT:   %struct.7: I = struct_value () [template]
// CHECK:STDOUT:   %.5: type = fn_type @.4 [template]
// CHECK:STDOUT:   %struct.8: <invalid> = struct_value () [template]
// CHECK:STDOUT:   %tuple: () = tuple_value () [template]
// CHECK:STDOUT:   %J: type = fn_type @J [template]
// CHECK:STDOUT:   %struct.9: J = struct_value () [template]
// CHECK:STDOUT:   %.6: type = fn_type @.5 [template]
// CHECK:STDOUT:   %struct.10: <invalid> = struct_value () [template]
// CHECK:STDOUT:   %K: type = fn_type @K [template]
// CHECK:STDOUT:   %struct.11: K = struct_value () [template]
// CHECK:STDOUT:   %.7: type = struct_type {} [template]
// CHECK:STDOUT:   %.8: type = fn_type @.6 [template]
// CHECK:STDOUT:   %struct.12: <invalid> = struct_value () [template]
// CHECK:STDOUT:   %struct.13: {} = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .J = %J.decl
// CHECK:STDOUT:     .K = %K.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.1] {}
// CHECK:STDOUT:   %.decl.loc15: <invalid> = fn_decl @.1 [template = constants.%struct.2] {
// CHECK:STDOUT:     %.loc15_10.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc15_10.2: type = converted %.loc15_10.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %x.loc15_6.1: () = param x
// CHECK:STDOUT:     @.1.%x: () = bind_name x, %x.loc15_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: G = fn_decl @G [template = constants.%struct.3] {
// CHECK:STDOUT:     %.loc17_10.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc17_10.2: type = converted %.loc17_10.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %x.loc17_6.1: () = param x
// CHECK:STDOUT:     @G.%x: () = bind_name x, %x.loc17_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl.loc25: <invalid> = fn_decl @.2 [template = constants.%struct.4] {}
// CHECK:STDOUT:   %H.decl: H = fn_decl @H [template = constants.%struct.5] {
// CHECK:STDOUT:     %.loc27_10.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc27_10.2: type = converted %.loc27_10.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %x.loc27_6.1: () = param x
// CHECK:STDOUT:     @H.%x: () = bind_name x, %x.loc27_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl.loc35: <invalid> = fn_decl @.3 [template = constants.%struct.6] {
// CHECK:STDOUT:     %x.loc35_6.1: bool = param x
// CHECK:STDOUT:     @.3.%x: bool = bind_name x, %x.loc35_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: I = fn_decl @I [template = constants.%struct.7] {}
// CHECK:STDOUT:   %.decl.loc45: <invalid> = fn_decl @.4 [template = constants.%struct.8] {
// CHECK:STDOUT:     %.loc45_12.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc45_12.2: type = converted %.loc45_12.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     @.4.%return: ref () = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.decl: J = fn_decl @J [template = constants.%struct.9] {
// CHECK:STDOUT:     %.loc47_12.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc47_12.2: type = converted %.loc47_12.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     @J.%return: ref () = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl.loc55: <invalid> = fn_decl @.5 [template = constants.%struct.10] {}
// CHECK:STDOUT:   %K.decl: K = fn_decl @K [template = constants.%struct.11] {
// CHECK:STDOUT:     %.loc57_12.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc57_12.2: type = converted %.loc57_12.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     @K.%return: ref () = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl.loc64: <invalid> = fn_decl @.6 [template = constants.%struct.12] {
// CHECK:STDOUT:     %.loc64_12.1: {} = struct_literal ()
// CHECK:STDOUT:     %.loc64_12.2: type = converted %.loc64_12.1, constants.%.7 [template = constants.%.7]
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
// CHECK:STDOUT:   %struct: {} = struct_value () [template = constants.%struct.13]
// CHECK:STDOUT:   %.loc64_25: {} = converted %.loc64_24, %struct [template = constants.%struct.13]
// CHECK:STDOUT:   return %.loc64_25
// CHECK:STDOUT: }
// CHECK:STDOUT:
