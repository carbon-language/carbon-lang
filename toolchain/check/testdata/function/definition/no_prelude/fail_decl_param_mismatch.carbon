// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/definition/no_prelude/fail_decl_param_mismatch.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/definition/no_prelude/fail_decl_param_mismatch.carbon

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
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE+4]]:9: ERROR: Name `Core` not found.
// CHECK:STDERR: fn H(x: bool) {}
// CHECK:STDERR:         ^~~~
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
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F: <type of F> = struct_value () [template]
// CHECK:STDOUT:   %.type.1: type = fn_type @.1 [template]
// CHECK:STDOUT:   %.2: <type of <invalid>> = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: <type of G> = struct_value () [template]
// CHECK:STDOUT:   %.type.2: type = fn_type @.2 [template]
// CHECK:STDOUT:   %.3: <type of <invalid>> = struct_value () [template]
// CHECK:STDOUT:   %H.type: type = fn_type @H [template]
// CHECK:STDOUT:   %H: <type of H> = struct_value () [template]
// CHECK:STDOUT:   %.type.3: type = fn_type @.3 [template]
// CHECK:STDOUT:   %.4: <type of <invalid>> = struct_value () [template]
// CHECK:STDOUT:   %I.type: type = fn_type @I [template]
// CHECK:STDOUT:   %I: <type of I> = struct_value () [template]
// CHECK:STDOUT:   %.type.4: type = fn_type @.4 [template]
// CHECK:STDOUT:   %.5: <type of <invalid>> = struct_value () [template]
// CHECK:STDOUT:   %tuple: () = tuple_value () [template]
// CHECK:STDOUT:   %J.type: type = fn_type @J [template]
// CHECK:STDOUT:   %J: <type of J> = struct_value () [template]
// CHECK:STDOUT:   %.type.5: type = fn_type @.5 [template]
// CHECK:STDOUT:   %.6: <type of <invalid>> = struct_value () [template]
// CHECK:STDOUT:   %K.type: type = fn_type @K [template]
// CHECK:STDOUT:   %K: <type of K> = struct_value () [template]
// CHECK:STDOUT:   %.7: type = struct_type {} [template]
// CHECK:STDOUT:   %.type.6: type = fn_type @.6 [template]
// CHECK:STDOUT:   %.8: <type of <invalid>> = struct_value () [template]
// CHECK:STDOUT:   %struct: {} = struct_value () [template]
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
// CHECK:STDOUT:   %F.decl: <type of F> = fn_decl @F [template = constants.%F] {}
// CHECK:STDOUT:   %.decl.loc19: <type of <invalid>> = fn_decl @.1 [template = constants.%.2] {
// CHECK:STDOUT:     %.loc19_10.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc19_10.2: type = converted %.loc19_10.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %x.loc19_6.1: () = param x
// CHECK:STDOUT:     @.1.%x: () = bind_name x, %x.loc19_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: <type of G> = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %.loc21_10.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc21_10.2: type = converted %.loc21_10.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %x.loc21_6.1: () = param x
// CHECK:STDOUT:     @G.%x: () = bind_name x, %x.loc21_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl.loc29: <type of <invalid>> = fn_decl @.2 [template = constants.%.3] {}
// CHECK:STDOUT:   %H.decl: <type of H> = fn_decl @H [template = constants.%H] {
// CHECK:STDOUT:     %.loc31_10.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc31_10.2: type = converted %.loc31_10.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %x.loc31_6.1: () = param x
// CHECK:STDOUT:     @H.%x: () = bind_name x, %x.loc31_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl.loc36: <type of <invalid>> = fn_decl @.3 [template = constants.%.4] {
// CHECK:STDOUT:     %x.loc36_6.1: <error> = param x
// CHECK:STDOUT:     @.3.%x: <error> = bind_name x, %x.loc36_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl: <type of I> = fn_decl @I [template = constants.%I] {}
// CHECK:STDOUT:   %.decl.loc46: <type of <invalid>> = fn_decl @.4 [template = constants.%.5] {
// CHECK:STDOUT:     %.loc46_12.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc46_12.2: type = converted %.loc46_12.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     @.4.%return: ref () = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.decl: <type of J> = fn_decl @J [template = constants.%J] {
// CHECK:STDOUT:     %.loc48_12.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc48_12.2: type = converted %.loc48_12.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     @J.%return: ref () = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl.loc56: <type of <invalid>> = fn_decl @.5 [template = constants.%.6] {}
// CHECK:STDOUT:   %K.decl: <type of K> = fn_decl @K [template = constants.%K] {
// CHECK:STDOUT:     %.loc58_12.1: () = tuple_literal ()
// CHECK:STDOUT:     %.loc58_12.2: type = converted %.loc58_12.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     @K.%return: ref () = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl.loc65: <type of <invalid>> = fn_decl @.6 [template = constants.%.8] {
// CHECK:STDOUT:     %.loc65_12.1: {} = struct_literal ()
// CHECK:STDOUT:     %.loc65_12.2: type = converted %.loc65_12.1, constants.%.7 [template = constants.%.7]
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
// CHECK:STDOUT: fn @.3(%x: <error>) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @I();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.4() -> () {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc46_24: () = tuple_literal ()
// CHECK:STDOUT:   %tuple: () = tuple_value () [template = constants.%tuple]
// CHECK:STDOUT:   %.loc46_25: () = converted %.loc46_24, %tuple [template = constants.%tuple]
// CHECK:STDOUT:   return %.loc46_25
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
// CHECK:STDOUT:   %.loc65_24: {} = struct_literal ()
// CHECK:STDOUT:   %struct: {} = struct_value () [template = constants.%struct]
// CHECK:STDOUT:   %.loc65_25: {} = converted %.loc65_24, %struct [template = constants.%struct]
// CHECK:STDOUT:   return %.loc65_25
// CHECK:STDOUT: }
// CHECK:STDOUT:
