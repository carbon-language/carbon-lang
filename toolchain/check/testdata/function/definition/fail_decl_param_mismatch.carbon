// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F();
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE+7]]:1: ERROR: Function redeclaration differs because of parameter count of 1.
// CHECK:STDERR: fn F(x: i32) {}
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE-4]]:1: Previously declared with parameter count of 0.
// CHECK:STDERR: fn F();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
fn F(x: i32) {}

fn G(x: i32);
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE+7]]:1: ERROR: Function redeclaration differs because of parameter count of 0.
// CHECK:STDERR: fn G() {}
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE-4]]:1: Previously declared with parameter count of 1.
// CHECK:STDERR: fn G(x: i32);
// CHECK:STDERR: ^~~~~~~~~~~~~
// CHECK:STDERR:
fn G() {}

fn H(x: i32);
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE+7]]:6: ERROR: Function redeclaration differs at parameter 1.
// CHECK:STDERR: fn H(x: bool) {}
// CHECK:STDERR:      ^
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE-4]]:6: Previous declaration's corresponding parameter here.
// CHECK:STDERR: fn H(x: i32);
// CHECK:STDERR:      ^
// CHECK:STDERR:
fn H(x: bool) {}

fn I();
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE+7]]:1: ERROR: Function redeclaration differs because return type is `i32`.
// CHECK:STDERR: fn I() -> i32 { return 0; }
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE-4]]:1: Previously declared with no return type.
// CHECK:STDERR: fn I();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
fn I() -> i32 { return 0; }

fn J() -> i32;
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE+7]]:1: ERROR: Function redeclaration differs because no return type is provided.
// CHECK:STDERR: fn J() {}
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE-4]]:1: Previously declared with return type `i32`.
// CHECK:STDERR: fn J() -> i32;
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
fn J() {}

fn K() -> i32;
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE+6]]:1: ERROR: Function redeclaration differs because return type is `bool`.
// CHECK:STDERR: fn K() -> bool { return false; }
// CHECK:STDERR: ^~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE-4]]:1: Previously declared with return type `i32`.
// CHECK:STDERR: fn K() -> i32;
// CHECK:STDERR: ^~~~~~~~~~~~~~
fn K() -> bool { return false; }

// CHECK:STDOUT: --- fail_decl_param_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.2: bool = bool_literal false [template]
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
// CHECK:STDOUT:   %.loc15: <function> = fn_decl @.1 [template] {
// CHECK:STDOUT:     %x.loc15_6.1: i32 = param x
// CHECK:STDOUT:     @.1.%x: i32 = bind_name x, %x.loc15_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G: <function> = fn_decl @G [template] {
// CHECK:STDOUT:     %x.loc17_6.1: i32 = param x
// CHECK:STDOUT:     @G.%x: i32 = bind_name x, %x.loc17_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc25: <function> = fn_decl @.2 [template] {}
// CHECK:STDOUT:   %H: <function> = fn_decl @H [template] {
// CHECK:STDOUT:     %x.loc27_6.1: i32 = param x
// CHECK:STDOUT:     @H.%x: i32 = bind_name x, %x.loc27_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc35: <function> = fn_decl @.3 [template] {
// CHECK:STDOUT:     %x.loc35_6.1: bool = param x
// CHECK:STDOUT:     @.3.%x: bool = bind_name x, %x.loc35_6.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I: <function> = fn_decl @I [template] {}
// CHECK:STDOUT:   %.loc45: <function> = fn_decl @.4 [template] {
// CHECK:STDOUT:     %return.var.loc45: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J: <function> = fn_decl @J [template] {
// CHECK:STDOUT:     %return.var.loc47: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc55: <function> = fn_decl @.5 [template] {}
// CHECK:STDOUT:   %K: <function> = fn_decl @K [template] {
// CHECK:STDOUT:     %return.var.loc57: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc64: <function> = fn_decl @.6 [template] {
// CHECK:STDOUT:     %return.var.loc64: ref bool = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1(%x: i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%x: i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H(%x: i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.3(%x: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @I();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.4() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc45: i32 = int_literal 0 [template = constants.%.1]
// CHECK:STDOUT:   return %.loc45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @J() -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.5() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @K() -> i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.6() -> bool {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc64: bool = bool_literal false [template = constants.%.2]
// CHECK:STDOUT:   return %.loc64
// CHECK:STDOUT: }
// CHECK:STDOUT:
