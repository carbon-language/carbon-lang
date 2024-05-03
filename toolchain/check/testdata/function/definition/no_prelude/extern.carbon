// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- fail_extern_def.carbon

library "extern_def" api;

// CHECK:STDERR: fail_extern_def.carbon:[[@LINE+4]]:1: ERROR: `extern` not allowed on `fn` declaration that provides a definition.
// CHECK:STDERR: extern fn F() {}
// CHECK:STDERR: ^~~~~~
// CHECK:STDERR:
extern fn F() {}

// --- fail_def_for_extern_decl.carbon

library "def_for_extern_decl" api;

extern fn F();
// CHECK:STDERR: fail_def_for_extern_decl.carbon:[[@LINE+7]]:1: ERROR: Redeclarations of `fn F` in the same library must match use of `extern`.
// CHECK:STDERR: fn F() {}
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_def_for_extern_decl.carbon:[[@LINE-4]]:1: Previously declared here.
// CHECK:STDERR: extern fn F();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
fn F() {}

// --- fail_extern_diag_suppressed.carbon

library "extern_diag_suppressed" api;

extern fn F();
// CHECK:STDERR: fail_extern_diag_suppressed.carbon:[[@LINE+7]]:1: ERROR: Redeclaration of `fn F` is redundant.
// CHECK:STDERR: fn F();
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR: fail_extern_diag_suppressed.carbon:[[@LINE-4]]:1: Previously declared here.
// CHECK:STDERR: extern fn F();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
fn F();
fn F() {}

// --- fail_extern_decl_after_def.carbon

library "extern_decl_after_def" api;

fn F() {}
// CHECK:STDERR: fail_extern_decl_after_def.carbon:[[@LINE+6]]:1: ERROR: Redeclaration of `fn F` is redundant.
// CHECK:STDERR: extern fn F();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_extern_decl_after_def.carbon:[[@LINE-4]]:1: Previously declared here.
// CHECK:STDERR: fn F() {}
// CHECK:STDERR: ^~~~~~~~
extern fn F();

// CHECK:STDOUT: --- fail_extern_def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: F = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_def_for_extern_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: F = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F.decl.loc12
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc4: F = fn_decl @F [template = constants.%struct] {}
// CHECK:STDOUT:   %F.decl.loc12: F = fn_decl @F [template = constants.%struct] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_extern_diag_suppressed.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: F = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F.decl.loc12
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc4: F = fn_decl @F [template = constants.%struct] {}
// CHECK:STDOUT:   %F.decl.loc12: F = fn_decl @F [template = constants.%struct] {}
// CHECK:STDOUT:   %F.decl.loc13: F = fn_decl @F [template = constants.%struct] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_extern_decl_after_def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: F = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F.decl.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc4: F = fn_decl @F [template = constants.%struct] {}
// CHECK:STDOUT:   %F.decl.loc11: F = fn_decl @F [template = constants.%struct] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
