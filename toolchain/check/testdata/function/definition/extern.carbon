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
// CHECK:STDERR: fail_def_for_extern_decl.carbon:[[@LINE+7]]:1: ERROR: Redeclaring `extern` function `F` as non-`extern`.
// CHECK:STDERR: fn F() {}
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR: fail_def_for_extern_decl.carbon:[[@LINE-4]]:1: Previously declared `extern` here.
// CHECK:STDERR: extern fn F();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR:
fn F() {}

// --- fail_extern_diag_suppressed.carbon

library "extern_diag_suppressed" api;

extern fn F();
// CHECK:STDERR: fail_extern_diag_suppressed.carbon:[[@LINE+7]]:1: ERROR: Redundant redeclaration of function F.
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
// CHECK:STDERR: fail_extern_decl_after_def.carbon:[[@LINE+6]]:1: ERROR: Redundant redeclaration of function F.
// CHECK:STDERR: extern fn F();
// CHECK:STDERR: ^~~~~~~~~~~~~~
// CHECK:STDERR: fail_extern_decl_after_def.carbon:[[@LINE-4]]:1: Previously declared here.
// CHECK:STDERR: fn F() {}
// CHECK:STDERR: ^~~~~~~~
extern fn F();

// CHECK:STDOUT: --- fail_extern_def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_def_for_extern_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.loc4: <function> = fn_decl @F [template] {}
// CHECK:STDOUT:   %F.loc12: <function> = fn_decl @F [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_extern_diag_suppressed.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.loc4: <function> = fn_decl @F [template] {}
// CHECK:STDOUT:   %F.loc12: <function> = fn_decl @F [template] {}
// CHECK:STDOUT:   %F.loc13: <function> = fn_decl @F [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_extern_decl_after_def.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F.loc4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.loc4: <function> = fn_decl @F [template] {}
// CHECK:STDOUT:   %F.loc11: <function> = fn_decl @F [template] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
