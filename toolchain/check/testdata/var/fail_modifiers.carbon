// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: ERROR: Semantics TODO: `private`.
// CHECK:STDERR: private var a: i32;
// CHECK:STDERR: ^
private var a: i32;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: ERROR: Semantics TODO: `protected`.
// CHECK:STDERR: protected var b: i32;
// CHECK:STDERR: ^
protected var b: i32;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+9]]:1: ERROR: Semantics TODO: `private`.
// CHECK:STDERR: private protected var c: i32;
// CHECK:STDERR: ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:9: ERROR: Modifier not allowed on the same declaration.
// CHECK:STDERR: private protected var c: i32;
// CHECK:STDERR:         ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: With this modifier.
// CHECK:STDERR: private protected var c: i32;
// CHECK:STDERR: ^
private protected var c: i32;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+9]]:1: ERROR: Semantics TODO: `protected`.
// CHECK:STDERR: protected protected var d: i32;
// CHECK:STDERR: ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:11: ERROR: Modifier repeated on the same declaration.
// CHECK:STDERR: protected protected var d: i32;
// CHECK:STDERR:           ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: Previously appeared here.
// CHECK:STDERR: protected protected var d: i32;
// CHECK:STDERR: ^
protected protected var d: i32;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:1: ERROR: Modifier not allowed.
// CHECK:STDERR: abstract var e: i32;
// CHECK:STDERR: ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:10: On this declaration.
// CHECK:STDERR: abstract var e: i32;
// CHECK:STDERR:          ^
abstract var e: i32;

// CHECK:STDOUT: file "fail_modifiers.carbon" {
// CHECK:STDOUT:   %a.var: ref i32 = var a
// CHECK:STDOUT:   %a: ref i32 = bind_name a, %a.var
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %c.var: ref i32 = var c
// CHECK:STDOUT:   %c: ref i32 = bind_name c, %c.var
// CHECK:STDOUT:   %d.var: ref i32 = var d
// CHECK:STDOUT:   %d: ref i32 = bind_name d, %d.var
// CHECK:STDOUT:   %e.var: ref i32 = var e
// CHECK:STDOUT:   %e: ref i32 = bind_name e, %e.var
// CHECK:STDOUT: }
