// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: ERROR: Semantics TODO: `private`.
// CHECK:STDERR: private let a: i32 = 1;
// CHECK:STDERR: ^
private let a: i32 = 1;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: ERROR: Semantics TODO: `protected`.
// CHECK:STDERR: protected let b: i32 = 1;
// CHECK:STDERR: ^
protected let b: i32 = 1;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: ERROR: Semantics TODO: `default`.
// CHECK:STDERR: default let c: i32 = 1;
// CHECK:STDERR: ^
default let c: i32 = 1;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: ERROR: Semantics TODO: `final`.
// CHECK:STDERR: final let d: i32 = 1;
// CHECK:STDERR: ^
final let d: i32 = 1;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: ERROR: `virtual` not allowed on `let` declaration.
// CHECK:STDERR: virtual let e: i32 = 1;
// CHECK:STDERR: ^
virtual let e: i32 = 1;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+9]]:1: ERROR: Semantics TODO: `default`.
// CHECK:STDERR: default final let f: i32 = 1;
// CHECK:STDERR: ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:9: ERROR: `final` not allowed on declaration with `default`.
// CHECK:STDERR: default final let f: i32 = 1;
// CHECK:STDERR:         ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: `default` previously appeared here.
// CHECK:STDERR: default final let f: i32 = 1;
// CHECK:STDERR: ^
default final let f: i32 = 1;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+9]]:1: ERROR: Semantics TODO: `default`.
// CHECK:STDERR: default default let g: i32 = 1;
// CHECK:STDERR: ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:9: ERROR: `default` repeated on declaration.
// CHECK:STDERR: default default let g: i32 = 1;
// CHECK:STDERR:         ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: `default` previously appeared here.
// CHECK:STDERR: default default let g: i32 = 1;
// CHECK:STDERR: ^
default default let g: i32 = 1;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+9]]:1: ERROR: Semantics TODO: `protected`.
// CHECK:STDERR: protected private let h: i32 = 1;
// CHECK:STDERR: ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:11: ERROR: `private` not allowed on declaration with `protected`.
// CHECK:STDERR: protected private let h: i32 = 1;
// CHECK:STDERR:           ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: `protected` previously appeared here.
// CHECK:STDERR: protected private let h: i32 = 1;
// CHECK:STDERR: ^
protected private let h: i32 = 1;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+9]]:1: ERROR: Semantics TODO: `protected`.
// CHECK:STDERR: protected protected let i: i32 = 1;
// CHECK:STDERR: ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:11: ERROR: `protected` repeated on declaration.
// CHECK:STDERR: protected protected let i: i32 = 1;
// CHECK:STDERR:           ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: `protected` previously appeared here.
// CHECK:STDERR: protected protected let i: i32 = 1;
// CHECK:STDERR: ^
protected protected let i: i32 = 1;

// CHECK:STDOUT: file "fail_modifiers.carbon" {
// CHECK:STDOUT:   %.loc10: i32 = int_literal 1
// CHECK:STDOUT:   %a: i32 = bind_name a, %.loc10
// CHECK:STDOUT:   %.loc15: i32 = int_literal 1
// CHECK:STDOUT:   %b: i32 = bind_name b, %.loc15
// CHECK:STDOUT:   %.loc20: i32 = int_literal 1
// CHECK:STDOUT:   %c: i32 = bind_name c, %.loc20
// CHECK:STDOUT:   %.loc25: i32 = int_literal 1
// CHECK:STDOUT:   %d: i32 = bind_name d, %.loc25
// CHECK:STDOUT:   %.loc30: i32 = int_literal 1
// CHECK:STDOUT:   %e: i32 = bind_name e, %.loc30
// CHECK:STDOUT:   %.loc41: i32 = int_literal 1
// CHECK:STDOUT:   %f: i32 = bind_name f, %.loc41
// CHECK:STDOUT:   %.loc52: i32 = int_literal 1
// CHECK:STDOUT:   %g: i32 = bind_name g, %.loc52
// CHECK:STDOUT:   %.loc63: i32 = int_literal 1
// CHECK:STDOUT:   %h: i32 = bind_name h, %.loc63
// CHECK:STDOUT:   %.loc74: i32 = int_literal 1
// CHECK:STDOUT:   %i: i32 = bind_name i, %.loc74
// CHECK:STDOUT: }
