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

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:1: ERROR: Modifier not allowed.
// CHECK:STDERR: virtual let e: i32 = 1;
// CHECK:STDERR: ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:9: On this declaration.
// CHECK:STDERR: virtual let e: i32 = 1;
// CHECK:STDERR:         ^
virtual let e: i32 = 1;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+9]]:1: ERROR: Semantics TODO: `default`.
// CHECK:STDERR: default final let f: i32 = 1;
// CHECK:STDERR: ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:9: ERROR: Modifier not allowed on the same declaration.
// CHECK:STDERR: default final let f: i32 = 1;
// CHECK:STDERR:         ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: With this modifier.
// CHECK:STDERR: default final let f: i32 = 1;
// CHECK:STDERR: ^
default final let f: i32 = 1;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+9]]:1: ERROR: Semantics TODO: `default`.
// CHECK:STDERR: default default let g: i32 = 1;
// CHECK:STDERR: ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:9: ERROR: Modifier repeated on the same declaration.
// CHECK:STDERR: default default let g: i32 = 1;
// CHECK:STDERR:         ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: Previously appeared here.
// CHECK:STDERR: default default let g: i32 = 1;
// CHECK:STDERR: ^
default default let g: i32 = 1;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+9]]:1: ERROR: Semantics TODO: `protected`.
// CHECK:STDERR: protected private let h: i32 = 1;
// CHECK:STDERR: ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:11: ERROR: Modifier not allowed on the same declaration.
// CHECK:STDERR: protected private let h: i32 = 1;
// CHECK:STDERR:           ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: With this modifier.
// CHECK:STDERR: protected private let h: i32 = 1;
// CHECK:STDERR: ^
protected private let h: i32 = 1;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+9]]:1: ERROR: Semantics TODO: `protected`.
// CHECK:STDERR: protected protected let i: i32 = 1;
// CHECK:STDERR: ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:11: ERROR: Modifier repeated on the same declaration.
// CHECK:STDERR: protected protected let i: i32 = 1;
// CHECK:STDERR:           ^
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: Previously appeared here.
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
// CHECK:STDOUT:   %.loc33: i32 = int_literal 1
// CHECK:STDOUT:   %e: i32 = bind_name e, %.loc33
// CHECK:STDOUT:   %.loc44: i32 = int_literal 1
// CHECK:STDOUT:   %f: i32 = bind_name f, %.loc44
// CHECK:STDOUT:   %.loc55: i32 = int_literal 1
// CHECK:STDOUT:   %g: i32 = bind_name g, %.loc55
// CHECK:STDOUT:   %.loc66: i32 = int_literal 1
// CHECK:STDOUT:   %h: i32 = bind_name h, %.loc66
// CHECK:STDOUT:   %.loc77: i32 = int_literal 1
// CHECK:STDOUT:   %i: i32 = bind_name i, %.loc77
// CHECK:STDOUT: }
