// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: ERROR: `protected` not allowed on `let` declaration at file scope, `protected` is only allowed on class members.
// CHECK:STDERR: protected let b: i32 = 1;
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
protected let b: i32 = 1;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: ERROR: `default` not allowed on `let` declaration outside of an interface.
// CHECK:STDERR: default let c: i32 = 1;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
default let c: i32 = 1;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: ERROR: `final` not allowed on `let` declaration outside of an interface.
// CHECK:STDERR: final let d: i32 = 1;
// CHECK:STDERR: ^~~~~
// CHECK:STDERR:
final let d: i32 = 1;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: ERROR: `virtual` not allowed on `let` declaration.
// CHECK:STDERR: virtual let e: i32 = 1;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
virtual let e: i32 = 1;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+11]]:1: ERROR: `default` not allowed on `let` declaration outside of an interface.
// CHECK:STDERR: default final let f: i32 = 1;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:9: ERROR: `final` not allowed on declaration with `default`.
// CHECK:STDERR: default final let f: i32 = 1;
// CHECK:STDERR:         ^~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: `default` previously appeared here.
// CHECK:STDERR: default final let f: i32 = 1;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
default final let f: i32 = 1;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+11]]:1: ERROR: `default` not allowed on `let` declaration outside of an interface.
// CHECK:STDERR: default default let g: i32 = 1;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:9: ERROR: `default` repeated on declaration.
// CHECK:STDERR: default default let g: i32 = 1;
// CHECK:STDERR:         ^~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: `default` previously appeared here.
// CHECK:STDERR: default default let g: i32 = 1;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
default default let g: i32 = 1;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+11]]:1: ERROR: `protected` not allowed on `let` declaration at file scope, `protected` is only allowed on class members.
// CHECK:STDERR: protected private let h: i32 = 1;
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:11: ERROR: `private` not allowed on declaration with `protected`.
// CHECK:STDERR: protected private let h: i32 = 1;
// CHECK:STDERR:           ^~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: `protected` previously appeared here.
// CHECK:STDERR: protected private let h: i32 = 1;
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
protected private let h: i32 = 1;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+10]]:1: ERROR: `protected` not allowed on `let` declaration at file scope, `protected` is only allowed on class members.
// CHECK:STDERR: protected protected let i: i32 = 1;
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+6]]:11: ERROR: `protected` repeated on declaration.
// CHECK:STDERR: protected protected let i: i32 = 1;
// CHECK:STDERR:           ^~~~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:1: `protected` previously appeared here.
// CHECK:STDERR: protected protected let i: i32 = 1;
// CHECK:STDERR: ^~~~~~~~~
protected protected let i: i32 = 1;

// CHECK:STDOUT: --- fail_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc11: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %b: i32 = bind_name b, %.loc11
// CHECK:STDOUT:   %.loc17: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %c: i32 = bind_name c, %.loc17
// CHECK:STDOUT:   %.loc23: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %d: i32 = bind_name d, %.loc23
// CHECK:STDOUT:   %.loc29: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %e: i32 = bind_name e, %.loc29
// CHECK:STDOUT:   %.loc42: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %f: i32 = bind_name f, %.loc42
// CHECK:STDOUT:   %.loc55: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %g: i32 = bind_name g, %.loc55
// CHECK:STDOUT:   %.loc68: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %h: i32 = bind_name h, %.loc68
// CHECK:STDOUT:   %.loc80: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %i: i32 = bind_name i, %.loc80
// CHECK:STDOUT: }
// CHECK:STDOUT:
