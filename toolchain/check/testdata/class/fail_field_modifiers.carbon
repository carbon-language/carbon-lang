// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {

  // CHECK:STDERR: fail_field_modifiers.carbon:[[@LINE+3]]:3: ERROR: `default` not allowed on `var` declaration.
  // CHECK:STDERR:   default var j: i32;
  // CHECK:STDERR:   ^~~~~~~
  default var j: i32;

  // CHECK:STDERR: fail_field_modifiers.carbon:[[@LINE+3]]:3: ERROR: `final` not allowed on `var` declaration.
  // CHECK:STDERR:   final var k: i32;
  // CHECK:STDERR:   ^~~~~
  final var k: i32;

  // CHECK:STDERR: fail_field_modifiers.carbon:[[@LINE+3]]:3: ERROR: `default` not allowed on `let` declaration outside of an interface.
  // CHECK:STDERR:   default let l: i32 = 0;
  // CHECK:STDERR:   ^~~~~~~
  default let l: i32 = 0;

  // CHECK:STDERR: fail_field_modifiers.carbon:[[@LINE+3]]:3: ERROR: `final` not allowed on `let` declaration outside of an interface.
  // CHECK:STDERR:   final let m: i32 = 1;
  // CHECK:STDERR:   ^~~~~
  final let m: i32 = 1;
}

// CHECK:STDOUT: --- fail_field_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc28: type = struct_type {.j: i32, .k: i32}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.Class = %Class.decl}
// CHECK:STDOUT:   %Class.decl = class_decl @Class, ()
// CHECK:STDOUT:   %Class: type = class_type @Class
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %.loc12_16.1: type = unbound_element_type Class, i32
// CHECK:STDOUT:   %.loc12_16.2: <unbound element of class Class> = field_decl j, element0
// CHECK:STDOUT:   %j: <unbound element of class Class> = bind_name j, %.loc12_16.2
// CHECK:STDOUT:   %.loc17_14.1: type = unbound_element_type Class, i32
// CHECK:STDOUT:   %.loc17_14.2: <unbound element of class Class> = field_decl k, element1
// CHECK:STDOUT:   %k: <unbound element of class Class> = bind_name k, %.loc17_14.2
// CHECK:STDOUT:   %.loc22: i32 = int_literal 0
// CHECK:STDOUT:   %l: i32 = bind_name l, %.loc22
// CHECK:STDOUT:   %.loc27: i32 = int_literal 1
// CHECK:STDOUT:   %m: i32 = bind_name m, %.loc27
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .j = %j
// CHECK:STDOUT:   .k = %k
// CHECK:STDOUT: }
// CHECK:STDOUT:
