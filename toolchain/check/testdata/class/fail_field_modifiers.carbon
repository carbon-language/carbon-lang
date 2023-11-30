// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {

  // CHECK:STDERR: fail_field_modifiers.carbon:[[@LINE+6]]:3: ERROR: Semantics TODO: `protected`.
  // CHECK:STDERR:   protected default var j: i32;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_field_modifiers.carbon:[[@LINE+3]]:13: ERROR: `default` not allowed on `var` declaration.
  // CHECK:STDERR:   protected default var j: i32;
  // CHECK:STDERR:             ^~~~~~~
  protected default var j: i32;

  // CHECK:STDERR: fail_field_modifiers.carbon:[[@LINE+6]]:3: ERROR: Semantics TODO: `private`.
  // CHECK:STDERR:   private final var k: i32;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_field_modifiers.carbon:[[@LINE+3]]:11: ERROR: `final` not allowed on `var` declaration.
  // CHECK:STDERR:   private final var k: i32;
  // CHECK:STDERR:           ^~~~~
  private final var k: i32;

  // CHECK:STDERR: fail_field_modifiers.carbon:[[@LINE+9]]:3: ERROR: Semantics TODO: `private`.
  // CHECK:STDERR:   private default let l: i32 = 0;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_field_modifiers.carbon:[[@LINE+6]]:11: ERROR: `default` not allowed on `let` declaration outside of an interface.
  // CHECK:STDERR:   private default let l: i32 = 0;
  // CHECK:STDERR:           ^~~~~~~
  // CHECK:STDERR: fail_field_modifiers.carbon:[[@LINE-24]]:1: Containing definition here.
  // CHECK:STDERR: class Class {
  // CHECK:STDERR: ^~~~~
  private default let l: i32 = 0;

  // CHECK:STDERR: fail_field_modifiers.carbon:[[@LINE+9]]:3: ERROR: Semantics TODO: `protected`.
  // CHECK:STDERR:   protected final let m: i32 = 1;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_field_modifiers.carbon:[[@LINE+6]]:13: ERROR: `final` not allowed on `let` declaration outside of an interface.
  // CHECK:STDERR:   protected final let m: i32 = 1;
  // CHECK:STDERR:             ^~~~~
  // CHECK:STDERR: fail_field_modifiers.carbon:[[@LINE-35]]:1: Containing definition here.
  // CHECK:STDERR: class Class {
  // CHECK:STDERR: ^~~~~
  protected final let m: i32 = 1;
}

// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc46: type = struct_type {.j: i32, .k: i32}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "fail_field_modifiers.carbon" {
// CHECK:STDOUT:   class_decl @Class, ()
// CHECK:STDOUT:   %Class: type = class_type @Class
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %.loc15_26.1: type = unbound_field_type Class, i32
// CHECK:STDOUT:   %.loc15_26.2: <unbound field of class Class> = field j, member0
// CHECK:STDOUT:   %j: <unbound field of class Class> = bind_name j, %.loc15_26.2
// CHECK:STDOUT:   %.loc23_22.1: type = unbound_field_type Class, i32
// CHECK:STDOUT:   %.loc23_22.2: <unbound field of class Class> = field k, member1
// CHECK:STDOUT:   %k: <unbound field of class Class> = bind_name k, %.loc23_22.2
// CHECK:STDOUT:   %.loc34: i32 = int_literal 0
// CHECK:STDOUT:   %l: i32 = bind_name l, %.loc34
// CHECK:STDOUT:   %.loc45: i32 = int_literal 1
// CHECK:STDOUT:   %m: i32 = bind_name m, %.loc45
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .j = %j
// CHECK:STDOUT:   .k = %k
// CHECK:STDOUT: }
