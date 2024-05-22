// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {

  // CHECK:STDERR: fail_field_modifiers.carbon:[[@LINE+4]]:3: ERROR: `default` not allowed on `var` declaration.
  // CHECK:STDERR:   default var j: i32;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  default var j: i32;

  // CHECK:STDERR: fail_field_modifiers.carbon:[[@LINE+4]]:3: ERROR: `final` not allowed on `var` declaration.
  // CHECK:STDERR:   final var k: i32;
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR:
  final var k: i32;

  // CHECK:STDERR: fail_field_modifiers.carbon:[[@LINE+4]]:3: ERROR: `default` not allowed on `let` declaration outside of an interface.
  // CHECK:STDERR:   default let l: i32 = 0;
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  default let l: i32 = 0;

  // CHECK:STDERR: fail_field_modifiers.carbon:[[@LINE+3]]:3: ERROR: `final` not allowed on `let` declaration outside of an interface.
  // CHECK:STDERR:   final let m: i32 = 1;
  // CHECK:STDERR:   ^~~~~
  final let m: i32 = 1;
}

// CHECK:STDOUT: --- fail_field_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Int32 = struct_value () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type Class, i32 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.5: type = struct_type {.j: i32, .k: i32} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct]
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct]
// CHECK:STDOUT:   %import_ref.3: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct]
// CHECK:STDOUT:   %import_ref.4: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %int.make_type_32.loc13: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc13_18.1: type = value_of_initializer %int.make_type_32.loc13 [template = i32]
// CHECK:STDOUT:   %.loc13_18.2: type = converted %int.make_type_32.loc13, %.loc13_18.1 [template = i32]
// CHECK:STDOUT:   %.loc13_16: <unbound element of class Class> = field_decl j, element0 [template]
// CHECK:STDOUT:   %int.make_type_32.loc19: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc19_16.1: type = value_of_initializer %int.make_type_32.loc19 [template = i32]
// CHECK:STDOUT:   %.loc19_16.2: type = converted %int.make_type_32.loc19, %.loc19_16.1 [template = i32]
// CHECK:STDOUT:   %.loc19_14: <unbound element of class Class> = field_decl k, element1 [template]
// CHECK:STDOUT:   %int.make_type_32.loc25: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc25_18.1: type = value_of_initializer %int.make_type_32.loc25 [template = i32]
// CHECK:STDOUT:   %.loc25_18.2: type = converted %int.make_type_32.loc25, %.loc25_18.1 [template = i32]
// CHECK:STDOUT:   %.loc25_24: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   %l: i32 = bind_name l, %.loc25_24
// CHECK:STDOUT:   %int.make_type_32.loc30: init type = call constants.%struct() [template = i32]
// CHECK:STDOUT:   %.loc30_16.1: type = value_of_initializer %int.make_type_32.loc30 [template = i32]
// CHECK:STDOUT:   %.loc30_16.2: type = converted %int.make_type_32.loc30, %.loc30_16.1 [template = i32]
// CHECK:STDOUT:   %.loc30_22: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %m: i32 = bind_name m, %.loc30_22
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .j = %.loc13_16
// CHECK:STDOUT:   .k = %.loc19_14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
