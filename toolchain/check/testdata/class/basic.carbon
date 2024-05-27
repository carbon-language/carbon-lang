// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class {
  fn F(n: i32) -> i32 {
    return n;
  }

  fn G(n: i32) -> i32;

  var k: i32;
}

fn Class.G(n: i32) -> i32 {
  return n;
}

fn Run() -> i32 {
  return Class.F(4);
}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Int32 = struct_value () [template]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %struct.2: F = struct_value () [template]
// CHECK:STDOUT:   %G: type = fn_type @G [template]
// CHECK:STDOUT:   %struct.3: G = struct_value () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type Class, i32 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.k: i32} [template]
// CHECK:STDOUT:   %Run: type = fn_type @Run [template]
// CHECK:STDOUT:   %struct.4: Run = struct_value () [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 4 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:     .Run = %Run.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.3: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.4: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.5: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.6: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.7: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %G.decl: G = fn_decl @G [template = constants.%struct.3] {
// CHECK:STDOUT:     %int.make_type_32.loc17_15: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc17_15.1: type = value_of_initializer %int.make_type_32.loc17_15 [template = i32]
// CHECK:STDOUT:     %.loc17_15.2: type = converted %int.make_type_32.loc17_15, %.loc17_15.1 [template = i32]
// CHECK:STDOUT:     %n.loc17_12.1: i32 = param n
// CHECK:STDOUT:     @G.%n: i32 = bind_name n, %n.loc17_12.1
// CHECK:STDOUT:     %int.make_type_32.loc17_23: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc17_23.1: type = value_of_initializer %int.make_type_32.loc17_23 [template = i32]
// CHECK:STDOUT:     %.loc17_23.2: type = converted %int.make_type_32.loc17_23, %.loc17_23.1 [template = i32]
// CHECK:STDOUT:     @G.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.8: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %Run.decl: Run = fn_decl @Run [template = constants.%struct.4] {
// CHECK:STDOUT:     %int.make_type_32.loc21: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc21_13.1: type = value_of_initializer %int.make_type_32.loc21 [template = i32]
// CHECK:STDOUT:     %.loc21_13.2: type = converted %int.make_type_32.loc21, %.loc21_13.1 [template = i32]
// CHECK:STDOUT:     @Run.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.2] {
// CHECK:STDOUT:     %int.make_type_32.loc8_11: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc8_11.1: type = value_of_initializer %int.make_type_32.loc8_11 [template = i32]
// CHECK:STDOUT:     %.loc8_11.2: type = converted %int.make_type_32.loc8_11, %.loc8_11.1 [template = i32]
// CHECK:STDOUT:     %n.loc8_8.1: i32 = param n
// CHECK:STDOUT:     %n.loc8_8.2: i32 = bind_name n, %n.loc8_8.1
// CHECK:STDOUT:     %int.make_type_32.loc8_19: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc8_19.1: type = value_of_initializer %int.make_type_32.loc8_19 [template = i32]
// CHECK:STDOUT:     %.loc8_19.2: type = converted %int.make_type_32.loc8_19, %.loc8_19.1 [template = i32]
// CHECK:STDOUT:     %return.var.loc8: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: G = fn_decl @G [template = constants.%struct.3] {
// CHECK:STDOUT:     %int.make_type_32.loc12_11: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc12_11.1: type = value_of_initializer %int.make_type_32.loc12_11 [template = i32]
// CHECK:STDOUT:     %.loc12_11.2: type = converted %int.make_type_32.loc12_11, %.loc12_11.1 [template = i32]
// CHECK:STDOUT:     %n.loc12_8.1: i32 = param n
// CHECK:STDOUT:     %n.loc12_8.2: i32 = bind_name n, %n.loc12_8.1
// CHECK:STDOUT:     %int.make_type_32.loc12_19: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc12_19.1: type = value_of_initializer %int.make_type_32.loc12_19 [template = i32]
// CHECK:STDOUT:     %.loc12_19.2: type = converted %int.make_type_32.loc12_19, %.loc12_19.1 [template = i32]
// CHECK:STDOUT:     %return.var.loc12: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int.make_type_32.loc14: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc14_10.1: type = value_of_initializer %int.make_type_32.loc14 [template = i32]
// CHECK:STDOUT:   %.loc14_10.2: type = converted %int.make_type_32.loc14, %.loc14_10.1 [template = i32]
// CHECK:STDOUT:   %.loc14_8: <unbound element of class Class> = field_decl k, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   .k = %.loc14_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(@Class.%n.loc8_8.2: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: i32 = name_ref n, @Class.%n.loc8_8.2
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%n: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: i32 = name_ref n, %n
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Class.ref: type = name_ref Class, file.%Class.decl [template = constants.%Class]
// CHECK:STDOUT:   %F.ref: F = name_ref F, @Class.%F.decl [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc22_18: i32 = int_literal 4 [template = constants.%.4]
// CHECK:STDOUT:   %F.call: init i32 = call %F.ref(%.loc22_18)
// CHECK:STDOUT:   %.loc22_20.1: i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc22_20.2: i32 = converted %F.call, %.loc22_20.1
// CHECK:STDOUT:   return %.loc22_20.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
