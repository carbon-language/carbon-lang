// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class Class;

class Class {
  fn F[self: Self](b: bool);
}

fn Class.F[self: Self](b: bool) {}

// CHECK:STDOUT: --- redeclaration.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %Bool: type = fn_type @Bool [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Bool = struct_value () [template]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %struct.2: F = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: type = ptr_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Class = %Class.decl.loc7
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Class.decl.loc7: type = class_decl @Class [template = constants.%Class] {}
// CHECK:STDOUT:   %Class.decl.loc9: type = class_decl @Class [template = constants.%Class] {}
// CHECK:STDOUT:   %import_ref.1: Bool = import_ref ir2, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.2: Bool = import_ref ir2, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.2] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %self.loc13_12.1: Class = param self
// CHECK:STDOUT:     @F.%self: Class = bind_name self, %self.loc13_12.1
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%struct.1() [template = bool]
// CHECK:STDOUT:     %.loc13_27.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc13_27.2: type = converted %bool.make_type, %.loc13_27.1 [template = bool]
// CHECK:STDOUT:     %b.loc13_24.1: bool = param b
// CHECK:STDOUT:     @F.%b: bool = bind_name b, %b.loc13_24.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.2] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %self.loc10_8.1: Class = param self
// CHECK:STDOUT:     %self.loc10_8.2: Class = bind_name self, %self.loc10_8.1
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%struct.1() [template = bool]
// CHECK:STDOUT:     %.loc10_23.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc10_23.2: type = converted %bool.make_type, %.loc10_23.1 [template = bool]
// CHECK:STDOUT:     %b.loc10_20.1: bool = param b
// CHECK:STDOUT:     %b.loc10_20.2: bool = bind_name b, %b.loc10_20.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bool() -> type = "bool.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[%self: Class](%b: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
