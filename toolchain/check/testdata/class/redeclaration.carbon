// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/redeclaration.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/redeclaration.carbon

class Class;

class Class {
  fn F[self: Self](b: bool);
}

fn Class.F[self: Self](b: bool) {}

// CHECK:STDOUT: --- redeclaration.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Bool = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Bool.type = import_ref Core//prelude/types/bool, inst+2, loaded [template = constants.%Bool]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl.loc11
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl.loc11: type = class_decl @Class [template = constants.%Class] {}
// CHECK:STDOUT:   %Class.decl.loc13: type = class_decl @Class [template = constants.%Class] {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %self.loc17_12.1: %Class = param self
// CHECK:STDOUT:     @F.%self: %Class = bind_name self, %self.loc17_12.1
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc17_27.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc17_27.2: type = converted %bool.make_type, %.loc17_27.1 [template = bool]
// CHECK:STDOUT:     %b.loc17_24.1: bool = param b
// CHECK:STDOUT:     @F.%b: bool = bind_name b, %b.loc17_24.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %self.loc14_8.1: %Class = param self
// CHECK:STDOUT:     %self.loc14_8.2: %Class = bind_name self, %self.loc14_8.1
// CHECK:STDOUT:     %bool.make_type: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:     %.loc14_23.1: type = value_of_initializer %bool.make_type [template = bool]
// CHECK:STDOUT:     %.loc14_23.2: type = converted %bool.make_type, %.loc14_23.1 [template = bool]
// CHECK:STDOUT:     %b.loc14_20.1: bool = param b
// CHECK:STDOUT:     %b.loc14_20.2: bool = bind_name b, %b.loc14_20.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Bool() -> type = "bool.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[%self: %Class](%b: bool) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
