// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/self.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/self.carbon

class Class {
  fn F[self: Self]() -> i32;
  fn G[addr self: Self*]() -> i32;

  var n: i32;
}

fn Class.F[self: Self]() -> i32 {
  return self.n;
}

fn Class.G[addr self: Self*]() -> i32 {
  return (*self).n;
}

// CHECK:STDOUT: --- self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = ptr_type %Class [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = unbound_element_type %Class, i32 [template]
// CHECK:STDOUT:   %.4: type = struct_type {.n: i32} [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.4 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %Self.ref.loc18: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %self.loc18_12.1: %Class = param self, runtime_param0
// CHECK:STDOUT:     @F.%self: %Class = bind_name self, %self.loc18_12.1
// CHECK:STDOUT:     %int.make_type_32.loc18: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc18_29.1: type = value_of_initializer %int.make_type_32.loc18 [template = i32]
// CHECK:STDOUT:     %.loc18_29.2: type = converted %int.make_type_32.loc18, %.loc18_29.1 [template = i32]
// CHECK:STDOUT:     @F.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %Self.ref.loc22: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %.loc22_27: type = ptr_type %Class [template = constants.%.2]
// CHECK:STDOUT:     %self.loc22_17.1: %.2 = param self, runtime_param0
// CHECK:STDOUT:     @G.%self: %.2 = bind_name self, %self.loc22_17.1
// CHECK:STDOUT:     @G.%.loc22: %.2 = addr_pattern @G.%self
// CHECK:STDOUT:     %int.make_type_32.loc22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc22_35.1: type = value_of_initializer %int.make_type_32.loc22 [template = i32]
// CHECK:STDOUT:     %.loc22_35.2: type = converted %int.make_type_32.loc22, %.loc22_35.1 [template = i32]
// CHECK:STDOUT:     @G.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %Self.ref.loc12: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %self.loc12_8.1: %Class = param self, runtime_param0
// CHECK:STDOUT:     %self.loc12_8.2: %Class = bind_name self, %self.loc12_8.1
// CHECK:STDOUT:     %int.make_type_32.loc12: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc12_25.1: type = value_of_initializer %int.make_type_32.loc12 [template = i32]
// CHECK:STDOUT:     %.loc12_25.2: type = converted %int.make_type_32.loc12, %.loc12_25.1 [template = i32]
// CHECK:STDOUT:     %return.var.loc12: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %Self.ref.loc13: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:     %.loc13_23: type = ptr_type %Class [template = constants.%.2]
// CHECK:STDOUT:     %self.loc13_13.1: %.2 = param self, runtime_param0
// CHECK:STDOUT:     %self.loc13_13.3: %.2 = bind_name self, %self.loc13_13.1
// CHECK:STDOUT:     %.loc13_8: %.2 = addr_pattern %self.loc13_13.3
// CHECK:STDOUT:     %int.make_type_32.loc13: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc13_31.1: type = value_of_initializer %int.make_type_32.loc13 [template = i32]
// CHECK:STDOUT:     %.loc13_31.2: type = converted %int.make_type_32.loc13, %.loc13_31.1 [template = i32]
// CHECK:STDOUT:     %return.var.loc13: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %int.make_type_32.loc15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc15_10.1: type = value_of_initializer %int.make_type_32.loc15 [template = i32]
// CHECK:STDOUT:   %.loc15_10.2: type = converted %int.make_type_32.loc15, %.loc15_10.1 [template = i32]
// CHECK:STDOUT:   %.loc15_8: %.3 = field_decl n, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   .n = %.loc15_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[%self: %Class]() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %Class = name_ref self, %self
// CHECK:STDOUT:   %n.ref: %.3 = name_ref n, @Class.%.loc15_8 [template = @Class.%.loc15_8]
// CHECK:STDOUT:   %.loc19_14.1: ref i32 = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc19_14.2: i32 = bind_value %.loc19_14.1
// CHECK:STDOUT:   return %.loc19_14.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G[addr %self: %.2]() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: %.2 = name_ref self, %self
// CHECK:STDOUT:   %.loc23_11: ref %Class = deref %self.ref
// CHECK:STDOUT:   %n.ref: %.3 = name_ref n, @Class.%.loc15_8 [template = @Class.%.loc15_8]
// CHECK:STDOUT:   %.loc23_17.1: ref i32 = class_element_access %.loc23_11, element0
// CHECK:STDOUT:   %.loc23_17.2: i32 = bind_value %.loc23_17.1
// CHECK:STDOUT:   return %.loc23_17.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
