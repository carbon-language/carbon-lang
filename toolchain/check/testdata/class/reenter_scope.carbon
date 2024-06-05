// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/reenter_scope.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/reenter_scope.carbon

class Class {
  fn F() -> i32;
  fn G() -> i32;
}

fn Class.F() -> i32 {
  Self.G();
  return G();
}

// CHECK:STDOUT: --- reenter_scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class: type = class_type @Class [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: <type of Int32> = struct_value () [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: <type of F> = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: <type of G> = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Class.decl: type = class_decl @Class [template = constants.%Class] {}
// CHECK:STDOUT:   %import_ref.1: <type of Int32> = import_ref ir3, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: <type of Int32> = import_ref ir3, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.3: <type of Int32> = import_ref ir3, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %F.decl: <type of F> = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc16_17.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc16_17.2: type = converted %int.make_type_32, %.loc16_17.1 [template = i32]
// CHECK:STDOUT:     @F.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %F.decl: <type of F> = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %int.make_type_32.loc12: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc12_13.1: type = value_of_initializer %int.make_type_32.loc12 [template = i32]
// CHECK:STDOUT:     %.loc12_13.2: type = converted %int.make_type_32.loc12, %.loc12_13.1 [template = i32]
// CHECK:STDOUT:     %return.var.loc12: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: <type of G> = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %int.make_type_32.loc13: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc13_13.1: type = value_of_initializer %int.make_type_32.loc13 [template = i32]
// CHECK:STDOUT:     %.loc13_13.2: type = converted %int.make_type_32.loc13, %.loc13_13.1 [template = i32]
// CHECK:STDOUT:     %return.var.loc13: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Self.ref: type = name_ref Self, constants.%Class [template = constants.%Class]
// CHECK:STDOUT:   %G.ref.loc17: <type of G> = name_ref G, @Class.%G.decl [template = constants.%G]
// CHECK:STDOUT:   %G.call.loc17: init i32 = call %G.ref.loc17()
// CHECK:STDOUT:   %G.ref.loc18: <type of G> = name_ref G, @Class.%G.decl [template = constants.%G]
// CHECK:STDOUT:   %G.call.loc18: init i32 = call %G.ref.loc18()
// CHECK:STDOUT:   %.loc18_13.1: i32 = value_of_initializer %G.call.loc18
// CHECK:STDOUT:   %.loc18_13.2: i32 = converted %G.call.loc18, %.loc18_13.1
// CHECK:STDOUT:   return %.loc18_13.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> i32;
// CHECK:STDOUT:
