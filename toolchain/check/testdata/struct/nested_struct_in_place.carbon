// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/struct/nested_struct_in_place.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/struct/nested_struct_in_place.carbon

fn F() -> (i32, i32, i32);

fn G() {
  var v: {.a: (i32, i32, i32), .b: (i32, i32, i32)} = {.a = F(), .b = F()};
}

// CHECK:STDOUT: --- nested_struct_in_place.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (type, type, type) [template]
// CHECK:STDOUT:   %.3: type = tuple_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = struct_type {.a: %.3, .b: %.3} [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.6: type = struct_type {.a: %.5, .b: %.5} [template]
// CHECK:STDOUT:   %.7: type = ptr_type %.6 [template]
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %int.make_type_32.loc11_12: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc11_17: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %int.make_type_32.loc11_22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc11_25.1: %.2 = tuple_literal (%int.make_type_32.loc11_12, %int.make_type_32.loc11_17, %int.make_type_32.loc11_22)
// CHECK:STDOUT:     %.loc11_25.2: type = value_of_initializer %int.make_type_32.loc11_12 [template = i32]
// CHECK:STDOUT:     %.loc11_25.3: type = converted %int.make_type_32.loc11_12, %.loc11_25.2 [template = i32]
// CHECK:STDOUT:     %.loc11_25.4: type = value_of_initializer %int.make_type_32.loc11_17 [template = i32]
// CHECK:STDOUT:     %.loc11_25.5: type = converted %int.make_type_32.loc11_17, %.loc11_25.4 [template = i32]
// CHECK:STDOUT:     %.loc11_25.6: type = value_of_initializer %int.make_type_32.loc11_22 [template = i32]
// CHECK:STDOUT:     %.loc11_25.7: type = converted %int.make_type_32.loc11_22, %.loc11_25.6 [template = i32]
// CHECK:STDOUT:     %.loc11_25.8: type = converted %.loc11_25.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:     @F.%return: ref %.3 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %.3;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32.loc14_16: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc14_21: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc14_26: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc14_29.1: %.2 = tuple_literal (%int.make_type_32.loc14_16, %int.make_type_32.loc14_21, %int.make_type_32.loc14_26)
// CHECK:STDOUT:   %.loc14_29.2: type = value_of_initializer %int.make_type_32.loc14_16 [template = i32]
// CHECK:STDOUT:   %.loc14_29.3: type = converted %int.make_type_32.loc14_16, %.loc14_29.2 [template = i32]
// CHECK:STDOUT:   %.loc14_29.4: type = value_of_initializer %int.make_type_32.loc14_21 [template = i32]
// CHECK:STDOUT:   %.loc14_29.5: type = converted %int.make_type_32.loc14_21, %.loc14_29.4 [template = i32]
// CHECK:STDOUT:   %.loc14_29.6: type = value_of_initializer %int.make_type_32.loc14_26 [template = i32]
// CHECK:STDOUT:   %.loc14_29.7: type = converted %int.make_type_32.loc14_26, %.loc14_29.6 [template = i32]
// CHECK:STDOUT:   %.loc14_29.8: type = converted %.loc14_29.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %int.make_type_32.loc14_37: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc14_42: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc14_47: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc14_50.1: %.2 = tuple_literal (%int.make_type_32.loc14_37, %int.make_type_32.loc14_42, %int.make_type_32.loc14_47)
// CHECK:STDOUT:   %.loc14_50.2: type = value_of_initializer %int.make_type_32.loc14_37 [template = i32]
// CHECK:STDOUT:   %.loc14_50.3: type = converted %int.make_type_32.loc14_37, %.loc14_50.2 [template = i32]
// CHECK:STDOUT:   %.loc14_50.4: type = value_of_initializer %int.make_type_32.loc14_42 [template = i32]
// CHECK:STDOUT:   %.loc14_50.5: type = converted %int.make_type_32.loc14_42, %.loc14_50.4 [template = i32]
// CHECK:STDOUT:   %.loc14_50.6: type = value_of_initializer %int.make_type_32.loc14_47 [template = i32]
// CHECK:STDOUT:   %.loc14_50.7: type = converted %int.make_type_32.loc14_47, %.loc14_50.6 [template = i32]
// CHECK:STDOUT:   %.loc14_50.8: type = converted %.loc14_50.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %.loc14_51: type = struct_type {.a: %.3, .b: %.3} [template = constants.%.4]
// CHECK:STDOUT:   %v.var: ref %.4 = var v
// CHECK:STDOUT:   %v: ref %.4 = bind_name v, %v.var
// CHECK:STDOUT:   %F.ref.loc14_61: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %.loc14_74.1: ref %.3 = struct_access %v.var, element0
// CHECK:STDOUT:   %F.call.loc14_62: init %.3 = call %F.ref.loc14_61() to %.loc14_74.1
// CHECK:STDOUT:   %F.ref.loc14_71: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %.loc14_74.2: ref %.3 = struct_access %v.var, element1
// CHECK:STDOUT:   %F.call.loc14_72: init %.3 = call %F.ref.loc14_71() to %.loc14_74.2
// CHECK:STDOUT:   %.loc14_74.3: %.4 = struct_literal (%F.call.loc14_62, %F.call.loc14_72)
// CHECK:STDOUT:   %.loc14_74.4: init %.4 = struct_init (%F.call.loc14_62, %F.call.loc14_72) to %v.var
// CHECK:STDOUT:   %.loc14_75: init %.4 = converted %.loc14_74.3, %.loc14_74.4
// CHECK:STDOUT:   assign %v.var, %.loc14_75
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
