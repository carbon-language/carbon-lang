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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32, %i32, %i32) [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %struct_type.a.b.1: type = struct_type {.a: %tuple.type.2, .b: %tuple.type.2} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:     %return.patt: %tuple.type.2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %tuple.type.2 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc11_12: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc11_12: init type = call constants.%Int(%int_32.loc11_12) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc11_17: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc11_17: init type = call constants.%Int(%int_32.loc11_17) [template = constants.%i32]
// CHECK:STDOUT:     %int_32.loc11_22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %int.make_type_signed.loc11_22: init type = call constants.%Int(%int_32.loc11_22) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_25.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc11_12, %int.make_type_signed.loc11_17, %int.make_type_signed.loc11_22)
// CHECK:STDOUT:     %.loc11_25.2: type = value_of_initializer %int.make_type_signed.loc11_12 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_25.3: type = converted %int.make_type_signed.loc11_12, %.loc11_25.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_25.4: type = value_of_initializer %int.make_type_signed.loc11_17 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_25.5: type = converted %int.make_type_signed.loc11_17, %.loc11_25.4 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_25.6: type = value_of_initializer %int.make_type_signed.loc11_22 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_25.7: type = converted %int.make_type_signed.loc11_22, %.loc11_25.6 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_25.8: type = converted %.loc11_25.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:     %return.param: ref %tuple.type.2 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %tuple.type.2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %tuple.type.2;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_32.loc14_16: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc14_16: init type = call constants.%Int(%int_32.loc14_16) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc14_21: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc14_21: init type = call constants.%Int(%int_32.loc14_21) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc14_26: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc14_26: init type = call constants.%Int(%int_32.loc14_26) [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_29.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc14_16, %int.make_type_signed.loc14_21, %int.make_type_signed.loc14_26)
// CHECK:STDOUT:   %.loc14_29.2: type = value_of_initializer %int.make_type_signed.loc14_16 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_29.3: type = converted %int.make_type_signed.loc14_16, %.loc14_29.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_29.4: type = value_of_initializer %int.make_type_signed.loc14_21 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_29.5: type = converted %int.make_type_signed.loc14_21, %.loc14_29.4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_29.6: type = value_of_initializer %int.make_type_signed.loc14_26 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_29.7: type = converted %int.make_type_signed.loc14_26, %.loc14_29.6 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_29.8: type = converted %.loc14_29.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %int_32.loc14_37: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc14_37: init type = call constants.%Int(%int_32.loc14_37) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc14_42: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc14_42: init type = call constants.%Int(%int_32.loc14_42) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc14_47: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc14_47: init type = call constants.%Int(%int_32.loc14_47) [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_50.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc14_37, %int.make_type_signed.loc14_42, %int.make_type_signed.loc14_47)
// CHECK:STDOUT:   %.loc14_50.2: type = value_of_initializer %int.make_type_signed.loc14_37 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_50.3: type = converted %int.make_type_signed.loc14_37, %.loc14_50.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_50.4: type = value_of_initializer %int.make_type_signed.loc14_42 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_50.5: type = converted %int.make_type_signed.loc14_42, %.loc14_50.4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_50.6: type = value_of_initializer %int.make_type_signed.loc14_47 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_50.7: type = converted %int.make_type_signed.loc14_47, %.loc14_50.6 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_50.8: type = converted %.loc14_50.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %tuple.type.2, .b: %tuple.type.2} [template = constants.%struct_type.a.b.1]
// CHECK:STDOUT:   %v.var: ref %struct_type.a.b.1 = var v
// CHECK:STDOUT:   %v: ref %struct_type.a.b.1 = bind_name v, %v.var
// CHECK:STDOUT:   %F.ref.loc14_61: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %.loc14_74.1: ref %tuple.type.2 = struct_access %v.var, element0
// CHECK:STDOUT:   %F.call.loc14_63: init %tuple.type.2 = call %F.ref.loc14_61() to %.loc14_74.1
// CHECK:STDOUT:   %F.ref.loc14_71: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %.loc14_74.2: ref %tuple.type.2 = struct_access %v.var, element1
// CHECK:STDOUT:   %F.call.loc14_73: init %tuple.type.2 = call %F.ref.loc14_71() to %.loc14_74.2
// CHECK:STDOUT:   %.loc14_74.3: %struct_type.a.b.1 = struct_literal (%F.call.loc14_63, %F.call.loc14_73)
// CHECK:STDOUT:   %.loc14_74.4: init %struct_type.a.b.1 = struct_init (%F.call.loc14_63, %F.call.loc14_73) to %v.var
// CHECK:STDOUT:   %.loc14_75: init %struct_type.a.b.1 = converted %.loc14_74.3, %.loc14_74.4
// CHECK:STDOUT:   assign %v.var, %.loc14_75
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
