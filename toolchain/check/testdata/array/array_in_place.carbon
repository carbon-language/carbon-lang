// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/array_in_place.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/array_in_place.carbon

fn F() -> (i32, i32, i32);

fn G() {
  var v: [(i32, i32, i32); 2] = (F(), F());
}

// CHECK:STDOUT: --- array_in_place.carbon
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
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %array_type: type = array_type %int_2, %tuple.type.2 [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (%tuple.type.2, %tuple.type.2) [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template]
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
// CHECK:STDOUT:   %int_32.loc14_12: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc14_12: init type = call constants.%Int(%int_32.loc14_12) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc14_17: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc14_17: init type = call constants.%Int(%int_32.loc14_17) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc14_22: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc14_22: init type = call constants.%Int(%int_32.loc14_22) [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_25.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc14_12, %int.make_type_signed.loc14_17, %int.make_type_signed.loc14_22)
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [template = constants.%int_2]
// CHECK:STDOUT:   %.loc14_25.2: type = value_of_initializer %int.make_type_signed.loc14_12 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_25.3: type = converted %int.make_type_signed.loc14_12, %.loc14_25.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_25.4: type = value_of_initializer %int.make_type_signed.loc14_17 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_25.5: type = converted %int.make_type_signed.loc14_17, %.loc14_25.4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_25.6: type = value_of_initializer %int.make_type_signed.loc14_22 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_25.7: type = converted %int.make_type_signed.loc14_22, %.loc14_25.6 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_25.8: type = converted %.loc14_25.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %array_type: type = array_type %int_2, %tuple.type.2 [template = constants.%array_type]
// CHECK:STDOUT:   %v.var: ref %array_type = var v
// CHECK:STDOUT:   %v: ref %array_type = bind_name v, %v.var
// CHECK:STDOUT:   %F.ref.loc14_34: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %.loc14_42.2: ref %tuple.type.2 = splice_block %.loc14_42.1 {
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:     %.loc14_42.1: ref %tuple.type.2 = array_index %v.var, %int_0
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.call.loc14_36: init %tuple.type.2 = call %F.ref.loc14_34() to %.loc14_42.2
// CHECK:STDOUT:   %F.ref.loc14_39: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %.loc14_42.4: ref %tuple.type.2 = splice_block %.loc14_42.3 {
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1]
// CHECK:STDOUT:     %.loc14_42.3: ref %tuple.type.2 = array_index %v.var, %int_1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.call.loc14_41: init %tuple.type.2 = call %F.ref.loc14_39() to %.loc14_42.4
// CHECK:STDOUT:   %.loc14_42.5: %tuple.type.3 = tuple_literal (%F.call.loc14_36, %F.call.loc14_41)
// CHECK:STDOUT:   %.loc14_42.6: init %array_type = array_init (%F.call.loc14_36, %F.call.loc14_41) to %v.var
// CHECK:STDOUT:   %.loc14_43: init %array_type = converted %.loc14_42.5, %.loc14_42.6
// CHECK:STDOUT:   assign %v.var, %.loc14_43
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
