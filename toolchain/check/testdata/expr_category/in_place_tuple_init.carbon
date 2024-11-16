// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/expr_category/in_place_tuple_init.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/expr_category/in_place_tuple_init.carbon

fn F() -> (i32, i32);

fn G() -> (i32, i32) {
  var v: (i32, i32) = F();
  v = F();
  return F();
}

fn H() -> i32 {
  return G().0;
}

// CHECK:STDOUT: --- in_place_tuple_init.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %H.type: type = fn_type @H [template]
// CHECK:STDOUT:   %H: %H.type = struct_value () [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 0 [template]
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
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %return.patt: %tuple.type.2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %tuple.type.2 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_12: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc11_12: init type = call constants.%Int(%.loc11_12) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_17: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc11_17: init type = call constants.%Int(%.loc11_17) [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_20.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc11_12, %int.make_type_signed.loc11_17)
// CHECK:STDOUT:     %.loc11_20.2: type = value_of_initializer %int.make_type_signed.loc11_12 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_20.3: type = converted %int.make_type_signed.loc11_12, %.loc11_20.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_20.4: type = value_of_initializer %int.make_type_signed.loc11_17 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_20.5: type = converted %int.make_type_signed.loc11_17, %.loc11_20.4 [template = constants.%i32]
// CHECK:STDOUT:     %.loc11_20.6: type = converted %.loc11_20.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:     %return.param: ref %tuple.type.2 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %tuple.type.2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %return.patt: %tuple.type.2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %tuple.type.2 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13_12: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc13_12: init type = call constants.%Int(%.loc13_12) [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_17: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc13_17: init type = call constants.%Int(%.loc13_17) [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_20.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc13_12, %int.make_type_signed.loc13_17)
// CHECK:STDOUT:     %.loc13_20.2: type = value_of_initializer %int.make_type_signed.loc13_12 [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_20.3: type = converted %int.make_type_signed.loc13_12, %.loc13_20.2 [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_20.4: type = value_of_initializer %int.make_type_signed.loc13_17 [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_20.5: type = converted %int.make_type_signed.loc13_17, %.loc13_20.4 [template = constants.%i32]
// CHECK:STDOUT:     %.loc13_20.6: type = converted %.loc13_20.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:     %return.param: ref %tuple.type.2 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %tuple.type.2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [template = constants.%H] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc19_11.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc19_11.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc19_11.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc19_11.3: type = converted %int.make_type_signed, %.loc19_11.2 [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %tuple.type.2;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %return: %tuple.type.2 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_11: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc14_11: init type = call constants.%Int(%.loc14_11) [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_16: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc14_16: init type = call constants.%Int(%.loc14_16) [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_19.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc14_11, %int.make_type_signed.loc14_16)
// CHECK:STDOUT:   %.loc14_19.2: type = value_of_initializer %int.make_type_signed.loc14_11 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_19.3: type = converted %int.make_type_signed.loc14_11, %.loc14_19.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_19.4: type = value_of_initializer %int.make_type_signed.loc14_16 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_19.5: type = converted %int.make_type_signed.loc14_16, %.loc14_19.4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_19.6: type = converted %.loc14_19.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %v.var: ref %tuple.type.2 = var v
// CHECK:STDOUT:   %v: ref %tuple.type.2 = bind_name v, %v.var
// CHECK:STDOUT:   %F.ref.loc14: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %.loc14_7: ref %tuple.type.2 = splice_block %v.var {}
// CHECK:STDOUT:   %F.call.loc14: init %tuple.type.2 = call %F.ref.loc14() to %.loc14_7
// CHECK:STDOUT:   assign %v.var, %F.call.loc14
// CHECK:STDOUT:   %v.ref: ref %tuple.type.2 = name_ref v, %v
// CHECK:STDOUT:   %F.ref.loc15: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %.loc15: ref %tuple.type.2 = splice_block %v.ref {}
// CHECK:STDOUT:   %F.call.loc15: init %tuple.type.2 = call %F.ref.loc15() to %.loc15
// CHECK:STDOUT:   assign %v.ref, %F.call.loc15
// CHECK:STDOUT:   %F.ref.loc16: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %.loc13_8.2: ref %tuple.type.2 = splice_block %return {}
// CHECK:STDOUT:   %F.call.loc16: init %tuple.type.2 = call %F.ref.loc16() to %.loc13_8.2
// CHECK:STDOUT:   return %F.call.loc16 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [template = constants.%G]
// CHECK:STDOUT:   %.loc20_11.1: ref %tuple.type.2 = temporary_storage
// CHECK:STDOUT:   %G.call: init %tuple.type.2 = call %G.ref() to %.loc20_11.1
// CHECK:STDOUT:   %.loc20_14: Core.IntLiteral = int_value 0 [template = constants.%.3]
// CHECK:STDOUT:   %.loc20_11.2: ref %tuple.type.2 = temporary %.loc20_11.1, %G.call
// CHECK:STDOUT:   %.loc20_13.1: ref %i32 = tuple_access %.loc20_11.2, element0
// CHECK:STDOUT:   %.loc20_13.2: %i32 = bind_value %.loc20_13.1
// CHECK:STDOUT:   return %.loc20_13.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
