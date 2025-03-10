// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/nested_tuple_in_place.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/nested_tuple_in_place.carbon

fn F() -> (i32, i32, i32);

fn G() {
  var v: ((i32, i32, i32), (i32, i32, i32)) = (F(), F());
}

fn H() {
  var v: (i32, (i32, i32, i32), i32) = (1, F(), 2);
}

// CHECK:STDOUT: --- nested_tuple_in_place.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.ff9: type = tuple_type (type, type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.189: type = tuple_type (%i32, %i32, %i32) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.f9f: type = tuple_type (%tuple.type.ff9, %tuple.type.ff9) [concrete]
// CHECK:STDOUT:   %tuple.type.99b: type = tuple_type (%tuple.type.189, %tuple.type.189) [concrete]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.3a3: type = tuple_type (type, %tuple.type.ff9, type) [concrete]
// CHECK:STDOUT:   %tuple.type.516: type = tuple_type (%i32, %tuple.type.189, %i32) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %tuple.type.667: type = tuple_type (Core.IntLiteral, %tuple.type.189, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%impl_witness.d39) [concrete]
// CHECK:STDOUT:   %.be7: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.9a1: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.b92: <bound method> = bound_method %int_2.ecc, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.patt: %tuple.type.189 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %tuple.type.189 = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc11_12: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11_12: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc11_17: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11_17: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc11_22: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11_22: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc11_25.1: %tuple.type.ff9 = tuple_literal (%i32.loc11_12, %i32.loc11_17, %i32.loc11_22)
// CHECK:STDOUT:     %.loc11_25.2: type = converted %.loc11_25.1, constants.%tuple.type.189 [concrete = constants.%tuple.type.189]
// CHECK:STDOUT:     %return.param: ref %tuple.type.189 = out_param call_param0
// CHECK:STDOUT:     %return: ref %tuple.type.189 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %tuple.type.189;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %tuple.type.99b = binding_pattern v
// CHECK:STDOUT:     %.loc14_3.1: %tuple.type.99b = var_pattern %v.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %tuple.type.99b = var v
// CHECK:STDOUT:   %F.ref.loc14_48: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %tuple.elem0: ref %tuple.type.189 = tuple_access %v.var, element0
// CHECK:STDOUT:   %F.call.loc14_50: init %tuple.type.189 = call %F.ref.loc14_48() to %tuple.elem0
// CHECK:STDOUT:   %F.ref.loc14_53: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %tuple.elem1: ref %tuple.type.189 = tuple_access %v.var, element1
// CHECK:STDOUT:   %F.call.loc14_55: init %tuple.type.189 = call %F.ref.loc14_53() to %tuple.elem1
// CHECK:STDOUT:   %.loc14_56.1: %tuple.type.99b = tuple_literal (%F.call.loc14_50, %F.call.loc14_55)
// CHECK:STDOUT:   %.loc14_56.2: init %tuple.type.99b = tuple_init (%F.call.loc14_50, %F.call.loc14_55) to %v.var
// CHECK:STDOUT:   %.loc14_3.2: init %tuple.type.99b = converted %.loc14_56.1, %.loc14_56.2
// CHECK:STDOUT:   assign %v.var, %.loc14_3.2
// CHECK:STDOUT:   %.loc14_43.1: type = splice_block %.loc14_43.5 [concrete = constants.%tuple.type.99b] {
// CHECK:STDOUT:     %int_32.loc14_12: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc14_12: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc14_17: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc14_17: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc14_22: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc14_22: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc14_25: %tuple.type.ff9 = tuple_literal (%i32.loc14_12, %i32.loc14_17, %i32.loc14_22)
// CHECK:STDOUT:     %int_32.loc14_29: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc14_29: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc14_34: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc14_34: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc14_39: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc14_39: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc14_42: %tuple.type.ff9 = tuple_literal (%i32.loc14_29, %i32.loc14_34, %i32.loc14_39)
// CHECK:STDOUT:     %.loc14_43.2: %tuple.type.f9f = tuple_literal (%.loc14_25, %.loc14_42)
// CHECK:STDOUT:     %.loc14_43.3: type = converted %.loc14_25, constants.%tuple.type.189 [concrete = constants.%tuple.type.189]
// CHECK:STDOUT:     %.loc14_43.4: type = converted %.loc14_42, constants.%tuple.type.189 [concrete = constants.%tuple.type.189]
// CHECK:STDOUT:     %.loc14_43.5: type = converted %.loc14_43.2, constants.%tuple.type.99b [concrete = constants.%tuple.type.99b]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %tuple.type.99b = bind_name v, %v.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %tuple.type.516 = binding_pattern v
// CHECK:STDOUT:     %.loc18_3.1: %tuple.type.516 = var_pattern %v.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %tuple.type.516 = var v
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %tuple.elem1: ref %tuple.type.189 = tuple_access %v.var, element1
// CHECK:STDOUT:   %F.call: init %tuple.type.189 = call %F.ref() to %tuple.elem1
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc18_50.1: %tuple.type.667 = tuple_literal (%int_1, %F.call, %int_2)
// CHECK:STDOUT:   %impl.elem0.loc18_50.1: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc18_50.1: <bound method> = bound_method %int_1, %impl.elem0.loc18_50.1 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc18_50.1: <specific function> = specific_function %impl.elem0.loc18_50.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_50.2: <bound method> = bound_method %int_1, %specific_fn.loc18_50.1 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc18_50.1: init %i32 = call %bound_method.loc18_50.2(%int_1) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc18_50.2: init %i32 = converted %int_1, %int.convert_checked.loc18_50.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %v.var, element0
// CHECK:STDOUT:   %.loc18_50.3: init %i32 = initialize_from %.loc18_50.2 to %tuple.elem0 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc18_50.2: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc18_50.3: <bound method> = bound_method %int_2, %impl.elem0.loc18_50.2 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc18_50.2: <specific function> = specific_function %impl.elem0.loc18_50.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_50.4: <bound method> = bound_method %int_2, %specific_fn.loc18_50.2 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc18_50.2: init %i32 = call %bound_method.loc18_50.4(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc18_50.4: init %i32 = converted %int_2, %int.convert_checked.loc18_50.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %tuple.elem2: ref %i32 = tuple_access %v.var, element2
// CHECK:STDOUT:   %.loc18_50.5: init %i32 = initialize_from %.loc18_50.4 to %tuple.elem2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc18_50.6: init %tuple.type.516 = tuple_init (%.loc18_50.3, %F.call, %.loc18_50.5) to %v.var
// CHECK:STDOUT:   %.loc18_3.2: init %tuple.type.516 = converted %.loc18_50.1, %.loc18_50.6
// CHECK:STDOUT:   assign %v.var, %.loc18_3.2
// CHECK:STDOUT:   %.loc18_36.1: type = splice_block %.loc18_36.4 [concrete = constants.%tuple.type.516] {
// CHECK:STDOUT:     %int_32.loc18_11: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc18_11: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc18_17: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc18_17: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc18_22: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc18_22: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc18_27: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc18_27: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc18_30: %tuple.type.ff9 = tuple_literal (%i32.loc18_17, %i32.loc18_22, %i32.loc18_27)
// CHECK:STDOUT:     %int_32.loc18_33: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc18_33: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc18_36.2: %tuple.type.3a3 = tuple_literal (%i32.loc18_11, %.loc18_30, %i32.loc18_33)
// CHECK:STDOUT:     %.loc18_36.3: type = converted %.loc18_30, constants.%tuple.type.189 [concrete = constants.%tuple.type.189]
// CHECK:STDOUT:     %.loc18_36.4: type = converted %.loc18_36.2, constants.%tuple.type.516 [concrete = constants.%tuple.type.516]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %tuple.type.516 = bind_name v, %v.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
