// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/let/convert.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/let/convert.carbon

fn F() -> i32 {
  var v: (i32, i32, i32) = (1, 2, 3);
  // Convert from object representation to value representation.
  let w: (i32, i32, i32) = v;
  return w.1;
}

// CHECK:STDOUT: --- convert.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.ff9: type = tuple_type (type, type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.189: type = tuple_type (%i32, %i32, %i32) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %tuple.type.37f: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [concrete]
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
// CHECK:STDOUT:   %Convert.bound.b30: <bound method> = bound_method %int_3.1ba, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.047: <bound method> = bound_method %int_3.1ba, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.189 = tuple_value (%int_1.5d2, %int_2.ef8, %int_3.822) [concrete]
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
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc11: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc11: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %tuple.type.189 = binding_pattern v
// CHECK:STDOUT:     %.loc12_3.1: %tuple.type.189 = var_pattern %v.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %tuple.type.189 = var v
// CHECK:STDOUT:   %int_1.loc12: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc12_36.1: %tuple.type.37f = tuple_literal (%int_1.loc12, %int_2, %int_3)
// CHECK:STDOUT:   %impl.elem0.loc12_36.1: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc12_36.1: <bound method> = bound_method %int_1.loc12, %impl.elem0.loc12_36.1 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc12_36.1: <specific function> = specific_function %impl.elem0.loc12_36.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_36.2: <bound method> = bound_method %int_1.loc12, %specific_fn.loc12_36.1 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc12_36.1: init %i32 = call %bound_method.loc12_36.2(%int_1.loc12) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc12_36.2: init %i32 = converted %int_1.loc12, %int.convert_checked.loc12_36.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %v.var, element0
// CHECK:STDOUT:   %.loc12_36.3: init %i32 = initialize_from %.loc12_36.2 to %tuple.elem0 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc12_36.2: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc12_36.3: <bound method> = bound_method %int_2, %impl.elem0.loc12_36.2 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc12_36.2: <specific function> = specific_function %impl.elem0.loc12_36.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_36.4: <bound method> = bound_method %int_2, %specific_fn.loc12_36.2 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc12_36.2: init %i32 = call %bound_method.loc12_36.4(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc12_36.4: init %i32 = converted %int_2, %int.convert_checked.loc12_36.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %tuple.elem1.loc12: ref %i32 = tuple_access %v.var, element1
// CHECK:STDOUT:   %.loc12_36.5: init %i32 = initialize_from %.loc12_36.4 to %tuple.elem1.loc12 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %impl.elem0.loc12_36.3: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc12_36.5: <bound method> = bound_method %int_3, %impl.elem0.loc12_36.3 [concrete = constants.%Convert.bound.b30]
// CHECK:STDOUT:   %specific_fn.loc12_36.3: <specific function> = specific_function %impl.elem0.loc12_36.3, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_36.6: <bound method> = bound_method %int_3, %specific_fn.loc12_36.3 [concrete = constants.%bound_method.047]
// CHECK:STDOUT:   %int.convert_checked.loc12_36.3: init %i32 = call %bound_method.loc12_36.6(%int_3) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc12_36.6: init %i32 = converted %int_3, %int.convert_checked.loc12_36.3 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %tuple.elem2: ref %i32 = tuple_access %v.var, element2
// CHECK:STDOUT:   %.loc12_36.7: init %i32 = initialize_from %.loc12_36.6 to %tuple.elem2 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc12_36.8: init %tuple.type.189 = tuple_init (%.loc12_36.3, %.loc12_36.5, %.loc12_36.7) to %v.var [concrete = constants.%tuple]
// CHECK:STDOUT:   %.loc12_3.2: init %tuple.type.189 = converted %.loc12_36.1, %.loc12_36.8 [concrete = constants.%tuple]
// CHECK:STDOUT:   assign %v.var, %.loc12_3.2
// CHECK:STDOUT:   %.loc12_24.1: type = splice_block %.loc12_24.3 [concrete = constants.%tuple.type.189] {
// CHECK:STDOUT:     %int_32.loc12_11: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc12_11: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc12_16: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc12_16: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc12_21: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc12_21: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc12_24.2: %tuple.type.ff9 = tuple_literal (%i32.loc12_11, %i32.loc12_16, %i32.loc12_21)
// CHECK:STDOUT:     %.loc12_24.3: type = converted %.loc12_24.2, constants.%tuple.type.189 [concrete = constants.%tuple.type.189]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %tuple.type.189 = bind_name v, %v.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %w.patt: %tuple.type.189 = binding_pattern w
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.ref: ref %tuple.type.189 = name_ref v, %v
// CHECK:STDOUT:   %.loc14_24.1: type = splice_block %.loc14_24.3 [concrete = constants.%tuple.type.189] {
// CHECK:STDOUT:     %int_32.loc14_11: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc14_11: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc14_16: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc14_16: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_32.loc14_21: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc14_21: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc14_24.2: %tuple.type.ff9 = tuple_literal (%i32.loc14_11, %i32.loc14_16, %i32.loc14_21)
// CHECK:STDOUT:     %.loc14_24.3: type = converted %.loc14_24.2, constants.%tuple.type.189 [concrete = constants.%tuple.type.189]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %w: ref %tuple.type.189 = bind_name w, %v.ref
// CHECK:STDOUT:   %w.ref: ref %tuple.type.189 = name_ref w, %w
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %tuple.elem1.loc15: ref %i32 = tuple_access %w.ref, element1
// CHECK:STDOUT:   %.loc15: %i32 = bind_value %tuple.elem1.loc15
// CHECK:STDOUT:   return %.loc15
// CHECK:STDOUT: }
// CHECK:STDOUT:
