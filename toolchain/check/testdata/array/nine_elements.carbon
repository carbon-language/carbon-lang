// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/nine_elements.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/nine_elements.carbon

var a: array(i32, 9) = (1, 2, 3, 4, 5, 6, 7, 8, 9);

// CHECK:STDOUT: --- nine_elements.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_9.988: Core.IntLiteral = int_value 9 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_9.988, %i32 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %int_6.462: Core.IntLiteral = int_value 6 [concrete]
// CHECK:STDOUT:   %int_7.29f: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %int_8.b85: Core.IntLiteral = int_value 8 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, %impl_witness.d39 [concrete]
// CHECK:STDOUT:   %.a0b: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
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
// CHECK:STDOUT:   %Convert.bound.ac3: <bound method> = bound_method %int_4.0c1, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.1da: <bound method> = bound_method %int_4.0c1, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.940: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %Convert.bound.4e6: <bound method> = bound_method %int_5.64b, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.a25: <bound method> = bound_method %int_5.64b, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_5.0f6: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %Convert.bound.ce9: <bound method> = bound_method %int_6.462, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.efa: <bound method> = bound_method %int_6.462, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_6.e56: %i32 = int_value 6 [concrete]
// CHECK:STDOUT:   %Convert.bound.208: <bound method> = bound_method %int_7.29f, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.3bd: <bound method> = bound_method %int_7.29f, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_7.0b1: %i32 = int_value 7 [concrete]
// CHECK:STDOUT:   %Convert.bound.e09: <bound method> = bound_method %int_8.b85, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.02d: <bound method> = bound_method %int_8.b85, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_8.98c: %i32 = int_value 8 [concrete]
// CHECK:STDOUT:   %Convert.bound.9e2: <bound method> = bound_method %int_9.988, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.cd3: <bound method> = bound_method %int_9.988, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_9.f88: %i32 = int_value 9 [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%int_1.5d2, %int_2.ef8, %int_3.822, %int_4.940, %int_5.0f6, %int_6.e56, %int_7.0b1, %int_8.98c, %int_9.f88) [concrete]
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
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %array_type = binding_pattern a
// CHECK:STDOUT:     %.loc11_1: %array_type = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type = var a
// CHECK:STDOUT:   %.loc11_20: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_9: Core.IntLiteral = int_value 9 [concrete = constants.%int_9.988]
// CHECK:STDOUT:     %array_type: type = array_type %int_9, %i32 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type = bind_name a, %a.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc11_25: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc11_28: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3.loc11_31: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %int_4.loc11_34: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %int_5.loc11_37: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %int_6.loc11_40: Core.IntLiteral = int_value 6 [concrete = constants.%int_6.462]
// CHECK:STDOUT:   %int_7.loc11_43: Core.IntLiteral = int_value 7 [concrete = constants.%int_7.29f]
// CHECK:STDOUT:   %int_8.loc11_46: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete = constants.%int_9.988]
// CHECK:STDOUT:   %.loc11_50.1: %tuple.type = tuple_literal (%int_1.loc11_25, %int_2.loc11_28, %int_3.loc11_31, %int_4.loc11_34, %int_5.loc11_37, %int_6.loc11_40, %int_7.loc11_43, %int_8.loc11_46, %int_9)
// CHECK:STDOUT:   %impl.elem0.loc11_50.1: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc11_50.1: <bound method> = bound_method %int_1.loc11_25, %impl.elem0.loc11_50.1 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc11_50.1: <specific function> = specific_function %impl.elem0.loc11_50.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_50.2: <bound method> = bound_method %int_1.loc11_25, %specific_fn.loc11_50.1 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc11_50.1: init %i32 = call %bound_method.loc11_50.2(%int_1.loc11_25) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc11_50.2: init %i32 = converted %int_1.loc11_25, %int.convert_checked.loc11_50.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc11_50.3: ref %i32 = array_index file.%a.var, %int_0
// CHECK:STDOUT:   %.loc11_50.4: init %i32 = initialize_from %.loc11_50.2 to %.loc11_50.3 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc11_50.2: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc11_50.3: <bound method> = bound_method %int_2.loc11_28, %impl.elem0.loc11_50.2 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc11_50.2: <specific function> = specific_function %impl.elem0.loc11_50.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_50.4: <bound method> = bound_method %int_2.loc11_28, %specific_fn.loc11_50.2 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc11_50.2: init %i32 = call %bound_method.loc11_50.4(%int_2.loc11_28) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc11_50.5: init %i32 = converted %int_2.loc11_28, %int.convert_checked.loc11_50.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %int_1.loc11_50: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc11_50.6: ref %i32 = array_index file.%a.var, %int_1.loc11_50
// CHECK:STDOUT:   %.loc11_50.7: init %i32 = initialize_from %.loc11_50.5 to %.loc11_50.6 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %impl.elem0.loc11_50.3: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc11_50.5: <bound method> = bound_method %int_3.loc11_31, %impl.elem0.loc11_50.3 [concrete = constants.%Convert.bound.b30]
// CHECK:STDOUT:   %specific_fn.loc11_50.3: <specific function> = specific_function %impl.elem0.loc11_50.3, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_50.6: <bound method> = bound_method %int_3.loc11_31, %specific_fn.loc11_50.3 [concrete = constants.%bound_method.047]
// CHECK:STDOUT:   %int.convert_checked.loc11_50.3: init %i32 = call %bound_method.loc11_50.6(%int_3.loc11_31) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc11_50.8: init %i32 = converted %int_3.loc11_31, %int.convert_checked.loc11_50.3 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %int_2.loc11_50: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc11_50.9: ref %i32 = array_index file.%a.var, %int_2.loc11_50
// CHECK:STDOUT:   %.loc11_50.10: init %i32 = initialize_from %.loc11_50.8 to %.loc11_50.9 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %impl.elem0.loc11_50.4: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc11_50.7: <bound method> = bound_method %int_4.loc11_34, %impl.elem0.loc11_50.4 [concrete = constants.%Convert.bound.ac3]
// CHECK:STDOUT:   %specific_fn.loc11_50.4: <specific function> = specific_function %impl.elem0.loc11_50.4, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_50.8: <bound method> = bound_method %int_4.loc11_34, %specific_fn.loc11_50.4 [concrete = constants.%bound_method.1da]
// CHECK:STDOUT:   %int.convert_checked.loc11_50.4: init %i32 = call %bound_method.loc11_50.8(%int_4.loc11_34) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc11_50.11: init %i32 = converted %int_4.loc11_34, %int.convert_checked.loc11_50.4 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %int_3.loc11_50: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc11_50.12: ref %i32 = array_index file.%a.var, %int_3.loc11_50
// CHECK:STDOUT:   %.loc11_50.13: init %i32 = initialize_from %.loc11_50.11 to %.loc11_50.12 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %impl.elem0.loc11_50.5: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc11_50.9: <bound method> = bound_method %int_5.loc11_37, %impl.elem0.loc11_50.5 [concrete = constants.%Convert.bound.4e6]
// CHECK:STDOUT:   %specific_fn.loc11_50.5: <specific function> = specific_function %impl.elem0.loc11_50.5, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_50.10: <bound method> = bound_method %int_5.loc11_37, %specific_fn.loc11_50.5 [concrete = constants.%bound_method.a25]
// CHECK:STDOUT:   %int.convert_checked.loc11_50.5: init %i32 = call %bound_method.loc11_50.10(%int_5.loc11_37) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc11_50.14: init %i32 = converted %int_5.loc11_37, %int.convert_checked.loc11_50.5 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %int_4.loc11_50: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %.loc11_50.15: ref %i32 = array_index file.%a.var, %int_4.loc11_50
// CHECK:STDOUT:   %.loc11_50.16: init %i32 = initialize_from %.loc11_50.14 to %.loc11_50.15 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %impl.elem0.loc11_50.6: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc11_50.11: <bound method> = bound_method %int_6.loc11_40, %impl.elem0.loc11_50.6 [concrete = constants.%Convert.bound.ce9]
// CHECK:STDOUT:   %specific_fn.loc11_50.6: <specific function> = specific_function %impl.elem0.loc11_50.6, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_50.12: <bound method> = bound_method %int_6.loc11_40, %specific_fn.loc11_50.6 [concrete = constants.%bound_method.efa]
// CHECK:STDOUT:   %int.convert_checked.loc11_50.6: init %i32 = call %bound_method.loc11_50.12(%int_6.loc11_40) [concrete = constants.%int_6.e56]
// CHECK:STDOUT:   %.loc11_50.17: init %i32 = converted %int_6.loc11_40, %int.convert_checked.loc11_50.6 [concrete = constants.%int_6.e56]
// CHECK:STDOUT:   %int_5.loc11_50: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %.loc11_50.18: ref %i32 = array_index file.%a.var, %int_5.loc11_50
// CHECK:STDOUT:   %.loc11_50.19: init %i32 = initialize_from %.loc11_50.17 to %.loc11_50.18 [concrete = constants.%int_6.e56]
// CHECK:STDOUT:   %impl.elem0.loc11_50.7: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc11_50.13: <bound method> = bound_method %int_7.loc11_43, %impl.elem0.loc11_50.7 [concrete = constants.%Convert.bound.208]
// CHECK:STDOUT:   %specific_fn.loc11_50.7: <specific function> = specific_function %impl.elem0.loc11_50.7, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_50.14: <bound method> = bound_method %int_7.loc11_43, %specific_fn.loc11_50.7 [concrete = constants.%bound_method.3bd]
// CHECK:STDOUT:   %int.convert_checked.loc11_50.7: init %i32 = call %bound_method.loc11_50.14(%int_7.loc11_43) [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:   %.loc11_50.20: init %i32 = converted %int_7.loc11_43, %int.convert_checked.loc11_50.7 [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:   %int_6.loc11_50: Core.IntLiteral = int_value 6 [concrete = constants.%int_6.462]
// CHECK:STDOUT:   %.loc11_50.21: ref %i32 = array_index file.%a.var, %int_6.loc11_50
// CHECK:STDOUT:   %.loc11_50.22: init %i32 = initialize_from %.loc11_50.20 to %.loc11_50.21 [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:   %impl.elem0.loc11_50.8: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc11_50.15: <bound method> = bound_method %int_8.loc11_46, %impl.elem0.loc11_50.8 [concrete = constants.%Convert.bound.e09]
// CHECK:STDOUT:   %specific_fn.loc11_50.8: <specific function> = specific_function %impl.elem0.loc11_50.8, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_50.16: <bound method> = bound_method %int_8.loc11_46, %specific_fn.loc11_50.8 [concrete = constants.%bound_method.02d]
// CHECK:STDOUT:   %int.convert_checked.loc11_50.8: init %i32 = call %bound_method.loc11_50.16(%int_8.loc11_46) [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc11_50.23: init %i32 = converted %int_8.loc11_46, %int.convert_checked.loc11_50.8 [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %int_7.loc11_50: Core.IntLiteral = int_value 7 [concrete = constants.%int_7.29f]
// CHECK:STDOUT:   %.loc11_50.24: ref %i32 = array_index file.%a.var, %int_7.loc11_50
// CHECK:STDOUT:   %.loc11_50.25: init %i32 = initialize_from %.loc11_50.23 to %.loc11_50.24 [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %impl.elem0.loc11_50.9: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc11_50.17: <bound method> = bound_method %int_9, %impl.elem0.loc11_50.9 [concrete = constants.%Convert.bound.9e2]
// CHECK:STDOUT:   %specific_fn.loc11_50.9: <specific function> = specific_function %impl.elem0.loc11_50.9, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_50.18: <bound method> = bound_method %int_9, %specific_fn.loc11_50.9 [concrete = constants.%bound_method.cd3]
// CHECK:STDOUT:   %int.convert_checked.loc11_50.9: init %i32 = call %bound_method.loc11_50.18(%int_9) [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   %.loc11_50.26: init %i32 = converted %int_9, %int.convert_checked.loc11_50.9 [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   %int_8.loc11_50: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %.loc11_50.27: ref %i32 = array_index file.%a.var, %int_8.loc11_50
// CHECK:STDOUT:   %.loc11_50.28: init %i32 = initialize_from %.loc11_50.26 to %.loc11_50.27 [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   %.loc11_50.29: init %array_type = array_init (%.loc11_50.4, %.loc11_50.7, %.loc11_50.10, %.loc11_50.13, %.loc11_50.16, %.loc11_50.19, %.loc11_50.22, %.loc11_50.25, %.loc11_50.28) to file.%a.var [concrete = constants.%array]
// CHECK:STDOUT:   %.loc11_1: init %array_type = converted %.loc11_50.1, %.loc11_50.29 [concrete = constants.%array]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
