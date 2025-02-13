// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/array/nine_elements.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/array/nine_elements.carbon

var a: [i32; 9] = (1, 2, 3, 4, 5, 6, 7, 8, 9);

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
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.1(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.1(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, %impl_witness.d39 [concrete]
// CHECK:STDOUT:   %.a0b: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn.70c: <specific function> = specific_function %Convert.bound.ab5, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn.787: <specific function> = specific_function %Convert.bound.ef9, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %Convert.bound.b30: <bound method> = bound_method %int_3.1ba, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn.b42: <specific function> = specific_function %Convert.bound.b30, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %Convert.bound.ac3: <bound method> = bound_method %int_4.0c1, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn.450: <specific function> = specific_function %Convert.bound.ac3, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %int_4.940: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %Convert.bound.4e6: <bound method> = bound_method %int_5.64b, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn.ba9: <specific function> = specific_function %Convert.bound.4e6, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %int_5.0f6: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %Convert.bound.ce9: <bound method> = bound_method %int_6.462, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn.631: <specific function> = specific_function %Convert.bound.ce9, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %int_6.e56: %i32 = int_value 6 [concrete]
// CHECK:STDOUT:   %Convert.bound.208: <bound method> = bound_method %int_7.29f, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn.c12: <specific function> = specific_function %Convert.bound.208, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %int_7.0b1: %i32 = int_value 7 [concrete]
// CHECK:STDOUT:   %Convert.bound.e09: <bound method> = bound_method %int_8.b85, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn.e0d: <specific function> = specific_function %Convert.bound.e09, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %int_8.98c: %i32 = int_value 8 [concrete]
// CHECK:STDOUT:   %Convert.bound.9e2: <bound method> = bound_method %int_9.988, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn.b02: <specific function> = specific_function %Convert.bound.9e2, @Convert.2(%int_32) [concrete]
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
// CHECK:STDOUT:   %.loc11_15: type = splice_block %array_type [concrete = constants.%array_type] {
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
// CHECK:STDOUT:   %int_1.loc11_20: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2.loc11_23: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3.loc11_26: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %int_4.loc11_29: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %int_5.loc11_32: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %int_6.loc11_35: Core.IntLiteral = int_value 6 [concrete = constants.%int_6.462]
// CHECK:STDOUT:   %int_7.loc11_38: Core.IntLiteral = int_value 7 [concrete = constants.%int_7.29f]
// CHECK:STDOUT:   %int_8.loc11_41: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete = constants.%int_9.988]
// CHECK:STDOUT:   %.loc11_45.1: %tuple.type = tuple_literal (%int_1.loc11_20, %int_2.loc11_23, %int_3.loc11_26, %int_4.loc11_29, %int_5.loc11_32, %int_6.loc11_35, %int_7.loc11_38, %int_8.loc11_41, %int_9)
// CHECK:STDOUT:   %impl.elem0.loc11_45.1: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc11_45.1: <bound method> = bound_method %int_1.loc11_20, %impl.elem0.loc11_45.1 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc11_45.1: <specific function> = specific_function %bound_method.loc11_45.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn.70c]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.1: init %i32 = call %specific_fn.loc11_45.1(%int_1.loc11_20) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc11_45.2: init %i32 = converted %int_1.loc11_20, %int.convert_checked.loc11_45.1 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc11_45.3: ref %i32 = array_index file.%a.var, %int_0
// CHECK:STDOUT:   %.loc11_45.4: init %i32 = initialize_from %.loc11_45.2 to %.loc11_45.3 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc11_45.2: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc11_45.2: <bound method> = bound_method %int_2.loc11_23, %impl.elem0.loc11_45.2 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc11_45.2: <specific function> = specific_function %bound_method.loc11_45.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn.787]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.2: init %i32 = call %specific_fn.loc11_45.2(%int_2.loc11_23) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc11_45.5: init %i32 = converted %int_2.loc11_23, %int.convert_checked.loc11_45.2 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %int_1.loc11_45: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc11_45.6: ref %i32 = array_index file.%a.var, %int_1.loc11_45
// CHECK:STDOUT:   %.loc11_45.7: init %i32 = initialize_from %.loc11_45.5 to %.loc11_45.6 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %impl.elem0.loc11_45.3: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc11_45.3: <bound method> = bound_method %int_3.loc11_26, %impl.elem0.loc11_45.3 [concrete = constants.%Convert.bound.b30]
// CHECK:STDOUT:   %specific_fn.loc11_45.3: <specific function> = specific_function %bound_method.loc11_45.3, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn.b42]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.3: init %i32 = call %specific_fn.loc11_45.3(%int_3.loc11_26) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc11_45.8: init %i32 = converted %int_3.loc11_26, %int.convert_checked.loc11_45.3 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %int_2.loc11_45: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc11_45.9: ref %i32 = array_index file.%a.var, %int_2.loc11_45
// CHECK:STDOUT:   %.loc11_45.10: init %i32 = initialize_from %.loc11_45.8 to %.loc11_45.9 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %impl.elem0.loc11_45.4: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc11_45.4: <bound method> = bound_method %int_4.loc11_29, %impl.elem0.loc11_45.4 [concrete = constants.%Convert.bound.ac3]
// CHECK:STDOUT:   %specific_fn.loc11_45.4: <specific function> = specific_function %bound_method.loc11_45.4, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn.450]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.4: init %i32 = call %specific_fn.loc11_45.4(%int_4.loc11_29) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc11_45.11: init %i32 = converted %int_4.loc11_29, %int.convert_checked.loc11_45.4 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %int_3.loc11_45: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc11_45.12: ref %i32 = array_index file.%a.var, %int_3.loc11_45
// CHECK:STDOUT:   %.loc11_45.13: init %i32 = initialize_from %.loc11_45.11 to %.loc11_45.12 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %impl.elem0.loc11_45.5: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc11_45.5: <bound method> = bound_method %int_5.loc11_32, %impl.elem0.loc11_45.5 [concrete = constants.%Convert.bound.4e6]
// CHECK:STDOUT:   %specific_fn.loc11_45.5: <specific function> = specific_function %bound_method.loc11_45.5, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn.ba9]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.5: init %i32 = call %specific_fn.loc11_45.5(%int_5.loc11_32) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc11_45.14: init %i32 = converted %int_5.loc11_32, %int.convert_checked.loc11_45.5 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %int_4.loc11_45: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %.loc11_45.15: ref %i32 = array_index file.%a.var, %int_4.loc11_45
// CHECK:STDOUT:   %.loc11_45.16: init %i32 = initialize_from %.loc11_45.14 to %.loc11_45.15 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %impl.elem0.loc11_45.6: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc11_45.6: <bound method> = bound_method %int_6.loc11_35, %impl.elem0.loc11_45.6 [concrete = constants.%Convert.bound.ce9]
// CHECK:STDOUT:   %specific_fn.loc11_45.6: <specific function> = specific_function %bound_method.loc11_45.6, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn.631]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.6: init %i32 = call %specific_fn.loc11_45.6(%int_6.loc11_35) [concrete = constants.%int_6.e56]
// CHECK:STDOUT:   %.loc11_45.17: init %i32 = converted %int_6.loc11_35, %int.convert_checked.loc11_45.6 [concrete = constants.%int_6.e56]
// CHECK:STDOUT:   %int_5.loc11_45: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %.loc11_45.18: ref %i32 = array_index file.%a.var, %int_5.loc11_45
// CHECK:STDOUT:   %.loc11_45.19: init %i32 = initialize_from %.loc11_45.17 to %.loc11_45.18 [concrete = constants.%int_6.e56]
// CHECK:STDOUT:   %impl.elem0.loc11_45.7: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc11_45.7: <bound method> = bound_method %int_7.loc11_38, %impl.elem0.loc11_45.7 [concrete = constants.%Convert.bound.208]
// CHECK:STDOUT:   %specific_fn.loc11_45.7: <specific function> = specific_function %bound_method.loc11_45.7, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn.c12]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.7: init %i32 = call %specific_fn.loc11_45.7(%int_7.loc11_38) [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:   %.loc11_45.20: init %i32 = converted %int_7.loc11_38, %int.convert_checked.loc11_45.7 [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:   %int_6.loc11_45: Core.IntLiteral = int_value 6 [concrete = constants.%int_6.462]
// CHECK:STDOUT:   %.loc11_45.21: ref %i32 = array_index file.%a.var, %int_6.loc11_45
// CHECK:STDOUT:   %.loc11_45.22: init %i32 = initialize_from %.loc11_45.20 to %.loc11_45.21 [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:   %impl.elem0.loc11_45.8: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc11_45.8: <bound method> = bound_method %int_8.loc11_41, %impl.elem0.loc11_45.8 [concrete = constants.%Convert.bound.e09]
// CHECK:STDOUT:   %specific_fn.loc11_45.8: <specific function> = specific_function %bound_method.loc11_45.8, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn.e0d]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.8: init %i32 = call %specific_fn.loc11_45.8(%int_8.loc11_41) [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %.loc11_45.23: init %i32 = converted %int_8.loc11_41, %int.convert_checked.loc11_45.8 [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %int_7.loc11_45: Core.IntLiteral = int_value 7 [concrete = constants.%int_7.29f]
// CHECK:STDOUT:   %.loc11_45.24: ref %i32 = array_index file.%a.var, %int_7.loc11_45
// CHECK:STDOUT:   %.loc11_45.25: init %i32 = initialize_from %.loc11_45.23 to %.loc11_45.24 [concrete = constants.%int_8.98c]
// CHECK:STDOUT:   %impl.elem0.loc11_45.9: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc11_45.9: <bound method> = bound_method %int_9, %impl.elem0.loc11_45.9 [concrete = constants.%Convert.bound.9e2]
// CHECK:STDOUT:   %specific_fn.loc11_45.9: <specific function> = specific_function %bound_method.loc11_45.9, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn.b02]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.9: init %i32 = call %specific_fn.loc11_45.9(%int_9) [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   %.loc11_45.26: init %i32 = converted %int_9, %int.convert_checked.loc11_45.9 [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   %int_8.loc11_45: Core.IntLiteral = int_value 8 [concrete = constants.%int_8.b85]
// CHECK:STDOUT:   %.loc11_45.27: ref %i32 = array_index file.%a.var, %int_8.loc11_45
// CHECK:STDOUT:   %.loc11_45.28: init %i32 = initialize_from %.loc11_45.26 to %.loc11_45.27 [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   %.loc11_45.29: init %array_type = array_init (%.loc11_45.4, %.loc11_45.7, %.loc11_45.10, %.loc11_45.13, %.loc11_45.16, %.loc11_45.19, %.loc11_45.22, %.loc11_45.25, %.loc11_45.28) to file.%a.var [concrete = constants.%array]
// CHECK:STDOUT:   %.loc11_1: init %array_type = converted %.loc11_45.1, %.loc11_45.29 [concrete = constants.%array]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
