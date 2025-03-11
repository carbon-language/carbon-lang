// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/eval/fail_aggregate.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/eval/fail_aggregate.carbon

// TODO: This creates an array temporary, which we don't yet support evaluating.

// CHECK:STDERR: fail_aggregate.carbon:[[@LINE+4]]:53: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var array_index: array(i32, 1) = (0,) as array(i32, ((5, 7, 1, 9) as array(i32, 4))[2]);
// CHECK:STDERR:                                                     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var array_index: array(i32, 1) = (0,) as array(i32, ((5, 7, 1, 9) as array(i32, 4))[2]);

// CHECK:STDOUT: --- fail_aggregate.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %array_type.0cb: type = array_type %int_1.5b8, %i32 [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %tuple.type.985: type = tuple_type (Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %int_7.29f: Core.IntLiteral = int_value 7 [concrete]
// CHECK:STDOUT:   %int_9.988: Core.IntLiteral = int_value 9 [concrete]
// CHECK:STDOUT:   %tuple.type.d46: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %array_type.f32: type = array_type %int_4, %i32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%impl_witness.d39) [concrete]
// CHECK:STDOUT:   %.be7: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.4e6: <bound method> = bound_method %int_5.64b, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.a25: <bound method> = bound_method %int_5.64b, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_5.0f6: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %Convert.bound.208: <bound method> = bound_method %int_7.29f, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.3bd: <bound method> = bound_method %int_7.29f, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_7.0b1: %i32 = int_value 7 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.9a1: <bound method> = bound_method %int_1.5b8, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Convert.bound.9e2: <bound method> = bound_method %int_9.988, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.cd3: <bound method> = bound_method %int_9.988, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_9.f88: %i32 = int_value 9 [concrete]
// CHECK:STDOUT:   %array: %array_type.f32 = tuple_value (%int_5.0f6, %int_7.0b1, %int_1.5d2, %int_9.f88) [concrete]
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
// CHECK:STDOUT:     .array_index = %array_index
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %array_index.patt: %array_type.0cb = binding_pattern array_index
// CHECK:STDOUT:     %.loc17_1: %array_type.0cb = var_pattern %array_index.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %array_index.var: ref %array_type.0cb = var array_index
// CHECK:STDOUT:   %.loc17_30: type = splice_block %array_type [concrete = constants.%array_type.0cb] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %array_type: type = array_type %int_1, %i32 [concrete = constants.%array_type.0cb]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %array_index: ref %array_type.0cb = bind_name array_index, %array_index.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0.loc17_35: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc17_37: %tuple.type.985 = tuple_literal (%int_0.loc17_35)
// CHECK:STDOUT:   %int_32.loc17_48: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc17_48: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [concrete = constants.%int_7.29f]
// CHECK:STDOUT:   %int_1.loc17_61: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete = constants.%int_9.988]
// CHECK:STDOUT:   %.loc17_65.1: %tuple.type.d46 = tuple_literal (%int_5, %int_7, %int_1.loc17_61, %int_9)
// CHECK:STDOUT:   %int_32.loc17_76: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc17_76: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:   %array_type: type = array_type %int_4, %i32 [concrete = constants.%array_type.f32]
// CHECK:STDOUT:   %impl.elem0.loc17_65.1: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc17_65.1: <bound method> = bound_method %int_5, %impl.elem0.loc17_65.1 [concrete = constants.%Convert.bound.4e6]
// CHECK:STDOUT:   %specific_fn.loc17_65.1: <specific function> = specific_function %impl.elem0.loc17_65.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc17_65.2: <bound method> = bound_method %int_5, %specific_fn.loc17_65.1 [concrete = constants.%bound_method.a25]
// CHECK:STDOUT:   %int.convert_checked.loc17_65.1: init %i32 = call %bound_method.loc17_65.2(%int_5) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc17_65.2: init %i32 = converted %int_5, %int.convert_checked.loc17_65.1 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc17_65.3: ref %array_type.f32 = temporary_storage
// CHECK:STDOUT:   %int_0.loc17_65: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc17_65.4: ref %i32 = array_index %.loc17_65.3, %int_0.loc17_65
// CHECK:STDOUT:   %.loc17_65.5: init %i32 = initialize_from %.loc17_65.2 to %.loc17_65.4 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %impl.elem0.loc17_65.2: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc17_65.3: <bound method> = bound_method %int_7, %impl.elem0.loc17_65.2 [concrete = constants.%Convert.bound.208]
// CHECK:STDOUT:   %specific_fn.loc17_65.2: <specific function> = specific_function %impl.elem0.loc17_65.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc17_65.4: <bound method> = bound_method %int_7, %specific_fn.loc17_65.2 [concrete = constants.%bound_method.3bd]
// CHECK:STDOUT:   %int.convert_checked.loc17_65.2: init %i32 = call %bound_method.loc17_65.4(%int_7) [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:   %.loc17_65.6: init %i32 = converted %int_7, %int.convert_checked.loc17_65.2 [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:   %int_1.loc17_65: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc17_65.7: ref %i32 = array_index %.loc17_65.3, %int_1.loc17_65
// CHECK:STDOUT:   %.loc17_65.8: init %i32 = initialize_from %.loc17_65.6 to %.loc17_65.7 [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:   %impl.elem0.loc17_65.3: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc17_65.5: <bound method> = bound_method %int_1.loc17_61, %impl.elem0.loc17_65.3 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc17_65.3: <specific function> = specific_function %impl.elem0.loc17_65.3, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc17_65.6: <bound method> = bound_method %int_1.loc17_61, %specific_fn.loc17_65.3 [concrete = constants.%bound_method.9a1]
// CHECK:STDOUT:   %int.convert_checked.loc17_65.3: init %i32 = call %bound_method.loc17_65.6(%int_1.loc17_61) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc17_65.9: init %i32 = converted %int_1.loc17_61, %int.convert_checked.loc17_65.3 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %int_2.loc17_65: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc17_65.10: ref %i32 = array_index %.loc17_65.3, %int_2.loc17_65
// CHECK:STDOUT:   %.loc17_65.11: init %i32 = initialize_from %.loc17_65.9 to %.loc17_65.10 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc17_65.4: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc17_65.7: <bound method> = bound_method %int_9, %impl.elem0.loc17_65.4 [concrete = constants.%Convert.bound.9e2]
// CHECK:STDOUT:   %specific_fn.loc17_65.4: <specific function> = specific_function %impl.elem0.loc17_65.4, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc17_65.8: <bound method> = bound_method %int_9, %specific_fn.loc17_65.4 [concrete = constants.%bound_method.cd3]
// CHECK:STDOUT:   %int.convert_checked.loc17_65.4: init %i32 = call %bound_method.loc17_65.8(%int_9) [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   %.loc17_65.12: init %i32 = converted %int_9, %int.convert_checked.loc17_65.4 [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:   %.loc17_65.13: ref %i32 = array_index %.loc17_65.3, %int_3
// CHECK:STDOUT:   %.loc17_65.14: init %i32 = initialize_from %.loc17_65.12 to %.loc17_65.13 [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   %.loc17_65.15: init %array_type.f32 = array_init (%.loc17_65.5, %.loc17_65.8, %.loc17_65.11, %.loc17_65.14) to %.loc17_65.3 [concrete = constants.%array]
// CHECK:STDOUT:   %.loc17_67.1: init %array_type.f32 = converted %.loc17_65.1, %.loc17_65.15 [concrete = constants.%array]
// CHECK:STDOUT:   %int_2.loc17_85: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc17_67.2: ref %array_type.f32 = temporary %.loc17_65.3, %.loc17_67.1
// CHECK:STDOUT:   %int_32.loc17_86: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc17_86: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc17_85: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc17_85.1: <bound method> = bound_method %int_2.loc17_85, %impl.elem0.loc17_85 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc17_85: <specific function> = specific_function %impl.elem0.loc17_85, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc17_85.2: <bound method> = bound_method %int_2.loc17_85, %specific_fn.loc17_85 [concrete = constants.%bound_method.b92]
// CHECK:STDOUT:   %int.convert_checked.loc17_85: init %i32 = call %bound_method.loc17_85.2(%int_2.loc17_85) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc17_85.1: %i32 = value_of_initializer %int.convert_checked.loc17_85 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc17_85.2: %i32 = converted %int_2.loc17_85, %.loc17_85.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc17_86.1: ref %i32 = array_index %.loc17_67.2, %.loc17_85.2
// CHECK:STDOUT:   %.loc17_86.2: %i32 = bind_value %.loc17_86.1
// CHECK:STDOUT:   assign file.%array_index.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
