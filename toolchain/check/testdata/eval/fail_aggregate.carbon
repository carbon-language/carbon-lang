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

// CHECK:STDERR: fail_aggregate.carbon:[[@LINE+4]]:43: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var array_index: [i32; 1] = (0,) as [i32; ((5, 7, 1, 9) as [i32; 4])[2]];
// CHECK:STDERR:                                           ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var array_index: [i32; 1] = (0,) as [i32; ((5, 7, 1, 9) as [i32; 4])[2]];

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
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.1(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.1(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, %impl_witness.d39 [concrete]
// CHECK:STDOUT:   %.a0b: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.4e6: <bound method> = bound_method %int_5.64b, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn.ba9: <specific function> = specific_function %Convert.bound.4e6, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %int_5.0f6: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %Convert.bound.208: <bound method> = bound_method %int_7.29f, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn.c12: <specific function> = specific_function %Convert.bound.208, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %int_7.0b1: %i32 = int_value 7 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Convert.bound.ab5: <bound method> = bound_method %int_1.5b8, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn.70c: <specific function> = specific_function %Convert.bound.ab5, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %int_1.5d2: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Convert.bound.9e2: <bound method> = bound_method %int_9.988, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn.b02: <specific function> = specific_function %Convert.bound.9e2, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %int_9.f88: %i32 = int_value 9 [concrete]
// CHECK:STDOUT:   %array: %array_type.f32 = tuple_value (%int_5.0f6, %int_7.0b1, %int_1.5d2, %int_9.f88) [concrete]
// CHECK:STDOUT:   %Convert.bound.ef9: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn.787: <specific function> = specific_function %Convert.bound.ef9, @Convert.2(%int_32) [concrete]
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
// CHECK:STDOUT:   %.loc17_25: type = splice_block %array_type [concrete = constants.%array_type.0cb] {
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
// CHECK:STDOUT:   %int_0.loc17_30: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc17_32: %tuple.type.985 = tuple_literal (%int_0.loc17_30)
// CHECK:STDOUT:   %int_32.loc17_38: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc17_38: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [concrete = constants.%int_7.29f]
// CHECK:STDOUT:   %int_1.loc17_51: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [concrete = constants.%int_9.988]
// CHECK:STDOUT:   %.loc17_55.1: %tuple.type.d46 = tuple_literal (%int_5, %int_7, %int_1.loc17_51, %int_9)
// CHECK:STDOUT:   %int_32.loc17_61: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc17_61: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4]
// CHECK:STDOUT:   %array_type: type = array_type %int_4, %i32 [concrete = constants.%array_type.f32]
// CHECK:STDOUT:   %impl.elem0.loc17_55.1: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc17_55.1: <bound method> = bound_method %int_5, %impl.elem0.loc17_55.1 [concrete = constants.%Convert.bound.4e6]
// CHECK:STDOUT:   %specific_fn.loc17_55.1: <specific function> = specific_function %bound_method.loc17_55.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn.ba9]
// CHECK:STDOUT:   %int.convert_checked.loc17_55.1: init %i32 = call %specific_fn.loc17_55.1(%int_5) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc17_55.2: init %i32 = converted %int_5, %int.convert_checked.loc17_55.1 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc17_55.3: ref %array_type.f32 = temporary_storage
// CHECK:STDOUT:   %int_0.loc17_55: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc17_55.4: ref %i32 = array_index %.loc17_55.3, %int_0.loc17_55
// CHECK:STDOUT:   %.loc17_55.5: init %i32 = initialize_from %.loc17_55.2 to %.loc17_55.4 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %impl.elem0.loc17_55.2: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc17_55.2: <bound method> = bound_method %int_7, %impl.elem0.loc17_55.2 [concrete = constants.%Convert.bound.208]
// CHECK:STDOUT:   %specific_fn.loc17_55.2: <specific function> = specific_function %bound_method.loc17_55.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn.c12]
// CHECK:STDOUT:   %int.convert_checked.loc17_55.2: init %i32 = call %specific_fn.loc17_55.2(%int_7) [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:   %.loc17_55.6: init %i32 = converted %int_7, %int.convert_checked.loc17_55.2 [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:   %int_1.loc17_55: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc17_55.7: ref %i32 = array_index %.loc17_55.3, %int_1.loc17_55
// CHECK:STDOUT:   %.loc17_55.8: init %i32 = initialize_from %.loc17_55.6 to %.loc17_55.7 [concrete = constants.%int_7.0b1]
// CHECK:STDOUT:   %impl.elem0.loc17_55.3: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc17_55.3: <bound method> = bound_method %int_1.loc17_51, %impl.elem0.loc17_55.3 [concrete = constants.%Convert.bound.ab5]
// CHECK:STDOUT:   %specific_fn.loc17_55.3: <specific function> = specific_function %bound_method.loc17_55.3, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn.70c]
// CHECK:STDOUT:   %int.convert_checked.loc17_55.3: init %i32 = call %specific_fn.loc17_55.3(%int_1.loc17_51) [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %.loc17_55.9: init %i32 = converted %int_1.loc17_51, %int.convert_checked.loc17_55.3 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %int_2.loc17_55: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc17_55.10: ref %i32 = array_index %.loc17_55.3, %int_2.loc17_55
// CHECK:STDOUT:   %.loc17_55.11: init %i32 = initialize_from %.loc17_55.9 to %.loc17_55.10 [concrete = constants.%int_1.5d2]
// CHECK:STDOUT:   %impl.elem0.loc17_55.4: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc17_55.4: <bound method> = bound_method %int_9, %impl.elem0.loc17_55.4 [concrete = constants.%Convert.bound.9e2]
// CHECK:STDOUT:   %specific_fn.loc17_55.4: <specific function> = specific_function %bound_method.loc17_55.4, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn.b02]
// CHECK:STDOUT:   %int.convert_checked.loc17_55.4: init %i32 = call %specific_fn.loc17_55.4(%int_9) [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   %.loc17_55.12: init %i32 = converted %int_9, %int.convert_checked.loc17_55.4 [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:   %.loc17_55.13: ref %i32 = array_index %.loc17_55.3, %int_3
// CHECK:STDOUT:   %.loc17_55.14: init %i32 = initialize_from %.loc17_55.12 to %.loc17_55.13 [concrete = constants.%int_9.f88]
// CHECK:STDOUT:   %.loc17_55.15: init %array_type.f32 = array_init (%.loc17_55.5, %.loc17_55.8, %.loc17_55.11, %.loc17_55.14) to %.loc17_55.3 [concrete = constants.%array]
// CHECK:STDOUT:   %.loc17_57.1: init %array_type.f32 = converted %.loc17_55.1, %.loc17_55.15 [concrete = constants.%array]
// CHECK:STDOUT:   %int_2.loc17_70: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc17_57.2: ref %array_type.f32 = temporary %.loc17_55.3, %.loc17_57.1
// CHECK:STDOUT:   %int_32.loc17_71: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc17_71: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc17_70: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc17_70: <bound method> = bound_method %int_2.loc17_70, %impl.elem0.loc17_70 [concrete = constants.%Convert.bound.ef9]
// CHECK:STDOUT:   %specific_fn.loc17_70: <specific function> = specific_function %bound_method.loc17_70, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn.787]
// CHECK:STDOUT:   %int.convert_checked.loc17_70: init %i32 = call %specific_fn.loc17_70(%int_2.loc17_70) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc17_70.1: %i32 = value_of_initializer %int.convert_checked.loc17_70 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc17_70.2: %i32 = converted %int_2.loc17_70, %.loc17_70.1 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc17_71.1: ref %i32 = array_index %.loc17_57.2, %.loc17_70.2
// CHECK:STDOUT:   %.loc17_71.2: %i32 = bind_value %.loc17_71.1
// CHECK:STDOUT:   assign file.%array_index.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
