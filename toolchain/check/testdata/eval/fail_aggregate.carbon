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

// CHECK:STDERR: fail_aggregate.carbon:[[@LINE+3]]:43: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var array_index: [i32; 1] = (0,) as [i32; ((5, 7, 1, 9) as [i32; 4])[2]];
// CHECK:STDERR:                                           ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
var array_index: [i32; 1] = (0,) as [i32; ((5, 7, 1, 9) as [i32; 4])[2]];

// CHECK:STDOUT: --- fail_aggregate.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %array_type.b8e: type = array_type %int_1.5b8, %i32 [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %tuple.type.985: type = tuple_type (Core.IntLiteral) [template]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %int_7.29f: Core.IntLiteral = int_value 7 [template]
// CHECK:STDOUT:   %int_9.988: Core.IntLiteral = int_value 9 [template]
// CHECK:STDOUT:   %tuple.type.d46: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %array_type.8d1: type = array_type %int_4, %i32 [template]
// CHECK:STDOUT:   %Convert.type.cd1: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.5b0: <witness> = impl_witness (imports.%import_ref.723), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.466: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.925: %Convert.type.466 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.b33: <bound method> = bound_method %int_5.64b, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.89d: <specific function> = specific_function %Convert.bound.b33, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_5.70b: %i32 = int_value 5 [template]
// CHECK:STDOUT:   %Convert.bound.eba: <bound method> = bound_method %int_7.29f, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.785: <specific function> = specific_function %Convert.bound.eba, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_7.2da: %i32 = int_value 7 [template]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.bound.afd: <bound method> = bound_method %int_1.5b8, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.b73: <specific function> = specific_function %Convert.bound.afd, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.c60: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %Convert.bound.cc8: <bound method> = bound_method %int_9.988, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.b8b: <specific function> = specific_function %Convert.bound.cc8, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_9.82c: %i32 = int_value 9 [template]
// CHECK:STDOUT:   %array: %array_type.8d1 = tuple_value (%int_5.70b, %int_7.2da, %int_1.c60, %int_9.82c) [template]
// CHECK:STDOUT:   %Convert.bound.f0b: <bound method> = bound_method %int_2.ecc, %Convert.925 [template]
// CHECK:STDOUT:   %Convert.specific_fn.20e: <specific function> = specific_function %Convert.bound.f0b, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.166: %i32 = int_value 2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.187
// CHECK:STDOUT:     .ImplicitAs = %import_ref.a69
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .array_index = %array_index
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %array_index.var: ref %array_type.b8e = var array_index
// CHECK:STDOUT:   %array_index: ref %array_type.b8e = bind_name array_index, %array_index.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0.loc16_30: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc16_32: %tuple.type.985 = tuple_literal (%int_0.loc16_30)
// CHECK:STDOUT:   %int_32.loc16_38: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc16_38: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template = constants.%int_5.64b]
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [template = constants.%int_7.29f]
// CHECK:STDOUT:   %int_1.loc16_51: Core.IntLiteral = int_value 1 [template = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [template = constants.%int_9.988]
// CHECK:STDOUT:   %.loc16_55.1: %tuple.type.d46 = tuple_literal (%int_5, %int_7, %int_1.loc16_51, %int_9)
// CHECK:STDOUT:   %int_32.loc16_61: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc16_61: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [template = constants.%int_4]
// CHECK:STDOUT:   %array_type: type = array_type %int_4, %i32 [template = constants.%array_type.8d1]
// CHECK:STDOUT:   %impl.elem0.loc16_55.1: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc16_55.1: <bound method> = bound_method %int_5, %impl.elem0.loc16_55.1 [template = constants.%Convert.bound.b33]
// CHECK:STDOUT:   %Convert.specific_fn.loc16_55.1: <specific function> = specific_function %Convert.bound.loc16_55.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.89d]
// CHECK:STDOUT:   %int.convert_checked.loc16_55.1: init %i32 = call %Convert.specific_fn.loc16_55.1(%int_5) [template = constants.%int_5.70b]
// CHECK:STDOUT:   %.loc16_55.2: init %i32 = converted %int_5, %int.convert_checked.loc16_55.1 [template = constants.%int_5.70b]
// CHECK:STDOUT:   %.loc16_55.3: ref %array_type.8d1 = temporary_storage
// CHECK:STDOUT:   %int_0.loc16_55: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc16_55.4: ref %i32 = array_index %.loc16_55.3, %int_0.loc16_55
// CHECK:STDOUT:   %.loc16_55.5: init %i32 = initialize_from %.loc16_55.2 to %.loc16_55.4 [template = constants.%int_5.70b]
// CHECK:STDOUT:   %impl.elem0.loc16_55.2: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc16_55.2: <bound method> = bound_method %int_7, %impl.elem0.loc16_55.2 [template = constants.%Convert.bound.eba]
// CHECK:STDOUT:   %Convert.specific_fn.loc16_55.2: <specific function> = specific_function %Convert.bound.loc16_55.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.785]
// CHECK:STDOUT:   %int.convert_checked.loc16_55.2: init %i32 = call %Convert.specific_fn.loc16_55.2(%int_7) [template = constants.%int_7.2da]
// CHECK:STDOUT:   %.loc16_55.6: init %i32 = converted %int_7, %int.convert_checked.loc16_55.2 [template = constants.%int_7.2da]
// CHECK:STDOUT:   %int_1.loc16_55: Core.IntLiteral = int_value 1 [template = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc16_55.7: ref %i32 = array_index %.loc16_55.3, %int_1.loc16_55
// CHECK:STDOUT:   %.loc16_55.8: init %i32 = initialize_from %.loc16_55.6 to %.loc16_55.7 [template = constants.%int_7.2da]
// CHECK:STDOUT:   %impl.elem0.loc16_55.3: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc16_55.3: <bound method> = bound_method %int_1.loc16_51, %impl.elem0.loc16_55.3 [template = constants.%Convert.bound.afd]
// CHECK:STDOUT:   %Convert.specific_fn.loc16_55.3: <specific function> = specific_function %Convert.bound.loc16_55.3, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.b73]
// CHECK:STDOUT:   %int.convert_checked.loc16_55.3: init %i32 = call %Convert.specific_fn.loc16_55.3(%int_1.loc16_51) [template = constants.%int_1.c60]
// CHECK:STDOUT:   %.loc16_55.9: init %i32 = converted %int_1.loc16_51, %int.convert_checked.loc16_55.3 [template = constants.%int_1.c60]
// CHECK:STDOUT:   %int_2.loc16_55: Core.IntLiteral = int_value 2 [template = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc16_55.10: ref %i32 = array_index %.loc16_55.3, %int_2.loc16_55
// CHECK:STDOUT:   %.loc16_55.11: init %i32 = initialize_from %.loc16_55.9 to %.loc16_55.10 [template = constants.%int_1.c60]
// CHECK:STDOUT:   %impl.elem0.loc16_55.4: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc16_55.4: <bound method> = bound_method %int_9, %impl.elem0.loc16_55.4 [template = constants.%Convert.bound.cc8]
// CHECK:STDOUT:   %Convert.specific_fn.loc16_55.4: <specific function> = specific_function %Convert.bound.loc16_55.4, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.b8b]
// CHECK:STDOUT:   %int.convert_checked.loc16_55.4: init %i32 = call %Convert.specific_fn.loc16_55.4(%int_9) [template = constants.%int_9.82c]
// CHECK:STDOUT:   %.loc16_55.12: init %i32 = converted %int_9, %int.convert_checked.loc16_55.4 [template = constants.%int_9.82c]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template = constants.%int_3]
// CHECK:STDOUT:   %.loc16_55.13: ref %i32 = array_index %.loc16_55.3, %int_3
// CHECK:STDOUT:   %.loc16_55.14: init %i32 = initialize_from %.loc16_55.12 to %.loc16_55.13 [template = constants.%int_9.82c]
// CHECK:STDOUT:   %.loc16_55.15: init %array_type.8d1 = array_init (%.loc16_55.5, %.loc16_55.8, %.loc16_55.11, %.loc16_55.14) to %.loc16_55.3 [template = constants.%array]
// CHECK:STDOUT:   %.loc16_57.1: init %array_type.8d1 = converted %.loc16_55.1, %.loc16_55.15 [template = constants.%array]
// CHECK:STDOUT:   %int_2.loc16_70: Core.IntLiteral = int_value 2 [template = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc16_57.2: ref %array_type.8d1 = temporary %.loc16_55.3, %.loc16_57.1
// CHECK:STDOUT:   %int_32.loc16_71: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32.loc16_71: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc16_70: %Convert.type.cd1 = impl_witness_access constants.%impl_witness.5b0, element0 [template = constants.%Convert.925]
// CHECK:STDOUT:   %Convert.bound.loc16_70: <bound method> = bound_method %int_2.loc16_70, %impl.elem0.loc16_70 [template = constants.%Convert.bound.f0b]
// CHECK:STDOUT:   %Convert.specific_fn.loc16_70: <specific function> = specific_function %Convert.bound.loc16_70, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.20e]
// CHECK:STDOUT:   %int.convert_checked.loc16_70: init %i32 = call %Convert.specific_fn.loc16_70(%int_2.loc16_70) [template = constants.%int_2.166]
// CHECK:STDOUT:   %.loc16_70.1: %i32 = value_of_initializer %int.convert_checked.loc16_70 [template = constants.%int_2.166]
// CHECK:STDOUT:   %.loc16_70.2: %i32 = converted %int_2.loc16_70, %.loc16_70.1 [template = constants.%int_2.166]
// CHECK:STDOUT:   %.loc16_71.1: ref %i32 = array_index %.loc16_57.2, %.loc16_70.2
// CHECK:STDOUT:   %.loc16_71.2: %i32 = bind_value %.loc16_71.1
// CHECK:STDOUT:   assign file.%array_index.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
