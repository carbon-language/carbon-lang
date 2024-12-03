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
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %array_type.1: type = array_type %int_1.1, %i32 [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (Core.IntLiteral) [template]
// CHECK:STDOUT:   %int_5.1: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %int_7.1: Core.IntLiteral = int_value 7 [template]
// CHECK:STDOUT:   %int_9.1: Core.IntLiteral = int_value 9 [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %array_type.2: type = array_type %int_4, %i32 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_5.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_5.2: %i32 = int_value 5 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_7.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_7.2: %i32 = int_value 7 [template]
// CHECK:STDOUT:   %int_2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_1.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %Convert.bound.4: <bound method> = bound_method %int_9.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.4: <specific function> = specific_function %Convert.bound.4, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_9.2: %i32 = int_value 9 [template]
// CHECK:STDOUT:   %array: %array_type.2 = tuple_value (%int_5.2, %int_7.2, %int_1.2, %int_9.2) [template]
// CHECK:STDOUT:   %Convert.bound.5: <bound method> = bound_method %int_2.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.5: <specific function> = specific_function %Convert.bound.5, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %.loc16_19.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc16_19.2: type = converted %int.make_type_signed, %.loc16_19.1 [template = constants.%i32]
// CHECK:STDOUT:   %array_type: type = array_type %int_1, %i32 [template = constants.%array_type.1]
// CHECK:STDOUT:   %array_index.var: ref %array_type.1 = var array_index
// CHECK:STDOUT:   %array_index: ref %array_type.1 = bind_name array_index, %array_index.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_0.loc16_30: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc16_32: %tuple.type.1 = tuple_literal (%int_0.loc16_30)
// CHECK:STDOUT:   %int_32.loc16_38: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc16_38: init type = call constants.%Int(%int_32.loc16_38) [template = constants.%i32]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template = constants.%int_5.1]
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [template = constants.%int_7.1]
// CHECK:STDOUT:   %int_1.loc16_51: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [template = constants.%int_9.1]
// CHECK:STDOUT:   %.loc16_55.1: %tuple.type.2 = tuple_literal (%int_5, %int_7, %int_1.loc16_51, %int_9)
// CHECK:STDOUT:   %int_32.loc16_61: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc16_61: init type = call constants.%Int(%int_32.loc16_61) [template = constants.%i32]
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [template = constants.%int_4]
// CHECK:STDOUT:   %.loc16_61.1: type = value_of_initializer %int.make_type_signed.loc16_61 [template = constants.%i32]
// CHECK:STDOUT:   %.loc16_61.2: type = converted %int.make_type_signed.loc16_61, %.loc16_61.1 [template = constants.%i32]
// CHECK:STDOUT:   %array_type: type = array_type %int_4, %i32 [template = constants.%array_type.2]
// CHECK:STDOUT:   %impl.elem0.loc16_55.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc16_55.1: <bound method> = bound_method %int_5, %impl.elem0.loc16_55.1 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc16_55.1: <specific function> = specific_function %Convert.bound.loc16_55.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc16_55.1: init %i32 = call %Convert.specific_fn.loc16_55.1(%int_5) [template = constants.%int_5.2]
// CHECK:STDOUT:   %.loc16_55.2: init %i32 = converted %int_5, %int.convert_checked.loc16_55.1 [template = constants.%int_5.2]
// CHECK:STDOUT:   %.loc16_55.3: ref %array_type.2 = temporary_storage
// CHECK:STDOUT:   %int_0.loc16_55: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %.loc16_55.4: ref %i32 = array_index %.loc16_55.3, %int_0.loc16_55
// CHECK:STDOUT:   %.loc16_55.5: init %i32 = initialize_from %.loc16_55.2 to %.loc16_55.4 [template = constants.%int_5.2]
// CHECK:STDOUT:   %impl.elem0.loc16_55.2: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc16_55.2: <bound method> = bound_method %int_7, %impl.elem0.loc16_55.2 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc16_55.2: <specific function> = specific_function %Convert.bound.loc16_55.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc16_55.2: init %i32 = call %Convert.specific_fn.loc16_55.2(%int_7) [template = constants.%int_7.2]
// CHECK:STDOUT:   %.loc16_55.6: init %i32 = converted %int_7, %int.convert_checked.loc16_55.2 [template = constants.%int_7.2]
// CHECK:STDOUT:   %int_1.loc16_55: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %.loc16_55.7: ref %i32 = array_index %.loc16_55.3, %int_1.loc16_55
// CHECK:STDOUT:   %.loc16_55.8: init %i32 = initialize_from %.loc16_55.6 to %.loc16_55.7 [template = constants.%int_7.2]
// CHECK:STDOUT:   %impl.elem0.loc16_55.3: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc16_55.3: <bound method> = bound_method %int_1.loc16_51, %impl.elem0.loc16_55.3 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc16_55.3: <specific function> = specific_function %Convert.bound.loc16_55.3, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc16_55.3: init %i32 = call %Convert.specific_fn.loc16_55.3(%int_1.loc16_51) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc16_55.9: init %i32 = converted %int_1.loc16_51, %int.convert_checked.loc16_55.3 [template = constants.%int_1.2]
// CHECK:STDOUT:   %int_2.loc16_55: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %.loc16_55.10: ref %i32 = array_index %.loc16_55.3, %int_2.loc16_55
// CHECK:STDOUT:   %.loc16_55.11: init %i32 = initialize_from %.loc16_55.9 to %.loc16_55.10 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc16_55.4: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc16_55.4: <bound method> = bound_method %int_9, %impl.elem0.loc16_55.4 [template = constants.%Convert.bound.4]
// CHECK:STDOUT:   %Convert.specific_fn.loc16_55.4: <specific function> = specific_function %Convert.bound.loc16_55.4, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.4]
// CHECK:STDOUT:   %int.convert_checked.loc16_55.4: init %i32 = call %Convert.specific_fn.loc16_55.4(%int_9) [template = constants.%int_9.2]
// CHECK:STDOUT:   %.loc16_55.12: init %i32 = converted %int_9, %int.convert_checked.loc16_55.4 [template = constants.%int_9.2]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [template = constants.%int_3]
// CHECK:STDOUT:   %.loc16_55.13: ref %i32 = array_index %.loc16_55.3, %int_3
// CHECK:STDOUT:   %.loc16_55.14: init %i32 = initialize_from %.loc16_55.12 to %.loc16_55.13 [template = constants.%int_9.2]
// CHECK:STDOUT:   %.loc16_55.15: init %array_type.2 = array_init (%.loc16_55.5, %.loc16_55.8, %.loc16_55.11, %.loc16_55.14) to %.loc16_55.3 [template = constants.%array]
// CHECK:STDOUT:   %.loc16_57.1: init %array_type.2 = converted %.loc16_55.1, %.loc16_55.15 [template = constants.%array]
// CHECK:STDOUT:   %int_2.loc16_70: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %.loc16_57.2: ref %array_type.2 = temporary %.loc16_55.3, %.loc16_57.1
// CHECK:STDOUT:   %int_32.loc16_71: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc16_71: init type = call constants.%Int(%int_32.loc16_71) [template = constants.%i32]
// CHECK:STDOUT:   %.loc16_71.1: type = value_of_initializer %int.make_type_signed.loc16_71 [template = constants.%i32]
// CHECK:STDOUT:   %.loc16_71.2: type = converted %int.make_type_signed.loc16_71, %.loc16_71.1 [template = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc16_70: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc16_70: <bound method> = bound_method %int_2.loc16_70, %impl.elem0.loc16_70 [template = constants.%Convert.bound.5]
// CHECK:STDOUT:   %Convert.specific_fn.loc16_70: <specific function> = specific_function %Convert.bound.loc16_70, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.5]
// CHECK:STDOUT:   %int.convert_checked.loc16_70: init %i32 = call %Convert.specific_fn.loc16_70(%int_2.loc16_70) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc16_70.1: %i32 = value_of_initializer %int.convert_checked.loc16_70 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc16_70.2: %i32 = converted %int_2.loc16_70, %.loc16_70.1 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc16_71.3: ref %i32 = array_index %.loc16_57.2, %.loc16_70.2
// CHECK:STDOUT:   %.loc16_71.4: %i32 = bind_value %.loc16_71.3
// CHECK:STDOUT:   %.loc16_38.1: type = value_of_initializer %int.make_type_signed.loc16_38 [template = constants.%i32]
// CHECK:STDOUT:   %.loc16_38.2: type = converted %int.make_type_signed.loc16_38, %.loc16_38.1 [template = constants.%i32]
// CHECK:STDOUT:   assign file.%array_index.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
