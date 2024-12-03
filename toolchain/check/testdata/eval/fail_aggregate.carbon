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
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int.32 [template]
// CHECK:STDOUT:   %int.1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.1: type = array_type %int.1.1, %i32 [template]
// CHECK:STDOUT:   %int.0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (Core.IntLiteral) [template]
// CHECK:STDOUT:   %int.5.1: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %int.7.1: Core.IntLiteral = int_value 7 [template]
// CHECK:STDOUT:   %int.9.1: Core.IntLiteral = int_value 9 [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %int.4: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %.3: type = array_type %int.4, %i32 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int.32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.28: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.29: <bound method> = bound_method %int.5.1, %Convert.14 [template]
// CHECK:STDOUT:   %.30: <specific function> = specific_function %.29, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.5.2: %i32 = int_value 5 [template]
// CHECK:STDOUT:   %.31: <bound method> = bound_method %int.7.1, %Convert.14 [template]
// CHECK:STDOUT:   %.32: <specific function> = specific_function %.31, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.7.2: %i32 = int_value 7 [template]
// CHECK:STDOUT:   %int.2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.33: <bound method> = bound_method %int.1.1, %Convert.14 [template]
// CHECK:STDOUT:   %.34: <specific function> = specific_function %.33, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %int.3: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.35: <bound method> = bound_method %int.9.1, %Convert.14 [template]
// CHECK:STDOUT:   %.36: <specific function> = specific_function %.35, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.9.2: %i32 = int_value 9 [template]
// CHECK:STDOUT:   %array: %.3 = tuple_value (%int.5.2, %int.7.2, %int.1.2, %int.9.2) [template]
// CHECK:STDOUT:   %.37: <bound method> = bound_method %int.2.1, %Convert.14 [template]
// CHECK:STDOUT:   %.38: <specific function> = specific_function %.37, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.2.2: %i32 = int_value 2 [template]
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
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%int.32) [template = constants.%i32]
// CHECK:STDOUT:   %int.1: Core.IntLiteral = int_value 1 [template = constants.%int.1.1]
// CHECK:STDOUT:   %.loc16_19.1: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc16_19.2: type = converted %int.make_type_signed, %.loc16_19.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc16_25: type = array_type %int.1, %i32 [template = constants.%.1]
// CHECK:STDOUT:   %array_index.var: ref %.1 = var array_index
// CHECK:STDOUT:   %array_index: ref %.1 = bind_name array_index, %array_index.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.0.loc16_30: Core.IntLiteral = int_value 0 [template = constants.%int.0]
// CHECK:STDOUT:   %.loc16_32: %tuple.type.1 = tuple_literal (%int.0.loc16_30)
// CHECK:STDOUT:   %int.32.loc16_38: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc16_38: init type = call constants.%Int(%int.32.loc16_38) [template = constants.%i32]
// CHECK:STDOUT:   %int.5: Core.IntLiteral = int_value 5 [template = constants.%int.5.1]
// CHECK:STDOUT:   %int.7: Core.IntLiteral = int_value 7 [template = constants.%int.7.1]
// CHECK:STDOUT:   %int.1.loc16_51: Core.IntLiteral = int_value 1 [template = constants.%int.1.1]
// CHECK:STDOUT:   %int.9: Core.IntLiteral = int_value 9 [template = constants.%int.9.1]
// CHECK:STDOUT:   %.loc16_55.1: %tuple.type.2 = tuple_literal (%int.5, %int.7, %int.1.loc16_51, %int.9)
// CHECK:STDOUT:   %int.32.loc16_61: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc16_61: init type = call constants.%Int(%int.32.loc16_61) [template = constants.%i32]
// CHECK:STDOUT:   %int.4: Core.IntLiteral = int_value 4 [template = constants.%int.4]
// CHECK:STDOUT:   %.loc16_61.1: type = value_of_initializer %int.make_type_signed.loc16_61 [template = constants.%i32]
// CHECK:STDOUT:   %.loc16_61.2: type = converted %int.make_type_signed.loc16_61, %.loc16_61.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc16_67: type = array_type %int.4, %i32 [template = constants.%.3]
// CHECK:STDOUT:   %.loc16_55.2: %Convert.type.2 = interface_witness_access constants.%.28, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc16_55.3: <bound method> = bound_method %int.5, %.loc16_55.2 [template = constants.%.29]
// CHECK:STDOUT:   %.loc16_55.4: <specific function> = specific_function %.loc16_55.3, @Convert.2(constants.%int.32) [template = constants.%.30]
// CHECK:STDOUT:   %int.convert_checked.loc16_55.1: init %i32 = call %.loc16_55.4(%int.5) [template = constants.%int.5.2]
// CHECK:STDOUT:   %.loc16_55.5: init %i32 = converted %int.5, %int.convert_checked.loc16_55.1 [template = constants.%int.5.2]
// CHECK:STDOUT:   %.loc16_55.6: ref %.3 = temporary_storage
// CHECK:STDOUT:   %int.0.loc16_55: Core.IntLiteral = int_value 0 [template = constants.%int.0]
// CHECK:STDOUT:   %.loc16_55.7: ref %i32 = array_index %.loc16_55.6, %int.0.loc16_55
// CHECK:STDOUT:   %.loc16_55.8: init %i32 = initialize_from %.loc16_55.5 to %.loc16_55.7 [template = constants.%int.5.2]
// CHECK:STDOUT:   %.loc16_55.9: %Convert.type.2 = interface_witness_access constants.%.28, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc16_55.10: <bound method> = bound_method %int.7, %.loc16_55.9 [template = constants.%.31]
// CHECK:STDOUT:   %.loc16_55.11: <specific function> = specific_function %.loc16_55.10, @Convert.2(constants.%int.32) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc16_55.2: init %i32 = call %.loc16_55.11(%int.7) [template = constants.%int.7.2]
// CHECK:STDOUT:   %.loc16_55.12: init %i32 = converted %int.7, %int.convert_checked.loc16_55.2 [template = constants.%int.7.2]
// CHECK:STDOUT:   %int.1.loc16_55: Core.IntLiteral = int_value 1 [template = constants.%int.1.1]
// CHECK:STDOUT:   %.loc16_55.13: ref %i32 = array_index %.loc16_55.6, %int.1.loc16_55
// CHECK:STDOUT:   %.loc16_55.14: init %i32 = initialize_from %.loc16_55.12 to %.loc16_55.13 [template = constants.%int.7.2]
// CHECK:STDOUT:   %.loc16_55.15: %Convert.type.2 = interface_witness_access constants.%.28, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc16_55.16: <bound method> = bound_method %int.1.loc16_51, %.loc16_55.15 [template = constants.%.33]
// CHECK:STDOUT:   %.loc16_55.17: <specific function> = specific_function %.loc16_55.16, @Convert.2(constants.%int.32) [template = constants.%.34]
// CHECK:STDOUT:   %int.convert_checked.loc16_55.3: init %i32 = call %.loc16_55.17(%int.1.loc16_51) [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc16_55.18: init %i32 = converted %int.1.loc16_51, %int.convert_checked.loc16_55.3 [template = constants.%int.1.2]
// CHECK:STDOUT:   %int.2.loc16_55: Core.IntLiteral = int_value 2 [template = constants.%int.2.1]
// CHECK:STDOUT:   %.loc16_55.19: ref %i32 = array_index %.loc16_55.6, %int.2.loc16_55
// CHECK:STDOUT:   %.loc16_55.20: init %i32 = initialize_from %.loc16_55.18 to %.loc16_55.19 [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc16_55.21: %Convert.type.2 = interface_witness_access constants.%.28, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc16_55.22: <bound method> = bound_method %int.9, %.loc16_55.21 [template = constants.%.35]
// CHECK:STDOUT:   %.loc16_55.23: <specific function> = specific_function %.loc16_55.22, @Convert.2(constants.%int.32) [template = constants.%.36]
// CHECK:STDOUT:   %int.convert_checked.loc16_55.4: init %i32 = call %.loc16_55.23(%int.9) [template = constants.%int.9.2]
// CHECK:STDOUT:   %.loc16_55.24: init %i32 = converted %int.9, %int.convert_checked.loc16_55.4 [template = constants.%int.9.2]
// CHECK:STDOUT:   %int.3: Core.IntLiteral = int_value 3 [template = constants.%int.3]
// CHECK:STDOUT:   %.loc16_55.25: ref %i32 = array_index %.loc16_55.6, %int.3
// CHECK:STDOUT:   %.loc16_55.26: init %i32 = initialize_from %.loc16_55.24 to %.loc16_55.25 [template = constants.%int.9.2]
// CHECK:STDOUT:   %.loc16_55.27: init %.3 = array_init (%.loc16_55.8, %.loc16_55.14, %.loc16_55.20, %.loc16_55.26) to %.loc16_55.6 [template = constants.%array]
// CHECK:STDOUT:   %.loc16_57.1: init %.3 = converted %.loc16_55.1, %.loc16_55.27 [template = constants.%array]
// CHECK:STDOUT:   %int.2.loc16_70: Core.IntLiteral = int_value 2 [template = constants.%int.2.1]
// CHECK:STDOUT:   %.loc16_57.2: ref %.3 = temporary %.loc16_55.6, %.loc16_57.1
// CHECK:STDOUT:   %.loc16_70.1: %Convert.type.2 = interface_witness_access constants.%.28, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc16_70.2: <bound method> = bound_method %int.2.loc16_70, %.loc16_70.1 [template = constants.%.37]
// CHECK:STDOUT:   %.loc16_70.3: <specific function> = specific_function %.loc16_70.2, @Convert.2(constants.%int.32) [template = constants.%.38]
// CHECK:STDOUT:   %int.convert_checked.loc16_70: init %i32 = call %.loc16_70.3(%int.2.loc16_70) [template = constants.%int.2.2]
// CHECK:STDOUT:   %.loc16_70.4: %i32 = value_of_initializer %int.convert_checked.loc16_70 [template = constants.%int.2.2]
// CHECK:STDOUT:   %.loc16_70.5: %i32 = converted %int.2.loc16_70, %.loc16_70.4 [template = constants.%int.2.2]
// CHECK:STDOUT:   %.loc16_71.1: ref %i32 = array_index %.loc16_57.2, %.loc16_70.5
// CHECK:STDOUT:   %.loc16_71.2: %i32 = bind_value %.loc16_71.1
// CHECK:STDOUT:   %.loc16_38.1: type = value_of_initializer %int.make_type_signed.loc16_38 [template = constants.%i32]
// CHECK:STDOUT:   %.loc16_38.2: type = converted %int.make_type_signed.loc16_38, %.loc16_38.1 [template = constants.%i32]
// CHECK:STDOUT:   assign file.%array_index.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
