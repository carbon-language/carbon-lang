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
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.3: type = array_type %.2, %i32 [template]
// CHECK:STDOUT:   %.5: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (Core.IntLiteral) [template]
// CHECK:STDOUT:   %.6: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %.7: Core.IntLiteral = int_value 7 [template]
// CHECK:STDOUT:   %.8: Core.IntLiteral = int_value 9 [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %.9: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %.10: type = array_type %.9, %i32 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.35: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.36: <bound method> = bound_method %.6, %Convert.14 [template]
// CHECK:STDOUT:   %.37: <specific function> = specific_function %.36, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.38: %i32 = int_value 5 [template]
// CHECK:STDOUT:   %.39: <bound method> = bound_method %.7, %Convert.14 [template]
// CHECK:STDOUT:   %.40: <specific function> = specific_function %.39, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.41: %i32 = int_value 7 [template]
// CHECK:STDOUT:   %.42: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.43: <bound method> = bound_method %.2, %Convert.14 [template]
// CHECK:STDOUT:   %.44: <specific function> = specific_function %.43, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.45: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %.46: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.47: <bound method> = bound_method %.8, %Convert.14 [template]
// CHECK:STDOUT:   %.48: <specific function> = specific_function %.47, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.49: %i32 = int_value 9 [template]
// CHECK:STDOUT:   %array: %.10 = tuple_value (%.38, %.41, %.45, %.49) [template]
// CHECK:STDOUT:   %.50: <bound method> = bound_method %.42, %Convert.14 [template]
// CHECK:STDOUT:   %.51: <specific function> = specific_function %.50, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.52: %i32 = int_value 2 [template]
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
// CHECK:STDOUT:   %.loc16_19.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%.loc16_19.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc16_24: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc16_19.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc16_19.3: type = converted %int.make_type_signed, %.loc16_19.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc16_25: type = array_type %.loc16_24, %i32 [template = constants.%.3]
// CHECK:STDOUT:   %array_index.var: ref %.3 = var array_index
// CHECK:STDOUT:   %array_index: ref %.3 = bind_name array_index, %array_index.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc16_30: Core.IntLiteral = int_value 0 [template = constants.%.5]
// CHECK:STDOUT:   %.loc16_32: %tuple.type.1 = tuple_literal (%.loc16_30)
// CHECK:STDOUT:   %.loc16_38.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc16_38: init type = call constants.%Int(%.loc16_38.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc16_45: Core.IntLiteral = int_value 5 [template = constants.%.6]
// CHECK:STDOUT:   %.loc16_48: Core.IntLiteral = int_value 7 [template = constants.%.7]
// CHECK:STDOUT:   %.loc16_51: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc16_54: Core.IntLiteral = int_value 9 [template = constants.%.8]
// CHECK:STDOUT:   %.loc16_55.1: %tuple.type.2 = tuple_literal (%.loc16_45, %.loc16_48, %.loc16_51, %.loc16_54)
// CHECK:STDOUT:   %.loc16_61.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc16_61: init type = call constants.%Int(%.loc16_61.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc16_66: Core.IntLiteral = int_value 4 [template = constants.%.9]
// CHECK:STDOUT:   %.loc16_61.2: type = value_of_initializer %int.make_type_signed.loc16_61 [template = constants.%i32]
// CHECK:STDOUT:   %.loc16_61.3: type = converted %int.make_type_signed.loc16_61, %.loc16_61.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc16_67: type = array_type %.loc16_66, %i32 [template = constants.%.10]
// CHECK:STDOUT:   %.loc16_55.2: %Convert.type.2 = interface_witness_access constants.%.35, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc16_55.3: <bound method> = bound_method %.loc16_45, %.loc16_55.2 [template = constants.%.36]
// CHECK:STDOUT:   %.loc16_55.4: <specific function> = specific_function %.loc16_55.3, @Convert.2(constants.%.1) [template = constants.%.37]
// CHECK:STDOUT:   %int.convert_checked.loc16_55.1: init %i32 = call %.loc16_55.4(%.loc16_45) [template = constants.%.38]
// CHECK:STDOUT:   %.loc16_55.5: init %i32 = converted %.loc16_45, %int.convert_checked.loc16_55.1 [template = constants.%.38]
// CHECK:STDOUT:   %.loc16_55.6: ref %.10 = temporary_storage
// CHECK:STDOUT:   %.loc16_55.7: Core.IntLiteral = int_value 0 [template = constants.%.5]
// CHECK:STDOUT:   %.loc16_55.8: ref %i32 = array_index %.loc16_55.6, %.loc16_55.7
// CHECK:STDOUT:   %.loc16_55.9: init %i32 = initialize_from %.loc16_55.5 to %.loc16_55.8 [template = constants.%.38]
// CHECK:STDOUT:   %.loc16_55.10: %Convert.type.2 = interface_witness_access constants.%.35, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc16_55.11: <bound method> = bound_method %.loc16_48, %.loc16_55.10 [template = constants.%.39]
// CHECK:STDOUT:   %.loc16_55.12: <specific function> = specific_function %.loc16_55.11, @Convert.2(constants.%.1) [template = constants.%.40]
// CHECK:STDOUT:   %int.convert_checked.loc16_55.2: init %i32 = call %.loc16_55.12(%.loc16_48) [template = constants.%.41]
// CHECK:STDOUT:   %.loc16_55.13: init %i32 = converted %.loc16_48, %int.convert_checked.loc16_55.2 [template = constants.%.41]
// CHECK:STDOUT:   %.loc16_55.14: Core.IntLiteral = int_value 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc16_55.15: ref %i32 = array_index %.loc16_55.6, %.loc16_55.14
// CHECK:STDOUT:   %.loc16_55.16: init %i32 = initialize_from %.loc16_55.13 to %.loc16_55.15 [template = constants.%.41]
// CHECK:STDOUT:   %.loc16_55.17: %Convert.type.2 = interface_witness_access constants.%.35, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc16_55.18: <bound method> = bound_method %.loc16_51, %.loc16_55.17 [template = constants.%.43]
// CHECK:STDOUT:   %.loc16_55.19: <specific function> = specific_function %.loc16_55.18, @Convert.2(constants.%.1) [template = constants.%.44]
// CHECK:STDOUT:   %int.convert_checked.loc16_55.3: init %i32 = call %.loc16_55.19(%.loc16_51) [template = constants.%.45]
// CHECK:STDOUT:   %.loc16_55.20: init %i32 = converted %.loc16_51, %int.convert_checked.loc16_55.3 [template = constants.%.45]
// CHECK:STDOUT:   %.loc16_55.21: Core.IntLiteral = int_value 2 [template = constants.%.42]
// CHECK:STDOUT:   %.loc16_55.22: ref %i32 = array_index %.loc16_55.6, %.loc16_55.21
// CHECK:STDOUT:   %.loc16_55.23: init %i32 = initialize_from %.loc16_55.20 to %.loc16_55.22 [template = constants.%.45]
// CHECK:STDOUT:   %.loc16_55.24: %Convert.type.2 = interface_witness_access constants.%.35, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc16_55.25: <bound method> = bound_method %.loc16_54, %.loc16_55.24 [template = constants.%.47]
// CHECK:STDOUT:   %.loc16_55.26: <specific function> = specific_function %.loc16_55.25, @Convert.2(constants.%.1) [template = constants.%.48]
// CHECK:STDOUT:   %int.convert_checked.loc16_55.4: init %i32 = call %.loc16_55.26(%.loc16_54) [template = constants.%.49]
// CHECK:STDOUT:   %.loc16_55.27: init %i32 = converted %.loc16_54, %int.convert_checked.loc16_55.4 [template = constants.%.49]
// CHECK:STDOUT:   %.loc16_55.28: Core.IntLiteral = int_value 3 [template = constants.%.46]
// CHECK:STDOUT:   %.loc16_55.29: ref %i32 = array_index %.loc16_55.6, %.loc16_55.28
// CHECK:STDOUT:   %.loc16_55.30: init %i32 = initialize_from %.loc16_55.27 to %.loc16_55.29 [template = constants.%.49]
// CHECK:STDOUT:   %.loc16_55.31: init %.10 = array_init (%.loc16_55.9, %.loc16_55.16, %.loc16_55.23, %.loc16_55.30) to %.loc16_55.6 [template = constants.%array]
// CHECK:STDOUT:   %.loc16_57.1: init %.10 = converted %.loc16_55.1, %.loc16_55.31 [template = constants.%array]
// CHECK:STDOUT:   %.loc16_70.1: Core.IntLiteral = int_value 2 [template = constants.%.42]
// CHECK:STDOUT:   %.loc16_57.2: ref %.10 = temporary %.loc16_55.6, %.loc16_57.1
// CHECK:STDOUT:   %.loc16_70.2: %Convert.type.2 = interface_witness_access constants.%.35, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc16_70.3: <bound method> = bound_method %.loc16_70.1, %.loc16_70.2 [template = constants.%.50]
// CHECK:STDOUT:   %.loc16_70.4: <specific function> = specific_function %.loc16_70.3, @Convert.2(constants.%.1) [template = constants.%.51]
// CHECK:STDOUT:   %int.convert_checked.loc16_70: init %i32 = call %.loc16_70.4(%.loc16_70.1) [template = constants.%.52]
// CHECK:STDOUT:   %.loc16_70.5: %i32 = value_of_initializer %int.convert_checked.loc16_70 [template = constants.%.52]
// CHECK:STDOUT:   %.loc16_70.6: %i32 = converted %.loc16_70.1, %.loc16_70.5 [template = constants.%.52]
// CHECK:STDOUT:   %.loc16_71.1: ref %i32 = array_index %.loc16_57.2, %.loc16_70.6
// CHECK:STDOUT:   %.loc16_71.2: %i32 = bind_value %.loc16_71.1
// CHECK:STDOUT:   %.loc16_38.2: type = value_of_initializer %int.make_type_signed.loc16_38 [template = constants.%i32]
// CHECK:STDOUT:   %.loc16_38.3: type = converted %int.make_type_signed.loc16_38, %.loc16_38.2 [template = constants.%i32]
// CHECK:STDOUT:   assign file.%array_index.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
