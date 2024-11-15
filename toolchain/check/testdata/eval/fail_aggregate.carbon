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
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.2: type = array_type %.1, i32 [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (Core.IntLiteral) [template]
// CHECK:STDOUT:   %.5: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %.6: Core.IntLiteral = int_value 7 [template]
// CHECK:STDOUT:   %.7: Core.IntLiteral = int_value 9 [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %.8: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %.9: type = array_type %.8, i32 [template]
// CHECK:STDOUT:   %.11: i32 = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.35: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.36: <bound method> = bound_method %.5, %Convert.15 [template]
// CHECK:STDOUT:   %.37: i32 = int_value 5 [template]
// CHECK:STDOUT:   %.38: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.39: <bound method> = bound_method %.6, %Convert.15 [template]
// CHECK:STDOUT:   %.40: i32 = int_value 7 [template]
// CHECK:STDOUT:   %.41: i32 = int_value 2 [template]
// CHECK:STDOUT:   %.42: <bound method> = bound_method %.1, %Convert.15 [template]
// CHECK:STDOUT:   %.43: i32 = int_value 3 [template]
// CHECK:STDOUT:   %.44: <bound method> = bound_method %.7, %Convert.15 [template]
// CHECK:STDOUT:   %.45: i32 = int_value 9 [template]
// CHECK:STDOUT:   %array: %.9 = tuple_value (%.37, %.40, %.38, %.45) [template]
// CHECK:STDOUT:   %.46: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.47: <bound method> = bound_method %.46, %Convert.15 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
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
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc16_24: Core.IntLiteral = int_value 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc16_19.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc16_19.2: type = converted %int.make_type_32, %.loc16_19.1 [template = i32]
// CHECK:STDOUT:   %.loc16_25: type = array_type %.loc16_24, i32 [template = constants.%.2]
// CHECK:STDOUT:   %array_index.var: ref %.2 = var array_index
// CHECK:STDOUT:   %array_index: ref %.2 = bind_name array_index, %array_index.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc16_30: Core.IntLiteral = int_value 0 [template = constants.%.4]
// CHECK:STDOUT:   %.loc16_32: %tuple.type.1 = tuple_literal (%.loc16_30)
// CHECK:STDOUT:   %int.make_type_32.loc16_38: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc16_45: Core.IntLiteral = int_value 5 [template = constants.%.5]
// CHECK:STDOUT:   %.loc16_48: Core.IntLiteral = int_value 7 [template = constants.%.6]
// CHECK:STDOUT:   %.loc16_51: Core.IntLiteral = int_value 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc16_54: Core.IntLiteral = int_value 9 [template = constants.%.7]
// CHECK:STDOUT:   %.loc16_55.1: %tuple.type.2 = tuple_literal (%.loc16_45, %.loc16_48, %.loc16_51, %.loc16_54)
// CHECK:STDOUT:   %int.make_type_32.loc16_61: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc16_66: Core.IntLiteral = int_value 4 [template = constants.%.8]
// CHECK:STDOUT:   %.loc16_61.1: type = value_of_initializer %int.make_type_32.loc16_61 [template = i32]
// CHECK:STDOUT:   %.loc16_61.2: type = converted %int.make_type_32.loc16_61, %.loc16_61.1 [template = i32]
// CHECK:STDOUT:   %.loc16_67: type = array_type %.loc16_66, i32 [template = constants.%.9]
// CHECK:STDOUT:   %.loc16_55.2: %Convert.type.2 = interface_witness_access constants.%.35, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc16_55.3: <bound method> = bound_method %.loc16_45, %.loc16_55.2 [template = constants.%.36]
// CHECK:STDOUT:   %int.convert_checked.loc16_55.1: init i32 = call %.loc16_55.3(%.loc16_45) [template = constants.%.37]
// CHECK:STDOUT:   %.loc16_55.4: init i32 = converted %.loc16_45, %int.convert_checked.loc16_55.1 [template = constants.%.37]
// CHECK:STDOUT:   %.loc16_55.5: ref %.9 = temporary_storage
// CHECK:STDOUT:   %.loc16_55.6: i32 = int_value 0 [template = constants.%.11]
// CHECK:STDOUT:   %.loc16_55.7: ref i32 = array_index %.loc16_55.5, %.loc16_55.6
// CHECK:STDOUT:   %.loc16_55.8: init i32 = initialize_from %.loc16_55.4 to %.loc16_55.7 [template = constants.%.37]
// CHECK:STDOUT:   %.loc16_55.9: %Convert.type.2 = interface_witness_access constants.%.35, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc16_55.10: <bound method> = bound_method %.loc16_48, %.loc16_55.9 [template = constants.%.39]
// CHECK:STDOUT:   %int.convert_checked.loc16_55.2: init i32 = call %.loc16_55.10(%.loc16_48) [template = constants.%.40]
// CHECK:STDOUT:   %.loc16_55.11: init i32 = converted %.loc16_48, %int.convert_checked.loc16_55.2 [template = constants.%.40]
// CHECK:STDOUT:   %.loc16_55.12: i32 = int_value 1 [template = constants.%.38]
// CHECK:STDOUT:   %.loc16_55.13: ref i32 = array_index %.loc16_55.5, %.loc16_55.12
// CHECK:STDOUT:   %.loc16_55.14: init i32 = initialize_from %.loc16_55.11 to %.loc16_55.13 [template = constants.%.40]
// CHECK:STDOUT:   %.loc16_55.15: %Convert.type.2 = interface_witness_access constants.%.35, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc16_55.16: <bound method> = bound_method %.loc16_51, %.loc16_55.15 [template = constants.%.42]
// CHECK:STDOUT:   %int.convert_checked.loc16_55.3: init i32 = call %.loc16_55.16(%.loc16_51) [template = constants.%.38]
// CHECK:STDOUT:   %.loc16_55.17: init i32 = converted %.loc16_51, %int.convert_checked.loc16_55.3 [template = constants.%.38]
// CHECK:STDOUT:   %.loc16_55.18: i32 = int_value 2 [template = constants.%.41]
// CHECK:STDOUT:   %.loc16_55.19: ref i32 = array_index %.loc16_55.5, %.loc16_55.18
// CHECK:STDOUT:   %.loc16_55.20: init i32 = initialize_from %.loc16_55.17 to %.loc16_55.19 [template = constants.%.38]
// CHECK:STDOUT:   %.loc16_55.21: %Convert.type.2 = interface_witness_access constants.%.35, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc16_55.22: <bound method> = bound_method %.loc16_54, %.loc16_55.21 [template = constants.%.44]
// CHECK:STDOUT:   %int.convert_checked.loc16_55.4: init i32 = call %.loc16_55.22(%.loc16_54) [template = constants.%.45]
// CHECK:STDOUT:   %.loc16_55.23: init i32 = converted %.loc16_54, %int.convert_checked.loc16_55.4 [template = constants.%.45]
// CHECK:STDOUT:   %.loc16_55.24: i32 = int_value 3 [template = constants.%.43]
// CHECK:STDOUT:   %.loc16_55.25: ref i32 = array_index %.loc16_55.5, %.loc16_55.24
// CHECK:STDOUT:   %.loc16_55.26: init i32 = initialize_from %.loc16_55.23 to %.loc16_55.25 [template = constants.%.45]
// CHECK:STDOUT:   %.loc16_55.27: init %.9 = array_init (%.loc16_55.8, %.loc16_55.14, %.loc16_55.20, %.loc16_55.26) to %.loc16_55.5 [template = constants.%array]
// CHECK:STDOUT:   %.loc16_57.1: init %.9 = converted %.loc16_55.1, %.loc16_55.27 [template = constants.%array]
// CHECK:STDOUT:   %.loc16_70.1: Core.IntLiteral = int_value 2 [template = constants.%.46]
// CHECK:STDOUT:   %.loc16_57.2: ref %.9 = temporary %.loc16_55.5, %.loc16_57.1
// CHECK:STDOUT:   %.loc16_70.2: %Convert.type.2 = interface_witness_access constants.%.35, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc16_70.3: <bound method> = bound_method %.loc16_70.1, %.loc16_70.2 [template = constants.%.47]
// CHECK:STDOUT:   %int.convert_checked.loc16_70: init i32 = call %.loc16_70.3(%.loc16_70.1) [template = constants.%.41]
// CHECK:STDOUT:   %.loc16_70.4: i32 = value_of_initializer %int.convert_checked.loc16_70 [template = constants.%.41]
// CHECK:STDOUT:   %.loc16_70.5: i32 = converted %.loc16_70.1, %.loc16_70.4 [template = constants.%.41]
// CHECK:STDOUT:   %.loc16_71.1: ref i32 = array_index %.loc16_57.2, %.loc16_70.5
// CHECK:STDOUT:   %.loc16_71.2: i32 = bind_value %.loc16_71.1
// CHECK:STDOUT:   %.loc16_38.1: type = value_of_initializer %int.make_type_32.loc16_38 [template = i32]
// CHECK:STDOUT:   %.loc16_38.2: type = converted %int.make_type_32.loc16_38, %.loc16_38.1 [template = i32]
// CHECK:STDOUT:   assign file.%array_index.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
