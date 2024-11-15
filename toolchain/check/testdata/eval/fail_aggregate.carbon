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
// CHECK:STDOUT:   %.1: i32 = int_value 1 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.25: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.26: <bound method> = bound_method %.1, %Convert.15 [template]
// CHECK:STDOUT:   %.27: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.28: type = array_type %.27, i32 [template]
// CHECK:STDOUT:   %.30: i32 = int_value 0 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.31: i32 = int_value 5 [template]
// CHECK:STDOUT:   %.32: i32 = int_value 7 [template]
// CHECK:STDOUT:   %.33: i32 = int_value 9 [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (i32, i32, i32, i32) [template]
// CHECK:STDOUT:   %.34: i32 = int_value 4 [template]
// CHECK:STDOUT:   %.35: <bound method> = bound_method %.34, %Convert.15 [template]
// CHECK:STDOUT:   %.36: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %.37: type = array_type %.36, i32 [template]
// CHECK:STDOUT:   %.39: i32 = int_value 2 [template]
// CHECK:STDOUT:   %.40: i32 = int_value 3 [template]
// CHECK:STDOUT:   %array: %.37 = tuple_value (%.31, %.32, %.1, %.33) [template]
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
// CHECK:STDOUT:   %.loc16_24.1: i32 = int_value 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc16_19.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc16_19.2: type = converted %int.make_type_32, %.loc16_19.1 [template = i32]
// CHECK:STDOUT:   %.loc16_24.2: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc16_24.3: <bound method> = bound_method %.loc16_24.1, %.loc16_24.2 [template = constants.%.26]
// CHECK:STDOUT:   %int.convert_checked: init Core.IntLiteral = call %.loc16_24.3(%.loc16_24.1) [template = constants.%.27]
// CHECK:STDOUT:   %.loc16_24.4: Core.IntLiteral = value_of_initializer %int.convert_checked [template = constants.%.27]
// CHECK:STDOUT:   %.loc16_24.5: Core.IntLiteral = converted %.loc16_24.1, %.loc16_24.4 [template = constants.%.27]
// CHECK:STDOUT:   %.loc16_25: type = array_type %.loc16_24.5, i32 [template = constants.%.28]
// CHECK:STDOUT:   %array_index.var: ref %.28 = var array_index
// CHECK:STDOUT:   %array_index: ref %.28 = bind_name array_index, %array_index.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc16_30: i32 = int_value 0 [template = constants.%.30]
// CHECK:STDOUT:   %.loc16_32: %tuple.type.1 = tuple_literal (%.loc16_30)
// CHECK:STDOUT:   %int.make_type_32.loc16_38: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc16_45: i32 = int_value 5 [template = constants.%.31]
// CHECK:STDOUT:   %.loc16_48: i32 = int_value 7 [template = constants.%.32]
// CHECK:STDOUT:   %.loc16_51: i32 = int_value 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc16_54: i32 = int_value 9 [template = constants.%.33]
// CHECK:STDOUT:   %.loc16_55.1: %tuple.type.2 = tuple_literal (%.loc16_45, %.loc16_48, %.loc16_51, %.loc16_54)
// CHECK:STDOUT:   %int.make_type_32.loc16_61: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc16_66.1: i32 = int_value 4 [template = constants.%.34]
// CHECK:STDOUT:   %.loc16_61.1: type = value_of_initializer %int.make_type_32.loc16_61 [template = i32]
// CHECK:STDOUT:   %.loc16_61.2: type = converted %int.make_type_32.loc16_61, %.loc16_61.1 [template = i32]
// CHECK:STDOUT:   %.loc16_66.2: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc16_66.3: <bound method> = bound_method %.loc16_66.1, %.loc16_66.2 [template = constants.%.35]
// CHECK:STDOUT:   %int.convert_checked: init Core.IntLiteral = call %.loc16_66.3(%.loc16_66.1) [template = constants.%.36]
// CHECK:STDOUT:   %.loc16_66.4: Core.IntLiteral = value_of_initializer %int.convert_checked [template = constants.%.36]
// CHECK:STDOUT:   %.loc16_66.5: Core.IntLiteral = converted %.loc16_66.1, %.loc16_66.4 [template = constants.%.36]
// CHECK:STDOUT:   %.loc16_67: type = array_type %.loc16_66.5, i32 [template = constants.%.37]
// CHECK:STDOUT:   %.loc16_55.2: ref %.37 = temporary_storage
// CHECK:STDOUT:   %.loc16_55.3: i32 = int_value 0 [template = constants.%.30]
// CHECK:STDOUT:   %.loc16_55.4: ref i32 = array_index %.loc16_55.2, %.loc16_55.3
// CHECK:STDOUT:   %.loc16_55.5: init i32 = initialize_from %.loc16_45 to %.loc16_55.4 [template = constants.%.31]
// CHECK:STDOUT:   %.loc16_55.6: i32 = int_value 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc16_55.7: ref i32 = array_index %.loc16_55.2, %.loc16_55.6
// CHECK:STDOUT:   %.loc16_55.8: init i32 = initialize_from %.loc16_48 to %.loc16_55.7 [template = constants.%.32]
// CHECK:STDOUT:   %.loc16_55.9: i32 = int_value 2 [template = constants.%.39]
// CHECK:STDOUT:   %.loc16_55.10: ref i32 = array_index %.loc16_55.2, %.loc16_55.9
// CHECK:STDOUT:   %.loc16_55.11: init i32 = initialize_from %.loc16_51 to %.loc16_55.10 [template = constants.%.1]
// CHECK:STDOUT:   %.loc16_55.12: i32 = int_value 3 [template = constants.%.40]
// CHECK:STDOUT:   %.loc16_55.13: ref i32 = array_index %.loc16_55.2, %.loc16_55.12
// CHECK:STDOUT:   %.loc16_55.14: init i32 = initialize_from %.loc16_54 to %.loc16_55.13 [template = constants.%.33]
// CHECK:STDOUT:   %.loc16_55.15: init %.37 = array_init (%.loc16_55.5, %.loc16_55.8, %.loc16_55.11, %.loc16_55.14) to %.loc16_55.2 [template = constants.%array]
// CHECK:STDOUT:   %.loc16_57.1: init %.37 = converted %.loc16_55.1, %.loc16_55.15 [template = constants.%array]
// CHECK:STDOUT:   %.loc16_70: i32 = int_value 2 [template = constants.%.39]
// CHECK:STDOUT:   %.loc16_57.2: ref %.37 = temporary %.loc16_55.2, %.loc16_57.1
// CHECK:STDOUT:   %.loc16_71.1: ref i32 = array_index %.loc16_57.2, %.loc16_70
// CHECK:STDOUT:   %.loc16_71.2: i32 = bind_value %.loc16_71.1
// CHECK:STDOUT:   %.loc16_38.1: type = value_of_initializer %int.make_type_32.loc16_38 [template = i32]
// CHECK:STDOUT:   %.loc16_38.2: type = converted %int.make_type_32.loc16_38, %.loc16_38.1 [template = i32]
// CHECK:STDOUT:   assign file.%array_index.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
