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
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.1: i32 = int_value 9 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.25: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.26: <bound method> = bound_method %.1, %Convert.15 [template]
// CHECK:STDOUT:   %.27: Core.IntLiteral = int_value 9 [template]
// CHECK:STDOUT:   %.28: type = array_type %.27, i32 [template]
// CHECK:STDOUT:   %.30: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.31: i32 = int_value 2 [template]
// CHECK:STDOUT:   %.32: i32 = int_value 3 [template]
// CHECK:STDOUT:   %.33: i32 = int_value 4 [template]
// CHECK:STDOUT:   %.34: i32 = int_value 5 [template]
// CHECK:STDOUT:   %.35: i32 = int_value 6 [template]
// CHECK:STDOUT:   %.36: i32 = int_value 7 [template]
// CHECK:STDOUT:   %.37: i32 = int_value 8 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (i32, i32, i32, i32, i32, i32, i32, i32, i32) [template]
// CHECK:STDOUT:   %.38: i32 = int_value 0 [template]
// CHECK:STDOUT:   %array: %.28 = tuple_value (%.30, %.31, %.32, %.33, %.34, %.35, %.36, %.37, %.1) [template]
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
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc11_14.1: i32 = int_value 9 [template = constants.%.1]
// CHECK:STDOUT:   %.loc11_9.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc11_9.2: type = converted %int.make_type_32, %.loc11_9.1 [template = i32]
// CHECK:STDOUT:   %.loc11_14.2: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc11_14.3: <bound method> = bound_method %.loc11_14.1, %.loc11_14.2 [template = constants.%.26]
// CHECK:STDOUT:   %int.convert_checked: init Core.IntLiteral = call %.loc11_14.3(%.loc11_14.1) [template = constants.%.27]
// CHECK:STDOUT:   %.loc11_14.4: Core.IntLiteral = value_of_initializer %int.convert_checked [template = constants.%.27]
// CHECK:STDOUT:   %.loc11_14.5: Core.IntLiteral = converted %.loc11_14.1, %.loc11_14.4 [template = constants.%.27]
// CHECK:STDOUT:   %.loc11_15: type = array_type %.loc11_14.5, i32 [template = constants.%.28]
// CHECK:STDOUT:   %a.var: ref %.28 = var a
// CHECK:STDOUT:   %a: ref %.28 = bind_name a, %a.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_20: i32 = int_value 1 [template = constants.%.30]
// CHECK:STDOUT:   %.loc11_23: i32 = int_value 2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc11_26: i32 = int_value 3 [template = constants.%.32]
// CHECK:STDOUT:   %.loc11_29: i32 = int_value 4 [template = constants.%.33]
// CHECK:STDOUT:   %.loc11_32: i32 = int_value 5 [template = constants.%.34]
// CHECK:STDOUT:   %.loc11_35: i32 = int_value 6 [template = constants.%.35]
// CHECK:STDOUT:   %.loc11_38: i32 = int_value 7 [template = constants.%.36]
// CHECK:STDOUT:   %.loc11_41: i32 = int_value 8 [template = constants.%.37]
// CHECK:STDOUT:   %.loc11_44: i32 = int_value 9 [template = constants.%.1]
// CHECK:STDOUT:   %.loc11_45.1: %tuple.type = tuple_literal (%.loc11_20, %.loc11_23, %.loc11_26, %.loc11_29, %.loc11_32, %.loc11_35, %.loc11_38, %.loc11_41, %.loc11_44)
// CHECK:STDOUT:   %.loc11_45.2: i32 = int_value 0 [template = constants.%.38]
// CHECK:STDOUT:   %.loc11_45.3: ref i32 = array_index file.%a.var, %.loc11_45.2
// CHECK:STDOUT:   %.loc11_45.4: init i32 = initialize_from %.loc11_20 to %.loc11_45.3 [template = constants.%.30]
// CHECK:STDOUT:   %.loc11_45.5: i32 = int_value 1 [template = constants.%.30]
// CHECK:STDOUT:   %.loc11_45.6: ref i32 = array_index file.%a.var, %.loc11_45.5
// CHECK:STDOUT:   %.loc11_45.7: init i32 = initialize_from %.loc11_23 to %.loc11_45.6 [template = constants.%.31]
// CHECK:STDOUT:   %.loc11_45.8: i32 = int_value 2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc11_45.9: ref i32 = array_index file.%a.var, %.loc11_45.8
// CHECK:STDOUT:   %.loc11_45.10: init i32 = initialize_from %.loc11_26 to %.loc11_45.9 [template = constants.%.32]
// CHECK:STDOUT:   %.loc11_45.11: i32 = int_value 3 [template = constants.%.32]
// CHECK:STDOUT:   %.loc11_45.12: ref i32 = array_index file.%a.var, %.loc11_45.11
// CHECK:STDOUT:   %.loc11_45.13: init i32 = initialize_from %.loc11_29 to %.loc11_45.12 [template = constants.%.33]
// CHECK:STDOUT:   %.loc11_45.14: i32 = int_value 4 [template = constants.%.33]
// CHECK:STDOUT:   %.loc11_45.15: ref i32 = array_index file.%a.var, %.loc11_45.14
// CHECK:STDOUT:   %.loc11_45.16: init i32 = initialize_from %.loc11_32 to %.loc11_45.15 [template = constants.%.34]
// CHECK:STDOUT:   %.loc11_45.17: i32 = int_value 5 [template = constants.%.34]
// CHECK:STDOUT:   %.loc11_45.18: ref i32 = array_index file.%a.var, %.loc11_45.17
// CHECK:STDOUT:   %.loc11_45.19: init i32 = initialize_from %.loc11_35 to %.loc11_45.18 [template = constants.%.35]
// CHECK:STDOUT:   %.loc11_45.20: i32 = int_value 6 [template = constants.%.35]
// CHECK:STDOUT:   %.loc11_45.21: ref i32 = array_index file.%a.var, %.loc11_45.20
// CHECK:STDOUT:   %.loc11_45.22: init i32 = initialize_from %.loc11_38 to %.loc11_45.21 [template = constants.%.36]
// CHECK:STDOUT:   %.loc11_45.23: i32 = int_value 7 [template = constants.%.36]
// CHECK:STDOUT:   %.loc11_45.24: ref i32 = array_index file.%a.var, %.loc11_45.23
// CHECK:STDOUT:   %.loc11_45.25: init i32 = initialize_from %.loc11_41 to %.loc11_45.24 [template = constants.%.37]
// CHECK:STDOUT:   %.loc11_45.26: i32 = int_value 8 [template = constants.%.37]
// CHECK:STDOUT:   %.loc11_45.27: ref i32 = array_index file.%a.var, %.loc11_45.26
// CHECK:STDOUT:   %.loc11_45.28: init i32 = initialize_from %.loc11_44 to %.loc11_45.27 [template = constants.%.1]
// CHECK:STDOUT:   %.loc11_45.29: init %.28 = array_init (%.loc11_45.4, %.loc11_45.7, %.loc11_45.10, %.loc11_45.13, %.loc11_45.16, %.loc11_45.19, %.loc11_45.22, %.loc11_45.25, %.loc11_45.28) to file.%a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc11_46: init %.28 = converted %.loc11_45.1, %.loc11_45.29 [template = constants.%array]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_46
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
