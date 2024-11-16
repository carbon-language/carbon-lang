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
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 9 [template]
// CHECK:STDOUT:   %.2: type = array_type %.1, i32 [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.5: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.6: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.7: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %.8: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %.9: Core.IntLiteral = int_value 6 [template]
// CHECK:STDOUT:   %.10: Core.IntLiteral = int_value 7 [template]
// CHECK:STDOUT:   %.11: Core.IntLiteral = int_value 8 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %.12: i32 = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.36: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.37: <bound method> = bound_method %.4, %Convert.15 [template]
// CHECK:STDOUT:   %.38: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.39: <bound method> = bound_method %.5, %Convert.15 [template]
// CHECK:STDOUT:   %.40: i32 = int_value 2 [template]
// CHECK:STDOUT:   %.41: <bound method> = bound_method %.6, %Convert.15 [template]
// CHECK:STDOUT:   %.42: i32 = int_value 3 [template]
// CHECK:STDOUT:   %.43: <bound method> = bound_method %.7, %Convert.15 [template]
// CHECK:STDOUT:   %.44: i32 = int_value 4 [template]
// CHECK:STDOUT:   %.45: <bound method> = bound_method %.8, %Convert.15 [template]
// CHECK:STDOUT:   %.46: i32 = int_value 5 [template]
// CHECK:STDOUT:   %.47: <bound method> = bound_method %.9, %Convert.15 [template]
// CHECK:STDOUT:   %.48: i32 = int_value 6 [template]
// CHECK:STDOUT:   %.49: <bound method> = bound_method %.10, %Convert.15 [template]
// CHECK:STDOUT:   %.50: i32 = int_value 7 [template]
// CHECK:STDOUT:   %.51: <bound method> = bound_method %.11, %Convert.15 [template]
// CHECK:STDOUT:   %.52: i32 = int_value 8 [template]
// CHECK:STDOUT:   %.53: <bound method> = bound_method %.1, %Convert.15 [template]
// CHECK:STDOUT:   %.54: i32 = int_value 9 [template]
// CHECK:STDOUT:   %array: %.2 = tuple_value (%.38, %.40, %.42, %.44, %.46, %.48, %.50, %.52, %.54) [template]
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
// CHECK:STDOUT:   %.loc11_14: Core.IntLiteral = int_value 9 [template = constants.%.1]
// CHECK:STDOUT:   %.loc11_9.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc11_9.2: type = converted %int.make_type_32, %.loc11_9.1 [template = i32]
// CHECK:STDOUT:   %.loc11_15: type = array_type %.loc11_14, i32 [template = constants.%.2]
// CHECK:STDOUT:   %a.var: ref %.2 = var a
// CHECK:STDOUT:   %a: ref %.2 = bind_name a, %a.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_20: Core.IntLiteral = int_value 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc11_23: Core.IntLiteral = int_value 2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_26: Core.IntLiteral = int_value 3 [template = constants.%.6]
// CHECK:STDOUT:   %.loc11_29: Core.IntLiteral = int_value 4 [template = constants.%.7]
// CHECK:STDOUT:   %.loc11_32: Core.IntLiteral = int_value 5 [template = constants.%.8]
// CHECK:STDOUT:   %.loc11_35: Core.IntLiteral = int_value 6 [template = constants.%.9]
// CHECK:STDOUT:   %.loc11_38: Core.IntLiteral = int_value 7 [template = constants.%.10]
// CHECK:STDOUT:   %.loc11_41: Core.IntLiteral = int_value 8 [template = constants.%.11]
// CHECK:STDOUT:   %.loc11_44: Core.IntLiteral = int_value 9 [template = constants.%.1]
// CHECK:STDOUT:   %.loc11_45.1: %tuple.type = tuple_literal (%.loc11_20, %.loc11_23, %.loc11_26, %.loc11_29, %.loc11_32, %.loc11_35, %.loc11_38, %.loc11_41, %.loc11_44)
// CHECK:STDOUT:   %.loc11_45.2: %Convert.type.2 = interface_witness_access constants.%.36, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc11_45.3: <bound method> = bound_method %.loc11_20, %.loc11_45.2 [template = constants.%.37]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.1: init i32 = call %.loc11_45.3(%.loc11_20) [template = constants.%.38]
// CHECK:STDOUT:   %.loc11_45.4: init i32 = converted %.loc11_20, %int.convert_checked.loc11_45.1 [template = constants.%.38]
// CHECK:STDOUT:   %.loc11_45.5: i32 = int_value 0 [template = constants.%.12]
// CHECK:STDOUT:   %.loc11_45.6: ref i32 = array_index file.%a.var, %.loc11_45.5
// CHECK:STDOUT:   %.loc11_45.7: init i32 = initialize_from %.loc11_45.4 to %.loc11_45.6 [template = constants.%.38]
// CHECK:STDOUT:   %.loc11_45.8: %Convert.type.2 = interface_witness_access constants.%.36, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc11_45.9: <bound method> = bound_method %.loc11_23, %.loc11_45.8 [template = constants.%.39]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.2: init i32 = call %.loc11_45.9(%.loc11_23) [template = constants.%.40]
// CHECK:STDOUT:   %.loc11_45.10: init i32 = converted %.loc11_23, %int.convert_checked.loc11_45.2 [template = constants.%.40]
// CHECK:STDOUT:   %.loc11_45.11: i32 = int_value 1 [template = constants.%.38]
// CHECK:STDOUT:   %.loc11_45.12: ref i32 = array_index file.%a.var, %.loc11_45.11
// CHECK:STDOUT:   %.loc11_45.13: init i32 = initialize_from %.loc11_45.10 to %.loc11_45.12 [template = constants.%.40]
// CHECK:STDOUT:   %.loc11_45.14: %Convert.type.2 = interface_witness_access constants.%.36, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc11_45.15: <bound method> = bound_method %.loc11_26, %.loc11_45.14 [template = constants.%.41]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.3: init i32 = call %.loc11_45.15(%.loc11_26) [template = constants.%.42]
// CHECK:STDOUT:   %.loc11_45.16: init i32 = converted %.loc11_26, %int.convert_checked.loc11_45.3 [template = constants.%.42]
// CHECK:STDOUT:   %.loc11_45.17: i32 = int_value 2 [template = constants.%.40]
// CHECK:STDOUT:   %.loc11_45.18: ref i32 = array_index file.%a.var, %.loc11_45.17
// CHECK:STDOUT:   %.loc11_45.19: init i32 = initialize_from %.loc11_45.16 to %.loc11_45.18 [template = constants.%.42]
// CHECK:STDOUT:   %.loc11_45.20: %Convert.type.2 = interface_witness_access constants.%.36, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc11_45.21: <bound method> = bound_method %.loc11_29, %.loc11_45.20 [template = constants.%.43]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.4: init i32 = call %.loc11_45.21(%.loc11_29) [template = constants.%.44]
// CHECK:STDOUT:   %.loc11_45.22: init i32 = converted %.loc11_29, %int.convert_checked.loc11_45.4 [template = constants.%.44]
// CHECK:STDOUT:   %.loc11_45.23: i32 = int_value 3 [template = constants.%.42]
// CHECK:STDOUT:   %.loc11_45.24: ref i32 = array_index file.%a.var, %.loc11_45.23
// CHECK:STDOUT:   %.loc11_45.25: init i32 = initialize_from %.loc11_45.22 to %.loc11_45.24 [template = constants.%.44]
// CHECK:STDOUT:   %.loc11_45.26: %Convert.type.2 = interface_witness_access constants.%.36, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc11_45.27: <bound method> = bound_method %.loc11_32, %.loc11_45.26 [template = constants.%.45]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.5: init i32 = call %.loc11_45.27(%.loc11_32) [template = constants.%.46]
// CHECK:STDOUT:   %.loc11_45.28: init i32 = converted %.loc11_32, %int.convert_checked.loc11_45.5 [template = constants.%.46]
// CHECK:STDOUT:   %.loc11_45.29: i32 = int_value 4 [template = constants.%.44]
// CHECK:STDOUT:   %.loc11_45.30: ref i32 = array_index file.%a.var, %.loc11_45.29
// CHECK:STDOUT:   %.loc11_45.31: init i32 = initialize_from %.loc11_45.28 to %.loc11_45.30 [template = constants.%.46]
// CHECK:STDOUT:   %.loc11_45.32: %Convert.type.2 = interface_witness_access constants.%.36, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc11_45.33: <bound method> = bound_method %.loc11_35, %.loc11_45.32 [template = constants.%.47]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.6: init i32 = call %.loc11_45.33(%.loc11_35) [template = constants.%.48]
// CHECK:STDOUT:   %.loc11_45.34: init i32 = converted %.loc11_35, %int.convert_checked.loc11_45.6 [template = constants.%.48]
// CHECK:STDOUT:   %.loc11_45.35: i32 = int_value 5 [template = constants.%.46]
// CHECK:STDOUT:   %.loc11_45.36: ref i32 = array_index file.%a.var, %.loc11_45.35
// CHECK:STDOUT:   %.loc11_45.37: init i32 = initialize_from %.loc11_45.34 to %.loc11_45.36 [template = constants.%.48]
// CHECK:STDOUT:   %.loc11_45.38: %Convert.type.2 = interface_witness_access constants.%.36, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc11_45.39: <bound method> = bound_method %.loc11_38, %.loc11_45.38 [template = constants.%.49]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.7: init i32 = call %.loc11_45.39(%.loc11_38) [template = constants.%.50]
// CHECK:STDOUT:   %.loc11_45.40: init i32 = converted %.loc11_38, %int.convert_checked.loc11_45.7 [template = constants.%.50]
// CHECK:STDOUT:   %.loc11_45.41: i32 = int_value 6 [template = constants.%.48]
// CHECK:STDOUT:   %.loc11_45.42: ref i32 = array_index file.%a.var, %.loc11_45.41
// CHECK:STDOUT:   %.loc11_45.43: init i32 = initialize_from %.loc11_45.40 to %.loc11_45.42 [template = constants.%.50]
// CHECK:STDOUT:   %.loc11_45.44: %Convert.type.2 = interface_witness_access constants.%.36, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc11_45.45: <bound method> = bound_method %.loc11_41, %.loc11_45.44 [template = constants.%.51]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.8: init i32 = call %.loc11_45.45(%.loc11_41) [template = constants.%.52]
// CHECK:STDOUT:   %.loc11_45.46: init i32 = converted %.loc11_41, %int.convert_checked.loc11_45.8 [template = constants.%.52]
// CHECK:STDOUT:   %.loc11_45.47: i32 = int_value 7 [template = constants.%.50]
// CHECK:STDOUT:   %.loc11_45.48: ref i32 = array_index file.%a.var, %.loc11_45.47
// CHECK:STDOUT:   %.loc11_45.49: init i32 = initialize_from %.loc11_45.46 to %.loc11_45.48 [template = constants.%.52]
// CHECK:STDOUT:   %.loc11_45.50: %Convert.type.2 = interface_witness_access constants.%.36, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc11_45.51: <bound method> = bound_method %.loc11_44, %.loc11_45.50 [template = constants.%.53]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.9: init i32 = call %.loc11_45.51(%.loc11_44) [template = constants.%.54]
// CHECK:STDOUT:   %.loc11_45.52: init i32 = converted %.loc11_44, %int.convert_checked.loc11_45.9 [template = constants.%.54]
// CHECK:STDOUT:   %.loc11_45.53: i32 = int_value 8 [template = constants.%.52]
// CHECK:STDOUT:   %.loc11_45.54: ref i32 = array_index file.%a.var, %.loc11_45.53
// CHECK:STDOUT:   %.loc11_45.55: init i32 = initialize_from %.loc11_45.52 to %.loc11_45.54 [template = constants.%.54]
// CHECK:STDOUT:   %.loc11_45.56: init %.2 = array_init (%.loc11_45.7, %.loc11_45.13, %.loc11_45.19, %.loc11_45.25, %.loc11_45.31, %.loc11_45.37, %.loc11_45.43, %.loc11_45.49, %.loc11_45.55) to file.%a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc11_46: init %.2 = converted %.loc11_45.1, %.loc11_45.56 [template = constants.%array]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_46
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
