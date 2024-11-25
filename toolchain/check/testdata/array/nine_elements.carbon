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
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 9 [template]
// CHECK:STDOUT:   %.3: type = array_type %.2, %i32 [template]
// CHECK:STDOUT:   %.5: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.6: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.7: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.8: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %.9: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %.10: Core.IntLiteral = int_value 6 [template]
// CHECK:STDOUT:   %.11: Core.IntLiteral = int_value 7 [template]
// CHECK:STDOUT:   %.12: Core.IntLiteral = int_value 8 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %.13: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.37: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.38: <bound method> = bound_method %.5, %Convert.14 [template]
// CHECK:STDOUT:   %.39: <specific function> = specific_function %.38, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.40: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %.41: <bound method> = bound_method %.6, %Convert.14 [template]
// CHECK:STDOUT:   %.42: <specific function> = specific_function %.41, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.43: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %.44: <bound method> = bound_method %.7, %Convert.14 [template]
// CHECK:STDOUT:   %.45: <specific function> = specific_function %.44, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.46: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %.47: <bound method> = bound_method %.8, %Convert.14 [template]
// CHECK:STDOUT:   %.48: <specific function> = specific_function %.47, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.49: %i32 = int_value 4 [template]
// CHECK:STDOUT:   %.50: <bound method> = bound_method %.9, %Convert.14 [template]
// CHECK:STDOUT:   %.51: <specific function> = specific_function %.50, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.52: %i32 = int_value 5 [template]
// CHECK:STDOUT:   %.53: <bound method> = bound_method %.10, %Convert.14 [template]
// CHECK:STDOUT:   %.54: <specific function> = specific_function %.53, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.55: %i32 = int_value 6 [template]
// CHECK:STDOUT:   %.56: <bound method> = bound_method %.11, %Convert.14 [template]
// CHECK:STDOUT:   %.57: <specific function> = specific_function %.56, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.58: %i32 = int_value 7 [template]
// CHECK:STDOUT:   %.59: <bound method> = bound_method %.12, %Convert.14 [template]
// CHECK:STDOUT:   %.60: <specific function> = specific_function %.59, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.61: %i32 = int_value 8 [template]
// CHECK:STDOUT:   %.62: <bound method> = bound_method %.2, %Convert.14 [template]
// CHECK:STDOUT:   %.63: <specific function> = specific_function %.62, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.64: %i32 = int_value 9 [template]
// CHECK:STDOUT:   %array: %.3 = tuple_value (%.40, %.43, %.46, %.49, %.52, %.55, %.58, %.61, %.64) [template]
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
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %.loc11_9.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%.loc11_9.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_14: Core.IntLiteral = int_value 9 [template = constants.%.2]
// CHECK:STDOUT:   %.loc11_9.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_9.3: type = converted %int.make_type_signed, %.loc11_9.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_15: type = array_type %.loc11_14, %i32 [template = constants.%.3]
// CHECK:STDOUT:   %a.var: ref %.3 = var a
// CHECK:STDOUT:   %a: ref %.3 = bind_name a, %a.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_20: Core.IntLiteral = int_value 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_23: Core.IntLiteral = int_value 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc11_26: Core.IntLiteral = int_value 3 [template = constants.%.7]
// CHECK:STDOUT:   %.loc11_29: Core.IntLiteral = int_value 4 [template = constants.%.8]
// CHECK:STDOUT:   %.loc11_32: Core.IntLiteral = int_value 5 [template = constants.%.9]
// CHECK:STDOUT:   %.loc11_35: Core.IntLiteral = int_value 6 [template = constants.%.10]
// CHECK:STDOUT:   %.loc11_38: Core.IntLiteral = int_value 7 [template = constants.%.11]
// CHECK:STDOUT:   %.loc11_41: Core.IntLiteral = int_value 8 [template = constants.%.12]
// CHECK:STDOUT:   %.loc11_44: Core.IntLiteral = int_value 9 [template = constants.%.2]
// CHECK:STDOUT:   %.loc11_45.1: %tuple.type = tuple_literal (%.loc11_20, %.loc11_23, %.loc11_26, %.loc11_29, %.loc11_32, %.loc11_35, %.loc11_38, %.loc11_41, %.loc11_44)
// CHECK:STDOUT:   %.loc11_45.2: %Convert.type.2 = interface_witness_access constants.%.37, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_45.3: <bound method> = bound_method %.loc11_20, %.loc11_45.2 [template = constants.%.38]
// CHECK:STDOUT:   %.loc11_45.4: <specific function> = specific_function %.loc11_45.3, @Convert.2(constants.%.1) [template = constants.%.39]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.1: init %i32 = call %.loc11_45.4(%.loc11_20) [template = constants.%.40]
// CHECK:STDOUT:   %.loc11_45.5: init %i32 = converted %.loc11_20, %int.convert_checked.loc11_45.1 [template = constants.%.40]
// CHECK:STDOUT:   %.loc11_45.6: Core.IntLiteral = int_value 0 [template = constants.%.13]
// CHECK:STDOUT:   %.loc11_45.7: ref %i32 = array_index file.%a.var, %.loc11_45.6
// CHECK:STDOUT:   %.loc11_45.8: init %i32 = initialize_from %.loc11_45.5 to %.loc11_45.7 [template = constants.%.40]
// CHECK:STDOUT:   %.loc11_45.9: %Convert.type.2 = interface_witness_access constants.%.37, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_45.10: <bound method> = bound_method %.loc11_23, %.loc11_45.9 [template = constants.%.41]
// CHECK:STDOUT:   %.loc11_45.11: <specific function> = specific_function %.loc11_45.10, @Convert.2(constants.%.1) [template = constants.%.42]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.2: init %i32 = call %.loc11_45.11(%.loc11_23) [template = constants.%.43]
// CHECK:STDOUT:   %.loc11_45.12: init %i32 = converted %.loc11_23, %int.convert_checked.loc11_45.2 [template = constants.%.43]
// CHECK:STDOUT:   %.loc11_45.13: Core.IntLiteral = int_value 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_45.14: ref %i32 = array_index file.%a.var, %.loc11_45.13
// CHECK:STDOUT:   %.loc11_45.15: init %i32 = initialize_from %.loc11_45.12 to %.loc11_45.14 [template = constants.%.43]
// CHECK:STDOUT:   %.loc11_45.16: %Convert.type.2 = interface_witness_access constants.%.37, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_45.17: <bound method> = bound_method %.loc11_26, %.loc11_45.16 [template = constants.%.44]
// CHECK:STDOUT:   %.loc11_45.18: <specific function> = specific_function %.loc11_45.17, @Convert.2(constants.%.1) [template = constants.%.45]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.3: init %i32 = call %.loc11_45.18(%.loc11_26) [template = constants.%.46]
// CHECK:STDOUT:   %.loc11_45.19: init %i32 = converted %.loc11_26, %int.convert_checked.loc11_45.3 [template = constants.%.46]
// CHECK:STDOUT:   %.loc11_45.20: Core.IntLiteral = int_value 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc11_45.21: ref %i32 = array_index file.%a.var, %.loc11_45.20
// CHECK:STDOUT:   %.loc11_45.22: init %i32 = initialize_from %.loc11_45.19 to %.loc11_45.21 [template = constants.%.46]
// CHECK:STDOUT:   %.loc11_45.23: %Convert.type.2 = interface_witness_access constants.%.37, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_45.24: <bound method> = bound_method %.loc11_29, %.loc11_45.23 [template = constants.%.47]
// CHECK:STDOUT:   %.loc11_45.25: <specific function> = specific_function %.loc11_45.24, @Convert.2(constants.%.1) [template = constants.%.48]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.4: init %i32 = call %.loc11_45.25(%.loc11_29) [template = constants.%.49]
// CHECK:STDOUT:   %.loc11_45.26: init %i32 = converted %.loc11_29, %int.convert_checked.loc11_45.4 [template = constants.%.49]
// CHECK:STDOUT:   %.loc11_45.27: Core.IntLiteral = int_value 3 [template = constants.%.7]
// CHECK:STDOUT:   %.loc11_45.28: ref %i32 = array_index file.%a.var, %.loc11_45.27
// CHECK:STDOUT:   %.loc11_45.29: init %i32 = initialize_from %.loc11_45.26 to %.loc11_45.28 [template = constants.%.49]
// CHECK:STDOUT:   %.loc11_45.30: %Convert.type.2 = interface_witness_access constants.%.37, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_45.31: <bound method> = bound_method %.loc11_32, %.loc11_45.30 [template = constants.%.50]
// CHECK:STDOUT:   %.loc11_45.32: <specific function> = specific_function %.loc11_45.31, @Convert.2(constants.%.1) [template = constants.%.51]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.5: init %i32 = call %.loc11_45.32(%.loc11_32) [template = constants.%.52]
// CHECK:STDOUT:   %.loc11_45.33: init %i32 = converted %.loc11_32, %int.convert_checked.loc11_45.5 [template = constants.%.52]
// CHECK:STDOUT:   %.loc11_45.34: Core.IntLiteral = int_value 4 [template = constants.%.8]
// CHECK:STDOUT:   %.loc11_45.35: ref %i32 = array_index file.%a.var, %.loc11_45.34
// CHECK:STDOUT:   %.loc11_45.36: init %i32 = initialize_from %.loc11_45.33 to %.loc11_45.35 [template = constants.%.52]
// CHECK:STDOUT:   %.loc11_45.37: %Convert.type.2 = interface_witness_access constants.%.37, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_45.38: <bound method> = bound_method %.loc11_35, %.loc11_45.37 [template = constants.%.53]
// CHECK:STDOUT:   %.loc11_45.39: <specific function> = specific_function %.loc11_45.38, @Convert.2(constants.%.1) [template = constants.%.54]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.6: init %i32 = call %.loc11_45.39(%.loc11_35) [template = constants.%.55]
// CHECK:STDOUT:   %.loc11_45.40: init %i32 = converted %.loc11_35, %int.convert_checked.loc11_45.6 [template = constants.%.55]
// CHECK:STDOUT:   %.loc11_45.41: Core.IntLiteral = int_value 5 [template = constants.%.9]
// CHECK:STDOUT:   %.loc11_45.42: ref %i32 = array_index file.%a.var, %.loc11_45.41
// CHECK:STDOUT:   %.loc11_45.43: init %i32 = initialize_from %.loc11_45.40 to %.loc11_45.42 [template = constants.%.55]
// CHECK:STDOUT:   %.loc11_45.44: %Convert.type.2 = interface_witness_access constants.%.37, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_45.45: <bound method> = bound_method %.loc11_38, %.loc11_45.44 [template = constants.%.56]
// CHECK:STDOUT:   %.loc11_45.46: <specific function> = specific_function %.loc11_45.45, @Convert.2(constants.%.1) [template = constants.%.57]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.7: init %i32 = call %.loc11_45.46(%.loc11_38) [template = constants.%.58]
// CHECK:STDOUT:   %.loc11_45.47: init %i32 = converted %.loc11_38, %int.convert_checked.loc11_45.7 [template = constants.%.58]
// CHECK:STDOUT:   %.loc11_45.48: Core.IntLiteral = int_value 6 [template = constants.%.10]
// CHECK:STDOUT:   %.loc11_45.49: ref %i32 = array_index file.%a.var, %.loc11_45.48
// CHECK:STDOUT:   %.loc11_45.50: init %i32 = initialize_from %.loc11_45.47 to %.loc11_45.49 [template = constants.%.58]
// CHECK:STDOUT:   %.loc11_45.51: %Convert.type.2 = interface_witness_access constants.%.37, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_45.52: <bound method> = bound_method %.loc11_41, %.loc11_45.51 [template = constants.%.59]
// CHECK:STDOUT:   %.loc11_45.53: <specific function> = specific_function %.loc11_45.52, @Convert.2(constants.%.1) [template = constants.%.60]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.8: init %i32 = call %.loc11_45.53(%.loc11_41) [template = constants.%.61]
// CHECK:STDOUT:   %.loc11_45.54: init %i32 = converted %.loc11_41, %int.convert_checked.loc11_45.8 [template = constants.%.61]
// CHECK:STDOUT:   %.loc11_45.55: Core.IntLiteral = int_value 7 [template = constants.%.11]
// CHECK:STDOUT:   %.loc11_45.56: ref %i32 = array_index file.%a.var, %.loc11_45.55
// CHECK:STDOUT:   %.loc11_45.57: init %i32 = initialize_from %.loc11_45.54 to %.loc11_45.56 [template = constants.%.61]
// CHECK:STDOUT:   %.loc11_45.58: %Convert.type.2 = interface_witness_access constants.%.37, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_45.59: <bound method> = bound_method %.loc11_44, %.loc11_45.58 [template = constants.%.62]
// CHECK:STDOUT:   %.loc11_45.60: <specific function> = specific_function %.loc11_45.59, @Convert.2(constants.%.1) [template = constants.%.63]
// CHECK:STDOUT:   %int.convert_checked.loc11_45.9: init %i32 = call %.loc11_45.60(%.loc11_44) [template = constants.%.64]
// CHECK:STDOUT:   %.loc11_45.61: init %i32 = converted %.loc11_44, %int.convert_checked.loc11_45.9 [template = constants.%.64]
// CHECK:STDOUT:   %.loc11_45.62: Core.IntLiteral = int_value 8 [template = constants.%.12]
// CHECK:STDOUT:   %.loc11_45.63: ref %i32 = array_index file.%a.var, %.loc11_45.62
// CHECK:STDOUT:   %.loc11_45.64: init %i32 = initialize_from %.loc11_45.61 to %.loc11_45.63 [template = constants.%.64]
// CHECK:STDOUT:   %.loc11_45.65: init %.3 = array_init (%.loc11_45.8, %.loc11_45.15, %.loc11_45.22, %.loc11_45.29, %.loc11_45.36, %.loc11_45.43, %.loc11_45.50, %.loc11_45.57, %.loc11_45.64) to file.%a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc11_46: init %.3 = converted %.loc11_45.1, %.loc11_45.65 [template = constants.%array]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_46
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
