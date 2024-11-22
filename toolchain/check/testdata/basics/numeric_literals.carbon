// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/basics/numeric_literals.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/basics/numeric_literals.carbon

fn F() {
  // 8 and 9 trigger special behavior in APInt when mishandling signed versus
  // unsigned, so we pay extra attention to those.
  var ints: [i32; 6] = (
    8,
    9,
    0x8,
    0b1000,
    2147483647,
    0x7FFFFFFF,
  );
  var floats: [f64; 6] = (
    0.9,
    8.0,
    80.0,
    1.0e7,
    1.0e8,
    1.0e-8
  );
}

// CHECK:STDOUT: --- numeric_literals.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 6 [template]
// CHECK:STDOUT:   %.3: type = array_type %.2, %i32 [template]
// CHECK:STDOUT:   %.5: Core.IntLiteral = int_value 8 [template]
// CHECK:STDOUT:   %.6: Core.IntLiteral = int_value 9 [template]
// CHECK:STDOUT:   %.7: Core.IntLiteral = int_value 2147483647 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %.8: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.32: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.33: <bound method> = bound_method %.5, %Convert.14 [template]
// CHECK:STDOUT:   %.34: <specific function> = specific_function %.33, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.35: %i32 = int_value 8 [template]
// CHECK:STDOUT:   %.36: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.37: <bound method> = bound_method %.6, %Convert.14 [template]
// CHECK:STDOUT:   %.38: <specific function> = specific_function %.37, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.39: %i32 = int_value 9 [template]
// CHECK:STDOUT:   %.40: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %.41: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.42: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %.43: <bound method> = bound_method %.7, %Convert.14 [template]
// CHECK:STDOUT:   %.44: <specific function> = specific_function %.43, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.45: %i32 = int_value 2147483647 [template]
// CHECK:STDOUT:   %.46: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %array.1: %.3 = tuple_value (%.35, %.39, %.35, %.35, %.45, %.45) [template]
// CHECK:STDOUT:   %.47: Core.IntLiteral = int_value 64 [template]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [template]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [template]
// CHECK:STDOUT:   %.48: type = array_type %.2, f64 [template]
// CHECK:STDOUT:   %.50: f64 = float_literal 0.90000000000000002 [template]
// CHECK:STDOUT:   %.51: f64 = float_literal 8 [template]
// CHECK:STDOUT:   %.52: f64 = float_literal 80 [template]
// CHECK:STDOUT:   %.53: f64 = float_literal 1.0E+7 [template]
// CHECK:STDOUT:   %.54: f64 = float_literal 1.0E+8 [template]
// CHECK:STDOUT:   %.55: f64 = float_literal 1.0E-8 [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (f64, f64, f64, f64, f64, f64) [template]
// CHECK:STDOUT:   %array.2: %.48 = tuple_value (%.50, %.51, %.52, %.53, %.54, %.55) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     .Float = %import_ref.38
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_14.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%.loc14_14.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_19: Core.IntLiteral = int_value 6 [template = constants.%.2]
// CHECK:STDOUT:   %.loc14_14.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_14.3: type = converted %int.make_type_signed, %.loc14_14.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_20: type = array_type %.loc14_19, %i32 [template = constants.%.3]
// CHECK:STDOUT:   %ints.var: ref %.3 = var ints
// CHECK:STDOUT:   %ints: ref %.3 = bind_name ints, %ints.var
// CHECK:STDOUT:   %.loc15: Core.IntLiteral = int_value 8 [template = constants.%.5]
// CHECK:STDOUT:   %.loc16: Core.IntLiteral = int_value 9 [template = constants.%.6]
// CHECK:STDOUT:   %.loc17: Core.IntLiteral = int_value 8 [template = constants.%.5]
// CHECK:STDOUT:   %.loc18: Core.IntLiteral = int_value 8 [template = constants.%.5]
// CHECK:STDOUT:   %.loc19: Core.IntLiteral = int_value 2147483647 [template = constants.%.7]
// CHECK:STDOUT:   %.loc20: Core.IntLiteral = int_value 2147483647 [template = constants.%.7]
// CHECK:STDOUT:   %.loc21_3.1: %tuple.type.1 = tuple_literal (%.loc15, %.loc16, %.loc17, %.loc18, %.loc19, %.loc20)
// CHECK:STDOUT:   %.loc21_3.2: %Convert.type.2 = interface_witness_access constants.%.32, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc21_3.3: <bound method> = bound_method %.loc15, %.loc21_3.2 [template = constants.%.33]
// CHECK:STDOUT:   %.loc21_3.4: <specific function> = specific_function %.loc21_3.3, @Convert.2(constants.%.1) [template = constants.%.34]
// CHECK:STDOUT:   %int.convert_checked.loc21_3.1: init %i32 = call %.loc21_3.4(%.loc15) [template = constants.%.35]
// CHECK:STDOUT:   %.loc21_3.5: init %i32 = converted %.loc15, %int.convert_checked.loc21_3.1 [template = constants.%.35]
// CHECK:STDOUT:   %.loc21_3.6: Core.IntLiteral = int_value 0 [template = constants.%.8]
// CHECK:STDOUT:   %.loc21_3.7: ref %i32 = array_index %ints.var, %.loc21_3.6
// CHECK:STDOUT:   %.loc21_3.8: init %i32 = initialize_from %.loc21_3.5 to %.loc21_3.7 [template = constants.%.35]
// CHECK:STDOUT:   %.loc21_3.9: %Convert.type.2 = interface_witness_access constants.%.32, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc21_3.10: <bound method> = bound_method %.loc16, %.loc21_3.9 [template = constants.%.37]
// CHECK:STDOUT:   %.loc21_3.11: <specific function> = specific_function %.loc21_3.10, @Convert.2(constants.%.1) [template = constants.%.38]
// CHECK:STDOUT:   %int.convert_checked.loc21_3.2: init %i32 = call %.loc21_3.11(%.loc16) [template = constants.%.39]
// CHECK:STDOUT:   %.loc21_3.12: init %i32 = converted %.loc16, %int.convert_checked.loc21_3.2 [template = constants.%.39]
// CHECK:STDOUT:   %.loc21_3.13: Core.IntLiteral = int_value 1 [template = constants.%.36]
// CHECK:STDOUT:   %.loc21_3.14: ref %i32 = array_index %ints.var, %.loc21_3.13
// CHECK:STDOUT:   %.loc21_3.15: init %i32 = initialize_from %.loc21_3.12 to %.loc21_3.14 [template = constants.%.39]
// CHECK:STDOUT:   %.loc21_3.16: %Convert.type.2 = interface_witness_access constants.%.32, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc21_3.17: <bound method> = bound_method %.loc17, %.loc21_3.16 [template = constants.%.33]
// CHECK:STDOUT:   %.loc21_3.18: <specific function> = specific_function %.loc21_3.17, @Convert.2(constants.%.1) [template = constants.%.34]
// CHECK:STDOUT:   %int.convert_checked.loc21_3.3: init %i32 = call %.loc21_3.18(%.loc17) [template = constants.%.35]
// CHECK:STDOUT:   %.loc21_3.19: init %i32 = converted %.loc17, %int.convert_checked.loc21_3.3 [template = constants.%.35]
// CHECK:STDOUT:   %.loc21_3.20: Core.IntLiteral = int_value 2 [template = constants.%.40]
// CHECK:STDOUT:   %.loc21_3.21: ref %i32 = array_index %ints.var, %.loc21_3.20
// CHECK:STDOUT:   %.loc21_3.22: init %i32 = initialize_from %.loc21_3.19 to %.loc21_3.21 [template = constants.%.35]
// CHECK:STDOUT:   %.loc21_3.23: %Convert.type.2 = interface_witness_access constants.%.32, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc21_3.24: <bound method> = bound_method %.loc18, %.loc21_3.23 [template = constants.%.33]
// CHECK:STDOUT:   %.loc21_3.25: <specific function> = specific_function %.loc21_3.24, @Convert.2(constants.%.1) [template = constants.%.34]
// CHECK:STDOUT:   %int.convert_checked.loc21_3.4: init %i32 = call %.loc21_3.25(%.loc18) [template = constants.%.35]
// CHECK:STDOUT:   %.loc21_3.26: init %i32 = converted %.loc18, %int.convert_checked.loc21_3.4 [template = constants.%.35]
// CHECK:STDOUT:   %.loc21_3.27: Core.IntLiteral = int_value 3 [template = constants.%.41]
// CHECK:STDOUT:   %.loc21_3.28: ref %i32 = array_index %ints.var, %.loc21_3.27
// CHECK:STDOUT:   %.loc21_3.29: init %i32 = initialize_from %.loc21_3.26 to %.loc21_3.28 [template = constants.%.35]
// CHECK:STDOUT:   %.loc21_3.30: %Convert.type.2 = interface_witness_access constants.%.32, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc21_3.31: <bound method> = bound_method %.loc19, %.loc21_3.30 [template = constants.%.43]
// CHECK:STDOUT:   %.loc21_3.32: <specific function> = specific_function %.loc21_3.31, @Convert.2(constants.%.1) [template = constants.%.44]
// CHECK:STDOUT:   %int.convert_checked.loc21_3.5: init %i32 = call %.loc21_3.32(%.loc19) [template = constants.%.45]
// CHECK:STDOUT:   %.loc21_3.33: init %i32 = converted %.loc19, %int.convert_checked.loc21_3.5 [template = constants.%.45]
// CHECK:STDOUT:   %.loc21_3.34: Core.IntLiteral = int_value 4 [template = constants.%.42]
// CHECK:STDOUT:   %.loc21_3.35: ref %i32 = array_index %ints.var, %.loc21_3.34
// CHECK:STDOUT:   %.loc21_3.36: init %i32 = initialize_from %.loc21_3.33 to %.loc21_3.35 [template = constants.%.45]
// CHECK:STDOUT:   %.loc21_3.37: %Convert.type.2 = interface_witness_access constants.%.32, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc21_3.38: <bound method> = bound_method %.loc20, %.loc21_3.37 [template = constants.%.43]
// CHECK:STDOUT:   %.loc21_3.39: <specific function> = specific_function %.loc21_3.38, @Convert.2(constants.%.1) [template = constants.%.44]
// CHECK:STDOUT:   %int.convert_checked.loc21_3.6: init %i32 = call %.loc21_3.39(%.loc20) [template = constants.%.45]
// CHECK:STDOUT:   %.loc21_3.40: init %i32 = converted %.loc20, %int.convert_checked.loc21_3.6 [template = constants.%.45]
// CHECK:STDOUT:   %.loc21_3.41: Core.IntLiteral = int_value 5 [template = constants.%.46]
// CHECK:STDOUT:   %.loc21_3.42: ref %i32 = array_index %ints.var, %.loc21_3.41
// CHECK:STDOUT:   %.loc21_3.43: init %i32 = initialize_from %.loc21_3.40 to %.loc21_3.42 [template = constants.%.45]
// CHECK:STDOUT:   %.loc21_3.44: init %.3 = array_init (%.loc21_3.8, %.loc21_3.15, %.loc21_3.22, %.loc21_3.29, %.loc21_3.36, %.loc21_3.43) to %ints.var [template = constants.%array.1]
// CHECK:STDOUT:   %.loc21_4: init %.3 = converted %.loc21_3.1, %.loc21_3.44 [template = constants.%array.1]
// CHECK:STDOUT:   assign %ints.var, %.loc21_4
// CHECK:STDOUT:   %.loc22_16.1: Core.IntLiteral = int_value 64 [template = constants.%.47]
// CHECK:STDOUT:   %float.make_type: init type = call constants.%Float(%.loc22_16.1) [template = f64]
// CHECK:STDOUT:   %.loc22_21: Core.IntLiteral = int_value 6 [template = constants.%.2]
// CHECK:STDOUT:   %.loc22_16.2: type = value_of_initializer %float.make_type [template = f64]
// CHECK:STDOUT:   %.loc22_16.3: type = converted %float.make_type, %.loc22_16.2 [template = f64]
// CHECK:STDOUT:   %.loc22_22: type = array_type %.loc22_21, f64 [template = constants.%.48]
// CHECK:STDOUT:   %floats.var: ref %.48 = var floats
// CHECK:STDOUT:   %floats: ref %.48 = bind_name floats, %floats.var
// CHECK:STDOUT:   %.loc23: f64 = float_literal 0.90000000000000002 [template = constants.%.50]
// CHECK:STDOUT:   %.loc24: f64 = float_literal 8 [template = constants.%.51]
// CHECK:STDOUT:   %.loc25: f64 = float_literal 80 [template = constants.%.52]
// CHECK:STDOUT:   %.loc26: f64 = float_literal 1.0E+7 [template = constants.%.53]
// CHECK:STDOUT:   %.loc27: f64 = float_literal 1.0E+8 [template = constants.%.54]
// CHECK:STDOUT:   %.loc28: f64 = float_literal 1.0E-8 [template = constants.%.55]
// CHECK:STDOUT:   %.loc29_3.1: %tuple.type.2 = tuple_literal (%.loc23, %.loc24, %.loc25, %.loc26, %.loc27, %.loc28)
// CHECK:STDOUT:   %.loc29_3.2: Core.IntLiteral = int_value 0 [template = constants.%.8]
// CHECK:STDOUT:   %.loc29_3.3: ref f64 = array_index %floats.var, %.loc29_3.2
// CHECK:STDOUT:   %.loc29_3.4: init f64 = initialize_from %.loc23 to %.loc29_3.3 [template = constants.%.50]
// CHECK:STDOUT:   %.loc29_3.5: Core.IntLiteral = int_value 1 [template = constants.%.36]
// CHECK:STDOUT:   %.loc29_3.6: ref f64 = array_index %floats.var, %.loc29_3.5
// CHECK:STDOUT:   %.loc29_3.7: init f64 = initialize_from %.loc24 to %.loc29_3.6 [template = constants.%.51]
// CHECK:STDOUT:   %.loc29_3.8: Core.IntLiteral = int_value 2 [template = constants.%.40]
// CHECK:STDOUT:   %.loc29_3.9: ref f64 = array_index %floats.var, %.loc29_3.8
// CHECK:STDOUT:   %.loc29_3.10: init f64 = initialize_from %.loc25 to %.loc29_3.9 [template = constants.%.52]
// CHECK:STDOUT:   %.loc29_3.11: Core.IntLiteral = int_value 3 [template = constants.%.41]
// CHECK:STDOUT:   %.loc29_3.12: ref f64 = array_index %floats.var, %.loc29_3.11
// CHECK:STDOUT:   %.loc29_3.13: init f64 = initialize_from %.loc26 to %.loc29_3.12 [template = constants.%.53]
// CHECK:STDOUT:   %.loc29_3.14: Core.IntLiteral = int_value 4 [template = constants.%.42]
// CHECK:STDOUT:   %.loc29_3.15: ref f64 = array_index %floats.var, %.loc29_3.14
// CHECK:STDOUT:   %.loc29_3.16: init f64 = initialize_from %.loc27 to %.loc29_3.15 [template = constants.%.54]
// CHECK:STDOUT:   %.loc29_3.17: Core.IntLiteral = int_value 5 [template = constants.%.46]
// CHECK:STDOUT:   %.loc29_3.18: ref f64 = array_index %floats.var, %.loc29_3.17
// CHECK:STDOUT:   %.loc29_3.19: init f64 = initialize_from %.loc28 to %.loc29_3.18 [template = constants.%.55]
// CHECK:STDOUT:   %.loc29_3.20: init %.48 = array_init (%.loc29_3.4, %.loc29_3.7, %.loc29_3.10, %.loc29_3.13, %.loc29_3.16, %.loc29_3.19) to %floats.var [template = constants.%array.2]
// CHECK:STDOUT:   %.loc29_4: init %.48 = converted %.loc29_3.1, %.loc29_3.20 [template = constants.%array.2]
// CHECK:STDOUT:   assign %floats.var, %.loc29_4
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
