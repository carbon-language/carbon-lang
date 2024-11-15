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
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.1: i32 = int_value 6 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.25: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.26: <bound method> = bound_method %.1, %Convert.15 [template]
// CHECK:STDOUT:   %.27: Core.IntLiteral = int_value 6 [template]
// CHECK:STDOUT:   %.28: type = array_type %.27, i32 [template]
// CHECK:STDOUT:   %.30: i32 = int_value 8 [template]
// CHECK:STDOUT:   %.31: i32 = int_value 9 [template]
// CHECK:STDOUT:   %.32: i32 = int_value 2147483647 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (i32, i32, i32, i32, i32, i32) [template]
// CHECK:STDOUT:   %.33: i32 = int_value 0 [template]
// CHECK:STDOUT:   %.34: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.35: i32 = int_value 2 [template]
// CHECK:STDOUT:   %.36: i32 = int_value 3 [template]
// CHECK:STDOUT:   %.37: i32 = int_value 4 [template]
// CHECK:STDOUT:   %.38: i32 = int_value 5 [template]
// CHECK:STDOUT:   %array.1: %.28 = tuple_value (%.30, %.31, %.30, %.30, %.32, %.32) [template]
// CHECK:STDOUT:   %.39: Core.IntLiteral = int_value 64 [template]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [template]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [template]
// CHECK:STDOUT:   %.40: type = array_type %.27, f64 [template]
// CHECK:STDOUT:   %.42: f64 = float_literal 0.90000000000000002 [template]
// CHECK:STDOUT:   %.43: f64 = float_literal 8 [template]
// CHECK:STDOUT:   %.44: f64 = float_literal 80 [template]
// CHECK:STDOUT:   %.45: f64 = float_literal 1.0E+7 [template]
// CHECK:STDOUT:   %.46: f64 = float_literal 1.0E+8 [template]
// CHECK:STDOUT:   %.47: f64 = float_literal 1.0E-8 [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (f64, f64, f64, f64, f64, f64) [template]
// CHECK:STDOUT:   %array.2: %.40 = tuple_value (%.42, %.43, %.44, %.45, %.46, %.47) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     .Float = %import_ref.51
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
// CHECK:STDOUT:   %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc14_19.1: i32 = int_value 6 [template = constants.%.1]
// CHECK:STDOUT:   %.loc14_14.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc14_14.2: type = converted %int.make_type_32, %.loc14_14.1 [template = i32]
// CHECK:STDOUT:   %.loc14_19.2: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc14_19.3: <bound method> = bound_method %.loc14_19.1, %.loc14_19.2 [template = constants.%.26]
// CHECK:STDOUT:   %int.convert_checked.loc14: init Core.IntLiteral = call %.loc14_19.3(%.loc14_19.1) [template = constants.%.27]
// CHECK:STDOUT:   %.loc14_19.4: Core.IntLiteral = value_of_initializer %int.convert_checked.loc14 [template = constants.%.27]
// CHECK:STDOUT:   %.loc14_19.5: Core.IntLiteral = converted %.loc14_19.1, %.loc14_19.4 [template = constants.%.27]
// CHECK:STDOUT:   %.loc14_20: type = array_type %.loc14_19.5, i32 [template = constants.%.28]
// CHECK:STDOUT:   %ints.var: ref %.28 = var ints
// CHECK:STDOUT:   %ints: ref %.28 = bind_name ints, %ints.var
// CHECK:STDOUT:   %.loc15: i32 = int_value 8 [template = constants.%.30]
// CHECK:STDOUT:   %.loc16: i32 = int_value 9 [template = constants.%.31]
// CHECK:STDOUT:   %.loc17: i32 = int_value 8 [template = constants.%.30]
// CHECK:STDOUT:   %.loc18: i32 = int_value 8 [template = constants.%.30]
// CHECK:STDOUT:   %.loc19: i32 = int_value 2147483647 [template = constants.%.32]
// CHECK:STDOUT:   %.loc20: i32 = int_value 2147483647 [template = constants.%.32]
// CHECK:STDOUT:   %.loc21_3.1: %tuple.type.1 = tuple_literal (%.loc15, %.loc16, %.loc17, %.loc18, %.loc19, %.loc20)
// CHECK:STDOUT:   %.loc21_3.2: i32 = int_value 0 [template = constants.%.33]
// CHECK:STDOUT:   %.loc21_3.3: ref i32 = array_index %ints.var, %.loc21_3.2
// CHECK:STDOUT:   %.loc21_3.4: init i32 = initialize_from %.loc15 to %.loc21_3.3 [template = constants.%.30]
// CHECK:STDOUT:   %.loc21_3.5: i32 = int_value 1 [template = constants.%.34]
// CHECK:STDOUT:   %.loc21_3.6: ref i32 = array_index %ints.var, %.loc21_3.5
// CHECK:STDOUT:   %.loc21_3.7: init i32 = initialize_from %.loc16 to %.loc21_3.6 [template = constants.%.31]
// CHECK:STDOUT:   %.loc21_3.8: i32 = int_value 2 [template = constants.%.35]
// CHECK:STDOUT:   %.loc21_3.9: ref i32 = array_index %ints.var, %.loc21_3.8
// CHECK:STDOUT:   %.loc21_3.10: init i32 = initialize_from %.loc17 to %.loc21_3.9 [template = constants.%.30]
// CHECK:STDOUT:   %.loc21_3.11: i32 = int_value 3 [template = constants.%.36]
// CHECK:STDOUT:   %.loc21_3.12: ref i32 = array_index %ints.var, %.loc21_3.11
// CHECK:STDOUT:   %.loc21_3.13: init i32 = initialize_from %.loc18 to %.loc21_3.12 [template = constants.%.30]
// CHECK:STDOUT:   %.loc21_3.14: i32 = int_value 4 [template = constants.%.37]
// CHECK:STDOUT:   %.loc21_3.15: ref i32 = array_index %ints.var, %.loc21_3.14
// CHECK:STDOUT:   %.loc21_3.16: init i32 = initialize_from %.loc19 to %.loc21_3.15 [template = constants.%.32]
// CHECK:STDOUT:   %.loc21_3.17: i32 = int_value 5 [template = constants.%.38]
// CHECK:STDOUT:   %.loc21_3.18: ref i32 = array_index %ints.var, %.loc21_3.17
// CHECK:STDOUT:   %.loc21_3.19: init i32 = initialize_from %.loc20 to %.loc21_3.18 [template = constants.%.32]
// CHECK:STDOUT:   %.loc21_3.20: init %.28 = array_init (%.loc21_3.4, %.loc21_3.7, %.loc21_3.10, %.loc21_3.13, %.loc21_3.16, %.loc21_3.19) to %ints.var [template = constants.%array.1]
// CHECK:STDOUT:   %.loc21_4: init %.28 = converted %.loc21_3.1, %.loc21_3.20 [template = constants.%array.1]
// CHECK:STDOUT:   assign %ints.var, %.loc21_4
// CHECK:STDOUT:   %.loc22_16.1: Core.IntLiteral = int_value 64 [template = constants.%.39]
// CHECK:STDOUT:   %float.make_type: init type = call constants.%Float(%.loc22_16.1) [template = f64]
// CHECK:STDOUT:   %.loc22_21.1: i32 = int_value 6 [template = constants.%.1]
// CHECK:STDOUT:   %.loc22_16.2: type = value_of_initializer %float.make_type [template = f64]
// CHECK:STDOUT:   %.loc22_16.3: type = converted %float.make_type, %.loc22_16.2 [template = f64]
// CHECK:STDOUT:   %.loc22_21.2: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc22_21.3: <bound method> = bound_method %.loc22_21.1, %.loc22_21.2 [template = constants.%.26]
// CHECK:STDOUT:   %int.convert_checked.loc22: init Core.IntLiteral = call %.loc22_21.3(%.loc22_21.1) [template = constants.%.27]
// CHECK:STDOUT:   %.loc22_21.4: Core.IntLiteral = value_of_initializer %int.convert_checked.loc22 [template = constants.%.27]
// CHECK:STDOUT:   %.loc22_21.5: Core.IntLiteral = converted %.loc22_21.1, %.loc22_21.4 [template = constants.%.27]
// CHECK:STDOUT:   %.loc22_22: type = array_type %.loc22_21.5, f64 [template = constants.%.40]
// CHECK:STDOUT:   %floats.var: ref %.40 = var floats
// CHECK:STDOUT:   %floats: ref %.40 = bind_name floats, %floats.var
// CHECK:STDOUT:   %.loc23: f64 = float_literal 0.90000000000000002 [template = constants.%.42]
// CHECK:STDOUT:   %.loc24: f64 = float_literal 8 [template = constants.%.43]
// CHECK:STDOUT:   %.loc25: f64 = float_literal 80 [template = constants.%.44]
// CHECK:STDOUT:   %.loc26: f64 = float_literal 1.0E+7 [template = constants.%.45]
// CHECK:STDOUT:   %.loc27: f64 = float_literal 1.0E+8 [template = constants.%.46]
// CHECK:STDOUT:   %.loc28: f64 = float_literal 1.0E-8 [template = constants.%.47]
// CHECK:STDOUT:   %.loc29_3.1: %tuple.type.2 = tuple_literal (%.loc23, %.loc24, %.loc25, %.loc26, %.loc27, %.loc28)
// CHECK:STDOUT:   %.loc29_3.2: i32 = int_value 0 [template = constants.%.33]
// CHECK:STDOUT:   %.loc29_3.3: ref f64 = array_index %floats.var, %.loc29_3.2
// CHECK:STDOUT:   %.loc29_3.4: init f64 = initialize_from %.loc23 to %.loc29_3.3 [template = constants.%.42]
// CHECK:STDOUT:   %.loc29_3.5: i32 = int_value 1 [template = constants.%.34]
// CHECK:STDOUT:   %.loc29_3.6: ref f64 = array_index %floats.var, %.loc29_3.5
// CHECK:STDOUT:   %.loc29_3.7: init f64 = initialize_from %.loc24 to %.loc29_3.6 [template = constants.%.43]
// CHECK:STDOUT:   %.loc29_3.8: i32 = int_value 2 [template = constants.%.35]
// CHECK:STDOUT:   %.loc29_3.9: ref f64 = array_index %floats.var, %.loc29_3.8
// CHECK:STDOUT:   %.loc29_3.10: init f64 = initialize_from %.loc25 to %.loc29_3.9 [template = constants.%.44]
// CHECK:STDOUT:   %.loc29_3.11: i32 = int_value 3 [template = constants.%.36]
// CHECK:STDOUT:   %.loc29_3.12: ref f64 = array_index %floats.var, %.loc29_3.11
// CHECK:STDOUT:   %.loc29_3.13: init f64 = initialize_from %.loc26 to %.loc29_3.12 [template = constants.%.45]
// CHECK:STDOUT:   %.loc29_3.14: i32 = int_value 4 [template = constants.%.37]
// CHECK:STDOUT:   %.loc29_3.15: ref f64 = array_index %floats.var, %.loc29_3.14
// CHECK:STDOUT:   %.loc29_3.16: init f64 = initialize_from %.loc27 to %.loc29_3.15 [template = constants.%.46]
// CHECK:STDOUT:   %.loc29_3.17: i32 = int_value 5 [template = constants.%.38]
// CHECK:STDOUT:   %.loc29_3.18: ref f64 = array_index %floats.var, %.loc29_3.17
// CHECK:STDOUT:   %.loc29_3.19: init f64 = initialize_from %.loc28 to %.loc29_3.18 [template = constants.%.47]
// CHECK:STDOUT:   %.loc29_3.20: init %.40 = array_init (%.loc29_3.4, %.loc29_3.7, %.loc29_3.10, %.loc29_3.13, %.loc29_3.16, %.loc29_3.19) to %floats.var [template = constants.%array.2]
// CHECK:STDOUT:   %.loc29_4: init %.40 = converted %.loc29_3.1, %.loc29_3.20 [template = constants.%array.2]
// CHECK:STDOUT:   assign %floats.var, %.loc29_4
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
