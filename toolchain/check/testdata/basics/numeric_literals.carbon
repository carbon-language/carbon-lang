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
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.1: i32 = int_value 6 [template]
// CHECK:STDOUT:   %.2: type = array_type %.1, i32 [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.2 [template]
// CHECK:STDOUT:   %.4: i32 = int_value 8 [template]
// CHECK:STDOUT:   %.5: i32 = int_value 9 [template]
// CHECK:STDOUT:   %.6: i32 = int_value 2147483647 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (i32, i32, i32, i32, i32, i32) [template]
// CHECK:STDOUT:   %.7: i32 = int_value 0 [template]
// CHECK:STDOUT:   %.8: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.9: i32 = int_value 2 [template]
// CHECK:STDOUT:   %.10: i32 = int_value 3 [template]
// CHECK:STDOUT:   %.11: i32 = int_value 4 [template]
// CHECK:STDOUT:   %.12: i32 = int_value 5 [template]
// CHECK:STDOUT:   %array.1: %.2 = tuple_value (%.4, %.5, %.4, %.4, %.6, %.6) [template]
// CHECK:STDOUT:   %.13: Core.IntLiteral = int_value 64 [template]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [template]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [template]
// CHECK:STDOUT:   %.14: type = array_type %.1, f64 [template]
// CHECK:STDOUT:   %.15: type = ptr_type %.14 [template]
// CHECK:STDOUT:   %.16: f64 = float_literal 0.90000000000000002 [template]
// CHECK:STDOUT:   %.17: f64 = float_literal 8 [template]
// CHECK:STDOUT:   %.18: f64 = float_literal 80 [template]
// CHECK:STDOUT:   %.19: f64 = float_literal 1.0E+7 [template]
// CHECK:STDOUT:   %.20: f64 = float_literal 1.0E+8 [template]
// CHECK:STDOUT:   %.21: f64 = float_literal 1.0E-8 [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (f64, f64, f64, f64, f64, f64) [template]
// CHECK:STDOUT:   %array.2: %.14 = tuple_value (%.16, %.17, %.18, %.19, %.20, %.21) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .Float = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: %Int32.type = import_ref Core//prelude/types, inst+15, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %import_ref.2: %Float.type = import_ref Core//prelude/types, inst+60, loaded [template = constants.%Float]
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
// CHECK:STDOUT:   %.loc14_19: i32 = int_value 6 [template = constants.%.1]
// CHECK:STDOUT:   %.loc14_14.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:   %.loc14_14.2: type = converted %int.make_type_32, %.loc14_14.1 [template = i32]
// CHECK:STDOUT:   %.loc14_20: type = array_type %.loc14_19, i32 [template = constants.%.2]
// CHECK:STDOUT:   %ints.var: ref %.2 = var ints
// CHECK:STDOUT:   %ints: ref %.2 = bind_name ints, %ints.var
// CHECK:STDOUT:   %.loc15: i32 = int_value 8 [template = constants.%.4]
// CHECK:STDOUT:   %.loc16: i32 = int_value 9 [template = constants.%.5]
// CHECK:STDOUT:   %.loc17: i32 = int_value 8 [template = constants.%.4]
// CHECK:STDOUT:   %.loc18: i32 = int_value 8 [template = constants.%.4]
// CHECK:STDOUT:   %.loc19: i32 = int_value 2147483647 [template = constants.%.6]
// CHECK:STDOUT:   %.loc20: i32 = int_value 2147483647 [template = constants.%.6]
// CHECK:STDOUT:   %.loc21_3.1: %tuple.type.1 = tuple_literal (%.loc15, %.loc16, %.loc17, %.loc18, %.loc19, %.loc20)
// CHECK:STDOUT:   %.loc21_3.2: i32 = int_value 0 [template = constants.%.7]
// CHECK:STDOUT:   %.loc21_3.3: ref i32 = array_index %ints.var, %.loc21_3.2
// CHECK:STDOUT:   %.loc21_3.4: init i32 = initialize_from %.loc15 to %.loc21_3.3 [template = constants.%.4]
// CHECK:STDOUT:   %.loc21_3.5: i32 = int_value 1 [template = constants.%.8]
// CHECK:STDOUT:   %.loc21_3.6: ref i32 = array_index %ints.var, %.loc21_3.5
// CHECK:STDOUT:   %.loc21_3.7: init i32 = initialize_from %.loc16 to %.loc21_3.6 [template = constants.%.5]
// CHECK:STDOUT:   %.loc21_3.8: i32 = int_value 2 [template = constants.%.9]
// CHECK:STDOUT:   %.loc21_3.9: ref i32 = array_index %ints.var, %.loc21_3.8
// CHECK:STDOUT:   %.loc21_3.10: init i32 = initialize_from %.loc17 to %.loc21_3.9 [template = constants.%.4]
// CHECK:STDOUT:   %.loc21_3.11: i32 = int_value 3 [template = constants.%.10]
// CHECK:STDOUT:   %.loc21_3.12: ref i32 = array_index %ints.var, %.loc21_3.11
// CHECK:STDOUT:   %.loc21_3.13: init i32 = initialize_from %.loc18 to %.loc21_3.12 [template = constants.%.4]
// CHECK:STDOUT:   %.loc21_3.14: i32 = int_value 4 [template = constants.%.11]
// CHECK:STDOUT:   %.loc21_3.15: ref i32 = array_index %ints.var, %.loc21_3.14
// CHECK:STDOUT:   %.loc21_3.16: init i32 = initialize_from %.loc19 to %.loc21_3.15 [template = constants.%.6]
// CHECK:STDOUT:   %.loc21_3.17: i32 = int_value 5 [template = constants.%.12]
// CHECK:STDOUT:   %.loc21_3.18: ref i32 = array_index %ints.var, %.loc21_3.17
// CHECK:STDOUT:   %.loc21_3.19: init i32 = initialize_from %.loc20 to %.loc21_3.18 [template = constants.%.6]
// CHECK:STDOUT:   %.loc21_3.20: init %.2 = array_init (%.loc21_3.4, %.loc21_3.7, %.loc21_3.10, %.loc21_3.13, %.loc21_3.16, %.loc21_3.19) to %ints.var [template = constants.%array.1]
// CHECK:STDOUT:   %.loc21_4: init %.2 = converted %.loc21_3.1, %.loc21_3.20 [template = constants.%array.1]
// CHECK:STDOUT:   assign %ints.var, %.loc21_4
// CHECK:STDOUT:   %.loc22_16.1: Core.IntLiteral = int_value 64 [template = constants.%.13]
// CHECK:STDOUT:   %float.make_type: init type = call constants.%Float(%.loc22_16.1) [template = f64]
// CHECK:STDOUT:   %.loc22_21: i32 = int_value 6 [template = constants.%.1]
// CHECK:STDOUT:   %.loc22_16.2: type = value_of_initializer %float.make_type [template = f64]
// CHECK:STDOUT:   %.loc22_16.3: type = converted %float.make_type, %.loc22_16.2 [template = f64]
// CHECK:STDOUT:   %.loc22_22: type = array_type %.loc22_21, f64 [template = constants.%.14]
// CHECK:STDOUT:   %floats.var: ref %.14 = var floats
// CHECK:STDOUT:   %floats: ref %.14 = bind_name floats, %floats.var
// CHECK:STDOUT:   %.loc23: f64 = float_literal 0.90000000000000002 [template = constants.%.16]
// CHECK:STDOUT:   %.loc24: f64 = float_literal 8 [template = constants.%.17]
// CHECK:STDOUT:   %.loc25: f64 = float_literal 80 [template = constants.%.18]
// CHECK:STDOUT:   %.loc26: f64 = float_literal 1.0E+7 [template = constants.%.19]
// CHECK:STDOUT:   %.loc27: f64 = float_literal 1.0E+8 [template = constants.%.20]
// CHECK:STDOUT:   %.loc28: f64 = float_literal 1.0E-8 [template = constants.%.21]
// CHECK:STDOUT:   %.loc29_3.1: %tuple.type.2 = tuple_literal (%.loc23, %.loc24, %.loc25, %.loc26, %.loc27, %.loc28)
// CHECK:STDOUT:   %.loc29_3.2: i32 = int_value 0 [template = constants.%.7]
// CHECK:STDOUT:   %.loc29_3.3: ref f64 = array_index %floats.var, %.loc29_3.2
// CHECK:STDOUT:   %.loc29_3.4: init f64 = initialize_from %.loc23 to %.loc29_3.3 [template = constants.%.16]
// CHECK:STDOUT:   %.loc29_3.5: i32 = int_value 1 [template = constants.%.8]
// CHECK:STDOUT:   %.loc29_3.6: ref f64 = array_index %floats.var, %.loc29_3.5
// CHECK:STDOUT:   %.loc29_3.7: init f64 = initialize_from %.loc24 to %.loc29_3.6 [template = constants.%.17]
// CHECK:STDOUT:   %.loc29_3.8: i32 = int_value 2 [template = constants.%.9]
// CHECK:STDOUT:   %.loc29_3.9: ref f64 = array_index %floats.var, %.loc29_3.8
// CHECK:STDOUT:   %.loc29_3.10: init f64 = initialize_from %.loc25 to %.loc29_3.9 [template = constants.%.18]
// CHECK:STDOUT:   %.loc29_3.11: i32 = int_value 3 [template = constants.%.10]
// CHECK:STDOUT:   %.loc29_3.12: ref f64 = array_index %floats.var, %.loc29_3.11
// CHECK:STDOUT:   %.loc29_3.13: init f64 = initialize_from %.loc26 to %.loc29_3.12 [template = constants.%.19]
// CHECK:STDOUT:   %.loc29_3.14: i32 = int_value 4 [template = constants.%.11]
// CHECK:STDOUT:   %.loc29_3.15: ref f64 = array_index %floats.var, %.loc29_3.14
// CHECK:STDOUT:   %.loc29_3.16: init f64 = initialize_from %.loc27 to %.loc29_3.15 [template = constants.%.20]
// CHECK:STDOUT:   %.loc29_3.17: i32 = int_value 5 [template = constants.%.12]
// CHECK:STDOUT:   %.loc29_3.18: ref f64 = array_index %floats.var, %.loc29_3.17
// CHECK:STDOUT:   %.loc29_3.19: init f64 = initialize_from %.loc28 to %.loc29_3.18 [template = constants.%.21]
// CHECK:STDOUT:   %.loc29_3.20: init %.14 = array_init (%.loc29_3.4, %.loc29_3.7, %.loc29_3.10, %.loc29_3.13, %.loc29_3.16, %.loc29_3.19) to %floats.var [template = constants.%array.2]
// CHECK:STDOUT:   %.loc29_4: init %.14 = converted %.loc29_3.1, %.loc29_3.20 [template = constants.%array.2]
// CHECK:STDOUT:   assign %floats.var, %.loc29_4
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Float(%size.param_patt: Core.IntLiteral) -> type = "float.make_type";
// CHECK:STDOUT:
