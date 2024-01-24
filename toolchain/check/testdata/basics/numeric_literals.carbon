// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() {
  // 8 and 9 trigger special behavior in APInt when mishandling signed versus
  // unsigned, so we pay extra attention to those.
  var ints: [i32; 5] = (
    8,
    9,
    0x8,
    0b1000,
    39999999999999999993,
  );
  var floats: [f64; 7] = (
    0.9,
    8.0,
    80.0,
    1.0e7,
    1.0e8,
    1.0e-8,
    39999999999999999993.0e39999999999999999993,
  );
}

// CHECK:STDOUT: --- numeric_literals.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 5 [template]
// CHECK:STDOUT:   %.2: type = array_type %.1, i32 [template]
// CHECK:STDOUT:   %.3: type = ptr_type [i32; 5] [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 8 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 9 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 39999999999999999993 [template]
// CHECK:STDOUT:   %.7: type = tuple_type (i32, i32, i32, i32, i32) [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.9: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.10: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.11: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.12: i32 = int_literal 4 [template]
// CHECK:STDOUT:   %.13: [i32; 5] = tuple_value (%.4, %.5, %.4, %.4, %.6) [template]
// CHECK:STDOUT:   %.14: i32 = int_literal 7 [template]
// CHECK:STDOUT:   %.15: type = array_type %.14, f64 [template]
// CHECK:STDOUT:   %.16: type = ptr_type [f64; 7] [template]
// CHECK:STDOUT:   %.17: f64 = real_literal 9e-1 [template]
// CHECK:STDOUT:   %.18: f64 = real_literal 80e-1 [template]
// CHECK:STDOUT:   %.19: f64 = real_literal 800e-1 [template]
// CHECK:STDOUT:   %.20: f64 = real_literal 10e6 [template]
// CHECK:STDOUT:   %.21: f64 = real_literal 10e7 [template]
// CHECK:STDOUT:   %.22: f64 = real_literal 10e-9 [template]
// CHECK:STDOUT:   %.23: f64 = real_literal 399999999999999999930e39999999999999999992 [template]
// CHECK:STDOUT:   %.24: type = tuple_type (f64, f64, f64, f64, f64, f64, f64) [template]
// CHECK:STDOUT:   %.25: i32 = int_literal 5 [template]
// CHECK:STDOUT:   %.26: i32 = int_literal 6 [template]
// CHECK:STDOUT:   %.27: [f64; 7] = tuple_value (%.17, %.18, %.19, %.20, %.21, %.22, %.23) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.F = %F} [template]
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_19: i32 = int_literal 5 [template = constants.%.1]
// CHECK:STDOUT:   %.loc10_20: type = array_type %.loc10_19, i32 [template = constants.%.2]
// CHECK:STDOUT:   %ints.var: ref [i32; 5] = var ints
// CHECK:STDOUT:   %ints: ref [i32; 5] = bind_name ints, %ints.var
// CHECK:STDOUT:   %.loc11: i32 = int_literal 8 [template = constants.%.4]
// CHECK:STDOUT:   %.loc12: i32 = int_literal 9 [template = constants.%.5]
// CHECK:STDOUT:   %.loc13: i32 = int_literal 8 [template = constants.%.4]
// CHECK:STDOUT:   %.loc14: i32 = int_literal 8 [template = constants.%.4]
// CHECK:STDOUT:   %.loc15: i32 = int_literal 39999999999999999993 [template = constants.%.6]
// CHECK:STDOUT:   %.loc16_3.1: (i32, i32, i32, i32, i32) = tuple_literal (%.loc11, %.loc12, %.loc13, %.loc14, %.loc15)
// CHECK:STDOUT:   %.loc16_3.2: i32 = int_literal 0 [template = constants.%.8]
// CHECK:STDOUT:   %.loc16_3.3: ref i32 = array_index %ints.var, %.loc16_3.2
// CHECK:STDOUT:   %.loc16_3.4: init i32 = initialize_from %.loc11 to %.loc16_3.3 [template = constants.%.4]
// CHECK:STDOUT:   %.loc16_3.5: i32 = int_literal 1 [template = constants.%.9]
// CHECK:STDOUT:   %.loc16_3.6: ref i32 = array_index %ints.var, %.loc16_3.5
// CHECK:STDOUT:   %.loc16_3.7: init i32 = initialize_from %.loc12 to %.loc16_3.6 [template = constants.%.5]
// CHECK:STDOUT:   %.loc16_3.8: i32 = int_literal 2 [template = constants.%.10]
// CHECK:STDOUT:   %.loc16_3.9: ref i32 = array_index %ints.var, %.loc16_3.8
// CHECK:STDOUT:   %.loc16_3.10: init i32 = initialize_from %.loc13 to %.loc16_3.9 [template = constants.%.4]
// CHECK:STDOUT:   %.loc16_3.11: i32 = int_literal 3 [template = constants.%.11]
// CHECK:STDOUT:   %.loc16_3.12: ref i32 = array_index %ints.var, %.loc16_3.11
// CHECK:STDOUT:   %.loc16_3.13: init i32 = initialize_from %.loc14 to %.loc16_3.12 [template = constants.%.4]
// CHECK:STDOUT:   %.loc16_3.14: i32 = int_literal 4 [template = constants.%.12]
// CHECK:STDOUT:   %.loc16_3.15: ref i32 = array_index %ints.var, %.loc16_3.14
// CHECK:STDOUT:   %.loc16_3.16: init i32 = initialize_from %.loc15 to %.loc16_3.15 [template = constants.%.6]
// CHECK:STDOUT:   %.loc16_3.17: init [i32; 5] = array_init (%.loc16_3.4, %.loc16_3.7, %.loc16_3.10, %.loc16_3.13, %.loc16_3.16) to %ints.var [template = constants.%.13]
// CHECK:STDOUT:   %.loc16_3.18: init [i32; 5] = converted %.loc16_3.1, %.loc16_3.17 [template = constants.%.13]
// CHECK:STDOUT:   assign %ints.var, %.loc16_3.18
// CHECK:STDOUT:   %.loc17_21: i32 = int_literal 7 [template = constants.%.14]
// CHECK:STDOUT:   %.loc17_22: type = array_type %.loc17_21, f64 [template = constants.%.15]
// CHECK:STDOUT:   %floats.var: ref [f64; 7] = var floats
// CHECK:STDOUT:   %floats: ref [f64; 7] = bind_name floats, %floats.var
// CHECK:STDOUT:   %.loc18: f64 = real_literal 9e-1 [template = constants.%.17]
// CHECK:STDOUT:   %.loc19: f64 = real_literal 80e-1 [template = constants.%.18]
// CHECK:STDOUT:   %.loc20: f64 = real_literal 800e-1 [template = constants.%.19]
// CHECK:STDOUT:   %.loc21: f64 = real_literal 10e6 [template = constants.%.20]
// CHECK:STDOUT:   %.loc22: f64 = real_literal 10e7 [template = constants.%.21]
// CHECK:STDOUT:   %.loc23: f64 = real_literal 10e-9 [template = constants.%.22]
// CHECK:STDOUT:   %.loc24: f64 = real_literal 399999999999999999930e39999999999999999992 [template = constants.%.23]
// CHECK:STDOUT:   %.loc25_3.1: (f64, f64, f64, f64, f64, f64, f64) = tuple_literal (%.loc18, %.loc19, %.loc20, %.loc21, %.loc22, %.loc23, %.loc24)
// CHECK:STDOUT:   %.loc25_3.2: i32 = int_literal 0 [template = constants.%.8]
// CHECK:STDOUT:   %.loc25_3.3: ref f64 = array_index %floats.var, %.loc25_3.2
// CHECK:STDOUT:   %.loc25_3.4: init f64 = initialize_from %.loc18 to %.loc25_3.3 [template = constants.%.17]
// CHECK:STDOUT:   %.loc25_3.5: i32 = int_literal 1 [template = constants.%.9]
// CHECK:STDOUT:   %.loc25_3.6: ref f64 = array_index %floats.var, %.loc25_3.5
// CHECK:STDOUT:   %.loc25_3.7: init f64 = initialize_from %.loc19 to %.loc25_3.6 [template = constants.%.18]
// CHECK:STDOUT:   %.loc25_3.8: i32 = int_literal 2 [template = constants.%.10]
// CHECK:STDOUT:   %.loc25_3.9: ref f64 = array_index %floats.var, %.loc25_3.8
// CHECK:STDOUT:   %.loc25_3.10: init f64 = initialize_from %.loc20 to %.loc25_3.9 [template = constants.%.19]
// CHECK:STDOUT:   %.loc25_3.11: i32 = int_literal 3 [template = constants.%.11]
// CHECK:STDOUT:   %.loc25_3.12: ref f64 = array_index %floats.var, %.loc25_3.11
// CHECK:STDOUT:   %.loc25_3.13: init f64 = initialize_from %.loc21 to %.loc25_3.12 [template = constants.%.20]
// CHECK:STDOUT:   %.loc25_3.14: i32 = int_literal 4 [template = constants.%.12]
// CHECK:STDOUT:   %.loc25_3.15: ref f64 = array_index %floats.var, %.loc25_3.14
// CHECK:STDOUT:   %.loc25_3.16: init f64 = initialize_from %.loc22 to %.loc25_3.15 [template = constants.%.21]
// CHECK:STDOUT:   %.loc25_3.17: i32 = int_literal 5 [template = constants.%.25]
// CHECK:STDOUT:   %.loc25_3.18: ref f64 = array_index %floats.var, %.loc25_3.17
// CHECK:STDOUT:   %.loc25_3.19: init f64 = initialize_from %.loc23 to %.loc25_3.18 [template = constants.%.22]
// CHECK:STDOUT:   %.loc25_3.20: i32 = int_literal 6 [template = constants.%.26]
// CHECK:STDOUT:   %.loc25_3.21: ref f64 = array_index %floats.var, %.loc25_3.20
// CHECK:STDOUT:   %.loc25_3.22: init f64 = initialize_from %.loc24 to %.loc25_3.21 [template = constants.%.23]
// CHECK:STDOUT:   %.loc25_3.23: init [f64; 7] = array_init (%.loc25_3.4, %.loc25_3.7, %.loc25_3.10, %.loc25_3.13, %.loc25_3.16, %.loc25_3.19, %.loc25_3.22) to %floats.var [template = constants.%.27]
// CHECK:STDOUT:   %.loc25_3.24: init [f64; 7] = converted %.loc25_3.1, %.loc25_3.23 [template = constants.%.27]
// CHECK:STDOUT:   assign %floats.var, %.loc25_3.24
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
