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

// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc10: type = ptr_type [i32; 5]
// CHECK:STDOUT:   %.loc16: type = tuple_type (i32, i32, i32, i32, i32)
// CHECK:STDOUT:   %.loc17: type = ptr_type [f64; 7]
// CHECK:STDOUT:   %.loc25: type = tuple_type (f64, f64, f64, f64, f64, f64, f64)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "numeric_literals.carbon" {
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_19: i32 = int_literal 5
// CHECK:STDOUT:   %.loc10_20: type = array_type %.loc10_19, i32
// CHECK:STDOUT:   %ints.var: ref [i32; 5] = var ints
// CHECK:STDOUT:   %ints: ref [i32; 5] = bind_name ints, %ints.var
// CHECK:STDOUT:   %.loc11: i32 = int_literal 8
// CHECK:STDOUT:   %.loc12: i32 = int_literal 9
// CHECK:STDOUT:   %.loc13: i32 = int_literal 8
// CHECK:STDOUT:   %.loc14: i32 = int_literal 8
// CHECK:STDOUT:   %.loc15: i32 = int_literal 39999999999999999993
// CHECK:STDOUT:   %.loc16_3.1: (i32, i32, i32, i32, i32) = tuple_literal (%.loc11, %.loc12, %.loc13, %.loc14, %.loc15)
// CHECK:STDOUT:   %.loc16_3.2: i32 = int_literal 0
// CHECK:STDOUT:   %.loc16_3.3: ref i32 = array_index %ints.var, %.loc16_3.2
// CHECK:STDOUT:   %.loc16_3.4: init i32 = initialize_from %.loc11 to %.loc16_3.3
// CHECK:STDOUT:   %.loc16_3.5: i32 = int_literal 1
// CHECK:STDOUT:   %.loc16_3.6: ref i32 = array_index %ints.var, %.loc16_3.5
// CHECK:STDOUT:   %.loc16_3.7: init i32 = initialize_from %.loc12 to %.loc16_3.6
// CHECK:STDOUT:   %.loc16_3.8: i32 = int_literal 2
// CHECK:STDOUT:   %.loc16_3.9: ref i32 = array_index %ints.var, %.loc16_3.8
// CHECK:STDOUT:   %.loc16_3.10: init i32 = initialize_from %.loc13 to %.loc16_3.9
// CHECK:STDOUT:   %.loc16_3.11: i32 = int_literal 3
// CHECK:STDOUT:   %.loc16_3.12: ref i32 = array_index %ints.var, %.loc16_3.11
// CHECK:STDOUT:   %.loc16_3.13: init i32 = initialize_from %.loc14 to %.loc16_3.12
// CHECK:STDOUT:   %.loc16_3.14: i32 = int_literal 4
// CHECK:STDOUT:   %.loc16_3.15: ref i32 = array_index %ints.var, %.loc16_3.14
// CHECK:STDOUT:   %.loc16_3.16: init i32 = initialize_from %.loc15 to %.loc16_3.15
// CHECK:STDOUT:   %.loc16_3.17: init [i32; 5] = array_init (%.loc16_3.4, %.loc16_3.7, %.loc16_3.10, %.loc16_3.13, %.loc16_3.16) to %ints.var
// CHECK:STDOUT:   %.loc16_3.18: init [i32; 5] = converted %.loc16_3.1, %.loc16_3.17
// CHECK:STDOUT:   assign %ints.var, %.loc16_3.18
// CHECK:STDOUT:   %.loc17_21: i32 = int_literal 7
// CHECK:STDOUT:   %.loc17_22: type = array_type %.loc17_21, f64
// CHECK:STDOUT:   %floats.var: ref [f64; 7] = var floats
// CHECK:STDOUT:   %floats: ref [f64; 7] = bind_name floats, %floats.var
// CHECK:STDOUT:   %.loc18: f64 = real_literal 9e-1
// CHECK:STDOUT:   %.loc19: f64 = real_literal 80e-1
// CHECK:STDOUT:   %.loc20: f64 = real_literal 800e-1
// CHECK:STDOUT:   %.loc21: f64 = real_literal 10e6
// CHECK:STDOUT:   %.loc22: f64 = real_literal 10e7
// CHECK:STDOUT:   %.loc23: f64 = real_literal 10e-9
// CHECK:STDOUT:   %.loc24: f64 = real_literal 399999999999999999930e39999999999999999992
// CHECK:STDOUT:   %.loc25_3.1: (f64, f64, f64, f64, f64, f64, f64) = tuple_literal (%.loc18, %.loc19, %.loc20, %.loc21, %.loc22, %.loc23, %.loc24)
// CHECK:STDOUT:   %.loc25_3.2: i32 = int_literal 0
// CHECK:STDOUT:   %.loc25_3.3: ref f64 = array_index %floats.var, %.loc25_3.2
// CHECK:STDOUT:   %.loc25_3.4: init f64 = initialize_from %.loc18 to %.loc25_3.3
// CHECK:STDOUT:   %.loc25_3.5: i32 = int_literal 1
// CHECK:STDOUT:   %.loc25_3.6: ref f64 = array_index %floats.var, %.loc25_3.5
// CHECK:STDOUT:   %.loc25_3.7: init f64 = initialize_from %.loc19 to %.loc25_3.6
// CHECK:STDOUT:   %.loc25_3.8: i32 = int_literal 2
// CHECK:STDOUT:   %.loc25_3.9: ref f64 = array_index %floats.var, %.loc25_3.8
// CHECK:STDOUT:   %.loc25_3.10: init f64 = initialize_from %.loc20 to %.loc25_3.9
// CHECK:STDOUT:   %.loc25_3.11: i32 = int_literal 3
// CHECK:STDOUT:   %.loc25_3.12: ref f64 = array_index %floats.var, %.loc25_3.11
// CHECK:STDOUT:   %.loc25_3.13: init f64 = initialize_from %.loc21 to %.loc25_3.12
// CHECK:STDOUT:   %.loc25_3.14: i32 = int_literal 4
// CHECK:STDOUT:   %.loc25_3.15: ref f64 = array_index %floats.var, %.loc25_3.14
// CHECK:STDOUT:   %.loc25_3.16: init f64 = initialize_from %.loc22 to %.loc25_3.15
// CHECK:STDOUT:   %.loc25_3.17: i32 = int_literal 5
// CHECK:STDOUT:   %.loc25_3.18: ref f64 = array_index %floats.var, %.loc25_3.17
// CHECK:STDOUT:   %.loc25_3.19: init f64 = initialize_from %.loc23 to %.loc25_3.18
// CHECK:STDOUT:   %.loc25_3.20: i32 = int_literal 6
// CHECK:STDOUT:   %.loc25_3.21: ref f64 = array_index %floats.var, %.loc25_3.20
// CHECK:STDOUT:   %.loc25_3.22: init f64 = initialize_from %.loc24 to %.loc25_3.21
// CHECK:STDOUT:   %.loc25_3.23: init [f64; 7] = array_init (%.loc25_3.4, %.loc25_3.7, %.loc25_3.10, %.loc25_3.13, %.loc25_3.16, %.loc25_3.19, %.loc25_3.22) to %floats.var
// CHECK:STDOUT:   %.loc25_3.24: init [f64; 7] = converted %.loc25_3.1, %.loc25_3.23
// CHECK:STDOUT:   assign %floats.var, %.loc25_3.24
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
