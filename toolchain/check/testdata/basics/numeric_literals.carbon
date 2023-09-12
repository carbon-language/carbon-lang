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

// CHECK:STDOUT: package {
// CHECK:STDOUT:   %.loc7 = fn_decl @F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_19: i32 = int_literal 5
// CHECK:STDOUT:   %.loc10_20: type = array_type %.loc10_19, i32
// CHECK:STDOUT:   %ints: ref [i32; 5] = var "ints"
// CHECK:STDOUT:   %.loc11_5.1: i32 = int_literal 8
// CHECK:STDOUT:   %.loc11_5.2: i32 = stub_reference %.loc11_5.1
// CHECK:STDOUT:   %.loc12_5.1: i32 = int_literal 9
// CHECK:STDOUT:   %.loc12_5.2: i32 = stub_reference %.loc12_5.1
// CHECK:STDOUT:   %.loc13_5.1: i32 = int_literal 8
// CHECK:STDOUT:   %.loc13_5.2: i32 = stub_reference %.loc13_5.1
// CHECK:STDOUT:   %.loc14_5.1: i32 = int_literal 8
// CHECK:STDOUT:   %.loc14_5.2: i32 = stub_reference %.loc14_5.1
// CHECK:STDOUT:   %.loc15_5.1: i32 = int_literal 39999999999999999993
// CHECK:STDOUT:   %.loc15_5.2: i32 = stub_reference %.loc15_5.1
// CHECK:STDOUT:   %.loc16_3.1: type = tuple_type (i32, i32, i32, i32, i32)
// CHECK:STDOUT:   %.loc16_3.2: (i32, i32, i32, i32, i32) = tuple_literal (%.loc11_5.2, %.loc12_5.2, %.loc13_5.2, %.loc14_5.2, %.loc15_5.2)
// CHECK:STDOUT:   %.loc16_3.3: [i32; 5] = array_value %.loc16_3.2
// CHECK:STDOUT:   assign %ints, %.loc16_3.3
// CHECK:STDOUT:   %.loc17_21: i32 = int_literal 7
// CHECK:STDOUT:   %.loc17_22: type = array_type %.loc17_21, f64
// CHECK:STDOUT:   %floats: ref [f64; 7] = var "floats"
// CHECK:STDOUT:   %.loc18_5.1: f64 = real_literal 9e-1
// CHECK:STDOUT:   %.loc18_5.2: f64 = stub_reference %.loc18_5.1
// CHECK:STDOUT:   %.loc19_5.1: f64 = real_literal 80e-1
// CHECK:STDOUT:   %.loc19_5.2: f64 = stub_reference %.loc19_5.1
// CHECK:STDOUT:   %.loc20_5.1: f64 = real_literal 800e-1
// CHECK:STDOUT:   %.loc20_5.2: f64 = stub_reference %.loc20_5.1
// CHECK:STDOUT:   %.loc21_5.1: f64 = real_literal 10e6
// CHECK:STDOUT:   %.loc21_5.2: f64 = stub_reference %.loc21_5.1
// CHECK:STDOUT:   %.loc22_5.1: f64 = real_literal 10e7
// CHECK:STDOUT:   %.loc22_5.2: f64 = stub_reference %.loc22_5.1
// CHECK:STDOUT:   %.loc23_5.1: f64 = real_literal 10e-9
// CHECK:STDOUT:   %.loc23_5.2: f64 = stub_reference %.loc23_5.1
// CHECK:STDOUT:   %.loc24_5.1: f64 = real_literal 399999999999999999930e39999999999999999992
// CHECK:STDOUT:   %.loc24_5.2: f64 = stub_reference %.loc24_5.1
// CHECK:STDOUT:   %.loc25_3.1: type = tuple_type (f64, f64, f64, f64, f64, f64, f64)
// CHECK:STDOUT:   %.loc25_3.2: (f64, f64, f64, f64, f64, f64, f64) = tuple_literal (%.loc18_5.2, %.loc19_5.2, %.loc20_5.2, %.loc21_5.2, %.loc22_5.2, %.loc23_5.2, %.loc24_5.2)
// CHECK:STDOUT:   %.loc25_3.3: [f64; 7] = array_value %.loc25_3.2
// CHECK:STDOUT:   assign %floats, %.loc25_3.3
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
