// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "numeric_literals.carbon" {

fn F() {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
  // 8 and 9 trigger special behavior in APInt when mishandling signed versus
  // unsigned, so we pay extra attention to those.
  var ints: [i32; 5] = (
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_19: i32 = int_literal 5
  // CHECK:STDOUT:   %.loc[[@LINE-2]]_20: type = array_type %.loc[[@LINE-2]]_19, i32
  // CHECK:STDOUT:   %ints: ref [i32; 5] = var "ints"
    8,
    // CHECK:STDOUT:   %.loc[[@LINE-1]]_5.1: i32 = int_literal 8
    // CHECK:STDOUT:   %.loc[[@LINE-2]]_5.2: i32 = stub_reference %.loc[[@LINE-2]]_5.1
    9,
    // CHECK:STDOUT:   %.loc[[@LINE-1]]_5.1: i32 = int_literal 9
    // CHECK:STDOUT:   %.loc[[@LINE-2]]_5.2: i32 = stub_reference %.loc[[@LINE-2]]_5.1
    0x8,
    // CHECK:STDOUT:   %.loc[[@LINE-1]]_5.1: i32 = int_literal 8
    // CHECK:STDOUT:   %.loc[[@LINE-2]]_5.2: i32 = stub_reference %.loc[[@LINE-2]]_5.1
    0b1000,
    // CHECK:STDOUT:   %.loc[[@LINE-1]]_5.1: i32 = int_literal 8
    // CHECK:STDOUT:   %.loc[[@LINE-2]]_5.2: i32 = stub_reference %.loc[[@LINE-2]]_5.1
    39999999999999999993,
    // CHECK:STDOUT:   %.loc[[@LINE-1]]_5.1: i32 = int_literal 39999999999999999993
    // CHECK:STDOUT:   %.loc[[@LINE-2]]_5.2: i32 = stub_reference %.loc[[@LINE-2]]_5.1
  );
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_3.1: type = tuple_type (i32, i32, i32, i32, i32)
  // CHECK:STDOUT:   %.loc[[@LINE-2]]_3.2: (i32, i32, i32, i32, i32) = tuple_literal (%.loc[[@LINE-17]]_5.2, %.loc[[@LINE-14]]_5.2, %.loc[[@LINE-11]]_5.2, %.loc[[@LINE-8]]_5.2, %.loc[[@LINE-5]]_5.2)
  // CHECK:STDOUT:   %.loc[[@LINE-3]]_3.3: [i32; 5] = array_value %.loc[[@LINE-3]]_3.2
  // CHECK:STDOUT:   assign %ints, %.loc[[@LINE-4]]_3.3
  var floats: [f64; 7] = (
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_21: i32 = int_literal 7
  // CHECK:STDOUT:   %.loc[[@LINE-2]]_22: type = array_type %.loc[[@LINE-2]]_21, f64
  // CHECK:STDOUT:   %floats: ref [f64; 7] = var "floats"
    0.9,
    // CHECK:STDOUT:   %.loc[[@LINE-1]]_5.1: f64 = real_literal 9e-1
    // CHECK:STDOUT:   %.loc[[@LINE-2]]_5.2: f64 = stub_reference %.loc[[@LINE-2]]_5.1
    8.0,
    // CHECK:STDOUT:   %.loc[[@LINE-1]]_5.1: f64 = real_literal 80e-1
    // CHECK:STDOUT:   %.loc[[@LINE-2]]_5.2: f64 = stub_reference %.loc[[@LINE-2]]_5.1
    80.0,
    // CHECK:STDOUT:   %.loc[[@LINE-1]]_5.1: f64 = real_literal 800e-1
    // CHECK:STDOUT:   %.loc[[@LINE-2]]_5.2: f64 = stub_reference %.loc[[@LINE-2]]_5.1
    1.0e7,
    // CHECK:STDOUT:   %.loc[[@LINE-1]]_5.1: f64 = real_literal 10e6
    // CHECK:STDOUT:   %.loc[[@LINE-2]]_5.2: f64 = stub_reference %.loc[[@LINE-2]]_5.1
    1.0e8,
    // CHECK:STDOUT:   %.loc[[@LINE-1]]_5.1: f64 = real_literal 10e7
    // CHECK:STDOUT:   %.loc[[@LINE-2]]_5.2: f64 = stub_reference %.loc[[@LINE-2]]_5.1
    1.0e-8,
    // CHECK:STDOUT:   %.loc[[@LINE-1]]_5.1: f64 = real_literal 10e-9
    // CHECK:STDOUT:   %.loc[[@LINE-2]]_5.2: f64 = stub_reference %.loc[[@LINE-2]]_5.1
    39999999999999999993.0e39999999999999999993,
    // CHECK:STDOUT:   %.loc[[@LINE-1]]_5.1: f64 = real_literal 399999999999999999930e39999999999999999992
    // CHECK:STDOUT:   %.loc[[@LINE-2]]_5.2: f64 = stub_reference %.loc[[@LINE-2]]_5.1
  );
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_3.1: type = tuple_type (f64, f64, f64, f64, f64, f64, f64)
  // CHECK:STDOUT:   %.loc[[@LINE-2]]_3.2: (f64, f64, f64, f64, f64, f64, f64) = tuple_literal (%.loc[[@LINE-23]]_5.2, %.loc[[@LINE-20]]_5.2, %.loc[[@LINE-17]]_5.2, %.loc[[@LINE-14]]_5.2, %.loc[[@LINE-11]]_5.2, %.loc[[@LINE-8]]_5.2, %.loc[[@LINE-5]]_5.2)
  // CHECK:STDOUT:   %.loc[[@LINE-3]]_3.3: [f64; 7] = array_value %.loc[[@LINE-3]]_3.2
  // CHECK:STDOUT:   assign %floats, %.loc[[@LINE-4]]_3.3
  // CHECK:STDOUT:   return
  // CHECK:STDOUT: }
}
