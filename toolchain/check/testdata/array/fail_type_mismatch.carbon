// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "fail_type_mismatch.carbon" {

// CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+3]]:27: Cannot implicitly convert from `(i32, f64) as type` to `[i32; 2]`.
// CHECK:STDERR: var a: [i32; 2] = (1, 2.5);
// CHECK:STDERR:                           ^
var a: [i32; 2] = (1, 2.5);
// CHECK:STDOUT:   %.loc[[@LINE-1]]_14: i32 = int_literal 2
// CHECK:STDOUT:   %.loc[[@LINE-2]]_15: type = array_type %.loc[[@LINE-2]]_14, i32
// CHECK:STDOUT:   %a: ref [i32; 2] = var "a"
// CHECK:STDOUT:   %.loc[[@LINE-4]]_20.1: i32 = int_literal 1
// CHECK:STDOUT:   %.loc[[@LINE-5]]_20.2: i32 = stub_reference %.loc[[@LINE-5]]_20.1
// CHECK:STDOUT:   %.loc[[@LINE-6]]_23.1: f64 = real_literal 25e-1
// CHECK:STDOUT:   %.loc[[@LINE-7]]_23.2: f64 = stub_reference %.loc[[@LINE-7]]_23.1
// CHECK:STDOUT:   %.loc[[@LINE-8]]_26.1: type = tuple_type (i32, f64)
// CHECK:STDOUT:   %.loc[[@LINE-9]]_26.2: (i32, f64) = tuple_literal (%.loc[[@LINE-9]]_20.2, %.loc[[@LINE-9]]_23.2)
// CHECK:STDOUT:   assign %a, <error>
// CHECK:STDOUT: }
