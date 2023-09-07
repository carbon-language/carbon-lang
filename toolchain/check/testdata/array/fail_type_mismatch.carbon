// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+3]]:26: Cannot implicitly convert from `f64` to `i32`.
// CHECK:STDERR: var a: [i32; 2] = (1, 2.5);
// CHECK:STDERR:                          ^
var a: [i32; 2] = (1, 2.5);

// CHECK:STDOUT: file "fail_type_mismatch.carbon" {
// CHECK:STDOUT:   %.loc10_14: i32 = int_literal 2
// CHECK:STDOUT:   %.loc10_15: type = array_type %.loc10_14, i32
// CHECK:STDOUT:   %a: ref [i32; 2] = var "a"
// CHECK:STDOUT:   %.loc10_20.1: i32 = int_literal 1
// CHECK:STDOUT:   %.loc10_20.2: i32 = stub_reference %.loc10_20.1
// CHECK:STDOUT:   %.loc10_23.1: f64 = real_literal 25e-1
// CHECK:STDOUT:   %.loc10_23.2: f64 = stub_reference %.loc10_23.1
// CHECK:STDOUT:   %.loc10_26.1: type = tuple_type (i32, f64)
// CHECK:STDOUT:   %.loc10_26.2: (i32, f64) = tuple_literal (%.loc10_20.2, %.loc10_23.2)
// CHECK:STDOUT:   %.loc10_26.3: ref [i32; 2] = stub_reference %a
// CHECK:STDOUT:   %.loc10_26.4: i32 = int_literal 0
// CHECK:STDOUT:   %.loc10_26.5: ref i32 = array_index %.loc10_26.3, %.loc10_26.4
// CHECK:STDOUT:   %.loc10_26.6 = initialize_from %.loc10_26.5 to %.loc10_20.2
// CHECK:STDOUT:   %.loc10_26.7: i32 = int_literal 1
// CHECK:STDOUT:   %.loc10_26.8: ref i32 = array_index %.loc10_26.3, %.loc10_26.7
// CHECK:STDOUT:   %.loc10_26.9 = initialize_from %.loc10_26.8 to <error>
// CHECK:STDOUT:   %.loc10_26.10: init [i32; 2] = array_init %.loc10_26.2, (%.loc10_26.6, %.loc10_26.9) to %.loc10_26.3
// CHECK:STDOUT:   assign %a, %.loc10_26.10
// CHECK:STDOUT: }
