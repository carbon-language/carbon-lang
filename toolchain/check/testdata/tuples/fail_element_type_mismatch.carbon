// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_element_type_mismatch.carbon:[[@LINE+3]]:31: Cannot implicitly convert from `(i32, f64) as type` to `(i32, i32) as type`.
// CHECK:STDERR: var x: (i32, i32) = (2, 65.89);
// CHECK:STDERR:                               ^
var x: (i32, i32) = (2, 65.89);

// CHECK:STDOUT: file "fail_element_type_mismatch.carbon" {
// CHECK:STDOUT:   %.loc10_9: type = stub_reference i32
// CHECK:STDOUT:   %.loc10_14: type = stub_reference i32
// CHECK:STDOUT:   %.loc10_17.1: type = tuple_type (type, type)
// CHECK:STDOUT:   %.loc10_17.2: (type, type) = tuple_literal (%.loc10_9, %.loc10_14)
// CHECK:STDOUT:   %.loc10_17.3: type = tuple_type (i32, i32)
// CHECK:STDOUT:   %x: ref (i32, i32) = var "x"
// CHECK:STDOUT:   %.loc10_22.1: i32 = int_literal 2
// CHECK:STDOUT:   %.loc10_22.2: i32 = stub_reference %.loc10_22.1
// CHECK:STDOUT:   %.loc10_25.1: f64 = real_literal 6589e-2
// CHECK:STDOUT:   %.loc10_25.2: f64 = stub_reference %.loc10_25.1
// CHECK:STDOUT:   %.loc10_30.1: type = tuple_type (i32, f64)
// CHECK:STDOUT:   %.loc10_30.2: (i32, f64) = tuple_literal (%.loc10_22.2, %.loc10_25.2)
// CHECK:STDOUT:   assign %x, <error>
// CHECK:STDOUT: }
