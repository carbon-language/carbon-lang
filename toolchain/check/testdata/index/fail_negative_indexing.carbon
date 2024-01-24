// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var a: (i32, i32) = (12, 6);
// CHECK:STDERR: fail_negative_indexing.carbon:[[@LINE+3]]:16: ERROR: Semantics TODO: `HandlePrefixOperatorMinus`.
// CHECK:STDERR: var b: i32 = a[-10];
// CHECK:STDERR:                ^~~
var b: i32 = a[-10];

// CHECK:STDOUT: --- fail_negative_indexing.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.2: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.3: type = ptr_type (i32, i32) [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 12 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 6 [template]
// CHECK:STDOUT:   %.6: (i32, i32) = tuple_value (%.4, %.5) [template]
// CHECK:STDOUT:   %.7: i32 = int_literal 10 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT: }
// CHECK:STDOUT:
