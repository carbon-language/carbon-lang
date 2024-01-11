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
// CHECK:STDOUT:   %.loc7_17.1: type = tuple_type (type, type), const
// CHECK:STDOUT:   %.loc7_17.2: type = tuple_type (i32, i32), const
// CHECK:STDOUT:   %.loc7_17.3: type = ptr_type (i32, i32), const
// CHECK:STDOUT:   %.loc7_22: i32 = int_literal 12, const
// CHECK:STDOUT:   %.loc7_26: i32 = int_literal 6, const
// CHECK:STDOUT:   %.loc11: i32 = int_literal 10, const
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT: }
// CHECK:STDOUT:
