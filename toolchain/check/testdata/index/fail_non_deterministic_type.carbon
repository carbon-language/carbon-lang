// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var a: (i32, i32) = (2, 3);
var b: i32 = 0;
// CHECK:STDERR: fail_non_deterministic_type.carbon:[[@LINE+3]]:17: Tuples indices must be integer literals.
// CHECK:STDERR: var c: i32 = a[b];
// CHECK:STDERR:                 ^
var c: i32 = a[b];

// CHECK:STDOUT: file "fail_non_deterministic_type.carbon" {
// CHECK:STDOUT:   %.loc7_9: type = stub_reference i32
// CHECK:STDOUT:   %.loc7_14: type = stub_reference i32
// CHECK:STDOUT:   %.loc7_17.1: type = tuple_type (type, type)
// CHECK:STDOUT:   %.loc7_17.2: (type, type) = tuple_literal (%.loc7_9, %.loc7_14)
// CHECK:STDOUT:   %.loc7_17.3: type = tuple_type (i32, i32)
// CHECK:STDOUT:   %a: ref (i32, i32) = var "a"
// CHECK:STDOUT:   %.loc7_22.1: i32 = int_literal 2
// CHECK:STDOUT:   %.loc7_22.2: i32 = stub_reference %.loc7_22.1
// CHECK:STDOUT:   %.loc7_25.1: i32 = int_literal 3
// CHECK:STDOUT:   %.loc7_25.2: i32 = stub_reference %.loc7_25.1
// CHECK:STDOUT:   %.loc7_26: (i32, i32) = tuple_literal (%.loc7_22.2, %.loc7_25.2)
// CHECK:STDOUT:   assign %a, %.loc7_26
// CHECK:STDOUT:   %b: ref i32 = var "b"
// CHECK:STDOUT:   %.loc8: i32 = int_literal 0
// CHECK:STDOUT:   assign %b, %.loc8
// CHECK:STDOUT:   %c: ref i32 = var "c"
// CHECK:STDOUT:   %.loc12: ref <error> = tuple_index %a, <error>
// CHECK:STDOUT:   assign %c, <error>
// CHECK:STDOUT: }
