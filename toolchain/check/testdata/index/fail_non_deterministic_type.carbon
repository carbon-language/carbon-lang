// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var a: (i32, i32) = (2, 3);
var b: i32 = 0;
// CHECK:STDERR: fail_non_deterministic_type.carbon:[[@LINE+3]]:17: ERROR: Tuples indices must be integer literals.
// CHECK:STDERR: var c: i32 = a[b];
// CHECK:STDERR:                 ^
var c: i32 = a[b];

// CHECK:STDOUT: file "fail_non_deterministic_type.carbon" {
// CHECK:STDOUT:   %.loc7_17.1: type = tuple_type (type, type)
// CHECK:STDOUT:   %.loc7_17.2: (type, type) = tuple_literal (i32, i32)
// CHECK:STDOUT:   %.loc7_17.3: type = tuple_type (i32, i32)
// CHECK:STDOUT:   %.loc7_17.4: type = ptr_type (i32, i32)
// CHECK:STDOUT:   %a: ref (i32, i32) = var "a"
// CHECK:STDOUT:   %.loc7_22: i32 = int_literal 2
// CHECK:STDOUT:   %.loc7_25: i32 = int_literal 3
// CHECK:STDOUT:   %.loc7_26.1: (i32, i32) = tuple_literal (%.loc7_22, %.loc7_25)
// CHECK:STDOUT:   %.loc7_26.2: ref i32 = tuple_access %a, member0
// CHECK:STDOUT:   %.loc7_26.3: init i32 = initialize_from %.loc7_22 to %.loc7_26.2
// CHECK:STDOUT:   %.loc7_26.4: ref i32 = tuple_access %a, member1
// CHECK:STDOUT:   %.loc7_26.5: init i32 = initialize_from %.loc7_25 to %.loc7_26.4
// CHECK:STDOUT:   %.loc7_26.6: init (i32, i32) = tuple_init %.loc7_26.1, (%.loc7_26.3, %.loc7_26.5)
// CHECK:STDOUT:   assign %a, %.loc7_26.6
// CHECK:STDOUT:   %b: ref i32 = var "b"
// CHECK:STDOUT:   %.loc8: i32 = int_literal 0
// CHECK:STDOUT:   assign %b, %.loc8
// CHECK:STDOUT:   %c: ref i32 = var "c"
// CHECK:STDOUT:   %a.ref: ref (i32, i32) = name_reference "a", %a
// CHECK:STDOUT:   %b.ref: ref i32 = name_reference "b", %b
// CHECK:STDOUT:   %.loc12: ref <error> = tuple_index %a.ref, <error>
// CHECK:STDOUT:   assign %c, <error>
// CHECK:STDOUT: }
