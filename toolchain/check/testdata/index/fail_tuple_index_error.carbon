// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var a: (i32, i32) = (12, 6);
// CHECK:STDERR: fail_tuple_index_error.carbon:[[@LINE+3]]:16: ERROR: Name `oops` not found.
// CHECK:STDERR: var b: i32 = a[oops];
// CHECK:STDERR:                ^~~~
var b: i32 = a[oops];

// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc7_17.1: type = tuple_type (type, type)
// CHECK:STDOUT:   %.loc7_17.2: type = tuple_type (i32, i32)
// CHECK:STDOUT:   %.loc7_17.3: type = ptr_type (i32, i32)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "fail_tuple_index_error.carbon" {
// CHECK:STDOUT:   %.loc7_17.1: (type, type) = tuple_literal (i32, i32)
// CHECK:STDOUT:   %.loc7_17.2: type = converted %.loc7_17.1, constants.%.loc7_17.2
// CHECK:STDOUT:   %a.var: ref (i32, i32) = var a
// CHECK:STDOUT:   %a: ref (i32, i32) = bind_name a, %a.var
// CHECK:STDOUT:   %.loc7_22: i32 = int_literal 12
// CHECK:STDOUT:   %.loc7_26: i32 = int_literal 6
// CHECK:STDOUT:   %.loc7_27.1: (i32, i32) = tuple_literal (%.loc7_22, %.loc7_26)
// CHECK:STDOUT:   %.loc7_27.2: ref i32 = tuple_access %a.var, member0
// CHECK:STDOUT:   %.loc7_27.3: init i32 = initialize_from %.loc7_22 to %.loc7_27.2
// CHECK:STDOUT:   %.loc7_27.4: ref i32 = tuple_access %a.var, member1
// CHECK:STDOUT:   %.loc7_27.5: init i32 = initialize_from %.loc7_26 to %.loc7_27.4
// CHECK:STDOUT:   %.loc7_27.6: init (i32, i32) = tuple_init (%.loc7_27.3, %.loc7_27.5) to %a.var
// CHECK:STDOUT:   %.loc7_27.7: init (i32, i32) = converted %.loc7_27.1, %.loc7_27.6
// CHECK:STDOUT:   assign %a.var, %.loc7_27.7
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %a.ref: ref (i32, i32) = name_ref a, %a
// CHECK:STDOUT:   %oops.ref: <error> = name_ref oops, <error>
// CHECK:STDOUT:   %.loc11: ref <error> = tuple_index %a.ref, %oops.ref
// CHECK:STDOUT:   assign %b.var, <error>
// CHECK:STDOUT: }
