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

// CHECK:STDOUT: --- fail_tuple_index_error.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.2: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.3: type = ptr_type (i32, i32) [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 12 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 6 [template]
// CHECK:STDOUT:   %.6: (i32, i32) = tuple_value (%.4, %.5) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.a = %a, .b = %b} [template]
// CHECK:STDOUT:   %.loc7_17.1: (type, type) = tuple_literal (i32, i32)
// CHECK:STDOUT:   %.loc7_17.2: type = converted %.loc7_17.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:   %a.var: ref (i32, i32) = var a
// CHECK:STDOUT:   %a: ref (i32, i32) = bind_name a, %a.var
// CHECK:STDOUT:   %.loc7_22: i32 = int_literal 12 [template = constants.%.4]
// CHECK:STDOUT:   %.loc7_26: i32 = int_literal 6 [template = constants.%.5]
// CHECK:STDOUT:   %.loc7_27.1: (i32, i32) = tuple_literal (%.loc7_22, %.loc7_26)
// CHECK:STDOUT:   %.loc7_27.2: ref i32 = tuple_access %a.var, element0
// CHECK:STDOUT:   %.loc7_27.3: init i32 = initialize_from %.loc7_22 to %.loc7_27.2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc7_27.4: ref i32 = tuple_access %a.var, element1
// CHECK:STDOUT:   %.loc7_27.5: init i32 = initialize_from %.loc7_26 to %.loc7_27.4 [template = constants.%.5]
// CHECK:STDOUT:   %.loc7_27.6: init (i32, i32) = tuple_init (%.loc7_27.3, %.loc7_27.5) to %a.var [template = constants.%.6]
// CHECK:STDOUT:   %.loc7_27.7: init (i32, i32) = converted %.loc7_27.1, %.loc7_27.6 [template = constants.%.6]
// CHECK:STDOUT:   assign %a.var, %.loc7_27.7
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %a.ref: ref (i32, i32) = name_ref a, %a
// CHECK:STDOUT:   %oops.ref: <error> = name_ref oops, <error> [template = <error>]
// CHECK:STDOUT:   %.loc11: ref <error> = tuple_index %a.ref, <error>
// CHECK:STDOUT:   assign %b.var, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
