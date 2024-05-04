// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var a: (i32, i32) = (12, 6);
// TODO: Only diagnose this once, and make the diagnostic clearer.
// CHECK:STDERR: fail_negative_indexing.carbon:[[@LINE+7]]:16: ERROR: Name `Negate` not found.
// CHECK:STDERR: var b: i32 = a[-10];
// CHECK:STDERR:                ^~~
// CHECK:STDERR:
// CHECK:STDERR: fail_negative_indexing.carbon:[[@LINE+3]]:16: ERROR: Semantics TODO: `missing or invalid operator interface, also avoid duplicate diagnostic if prelude is unavailable`.
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
// CHECK:STDOUT:   %tuple: (i32, i32) = tuple_value (%.4, %.5) [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 10 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc7_17.1: (type, type) = tuple_literal (i32, i32)
// CHECK:STDOUT:   %.loc7_17.2: type = converted %.loc7_17.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:   %a.var: ref (i32, i32) = var a
// CHECK:STDOUT:   %a: ref (i32, i32) = bind_name a, %a.var
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b: ref i32 = bind_name b, %b.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_22: i32 = int_literal 12 [template = constants.%.4]
// CHECK:STDOUT:   %.loc7_26: i32 = int_literal 6 [template = constants.%.5]
// CHECK:STDOUT:   %.loc7_27.1: (i32, i32) = tuple_literal (%.loc7_22, %.loc7_26)
// CHECK:STDOUT:   %.loc7_27.2: ref i32 = tuple_access file.%a.var, element0
// CHECK:STDOUT:   %.loc7_27.3: init i32 = initialize_from %.loc7_22 to %.loc7_27.2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc7_27.4: ref i32 = tuple_access file.%a.var, element1
// CHECK:STDOUT:   %.loc7_27.5: init i32 = initialize_from %.loc7_26 to %.loc7_27.4 [template = constants.%.5]
// CHECK:STDOUT:   %.loc7_27.6: init (i32, i32) = tuple_init (%.loc7_27.3, %.loc7_27.5) to file.%a.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc7_28: init (i32, i32) = converted %.loc7_27.1, %.loc7_27.6 [template = constants.%tuple]
// CHECK:STDOUT:   assign file.%a.var, %.loc7_28
// CHECK:STDOUT:   %a.ref: ref (i32, i32) = name_ref a, file.%a
// CHECK:STDOUT:   %.loc16_17: i32 = int_literal 10 [template = constants.%.6]
// CHECK:STDOUT:   %.loc16_19: ref <error> = tuple_index %a.ref, <error>
// CHECK:STDOUT:   assign file.%b.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
