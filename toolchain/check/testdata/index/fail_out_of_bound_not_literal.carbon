// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var a: (i32, i32) = (12, 34);
// CHECK:STDERR: fail_out_of_bound_not_literal.carbon:[[@LINE+3]]:14: ERROR: Tuple element index `2` is past the end of type `(i32, i32)`.
// CHECK:STDERR: var b: i32 = a[{.index = 2}.index];
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~~
var b: i32 = a[{.index = 2}.index];

// CHECK:STDOUT: --- fail_out_of_bound_not_literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.2: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.3: type = ptr_type (i32, i32) [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 12 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 34 [template]
// CHECK:STDOUT:   %.6: (i32, i32) = tuple_value (%.4, %.5) [template]
// CHECK:STDOUT:   %.7: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.8: type = struct_type {.index: i32} [template]
// CHECK:STDOUT:   %.9: {.index: i32} = struct_value (%.7) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.a = %a, .b = %b} [template]
// CHECK:STDOUT:   %.loc7_17.1: (type, type) = tuple_literal (i32, i32)
// CHECK:STDOUT:   %.loc7_17.2: type = converted %.loc7_17.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:   %a.var: ref (i32, i32) = var a
// CHECK:STDOUT:   %a: ref (i32, i32) = bind_name a, %a.var
// CHECK:STDOUT:   %.loc7_22: i32 = int_literal 12 [template = constants.%.4]
// CHECK:STDOUT:   %.loc7_26: i32 = int_literal 34 [template = constants.%.5]
// CHECK:STDOUT:   %.loc7_28.1: (i32, i32) = tuple_literal (%.loc7_22, %.loc7_26)
// CHECK:STDOUT:   %.loc7_28.2: ref i32 = tuple_access %a.var, element0
// CHECK:STDOUT:   %.loc7_28.3: init i32 = initialize_from %.loc7_22 to %.loc7_28.2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc7_28.4: ref i32 = tuple_access %a.var, element1
// CHECK:STDOUT:   %.loc7_28.5: init i32 = initialize_from %.loc7_26 to %.loc7_28.4 [template = constants.%.5]
// CHECK:STDOUT:   %.loc7_28.6: init (i32, i32) = tuple_init (%.loc7_28.3, %.loc7_28.5) to %a.var [template = constants.%.6]
// CHECK:STDOUT:   %.loc7_28.7: init (i32, i32) = converted %.loc7_28.1, %.loc7_28.6 [template = constants.%.6]
// CHECK:STDOUT:   assign %a.var, %.loc7_28.7
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %a.ref: ref (i32, i32) = name_ref a, %a
// CHECK:STDOUT:   %.loc11_26: i32 = int_literal 2 [template = constants.%.7]
// CHECK:STDOUT:   %.loc11_27.1: {.index: i32} = struct_literal (%.loc11_26)
// CHECK:STDOUT:   %.loc11_27.2: {.index: i32} = struct_value (%.loc11_26) [template = constants.%.9]
// CHECK:STDOUT:   %.loc11_27.3: {.index: i32} = converted %.loc11_27.1, %.loc11_27.2 [template = constants.%.9]
// CHECK:STDOUT:   %.loc11_28: i32 = struct_access %.loc11_27.3, element0 [template = constants.%.7]
// CHECK:STDOUT:   %.loc11_34: ref <error> = tuple_index %a.ref, <error>
// CHECK:STDOUT:   assign %b.var, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
