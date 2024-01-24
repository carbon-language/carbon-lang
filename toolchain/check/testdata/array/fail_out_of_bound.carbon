// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_out_of_bound.carbon:[[@LINE+3]]:19: ERROR: Cannot initialize array of 1 element(s) from 3 initializer(s).
// CHECK:STDERR: var a: [i32; 1] = (1, 2, 3);
// CHECK:STDERR:                   ^~~~~~~~~
var a: [i32; 1] = (1, 2, 3);

// CHECK:STDOUT: --- fail_out_of_bound.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.2: type = array_type %.1, i32 [template]
// CHECK:STDOUT:   %.3: type = ptr_type [i32; 1] [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.6: type = tuple_type (i32, i32, i32) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.a = %a} [template]
// CHECK:STDOUT:   %.loc10_14: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc10_15: type = array_type %.loc10_14, i32 [template = constants.%.2]
// CHECK:STDOUT:   %a.var: ref [i32; 1] = var a
// CHECK:STDOUT:   %a: ref [i32; 1] = bind_name a, %a.var
// CHECK:STDOUT:   %.loc10_20: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc10_23: i32 = int_literal 2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc10_26: i32 = int_literal 3 [template = constants.%.5]
// CHECK:STDOUT:   %.loc10_27: (i32, i32, i32) = tuple_literal (%.loc10_20, %.loc10_23, %.loc10_26)
// CHECK:STDOUT:   assign %a.var, <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
