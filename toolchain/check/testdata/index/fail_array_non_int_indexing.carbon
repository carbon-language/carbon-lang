// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var a: [i32; 1] = (12,);
// CHECK:STDERR: fail_array_non_int_indexing.carbon:[[@LINE+3]]:16: ERROR: Cannot implicitly convert from `f64` to `i32`.
// CHECK:STDERR: var b: i32 = a[2.6];
// CHECK:STDERR:                ^~~
var b: i32 = a[2.6];

// CHECK:STDOUT: --- fail_array_non_int_indexing.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.2: type = array_type %.1, i32 [template]
// CHECK:STDOUT:   %.3: type = ptr_type [i32; 1] [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 12 [template]
// CHECK:STDOUT:   %.5: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.7: [i32; 1] = tuple_value (%.4) [template]
// CHECK:STDOUT:   %.8: f64 = real_literal 26e-1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.a = %a, .b = %b} [template]
// CHECK:STDOUT:   %.loc7_14: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc7_15: type = array_type %.loc7_14, i32 [template = constants.%.2]
// CHECK:STDOUT:   %a.var: ref [i32; 1] = var a
// CHECK:STDOUT:   %a: ref [i32; 1] = bind_name a, %a.var
// CHECK:STDOUT:   %.loc7_20: i32 = int_literal 12 [template = constants.%.4]
// CHECK:STDOUT:   %.loc7_23.1: (i32,) = tuple_literal (%.loc7_20)
// CHECK:STDOUT:   %.loc7_23.2: i32 = int_literal 0 [template = constants.%.6]
// CHECK:STDOUT:   %.loc7_23.3: ref i32 = array_index %a.var, %.loc7_23.2
// CHECK:STDOUT:   %.loc7_23.4: init i32 = initialize_from %.loc7_20 to %.loc7_23.3 [template = constants.%.4]
// CHECK:STDOUT:   %.loc7_23.5: init [i32; 1] = array_init (%.loc7_23.4) to %a.var [template = constants.%.7]
// CHECK:STDOUT:   %.loc7_23.6: init [i32; 1] = converted %.loc7_23.1, %.loc7_23.5 [template = constants.%.7]
// CHECK:STDOUT:   assign %a.var, %.loc7_23.6
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %a.ref: ref [i32; 1] = name_ref a, %a
// CHECK:STDOUT:   %.loc11_16: f64 = real_literal 26e-1 [template = constants.%.8]
// CHECK:STDOUT:   %.loc11_19.1: ref i32 = array_index %a.ref, <error>
// CHECK:STDOUT:   %.loc11_19.2: i32 = bind_value %.loc11_19.1
// CHECK:STDOUT:   assign %b.var, %.loc11_19.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
