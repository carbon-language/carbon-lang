// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var a: (i32,) = (12,);
var b: (i32,) = a;
// CHECK:STDERR: fail_tuple_large_index.carbon:[[@LINE+4]]:14: ERROR: Tuple element index `1` is past the end of type `(i32,)`.
// CHECK:STDERR: var c: i32 = b[1];
// CHECK:STDERR:              ^~~~
// CHECK:STDERR:
var c: i32 = b[1];
// CHECK:STDERR: fail_tuple_large_index.carbon:[[@LINE+3]]:14: ERROR: Tuple element index `2147483647` is past the end of type `(i32,)`.
// CHECK:STDERR: var d: i32 = b[0x7FFF_FFFF];
// CHECK:STDERR:              ^~~~~~~~~~~~~~
var d: i32 = b[0x7FFF_FFFF];

// CHECK:STDOUT: --- fail_tuple_large_index.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type (type) [template]
// CHECK:STDOUT:   %.2: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 12 [template]
// CHECK:STDOUT:   %.4: (i32,) = tuple_value (%.3) [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 2147483647 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7_13.1: (type,) = tuple_literal (i32)
// CHECK:STDOUT:   %.loc7_13.2: type = converted %.loc7_13.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:   %a.var: ref (i32,) = var a
// CHECK:STDOUT:   %a: ref (i32,) = bind_name a, %a.var
// CHECK:STDOUT:   %.loc8_13.1: (type,) = tuple_literal (i32)
// CHECK:STDOUT:   %.loc8_13.2: type = converted %.loc8_13.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:   %b.var: ref (i32,) = var b
// CHECK:STDOUT:   %b: ref (i32,) = bind_name b, %b.var
// CHECK:STDOUT:   %c.var: ref i32 = var c
// CHECK:STDOUT:   %c: ref i32 = bind_name c, %c.var
// CHECK:STDOUT:   %d.var: ref i32 = var d
// CHECK:STDOUT:   %d: ref i32 = bind_name d, %d.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_18: i32 = int_literal 12 [template = constants.%.3]
// CHECK:STDOUT:   %.loc7_21.1: (i32,) = tuple_literal (%.loc7_18)
// CHECK:STDOUT:   %.loc7_21.2: init (i32,) = tuple_init (%.loc7_18) to file.%a.var [template = constants.%.4]
// CHECK:STDOUT:   %.loc7_21.3: init (i32,) = converted %.loc7_21.1, %.loc7_21.2 [template = constants.%.4]
// CHECK:STDOUT:   assign file.%a.var, %.loc7_21.3
// CHECK:STDOUT:   %a.ref: ref (i32,) = name_ref a, file.%a
// CHECK:STDOUT:   %.loc8_17.1: ref i32 = tuple_access %a.ref, element0
// CHECK:STDOUT:   %.loc8_17.2: i32 = bind_value %.loc8_17.1
// CHECK:STDOUT:   %.loc8_17.3: init (i32,) = tuple_init (%.loc8_17.2) to file.%b.var
// CHECK:STDOUT:   %.loc8_17.4: init (i32,) = converted %a.ref, %.loc8_17.3
// CHECK:STDOUT:   assign file.%b.var, %.loc8_17.4
// CHECK:STDOUT:   %b.ref.loc13: ref (i32,) = name_ref b, file.%b
// CHECK:STDOUT:   %.loc13_16: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc13_17: ref <error> = tuple_index %b.ref.loc13, <error>
// CHECK:STDOUT:   assign file.%c.var, <error>
// CHECK:STDOUT:   %b.ref.loc17: ref (i32,) = name_ref b, file.%b
// CHECK:STDOUT:   %.loc17_16: i32 = int_literal 2147483647 [template = constants.%.6]
// CHECK:STDOUT:   %.loc17_27: ref <error> = tuple_index %b.ref.loc17, <error>
// CHECK:STDOUT:   assign file.%d.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
