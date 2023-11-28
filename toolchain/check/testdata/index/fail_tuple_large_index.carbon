// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var a: (i32,) = (12,);
var b: (i32,) = a;
// CHECK:STDERR: fail_tuple_large_index.carbon:[[@LINE+3]]:14: ERROR: Index `295147905179352825855` is past the end of `(i32,) as type`.
// CHECK:STDERR: var c: i32 = b[0xFFFFFFFFFFFFFFFFF];
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~~~
var c: i32 = b[0xFFFFFFFFFFFFFFFFF];

// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc7_13.1: type = tuple_type (type)
// CHECK:STDOUT:   %.loc7_13.2: type = tuple_type (i32)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "fail_tuple_large_index.carbon" {
// CHECK:STDOUT:   %.loc7_13.1: (type,) = tuple_literal (i32)
// CHECK:STDOUT:   %.loc7_13.2: type = converted %.loc7_13.1, constants.%.loc7_13.2
// CHECK:STDOUT:   %a.var: ref (i32,) = var a
// CHECK:STDOUT:   %a: ref (i32,) = bind_name a, %a.var
// CHECK:STDOUT:   %.loc7_18: i32 = int_literal 12
// CHECK:STDOUT:   %.loc7_21.1: (i32,) = tuple_literal (%.loc7_18)
// CHECK:STDOUT:   %.loc7_21.2: init (i32,) = tuple_init (%.loc7_18) to %a.var
// CHECK:STDOUT:   %.loc7_21.3: init (i32,) = converted %.loc7_21.1, %.loc7_21.2
// CHECK:STDOUT:   assign %a.var, %.loc7_21.3
// CHECK:STDOUT:   %.loc8_13: (type,) = tuple_literal (i32)
// CHECK:STDOUT:   %.loc7_13.3: type = converted %.loc8_13, constants.%.loc7_13.2
// CHECK:STDOUT:   %b.var: ref (i32,) = var b
// CHECK:STDOUT:   %b: ref (i32,) = bind_name b, %b.var
// CHECK:STDOUT:   %a.ref: ref (i32,) = name_ref a, %a
// CHECK:STDOUT:   %.loc8_17.1: ref i32 = tuple_access %a.ref, member0
// CHECK:STDOUT:   %.loc8_17.2: i32 = bind_value %.loc8_17.1
// CHECK:STDOUT:   %.loc8_17.3: init (i32,) = tuple_init (%.loc8_17.2) to %b.var
// CHECK:STDOUT:   %.loc8_17.4: init (i32,) = converted %a.ref, %.loc8_17.3
// CHECK:STDOUT:   assign %b.var, %.loc8_17.4
// CHECK:STDOUT:   %c.var: ref i32 = var c
// CHECK:STDOUT:   %c: ref i32 = bind_name c, %c.var
// CHECK:STDOUT:   %b.ref: ref (i32,) = name_ref b, %b
// CHECK:STDOUT:   %.loc12_16: i32 = int_literal 295147905179352825855
// CHECK:STDOUT:   %.loc12_35: ref <error> = tuple_index %b.ref, <error>
// CHECK:STDOUT:   assign %c.var, <error>
// CHECK:STDOUT: }
