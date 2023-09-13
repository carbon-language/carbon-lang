// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var a: (i32,) = (12,);
var b: (i32,) = a;
// CHECK:STDERR: fail_tuple_large_index.carbon:[[@LINE+3]]:35: Index `295147905179352825855` is past the end of `(i32,) as type`.
// CHECK:STDERR: var c: i32 = b[0xFFFFFFFFFFFFFFFFF];
// CHECK:STDERR:                                   ^
var c: i32 = b[0xFFFFFFFFFFFFFFFFF];

// CHECK:STDOUT: file "fail_tuple_large_index.carbon" {
// CHECK:STDOUT:   %.loc7_9: type = stub_reference i32
// CHECK:STDOUT:   %.loc7_13.1: type = tuple_type (type)
// CHECK:STDOUT:   %.loc7_13.2: (type,) = tuple_literal (%.loc7_9)
// CHECK:STDOUT:   %.loc7_13.3: type = tuple_type (i32)
// CHECK:STDOUT:   %a: ref (i32,) = var "a"
// CHECK:STDOUT:   %.loc7_18.1: i32 = int_literal 12
// CHECK:STDOUT:   %.loc7_18.2: i32 = stub_reference %.loc7_18.1
// CHECK:STDOUT:   %.loc7_21: (i32,) = tuple_literal (%.loc7_18.2)
// CHECK:STDOUT:   assign %a, %.loc7_21
// CHECK:STDOUT:   %.loc8_9: type = stub_reference i32
// CHECK:STDOUT:   %.loc8_13: (type,) = tuple_literal (%.loc8_9)
// CHECK:STDOUT:   %b: ref (i32,) = var "b"
// CHECK:STDOUT:   %.loc7_5: (i32,) = bind_value %a
// CHECK:STDOUT:   assign %b, %.loc7_5
// CHECK:STDOUT:   %c: ref i32 = var "c"
// CHECK:STDOUT:   %.loc12_16: i32 = int_literal 295147905179352825855
// CHECK:STDOUT:   %.loc12_35: ref <error> = tuple_index %b, <error>
// CHECK:STDOUT:   assign %c, <error>
// CHECK:STDOUT: }
