// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var a: (i32,) = (12,);
var b: (i32,) = a;
var c: i32 = b[0];

// CHECK:STDOUT: file "tuple_element_access.carbon" {
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
// CHECK:STDOUT:   %.loc9_16: i32 = int_literal 0
// CHECK:STDOUT:   %.loc9_17.1: ref i32 = tuple_index %b, %.loc9_16
// CHECK:STDOUT:   %.loc9_17.2: i32 = bind_value %.loc9_17.1
// CHECK:STDOUT:   assign %c, %.loc9_17.2
// CHECK:STDOUT: }
