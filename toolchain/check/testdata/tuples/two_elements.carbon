// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var x: (i32, i32) = (4, 102);
var y: (i32, i32) = x;

// CHECK:STDOUT: file "two_elements.carbon" {
// CHECK:STDOUT:   %.loc7_9: type = stub_reference i32
// CHECK:STDOUT:   %.loc7_14: type = stub_reference i32
// CHECK:STDOUT:   %.loc7_17.1: type = tuple_type (type, type)
// CHECK:STDOUT:   %.loc7_17.2: (type, type) = tuple_literal (%.loc7_9, %.loc7_14)
// CHECK:STDOUT:   %.loc7_17.3: type = tuple_type (i32, i32)
// CHECK:STDOUT:   %x: ref (i32, i32) = var "x"
// CHECK:STDOUT:   %.loc7_22.1: i32 = int_literal 4
// CHECK:STDOUT:   %.loc7_22.2: i32 = stub_reference %.loc7_22.1
// CHECK:STDOUT:   %.loc7_25.1: i32 = int_literal 102
// CHECK:STDOUT:   %.loc7_25.2: i32 = stub_reference %.loc7_25.1
// CHECK:STDOUT:   %.loc7_28: (i32, i32) = tuple_literal (%.loc7_22.2, %.loc7_25.2)
// CHECK:STDOUT:   assign %x, %.loc7_28
// CHECK:STDOUT:   %.loc8_9: type = stub_reference i32
// CHECK:STDOUT:   %.loc8_14: type = stub_reference i32
// CHECK:STDOUT:   %.loc8_17: (type, type) = tuple_literal (%.loc8_9, %.loc8_14)
// CHECK:STDOUT:   %y: ref (i32, i32) = var "y"
// CHECK:STDOUT:   %.loc7_5: (i32, i32) = bind_value %x
// CHECK:STDOUT:   assign %y, %.loc7_5
// CHECK:STDOUT: }
