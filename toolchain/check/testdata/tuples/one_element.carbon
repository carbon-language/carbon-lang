// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var x: (i32,) = (4,);
var y: (i32,) = x;

// CHECK:STDOUT: file "one_element.carbon" {
// CHECK:STDOUT:   %.loc7_9: type = stub_reference i32
// CHECK:STDOUT:   %.loc7_13.1: type = tuple_type (type)
// CHECK:STDOUT:   %.loc7_13.2: (type,) = tuple_literal (%.loc7_9)
// CHECK:STDOUT:   %.loc7_13.3: type = tuple_type (i32)
// CHECK:STDOUT:   %x: ref (i32,) = var "x"
// CHECK:STDOUT:   %.loc7_18.1: i32 = int_literal 4
// CHECK:STDOUT:   %.loc7_18.2: i32 = stub_reference %.loc7_18.1
// CHECK:STDOUT:   %.loc7_20: (i32,) = tuple_literal (%.loc7_18.2)
// CHECK:STDOUT:   assign %x, %.loc7_20
// CHECK:STDOUT:   %.loc8_9: type = stub_reference i32
// CHECK:STDOUT:   %.loc8_13: (type,) = tuple_literal (%.loc8_9)
// CHECK:STDOUT:   %y: ref (i32,) = var "y"
// CHECK:STDOUT:   %.loc7_5: (i32,) = bind_value %x
// CHECK:STDOUT:   assign %y, %.loc7_5
// CHECK:STDOUT: }
