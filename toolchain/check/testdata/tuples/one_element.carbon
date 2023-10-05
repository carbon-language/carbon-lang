// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var x: (i32,) = (4,);
var y: (i32,) = x;

// CHECK:STDOUT: file "one_element.carbon" {
// CHECK:STDOUT:   %.loc7_13.1: type = tuple_type (type)
// CHECK:STDOUT:   %.loc7_13.2: (type,) = tuple_literal (i32)
// CHECK:STDOUT:   %.loc7_13.3: type = tuple_type (i32)
// CHECK:STDOUT:   %x: ref (i32,) = var "x"
// CHECK:STDOUT:   %.loc7_18: i32 = int_literal 4
// CHECK:STDOUT:   %.loc7_20.1: (i32,) = tuple_literal (%.loc7_18)
// CHECK:STDOUT:   %.loc7_20.2: init (i32,) = tuple_init %.loc7_20.1, (%.loc7_18)
// CHECK:STDOUT:   assign %x, %.loc7_20.2
// CHECK:STDOUT:   %.loc8_13: (type,) = tuple_literal (i32)
// CHECK:STDOUT:   %y: ref (i32,) = var "y"
// CHECK:STDOUT:   %x.ref: ref (i32,) = name_reference "x", %x
// CHECK:STDOUT:   %.loc8_17.1: ref i32 = tuple_access %x.ref, member0
// CHECK:STDOUT:   %.loc8_17.2: i32 = bind_value %.loc8_17.1
// CHECK:STDOUT:   %.loc8_17.3: init (i32,) = tuple_init %x.ref, (%.loc8_17.2)
// CHECK:STDOUT:   assign %y, %.loc8_17.3
// CHECK:STDOUT: }
