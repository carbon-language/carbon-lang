// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var x: {.a: i32} = {.a = 4};
var y: {.a: i32} = x;

// CHECK:STDOUT: file "one_entry.carbon" {
// CHECK:STDOUT:   %.loc7_16: type = struct_type {.a: i32}
// CHECK:STDOUT:   %x.loc7_5.1: ref {.a: i32} = var "x"
// CHECK:STDOUT:   %x.loc7_5.2: ref {.a: i32} = bind_name "x", %x.loc7_5.1
// CHECK:STDOUT:   %.loc7_26: i32 = int_literal 4
// CHECK:STDOUT:   %.loc7_27.1: {.a: i32} = struct_literal (%.loc7_26)
// CHECK:STDOUT:   %.loc7_27.2: init {.a: i32} = struct_init %.loc7_27.1, (%.loc7_26)
// CHECK:STDOUT:   assign %x.loc7_5.1, %.loc7_27.2
// CHECK:STDOUT:   %.loc8_16: type = struct_type {.a: i32}
// CHECK:STDOUT:   %y.loc8_5.1: ref {.a: i32} = var "y"
// CHECK:STDOUT:   %y.loc8_5.2: ref {.a: i32} = bind_name "y", %y.loc8_5.1
// CHECK:STDOUT:   %x.ref: ref {.a: i32} = name_reference "x", %x.loc7_5.2
// CHECK:STDOUT:   %.loc8_20.1: ref i32 = struct_access %x.ref, member0
// CHECK:STDOUT:   %.loc8_20.2: i32 = bind_value %.loc8_20.1
// CHECK:STDOUT:   %.loc8_20.3: init {.a: i32} = struct_init %x.ref, (%.loc8_20.2)
// CHECK:STDOUT:   assign %y.loc8_5.1, %.loc8_20.3
// CHECK:STDOUT: }
