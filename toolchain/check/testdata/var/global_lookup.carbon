// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var x: i32 = 0;
var y: i32 = x;

// CHECK:STDOUT: file "global_lookup.carbon" {
// CHECK:STDOUT:   %x: ref i32 = var "x"
// CHECK:STDOUT:   %.loc7_14: i32 = int_literal 0
// CHECK:STDOUT:   assign %x, %.loc7_14
// CHECK:STDOUT:   %y: ref i32 = var "y"
// CHECK:STDOUT:   %.loc7_5: i32 = bind_value %x
// CHECK:STDOUT:   assign %y, %.loc7_5
// CHECK:STDOUT: }
