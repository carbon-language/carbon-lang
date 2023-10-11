// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var x: i32 = 0;
var y: i32 = x;

// CHECK:STDOUT: file "global_lookup.carbon" {
// CHECK:STDOUT:   %x: ref i32 = var "x"
// CHECK:STDOUT:   %.loc7: i32 = int_literal 0
// CHECK:STDOUT:   assign %x, %.loc7
// CHECK:STDOUT:   %y: ref i32 = var "y"
// CHECK:STDOUT:   %x.ref: ref i32 = name_reference "x", %x
// CHECK:STDOUT:   %.loc8: i32 = bind_value %x.ref
// CHECK:STDOUT:   assign %y, %.loc8
// CHECK:STDOUT: }
