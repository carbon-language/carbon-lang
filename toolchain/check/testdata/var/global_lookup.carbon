// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "global_lookup.carbon" {
// CHECK:STDOUT:   %x: ref i32 = var "x"

var x: i32 = 0;
// CHECK:STDOUT:   %.loc[[@LINE-1]]_14: i32 = int_literal 0
// CHECK:STDOUT:   assign %x, %.loc[[@LINE-2]]_14
// CHECK:STDOUT:   %y: ref i32 = var "y"
// CHECK:STDOUT:   %.loc[[@LINE-4]]_5: i32 = bind_value %x
// CHECK:STDOUT:   assign %y, %.loc[[@LINE-5]]_5
// CHECK:STDOUT: }
var y: i32 = x;
