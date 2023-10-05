// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var test_i32: i32 = ((1) + (2));

// CHECK:STDOUT: file "parens.carbon" {
// CHECK:STDOUT:   %test_i32: ref i32 = var "test_i32"
// CHECK:STDOUT:   %.loc7_23: i32 = int_literal 1
// CHECK:STDOUT:   %.loc7_29: i32 = int_literal 2
// CHECK:STDOUT:   %.loc7_26: i32 = add %.loc7_23, %.loc7_29
// CHECK:STDOUT:   assign %test_i32, %.loc7_26
// CHECK:STDOUT: }
