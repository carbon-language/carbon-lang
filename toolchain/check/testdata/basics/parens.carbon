// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "parens.carbon" {
// CHECK:STDOUT:   %test_i32: ref i32 = var "test_i32"

var test_i32: i32 = ((1) + (2));
// CHECK:STDOUT:   %.loc[[@LINE-1]]_23: i32 = int_literal 1
// CHECK:STDOUT:   %.loc[[@LINE-2]]_29: i32 = int_literal 2
// CHECK:STDOUT:   %.loc[[@LINE-3]]_26: i32 = add %.loc[[@LINE-3]]_23, %.loc[[@LINE-3]]_29
// CHECK:STDOUT:   assign %test_i32, %.loc[[@LINE-4]]_26
// CHECK:STDOUT: }
