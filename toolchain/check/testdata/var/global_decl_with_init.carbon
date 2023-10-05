// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var x: i32 = 0;

// CHECK:STDOUT: file "global_decl_with_init.carbon" {
// CHECK:STDOUT:   %x: ref i32 = var "x"
// CHECK:STDOUT:   %.loc7: i32 = int_literal 0
// CHECK:STDOUT:   assign %x, %.loc7
// CHECK:STDOUT: }
