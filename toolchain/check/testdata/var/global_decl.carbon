// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var x: i32;

// CHECK:STDOUT: file "global_decl.carbon" {
// CHECK:STDOUT:   %x.loc7_5.1: ref i32 = var "x"
// CHECK:STDOUT:   %x.loc7_5.2: ref i32 = bind_name "x", %x.loc7_5.1
// CHECK:STDOUT: }
