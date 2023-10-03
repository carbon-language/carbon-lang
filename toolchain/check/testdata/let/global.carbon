// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

let n: i32 = 1;

fn F() -> i32 { return n; }

// CHECK:STDOUT: file "global.carbon" {
// CHECK:STDOUT:   %.loc7: i32 = int_literal 1
// CHECK:STDOUT:   %n: i32 = bind_name "n", %.loc7
// CHECK:STDOUT:   %.loc9 = fn_decl @F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return package.%n
// CHECK:STDOUT: }
