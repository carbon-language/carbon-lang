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
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: i32 = name_reference "n", package.%n
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
