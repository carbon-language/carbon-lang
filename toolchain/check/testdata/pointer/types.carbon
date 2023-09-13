// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Ptr(p: i32*) -> i32* {
  return p;
}

fn ConstPtr(p: const i32*) -> (const i32)* {
  return p;
}

// CHECK:STDOUT: file "types.carbon" {
// CHECK:STDOUT:   %.loc7 = fn_decl @Ptr
// CHECK:STDOUT:   %.loc11 = fn_decl @ConstPtr
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Ptr(%p: i32*) -> i32* {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return %p
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConstPtr(%p: const i32*) -> const i32* {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return %p
// CHECK:STDOUT: }
