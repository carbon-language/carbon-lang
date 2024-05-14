// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F(p: i32**) -> i32 {
  var a: i32** = p;
  var b: i32* = *p;
  var c: i32** = &b;
  return **c;
}

// CHECK:STDOUT: ; ModuleID = 'pointer_to_pointer.carbon'
// CHECK:STDOUT: source_filename = "pointer_to_pointer.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @F(ptr %p) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %p, ptr %a.var, align 8
// CHECK:STDOUT:   %b.var = alloca ptr, align 8
// CHECK:STDOUT:   %.loc9_17.2 = load ptr, ptr %p, align 8
// CHECK:STDOUT:   store ptr %.loc9_17.2, ptr %b.var, align 8
// CHECK:STDOUT:   %c.var = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %b.var, ptr %c.var, align 8
// CHECK:STDOUT:   %.loc11_12 = load ptr, ptr %c.var, align 8
// CHECK:STDOUT:   %.loc11_11.2 = load ptr, ptr %.loc11_12, align 8
// CHECK:STDOUT:   %.loc11_10.2 = load i32, ptr %.loc11_11.2, align 4
// CHECK:STDOUT:   ret i32 %.loc11_10.2
// CHECK:STDOUT: }
