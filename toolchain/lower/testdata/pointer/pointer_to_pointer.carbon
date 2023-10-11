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
// CHECK:STDOUT:   %a = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %p, ptr %a, align 8
// CHECK:STDOUT:   %b = alloca ptr, align 8
// CHECK:STDOUT:   %1 = load ptr, ptr %p, align 8
// CHECK:STDOUT:   store ptr %1, ptr %b, align 8
// CHECK:STDOUT:   %c = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %b, ptr %c, align 8
// CHECK:STDOUT:   %2 = load ptr, ptr %c, align 8
// CHECK:STDOUT:   %3 = load ptr, ptr %2, align 8
// CHECK:STDOUT:   %4 = load i32, ptr %3, align 4
// CHECK:STDOUT:   ret i32 %4
// CHECK:STDOUT: }
