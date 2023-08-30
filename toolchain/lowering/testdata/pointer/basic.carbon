// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn G(p: i32*) -> i32 {
  // TODO: The LLVM IR generated here is wrong, missing a load.
  return *p;
}

fn F() -> i32 {
  var n: i32 = 0;
  return G(&n);
}

// CHECK:STDOUT: ; ModuleID = 'basic.carbon'
// CHECK:STDOUT: source_filename = "basic.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @G(ptr %p) {
// CHECK:STDOUT:   %1 = load i32, ptr %p, align 4
// CHECK:STDOUT:   ret i32 %1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @F() {
// CHECK:STDOUT:   %n = alloca i32, align 4
// CHECK:STDOUT:   store i32 0, ptr %n, align 4
// CHECK:STDOUT:   %1 = load ptr, ptr %n, align 8
// CHECK:STDOUT:   %G = call i32 @G(ptr %1)
// CHECK:STDOUT:   %temp = alloca i32, align 4
// CHECK:STDOUT:   store i32 %G, ptr %temp, align 4
// CHECK:STDOUT:   %2 = load i32, ptr %temp, align 4
// CHECK:STDOUT:   ret i32 %2
// CHECK:STDOUT: }
