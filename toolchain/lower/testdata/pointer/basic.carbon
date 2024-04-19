// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn G(p: i32*) -> i32 {
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
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc8_10.2 = load i32, ptr %p, align 4
// CHECK:STDOUT:   ret i32 %.loc8_10.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @F() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %n.var = alloca i32, align 4
// CHECK:STDOUT:   store i32 0, ptr %n.var, align 4
// CHECK:STDOUT:   %G.call = call i32 @G(ptr %n.var)
// CHECK:STDOUT:   ret i32 %G.call
// CHECK:STDOUT: }
