// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class C {
  var a: i32;
  var b: C*;
}

fn F(c: C) -> i32 {
  return (*c.b).a;
}

fn Run() -> i32 {
  var c: C;
  c.a = 1;
  c.b = &c;
  return F(c);
}

// CHECK:STDOUT: ; ModuleID = 'field.carbon'
// CHECK:STDOUT: source_filename = "field.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @F(ptr %c) {
// CHECK:STDOUT:   %b = getelementptr inbounds { i32, ptr }, ptr %c, i32 0, i32 1
// CHECK:STDOUT:   %1 = load ptr, ptr %b, align 8
// CHECK:STDOUT:   %a = getelementptr inbounds { i32, ptr }, ptr %1, i32 0, i32 0
// CHECK:STDOUT:   %2 = load i32, ptr %a, align 4
// CHECK:STDOUT:   ret i32 %2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @main() {
// CHECK:STDOUT:   %c = alloca { i32, ptr }, align 8
// CHECK:STDOUT:   %a = getelementptr inbounds { i32, ptr }, ptr %c, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %a, align 4
// CHECK:STDOUT:   %b = getelementptr inbounds { i32, ptr }, ptr %c, i32 0, i32 1
// CHECK:STDOUT:   store ptr %c, ptr %b, align 8
// CHECK:STDOUT:   %F = call i32 @F(ptr %c)
// CHECK:STDOUT:   ret i32 %F
// CHECK:STDOUT: }
