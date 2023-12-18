// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

base class Base {
  var b: i32;
}

class Derived {
  extend base: Base;

  var d: i32;
}

fn Make() -> Derived {
  return {.base = {.b = 4}, .d = 7};
}

fn Access(d: Derived) -> (i32, i32) {
  return (d.d, d.base.b);
}

fn Convert(p: Derived*) -> Base* {
  return p;
}

// CHECK:STDOUT: ; ModuleID = 'base.carbon'
// CHECK:STDOUT: source_filename = "base.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Make(ptr sret({ { i32 }, i32 }) %return) {
// CHECK:STDOUT:   %base = getelementptr inbounds { { i32 }, i32 }, ptr %return, i32 0, i32 0
// CHECK:STDOUT:   %b = getelementptr inbounds { i32 }, ptr %base, i32 0, i32 0
// CHECK:STDOUT:   store i32 4, ptr %b, align 4
// CHECK:STDOUT:   %d = getelementptr inbounds { { i32 }, i32 }, ptr %return, i32 0, i32 1
// CHECK:STDOUT:   store i32 7, ptr %d, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Access(ptr sret({ i32, i32 }) %return, ptr %d) {
// CHECK:STDOUT:   %d1 = getelementptr inbounds { { i32 }, i32 }, ptr %d, i32 0, i32 1
// CHECK:STDOUT:   %1 = load i32, ptr %d1, align 4
// CHECK:STDOUT:   %base = getelementptr inbounds { { i32 }, i32 }, ptr %d, i32 0, i32 0
// CHECK:STDOUT:   %b = getelementptr inbounds { i32 }, ptr %base, i32 0, i32 0
// CHECK:STDOUT:   %2 = load i32, ptr %b, align 4
// CHECK:STDOUT:   %tuple.elem = getelementptr inbounds { i32, i32 }, ptr %return, i32 0, i32 0
// CHECK:STDOUT:   store i32 %1, ptr %tuple.elem, align 4
// CHECK:STDOUT:   %tuple.elem2 = getelementptr inbounds { i32, i32 }, ptr %return, i32 0, i32 1
// CHECK:STDOUT:   store i32 %2, ptr %tuple.elem2, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define ptr @Convert(ptr %p) {
// CHECK:STDOUT:   %base = getelementptr inbounds { { i32 }, i32 }, ptr %p, i32 0, i32 0
// CHECK:STDOUT:   ret ptr %base
// CHECK:STDOUT: }
