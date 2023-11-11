// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class C {
  var a: i32;

  fn Get[self: C]() -> i32;
  fn Set[addr self: C*]();
}

fn C.Get[self: C]() -> i32 {
  return self.a;
}

fn C.Set[addr self: C*]() {
  (*self).a = 1;
}

// CHECK:STDOUT: ; ModuleID = 'self.carbon'
// CHECK:STDOUT: source_filename = "self.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @Get(ptr %self) {
// CHECK:STDOUT:   %a = getelementptr inbounds { i32 }, ptr %self, i32 0, i32 0
// CHECK:STDOUT:   %1 = load i32, ptr %a, align 4
// CHECK:STDOUT:   ret i32 %1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Set(ptr %self) {
// CHECK:STDOUT:   %a = getelementptr inbounds { i32 }, ptr %self, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %a, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
