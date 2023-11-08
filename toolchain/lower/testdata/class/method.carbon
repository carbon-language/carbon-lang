// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class C {
  var a: i32;

  fn Get[self: C]() -> i32;
  fn Set[addr self: C*](n: i32);
}

fn F(p: C*) {
  let n: i32 = (*p).Get();
  (*p).Set(n);
}

// CHECK:STDOUT: ; ModuleID = 'method.carbon'
// CHECK:STDOUT: source_filename = "method.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @Get(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @Set(ptr, i32)
// CHECK:STDOUT:
// CHECK:STDOUT: define void @F(ptr %p) {
// CHECK:STDOUT:   %Get = call i32 @Get(ptr %p)
// CHECK:STDOUT:   call void @Set(ptr %p, i32 %Get)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
