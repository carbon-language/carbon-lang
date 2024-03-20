// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

interface I {
  fn F[self: Self]() -> i32;
}

class A {
  alias F = I.F;
  var n: i32;
}

impl A as I {
  fn F[self: A]() -> i32 {
    return self.n;
  }
}

fn Call(a: A) -> i32 {
  return a.F();
}

// CHECK:STDOUT: ; ModuleID = 'assoc_fn_alias.carbon'
// CHECK:STDOUT: source_filename = "assoc_fn_alias.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @F(ptr %self) {
// CHECK:STDOUT:   %n = getelementptr inbounds { i32 }, ptr %self, i32 0, i32 0
// CHECK:STDOUT:   %1 = load i32, ptr %n, align 4
// CHECK:STDOUT:   ret i32 %1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @Call(ptr %a) {
// CHECK:STDOUT:   %F = call i32 @F(ptr %a)
// CHECK:STDOUT:   ret i32 %F
// CHECK:STDOUT: }
