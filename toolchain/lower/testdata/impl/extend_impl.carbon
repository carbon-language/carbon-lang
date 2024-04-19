// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

interface I {
  fn F() -> i32;
}

class A {
  extend impl as I {
    fn F() -> i32 {
      return 0;
    }
  }
}

fn TypeAccess() -> i32 {
  return A.F();
}

fn InstanceAccess(a: A) -> i32 {
  return a.F();
}

// CHECK:STDOUT: ; ModuleID = 'extend_impl.carbon'
// CHECK:STDOUT: source_filename = "extend_impl.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @F() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @TypeAccess() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F.call = call i32 @F()
// CHECK:STDOUT:   ret i32 %F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @InstanceAccess(ptr %a) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F.call = call i32 @F()
// CHECK:STDOUT:   ret i32 %F.call
// CHECK:STDOUT: }
