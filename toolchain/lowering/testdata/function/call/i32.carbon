// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: ; ModuleID = 'i32.carbon'
// CHECK:STDOUT: source_filename = "i32.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: %EmptyTupleType = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @Echo(i32 %a) {
// CHECK:STDOUT:   ret i32 %a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define %EmptyTupleType @Main() {
// CHECK:STDOUT:   %var = alloca i32, align 4
// CHECK:STDOUT:   %Echo = call i32 @Echo(i32 1)
// CHECK:STDOUT:   store i32 %Echo, ptr %var, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }

fn Echo(a: i32) -> i32 {
  return a;
}

fn Main() {
  var b: i32 = Echo(1);
}
