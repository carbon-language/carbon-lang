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
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define %EmptyTupleType @Main() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %var = alloca i32, align 4
// CHECK:STDOUT:   %Echo = call i32 @Echo(i32 1)
// CHECK:STDOUT:   store i32 %Echo, ptr %var, align 4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder i32 1, { 1, 2, 0 }

fn Echo(a: i32) -> i32 {
  // TODO: `return a;` requires the parameter to be in name lookup.
  return 1;
}

fn Main() {
  var b: i32 = Echo(1);
}
