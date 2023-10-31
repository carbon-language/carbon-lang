// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class C {
  var a: (i32, i32, i32);
}

fn F(c: C) -> i32 {
  return c.a[1];
}

// CHECK:STDOUT: ; ModuleID = 'value_access.carbon'
// CHECK:STDOUT: source_filename = "value_access.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @F(ptr %c) {
// CHECK:STDOUT:   %a = getelementptr inbounds { { i32, i32, i32 } }, ptr %c, i32 0, i32 0
// CHECK:STDOUT:   %tuple.index = getelementptr inbounds { i32, i32, i32 }, ptr %a, i32 0, i32 1
// CHECK:STDOUT:   %1 = load i32, ptr %tuple.index, align 4
// CHECK:STDOUT:   ret i32 %1
// CHECK:STDOUT: }
