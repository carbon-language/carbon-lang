// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class C {
  var a: (i32, i32, i32);
}

fn F(c: C) -> i32 {
  // TODO: `c.a` is a value expression here, which forces a value binding as
  // part of the member access, creating a tuple value temporary. We could
  // defer performing the value binding to avoid creating this temporary.
  return c.a[1];
}

// CHECK:STDOUT: ; ModuleID = 'value_access.carbon'
// CHECK:STDOUT: source_filename = "value_access.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @F(ptr %c) {
// CHECK:STDOUT:   %a = getelementptr inbounds { { i32, i32, i32 } }, ptr %c, i32 0, i32 0
// CHECK:STDOUT:   %tuple.elem = getelementptr inbounds { i32, i32, i32 }, ptr %a, i32 0, i32 0
// CHECK:STDOUT:   %1 = load i32, ptr %tuple.elem, align 4
// CHECK:STDOUT:   %tuple.elem1 = getelementptr inbounds { i32, i32, i32 }, ptr %a, i32 0, i32 1
// CHECK:STDOUT:   %2 = load i32, ptr %tuple.elem1, align 4
// CHECK:STDOUT:   %tuple.elem2 = getelementptr inbounds { i32, i32, i32 }, ptr %a, i32 0, i32 2
// CHECK:STDOUT:   %3 = load i32, ptr %tuple.elem2, align 4
// CHECK:STDOUT:   %tuple = alloca { i32, i32, i32 }, align 8
// CHECK:STDOUT:   %4 = getelementptr inbounds { i32, i32, i32 }, ptr %tuple, i32 0, i32 0
// CHECK:STDOUT:   store i32 %1, ptr %4, align 4
// CHECK:STDOUT:   %5 = getelementptr inbounds { i32, i32, i32 }, ptr %tuple, i32 0, i32 1
// CHECK:STDOUT:   store i32 %2, ptr %5, align 4
// CHECK:STDOUT:   %6 = getelementptr inbounds { i32, i32, i32 }, ptr %tuple, i32 0, i32 2
// CHECK:STDOUT:   store i32 %3, ptr %6, align 4
// CHECK:STDOUT:   %tuple.index = getelementptr inbounds { i32, i32, i32 }, ptr %tuple, i32 0, i32 1
// CHECK:STDOUT:   %tuple.index.load = load i32, ptr %tuple.index, align 4
// CHECK:STDOUT:   ret i32 %tuple.index.load
// CHECK:STDOUT: }
