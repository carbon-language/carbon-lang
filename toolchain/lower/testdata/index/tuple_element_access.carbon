// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Run() -> i32 {
  var a: (i32, i32, i32) = (0, 1, 2);
  var b: i32 = a[0];
  var c: i32 = a[2];
  return 0;
}

// CHECK:STDOUT: ; ModuleID = 'tuple_element_access.carbon'
// CHECK:STDOUT: source_filename = "tuple_element_access.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @main() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca { i32, i32, i32 }, align 8
// CHECK:STDOUT:   %.loc8_36.2.tuple.elem = getelementptr inbounds { i32, i32, i32 }, ptr %a.var, i32 0, i32 0
// CHECK:STDOUT:   store i32 0, ptr %.loc8_36.2.tuple.elem, align 4
// CHECK:STDOUT:   %.loc8_36.4.tuple.elem = getelementptr inbounds { i32, i32, i32 }, ptr %a.var, i32 0, i32 1
// CHECK:STDOUT:   store i32 1, ptr %.loc8_36.4.tuple.elem, align 4
// CHECK:STDOUT:   %.loc8_36.6.tuple.elem = getelementptr inbounds { i32, i32, i32 }, ptr %a.var, i32 0, i32 2
// CHECK:STDOUT:   store i32 2, ptr %.loc8_36.6.tuple.elem, align 4
// CHECK:STDOUT:   %b.var = alloca i32, align 4
// CHECK:STDOUT:   %.loc9_19.1.tuple.index = getelementptr inbounds { i32, i32, i32 }, ptr %a.var, i32 0, i32 0
// CHECK:STDOUT:   %.loc9_19.2 = load i32, ptr %.loc9_19.1.tuple.index, align 4
// CHECK:STDOUT:   store i32 %.loc9_19.2, ptr %b.var, align 4
// CHECK:STDOUT:   %c.var = alloca i32, align 4
// CHECK:STDOUT:   %.loc10_19.1.tuple.index = getelementptr inbounds { i32, i32, i32 }, ptr %a.var, i32 0, i32 2
// CHECK:STDOUT:   %.loc10_19.2 = load i32, ptr %.loc10_19.1.tuple.index, align 4
// CHECK:STDOUT:   store i32 %.loc10_19.2, ptr %c.var, align 4
// CHECK:STDOUT:   ret i32 0
// CHECK:STDOUT: }
