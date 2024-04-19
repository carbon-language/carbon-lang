// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> i32 {
  var a: (i32, i32, i32) = (1, 2, 3);
  var b: (i32, i32) = (4, 5);
  let c: ((i32, i32, i32), (i32, i32)) = (a, b);
  return c[1][1];
}

// CHECK:STDOUT: ; ModuleID = 'tuple.carbon'
// CHECK:STDOUT: source_filename = "tuple.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @F() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca { i32, i32, i32 }, align 8
// CHECK:STDOUT:   %.loc8_36.2.tuple.elem = getelementptr inbounds { i32, i32, i32 }, ptr %a.var, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %.loc8_36.2.tuple.elem, align 4
// CHECK:STDOUT:   %.loc8_36.4.tuple.elem = getelementptr inbounds { i32, i32, i32 }, ptr %a.var, i32 0, i32 1
// CHECK:STDOUT:   store i32 2, ptr %.loc8_36.4.tuple.elem, align 4
// CHECK:STDOUT:   %.loc8_36.6.tuple.elem = getelementptr inbounds { i32, i32, i32 }, ptr %a.var, i32 0, i32 2
// CHECK:STDOUT:   store i32 3, ptr %.loc8_36.6.tuple.elem, align 4
// CHECK:STDOUT:   %b.var = alloca { i32, i32 }, align 8
// CHECK:STDOUT:   %.loc9_28.2.tuple.elem = getelementptr inbounds { i32, i32 }, ptr %b.var, i32 0, i32 0
// CHECK:STDOUT:   store i32 4, ptr %.loc9_28.2.tuple.elem, align 4
// CHECK:STDOUT:   %.loc9_28.4.tuple.elem = getelementptr inbounds { i32, i32 }, ptr %b.var, i32 0, i32 1
// CHECK:STDOUT:   store i32 5, ptr %.loc9_28.4.tuple.elem, align 4
// CHECK:STDOUT:   %.loc10_43.1.tuple.elem = getelementptr inbounds { i32, i32, i32 }, ptr %a.var, i32 0, i32 0
// CHECK:STDOUT:   %.loc10_43.2 = load i32, ptr %.loc10_43.1.tuple.elem, align 4
// CHECK:STDOUT:   %.loc10_43.3.tuple.elem = getelementptr inbounds { i32, i32, i32 }, ptr %a.var, i32 0, i32 1
// CHECK:STDOUT:   %.loc10_43.4 = load i32, ptr %.loc10_43.3.tuple.elem, align 4
// CHECK:STDOUT:   %.loc10_43.5.tuple.elem = getelementptr inbounds { i32, i32, i32 }, ptr %a.var, i32 0, i32 2
// CHECK:STDOUT:   %.loc10_43.6 = load i32, ptr %.loc10_43.5.tuple.elem, align 4
// CHECK:STDOUT:   %.loc10_43.7.tuple = alloca { i32, i32, i32 }, align 8
// CHECK:STDOUT:   %.loc10_43.7 = getelementptr inbounds { i32, i32, i32 }, ptr %.loc10_43.7.tuple, i32 0, i32 0
// CHECK:STDOUT:   store i32 %.loc10_43.2, ptr %.loc10_43.7, align 4
// CHECK:STDOUT:   %.loc10_43.71 = getelementptr inbounds { i32, i32, i32 }, ptr %.loc10_43.7.tuple, i32 0, i32 1
// CHECK:STDOUT:   store i32 %.loc10_43.4, ptr %.loc10_43.71, align 4
// CHECK:STDOUT:   %.loc10_43.72 = getelementptr inbounds { i32, i32, i32 }, ptr %.loc10_43.7.tuple, i32 0, i32 2
// CHECK:STDOUT:   store i32 %.loc10_43.6, ptr %.loc10_43.72, align 4
// CHECK:STDOUT:   %.loc10_46.1.tuple.elem = getelementptr inbounds { i32, i32 }, ptr %b.var, i32 0, i32 0
// CHECK:STDOUT:   %.loc10_46.2 = load i32, ptr %.loc10_46.1.tuple.elem, align 4
// CHECK:STDOUT:   %.loc10_46.3.tuple.elem = getelementptr inbounds { i32, i32 }, ptr %b.var, i32 0, i32 1
// CHECK:STDOUT:   %.loc10_46.4 = load i32, ptr %.loc10_46.3.tuple.elem, align 4
// CHECK:STDOUT:   %.loc10_46.5.tuple = alloca { i32, i32 }, align 8
// CHECK:STDOUT:   %.loc10_46.5 = getelementptr inbounds { i32, i32 }, ptr %.loc10_46.5.tuple, i32 0, i32 0
// CHECK:STDOUT:   store i32 %.loc10_46.2, ptr %.loc10_46.5, align 4
// CHECK:STDOUT:   %.loc10_46.53 = getelementptr inbounds { i32, i32 }, ptr %.loc10_46.5.tuple, i32 0, i32 1
// CHECK:STDOUT:   store i32 %.loc10_46.4, ptr %.loc10_46.53, align 4
// CHECK:STDOUT:   %.loc10_47.4.tuple = alloca { ptr, ptr }, align 8
// CHECK:STDOUT:   %.loc10_47.4 = getelementptr inbounds { ptr, ptr }, ptr %.loc10_47.4.tuple, i32 0, i32 0
// CHECK:STDOUT:   store ptr %.loc10_43.7.tuple, ptr %.loc10_47.4, align 8
// CHECK:STDOUT:   %.loc10_47.44 = getelementptr inbounds { ptr, ptr }, ptr %.loc10_47.4.tuple, i32 0, i32 1
// CHECK:STDOUT:   store ptr %.loc10_46.5.tuple, ptr %.loc10_47.44, align 8
// CHECK:STDOUT:   %.loc11_13.tuple.index = getelementptr inbounds { ptr, ptr }, ptr %.loc10_47.4.tuple, i32 0, i32 1
// CHECK:STDOUT:   %.loc11_13.tuple.index.load = load ptr, ptr %.loc11_13.tuple.index, align 8
// CHECK:STDOUT:   %.loc11_16.tuple.index = getelementptr inbounds { i32, i32 }, ptr %.loc11_13.tuple.index.load, i32 0, i32 1
// CHECK:STDOUT:   %.loc11_16.tuple.index.load = load i32, ptr %.loc11_16.tuple.index, align 4
// CHECK:STDOUT:   ret i32 %.loc11_16.tuple.index.load
// CHECK:STDOUT: }
