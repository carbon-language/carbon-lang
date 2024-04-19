// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn G(ab: ((i32, i32, i32), (i32, i32, i32)));

fn F() {
  var a: (i32, i32, i32);
  var b: (i32, i32, i32);
  G((a, b));
}

// CHECK:STDOUT: ; ModuleID = 'value_formation.carbon'
// CHECK:STDOUT: source_filename = "value_formation.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @G(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: define void @F() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca { i32, i32, i32 }, align 8
// CHECK:STDOUT:   %b.var = alloca { i32, i32, i32 }, align 8
// CHECK:STDOUT:   %.loc12_6.1.tuple.elem = getelementptr inbounds { i32, i32, i32 }, ptr %a.var, i32 0, i32 0
// CHECK:STDOUT:   %.loc12_6.2 = load i32, ptr %.loc12_6.1.tuple.elem, align 4
// CHECK:STDOUT:   %.loc12_6.3.tuple.elem = getelementptr inbounds { i32, i32, i32 }, ptr %a.var, i32 0, i32 1
// CHECK:STDOUT:   %.loc12_6.4 = load i32, ptr %.loc12_6.3.tuple.elem, align 4
// CHECK:STDOUT:   %.loc12_6.5.tuple.elem = getelementptr inbounds { i32, i32, i32 }, ptr %a.var, i32 0, i32 2
// CHECK:STDOUT:   %.loc12_6.6 = load i32, ptr %.loc12_6.5.tuple.elem, align 4
// CHECK:STDOUT:   %.loc12_6.7.tuple = alloca { i32, i32, i32 }, align 8
// CHECK:STDOUT:   %.loc12_6.7 = getelementptr inbounds { i32, i32, i32 }, ptr %.loc12_6.7.tuple, i32 0, i32 0
// CHECK:STDOUT:   store i32 %.loc12_6.2, ptr %.loc12_6.7, align 4
// CHECK:STDOUT:   %.loc12_6.71 = getelementptr inbounds { i32, i32, i32 }, ptr %.loc12_6.7.tuple, i32 0, i32 1
// CHECK:STDOUT:   store i32 %.loc12_6.4, ptr %.loc12_6.71, align 4
// CHECK:STDOUT:   %.loc12_6.72 = getelementptr inbounds { i32, i32, i32 }, ptr %.loc12_6.7.tuple, i32 0, i32 2
// CHECK:STDOUT:   store i32 %.loc12_6.6, ptr %.loc12_6.72, align 4
// CHECK:STDOUT:   %.loc12_9.1.tuple.elem = getelementptr inbounds { i32, i32, i32 }, ptr %b.var, i32 0, i32 0
// CHECK:STDOUT:   %.loc12_9.2 = load i32, ptr %.loc12_9.1.tuple.elem, align 4
// CHECK:STDOUT:   %.loc12_9.3.tuple.elem = getelementptr inbounds { i32, i32, i32 }, ptr %b.var, i32 0, i32 1
// CHECK:STDOUT:   %.loc12_9.4 = load i32, ptr %.loc12_9.3.tuple.elem, align 4
// CHECK:STDOUT:   %.loc12_9.5.tuple.elem = getelementptr inbounds { i32, i32, i32 }, ptr %b.var, i32 0, i32 2
// CHECK:STDOUT:   %.loc12_9.6 = load i32, ptr %.loc12_9.5.tuple.elem, align 4
// CHECK:STDOUT:   %.loc12_9.7.tuple = alloca { i32, i32, i32 }, align 8
// CHECK:STDOUT:   %.loc12_9.7 = getelementptr inbounds { i32, i32, i32 }, ptr %.loc12_9.7.tuple, i32 0, i32 0
// CHECK:STDOUT:   store i32 %.loc12_9.2, ptr %.loc12_9.7, align 4
// CHECK:STDOUT:   %.loc12_9.73 = getelementptr inbounds { i32, i32, i32 }, ptr %.loc12_9.7.tuple, i32 0, i32 1
// CHECK:STDOUT:   store i32 %.loc12_9.4, ptr %.loc12_9.73, align 4
// CHECK:STDOUT:   %.loc12_9.74 = getelementptr inbounds { i32, i32, i32 }, ptr %.loc12_9.7.tuple, i32 0, i32 2
// CHECK:STDOUT:   store i32 %.loc12_9.6, ptr %.loc12_9.74, align 4
// CHECK:STDOUT:   %.loc12_10.4.tuple = alloca { ptr, ptr }, align 8
// CHECK:STDOUT:   %.loc12_10.4 = getelementptr inbounds { ptr, ptr }, ptr %.loc12_10.4.tuple, i32 0, i32 0
// CHECK:STDOUT:   store ptr %.loc12_6.7.tuple, ptr %.loc12_10.4, align 8
// CHECK:STDOUT:   %.loc12_10.45 = getelementptr inbounds { ptr, ptr }, ptr %.loc12_10.4.tuple, i32 0, i32 1
// CHECK:STDOUT:   store ptr %.loc12_9.7.tuple, ptr %.loc12_10.45, align 8
// CHECK:STDOUT:   call void @G(ptr %.loc12_10.4.tuple)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
