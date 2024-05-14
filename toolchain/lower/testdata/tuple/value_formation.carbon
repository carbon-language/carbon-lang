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
// CHECK:STDOUT:   %tuple.loc12_6 = alloca { i32, i32, i32 }, align 8
// CHECK:STDOUT:   %tuple.loc12_61 = getelementptr inbounds { i32, i32, i32 }, ptr %tuple.loc12_6, i32 0, i32 0
// CHECK:STDOUT:   store i32 %.loc12_6.2, ptr %tuple.loc12_61, align 4
// CHECK:STDOUT:   %tuple.loc12_62 = getelementptr inbounds { i32, i32, i32 }, ptr %tuple.loc12_6, i32 0, i32 1
// CHECK:STDOUT:   store i32 %.loc12_6.4, ptr %tuple.loc12_62, align 4
// CHECK:STDOUT:   %tuple.loc12_63 = getelementptr inbounds { i32, i32, i32 }, ptr %tuple.loc12_6, i32 0, i32 2
// CHECK:STDOUT:   store i32 %.loc12_6.6, ptr %tuple.loc12_63, align 4
// CHECK:STDOUT:   %.loc12_9.1.tuple.elem = getelementptr inbounds { i32, i32, i32 }, ptr %b.var, i32 0, i32 0
// CHECK:STDOUT:   %.loc12_9.2 = load i32, ptr %.loc12_9.1.tuple.elem, align 4
// CHECK:STDOUT:   %.loc12_9.3.tuple.elem = getelementptr inbounds { i32, i32, i32 }, ptr %b.var, i32 0, i32 1
// CHECK:STDOUT:   %.loc12_9.4 = load i32, ptr %.loc12_9.3.tuple.elem, align 4
// CHECK:STDOUT:   %.loc12_9.5.tuple.elem = getelementptr inbounds { i32, i32, i32 }, ptr %b.var, i32 0, i32 2
// CHECK:STDOUT:   %.loc12_9.6 = load i32, ptr %.loc12_9.5.tuple.elem, align 4
// CHECK:STDOUT:   %tuple.loc12_9 = alloca { i32, i32, i32 }, align 8
// CHECK:STDOUT:   %tuple.loc12_94 = getelementptr inbounds { i32, i32, i32 }, ptr %tuple.loc12_9, i32 0, i32 0
// CHECK:STDOUT:   store i32 %.loc12_9.2, ptr %tuple.loc12_94, align 4
// CHECK:STDOUT:   %tuple.loc12_95 = getelementptr inbounds { i32, i32, i32 }, ptr %tuple.loc12_9, i32 0, i32 1
// CHECK:STDOUT:   store i32 %.loc12_9.4, ptr %tuple.loc12_95, align 4
// CHECK:STDOUT:   %tuple.loc12_96 = getelementptr inbounds { i32, i32, i32 }, ptr %tuple.loc12_9, i32 0, i32 2
// CHECK:STDOUT:   store i32 %.loc12_9.6, ptr %tuple.loc12_96, align 4
// CHECK:STDOUT:   %tuple.loc12_10 = alloca { ptr, ptr }, align 8
// CHECK:STDOUT:   %tuple.loc12_107 = getelementptr inbounds { ptr, ptr }, ptr %tuple.loc12_10, i32 0, i32 0
// CHECK:STDOUT:   store ptr %tuple.loc12_6, ptr %tuple.loc12_107, align 8
// CHECK:STDOUT:   %tuple.loc12_108 = getelementptr inbounds { ptr, ptr }, ptr %tuple.loc12_10, i32 0, i32 1
// CHECK:STDOUT:   store ptr %tuple.loc12_9, ptr %tuple.loc12_108, align 8
// CHECK:STDOUT:   call void @G(ptr %tuple.loc12_10)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
