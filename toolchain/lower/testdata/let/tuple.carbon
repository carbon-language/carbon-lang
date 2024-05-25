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
// CHECK:STDOUT: @tuple.1.loc8_37 = internal constant { i32, i32, i32 } { i32 1, i32 2, i32 3 }
// CHECK:STDOUT: @tuple.2.loc9_29 = internal constant { i32, i32 } { i32 4, i32 5 }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @F() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca { i32, i32, i32 }, align 8
// CHECK:STDOUT:   %.loc8_36.2.tuple.elem = getelementptr inbounds { i32, i32, i32 }, ptr %a.var, i32 0, i32 0
// CHECK:STDOUT:   %.loc8_36.4.tuple.elem = getelementptr inbounds { i32, i32, i32 }, ptr %a.var, i32 0, i32 1
// CHECK:STDOUT:   %.loc8_36.6.tuple.elem = getelementptr inbounds { i32, i32, i32 }, ptr %a.var, i32 0, i32 2
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %a.var, ptr align 4 @tuple.1.loc8_37, i64 12, i1 false)
// CHECK:STDOUT:   %b.var = alloca { i32, i32 }, align 8
// CHECK:STDOUT:   %.loc9_28.2.tuple.elem = getelementptr inbounds { i32, i32 }, ptr %b.var, i32 0, i32 0
// CHECK:STDOUT:   %.loc9_28.4.tuple.elem = getelementptr inbounds { i32, i32 }, ptr %b.var, i32 0, i32 1
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %b.var, ptr align 4 @tuple.2.loc9_29, i64 8, i1 false)
// CHECK:STDOUT:   %.loc10_43.1.tuple.elem = getelementptr inbounds { i32, i32, i32 }, ptr %a.var, i32 0, i32 0
// CHECK:STDOUT:   %.loc10_43.2 = load i32, ptr %.loc10_43.1.tuple.elem, align 4
// CHECK:STDOUT:   %.loc10_43.3.tuple.elem = getelementptr inbounds { i32, i32, i32 }, ptr %a.var, i32 0, i32 1
// CHECK:STDOUT:   %.loc10_43.4 = load i32, ptr %.loc10_43.3.tuple.elem, align 4
// CHECK:STDOUT:   %.loc10_43.5.tuple.elem = getelementptr inbounds { i32, i32, i32 }, ptr %a.var, i32 0, i32 2
// CHECK:STDOUT:   %.loc10_43.6 = load i32, ptr %.loc10_43.5.tuple.elem, align 4
// CHECK:STDOUT:   %tuple.loc10_43 = alloca { i32, i32, i32 }, align 8
// CHECK:STDOUT:   %tuple.loc10_431 = getelementptr inbounds { i32, i32, i32 }, ptr %tuple.loc10_43, i32 0, i32 0
// CHECK:STDOUT:   store i32 %.loc10_43.2, ptr %tuple.loc10_431, align 4
// CHECK:STDOUT:   %tuple.loc10_432 = getelementptr inbounds { i32, i32, i32 }, ptr %tuple.loc10_43, i32 0, i32 1
// CHECK:STDOUT:   store i32 %.loc10_43.4, ptr %tuple.loc10_432, align 4
// CHECK:STDOUT:   %tuple.loc10_433 = getelementptr inbounds { i32, i32, i32 }, ptr %tuple.loc10_43, i32 0, i32 2
// CHECK:STDOUT:   store i32 %.loc10_43.6, ptr %tuple.loc10_433, align 4
// CHECK:STDOUT:   %.loc10_46.1.tuple.elem = getelementptr inbounds { i32, i32 }, ptr %b.var, i32 0, i32 0
// CHECK:STDOUT:   %.loc10_46.2 = load i32, ptr %.loc10_46.1.tuple.elem, align 4
// CHECK:STDOUT:   %.loc10_46.3.tuple.elem = getelementptr inbounds { i32, i32 }, ptr %b.var, i32 0, i32 1
// CHECK:STDOUT:   %.loc10_46.4 = load i32, ptr %.loc10_46.3.tuple.elem, align 4
// CHECK:STDOUT:   %tuple.loc10_46 = alloca { i32, i32 }, align 8
// CHECK:STDOUT:   %tuple.loc10_464 = getelementptr inbounds { i32, i32 }, ptr %tuple.loc10_46, i32 0, i32 0
// CHECK:STDOUT:   store i32 %.loc10_46.2, ptr %tuple.loc10_464, align 4
// CHECK:STDOUT:   %tuple.loc10_465 = getelementptr inbounds { i32, i32 }, ptr %tuple.loc10_46, i32 0, i32 1
// CHECK:STDOUT:   store i32 %.loc10_46.4, ptr %tuple.loc10_465, align 4
// CHECK:STDOUT:   %tuple.loc10_47 = alloca { ptr, ptr }, align 8
// CHECK:STDOUT:   %tuple.loc10_476 = getelementptr inbounds { ptr, ptr }, ptr %tuple.loc10_47, i32 0, i32 0
// CHECK:STDOUT:   store ptr %tuple.loc10_43, ptr %tuple.loc10_476, align 8
// CHECK:STDOUT:   %tuple.loc10_477 = getelementptr inbounds { ptr, ptr }, ptr %tuple.loc10_47, i32 0, i32 1
// CHECK:STDOUT:   store ptr %tuple.loc10_46, ptr %tuple.loc10_477, align 8
// CHECK:STDOUT:   %.loc11_13.tuple.index = getelementptr inbounds { ptr, ptr }, ptr %tuple.loc10_47, i32 0, i32 1
// CHECK:STDOUT:   %.loc11_13.tuple.index.load = load ptr, ptr %.loc11_13.tuple.index, align 8
// CHECK:STDOUT:   %.loc11_16.tuple.index = getelementptr inbounds { i32, i32 }, ptr %.loc11_13.tuple.index.load, i32 0, i32 1
// CHECK:STDOUT:   %.loc11_16.tuple.index.load = load i32, ptr %.loc11_16.tuple.index, align 4
// CHECK:STDOUT:   ret i32 %.loc11_16.tuple.index.load
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #0
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
