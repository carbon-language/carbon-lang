// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Run() -> i32 {
  var x: (i32, i32) = (12, 7);
  var y: (i32, i32) = x;
  return 0;
}

// CHECK:STDOUT: ; ModuleID = 'two_entries.carbon'
// CHECK:STDOUT: source_filename = "two_entries.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @tuple.loc8_30 = internal constant { i32, i32 } { i32 12, i32 7 }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @main() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca { i32, i32 }, align 8
// CHECK:STDOUT:   %.loc8_29.2.tuple.elem = getelementptr inbounds { i32, i32 }, ptr %x.var, i32 0, i32 0
// CHECK:STDOUT:   %.loc8_29.4.tuple.elem = getelementptr inbounds { i32, i32 }, ptr %x.var, i32 0, i32 1
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %x.var, ptr align 4 @tuple.loc8_30, i64 8, i1 false)
// CHECK:STDOUT:   %y.var = alloca { i32, i32 }, align 8
// CHECK:STDOUT:   %.loc9_23.1.tuple.elem = getelementptr inbounds { i32, i32 }, ptr %x.var, i32 0, i32 0
// CHECK:STDOUT:   %.loc9_23.2 = load i32, ptr %.loc9_23.1.tuple.elem, align 4
// CHECK:STDOUT:   %.loc9_23.3.tuple.elem = getelementptr inbounds { i32, i32 }, ptr %y.var, i32 0, i32 0
// CHECK:STDOUT:   store i32 %.loc9_23.2, ptr %.loc9_23.3.tuple.elem, align 4
// CHECK:STDOUT:   %.loc9_23.5.tuple.elem = getelementptr inbounds { i32, i32 }, ptr %x.var, i32 0, i32 1
// CHECK:STDOUT:   %.loc9_23.6 = load i32, ptr %.loc9_23.5.tuple.elem, align 4
// CHECK:STDOUT:   %.loc9_23.7.tuple.elem = getelementptr inbounds { i32, i32 }, ptr %y.var, i32 0, i32 1
// CHECK:STDOUT:   store i32 %.loc9_23.6, ptr %.loc9_23.7.tuple.elem, align 4
// CHECK:STDOUT:   ret i32 0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #0
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
