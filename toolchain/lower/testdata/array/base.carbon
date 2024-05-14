// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Run() {
  var a: [i32; 1] = (1,);
  var b: [f64; 2] = (11.1, 2.2,);
  var c: [(); 5] = ((), (), (), (), (),);
  var d: (i32, i32, i32) = (1, 2, 3);
  var e: [i32; 3] = d;
}

// CHECK:STDOUT: ; ModuleID = 'base.carbon'
// CHECK:STDOUT: source_filename = "base.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @array.1.loc8_25 = internal constant [1 x i32] [i32 1]
// CHECK:STDOUT: @array.2.loc9_33 = internal constant [2 x double] [double 0x4026333333333334, double 2.200000e+00]
// CHECK:STDOUT: @array.3.loc10_41 = internal constant [5 x {}] zeroinitializer
// CHECK:STDOUT: @tuple.2.loc11_37 = internal constant { i32, i32, i32 } { i32 1, i32 2, i32 3 }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @main() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca [1 x i32], align 4
// CHECK:STDOUT:   %.loc8_24.3.array.index = getelementptr inbounds [1 x i32], ptr %a.var, i32 0, i32 0
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %a.var, ptr align 4 @array.1.loc8_25, i64 4, i1 false)
// CHECK:STDOUT:   %b.var = alloca [2 x double], align 8
// CHECK:STDOUT:   %.loc9_32.3.array.index = getelementptr inbounds [2 x double], ptr %b.var, i32 0, i32 0
// CHECK:STDOUT:   %.loc9_32.6.array.index = getelementptr inbounds [2 x double], ptr %b.var, i32 0, i32 1
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %b.var, ptr align 8 @array.2.loc9_33, i64 16, i1 false)
// CHECK:STDOUT:   %c.var = alloca [5 x {}], align 8
// CHECK:STDOUT:   %.loc10_40.3.array.index = getelementptr inbounds [5 x {}], ptr %c.var, i32 0, i32 0
// CHECK:STDOUT:   %.loc10_40.6.array.index = getelementptr inbounds [5 x {}], ptr %c.var, i32 0, i32 1
// CHECK:STDOUT:   %.loc10_40.9.array.index = getelementptr inbounds [5 x {}], ptr %c.var, i32 0, i32 2
// CHECK:STDOUT:   %.loc10_40.12.array.index = getelementptr inbounds [5 x {}], ptr %c.var, i32 0, i32 3
// CHECK:STDOUT:   %.loc10_40.15.array.index = getelementptr inbounds [5 x {}], ptr %c.var, i32 0, i32 4
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @array.3.loc10_41, i64 0, i1 false)
// CHECK:STDOUT:   %d.var = alloca { i32, i32, i32 }, align 8
// CHECK:STDOUT:   %.loc11_36.2.tuple.elem = getelementptr inbounds { i32, i32, i32 }, ptr %d.var, i32 0, i32 0
// CHECK:STDOUT:   %.loc11_36.4.tuple.elem = getelementptr inbounds { i32, i32, i32 }, ptr %d.var, i32 0, i32 1
// CHECK:STDOUT:   %.loc11_36.6.tuple.elem = getelementptr inbounds { i32, i32, i32 }, ptr %d.var, i32 0, i32 2
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %d.var, ptr align 4 @tuple.2.loc11_37, i64 12, i1 false)
// CHECK:STDOUT:   %e.var = alloca [3 x i32], align 4
// CHECK:STDOUT:   %.loc12_21.1.tuple.elem = getelementptr inbounds { i32, i32, i32 }, ptr %d.var, i32 0, i32 0
// CHECK:STDOUT:   %.loc12_21.2 = load i32, ptr %.loc12_21.1.tuple.elem, align 4
// CHECK:STDOUT:   %.loc12_21.4.array.index = getelementptr inbounds [3 x i32], ptr %e.var, i32 0, i32 0
// CHECK:STDOUT:   store i32 %.loc12_21.2, ptr %.loc12_21.4.array.index, align 4
// CHECK:STDOUT:   %.loc12_21.6.tuple.elem = getelementptr inbounds { i32, i32, i32 }, ptr %d.var, i32 0, i32 1
// CHECK:STDOUT:   %.loc12_21.7 = load i32, ptr %.loc12_21.6.tuple.elem, align 4
// CHECK:STDOUT:   %.loc12_21.9.array.index = getelementptr inbounds [3 x i32], ptr %e.var, i32 0, i32 1
// CHECK:STDOUT:   store i32 %.loc12_21.7, ptr %.loc12_21.9.array.index, align 4
// CHECK:STDOUT:   %.loc12_21.11.tuple.elem = getelementptr inbounds { i32, i32, i32 }, ptr %d.var, i32 0, i32 2
// CHECK:STDOUT:   %.loc12_21.12 = load i32, ptr %.loc12_21.11.tuple.elem, align 4
// CHECK:STDOUT:   %.loc12_21.14.array.index = getelementptr inbounds [3 x i32], ptr %e.var, i32 0, i32 2
// CHECK:STDOUT:   store i32 %.loc12_21.12, ptr %.loc12_21.14.array.index, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #0
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
