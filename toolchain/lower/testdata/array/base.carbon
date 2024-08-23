// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/array/base.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/array/base.carbon

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
// CHECK:STDOUT: @array.1.loc12_25 = internal constant [1 x i32] [i32 1]
// CHECK:STDOUT: @array.2.loc13_33 = internal constant [2 x double] [double 0x4026333333333334, double 2.200000e+00]
// CHECK:STDOUT: @array.3.loc14_41 = internal constant [5 x {}] zeroinitializer
// CHECK:STDOUT: @tuple.2.loc15_37 = internal constant { i32, i32, i32 } { i32 1, i32 2, i32 3 }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca [1 x i32], align 4
// CHECK:STDOUT:   %.loc12_24.3.array.index = getelementptr inbounds [1 x i32], ptr %a.var, i32 0, i32 0
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %a.var, ptr align 4 @array.1.loc12_25, i64 4, i1 false)
// CHECK:STDOUT:   %b.var = alloca [2 x double], align 8
// CHECK:STDOUT:   %.loc13_32.3.array.index = getelementptr inbounds [2 x double], ptr %b.var, i32 0, i32 0
// CHECK:STDOUT:   %.loc13_32.6.array.index = getelementptr inbounds [2 x double], ptr %b.var, i32 0, i32 1
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %b.var, ptr align 8 @array.2.loc13_33, i64 16, i1 false)
// CHECK:STDOUT:   %c.var = alloca [5 x {}], align 8
// CHECK:STDOUT:   %.loc14_40.3.array.index = getelementptr inbounds [5 x {}], ptr %c.var, i32 0, i32 0
// CHECK:STDOUT:   %.loc14_40.6.array.index = getelementptr inbounds [5 x {}], ptr %c.var, i32 0, i32 1
// CHECK:STDOUT:   %.loc14_40.9.array.index = getelementptr inbounds [5 x {}], ptr %c.var, i32 0, i32 2
// CHECK:STDOUT:   %.loc14_40.12.array.index = getelementptr inbounds [5 x {}], ptr %c.var, i32 0, i32 3
// CHECK:STDOUT:   %.loc14_40.15.array.index = getelementptr inbounds [5 x {}], ptr %c.var, i32 0, i32 4
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @array.3.loc14_41, i64 0, i1 false)
// CHECK:STDOUT:   %d.var = alloca { i32, i32, i32 }, align 8
// CHECK:STDOUT:   %.loc15_36.2.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %d.var, i32 0, i32 0
// CHECK:STDOUT:   %.loc15_36.4.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %d.var, i32 0, i32 1
// CHECK:STDOUT:   %.loc15_36.6.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %d.var, i32 0, i32 2
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %d.var, ptr align 4 @tuple.2.loc15_37, i64 12, i1 false)
// CHECK:STDOUT:   %e.var = alloca [3 x i32], align 4
// CHECK:STDOUT:   %.loc16_21.1.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %d.var, i32 0, i32 0
// CHECK:STDOUT:   %.loc16_21.2 = load i32, ptr %.loc16_21.1.tuple.elem, align 4
// CHECK:STDOUT:   %.loc16_21.4.array.index = getelementptr inbounds [3 x i32], ptr %e.var, i32 0, i32 0
// CHECK:STDOUT:   store i32 %.loc16_21.2, ptr %.loc16_21.4.array.index, align 4
// CHECK:STDOUT:   %.loc16_21.6.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %d.var, i32 0, i32 1
// CHECK:STDOUT:   %.loc16_21.7 = load i32, ptr %.loc16_21.6.tuple.elem, align 4
// CHECK:STDOUT:   %.loc16_21.9.array.index = getelementptr inbounds [3 x i32], ptr %e.var, i32 0, i32 1
// CHECK:STDOUT:   store i32 %.loc16_21.7, ptr %.loc16_21.9.array.index, align 4
// CHECK:STDOUT:   %.loc16_21.11.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %d.var, i32 0, i32 2
// CHECK:STDOUT:   %.loc16_21.12 = load i32, ptr %.loc16_21.11.tuple.elem, align 4
// CHECK:STDOUT:   %.loc16_21.14.array.index = getelementptr inbounds [3 x i32], ptr %e.var, i32 0, i32 2
// CHECK:STDOUT:   store i32 %.loc16_21.12, ptr %.loc16_21.14.array.index, align 4
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
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "base.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "main", scope: null, file: !3, line: 11, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
