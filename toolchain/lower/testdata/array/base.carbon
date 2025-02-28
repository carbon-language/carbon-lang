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
  var a: array(i32, 1) = (1,);
  var b: array(f64, 2) = (11.1, 2.2,);
  var c: array((), 5) = ((), (), (), (), (),);
  var d: (i32, i32, i32) = (1, 2, 3);
  var e: array(i32, 3) = d;
}

// CHECK:STDOUT: ; ModuleID = 'base.carbon'
// CHECK:STDOUT: source_filename = "base.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @array.237.loc12_3.2 = internal constant [1 x i32] [i32 1]
// CHECK:STDOUT: @array.6a2.loc13_3.2 = internal constant [2 x double] [double 0x4026333333333334, double 2.200000e+00]
// CHECK:STDOUT: @array.1cb.loc14_3.2 = internal constant [5 x {}] zeroinitializer
// CHECK:STDOUT: @tuple.loc15_3.2 = internal constant { i32, i32, i32 } { i32 1, i32 2, i32 3 }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca [1 x i32], align 4, !dbg !7
// CHECK:STDOUT:   %b.var = alloca [2 x double], align 8, !dbg !7
// CHECK:STDOUT:   %c.var = alloca [5 x {}], align 8, !dbg !7
// CHECK:STDOUT:   %d.var = alloca { i32, i32, i32 }, align 8, !dbg !7
// CHECK:STDOUT:   %e.var = alloca [3 x i32], align 4, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 4, ptr %a.var), !dbg !7
// CHECK:STDOUT:   %.loc12_29.3.array.index = getelementptr inbounds [1 x i32], ptr %a.var, i32 0, i64 0, !dbg !8
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %a.var, ptr align 4 @array.237.loc12_3.2, i64 4, i1 false), !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 16, ptr %b.var), !dbg !7
// CHECK:STDOUT:   %.loc13_37.2.array.index = getelementptr inbounds [2 x double], ptr %b.var, i32 0, i64 0, !dbg !9
// CHECK:STDOUT:   %.loc13_37.4.array.index = getelementptr inbounds [2 x double], ptr %b.var, i32 0, i64 1, !dbg !9
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %b.var, ptr align 8 @array.6a2.loc13_3.2, i64 16, i1 false), !dbg !10
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 0, ptr %c.var), !dbg !7
// CHECK:STDOUT:   %.loc14_45.2.array.index = getelementptr inbounds [5 x {}], ptr %c.var, i32 0, i64 0, !dbg !11
// CHECK:STDOUT:   %.loc14_45.4.array.index = getelementptr inbounds [5 x {}], ptr %c.var, i32 0, i64 1, !dbg !11
// CHECK:STDOUT:   %.loc14_45.6.array.index = getelementptr inbounds [5 x {}], ptr %c.var, i32 0, i64 2, !dbg !11
// CHECK:STDOUT:   %.loc14_45.8.array.index = getelementptr inbounds [5 x {}], ptr %c.var, i32 0, i64 3, !dbg !11
// CHECK:STDOUT:   %.loc14_45.10.array.index = getelementptr inbounds [5 x {}], ptr %c.var, i32 0, i64 4, !dbg !11
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %c.var, ptr align 1 @array.1cb.loc14_3.2, i64 0, i1 false), !dbg !12
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 12, ptr %d.var), !dbg !7
// CHECK:STDOUT:   %tuple.elem0.loc15.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %d.var, i32 0, i32 0, !dbg !13
// CHECK:STDOUT:   %tuple.elem1.loc15.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %d.var, i32 0, i32 1, !dbg !13
// CHECK:STDOUT:   %tuple.elem2.loc15.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %d.var, i32 0, i32 2, !dbg !13
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %d.var, ptr align 4 @tuple.loc15_3.2, i64 12, i1 false), !dbg !14
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 12, ptr %e.var), !dbg !7
// CHECK:STDOUT:   %tuple.elem0.loc16.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %d.var, i32 0, i32 0, !dbg !15
// CHECK:STDOUT:   %.loc16_26.1 = load i32, ptr %tuple.elem0.loc16.tuple.elem, align 4, !dbg !15
// CHECK:STDOUT:   %.loc16_26.2.array.index = getelementptr inbounds [3 x i32], ptr %e.var, i32 0, i64 0, !dbg !15
// CHECK:STDOUT:   store i32 %.loc16_26.1, ptr %.loc16_26.2.array.index, align 4, !dbg !15
// CHECK:STDOUT:   %tuple.elem1.loc16.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %d.var, i32 0, i32 1, !dbg !15
// CHECK:STDOUT:   %.loc16_26.4 = load i32, ptr %tuple.elem1.loc16.tuple.elem, align 4, !dbg !15
// CHECK:STDOUT:   %.loc16_26.5.array.index = getelementptr inbounds [3 x i32], ptr %e.var, i32 0, i64 1, !dbg !15
// CHECK:STDOUT:   store i32 %.loc16_26.4, ptr %.loc16_26.5.array.index, align 4, !dbg !15
// CHECK:STDOUT:   %tuple.elem2.loc16.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %d.var, i32 0, i32 2, !dbg !15
// CHECK:STDOUT:   %.loc16_26.7 = load i32, ptr %tuple.elem2.loc16.tuple.elem, align 4, !dbg !15
// CHECK:STDOUT:   %.loc16_26.8.array.index = getelementptr inbounds [3 x i32], ptr %e.var, i32 0, i64 2, !dbg !15
// CHECK:STDOUT:   store i32 %.loc16_26.7, ptr %.loc16_26.8.array.index, align 4, !dbg !15
// CHECK:STDOUT:   ret void, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder i32 1, { 0, 2, 1, 3, 4, 5, 6, 7 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "base.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !3, line: 11, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 12, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 12, column: 26, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 13, column: 26, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 13, column: 3, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 14, column: 25, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 14, column: 3, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 15, column: 28, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 15, column: 3, scope: !4)
// CHECK:STDOUT: !15 = !DILocation(line: 16, column: 26, scope: !4)
// CHECK:STDOUT: !16 = !DILocation(line: 11, column: 1, scope: !4)
