// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/array/base.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/array/base.carbon

fn Run() {
  var _: array(i32, 1) = (1,);
  var _: array(f64, 2) = (11.1, 2.2,);
  var _: array((), 5) = ((), (), (), (), (),);
  var d: (i32, i32, i32) = (1, 2, 3);
  var _: array(i32, 3) = d;
}

// CHECK:STDOUT: ; ModuleID = 'base.carbon'
// CHECK:STDOUT: source_filename = "base.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @array.765.loc14_3 = internal constant [1 x i32] [i32 1]
// CHECK:STDOUT: @array.c9a.loc15_3 = internal constant [2 x double] [double 1.110000e+01, double 2.200000e+00]
// CHECK:STDOUT: @array.1cb.loc16_3 = internal constant [5 x {}] zeroinitializer
// CHECK:STDOUT: @tuple.8d2.loc17_3 = internal constant { i32, i32, i32 } { i32 1, i32 2, i32 3 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var.loc14 = alloca [1 x i32], align 4, !dbg !8
// CHECK:STDOUT:   %_.var.loc15 = alloca [2 x double], align 8, !dbg !9
// CHECK:STDOUT:   %_.var.loc16 = alloca [5 x {}], align 8, !dbg !10
// CHECK:STDOUT:   %d.var = alloca { i32, i32, i32 }, align 8, !dbg !11
// CHECK:STDOUT:   %_.var.loc18 = alloca [3 x i32], align 4, !dbg !12
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc14), !dbg !8
// CHECK:STDOUT:   %.loc14_29.3.array.index = getelementptr inbounds [1 x i32], ptr %_.var.loc14, i32 0, i64 0, !dbg !13
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %_.var.loc14, ptr align 4 @array.765.loc14_3, i64 4, i1 false), !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc15), !dbg !9
// CHECK:STDOUT:   %.loc15_37.3.array.index = getelementptr inbounds [2 x double], ptr %_.var.loc15, i32 0, i64 0, !dbg !14
// CHECK:STDOUT:   %.loc15_37.6.array.index = getelementptr inbounds [2 x double], ptr %_.var.loc15, i32 0, i64 1, !dbg !14
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %_.var.loc15, ptr align 8 @array.c9a.loc15_3, i64 16, i1 false), !dbg !9
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc16), !dbg !10
// CHECK:STDOUT:   %.loc16_45.2.array.index = getelementptr inbounds [5 x {}], ptr %_.var.loc16, i32 0, i64 0, !dbg !15
// CHECK:STDOUT:   %.loc16_45.5.array.index = getelementptr inbounds [5 x {}], ptr %_.var.loc16, i32 0, i64 1, !dbg !15
// CHECK:STDOUT:   %.loc16_45.8.array.index = getelementptr inbounds [5 x {}], ptr %_.var.loc16, i32 0, i64 2, !dbg !15
// CHECK:STDOUT:   %.loc16_45.11.array.index = getelementptr inbounds [5 x {}], ptr %_.var.loc16, i32 0, i64 3, !dbg !15
// CHECK:STDOUT:   %.loc16_45.14.array.index = getelementptr inbounds [5 x {}], ptr %_.var.loc16, i32 0, i64 4, !dbg !15
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %_.var.loc16, ptr align 1 @array.1cb.loc16_3, i64 0, i1 false), !dbg !10
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %d.var), !dbg !11
// CHECK:STDOUT:   %tuple.elem0.loc17.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %d.var, i32 0, i32 0, !dbg !16
// CHECK:STDOUT:   %tuple.elem1.loc17.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %d.var, i32 0, i32 1, !dbg !16
// CHECK:STDOUT:   %tuple.elem2.loc17.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %d.var, i32 0, i32 2, !dbg !16
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %d.var, ptr align 4 @tuple.8d2.loc17_3, i64 12, i1 false), !dbg !11
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc18), !dbg !12
// CHECK:STDOUT:   %tuple.elem0.loc18.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %d.var, i32 0, i32 0, !dbg !17
// CHECK:STDOUT:   %.loc18_26.1 = load i32, ptr %tuple.elem0.loc18.tuple.elem, align 4, !dbg !17
// CHECK:STDOUT:   %.loc18_26.2.array.index = getelementptr inbounds [3 x i32], ptr %_.var.loc18, i32 0, i64 0, !dbg !17
// CHECK:STDOUT:   store i32 %.loc18_26.1, ptr %.loc18_26.2.array.index, align 4, !dbg !17
// CHECK:STDOUT:   %tuple.elem1.loc18.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %d.var, i32 0, i32 1, !dbg !17
// CHECK:STDOUT:   %.loc18_26.4 = load i32, ptr %tuple.elem1.loc18.tuple.elem, align 4, !dbg !17
// CHECK:STDOUT:   %.loc18_26.5.array.index = getelementptr inbounds [3 x i32], ptr %_.var.loc18, i32 0, i64 1, !dbg !17
// CHECK:STDOUT:   store i32 %.loc18_26.4, ptr %.loc18_26.5.array.index, align 4, !dbg !17
// CHECK:STDOUT:   %tuple.elem2.loc18.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %d.var, i32 0, i32 2, !dbg !17
// CHECK:STDOUT:   %.loc18_26.7 = load i32, ptr %tuple.elem2.loc18.tuple.elem, align 4, !dbg !17
// CHECK:STDOUT:   %.loc18_26.8.array.index = getelementptr inbounds [3 x i32], ptr %_.var.loc18, i32 0, i64 2, !dbg !17
// CHECK:STDOUT:   store i32 %.loc18_26.7, ptr %.loc18_26.8.array.index, align 4, !dbg !17
// CHECK:STDOUT:   call void @"_COp.1c95d180afb5f529:core.Destroy.Core"(ptr %_.var.loc18), !dbg !12
// CHECK:STDOUT:   call void @"_COp.9eed32d146d768a4:core.Destroy.Core"(ptr %d.var), !dbg !11
// CHECK:STDOUT:   call void @"_COp.fd2b1e64e898d326:core.Destroy.Core"(ptr %_.var.loc16), !dbg !10
// CHECK:STDOUT:   call void @"_COp.3f75afeba3821ad2:core.Destroy.Core"(ptr %_.var.loc15), !dbg !9
// CHECK:STDOUT:   call void @"_COp.686b145eea938d24:core.Destroy.Core"(ptr %_.var.loc14), !dbg !8
// CHECK:STDOUT:   ret i32 0, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.94e3ca86614aba5d:core.Destroy.Core"(ptr %self) #0 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.1c95d180afb5f529:core.Destroy.Core"(ptr %self) #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.9eed32d146d768a4:core.Destroy.Core"(ptr %self) #0 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.fd2b1e64e898d326:core.Destroy.Core"(ptr %self) #0 !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.b46d7363edf385cc:core.Destroy.Core"(ptr %self) #0 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.3f75afeba3821ad2:core.Destroy.Core"(ptr %self) #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.686b145eea938d24:core.Destroy.Core"(ptr %self) #0 !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "base.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !3, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7}
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !DILocation(line: 14, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 15, column: 3, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 16, column: 3, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 17, column: 3, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 18, column: 3, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 14, column: 26, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 15, column: 26, scope: !4)
// CHECK:STDOUT: !15 = !DILocation(line: 16, column: 25, scope: !4)
// CHECK:STDOUT: !16 = !DILocation(line: 17, column: 28, scope: !4)
// CHECK:STDOUT: !17 = !DILocation(line: 18, column: 26, scope: !4)
// CHECK:STDOUT: !18 = !DILocation(line: 13, column: 1, scope: !4)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "Op", linkageName: "_COp.94e3ca86614aba5d:core.Destroy.Core", scope: null, file: !3, line: 18, type: !20, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !22)
// CHECK:STDOUT: !20 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !21 = !{null, !7}
// CHECK:STDOUT: !22 = !{!23}
// CHECK:STDOUT: !23 = !DILocalVariable(arg: 1, scope: !19, type: !7)
// CHECK:STDOUT: !24 = !DILocation(line: 18, column: 3, scope: !19)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "Op", linkageName: "_COp.1c95d180afb5f529:core.Destroy.Core", scope: null, file: !3, line: 18, type: !26, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !29)
// CHECK:STDOUT: !26 = !DISubroutineType(types: !27)
// CHECK:STDOUT: !27 = !{null, !28}
// CHECK:STDOUT: !28 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !29 = !{!30}
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 1, scope: !25, type: !28)
// CHECK:STDOUT: !31 = !DILocation(line: 18, column: 3, scope: !25)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "Op", linkageName: "_COp.9eed32d146d768a4:core.Destroy.Core", scope: null, file: !3, line: 17, type: !26, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !33)
// CHECK:STDOUT: !33 = !{!34}
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 1, scope: !32, type: !28)
// CHECK:STDOUT: !35 = !DILocation(line: 17, column: 3, scope: !32)
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "Op", linkageName: "_COp.fd2b1e64e898d326:core.Destroy.Core", scope: null, file: !3, line: 16, type: !26, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !37)
// CHECK:STDOUT: !37 = !{!38}
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !36, type: !28)
// CHECK:STDOUT: !39 = !DILocation(line: 16, column: 3, scope: !36)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "Op", linkageName: "_COp.b46d7363edf385cc:core.Destroy.Core", scope: null, file: !3, line: 15, type: !26, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !41)
// CHECK:STDOUT: !41 = !{!42}
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 1, scope: !40, type: !28)
// CHECK:STDOUT: !43 = !DILocation(line: 15, column: 3, scope: !40)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "Op", linkageName: "_COp.3f75afeba3821ad2:core.Destroy.Core", scope: null, file: !3, line: 15, type: !26, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !45)
// CHECK:STDOUT: !45 = !{!46}
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 1, scope: !44, type: !28)
// CHECK:STDOUT: !47 = !DILocation(line: 15, column: 3, scope: !44)
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "Op", linkageName: "_COp.686b145eea938d24:core.Destroy.Core", scope: null, file: !3, line: 14, type: !26, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !49)
// CHECK:STDOUT: !49 = !{!50}
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 1, scope: !48, type: !28)
// CHECK:STDOUT: !51 = !DILocation(line: 14, column: 3, scope: !48)
