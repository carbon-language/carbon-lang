// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/tuple/value_formation.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/tuple/value_formation.carbon

fn G(ab: ((i32, i32, i32), (i32, i32, i32)));

fn F() {
  var a: (i32, i32, i32);
  var b: (i32, i32, i32);
  G((a, b));
}

// CHECK:STDOUT: ; ModuleID = 'value_formation.carbon'
// CHECK:STDOUT: source_filename = "value_formation.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca { i32, i32, i32 }, align 4, !dbg !7
// CHECK:STDOUT:   %b.var = alloca { i32, i32, i32 }, align 4, !dbg !8
// CHECK:STDOUT:   %tuple.loc18_6 = alloca { i32, i32, i32 }, align 4, !dbg !9
// CHECK:STDOUT:   %tuple.loc18_9 = alloca { i32, i32, i32 }, align 4, !dbg !10
// CHECK:STDOUT:   %tuple.loc18_10 = alloca { ptr, ptr }, align 8, !dbg !11
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !8
// CHECK:STDOUT:   %tuple.elem0.loc18_6.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 0, !dbg !9
// CHECK:STDOUT:   %.loc18_6.1 = load i32, ptr %tuple.elem0.loc18_6.tuple.elem, align 4, !dbg !9
// CHECK:STDOUT:   %tuple.elem1.loc18_6.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 1, !dbg !9
// CHECK:STDOUT:   %.loc18_6.2 = load i32, ptr %tuple.elem1.loc18_6.tuple.elem, align 4, !dbg !9
// CHECK:STDOUT:   %tuple.elem2.loc18_6.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 2, !dbg !9
// CHECK:STDOUT:   %.loc18_6.3 = load i32, ptr %tuple.elem2.loc18_6.tuple.elem, align 4, !dbg !9
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %tuple.loc18_6), !dbg !9
// CHECK:STDOUT:   %tuple.loc18_61 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc18_6, i32 0, i32 0, !dbg !9
// CHECK:STDOUT:   store i32 %.loc18_6.1, ptr %tuple.loc18_61, align 4, !dbg !9
// CHECK:STDOUT:   %tuple.loc18_62 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc18_6, i32 0, i32 1, !dbg !9
// CHECK:STDOUT:   store i32 %.loc18_6.2, ptr %tuple.loc18_62, align 4, !dbg !9
// CHECK:STDOUT:   %tuple.loc18_63 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc18_6, i32 0, i32 2, !dbg !9
// CHECK:STDOUT:   store i32 %.loc18_6.3, ptr %tuple.loc18_63, align 4, !dbg !9
// CHECK:STDOUT:   %tuple.elem0.loc18_9.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %b.var, i32 0, i32 0, !dbg !10
// CHECK:STDOUT:   %.loc18_9.1 = load i32, ptr %tuple.elem0.loc18_9.tuple.elem, align 4, !dbg !10
// CHECK:STDOUT:   %tuple.elem1.loc18_9.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %b.var, i32 0, i32 1, !dbg !10
// CHECK:STDOUT:   %.loc18_9.2 = load i32, ptr %tuple.elem1.loc18_9.tuple.elem, align 4, !dbg !10
// CHECK:STDOUT:   %tuple.elem2.loc18_9.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %b.var, i32 0, i32 2, !dbg !10
// CHECK:STDOUT:   %.loc18_9.3 = load i32, ptr %tuple.elem2.loc18_9.tuple.elem, align 4, !dbg !10
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %tuple.loc18_9), !dbg !10
// CHECK:STDOUT:   %tuple.loc18_94 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc18_9, i32 0, i32 0, !dbg !10
// CHECK:STDOUT:   store i32 %.loc18_9.1, ptr %tuple.loc18_94, align 4, !dbg !10
// CHECK:STDOUT:   %tuple.loc18_95 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc18_9, i32 0, i32 1, !dbg !10
// CHECK:STDOUT:   store i32 %.loc18_9.2, ptr %tuple.loc18_95, align 4, !dbg !10
// CHECK:STDOUT:   %tuple.loc18_96 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc18_9, i32 0, i32 2, !dbg !10
// CHECK:STDOUT:   store i32 %.loc18_9.3, ptr %tuple.loc18_96, align 4, !dbg !10
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %tuple.loc18_10), !dbg !11
// CHECK:STDOUT:   %tuple.loc18_107 = getelementptr inbounds nuw { ptr, ptr }, ptr %tuple.loc18_10, i32 0, i32 0, !dbg !11
// CHECK:STDOUT:   store ptr %tuple.loc18_6, ptr %tuple.loc18_107, align 8, !dbg !11
// CHECK:STDOUT:   %tuple.loc18_108 = getelementptr inbounds nuw { ptr, ptr }, ptr %tuple.loc18_10, i32 0, i32 1, !dbg !11
// CHECK:STDOUT:   store ptr %tuple.loc18_9, ptr %tuple.loc18_108, align 8, !dbg !11
// CHECK:STDOUT:   call void @_CG.Main(ptr %tuple.loc18_10), !dbg !12
// CHECK:STDOUT:   call void @"_COp.9eed32d146d768a4:core.Destroy.Core"(ptr %b.var), !dbg !8
// CHECK:STDOUT:   call void @"_COp.9eed32d146d768a4:core.Destroy.Core"(ptr %a.var), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.94e3ca86614aba5d:core.Destroy.Core"(ptr %self) #0 !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.9eed32d146d768a4:core.Destroy.Core"(ptr %self) #0 !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_COp.9eed32d146d768a4:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 4, 3, 2, 1, 0 }
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
// CHECK:STDOUT: !3 = !DIFile(filename: "value_formation.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !3, line: 15, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 16, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 17, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 18, column: 6, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 18, column: 9, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 18, column: 5, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 18, column: 3, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 15, column: 1, scope: !4)
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "Op", linkageName: "_COp.94e3ca86614aba5d:core.Destroy.Core", scope: null, file: !3, line: 17, type: !15, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !18)
// CHECK:STDOUT: !15 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !16 = !{null, !17}
// CHECK:STDOUT: !17 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !18 = !{!19}
// CHECK:STDOUT: !19 = !DILocalVariable(arg: 1, scope: !14, type: !17)
// CHECK:STDOUT: !20 = !DILocation(line: 17, column: 3, scope: !14)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "Op", linkageName: "_COp.9eed32d146d768a4:core.Destroy.Core", scope: null, file: !3, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !25)
// CHECK:STDOUT: !22 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !23 = !{null, !24}
// CHECK:STDOUT: !24 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !25 = !{!26}
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 1, scope: !21, type: !24)
// CHECK:STDOUT: !27 = !DILocation(line: 17, column: 3, scope: !21)
