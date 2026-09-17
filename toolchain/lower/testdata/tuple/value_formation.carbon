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

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'value_formation.carbon'
// CHECK:STDOUT: source_filename = "value_formation.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca { i32, i32, i32 }, align 4, !dbg !29
// CHECK:STDOUT:   %b.var = alloca { i32, i32, i32 }, align 4, !dbg !31
// CHECK:STDOUT:   %tuple.loc18_6 = alloca { i32, i32, i32 }, align 4, !dbg !33
// CHECK:STDOUT:   %tuple.loc18_9 = alloca { i32, i32, i32 }, align 4, !dbg !34
// CHECK:STDOUT:   %tuple.loc18_10 = alloca { ptr, ptr }, align 8, !dbg !35
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !29
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !31
// CHECK:STDOUT:   %tuple.elem0.loc18_6.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 0, !dbg !33
// CHECK:STDOUT:   %.loc18_6.1 = load i32, ptr %tuple.elem0.loc18_6.tuple.elem, align 4, !dbg !33
// CHECK:STDOUT:   %tuple.elem1.loc18_6.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 1, !dbg !33
// CHECK:STDOUT:   %.loc18_6.2 = load i32, ptr %tuple.elem1.loc18_6.tuple.elem, align 4, !dbg !33
// CHECK:STDOUT:   %tuple.elem2.loc18_6.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 2, !dbg !33
// CHECK:STDOUT:   %.loc18_6.3 = load i32, ptr %tuple.elem2.loc18_6.tuple.elem, align 4, !dbg !33
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %tuple.loc18_6), !dbg !33
// CHECK:STDOUT:   %tuple.loc18_61 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc18_6, i32 0, i32 0, !dbg !33
// CHECK:STDOUT:   store i32 %.loc18_6.1, ptr %tuple.loc18_61, align 4, !dbg !33
// CHECK:STDOUT:   %tuple.loc18_62 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc18_6, i32 0, i32 1, !dbg !33
// CHECK:STDOUT:   store i32 %.loc18_6.2, ptr %tuple.loc18_62, align 4, !dbg !33
// CHECK:STDOUT:   %tuple.loc18_63 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc18_6, i32 0, i32 2, !dbg !33
// CHECK:STDOUT:   store i32 %.loc18_6.3, ptr %tuple.loc18_63, align 4, !dbg !33
// CHECK:STDOUT:   %tuple.elem0.loc18_9.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %b.var, i32 0, i32 0, !dbg !34
// CHECK:STDOUT:   %.loc18_9.1 = load i32, ptr %tuple.elem0.loc18_9.tuple.elem, align 4, !dbg !34
// CHECK:STDOUT:   %tuple.elem1.loc18_9.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %b.var, i32 0, i32 1, !dbg !34
// CHECK:STDOUT:   %.loc18_9.2 = load i32, ptr %tuple.elem1.loc18_9.tuple.elem, align 4, !dbg !34
// CHECK:STDOUT:   %tuple.elem2.loc18_9.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %b.var, i32 0, i32 2, !dbg !34
// CHECK:STDOUT:   %.loc18_9.3 = load i32, ptr %tuple.elem2.loc18_9.tuple.elem, align 4, !dbg !34
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %tuple.loc18_9), !dbg !34
// CHECK:STDOUT:   %tuple.loc18_94 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc18_9, i32 0, i32 0, !dbg !34
// CHECK:STDOUT:   store i32 %.loc18_9.1, ptr %tuple.loc18_94, align 4, !dbg !34
// CHECK:STDOUT:   %tuple.loc18_95 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc18_9, i32 0, i32 1, !dbg !34
// CHECK:STDOUT:   store i32 %.loc18_9.2, ptr %tuple.loc18_95, align 4, !dbg !34
// CHECK:STDOUT:   %tuple.loc18_96 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc18_9, i32 0, i32 2, !dbg !34
// CHECK:STDOUT:   store i32 %.loc18_9.3, ptr %tuple.loc18_96, align 4, !dbg !34
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %tuple.loc18_10), !dbg !35
// CHECK:STDOUT:   %tuple.loc18_107 = getelementptr inbounds nuw { ptr, ptr }, ptr %tuple.loc18_10, i32 0, i32 0, !dbg !35
// CHECK:STDOUT:   store ptr %tuple.loc18_6, ptr %tuple.loc18_107, align 8, !dbg !35
// CHECK:STDOUT:   %tuple.loc18_108 = getelementptr inbounds nuw { ptr, ptr }, ptr %tuple.loc18_10, i32 0, i32 1, !dbg !35
// CHECK:STDOUT:   store ptr %tuple.loc18_9, ptr %tuple.loc18_108, align 8, !dbg !35
// CHECK:STDOUT:   call void @_CG.Main(ptr %tuple.loc18_10), !dbg !36
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.dd851822db46e4f5(ptr %b.var), !dbg !31
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.dd851822db46e4f5(ptr %a.var), !dbg !29
// CHECK:STDOUT:   ret void, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.555b1314256372b8:core.Destroy.Core"(ptr %self) #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.dd851822db46e4f5(ptr %self) #0 !dbg !46 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.555b1314256372b8:core.Destroy.Core"(ptr %self), !dbg !50
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.dd851822db46e4f5, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !18 = !DIFile(filename: "value_formation.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{null, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !{null}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{null, !20}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !18, line: 15, type: !25, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !29 = !DILocation(line: 16, column: 3, scope: !28)
// CHECK:STDOUT: !31 = !DILocation(line: 17, column: 3, scope: !28)
// CHECK:STDOUT: !33 = !DILocation(line: 18, column: 6, scope: !28)
// CHECK:STDOUT: !34 = !DILocation(line: 18, column: 9, scope: !28)
// CHECK:STDOUT: !35 = !DILocation(line: 18, column: 5, scope: !28)
// CHECK:STDOUT: !36 = !DILocation(line: 18, column: 3, scope: !28)
// CHECK:STDOUT: !37 = !DILocation(line: 15, column: 1, scope: !28)
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !18, line: 17, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !41)
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !38, type: !20)
// CHECK:STDOUT: !40 = !DILocation(line: 17, column: 3, scope: !38)
// CHECK:STDOUT: !41 = !{!39}
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.555b1314256372b8:core.Destroy.Core", scope: null, file: !18, line: 17, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !45)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !42, type: !21)
// CHECK:STDOUT: !44 = !DILocation(line: 17, column: 3, scope: !42)
// CHECK:STDOUT: !45 = !{!43}
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.dd851822db46e4f5", scope: null, file: !14, line: 24, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !52)
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !46, type: !21)
// CHECK:STDOUT: !50 = !DILocation(line: 26, column: 5, scope: !46)
// CHECK:STDOUT: !51 = !DILocation(line: 24, column: 3, scope: !46)
// CHECK:STDOUT: !52 = !{!47}
// CHECK:STDOUT:
