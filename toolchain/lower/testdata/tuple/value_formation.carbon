// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// CHECK:STDOUT: define void @_CF.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca { i32, i32, i32 }, align 8, !dbg !7
// CHECK:STDOUT:   %b.var = alloca { i32, i32, i32 }, align 8, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 12, ptr %a.var), !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 12, ptr %b.var), !dbg !7
// CHECK:STDOUT:   %tuple.elem0.loc16_6.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 0, !dbg !8
// CHECK:STDOUT:   %.loc16_6.1 = load i32, ptr %tuple.elem0.loc16_6.tuple.elem, align 4, !dbg !8
// CHECK:STDOUT:   %tuple.elem1.loc16_6.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 1, !dbg !8
// CHECK:STDOUT:   %.loc16_6.2 = load i32, ptr %tuple.elem1.loc16_6.tuple.elem, align 4, !dbg !8
// CHECK:STDOUT:   %tuple.elem2.loc16_6.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 2, !dbg !8
// CHECK:STDOUT:   %.loc16_6.3 = load i32, ptr %tuple.elem2.loc16_6.tuple.elem, align 4, !dbg !8
// CHECK:STDOUT:   %tuple.loc16_6 = alloca { i32, i32, i32 }, align 8, !dbg !8
// CHECK:STDOUT:   %tuple.loc16_61 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc16_6, i32 0, i32 0, !dbg !8
// CHECK:STDOUT:   store i32 %.loc16_6.1, ptr %tuple.loc16_61, align 4, !dbg !8
// CHECK:STDOUT:   %tuple.loc16_62 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc16_6, i32 0, i32 1, !dbg !8
// CHECK:STDOUT:   store i32 %.loc16_6.2, ptr %tuple.loc16_62, align 4, !dbg !8
// CHECK:STDOUT:   %tuple.loc16_63 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc16_6, i32 0, i32 2, !dbg !8
// CHECK:STDOUT:   store i32 %.loc16_6.3, ptr %tuple.loc16_63, align 4, !dbg !8
// CHECK:STDOUT:   %tuple.elem0.loc16_9.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %b.var, i32 0, i32 0, !dbg !9
// CHECK:STDOUT:   %.loc16_9.1 = load i32, ptr %tuple.elem0.loc16_9.tuple.elem, align 4, !dbg !9
// CHECK:STDOUT:   %tuple.elem1.loc16_9.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %b.var, i32 0, i32 1, !dbg !9
// CHECK:STDOUT:   %.loc16_9.2 = load i32, ptr %tuple.elem1.loc16_9.tuple.elem, align 4, !dbg !9
// CHECK:STDOUT:   %tuple.elem2.loc16_9.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %b.var, i32 0, i32 2, !dbg !9
// CHECK:STDOUT:   %.loc16_9.3 = load i32, ptr %tuple.elem2.loc16_9.tuple.elem, align 4, !dbg !9
// CHECK:STDOUT:   %tuple.loc16_9 = alloca { i32, i32, i32 }, align 8, !dbg !9
// CHECK:STDOUT:   %tuple.loc16_94 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc16_9, i32 0, i32 0, !dbg !9
// CHECK:STDOUT:   store i32 %.loc16_9.1, ptr %tuple.loc16_94, align 4, !dbg !9
// CHECK:STDOUT:   %tuple.loc16_95 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc16_9, i32 0, i32 1, !dbg !9
// CHECK:STDOUT:   store i32 %.loc16_9.2, ptr %tuple.loc16_95, align 4, !dbg !9
// CHECK:STDOUT:   %tuple.loc16_96 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc16_9, i32 0, i32 2, !dbg !9
// CHECK:STDOUT:   store i32 %.loc16_9.3, ptr %tuple.loc16_96, align 4, !dbg !9
// CHECK:STDOUT:   %tuple.loc16_10 = alloca { ptr, ptr }, align 8, !dbg !10
// CHECK:STDOUT:   %tuple.loc16_107 = getelementptr inbounds nuw { ptr, ptr }, ptr %tuple.loc16_10, i32 0, i32 0, !dbg !10
// CHECK:STDOUT:   store ptr %tuple.loc16_6, ptr %tuple.loc16_107, align 8, !dbg !10
// CHECK:STDOUT:   %tuple.loc16_108 = getelementptr inbounds nuw { ptr, ptr }, ptr %tuple.loc16_10, i32 0, i32 1, !dbg !10
// CHECK:STDOUT:   store ptr %tuple.loc16_9, ptr %tuple.loc16_108, align 8, !dbg !10
// CHECK:STDOUT:   call void @_CG.Main(ptr %tuple.loc16_10), !dbg !11
// CHECK:STDOUT:   ret void, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "value_formation.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !3, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 14, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 16, column: 6, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 16, column: 9, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 16, column: 5, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 16, column: 3, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 13, column: 1, scope: !4)
