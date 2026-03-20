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
// CHECK:STDOUT:   %a.var = alloca { i32, i32, i32 }, align 8, !dbg !7
// CHECK:STDOUT:   %b.var = alloca { i32, i32, i32 }, align 8, !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !8
// CHECK:STDOUT:   %tuple.elem0.loc55.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 0, !dbg !9
// CHECK:STDOUT:   %.loc55.1 = load i32, ptr %tuple.elem0.loc55.tuple.elem, align 4, !dbg !9
// CHECK:STDOUT:   %tuple.elem1.loc55.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 1, !dbg !9
// CHECK:STDOUT:   %.loc55.2 = load i32, ptr %tuple.elem1.loc55.tuple.elem, align 4, !dbg !9
// CHECK:STDOUT:   %tuple.elem2.loc55.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 2, !dbg !9
// CHECK:STDOUT:   %.loc55.3 = load i32, ptr %tuple.elem2.loc55.tuple.elem, align 4, !dbg !9
// CHECK:STDOUT:   %tuple.loc55 = alloca { i32, i32, i32 }, align 8, !dbg !9
// CHECK:STDOUT:   %tuple.loc551 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc55, i32 0, i32 0, !dbg !9
// CHECK:STDOUT:   store i32 %.loc55.1, ptr %tuple.loc551, align 4, !dbg !9
// CHECK:STDOUT:   %tuple.loc552 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc55, i32 0, i32 1, !dbg !9
// CHECK:STDOUT:   store i32 %.loc55.2, ptr %tuple.loc552, align 4, !dbg !9
// CHECK:STDOUT:   %tuple.loc553 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc55, i32 0, i32 2, !dbg !9
// CHECK:STDOUT:   store i32 %.loc55.3, ptr %tuple.loc553, align 4, !dbg !9
// CHECK:STDOUT:   %tuple.elem0.loc57.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %b.var, i32 0, i32 0, !dbg !10
// CHECK:STDOUT:   %.loc57.1 = load i32, ptr %tuple.elem0.loc57.tuple.elem, align 4, !dbg !10
// CHECK:STDOUT:   %tuple.elem1.loc57.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %b.var, i32 0, i32 1, !dbg !10
// CHECK:STDOUT:   %.loc57.2 = load i32, ptr %tuple.elem1.loc57.tuple.elem, align 4, !dbg !10
// CHECK:STDOUT:   %tuple.elem2.loc57.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %b.var, i32 0, i32 2, !dbg !10
// CHECK:STDOUT:   %.loc57.3 = load i32, ptr %tuple.elem2.loc57.tuple.elem, align 4, !dbg !10
// CHECK:STDOUT:   %tuple.loc57 = alloca { i32, i32, i32 }, align 8, !dbg !10
// CHECK:STDOUT:   %tuple.loc574 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc57, i32 0, i32 0, !dbg !10
// CHECK:STDOUT:   store i32 %.loc57.1, ptr %tuple.loc574, align 4, !dbg !10
// CHECK:STDOUT:   %tuple.loc575 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc57, i32 0, i32 1, !dbg !10
// CHECK:STDOUT:   store i32 %.loc57.2, ptr %tuple.loc575, align 4, !dbg !10
// CHECK:STDOUT:   %tuple.loc576 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc57, i32 0, i32 2, !dbg !10
// CHECK:STDOUT:   store i32 %.loc57.3, ptr %tuple.loc576, align 4, !dbg !10
// CHECK:STDOUT:   %tuple.loc58 = alloca { ptr, ptr }, align 8, !dbg !11
// CHECK:STDOUT:   %tuple.loc587 = getelementptr inbounds nuw { ptr, ptr }, ptr %tuple.loc58, i32 0, i32 0, !dbg !11
// CHECK:STDOUT:   store ptr %tuple.loc55, ptr %tuple.loc587, align 8, !dbg !11
// CHECK:STDOUT:   %tuple.loc588 = getelementptr inbounds nuw { ptr, ptr }, ptr %tuple.loc58, i32 0, i32 1, !dbg !11
// CHECK:STDOUT:   store ptr %tuple.loc57, ptr %tuple.loc588, align 8, !dbg !11
// CHECK:STDOUT:   call void @_CG.Main(ptr %tuple.loc58), !dbg !12
// CHECK:STDOUT:   ret void, !dbg !13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
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
