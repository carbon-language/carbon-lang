// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/let/tuple.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/let/tuple.carbon

fn F() -> i32 {
  var a: (i32, i32, i32) = (1, 2, 3);
  var b: (i32, i32) = (4, 5);
  let c: ((i32, i32, i32), (i32, i32)) = (a, b);
  return c.1.(1);
}

// CHECK:STDOUT: ; ModuleID = 'tuple.carbon'
// CHECK:STDOUT: source_filename = "tuple.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @tuple.ee6.loc8 = internal constant { i32, i32, i32 } { i32 1, i32 2, i32 3 }
// CHECK:STDOUT: @tuple.0a0.loc27 = internal constant { i32, i32 } { i32 4, i32 5 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CF.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca { i32, i32, i32 }, align 8, !dbg !8
// CHECK:STDOUT:   %b.var = alloca { i32, i32 }, align 8, !dbg !9
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !8
// CHECK:STDOUT:   %tuple.elem0.loc25.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 0, !dbg !10
// CHECK:STDOUT:   %tuple.elem1.loc25.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 1, !dbg !10
// CHECK:STDOUT:   %tuple.elem2.loc25.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 2, !dbg !10
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %a.var, ptr align 4 @tuple.ee6.loc8, i64 12, i1 false), !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !9
// CHECK:STDOUT:   %tuple.elem0.loc40.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %b.var, i32 0, i32 0, !dbg !11
// CHECK:STDOUT:   %tuple.elem1.loc40.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %b.var, i32 0, i32 1, !dbg !11
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %b.var, ptr align 4 @tuple.0a0.loc27, i64 8, i1 false), !dbg !9
// CHECK:STDOUT:   %tuple.elem0.loc62.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 0, !dbg !12
// CHECK:STDOUT:   %.loc62.1 = load i32, ptr %tuple.elem0.loc62.tuple.elem, align 4, !dbg !12
// CHECK:STDOUT:   %tuple.elem1.loc62.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 1, !dbg !12
// CHECK:STDOUT:   %.loc62.2 = load i32, ptr %tuple.elem1.loc62.tuple.elem, align 4, !dbg !12
// CHECK:STDOUT:   %tuple.elem2.loc62.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 2, !dbg !12
// CHECK:STDOUT:   %.loc62.3 = load i32, ptr %tuple.elem2.loc62.tuple.elem, align 4, !dbg !12
// CHECK:STDOUT:   %tuple.loc62 = alloca { i32, i32, i32 }, align 8, !dbg !12
// CHECK:STDOUT:   %tuple.loc621 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc62, i32 0, i32 0, !dbg !12
// CHECK:STDOUT:   store i32 %.loc62.1, ptr %tuple.loc621, align 4, !dbg !12
// CHECK:STDOUT:   %tuple.loc622 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc62, i32 0, i32 1, !dbg !12
// CHECK:STDOUT:   store i32 %.loc62.2, ptr %tuple.loc622, align 4, !dbg !12
// CHECK:STDOUT:   %tuple.loc623 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc62, i32 0, i32 2, !dbg !12
// CHECK:STDOUT:   store i32 %.loc62.3, ptr %tuple.loc623, align 4, !dbg !12
// CHECK:STDOUT:   %tuple.elem0.loc64.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %b.var, i32 0, i32 0, !dbg !13
// CHECK:STDOUT:   %.loc64.1 = load i32, ptr %tuple.elem0.loc64.tuple.elem, align 4, !dbg !13
// CHECK:STDOUT:   %tuple.elem1.loc64.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %b.var, i32 0, i32 1, !dbg !13
// CHECK:STDOUT:   %.loc64.2 = load i32, ptr %tuple.elem1.loc64.tuple.elem, align 4, !dbg !13
// CHECK:STDOUT:   %tuple.loc64 = alloca { i32, i32 }, align 8, !dbg !13
// CHECK:STDOUT:   %tuple.loc644 = getelementptr inbounds nuw { i32, i32 }, ptr %tuple.loc64, i32 0, i32 0, !dbg !13
// CHECK:STDOUT:   store i32 %.loc64.1, ptr %tuple.loc644, align 4, !dbg !13
// CHECK:STDOUT:   %tuple.loc645 = getelementptr inbounds nuw { i32, i32 }, ptr %tuple.loc64, i32 0, i32 1, !dbg !13
// CHECK:STDOUT:   store i32 %.loc64.2, ptr %tuple.loc645, align 4, !dbg !13
// CHECK:STDOUT:   %tuple.loc65 = alloca { ptr, ptr }, align 8, !dbg !14
// CHECK:STDOUT:   %tuple.loc656 = getelementptr inbounds nuw { ptr, ptr }, ptr %tuple.loc65, i32 0, i32 0, !dbg !14
// CHECK:STDOUT:   store ptr %tuple.loc62, ptr %tuple.loc656, align 8, !dbg !14
// CHECK:STDOUT:   %tuple.loc657 = getelementptr inbounds nuw { ptr, ptr }, ptr %tuple.loc65, i32 0, i32 1, !dbg !14
// CHECK:STDOUT:   store ptr %tuple.loc64, ptr %tuple.loc657, align 8, !dbg !14
// CHECK:STDOUT:   %tuple.elem1.loc69.tuple.elem = getelementptr inbounds nuw { ptr, ptr }, ptr %tuple.loc65, i32 0, i32 1, !dbg !15
// CHECK:STDOUT:   %tuple.elem1.loc69.tuple.elem.load = load ptr, ptr %tuple.elem1.loc69.tuple.elem, align 8, !dbg !15
// CHECK:STDOUT:   %tuple.elem1.loc71.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %tuple.elem1.loc69.tuple.elem.load, i32 0, i32 1, !dbg !15
// CHECK:STDOUT:   %tuple.elem1.loc71.tuple.elem.load = load i32, ptr %tuple.elem1.loc71.tuple.elem, align 4, !dbg !15
// CHECK:STDOUT:   ret i32 %tuple.elem1.loc71.tuple.elem.load, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "tuple.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !3, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7}
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !DILocation(line: 14, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 15, column: 3, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 14, column: 28, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 15, column: 23, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 16, column: 43, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 16, column: 46, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 16, column: 42, scope: !4)
// CHECK:STDOUT: !15 = !DILocation(line: 17, column: 10, scope: !4)
// CHECK:STDOUT: !16 = !DILocation(line: 17, column: 3, scope: !4)
