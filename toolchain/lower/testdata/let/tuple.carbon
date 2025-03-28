// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// CHECK:STDOUT: @tuple.ee6.loc12_3.2 = internal constant { i32, i32, i32 } { i32 1, i32 2, i32 3 }
// CHECK:STDOUT: @tuple.0a0.loc13_3.2 = internal constant { i32, i32 } { i32 4, i32 5 }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CF.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca { i32, i32, i32 }, align 8, !dbg !7
// CHECK:STDOUT:   %b.var = alloca { i32, i32 }, align 8, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 12, ptr %a.var), !dbg !7
// CHECK:STDOUT:   %tuple.elem0.loc12.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 0, !dbg !8
// CHECK:STDOUT:   %tuple.elem1.loc12.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 1, !dbg !8
// CHECK:STDOUT:   %tuple.elem2.loc12.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 2, !dbg !8
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %a.var, ptr align 4 @tuple.ee6.loc12_3.2, i64 12, i1 false), !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %b.var), !dbg !7
// CHECK:STDOUT:   %tuple.elem0.loc13.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %b.var, i32 0, i32 0, !dbg !9
// CHECK:STDOUT:   %tuple.elem1.loc13.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %b.var, i32 0, i32 1, !dbg !9
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %b.var, ptr align 4 @tuple.0a0.loc13_3.2, i64 8, i1 false), !dbg !10
// CHECK:STDOUT:   %tuple.elem0.loc14_43.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 0, !dbg !11
// CHECK:STDOUT:   %.loc14_43.1 = load i32, ptr %tuple.elem0.loc14_43.tuple.elem, align 4, !dbg !11
// CHECK:STDOUT:   %tuple.elem1.loc14_43.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 1, !dbg !11
// CHECK:STDOUT:   %.loc14_43.2 = load i32, ptr %tuple.elem1.loc14_43.tuple.elem, align 4, !dbg !11
// CHECK:STDOUT:   %tuple.elem2.loc14.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 2, !dbg !11
// CHECK:STDOUT:   %.loc14_43.3 = load i32, ptr %tuple.elem2.loc14.tuple.elem, align 4, !dbg !11
// CHECK:STDOUT:   %tuple.loc14_43 = alloca { i32, i32, i32 }, align 8, !dbg !11
// CHECK:STDOUT:   %tuple.loc14_431 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc14_43, i32 0, i32 0, !dbg !11
// CHECK:STDOUT:   store i32 %.loc14_43.1, ptr %tuple.loc14_431, align 4, !dbg !11
// CHECK:STDOUT:   %tuple.loc14_432 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc14_43, i32 0, i32 1, !dbg !11
// CHECK:STDOUT:   store i32 %.loc14_43.2, ptr %tuple.loc14_432, align 4, !dbg !11
// CHECK:STDOUT:   %tuple.loc14_433 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc14_43, i32 0, i32 2, !dbg !11
// CHECK:STDOUT:   store i32 %.loc14_43.3, ptr %tuple.loc14_433, align 4, !dbg !11
// CHECK:STDOUT:   %tuple.elem0.loc14_46.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %b.var, i32 0, i32 0, !dbg !12
// CHECK:STDOUT:   %.loc14_46.1 = load i32, ptr %tuple.elem0.loc14_46.tuple.elem, align 4, !dbg !12
// CHECK:STDOUT:   %tuple.elem1.loc14_46.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %b.var, i32 0, i32 1, !dbg !12
// CHECK:STDOUT:   %.loc14_46.2 = load i32, ptr %tuple.elem1.loc14_46.tuple.elem, align 4, !dbg !12
// CHECK:STDOUT:   %tuple.loc14_46 = alloca { i32, i32 }, align 8, !dbg !12
// CHECK:STDOUT:   %tuple.loc14_464 = getelementptr inbounds nuw { i32, i32 }, ptr %tuple.loc14_46, i32 0, i32 0, !dbg !12
// CHECK:STDOUT:   store i32 %.loc14_46.1, ptr %tuple.loc14_464, align 4, !dbg !12
// CHECK:STDOUT:   %tuple.loc14_465 = getelementptr inbounds nuw { i32, i32 }, ptr %tuple.loc14_46, i32 0, i32 1, !dbg !12
// CHECK:STDOUT:   store i32 %.loc14_46.2, ptr %tuple.loc14_465, align 4, !dbg !12
// CHECK:STDOUT:   %tuple.loc14_47 = alloca { ptr, ptr }, align 8, !dbg !13
// CHECK:STDOUT:   %tuple.loc14_476 = getelementptr inbounds nuw { ptr, ptr }, ptr %tuple.loc14_47, i32 0, i32 0, !dbg !13
// CHECK:STDOUT:   store ptr %tuple.loc14_43, ptr %tuple.loc14_476, align 8, !dbg !13
// CHECK:STDOUT:   %tuple.loc14_477 = getelementptr inbounds nuw { ptr, ptr }, ptr %tuple.loc14_47, i32 0, i32 1, !dbg !13
// CHECK:STDOUT:   store ptr %tuple.loc14_46, ptr %tuple.loc14_477, align 8, !dbg !13
// CHECK:STDOUT:   %tuple.elem1.loc15_11.tuple.elem = getelementptr inbounds nuw { ptr, ptr }, ptr %tuple.loc14_47, i32 0, i32 1, !dbg !14
// CHECK:STDOUT:   %tuple.elem1.loc15_11.tuple.elem.load = load ptr, ptr %tuple.elem1.loc15_11.tuple.elem, align 8, !dbg !14
// CHECK:STDOUT:   %tuple.elem1.loc15_13.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %tuple.elem1.loc15_11.tuple.elem.load, i32 0, i32 1, !dbg !14
// CHECK:STDOUT:   %tuple.elem1.loc15_13.tuple.elem.load = load i32, ptr %tuple.elem1.loc15_13.tuple.elem, align 4, !dbg !14
// CHECK:STDOUT:   ret i32 %tuple.elem1.loc15_13.tuple.elem.load, !dbg !15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder i32 1, { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 11, 13, 14 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
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
// CHECK:STDOUT: !3 = !DIFile(filename: "tuple.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !3, line: 11, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 12, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 12, column: 28, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 13, column: 23, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 13, column: 3, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 14, column: 43, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 14, column: 46, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 14, column: 42, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 15, column: 10, scope: !4)
// CHECK:STDOUT: !15 = !DILocation(line: 15, column: 3, scope: !4)
