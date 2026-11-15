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

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'tuple.carbon'
// CHECK:STDOUT: source_filename = "tuple.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @tuple.80e.loc14_3 = internal constant { i32, i32, i32 } { i32 1, i32 2, i32 3 }
// CHECK:STDOUT: @tuple.a84.loc15_3 = internal constant { i32, i32 } { i32 4, i32 5 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CF.Main() #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca { i32, i32, i32 }, align 4, !dbg !28
// CHECK:STDOUT:   %b.var = alloca { i32, i32 }, align 4, !dbg !31
// CHECK:STDOUT:   %tuple.loc16_43 = alloca { i32, i32, i32 }, align 4, !dbg !34
// CHECK:STDOUT:   %tuple.loc16_46 = alloca { i32, i32 }, align 4, !dbg !35
// CHECK:STDOUT:   %tuple.loc16_47 = alloca { ptr, ptr }, align 8, !dbg !36
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !28
// CHECK:STDOUT:   %tuple.elem0.loc14.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 0, !dbg !29
// CHECK:STDOUT:   %tuple.elem1.loc14.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 1, !dbg !29
// CHECK:STDOUT:   %tuple.elem2.loc14.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 2, !dbg !29
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %a.var, ptr align 4 @tuple.80e.loc14_3, i64 12, i1 false), !dbg !28
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !31
// CHECK:STDOUT:   %tuple.elem0.loc15.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %b.var, i32 0, i32 0, !dbg !32
// CHECK:STDOUT:   %tuple.elem1.loc15.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %b.var, i32 0, i32 1, !dbg !32
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %b.var, ptr align 4 @tuple.a84.loc15_3, i64 8, i1 false), !dbg !31
// CHECK:STDOUT:   %tuple.elem0.loc16_43.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 0, !dbg !34
// CHECK:STDOUT:   %.loc16_43.1 = load i32, ptr %tuple.elem0.loc16_43.tuple.elem, align 4, !dbg !34
// CHECK:STDOUT:   %tuple.elem1.loc16_43.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 1, !dbg !34
// CHECK:STDOUT:   %.loc16_43.2 = load i32, ptr %tuple.elem1.loc16_43.tuple.elem, align 4, !dbg !34
// CHECK:STDOUT:   %tuple.elem2.loc16.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 2, !dbg !34
// CHECK:STDOUT:   %.loc16_43.3 = load i32, ptr %tuple.elem2.loc16.tuple.elem, align 4, !dbg !34
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %tuple.loc16_43), !dbg !34
// CHECK:STDOUT:   %tuple.loc16_431 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc16_43, i32 0, i32 0, !dbg !34
// CHECK:STDOUT:   store i32 %.loc16_43.1, ptr %tuple.loc16_431, align 4, !dbg !34
// CHECK:STDOUT:   %tuple.loc16_432 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc16_43, i32 0, i32 1, !dbg !34
// CHECK:STDOUT:   store i32 %.loc16_43.2, ptr %tuple.loc16_432, align 4, !dbg !34
// CHECK:STDOUT:   %tuple.loc16_433 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc16_43, i32 0, i32 2, !dbg !34
// CHECK:STDOUT:   store i32 %.loc16_43.3, ptr %tuple.loc16_433, align 4, !dbg !34
// CHECK:STDOUT:   %tuple.elem0.loc16_46.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %b.var, i32 0, i32 0, !dbg !35
// CHECK:STDOUT:   %.loc16_46.1 = load i32, ptr %tuple.elem0.loc16_46.tuple.elem, align 4, !dbg !35
// CHECK:STDOUT:   %tuple.elem1.loc16_46.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %b.var, i32 0, i32 1, !dbg !35
// CHECK:STDOUT:   %.loc16_46.2 = load i32, ptr %tuple.elem1.loc16_46.tuple.elem, align 4, !dbg !35
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %tuple.loc16_46), !dbg !35
// CHECK:STDOUT:   %tuple.loc16_464 = getelementptr inbounds nuw { i32, i32 }, ptr %tuple.loc16_46, i32 0, i32 0, !dbg !35
// CHECK:STDOUT:   store i32 %.loc16_46.1, ptr %tuple.loc16_464, align 4, !dbg !35
// CHECK:STDOUT:   %tuple.loc16_465 = getelementptr inbounds nuw { i32, i32 }, ptr %tuple.loc16_46, i32 0, i32 1, !dbg !35
// CHECK:STDOUT:   store i32 %.loc16_46.2, ptr %tuple.loc16_465, align 4, !dbg !35
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %tuple.loc16_47), !dbg !36
// CHECK:STDOUT:   %tuple.loc16_476 = getelementptr inbounds nuw { ptr, ptr }, ptr %tuple.loc16_47, i32 0, i32 0, !dbg !36
// CHECK:STDOUT:   store ptr %tuple.loc16_43, ptr %tuple.loc16_476, align 8, !dbg !36
// CHECK:STDOUT:   %tuple.loc16_477 = getelementptr inbounds nuw { ptr, ptr }, ptr %tuple.loc16_47, i32 0, i32 1, !dbg !36
// CHECK:STDOUT:   store ptr %tuple.loc16_46, ptr %tuple.loc16_477, align 8, !dbg !36
// CHECK:STDOUT:   %tuple.elem1.loc17_11.tuple.elem = getelementptr inbounds nuw { ptr, ptr }, ptr %tuple.loc16_47, i32 0, i32 1, !dbg !39
// CHECK:STDOUT:   %tuple.elem1.loc17_11.tuple.elem.load = load ptr, ptr %tuple.elem1.loc17_11.tuple.elem, align 8, !dbg !39
// CHECK:STDOUT:   %tuple.elem1.loc17_13.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %tuple.elem1.loc17_11.tuple.elem.load, i32 0, i32 1, !dbg !39
// CHECK:STDOUT:   %tuple.elem1.loc17_13.tuple.elem.load = load i32, ptr %tuple.elem1.loc17_13.tuple.elem, align 4, !dbg !39
// CHECK:STDOUT:   ret i32 %tuple.elem1.loc17_13.tuple.elem.load, !dbg !40
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
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIFile(filename: "tuple.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !23 = !{!22}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !20, line: 13, type: !24, spFlags: DISPFlagDefinition, unit: !21)
// CHECK:STDOUT: !28 = !DILocation(line: 14, column: 3, scope: !25)
// CHECK:STDOUT: !29 = !DILocation(line: 14, column: 28, scope: !25)
// CHECK:STDOUT: !31 = !DILocation(line: 15, column: 3, scope: !25)
// CHECK:STDOUT: !32 = !DILocation(line: 15, column: 23, scope: !25)
// CHECK:STDOUT: !34 = !DILocation(line: 16, column: 43, scope: !25)
// CHECK:STDOUT: !35 = !DILocation(line: 16, column: 46, scope: !25)
// CHECK:STDOUT: !36 = !DILocation(line: 16, column: 42, scope: !25)
// CHECK:STDOUT: !39 = !DILocation(line: 17, column: 10, scope: !25)
// CHECK:STDOUT: !40 = !DILocation(line: 17, column: 3, scope: !25)
// CHECK:STDOUT:
