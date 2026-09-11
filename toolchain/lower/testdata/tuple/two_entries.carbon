// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/tuple/two_entries.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/tuple/two_entries.carbon

fn Run() -> i32 {
  var x: (i32, i32) = (12, 7);
  var _: (i32, i32) = x;
  return 0;
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'two_entries.carbon'
// CHECK:STDOUT: source_filename = "two_entries.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @tuple.328.loc14_3.1 = internal constant { i32, i32 } { i32 12, i32 7 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca { i32, i32 }, align 4, !dbg !8
// CHECK:STDOUT:   %_.var = alloca { i32, i32 }, align 4, !dbg !9
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %x.var), !dbg !8
// CHECK:STDOUT:   %tuple.elem0.loc14.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %x.var, i32 0, i32 0, !dbg !10
// CHECK:STDOUT:   %tuple.elem1.loc14.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %x.var, i32 0, i32 1, !dbg !10
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %x.var, ptr align 4 @tuple.328.loc14_3.1, i64 8, i1 false), !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !9
// CHECK:STDOUT:   %tuple.elem0.loc15_23.1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %x.var, i32 0, i32 0, !dbg !11
// CHECK:STDOUT:   %.loc15_23.1 = load i32, ptr %tuple.elem0.loc15_23.1.tuple.elem, align 4, !dbg !11
// CHECK:STDOUT:   %tuple.elem0.loc15_23.2.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %_.var, i32 0, i32 0, !dbg !11
// CHECK:STDOUT:   store i32 %.loc15_23.1, ptr %tuple.elem0.loc15_23.2.tuple.elem, align 4, !dbg !11
// CHECK:STDOUT:   %tuple.elem1.loc15_23.1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %x.var, i32 0, i32 1, !dbg !11
// CHECK:STDOUT:   %.loc15_23.3 = load i32, ptr %tuple.elem1.loc15_23.1.tuple.elem, align 4, !dbg !11
// CHECK:STDOUT:   %tuple.elem1.loc15_23.2.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %_.var, i32 0, i32 1, !dbg !11
// CHECK:STDOUT:   store i32 %.loc15_23.3, ptr %tuple.elem1.loc15_23.2.tuple.elem, align 4, !dbg !11
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.c6bc553020e9306c(ptr %_.var), !dbg !9
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.c6bc553020e9306c(ptr %x.var), !dbg !8
// CHECK:STDOUT:   ret i32 0, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.70db2d92ffddbef7:core.Destroy.Core"(ptr %self) #0 !dbg !13 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.7ebc50e8bd9eb2f1:core.Destroy.Core"(ptr %self) #0 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.c6bc553020e9306c(ptr %self) #0 !dbg !26 {
// CHECK:STDOUT:   call void @"_COp.7ebc50e8bd9eb2f1:core.Destroy.Core"(ptr %self), !dbg !30
// CHECK:STDOUT:   call void @_CSubobjectDestroy.Destroy.Core.c6bc553020e9306c(ptr %self), !dbg !31
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSubobjectDestroy.Destroy.Core.c6bc553020e9306c(ptr %self) #0 !dbg !33 {
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.c6bc553020e9306c, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "two_entries.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !1, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7}
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !DILocation(line: 14, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 15, column: 3, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 14, column: 23, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 15, column: 23, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 16, column: 3, scope: !4)
// CHECK:STDOUT: !13 = distinct !DISubprogram(name: "Op", linkageName: "_COp.70db2d92ffddbef7:core.Destroy.Core", scope: null, file: !1, line: 15, type: !14, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !16)
// CHECK:STDOUT: !14 = !DISubroutineType(types: !15)
// CHECK:STDOUT: !15 = !{null, !7}
// CHECK:STDOUT: !16 = !{!17}
// CHECK:STDOUT: !17 = !DILocalVariable(arg: 1, scope: !13, type: !7)
// CHECK:STDOUT: !18 = !DILocation(line: 15, column: 3, scope: !13)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "Op", linkageName: "_COp.7ebc50e8bd9eb2f1:core.Destroy.Core", scope: null, file: !1, line: 15, type: !20, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !23)
// CHECK:STDOUT: !20 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !21 = !{null, !22}
// CHECK:STDOUT: !22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !23 = !{!24}
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !19, type: !22)
// CHECK:STDOUT: !25 = !DILocation(line: 15, column: 3, scope: !19)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.c6bc553020e9306c", scope: null, file: !27, line: 24, type: !20, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !28)
// CHECK:STDOUT: !27 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !28 = !{!29}
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 1, scope: !26, type: !22)
// CHECK:STDOUT: !30 = !DILocation(line: 25, column: 5, scope: !26)
// CHECK:STDOUT: !31 = !DILocation(line: 26, column: 5, scope: !26)
// CHECK:STDOUT: !32 = !DILocation(line: 24, column: 3, scope: !26)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.Destroy.Core.c6bc553020e9306c", scope: null, file: !27, line: 17, type: !20, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !34)
// CHECK:STDOUT: !34 = !{!35}
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !33, type: !22)
// CHECK:STDOUT: !36 = !DILocation(line: 17, column: 3, scope: !33)
// CHECK:STDOUT:
