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
// CHECK:STDOUT: @tuple.fbf.loc14_3.1 = internal constant { i32, i32 } { i32 12, i32 7 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.var = alloca { i32, i32 }, align 4, !dbg !29
// CHECK:STDOUT:   %_.var = alloca { i32, i32 }, align 4, !dbg !32
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %x.var), !dbg !29
// CHECK:STDOUT:   %tuple.elem0.loc14.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %x.var, i32 0, i32 0, !dbg !30
// CHECK:STDOUT:   %tuple.elem1.loc14.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %x.var, i32 0, i32 1, !dbg !30
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %x.var, ptr align 4 @tuple.fbf.loc14_3.1, i64 8, i1 false), !dbg !29
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !32
// CHECK:STDOUT:   %tuple.elem0.loc15_23.1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %x.var, i32 0, i32 0, !dbg !33
// CHECK:STDOUT:   %.loc15_23.1 = load i32, ptr %tuple.elem0.loc15_23.1.tuple.elem, align 4, !dbg !33
// CHECK:STDOUT:   %tuple.elem0.loc15_23.2.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %_.var, i32 0, i32 0, !dbg !33
// CHECK:STDOUT:   store i32 %.loc15_23.1, ptr %tuple.elem0.loc15_23.2.tuple.elem, align 4, !dbg !33
// CHECK:STDOUT:   %tuple.elem1.loc15_23.1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %x.var, i32 0, i32 1, !dbg !33
// CHECK:STDOUT:   %.loc15_23.3 = load i32, ptr %tuple.elem1.loc15_23.1.tuple.elem, align 4, !dbg !33
// CHECK:STDOUT:   %tuple.elem1.loc15_23.2.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %_.var, i32 0, i32 1, !dbg !33
// CHECK:STDOUT:   store i32 %.loc15_23.3, ptr %tuple.elem1.loc15_23.2.tuple.elem, align 4, !dbg !33
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ffbc82e0019b7bf2(ptr %_.var), !dbg !32
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ffbc82e0019b7bf2(ptr %x.var), !dbg !29
// CHECK:STDOUT:   ret i32 0, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.35f6012581c5aabb:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !38
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %tuple.elem1.tuple.elem), !dbg !38
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !38
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %tuple.elem0.tuple.elem), !dbg !38
// CHECK:STDOUT:   ret void, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.ffbc82e0019b7bf2(ptr %t) #0 !dbg !42 {
// CHECK:STDOUT:   call void @"_COp.35f6012581c5aabb:core.SubobjectDestroy.Core"(ptr %t), !dbg !46
// CHECK:STDOUT:   ret void, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f127f30143696d51(ptr %t) #0 !dbg !49 {
// CHECK:STDOUT:   ret void, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.ffbc82e0019b7bf2, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.f127f30143696d51, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !18 = !DIFile(filename: "two_entries.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{!20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{null, !23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !18, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !29 = !DILocation(line: 14, column: 3, scope: !26)
// CHECK:STDOUT: !30 = !DILocation(line: 14, column: 23, scope: !26)
// CHECK:STDOUT: !32 = !DILocation(line: 15, column: 3, scope: !26)
// CHECK:STDOUT: !33 = !DILocation(line: 15, column: 23, scope: !26)
// CHECK:STDOUT: !35 = !DILocation(line: 16, column: 3, scope: !26)
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "Op", linkageName: "_COp.35f6012581c5aabb:core.SubobjectDestroy.Core", scope: null, file: !18, line: 15, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !41)
// CHECK:STDOUT: !37 = !DILocalVariable(arg: 1, scope: !36, type: !23)
// CHECK:STDOUT: !38 = !DILocation(line: 15, column: 3, scope: !36)
// CHECK:STDOUT: !39 = !{null, !20}
// CHECK:STDOUT: !40 = !DISubroutineType(types: !39)
// CHECK:STDOUT: !41 = !{!37}
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.ffbc82e0019b7bf2", scope: null, file: !10, line: 32, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !48)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !42, type: !23)
// CHECK:STDOUT: !46 = !DILocation(line: 34, column: 5, scope: !42)
// CHECK:STDOUT: !47 = !DILocation(line: 32, column: 1, scope: !42)
// CHECK:STDOUT: !48 = !{!43}
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f127f30143696d51", scope: null, file: !10, line: 32, type: !40, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !55)
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 1, scope: !49, type: !20)
// CHECK:STDOUT: !54 = !DILocation(line: 32, column: 1, scope: !49)
// CHECK:STDOUT: !55 = !{!50}
// CHECK:STDOUT:
