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
// CHECK:STDOUT: @tuple.479.loc14_3.1 = internal constant { i32, i32, i32 } { i32 1, i32 2, i32 3 }
// CHECK:STDOUT: @tuple.8a1.loc15_3.1 = internal constant { i32, i32 } { i32 4, i32 5 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CF.Main() #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca { i32, i32, i32 }, align 4, !dbg !29
// CHECK:STDOUT:   %b.var = alloca { i32, i32 }, align 4, !dbg !32
// CHECK:STDOUT:   %tuple.loc16_43 = alloca { i32, i32, i32 }, align 4, !dbg !35
// CHECK:STDOUT:   %tuple.loc16_46 = alloca { i32, i32 }, align 4, !dbg !36
// CHECK:STDOUT:   %tuple.loc16_47 = alloca { ptr, ptr }, align 8, !dbg !37
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !29
// CHECK:STDOUT:   %tuple.elem0.loc14.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 0, !dbg !30
// CHECK:STDOUT:   %tuple.elem1.loc14.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 1, !dbg !30
// CHECK:STDOUT:   %tuple.elem2.loc14.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 2, !dbg !30
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %a.var, ptr align 4 @tuple.479.loc14_3.1, i64 12, i1 false), !dbg !29
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b.var), !dbg !32
// CHECK:STDOUT:   %tuple.elem0.loc15.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %b.var, i32 0, i32 0, !dbg !33
// CHECK:STDOUT:   %tuple.elem1.loc15.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %b.var, i32 0, i32 1, !dbg !33
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %b.var, ptr align 4 @tuple.8a1.loc15_3.1, i64 8, i1 false), !dbg !32
// CHECK:STDOUT:   %tuple.elem0.loc16_43.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 0, !dbg !35
// CHECK:STDOUT:   %.loc16_43.1 = load i32, ptr %tuple.elem0.loc16_43.tuple.elem, align 4, !dbg !35
// CHECK:STDOUT:   %tuple.elem1.loc16_43.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 1, !dbg !35
// CHECK:STDOUT:   %.loc16_43.2 = load i32, ptr %tuple.elem1.loc16_43.tuple.elem, align 4, !dbg !35
// CHECK:STDOUT:   %tuple.elem2.loc16.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 2, !dbg !35
// CHECK:STDOUT:   %.loc16_43.3 = load i32, ptr %tuple.elem2.loc16.tuple.elem, align 4, !dbg !35
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %tuple.loc16_43), !dbg !35
// CHECK:STDOUT:   %tuple.loc16_431 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc16_43, i32 0, i32 0, !dbg !35
// CHECK:STDOUT:   store i32 %.loc16_43.1, ptr %tuple.loc16_431, align 4, !dbg !35
// CHECK:STDOUT:   %tuple.loc16_432 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc16_43, i32 0, i32 1, !dbg !35
// CHECK:STDOUT:   store i32 %.loc16_43.2, ptr %tuple.loc16_432, align 4, !dbg !35
// CHECK:STDOUT:   %tuple.loc16_433 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc16_43, i32 0, i32 2, !dbg !35
// CHECK:STDOUT:   store i32 %.loc16_43.3, ptr %tuple.loc16_433, align 4, !dbg !35
// CHECK:STDOUT:   %tuple.elem0.loc16_46.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %b.var, i32 0, i32 0, !dbg !36
// CHECK:STDOUT:   %.loc16_46.1 = load i32, ptr %tuple.elem0.loc16_46.tuple.elem, align 4, !dbg !36
// CHECK:STDOUT:   %tuple.elem1.loc16_46.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %b.var, i32 0, i32 1, !dbg !36
// CHECK:STDOUT:   %.loc16_46.2 = load i32, ptr %tuple.elem1.loc16_46.tuple.elem, align 4, !dbg !36
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %tuple.loc16_46), !dbg !36
// CHECK:STDOUT:   %tuple.loc16_464 = getelementptr inbounds nuw { i32, i32 }, ptr %tuple.loc16_46, i32 0, i32 0, !dbg !36
// CHECK:STDOUT:   store i32 %.loc16_46.1, ptr %tuple.loc16_464, align 4, !dbg !36
// CHECK:STDOUT:   %tuple.loc16_465 = getelementptr inbounds nuw { i32, i32 }, ptr %tuple.loc16_46, i32 0, i32 1, !dbg !36
// CHECK:STDOUT:   store i32 %.loc16_46.2, ptr %tuple.loc16_465, align 4, !dbg !36
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %tuple.loc16_47), !dbg !37
// CHECK:STDOUT:   %tuple.loc16_476 = getelementptr inbounds nuw { ptr, ptr }, ptr %tuple.loc16_47, i32 0, i32 0, !dbg !37
// CHECK:STDOUT:   store ptr %tuple.loc16_43, ptr %tuple.loc16_476, align 8, !dbg !37
// CHECK:STDOUT:   %tuple.loc16_477 = getelementptr inbounds nuw { ptr, ptr }, ptr %tuple.loc16_47, i32 0, i32 1, !dbg !37
// CHECK:STDOUT:   store ptr %tuple.loc16_46, ptr %tuple.loc16_477, align 8, !dbg !37
// CHECK:STDOUT:   %tuple.elem1.loc17_11.tuple.elem = getelementptr inbounds nuw { ptr, ptr }, ptr %tuple.loc16_47, i32 0, i32 1, !dbg !40
// CHECK:STDOUT:   %tuple.elem1.loc17_11.tuple.elem.load = load ptr, ptr %tuple.elem1.loc17_11.tuple.elem, align 8, !dbg !40
// CHECK:STDOUT:   %tuple.elem1.loc17_13.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %tuple.elem1.loc17_11.tuple.elem.load, i32 0, i32 1, !dbg !40
// CHECK:STDOUT:   %tuple.elem1.loc17_13.tuple.elem.load = load i32, ptr %tuple.elem1.loc17_13.tuple.elem, align 4, !dbg !40
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.5d748605768e9d50(ptr %b.var), !dbg !32
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.d2d0f824a06da6ce(ptr %a.var), !dbg !29
// CHECK:STDOUT:   ret i32 %tuple.elem1.loc17_13.tuple.elem.load, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.e554695bf635d9c1:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 1, !dbg !44
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3b18dc1d3d647932(ptr %tuple.elem1.tuple.elem), !dbg !44
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32 }, ptr %self, i32 0, i32 0, !dbg !44
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3b18dc1d3d647932(ptr %tuple.elem0.tuple.elem), !dbg !44
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.555b1314256372b8:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem2.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 2, !dbg !50
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3b18dc1d3d647932(ptr %tuple.elem2.tuple.elem), !dbg !50
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 1, !dbg !50
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3b18dc1d3d647932(ptr %tuple.elem1.tuple.elem), !dbg !50
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 0, !dbg !50
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3b18dc1d3d647932(ptr %tuple.elem0.tuple.elem), !dbg !50
// CHECK:STDOUT:   ret void, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.5d748605768e9d50(ptr %t) #0 !dbg !52 {
// CHECK:STDOUT:   call void @"_COp.e554695bf635d9c1:core.SubobjectDestroy.Core"(ptr %t), !dbg !56
// CHECK:STDOUT:   ret void, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.d2d0f824a06da6ce(ptr %t) #0 !dbg !59 {
// CHECK:STDOUT:   call void @"_COp.555b1314256372b8:core.SubobjectDestroy.Core"(ptr %t), !dbg !63
// CHECK:STDOUT:   ret void, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.3b18dc1d3d647932(ptr %t) #0 !dbg !66 {
// CHECK:STDOUT:   ret void, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.3b18dc1d3d647932, { 4, 3, 2, 1, 0 }
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
// CHECK:STDOUT: !18 = !DIFile(filename: "tuple.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{!20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{null, !23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !18, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !29 = !DILocation(line: 14, column: 3, scope: !26)
// CHECK:STDOUT: !30 = !DILocation(line: 14, column: 28, scope: !26)
// CHECK:STDOUT: !32 = !DILocation(line: 15, column: 3, scope: !26)
// CHECK:STDOUT: !33 = !DILocation(line: 15, column: 23, scope: !26)
// CHECK:STDOUT: !35 = !DILocation(line: 16, column: 43, scope: !26)
// CHECK:STDOUT: !36 = !DILocation(line: 16, column: 46, scope: !26)
// CHECK:STDOUT: !37 = !DILocation(line: 16, column: 42, scope: !26)
// CHECK:STDOUT: !40 = !DILocation(line: 17, column: 10, scope: !26)
// CHECK:STDOUT: !41 = !DILocation(line: 17, column: 3, scope: !26)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "Op", linkageName: "_COp.e554695bf635d9c1:core.SubobjectDestroy.Core", scope: null, file: !18, line: 15, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !47)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !42, type: !23)
// CHECK:STDOUT: !44 = !DILocation(line: 15, column: 3, scope: !42)
// CHECK:STDOUT: !45 = !{null, !20}
// CHECK:STDOUT: !46 = !DISubroutineType(types: !45)
// CHECK:STDOUT: !47 = !{!43}
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "Op", linkageName: "_COp.555b1314256372b8:core.SubobjectDestroy.Core", scope: null, file: !18, line: 14, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !51)
// CHECK:STDOUT: !49 = !DILocalVariable(arg: 1, scope: !48, type: !23)
// CHECK:STDOUT: !50 = !DILocation(line: 14, column: 3, scope: !48)
// CHECK:STDOUT: !51 = !{!49}
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.5d748605768e9d50", scope: null, file: !10, line: 26, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !58)
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 1, scope: !52, type: !23)
// CHECK:STDOUT: !56 = !DILocation(line: 28, column: 5, scope: !52)
// CHECK:STDOUT: !57 = !DILocation(line: 26, column: 1, scope: !52)
// CHECK:STDOUT: !58 = !{!53}
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.d2d0f824a06da6ce", scope: null, file: !10, line: 26, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !65)
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 1, scope: !59, type: !23)
// CHECK:STDOUT: !63 = !DILocation(line: 28, column: 5, scope: !59)
// CHECK:STDOUT: !64 = !DILocation(line: 26, column: 1, scope: !59)
// CHECK:STDOUT: !65 = !{!60}
// CHECK:STDOUT: !66 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.3b18dc1d3d647932", scope: null, file: !10, line: 26, type: !46, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !72)
// CHECK:STDOUT: !67 = !DILocalVariable(arg: 1, scope: !66, type: !20)
// CHECK:STDOUT: !71 = !DILocation(line: 26, column: 1, scope: !66)
// CHECK:STDOUT: !72 = !{!67}
// CHECK:STDOUT:
