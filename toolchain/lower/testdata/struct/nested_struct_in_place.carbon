// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/struct/nested_struct_in_place.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/struct/nested_struct_in_place.carbon

fn F() -> (i32, i32, i32);

fn G() {
  var _: {.a: (i32, i32, i32), .b: (i32, i32, i32)} = {.a = F(), .b = F()};
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'nested_struct_in_place.carbon'
// CHECK:STDOUT: source_filename = "nested_struct_in_place.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Main(ptr sret({ i32, i32, i32 }))
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca { { i32, i32, i32 }, { i32, i32, i32 } }, align 4, !dbg !29
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !29
// CHECK:STDOUT:   %.loc16_74.1.a = getelementptr inbounds nuw { { i32, i32, i32 }, { i32, i32, i32 } }, ptr %_.var, i32 0, i32 0, !dbg !30
// CHECK:STDOUT:   call void @_CF.Main(ptr %.loc16_74.1.a), !dbg !31
// CHECK:STDOUT:   %.loc16_74.2.b = getelementptr inbounds nuw { { i32, i32, i32 }, { i32, i32, i32 } }, ptr %_.var, i32 0, i32 1, !dbg !30
// CHECK:STDOUT:   call void @_CF.Main(ptr %.loc16_74.2.b), !dbg !32
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b222e6e2cb5261f1(ptr %_.var), !dbg !29
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.9f53a29b4522c35d:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %tuple.elem2.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 2, !dbg !37
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7f8dce37f94a83bf(ptr %tuple.elem2.tuple.elem), !dbg !37
// CHECK:STDOUT:   %tuple.elem1.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 1, !dbg !37
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7f8dce37f94a83bf(ptr %tuple.elem1.tuple.elem), !dbg !37
// CHECK:STDOUT:   %tuple.elem0.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 0, !dbg !37
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7f8dce37f94a83bf(ptr %tuple.elem0.tuple.elem), !dbg !37
// CHECK:STDOUT:   ret void, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.ce2469301e89e9e4:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc16_3.1.b = getelementptr inbounds nuw { { i32, i32, i32 }, { i32, i32, i32 } }, ptr %self, i32 0, i32 1, !dbg !43
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.e007740e1faeeffa(ptr %.loc16_3.1.b), !dbg !43
// CHECK:STDOUT:   %.loc16_3.4.a = getelementptr inbounds nuw { { i32, i32, i32 }, { i32, i32, i32 } }, ptr %self, i32 0, i32 0, !dbg !43
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.e007740e1faeeffa(ptr %.loc16_3.4.a), !dbg !43
// CHECK:STDOUT:   ret void, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.b222e6e2cb5261f1(ptr %t) #0 !dbg !45 {
// CHECK:STDOUT:   call void @"_COp.ce2469301e89e9e4:core.SubobjectDestroy.Core"(ptr %t), !dbg !49
// CHECK:STDOUT:   ret void, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.7f8dce37f94a83bf(ptr %t) #0 !dbg !52 {
// CHECK:STDOUT:   ret void, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.e007740e1faeeffa(ptr %t) #0 !dbg !59 {
// CHECK:STDOUT:   call void @"_COp.9f53a29b4522c35d:core.SubobjectDestroy.Core"(ptr %t), !dbg !63
// CHECK:STDOUT:   ret void, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.7f8dce37f94a83bf, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.e007740e1faeeffa, { 1, 0 }
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
// CHECK:STDOUT: !18 = !DIFile(filename: "nested_struct_in_place.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{null}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{null, !21}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !18, line: 15, type: !25, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !29 = !DILocation(line: 16, column: 3, scope: !28)
// CHECK:STDOUT: !30 = !DILocation(line: 16, column: 55, scope: !28)
// CHECK:STDOUT: !31 = !DILocation(line: 16, column: 61, scope: !28)
// CHECK:STDOUT: !32 = !DILocation(line: 16, column: 71, scope: !28)
// CHECK:STDOUT: !34 = !DILocation(line: 15, column: 1, scope: !28)
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "Op", linkageName: "_COp.9f53a29b4522c35d:core.SubobjectDestroy.Core", scope: null, file: !18, line: 16, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !40)
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !35, type: !21)
// CHECK:STDOUT: !37 = !DILocation(line: 16, column: 3, scope: !35)
// CHECK:STDOUT: !38 = !{null, !20}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !40 = !{!36}
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "Op", linkageName: "_COp.ce2469301e89e9e4:core.SubobjectDestroy.Core", scope: null, file: !18, line: 16, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !44)
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 1, scope: !41, type: !21)
// CHECK:STDOUT: !43 = !DILocation(line: 16, column: 3, scope: !41)
// CHECK:STDOUT: !44 = !{!42}
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.b222e6e2cb5261f1", scope: null, file: !10, line: 30, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !51)
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 1, scope: !45, type: !21)
// CHECK:STDOUT: !49 = !DILocation(line: 32, column: 5, scope: !45)
// CHECK:STDOUT: !50 = !DILocation(line: 30, column: 1, scope: !45)
// CHECK:STDOUT: !51 = !{!46}
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.7f8dce37f94a83bf", scope: null, file: !10, line: 30, type: !39, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !58)
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 1, scope: !52, type: !20)
// CHECK:STDOUT: !57 = !DILocation(line: 30, column: 1, scope: !52)
// CHECK:STDOUT: !58 = !{!53}
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.e007740e1faeeffa", scope: null, file: !10, line: 30, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !65)
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 1, scope: !59, type: !21)
// CHECK:STDOUT: !63 = !DILocation(line: 32, column: 5, scope: !59)
// CHECK:STDOUT: !64 = !DILocation(line: 30, column: 1, scope: !59)
// CHECK:STDOUT: !65 = !{!60}
// CHECK:STDOUT:
