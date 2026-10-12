// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/struct/layout.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/struct/layout.carbon

// --- no_padding.carbon
library "[[@TEST_NAME]]";

fn G(p: i32*);

fn F() {
  // LLVM {i32, i32, i32} matches the Carbon layout, so we use it.
  var no_padding: {.x: i32, .y: i32, .z: i32};
  G(&no_padding.x);
  G(&no_padding.y);
  G(&no_padding.z);
}

// --- avoid_tail_padding.carbon
library "[[@TEST_NAME]]";

fn G(p: i32*);

fn F() {
  // LLVM {ptr, i32} would have a size of 16, so we use a packed struct.
  var avoid_tail_padding: {.x: ()*, .y: i32};
  G(&avoid_tail_padding.y);
}

// --- use_tail_padding.carbon
library "[[@TEST_NAME]]";

fn G(p: i32*);

fn F() {
  // LLVM {ptr, i32} would have a size of 16, so we use a packed struct as the
  // type of `.a`. `.b` goes in the tail padding.
  var use_tail_padding: {.a: {.x: ()*, .y: i32}, .b: i32};
  G(&use_tail_padding.a.y);
  G(&use_tail_padding.b);
}

// --- implicit_padding.carbon
library "[[@TEST_NAME]]";

fn G(p: ()**);

fn F() {
  // LLVM {<{ptr, i32}>, ptr} inserts the right padding before `.b`, so we use
  // it directly.
  var implicit_padding: {.a: {.x: ()*, .y: i32}, .b: ()*};
  G(&implicit_padding.a.x);
  G(&implicit_padding.b);
}

// --- insert_padding.carbon
library "[[@TEST_NAME]]";

fn G(p: ()**);

fn F() {
  // LLVM {<{i32, i32, i32}>, ptr, i32} would have extra tail padding, so we use
  // a packed struct. That requires that we explicitly add padding between `.a`
  // and `.b`.
  var insert_padding: {.a: {.x: i32, .y: i32, .z: i32}, .b: ()*, .c: i32};
  G(&insert_padding.b);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'no_padding.carbon'
// CHECK:STDOUT: source_filename = "no_padding.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %no_padding.var = alloca { i32, i32, i32 }, align 4, !dbg !27
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %no_padding.var), !dbg !27
// CHECK:STDOUT:   %.loc8.x = getelementptr inbounds nuw { i32, i32, i32 }, ptr %no_padding.var, i32 0, i32 0, !dbg !29
// CHECK:STDOUT:   call void @_CG.Main(ptr %.loc8.x), !dbg !31
// CHECK:STDOUT:   %.loc9.y = getelementptr inbounds nuw { i32, i32, i32 }, ptr %no_padding.var, i32 0, i32 1, !dbg !32
// CHECK:STDOUT:   call void @_CG.Main(ptr %.loc9.y), !dbg !34
// CHECK:STDOUT:   %.loc10.z = getelementptr inbounds nuw { i32, i32, i32 }, ptr %no_padding.var, i32 0, i32 2, !dbg !35
// CHECK:STDOUT:   call void @_CG.Main(ptr %.loc10.z), !dbg !37
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.d978dd91908bcbb7(ptr %no_padding.var), !dbg !27
// CHECK:STDOUT:   ret void, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.2c905546659eba19:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_3.1.z = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 2, !dbg !41
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %.loc7_3.1.z), !dbg !41
// CHECK:STDOUT:   %.loc7_3.4.y = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 1, !dbg !41
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %.loc7_3.4.y), !dbg !41
// CHECK:STDOUT:   %.loc7_3.7.x = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 0, !dbg !41
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %.loc7_3.7.x), !dbg !41
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.d978dd91908bcbb7(ptr %t) #0 !dbg !45 {
// CHECK:STDOUT:   call void @"_COp.2c905546659eba19:core.SubobjectDestroy.Core"(ptr %t), !dbg !49
// CHECK:STDOUT:   ret void, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %t) #0 !dbg !52 {
// CHECK:STDOUT:   ret void, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.0aefc5da911f5051, { 2, 1, 0 }
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
// CHECK:STDOUT: !18 = !DIFile(filename: "no_padding.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{null, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !{null}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !18, line: 5, type: !25, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !27 = !DILocation(line: 7, column: 3, scope: !26)
// CHECK:STDOUT: !29 = !DILocation(line: 8, column: 6, scope: !26)
// CHECK:STDOUT: !31 = !DILocation(line: 8, column: 3, scope: !26)
// CHECK:STDOUT: !32 = !DILocation(line: 9, column: 6, scope: !26)
// CHECK:STDOUT: !34 = !DILocation(line: 9, column: 3, scope: !26)
// CHECK:STDOUT: !35 = !DILocation(line: 10, column: 6, scope: !26)
// CHECK:STDOUT: !37 = !DILocation(line: 10, column: 3, scope: !26)
// CHECK:STDOUT: !38 = !DILocation(line: 5, column: 1, scope: !26)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "Op", linkageName: "_COp.2c905546659eba19:core.SubobjectDestroy.Core", scope: null, file: !18, line: 7, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !44)
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !39, type: !21)
// CHECK:STDOUT: !41 = !DILocation(line: 7, column: 3, scope: !39)
// CHECK:STDOUT: !42 = !{null, !20}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !44 = !{!40}
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.d978dd91908bcbb7", scope: null, file: !10, line: 26, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !51)
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 1, scope: !45, type: !21)
// CHECK:STDOUT: !49 = !DILocation(line: 28, column: 5, scope: !45)
// CHECK:STDOUT: !50 = !DILocation(line: 26, column: 1, scope: !45)
// CHECK:STDOUT: !51 = !{!46}
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.0aefc5da911f5051", scope: null, file: !10, line: 26, type: !43, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !58)
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 1, scope: !52, type: !20)
// CHECK:STDOUT: !57 = !DILocation(line: 26, column: 1, scope: !52)
// CHECK:STDOUT: !58 = !{!53}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'avoid_tail_padding.carbon'
// CHECK:STDOUT: source_filename = "avoid_tail_padding.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !61 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %avoid_tail_padding.var = alloca <{ ptr, i32 }>, align 8, !dbg !62
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %avoid_tail_padding.var), !dbg !62
// CHECK:STDOUT:   %.loc8.y = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %avoid_tail_padding.var, i32 0, i32 1, !dbg !64
// CHECK:STDOUT:   call void @_CG.Main(ptr %.loc8.y), !dbg !66
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.887d07eea57f466d(ptr %avoid_tail_padding.var), !dbg !62
// CHECK:STDOUT:   ret void, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.a6800c4c99d7a8c1:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !68 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_3.1.y = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %self, i32 0, i32 1, !dbg !70
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %.loc7_3.1.y), !dbg !70
// CHECK:STDOUT:   %.loc7_3.4.x = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %self, i32 0, i32 0, !dbg !70
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.437d3b42b67f7a2e(ptr %.loc7_3.4.x), !dbg !70
// CHECK:STDOUT:   ret void, !dbg !70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.887d07eea57f466d(ptr %t) #0 !dbg !72 {
// CHECK:STDOUT:   call void @"_COp.a6800c4c99d7a8c1:core.SubobjectDestroy.Core"(ptr %t), !dbg !76
// CHECK:STDOUT:   ret void, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %t) #0 !dbg !79 {
// CHECK:STDOUT:   ret void, !dbg !84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.437d3b42b67f7a2e(ptr %t) #0 !dbg !86 {
// CHECK:STDOUT:   ret void, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!60}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{null, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !{null}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !42 = !{null, !20}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !59 = !DIFile(filename: "avoid_tail_padding.carbon", directory: "")
// CHECK:STDOUT: !60 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !59, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !61 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !59, line: 5, type: !25, spFlags: DISPFlagDefinition, unit: !60)
// CHECK:STDOUT: !62 = !DILocation(line: 7, column: 3, scope: !61)
// CHECK:STDOUT: !64 = !DILocation(line: 8, column: 6, scope: !61)
// CHECK:STDOUT: !66 = !DILocation(line: 8, column: 3, scope: !61)
// CHECK:STDOUT: !67 = !DILocation(line: 5, column: 1, scope: !61)
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "Op", linkageName: "_COp.a6800c4c99d7a8c1:core.SubobjectDestroy.Core", scope: null, file: !59, line: 7, type: !23, spFlags: DISPFlagDefinition, unit: !60, retainedNodes: !71)
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 1, scope: !68, type: !21)
// CHECK:STDOUT: !70 = !DILocation(line: 7, column: 3, scope: !68)
// CHECK:STDOUT: !71 = !{!69}
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.887d07eea57f466d", scope: null, file: !10, line: 26, type: !23, spFlags: DISPFlagDefinition, unit: !60, retainedNodes: !78)
// CHECK:STDOUT: !73 = !DILocalVariable(arg: 1, scope: !72, type: !21)
// CHECK:STDOUT: !76 = !DILocation(line: 28, column: 5, scope: !72)
// CHECK:STDOUT: !77 = !DILocation(line: 26, column: 1, scope: !72)
// CHECK:STDOUT: !78 = !{!73}
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.0aefc5da911f5051", scope: null, file: !10, line: 26, type: !43, spFlags: DISPFlagDefinition, unit: !60, retainedNodes: !85)
// CHECK:STDOUT: !80 = !DILocalVariable(arg: 1, scope: !79, type: !20)
// CHECK:STDOUT: !84 = !DILocation(line: 26, column: 1, scope: !79)
// CHECK:STDOUT: !85 = !{!80}
// CHECK:STDOUT: !86 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.437d3b42b67f7a2e", scope: null, file: !10, line: 26, type: !23, spFlags: DISPFlagDefinition, unit: !60, retainedNodes: !92)
// CHECK:STDOUT: !87 = !DILocalVariable(arg: 1, scope: !86, type: !21)
// CHECK:STDOUT: !91 = !DILocation(line: 26, column: 1, scope: !86)
// CHECK:STDOUT: !92 = !{!87}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'use_tail_padding.carbon'
// CHECK:STDOUT: source_filename = "use_tail_padding.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !95 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %use_tail_padding.var = alloca { <{ ptr, i32 }>, i32 }, align 8, !dbg !96
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %use_tail_padding.var), !dbg !96
// CHECK:STDOUT:   %.loc9_22.a = getelementptr inbounds nuw { <{ ptr, i32 }>, i32 }, ptr %use_tail_padding.var, i32 0, i32 0, !dbg !98
// CHECK:STDOUT:   %.loc9_24.y = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %.loc9_22.a, i32 0, i32 1, !dbg !98
// CHECK:STDOUT:   call void @_CG.Main(ptr %.loc9_24.y), !dbg !100
// CHECK:STDOUT:   %.loc10.b = getelementptr inbounds nuw { <{ ptr, i32 }>, i32 }, ptr %use_tail_padding.var, i32 0, i32 1, !dbg !101
// CHECK:STDOUT:   call void @_CG.Main(ptr %.loc10.b), !dbg !103
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3d847764181cc6e2(ptr %use_tail_padding.var), !dbg !96
// CHECK:STDOUT:   ret void, !dbg !104
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.a6800c4c99d7a8c1:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !105 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc8_3.1.y = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %self, i32 0, i32 1, !dbg !107
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %.loc8_3.1.y), !dbg !107
// CHECK:STDOUT:   %.loc8_3.4.x = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %self, i32 0, i32 0, !dbg !107
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.437d3b42b67f7a2e(ptr %.loc8_3.4.x), !dbg !107
// CHECK:STDOUT:   ret void, !dbg !107
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.b8a9dc445fab942c:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !109 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc8_3.1.b = getelementptr inbounds nuw { <{ ptr, i32 }>, i32 }, ptr %self, i32 0, i32 1, !dbg !111
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %.loc8_3.1.b), !dbg !111
// CHECK:STDOUT:   %.loc8_3.4.a = getelementptr inbounds nuw { <{ ptr, i32 }>, i32 }, ptr %self, i32 0, i32 0, !dbg !111
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.887d07eea57f466d(ptr %.loc8_3.4.a), !dbg !111
// CHECK:STDOUT:   ret void, !dbg !111
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.3d847764181cc6e2(ptr %t) #0 !dbg !113 {
// CHECK:STDOUT:   call void @"_COp.b8a9dc445fab942c:core.SubobjectDestroy.Core"(ptr %t), !dbg !117
// CHECK:STDOUT:   ret void, !dbg !118
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %t) #0 !dbg !120 {
// CHECK:STDOUT:   ret void, !dbg !125
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.437d3b42b67f7a2e(ptr %t) #0 !dbg !127 {
// CHECK:STDOUT:   ret void, !dbg !132
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.887d07eea57f466d(ptr %t) #0 !dbg !134 {
// CHECK:STDOUT:   call void @"_COp.a6800c4c99d7a8c1:core.SubobjectDestroy.Core"(ptr %t), !dbg !138
// CHECK:STDOUT:   ret void, !dbg !139
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.0aefc5da911f5051, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!94}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{null, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !{null}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !42 = !{null, !20}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !93 = !DIFile(filename: "use_tail_padding.carbon", directory: "")
// CHECK:STDOUT: !94 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !93, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !95 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !93, line: 5, type: !25, spFlags: DISPFlagDefinition, unit: !94)
// CHECK:STDOUT: !96 = !DILocation(line: 8, column: 3, scope: !95)
// CHECK:STDOUT: !98 = !DILocation(line: 9, column: 6, scope: !95)
// CHECK:STDOUT: !100 = !DILocation(line: 9, column: 3, scope: !95)
// CHECK:STDOUT: !101 = !DILocation(line: 10, column: 6, scope: !95)
// CHECK:STDOUT: !103 = !DILocation(line: 10, column: 3, scope: !95)
// CHECK:STDOUT: !104 = !DILocation(line: 5, column: 1, scope: !95)
// CHECK:STDOUT: !105 = distinct !DISubprogram(name: "Op", linkageName: "_COp.a6800c4c99d7a8c1:core.SubobjectDestroy.Core", scope: null, file: !93, line: 8, type: !23, spFlags: DISPFlagDefinition, unit: !94, retainedNodes: !108)
// CHECK:STDOUT: !106 = !DILocalVariable(arg: 1, scope: !105, type: !21)
// CHECK:STDOUT: !107 = !DILocation(line: 8, column: 3, scope: !105)
// CHECK:STDOUT: !108 = !{!106}
// CHECK:STDOUT: !109 = distinct !DISubprogram(name: "Op", linkageName: "_COp.b8a9dc445fab942c:core.SubobjectDestroy.Core", scope: null, file: !93, line: 8, type: !23, spFlags: DISPFlagDefinition, unit: !94, retainedNodes: !112)
// CHECK:STDOUT: !110 = !DILocalVariable(arg: 1, scope: !109, type: !21)
// CHECK:STDOUT: !111 = !DILocation(line: 8, column: 3, scope: !109)
// CHECK:STDOUT: !112 = !{!110}
// CHECK:STDOUT: !113 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.3d847764181cc6e2", scope: null, file: !10, line: 26, type: !23, spFlags: DISPFlagDefinition, unit: !94, retainedNodes: !119)
// CHECK:STDOUT: !114 = !DILocalVariable(arg: 1, scope: !113, type: !21)
// CHECK:STDOUT: !117 = !DILocation(line: 28, column: 5, scope: !113)
// CHECK:STDOUT: !118 = !DILocation(line: 26, column: 1, scope: !113)
// CHECK:STDOUT: !119 = !{!114}
// CHECK:STDOUT: !120 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.0aefc5da911f5051", scope: null, file: !10, line: 26, type: !43, spFlags: DISPFlagDefinition, unit: !94, retainedNodes: !126)
// CHECK:STDOUT: !121 = !DILocalVariable(arg: 1, scope: !120, type: !20)
// CHECK:STDOUT: !125 = !DILocation(line: 26, column: 1, scope: !120)
// CHECK:STDOUT: !126 = !{!121}
// CHECK:STDOUT: !127 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.437d3b42b67f7a2e", scope: null, file: !10, line: 26, type: !23, spFlags: DISPFlagDefinition, unit: !94, retainedNodes: !133)
// CHECK:STDOUT: !128 = !DILocalVariable(arg: 1, scope: !127, type: !21)
// CHECK:STDOUT: !132 = !DILocation(line: 26, column: 1, scope: !127)
// CHECK:STDOUT: !133 = !{!128}
// CHECK:STDOUT: !134 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.887d07eea57f466d", scope: null, file: !10, line: 26, type: !23, spFlags: DISPFlagDefinition, unit: !94, retainedNodes: !140)
// CHECK:STDOUT: !135 = !DILocalVariable(arg: 1, scope: !134, type: !21)
// CHECK:STDOUT: !138 = !DILocation(line: 28, column: 5, scope: !134)
// CHECK:STDOUT: !139 = !DILocation(line: 26, column: 1, scope: !134)
// CHECK:STDOUT: !140 = !{!135}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'implicit_padding.carbon'
// CHECK:STDOUT: source_filename = "implicit_padding.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !143 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %implicit_padding.var = alloca { <{ ptr, i32 }>, ptr }, align 8, !dbg !144
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %implicit_padding.var), !dbg !144
// CHECK:STDOUT:   %.loc9_22.a = getelementptr inbounds nuw { <{ ptr, i32 }>, ptr }, ptr %implicit_padding.var, i32 0, i32 0, !dbg !146
// CHECK:STDOUT:   %.loc9_24.x = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %.loc9_22.a, i32 0, i32 0, !dbg !146
// CHECK:STDOUT:   call void @_CG.Main(ptr %.loc9_24.x), !dbg !148
// CHECK:STDOUT:   %.loc10.b = getelementptr inbounds nuw { <{ ptr, i32 }>, ptr }, ptr %implicit_padding.var, i32 0, i32 1, !dbg !149
// CHECK:STDOUT:   call void @_CG.Main(ptr %.loc10.b), !dbg !151
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3f70cb6243cc249e(ptr %implicit_padding.var), !dbg !144
// CHECK:STDOUT:   ret void, !dbg !152
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.a6800c4c99d7a8c1:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !153 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc8_3.1.y = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %self, i32 0, i32 1, !dbg !155
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %.loc8_3.1.y), !dbg !155
// CHECK:STDOUT:   %.loc8_3.4.x = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %self, i32 0, i32 0, !dbg !155
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.437d3b42b67f7a2e(ptr %.loc8_3.4.x), !dbg !155
// CHECK:STDOUT:   ret void, !dbg !155
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.aad81ae9f983033e:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !157 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc8_3.1.b = getelementptr inbounds nuw { <{ ptr, i32 }>, ptr }, ptr %self, i32 0, i32 1, !dbg !159
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.437d3b42b67f7a2e(ptr %.loc8_3.1.b), !dbg !159
// CHECK:STDOUT:   %.loc8_3.4.a = getelementptr inbounds nuw { <{ ptr, i32 }>, ptr }, ptr %self, i32 0, i32 0, !dbg !159
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.887d07eea57f466d(ptr %.loc8_3.4.a), !dbg !159
// CHECK:STDOUT:   ret void, !dbg !159
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.3f70cb6243cc249e(ptr %t) #0 !dbg !161 {
// CHECK:STDOUT:   call void @"_COp.aad81ae9f983033e:core.SubobjectDestroy.Core"(ptr %t), !dbg !165
// CHECK:STDOUT:   ret void, !dbg !166
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %t) #0 !dbg !168 {
// CHECK:STDOUT:   ret void, !dbg !173
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.437d3b42b67f7a2e(ptr %t) #0 !dbg !175 {
// CHECK:STDOUT:   ret void, !dbg !180
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.887d07eea57f466d(ptr %t) #0 !dbg !182 {
// CHECK:STDOUT:   call void @"_COp.a6800c4c99d7a8c1:core.SubobjectDestroy.Core"(ptr %t), !dbg !186
// CHECK:STDOUT:   ret void, !dbg !187
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.437d3b42b67f7a2e, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!142}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{null, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !{null}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !42 = !{null, !20}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !141 = !DIFile(filename: "implicit_padding.carbon", directory: "")
// CHECK:STDOUT: !142 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !141, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !143 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !141, line: 5, type: !25, spFlags: DISPFlagDefinition, unit: !142)
// CHECK:STDOUT: !144 = !DILocation(line: 8, column: 3, scope: !143)
// CHECK:STDOUT: !146 = !DILocation(line: 9, column: 6, scope: !143)
// CHECK:STDOUT: !148 = !DILocation(line: 9, column: 3, scope: !143)
// CHECK:STDOUT: !149 = !DILocation(line: 10, column: 6, scope: !143)
// CHECK:STDOUT: !151 = !DILocation(line: 10, column: 3, scope: !143)
// CHECK:STDOUT: !152 = !DILocation(line: 5, column: 1, scope: !143)
// CHECK:STDOUT: !153 = distinct !DISubprogram(name: "Op", linkageName: "_COp.a6800c4c99d7a8c1:core.SubobjectDestroy.Core", scope: null, file: !141, line: 8, type: !23, spFlags: DISPFlagDefinition, unit: !142, retainedNodes: !156)
// CHECK:STDOUT: !154 = !DILocalVariable(arg: 1, scope: !153, type: !21)
// CHECK:STDOUT: !155 = !DILocation(line: 8, column: 3, scope: !153)
// CHECK:STDOUT: !156 = !{!154}
// CHECK:STDOUT: !157 = distinct !DISubprogram(name: "Op", linkageName: "_COp.aad81ae9f983033e:core.SubobjectDestroy.Core", scope: null, file: !141, line: 8, type: !23, spFlags: DISPFlagDefinition, unit: !142, retainedNodes: !160)
// CHECK:STDOUT: !158 = !DILocalVariable(arg: 1, scope: !157, type: !21)
// CHECK:STDOUT: !159 = !DILocation(line: 8, column: 3, scope: !157)
// CHECK:STDOUT: !160 = !{!158}
// CHECK:STDOUT: !161 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.3f70cb6243cc249e", scope: null, file: !10, line: 26, type: !23, spFlags: DISPFlagDefinition, unit: !142, retainedNodes: !167)
// CHECK:STDOUT: !162 = !DILocalVariable(arg: 1, scope: !161, type: !21)
// CHECK:STDOUT: !165 = !DILocation(line: 28, column: 5, scope: !161)
// CHECK:STDOUT: !166 = !DILocation(line: 26, column: 1, scope: !161)
// CHECK:STDOUT: !167 = !{!162}
// CHECK:STDOUT: !168 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.0aefc5da911f5051", scope: null, file: !10, line: 26, type: !43, spFlags: DISPFlagDefinition, unit: !142, retainedNodes: !174)
// CHECK:STDOUT: !169 = !DILocalVariable(arg: 1, scope: !168, type: !20)
// CHECK:STDOUT: !173 = !DILocation(line: 26, column: 1, scope: !168)
// CHECK:STDOUT: !174 = !{!169}
// CHECK:STDOUT: !175 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.437d3b42b67f7a2e", scope: null, file: !10, line: 26, type: !23, spFlags: DISPFlagDefinition, unit: !142, retainedNodes: !181)
// CHECK:STDOUT: !176 = !DILocalVariable(arg: 1, scope: !175, type: !21)
// CHECK:STDOUT: !180 = !DILocation(line: 26, column: 1, scope: !175)
// CHECK:STDOUT: !181 = !{!176}
// CHECK:STDOUT: !182 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.887d07eea57f466d", scope: null, file: !10, line: 26, type: !23, spFlags: DISPFlagDefinition, unit: !142, retainedNodes: !188)
// CHECK:STDOUT: !183 = !DILocalVariable(arg: 1, scope: !182, type: !21)
// CHECK:STDOUT: !186 = !DILocation(line: 28, column: 5, scope: !182)
// CHECK:STDOUT: !187 = !DILocation(line: 26, column: 1, scope: !182)
// CHECK:STDOUT: !188 = !{!183}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'insert_padding.carbon'
// CHECK:STDOUT: source_filename = "insert_padding.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !191 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %insert_padding.var = alloca <{ <{ { i32, i32, i32 }, [4 x i8] }>, ptr, i32 }>, align 8, !dbg !192
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %insert_padding.var), !dbg !192
// CHECK:STDOUT:   %.loc10.b = getelementptr inbounds nuw <{ <{ { i32, i32, i32 }, [4 x i8] }>, ptr, i32 }>, ptr %insert_padding.var, i32 0, i32 1, !dbg !194
// CHECK:STDOUT:   call void @_CG.Main(ptr %.loc10.b), !dbg !196
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f4a47bb07818f0ac(ptr %insert_padding.var), !dbg !192
// CHECK:STDOUT:   ret void, !dbg !197
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.2c905546659eba19:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !198 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_3.1.z = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 2, !dbg !200
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %.loc9_3.1.z), !dbg !200
// CHECK:STDOUT:   %.loc9_3.4.y = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 1, !dbg !200
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %.loc9_3.4.y), !dbg !200
// CHECK:STDOUT:   %.loc9_3.7.x = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 0, !dbg !200
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %.loc9_3.7.x), !dbg !200
// CHECK:STDOUT:   ret void, !dbg !200
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.1290879a15319f2b:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !202 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_3.1.c = getelementptr inbounds nuw <{ <{ { i32, i32, i32 }, [4 x i8] }>, ptr, i32 }>, ptr %self, i32 0, i32 2, !dbg !204
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %.loc9_3.1.c), !dbg !204
// CHECK:STDOUT:   %.loc9_3.4.b = getelementptr inbounds nuw <{ <{ { i32, i32, i32 }, [4 x i8] }>, ptr, i32 }>, ptr %self, i32 0, i32 1, !dbg !204
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.437d3b42b67f7a2e(ptr %.loc9_3.4.b), !dbg !204
// CHECK:STDOUT:   %.loc9_3.7.a = getelementptr inbounds nuw <{ <{ { i32, i32, i32 }, [4 x i8] }>, ptr, i32 }>, ptr %self, i32 0, i32 0, !dbg !204
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.d978dd91908bcbb7(ptr %.loc9_3.7.a), !dbg !204
// CHECK:STDOUT:   ret void, !dbg !204
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f4a47bb07818f0ac(ptr %t) #0 !dbg !206 {
// CHECK:STDOUT:   call void @"_COp.1290879a15319f2b:core.SubobjectDestroy.Core"(ptr %t), !dbg !210
// CHECK:STDOUT:   ret void, !dbg !211
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.0aefc5da911f5051(ptr %t) #0 !dbg !213 {
// CHECK:STDOUT:   ret void, !dbg !218
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.437d3b42b67f7a2e(ptr %t) #0 !dbg !220 {
// CHECK:STDOUT:   ret void, !dbg !225
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.d978dd91908bcbb7(ptr %t) #0 !dbg !227 {
// CHECK:STDOUT:   call void @"_COp.2c905546659eba19:core.SubobjectDestroy.Core"(ptr %t), !dbg !231
// CHECK:STDOUT:   ret void, !dbg !232
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.0aefc5da911f5051, { 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!190}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{null, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !{null}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !42 = !{null, !20}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !189 = !DIFile(filename: "insert_padding.carbon", directory: "")
// CHECK:STDOUT: !190 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !189, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !191 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !189, line: 5, type: !25, spFlags: DISPFlagDefinition, unit: !190)
// CHECK:STDOUT: !192 = !DILocation(line: 9, column: 3, scope: !191)
// CHECK:STDOUT: !194 = !DILocation(line: 10, column: 6, scope: !191)
// CHECK:STDOUT: !196 = !DILocation(line: 10, column: 3, scope: !191)
// CHECK:STDOUT: !197 = !DILocation(line: 5, column: 1, scope: !191)
// CHECK:STDOUT: !198 = distinct !DISubprogram(name: "Op", linkageName: "_COp.2c905546659eba19:core.SubobjectDestroy.Core", scope: null, file: !189, line: 9, type: !23, spFlags: DISPFlagDefinition, unit: !190, retainedNodes: !201)
// CHECK:STDOUT: !199 = !DILocalVariable(arg: 1, scope: !198, type: !21)
// CHECK:STDOUT: !200 = !DILocation(line: 9, column: 3, scope: !198)
// CHECK:STDOUT: !201 = !{!199}
// CHECK:STDOUT: !202 = distinct !DISubprogram(name: "Op", linkageName: "_COp.1290879a15319f2b:core.SubobjectDestroy.Core", scope: null, file: !189, line: 9, type: !23, spFlags: DISPFlagDefinition, unit: !190, retainedNodes: !205)
// CHECK:STDOUT: !203 = !DILocalVariable(arg: 1, scope: !202, type: !21)
// CHECK:STDOUT: !204 = !DILocation(line: 9, column: 3, scope: !202)
// CHECK:STDOUT: !205 = !{!203}
// CHECK:STDOUT: !206 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f4a47bb07818f0ac", scope: null, file: !10, line: 26, type: !23, spFlags: DISPFlagDefinition, unit: !190, retainedNodes: !212)
// CHECK:STDOUT: !207 = !DILocalVariable(arg: 1, scope: !206, type: !21)
// CHECK:STDOUT: !210 = !DILocation(line: 28, column: 5, scope: !206)
// CHECK:STDOUT: !211 = !DILocation(line: 26, column: 1, scope: !206)
// CHECK:STDOUT: !212 = !{!207}
// CHECK:STDOUT: !213 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.0aefc5da911f5051", scope: null, file: !10, line: 26, type: !43, spFlags: DISPFlagDefinition, unit: !190, retainedNodes: !219)
// CHECK:STDOUT: !214 = !DILocalVariable(arg: 1, scope: !213, type: !20)
// CHECK:STDOUT: !218 = !DILocation(line: 26, column: 1, scope: !213)
// CHECK:STDOUT: !219 = !{!214}
// CHECK:STDOUT: !220 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.437d3b42b67f7a2e", scope: null, file: !10, line: 26, type: !23, spFlags: DISPFlagDefinition, unit: !190, retainedNodes: !226)
// CHECK:STDOUT: !221 = !DILocalVariable(arg: 1, scope: !220, type: !21)
// CHECK:STDOUT: !225 = !DILocation(line: 26, column: 1, scope: !220)
// CHECK:STDOUT: !226 = !{!221}
// CHECK:STDOUT: !227 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.d978dd91908bcbb7", scope: null, file: !10, line: 26, type: !23, spFlags: DISPFlagDefinition, unit: !190, retainedNodes: !233)
// CHECK:STDOUT: !228 = !DILocalVariable(arg: 1, scope: !227, type: !21)
// CHECK:STDOUT: !231 = !DILocation(line: 28, column: 5, scope: !227)
// CHECK:STDOUT: !232 = !DILocation(line: 26, column: 1, scope: !227)
// CHECK:STDOUT: !233 = !{!228}
// CHECK:STDOUT:
