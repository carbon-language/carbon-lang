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
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %no_padding.var = alloca { i32, i32, i32 }, align 4, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %no_padding.var), !dbg !7
// CHECK:STDOUT:   %.loc8.x = getelementptr inbounds nuw { i32, i32, i32 }, ptr %no_padding.var, i32 0, i32 0, !dbg !8
// CHECK:STDOUT:   call void @_CG.Main(ptr %.loc8.x), !dbg !9
// CHECK:STDOUT:   %.loc9.y = getelementptr inbounds nuw { i32, i32, i32 }, ptr %no_padding.var, i32 0, i32 1, !dbg !10
// CHECK:STDOUT:   call void @_CG.Main(ptr %.loc9.y), !dbg !11
// CHECK:STDOUT:   %.loc10.z = getelementptr inbounds nuw { i32, i32, i32 }, ptr %no_padding.var, i32 0, i32 2, !dbg !12
// CHECK:STDOUT:   call void @_CG.Main(ptr %.loc10.z), !dbg !13
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.a876613cdec73e2b(ptr %no_padding.var), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.70db2d92ffddbef7:core.Destroy.Core"(ptr %self) #0 !dbg !15 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.ef264ddbcdaf6301:core.Destroy.Core"(ptr %self) #0 !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.a876613cdec73e2b(ptr %self) #0 !dbg !29 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.ef264ddbcdaf6301:core.Destroy.Core"(ptr %self), !dbg !33
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "no_padding.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !1, line: 5, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 7, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 8, column: 6, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 8, column: 3, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 9, column: 6, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 9, column: 3, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 10, column: 6, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 10, column: 3, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 5, column: 1, scope: !4)
// CHECK:STDOUT: !15 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.70db2d92ffddbef7:core.Destroy.Core", scope: null, file: !1, line: 7, type: !16, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !19)
// CHECK:STDOUT: !16 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !17 = !{null, !18}
// CHECK:STDOUT: !18 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !19 = !{!20}
// CHECK:STDOUT: !20 = !DILocalVariable(arg: 1, scope: !15, type: !18)
// CHECK:STDOUT: !21 = !DILocation(line: 7, column: 3, scope: !15)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.ef264ddbcdaf6301:core.Destroy.Core", scope: null, file: !1, line: 7, type: !23, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !26)
// CHECK:STDOUT: !23 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !24 = !{null, !25}
// CHECK:STDOUT: !25 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !26 = !{!27}
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !22, type: !25)
// CHECK:STDOUT: !28 = !DILocation(line: 7, column: 3, scope: !22)
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.a876613cdec73e2b", scope: null, file: !30, line: 24, type: !23, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !31)
// CHECK:STDOUT: !30 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !31 = !{!32}
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !29, type: !25)
// CHECK:STDOUT: !33 = !DILocation(line: 26, column: 5, scope: !29)
// CHECK:STDOUT: !34 = !DILocation(line: 24, column: 3, scope: !29)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'avoid_tail_padding.carbon'
// CHECK:STDOUT: source_filename = "avoid_tail_padding.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %avoid_tail_padding.var = alloca <{ ptr, i32 }>, align 8, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %avoid_tail_padding.var), !dbg !7
// CHECK:STDOUT:   %.loc8.y = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %avoid_tail_padding.var, i32 0, i32 1, !dbg !8
// CHECK:STDOUT:   call void @_CG.Main(ptr %.loc8.y), !dbg !9
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.f58fd6d7a2f83df4(ptr %avoid_tail_padding.var), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.70db2d92ffddbef7:core.Destroy.Core"(ptr %self) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.39f44b5e2be36eae:core.Destroy.Core"(ptr %self) #0 !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.f58fd6d7a2f83df4(ptr %self) #0 !dbg !25 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.39f44b5e2be36eae:core.Destroy.Core"(ptr %self), !dbg !29
// CHECK:STDOUT:   ret void, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "avoid_tail_padding.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !1, line: 5, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 7, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 8, column: 6, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 8, column: 3, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 5, column: 1, scope: !4)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.70db2d92ffddbef7:core.Destroy.Core", scope: null, file: !1, line: 7, type: !12, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !15)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{null, !14}
// CHECK:STDOUT: !14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !15 = !{!16}
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 1, scope: !11, type: !14)
// CHECK:STDOUT: !17 = !DILocation(line: 7, column: 3, scope: !11)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.39f44b5e2be36eae:core.Destroy.Core", scope: null, file: !1, line: 7, type: !19, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !22)
// CHECK:STDOUT: !19 = !DISubroutineType(types: !20)
// CHECK:STDOUT: !20 = !{null, !21}
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!23}
// CHECK:STDOUT: !23 = !DILocalVariable(arg: 1, scope: !18, type: !21)
// CHECK:STDOUT: !24 = !DILocation(line: 7, column: 3, scope: !18)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.f58fd6d7a2f83df4", scope: null, file: !26, line: 24, type: !19, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !27)
// CHECK:STDOUT: !26 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !27 = !{!28}
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 1, scope: !25, type: !21)
// CHECK:STDOUT: !29 = !DILocation(line: 26, column: 5, scope: !25)
// CHECK:STDOUT: !30 = !DILocation(line: 24, column: 3, scope: !25)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'use_tail_padding.carbon'
// CHECK:STDOUT: source_filename = "use_tail_padding.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %use_tail_padding.var = alloca { <{ ptr, i32 }>, i32 }, align 8, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %use_tail_padding.var), !dbg !7
// CHECK:STDOUT:   %.loc9_22.a = getelementptr inbounds nuw { <{ ptr, i32 }>, i32 }, ptr %use_tail_padding.var, i32 0, i32 0, !dbg !8
// CHECK:STDOUT:   %.loc9_24.y = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %.loc9_22.a, i32 0, i32 1, !dbg !8
// CHECK:STDOUT:   call void @_CG.Main(ptr %.loc9_24.y), !dbg !9
// CHECK:STDOUT:   %.loc10.b = getelementptr inbounds nuw { <{ ptr, i32 }>, i32 }, ptr %use_tail_padding.var, i32 0, i32 1, !dbg !10
// CHECK:STDOUT:   call void @_CG.Main(ptr %.loc10.b), !dbg !11
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.608682cf29acc5fb(ptr %use_tail_padding.var), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.70db2d92ffddbef7:core.Destroy.Core"(ptr %self) #0 !dbg !13 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.39f44b5e2be36eae:core.Destroy.Core"(ptr %self) #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.43335663c40fb4bc:core.Destroy.Core"(ptr %self) #0 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.608682cf29acc5fb(ptr %self) #0 !dbg !31 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.43335663c40fb4bc:core.Destroy.Core"(ptr %self), !dbg !35
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "use_tail_padding.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !1, line: 5, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 8, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 9, column: 6, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 9, column: 3, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 10, column: 6, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 10, column: 3, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 5, column: 1, scope: !4)
// CHECK:STDOUT: !13 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.70db2d92ffddbef7:core.Destroy.Core", scope: null, file: !1, line: 8, type: !14, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !17)
// CHECK:STDOUT: !14 = !DISubroutineType(types: !15)
// CHECK:STDOUT: !15 = !{null, !16}
// CHECK:STDOUT: !16 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !17 = !{!18}
// CHECK:STDOUT: !18 = !DILocalVariable(arg: 1, scope: !13, type: !16)
// CHECK:STDOUT: !19 = !DILocation(line: 8, column: 3, scope: !13)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.39f44b5e2be36eae:core.Destroy.Core", scope: null, file: !1, line: 8, type: !21, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !24)
// CHECK:STDOUT: !21 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !22 = !{null, !23}
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!25}
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !20, type: !23)
// CHECK:STDOUT: !26 = !DILocation(line: 8, column: 3, scope: !20)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.43335663c40fb4bc:core.Destroy.Core", scope: null, file: !1, line: 8, type: !21, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !28)
// CHECK:STDOUT: !28 = !{!29}
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 1, scope: !27, type: !23)
// CHECK:STDOUT: !30 = !DILocation(line: 8, column: 3, scope: !27)
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.608682cf29acc5fb", scope: null, file: !32, line: 24, type: !21, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !33)
// CHECK:STDOUT: !32 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !33 = !{!34}
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 1, scope: !31, type: !23)
// CHECK:STDOUT: !35 = !DILocation(line: 26, column: 5, scope: !31)
// CHECK:STDOUT: !36 = !DILocation(line: 24, column: 3, scope: !31)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'implicit_padding.carbon'
// CHECK:STDOUT: source_filename = "implicit_padding.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %implicit_padding.var = alloca { <{ ptr, i32 }>, ptr }, align 8, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %implicit_padding.var), !dbg !7
// CHECK:STDOUT:   %.loc9_22.a = getelementptr inbounds nuw { <{ ptr, i32 }>, ptr }, ptr %implicit_padding.var, i32 0, i32 0, !dbg !8
// CHECK:STDOUT:   %.loc9_24.x = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %.loc9_22.a, i32 0, i32 0, !dbg !8
// CHECK:STDOUT:   call void @_CG.Main(ptr %.loc9_24.x), !dbg !9
// CHECK:STDOUT:   %.loc10.b = getelementptr inbounds nuw { <{ ptr, i32 }>, ptr }, ptr %implicit_padding.var, i32 0, i32 1, !dbg !10
// CHECK:STDOUT:   call void @_CG.Main(ptr %.loc10.b), !dbg !11
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.2dab6b4f72ef5ea6(ptr %implicit_padding.var), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.70db2d92ffddbef7:core.Destroy.Core"(ptr %self) #0 !dbg !13 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.39f44b5e2be36eae:core.Destroy.Core"(ptr %self) #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.bbaa45a6eaf1eb33:core.Destroy.Core"(ptr %self) #0 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.2dab6b4f72ef5ea6(ptr %self) #0 !dbg !31 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.bbaa45a6eaf1eb33:core.Destroy.Core"(ptr %self), !dbg !35
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "implicit_padding.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !1, line: 5, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 8, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 9, column: 6, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 9, column: 3, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 10, column: 6, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 10, column: 3, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 5, column: 1, scope: !4)
// CHECK:STDOUT: !13 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.70db2d92ffddbef7:core.Destroy.Core", scope: null, file: !1, line: 8, type: !14, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !17)
// CHECK:STDOUT: !14 = !DISubroutineType(types: !15)
// CHECK:STDOUT: !15 = !{null, !16}
// CHECK:STDOUT: !16 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !17 = !{!18}
// CHECK:STDOUT: !18 = !DILocalVariable(arg: 1, scope: !13, type: !16)
// CHECK:STDOUT: !19 = !DILocation(line: 8, column: 3, scope: !13)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.39f44b5e2be36eae:core.Destroy.Core", scope: null, file: !1, line: 8, type: !21, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !24)
// CHECK:STDOUT: !21 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !22 = !{null, !23}
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!25}
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !20, type: !23)
// CHECK:STDOUT: !26 = !DILocation(line: 8, column: 3, scope: !20)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.bbaa45a6eaf1eb33:core.Destroy.Core", scope: null, file: !1, line: 8, type: !21, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !28)
// CHECK:STDOUT: !28 = !{!29}
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 1, scope: !27, type: !23)
// CHECK:STDOUT: !30 = !DILocation(line: 8, column: 3, scope: !27)
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.2dab6b4f72ef5ea6", scope: null, file: !32, line: 24, type: !21, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !33)
// CHECK:STDOUT: !32 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !33 = !{!34}
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 1, scope: !31, type: !23)
// CHECK:STDOUT: !35 = !DILocation(line: 26, column: 5, scope: !31)
// CHECK:STDOUT: !36 = !DILocation(line: 24, column: 3, scope: !31)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'insert_padding.carbon'
// CHECK:STDOUT: source_filename = "insert_padding.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %insert_padding.var = alloca <{ <{ { i32, i32, i32 }, [4 x i8] }>, ptr, i32 }>, align 8, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %insert_padding.var), !dbg !7
// CHECK:STDOUT:   %.loc10.b = getelementptr inbounds nuw <{ <{ { i32, i32, i32 }, [4 x i8] }>, ptr, i32 }>, ptr %insert_padding.var, i32 0, i32 1, !dbg !8
// CHECK:STDOUT:   call void @_CG.Main(ptr %.loc10.b), !dbg !9
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.1eb6a0f3fc1f88f0(ptr %insert_padding.var), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.70db2d92ffddbef7:core.Destroy.Core"(ptr %self) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.ef264ddbcdaf6301:core.Destroy.Core"(ptr %self) #0 !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.97f42b497d28c70c:core.Destroy.Core"(ptr %self) #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.1eb6a0f3fc1f88f0(ptr %self) #0 !dbg !29 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.97f42b497d28c70c:core.Destroy.Core"(ptr %self), !dbg !33
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "insert_padding.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !1, line: 5, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 9, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 10, column: 6, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 10, column: 3, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 5, column: 1, scope: !4)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.70db2d92ffddbef7:core.Destroy.Core", scope: null, file: !1, line: 9, type: !12, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !15)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{null, !14}
// CHECK:STDOUT: !14 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !15 = !{!16}
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 1, scope: !11, type: !14)
// CHECK:STDOUT: !17 = !DILocation(line: 9, column: 3, scope: !11)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.ef264ddbcdaf6301:core.Destroy.Core", scope: null, file: !1, line: 9, type: !19, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !22)
// CHECK:STDOUT: !19 = !DISubroutineType(types: !20)
// CHECK:STDOUT: !20 = !{null, !21}
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!23}
// CHECK:STDOUT: !23 = !DILocalVariable(arg: 1, scope: !18, type: !21)
// CHECK:STDOUT: !24 = !DILocation(line: 9, column: 3, scope: !18)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.97f42b497d28c70c:core.Destroy.Core", scope: null, file: !1, line: 9, type: !19, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !26)
// CHECK:STDOUT: !26 = !{!27}
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !25, type: !21)
// CHECK:STDOUT: !28 = !DILocation(line: 9, column: 3, scope: !25)
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.1eb6a0f3fc1f88f0", scope: null, file: !30, line: 24, type: !19, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !31)
// CHECK:STDOUT: !30 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !31 = !{!32}
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !29, type: !21)
// CHECK:STDOUT: !33 = !DILocation(line: 26, column: 5, scope: !29)
// CHECK:STDOUT: !34 = !DILocation(line: 24, column: 3, scope: !29)
// CHECK:STDOUT:
