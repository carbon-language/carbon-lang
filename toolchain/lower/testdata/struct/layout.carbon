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
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %no_padding.var = alloca { i32, i32, i32 }, align 4, !dbg !29
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %no_padding.var), !dbg !29
// CHECK:STDOUT:   %.loc8.x = getelementptr inbounds nuw { i32, i32, i32 }, ptr %no_padding.var, i32 0, i32 0, !dbg !31
// CHECK:STDOUT:   call void @_CG.Main(ptr %.loc8.x), !dbg !33
// CHECK:STDOUT:   %.loc9.y = getelementptr inbounds nuw { i32, i32, i32 }, ptr %no_padding.var, i32 0, i32 1, !dbg !34
// CHECK:STDOUT:   call void @_CG.Main(ptr %.loc9.y), !dbg !36
// CHECK:STDOUT:   %.loc10.z = getelementptr inbounds nuw { i32, i32, i32 }, ptr %no_padding.var, i32 0, i32 2, !dbg !37
// CHECK:STDOUT:   call void @_CG.Main(ptr %.loc10.z), !dbg !39
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.a876613cdec73e2b(ptr %no_padding.var), !dbg !29
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.cc12b768126bed5c:core.Destroy.Core"(ptr %self) #0 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.a876613cdec73e2b(ptr %self) #0 !dbg !49 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.cc12b768126bed5c:core.Destroy.Core"(ptr %self), !dbg !53
// CHECK:STDOUT:   ret void, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !18 = !DIFile(filename: "no_padding.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{null, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !{null}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{null, !20}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !18, line: 5, type: !25, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !29 = !DILocation(line: 7, column: 3, scope: !28)
// CHECK:STDOUT: !31 = !DILocation(line: 8, column: 6, scope: !28)
// CHECK:STDOUT: !33 = !DILocation(line: 8, column: 3, scope: !28)
// CHECK:STDOUT: !34 = !DILocation(line: 9, column: 6, scope: !28)
// CHECK:STDOUT: !36 = !DILocation(line: 9, column: 3, scope: !28)
// CHECK:STDOUT: !37 = !DILocation(line: 10, column: 6, scope: !28)
// CHECK:STDOUT: !39 = !DILocation(line: 10, column: 3, scope: !28)
// CHECK:STDOUT: !40 = !DILocation(line: 5, column: 1, scope: !28)
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !18, line: 7, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !44)
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 1, scope: !41, type: !20)
// CHECK:STDOUT: !43 = !DILocation(line: 7, column: 3, scope: !41)
// CHECK:STDOUT: !44 = !{!42}
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.cc12b768126bed5c:core.Destroy.Core", scope: null, file: !18, line: 7, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !48)
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 1, scope: !45, type: !21)
// CHECK:STDOUT: !47 = !DILocation(line: 7, column: 3, scope: !45)
// CHECK:STDOUT: !48 = !{!46}
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.a876613cdec73e2b", scope: null, file: !14, line: 24, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !55)
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 1, scope: !49, type: !21)
// CHECK:STDOUT: !53 = !DILocation(line: 26, column: 5, scope: !49)
// CHECK:STDOUT: !54 = !DILocation(line: 24, column: 3, scope: !49)
// CHECK:STDOUT: !55 = !{!50}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'avoid_tail_padding.carbon'
// CHECK:STDOUT: source_filename = "avoid_tail_padding.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !58 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %avoid_tail_padding.var = alloca <{ ptr, i32 }>, align 8, !dbg !59
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %avoid_tail_padding.var), !dbg !59
// CHECK:STDOUT:   %.loc8.y = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %avoid_tail_padding.var, i32 0, i32 1, !dbg !61
// CHECK:STDOUT:   call void @_CG.Main(ptr %.loc8.y), !dbg !63
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.f58fd6d7a2f83df4(ptr %avoid_tail_padding.var), !dbg !59
// CHECK:STDOUT:   ret void, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !65 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.90798694d9a177e6:core.Destroy.Core"(ptr %self) #0 !dbg !69 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.f58fd6d7a2f83df4(ptr %self) #0 !dbg !73 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.90798694d9a177e6:core.Destroy.Core"(ptr %self), !dbg !77
// CHECK:STDOUT:   ret void, !dbg !78
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!57}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{null, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !{null}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{null, !20}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !56 = !DIFile(filename: "avoid_tail_padding.carbon", directory: "")
// CHECK:STDOUT: !57 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !56, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !56, line: 5, type: !25, spFlags: DISPFlagDefinition, unit: !57)
// CHECK:STDOUT: !59 = !DILocation(line: 7, column: 3, scope: !58)
// CHECK:STDOUT: !61 = !DILocation(line: 8, column: 6, scope: !58)
// CHECK:STDOUT: !63 = !DILocation(line: 8, column: 3, scope: !58)
// CHECK:STDOUT: !64 = !DILocation(line: 5, column: 1, scope: !58)
// CHECK:STDOUT: !65 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !56, line: 7, type: !27, spFlags: DISPFlagDefinition, unit: !57, retainedNodes: !68)
// CHECK:STDOUT: !66 = !DILocalVariable(arg: 1, scope: !65, type: !20)
// CHECK:STDOUT: !67 = !DILocation(line: 7, column: 3, scope: !65)
// CHECK:STDOUT: !68 = !{!66}
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.90798694d9a177e6:core.Destroy.Core", scope: null, file: !56, line: 7, type: !23, spFlags: DISPFlagDefinition, unit: !57, retainedNodes: !72)
// CHECK:STDOUT: !70 = !DILocalVariable(arg: 1, scope: !69, type: !21)
// CHECK:STDOUT: !71 = !DILocation(line: 7, column: 3, scope: !69)
// CHECK:STDOUT: !72 = !{!70}
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.f58fd6d7a2f83df4", scope: null, file: !14, line: 24, type: !23, spFlags: DISPFlagDefinition, unit: !57, retainedNodes: !79)
// CHECK:STDOUT: !74 = !DILocalVariable(arg: 1, scope: !73, type: !21)
// CHECK:STDOUT: !77 = !DILocation(line: 26, column: 5, scope: !73)
// CHECK:STDOUT: !78 = !DILocation(line: 24, column: 3, scope: !73)
// CHECK:STDOUT: !79 = !{!74}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'use_tail_padding.carbon'
// CHECK:STDOUT: source_filename = "use_tail_padding.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !82 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %use_tail_padding.var = alloca { <{ ptr, i32 }>, i32 }, align 8, !dbg !83
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %use_tail_padding.var), !dbg !83
// CHECK:STDOUT:   %.loc9_22.a = getelementptr inbounds nuw { <{ ptr, i32 }>, i32 }, ptr %use_tail_padding.var, i32 0, i32 0, !dbg !85
// CHECK:STDOUT:   %.loc9_24.y = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %.loc9_22.a, i32 0, i32 1, !dbg !85
// CHECK:STDOUT:   call void @_CG.Main(ptr %.loc9_24.y), !dbg !87
// CHECK:STDOUT:   %.loc10.b = getelementptr inbounds nuw { <{ ptr, i32 }>, i32 }, ptr %use_tail_padding.var, i32 0, i32 1, !dbg !88
// CHECK:STDOUT:   call void @_CG.Main(ptr %.loc10.b), !dbg !90
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.608682cf29acc5fb(ptr %use_tail_padding.var), !dbg !83
// CHECK:STDOUT:   ret void, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !92 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !94
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.90798694d9a177e6:core.Destroy.Core"(ptr %self) #0 !dbg !96 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !98
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.3862e4ac11efcf1c:core.Destroy.Core"(ptr %self) #0 !dbg !100 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !102
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.608682cf29acc5fb(ptr %self) #0 !dbg !104 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.3862e4ac11efcf1c:core.Destroy.Core"(ptr %self), !dbg !108
// CHECK:STDOUT:   ret void, !dbg !109
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!81}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{null, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !{null}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{null, !20}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !80 = !DIFile(filename: "use_tail_padding.carbon", directory: "")
// CHECK:STDOUT: !81 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !80, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !82 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !80, line: 5, type: !25, spFlags: DISPFlagDefinition, unit: !81)
// CHECK:STDOUT: !83 = !DILocation(line: 8, column: 3, scope: !82)
// CHECK:STDOUT: !85 = !DILocation(line: 9, column: 6, scope: !82)
// CHECK:STDOUT: !87 = !DILocation(line: 9, column: 3, scope: !82)
// CHECK:STDOUT: !88 = !DILocation(line: 10, column: 6, scope: !82)
// CHECK:STDOUT: !90 = !DILocation(line: 10, column: 3, scope: !82)
// CHECK:STDOUT: !91 = !DILocation(line: 5, column: 1, scope: !82)
// CHECK:STDOUT: !92 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !80, line: 8, type: !27, spFlags: DISPFlagDefinition, unit: !81, retainedNodes: !95)
// CHECK:STDOUT: !93 = !DILocalVariable(arg: 1, scope: !92, type: !20)
// CHECK:STDOUT: !94 = !DILocation(line: 8, column: 3, scope: !92)
// CHECK:STDOUT: !95 = !{!93}
// CHECK:STDOUT: !96 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.90798694d9a177e6:core.Destroy.Core", scope: null, file: !80, line: 8, type: !23, spFlags: DISPFlagDefinition, unit: !81, retainedNodes: !99)
// CHECK:STDOUT: !97 = !DILocalVariable(arg: 1, scope: !96, type: !21)
// CHECK:STDOUT: !98 = !DILocation(line: 8, column: 3, scope: !96)
// CHECK:STDOUT: !99 = !{!97}
// CHECK:STDOUT: !100 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.3862e4ac11efcf1c:core.Destroy.Core", scope: null, file: !80, line: 8, type: !23, spFlags: DISPFlagDefinition, unit: !81, retainedNodes: !103)
// CHECK:STDOUT: !101 = !DILocalVariable(arg: 1, scope: !100, type: !21)
// CHECK:STDOUT: !102 = !DILocation(line: 8, column: 3, scope: !100)
// CHECK:STDOUT: !103 = !{!101}
// CHECK:STDOUT: !104 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.608682cf29acc5fb", scope: null, file: !14, line: 24, type: !23, spFlags: DISPFlagDefinition, unit: !81, retainedNodes: !110)
// CHECK:STDOUT: !105 = !DILocalVariable(arg: 1, scope: !104, type: !21)
// CHECK:STDOUT: !108 = !DILocation(line: 26, column: 5, scope: !104)
// CHECK:STDOUT: !109 = !DILocation(line: 24, column: 3, scope: !104)
// CHECK:STDOUT: !110 = !{!105}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'implicit_padding.carbon'
// CHECK:STDOUT: source_filename = "implicit_padding.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !113 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %implicit_padding.var = alloca { <{ ptr, i32 }>, ptr }, align 8, !dbg !114
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %implicit_padding.var), !dbg !114
// CHECK:STDOUT:   %.loc9_22.a = getelementptr inbounds nuw { <{ ptr, i32 }>, ptr }, ptr %implicit_padding.var, i32 0, i32 0, !dbg !116
// CHECK:STDOUT:   %.loc9_24.x = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %.loc9_22.a, i32 0, i32 0, !dbg !116
// CHECK:STDOUT:   call void @_CG.Main(ptr %.loc9_24.x), !dbg !118
// CHECK:STDOUT:   %.loc10.b = getelementptr inbounds nuw { <{ ptr, i32 }>, ptr }, ptr %implicit_padding.var, i32 0, i32 1, !dbg !119
// CHECK:STDOUT:   call void @_CG.Main(ptr %.loc10.b), !dbg !121
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.2dab6b4f72ef5ea6(ptr %implicit_padding.var), !dbg !114
// CHECK:STDOUT:   ret void, !dbg !122
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !123 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !125
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.90798694d9a177e6:core.Destroy.Core"(ptr %self) #0 !dbg !127 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !129
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.fb4d9eb13d71ce13:core.Destroy.Core"(ptr %self) #0 !dbg !131 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !133
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.2dab6b4f72ef5ea6(ptr %self) #0 !dbg !135 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.fb4d9eb13d71ce13:core.Destroy.Core"(ptr %self), !dbg !139
// CHECK:STDOUT:   ret void, !dbg !140
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!112}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{null, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !{null}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{null, !20}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !111 = !DIFile(filename: "implicit_padding.carbon", directory: "")
// CHECK:STDOUT: !112 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !111, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !113 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !111, line: 5, type: !25, spFlags: DISPFlagDefinition, unit: !112)
// CHECK:STDOUT: !114 = !DILocation(line: 8, column: 3, scope: !113)
// CHECK:STDOUT: !116 = !DILocation(line: 9, column: 6, scope: !113)
// CHECK:STDOUT: !118 = !DILocation(line: 9, column: 3, scope: !113)
// CHECK:STDOUT: !119 = !DILocation(line: 10, column: 6, scope: !113)
// CHECK:STDOUT: !121 = !DILocation(line: 10, column: 3, scope: !113)
// CHECK:STDOUT: !122 = !DILocation(line: 5, column: 1, scope: !113)
// CHECK:STDOUT: !123 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !111, line: 8, type: !27, spFlags: DISPFlagDefinition, unit: !112, retainedNodes: !126)
// CHECK:STDOUT: !124 = !DILocalVariable(arg: 1, scope: !123, type: !20)
// CHECK:STDOUT: !125 = !DILocation(line: 8, column: 3, scope: !123)
// CHECK:STDOUT: !126 = !{!124}
// CHECK:STDOUT: !127 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.90798694d9a177e6:core.Destroy.Core", scope: null, file: !111, line: 8, type: !23, spFlags: DISPFlagDefinition, unit: !112, retainedNodes: !130)
// CHECK:STDOUT: !128 = !DILocalVariable(arg: 1, scope: !127, type: !21)
// CHECK:STDOUT: !129 = !DILocation(line: 8, column: 3, scope: !127)
// CHECK:STDOUT: !130 = !{!128}
// CHECK:STDOUT: !131 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.fb4d9eb13d71ce13:core.Destroy.Core", scope: null, file: !111, line: 8, type: !23, spFlags: DISPFlagDefinition, unit: !112, retainedNodes: !134)
// CHECK:STDOUT: !132 = !DILocalVariable(arg: 1, scope: !131, type: !21)
// CHECK:STDOUT: !133 = !DILocation(line: 8, column: 3, scope: !131)
// CHECK:STDOUT: !134 = !{!132}
// CHECK:STDOUT: !135 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.2dab6b4f72ef5ea6", scope: null, file: !14, line: 24, type: !23, spFlags: DISPFlagDefinition, unit: !112, retainedNodes: !141)
// CHECK:STDOUT: !136 = !DILocalVariable(arg: 1, scope: !135, type: !21)
// CHECK:STDOUT: !139 = !DILocation(line: 26, column: 5, scope: !135)
// CHECK:STDOUT: !140 = !DILocation(line: 24, column: 3, scope: !135)
// CHECK:STDOUT: !141 = !{!136}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'insert_padding.carbon'
// CHECK:STDOUT: source_filename = "insert_padding.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !144 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %insert_padding.var = alloca <{ <{ { i32, i32, i32 }, [4 x i8] }>, ptr, i32 }>, align 8, !dbg !145
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %insert_padding.var), !dbg !145
// CHECK:STDOUT:   %.loc10.b = getelementptr inbounds nuw <{ <{ { i32, i32, i32 }, [4 x i8] }>, ptr, i32 }>, ptr %insert_padding.var, i32 0, i32 1, !dbg !147
// CHECK:STDOUT:   call void @_CG.Main(ptr %.loc10.b), !dbg !149
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.1eb6a0f3fc1f88f0(ptr %insert_padding.var), !dbg !145
// CHECK:STDOUT:   ret void, !dbg !150
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !151 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !153
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.cc12b768126bed5c:core.Destroy.Core"(ptr %self) #0 !dbg !155 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !157
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.b496992173693e26:core.Destroy.Core"(ptr %self) #0 !dbg !159 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !161
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.1eb6a0f3fc1f88f0(ptr %self) #0 !dbg !163 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.b496992173693e26:core.Destroy.Core"(ptr %self), !dbg !167
// CHECK:STDOUT:   ret void, !dbg !168
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!143}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{null, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !{null}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{null, !20}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !142 = !DIFile(filename: "insert_padding.carbon", directory: "")
// CHECK:STDOUT: !143 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !142, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !144 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !142, line: 5, type: !25, spFlags: DISPFlagDefinition, unit: !143)
// CHECK:STDOUT: !145 = !DILocation(line: 9, column: 3, scope: !144)
// CHECK:STDOUT: !147 = !DILocation(line: 10, column: 6, scope: !144)
// CHECK:STDOUT: !149 = !DILocation(line: 10, column: 3, scope: !144)
// CHECK:STDOUT: !150 = !DILocation(line: 5, column: 1, scope: !144)
// CHECK:STDOUT: !151 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !142, line: 9, type: !27, spFlags: DISPFlagDefinition, unit: !143, retainedNodes: !154)
// CHECK:STDOUT: !152 = !DILocalVariable(arg: 1, scope: !151, type: !20)
// CHECK:STDOUT: !153 = !DILocation(line: 9, column: 3, scope: !151)
// CHECK:STDOUT: !154 = !{!152}
// CHECK:STDOUT: !155 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.cc12b768126bed5c:core.Destroy.Core", scope: null, file: !142, line: 9, type: !23, spFlags: DISPFlagDefinition, unit: !143, retainedNodes: !158)
// CHECK:STDOUT: !156 = !DILocalVariable(arg: 1, scope: !155, type: !21)
// CHECK:STDOUT: !157 = !DILocation(line: 9, column: 3, scope: !155)
// CHECK:STDOUT: !158 = !{!156}
// CHECK:STDOUT: !159 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.b496992173693e26:core.Destroy.Core", scope: null, file: !142, line: 9, type: !23, spFlags: DISPFlagDefinition, unit: !143, retainedNodes: !162)
// CHECK:STDOUT: !160 = !DILocalVariable(arg: 1, scope: !159, type: !21)
// CHECK:STDOUT: !161 = !DILocation(line: 9, column: 3, scope: !159)
// CHECK:STDOUT: !162 = !{!160}
// CHECK:STDOUT: !163 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.1eb6a0f3fc1f88f0", scope: null, file: !14, line: 24, type: !23, spFlags: DISPFlagDefinition, unit: !143, retainedNodes: !169)
// CHECK:STDOUT: !164 = !DILocalVariable(arg: 1, scope: !163, type: !21)
// CHECK:STDOUT: !167 = !DILocation(line: 26, column: 5, scope: !163)
// CHECK:STDOUT: !168 = !DILocation(line: 24, column: 3, scope: !163)
// CHECK:STDOUT: !169 = !{!164}
// CHECK:STDOUT:
