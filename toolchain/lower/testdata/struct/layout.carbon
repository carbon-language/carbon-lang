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
// CHECK:STDOUT:   call void @"_CSelfDestruct.cc12b768126bed5c:core.Destroy.Core"(ptr %no_padding.var), !dbg !29
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.cc12b768126bed5c:core.Destroy.Core"(ptr %self) #0 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_3.1.z = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 2, !dbg !47
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc7_3.1.z), !dbg !47
// CHECK:STDOUT:   %.loc7_3.2.y = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 1, !dbg !47
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc7_3.2.y), !dbg !47
// CHECK:STDOUT:   %.loc7_3.3.x = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 0, !dbg !47
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc7_3.3.x), !dbg !47
// CHECK:STDOUT:   ret void, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cc12b768126bed5c:core.Destroy.Core"(ptr %self) #0 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.cc12b768126bed5c:core.Destroy.Core"(ptr %self), !dbg !51
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
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
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !18, line: 7, type: !27, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !44)
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 1, scope: !41, type: !20)
// CHECK:STDOUT: !43 = !DILocation(line: 7, column: 3, scope: !41)
// CHECK:STDOUT: !44 = !{!42}
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.cc12b768126bed5c:core.Destroy.Core", scope: null, file: !18, line: 7, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !48)
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 1, scope: !45, type: !21)
// CHECK:STDOUT: !47 = !DILocation(line: 7, column: 3, scope: !45)
// CHECK:STDOUT: !48 = !{!46}
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cc12b768126bed5c:core.Destroy.Core", scope: null, file: !18, line: 7, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !52)
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 1, scope: !49, type: !21)
// CHECK:STDOUT: !51 = !DILocation(line: 7, column: 3, scope: !49)
// CHECK:STDOUT: !52 = !{!50}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'avoid_tail_padding.carbon'
// CHECK:STDOUT: source_filename = "avoid_tail_padding.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %avoid_tail_padding.var = alloca <{ ptr, i32 }>, align 8, !dbg !56
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %avoid_tail_padding.var), !dbg !56
// CHECK:STDOUT:   %.loc8.y = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %avoid_tail_padding.var, i32 0, i32 1, !dbg !58
// CHECK:STDOUT:   call void @_CG.Main(ptr %.loc8.y), !dbg !60
// CHECK:STDOUT:   call void @"_CSelfDestruct.90798694d9a177e6:core.Destroy.Core"(ptr %avoid_tail_padding.var), !dbg !56
// CHECK:STDOUT:   ret void, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.8436f795915f0543:core.Destroy.Core"(ptr %self) #0 !dbg !62 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !66 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.90798694d9a177e6:core.Destroy.Core"(ptr %self) #0 !dbg !70 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_3.1.y = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %self, i32 0, i32 1, !dbg !72
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc7_3.1.y), !dbg !72
// CHECK:STDOUT:   %.loc7_3.2.x = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %self, i32 0, i32 0, !dbg !72
// CHECK:STDOUT:   call void @"_CSelfDestruct.8436f795915f0543:core.Destroy.Core"(ptr %.loc7_3.2.x), !dbg !72
// CHECK:STDOUT:   ret void, !dbg !72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.90798694d9a177e6:core.Destroy.Core"(ptr %self) #0 !dbg !74 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.90798694d9a177e6:core.Destroy.Core"(ptr %self), !dbg !76
// CHECK:STDOUT:   ret void, !dbg !76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!54}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{null, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !{null}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{null, !20}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !53 = !DIFile(filename: "avoid_tail_padding.carbon", directory: "")
// CHECK:STDOUT: !54 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !53, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !53, line: 5, type: !25, spFlags: DISPFlagDefinition, unit: !54)
// CHECK:STDOUT: !56 = !DILocation(line: 7, column: 3, scope: !55)
// CHECK:STDOUT: !58 = !DILocation(line: 8, column: 6, scope: !55)
// CHECK:STDOUT: !60 = !DILocation(line: 8, column: 3, scope: !55)
// CHECK:STDOUT: !61 = !DILocation(line: 5, column: 1, scope: !55)
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.8436f795915f0543:core.Destroy.Core", scope: null, file: !53, line: 7, type: !23, spFlags: DISPFlagDefinition, unit: !54, retainedNodes: !65)
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !62, type: !21)
// CHECK:STDOUT: !64 = !DILocation(line: 7, column: 3, scope: !62)
// CHECK:STDOUT: !65 = !{!63}
// CHECK:STDOUT: !66 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !53, line: 7, type: !27, spFlags: DISPFlagDefinition, unit: !54, retainedNodes: !69)
// CHECK:STDOUT: !67 = !DILocalVariable(arg: 1, scope: !66, type: !20)
// CHECK:STDOUT: !68 = !DILocation(line: 7, column: 3, scope: !66)
// CHECK:STDOUT: !69 = !{!67}
// CHECK:STDOUT: !70 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.90798694d9a177e6:core.Destroy.Core", scope: null, file: !53, line: 7, type: !23, spFlags: DISPFlagDefinition, unit: !54, retainedNodes: !73)
// CHECK:STDOUT: !71 = !DILocalVariable(arg: 1, scope: !70, type: !21)
// CHECK:STDOUT: !72 = !DILocation(line: 7, column: 3, scope: !70)
// CHECK:STDOUT: !73 = !{!71}
// CHECK:STDOUT: !74 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.90798694d9a177e6:core.Destroy.Core", scope: null, file: !53, line: 7, type: !23, spFlags: DISPFlagDefinition, unit: !54, retainedNodes: !77)
// CHECK:STDOUT: !75 = !DILocalVariable(arg: 1, scope: !74, type: !21)
// CHECK:STDOUT: !76 = !DILocation(line: 7, column: 3, scope: !74)
// CHECK:STDOUT: !77 = !{!75}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'use_tail_padding.carbon'
// CHECK:STDOUT: source_filename = "use_tail_padding.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !80 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %use_tail_padding.var = alloca { <{ ptr, i32 }>, i32 }, align 8, !dbg !81
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %use_tail_padding.var), !dbg !81
// CHECK:STDOUT:   %.loc9_22.a = getelementptr inbounds nuw { <{ ptr, i32 }>, i32 }, ptr %use_tail_padding.var, i32 0, i32 0, !dbg !83
// CHECK:STDOUT:   %.loc9_24.y = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %.loc9_22.a, i32 0, i32 1, !dbg !83
// CHECK:STDOUT:   call void @_CG.Main(ptr %.loc9_24.y), !dbg !85
// CHECK:STDOUT:   %.loc10.b = getelementptr inbounds nuw { <{ ptr, i32 }>, i32 }, ptr %use_tail_padding.var, i32 0, i32 1, !dbg !86
// CHECK:STDOUT:   call void @_CG.Main(ptr %.loc10.b), !dbg !88
// CHECK:STDOUT:   call void @"_CSelfDestruct.3862e4ac11efcf1c:core.Destroy.Core"(ptr %use_tail_padding.var), !dbg !81
// CHECK:STDOUT:   ret void, !dbg !89
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.8436f795915f0543:core.Destroy.Core"(ptr %self) #0 !dbg !90 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !92
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !94 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !96
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.90798694d9a177e6:core.Destroy.Core"(ptr %self) #0 !dbg !98 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc8_3.1.y = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %self, i32 0, i32 1, !dbg !100
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc8_3.1.y), !dbg !100
// CHECK:STDOUT:   %.loc8_3.2.x = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %self, i32 0, i32 0, !dbg !100
// CHECK:STDOUT:   call void @"_CSelfDestruct.8436f795915f0543:core.Destroy.Core"(ptr %.loc8_3.2.x), !dbg !100
// CHECK:STDOUT:   ret void, !dbg !100
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.90798694d9a177e6:core.Destroy.Core"(ptr %self) #0 !dbg !102 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.90798694d9a177e6:core.Destroy.Core"(ptr %self), !dbg !104
// CHECK:STDOUT:   ret void, !dbg !104
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.3862e4ac11efcf1c:core.Destroy.Core"(ptr %self) #0 !dbg !106 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc8_3.1.b = getelementptr inbounds nuw { <{ ptr, i32 }>, i32 }, ptr %self, i32 0, i32 1, !dbg !108
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc8_3.1.b), !dbg !108
// CHECK:STDOUT:   %.loc8_3.2.a = getelementptr inbounds nuw { <{ ptr, i32 }>, i32 }, ptr %self, i32 0, i32 0, !dbg !108
// CHECK:STDOUT:   call void @"_CSelfDestruct.90798694d9a177e6:core.Destroy.Core"(ptr %.loc8_3.2.a), !dbg !108
// CHECK:STDOUT:   ret void, !dbg !108
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.3862e4ac11efcf1c:core.Destroy.Core"(ptr %self) #0 !dbg !110 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.3862e4ac11efcf1c:core.Destroy.Core"(ptr %self), !dbg !112
// CHECK:STDOUT:   ret void, !dbg !112
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!79}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{null, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !{null}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{null, !20}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !78 = !DIFile(filename: "use_tail_padding.carbon", directory: "")
// CHECK:STDOUT: !79 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !78, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !80 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !78, line: 5, type: !25, spFlags: DISPFlagDefinition, unit: !79)
// CHECK:STDOUT: !81 = !DILocation(line: 8, column: 3, scope: !80)
// CHECK:STDOUT: !83 = !DILocation(line: 9, column: 6, scope: !80)
// CHECK:STDOUT: !85 = !DILocation(line: 9, column: 3, scope: !80)
// CHECK:STDOUT: !86 = !DILocation(line: 10, column: 6, scope: !80)
// CHECK:STDOUT: !88 = !DILocation(line: 10, column: 3, scope: !80)
// CHECK:STDOUT: !89 = !DILocation(line: 5, column: 1, scope: !80)
// CHECK:STDOUT: !90 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.8436f795915f0543:core.Destroy.Core", scope: null, file: !78, line: 8, type: !23, spFlags: DISPFlagDefinition, unit: !79, retainedNodes: !93)
// CHECK:STDOUT: !91 = !DILocalVariable(arg: 1, scope: !90, type: !21)
// CHECK:STDOUT: !92 = !DILocation(line: 8, column: 3, scope: !90)
// CHECK:STDOUT: !93 = !{!91}
// CHECK:STDOUT: !94 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !78, line: 8, type: !27, spFlags: DISPFlagDefinition, unit: !79, retainedNodes: !97)
// CHECK:STDOUT: !95 = !DILocalVariable(arg: 1, scope: !94, type: !20)
// CHECK:STDOUT: !96 = !DILocation(line: 8, column: 3, scope: !94)
// CHECK:STDOUT: !97 = !{!95}
// CHECK:STDOUT: !98 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.90798694d9a177e6:core.Destroy.Core", scope: null, file: !78, line: 8, type: !23, spFlags: DISPFlagDefinition, unit: !79, retainedNodes: !101)
// CHECK:STDOUT: !99 = !DILocalVariable(arg: 1, scope: !98, type: !21)
// CHECK:STDOUT: !100 = !DILocation(line: 8, column: 3, scope: !98)
// CHECK:STDOUT: !101 = !{!99}
// CHECK:STDOUT: !102 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.90798694d9a177e6:core.Destroy.Core", scope: null, file: !78, line: 8, type: !23, spFlags: DISPFlagDefinition, unit: !79, retainedNodes: !105)
// CHECK:STDOUT: !103 = !DILocalVariable(arg: 1, scope: !102, type: !21)
// CHECK:STDOUT: !104 = !DILocation(line: 8, column: 3, scope: !102)
// CHECK:STDOUT: !105 = !{!103}
// CHECK:STDOUT: !106 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.3862e4ac11efcf1c:core.Destroy.Core", scope: null, file: !78, line: 8, type: !23, spFlags: DISPFlagDefinition, unit: !79, retainedNodes: !109)
// CHECK:STDOUT: !107 = !DILocalVariable(arg: 1, scope: !106, type: !21)
// CHECK:STDOUT: !108 = !DILocation(line: 8, column: 3, scope: !106)
// CHECK:STDOUT: !109 = !{!107}
// CHECK:STDOUT: !110 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.3862e4ac11efcf1c:core.Destroy.Core", scope: null, file: !78, line: 8, type: !23, spFlags: DISPFlagDefinition, unit: !79, retainedNodes: !113)
// CHECK:STDOUT: !111 = !DILocalVariable(arg: 1, scope: !110, type: !21)
// CHECK:STDOUT: !112 = !DILocation(line: 8, column: 3, scope: !110)
// CHECK:STDOUT: !113 = !{!111}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'implicit_padding.carbon'
// CHECK:STDOUT: source_filename = "implicit_padding.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !116 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %implicit_padding.var = alloca { <{ ptr, i32 }>, ptr }, align 8, !dbg !117
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %implicit_padding.var), !dbg !117
// CHECK:STDOUT:   %.loc9_22.a = getelementptr inbounds nuw { <{ ptr, i32 }>, ptr }, ptr %implicit_padding.var, i32 0, i32 0, !dbg !119
// CHECK:STDOUT:   %.loc9_24.x = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %.loc9_22.a, i32 0, i32 0, !dbg !119
// CHECK:STDOUT:   call void @_CG.Main(ptr %.loc9_24.x), !dbg !121
// CHECK:STDOUT:   %.loc10.b = getelementptr inbounds nuw { <{ ptr, i32 }>, ptr }, ptr %implicit_padding.var, i32 0, i32 1, !dbg !122
// CHECK:STDOUT:   call void @_CG.Main(ptr %.loc10.b), !dbg !124
// CHECK:STDOUT:   call void @"_CSelfDestruct.fb4d9eb13d71ce13:core.Destroy.Core"(ptr %implicit_padding.var), !dbg !117
// CHECK:STDOUT:   ret void, !dbg !125
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.8436f795915f0543:core.Destroy.Core"(ptr %self) #0 !dbg !126 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !128
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !130 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !132
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.90798694d9a177e6:core.Destroy.Core"(ptr %self) #0 !dbg !134 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc8_3.1.y = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %self, i32 0, i32 1, !dbg !136
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc8_3.1.y), !dbg !136
// CHECK:STDOUT:   %.loc8_3.2.x = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %self, i32 0, i32 0, !dbg !136
// CHECK:STDOUT:   call void @"_CSelfDestruct.8436f795915f0543:core.Destroy.Core"(ptr %.loc8_3.2.x), !dbg !136
// CHECK:STDOUT:   ret void, !dbg !136
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.90798694d9a177e6:core.Destroy.Core"(ptr %self) #0 !dbg !138 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.90798694d9a177e6:core.Destroy.Core"(ptr %self), !dbg !140
// CHECK:STDOUT:   ret void, !dbg !140
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.fb4d9eb13d71ce13:core.Destroy.Core"(ptr %self) #0 !dbg !142 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc8_3.1.b = getelementptr inbounds nuw { <{ ptr, i32 }>, ptr }, ptr %self, i32 0, i32 1, !dbg !144
// CHECK:STDOUT:   call void @"_CSelfDestruct.8436f795915f0543:core.Destroy.Core"(ptr %.loc8_3.1.b), !dbg !144
// CHECK:STDOUT:   %.loc8_3.2.a = getelementptr inbounds nuw { <{ ptr, i32 }>, ptr }, ptr %self, i32 0, i32 0, !dbg !144
// CHECK:STDOUT:   call void @"_CSelfDestruct.90798694d9a177e6:core.Destroy.Core"(ptr %.loc8_3.2.a), !dbg !144
// CHECK:STDOUT:   ret void, !dbg !144
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.fb4d9eb13d71ce13:core.Destroy.Core"(ptr %self) #0 !dbg !146 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.fb4d9eb13d71ce13:core.Destroy.Core"(ptr %self), !dbg !148
// CHECK:STDOUT:   ret void, !dbg !148
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!115}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{null, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !{null}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{null, !20}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !114 = !DIFile(filename: "implicit_padding.carbon", directory: "")
// CHECK:STDOUT: !115 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !114, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !116 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !114, line: 5, type: !25, spFlags: DISPFlagDefinition, unit: !115)
// CHECK:STDOUT: !117 = !DILocation(line: 8, column: 3, scope: !116)
// CHECK:STDOUT: !119 = !DILocation(line: 9, column: 6, scope: !116)
// CHECK:STDOUT: !121 = !DILocation(line: 9, column: 3, scope: !116)
// CHECK:STDOUT: !122 = !DILocation(line: 10, column: 6, scope: !116)
// CHECK:STDOUT: !124 = !DILocation(line: 10, column: 3, scope: !116)
// CHECK:STDOUT: !125 = !DILocation(line: 5, column: 1, scope: !116)
// CHECK:STDOUT: !126 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.8436f795915f0543:core.Destroy.Core", scope: null, file: !114, line: 8, type: !23, spFlags: DISPFlagDefinition, unit: !115, retainedNodes: !129)
// CHECK:STDOUT: !127 = !DILocalVariable(arg: 1, scope: !126, type: !21)
// CHECK:STDOUT: !128 = !DILocation(line: 8, column: 3, scope: !126)
// CHECK:STDOUT: !129 = !{!127}
// CHECK:STDOUT: !130 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !114, line: 8, type: !27, spFlags: DISPFlagDefinition, unit: !115, retainedNodes: !133)
// CHECK:STDOUT: !131 = !DILocalVariable(arg: 1, scope: !130, type: !20)
// CHECK:STDOUT: !132 = !DILocation(line: 8, column: 3, scope: !130)
// CHECK:STDOUT: !133 = !{!131}
// CHECK:STDOUT: !134 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.90798694d9a177e6:core.Destroy.Core", scope: null, file: !114, line: 8, type: !23, spFlags: DISPFlagDefinition, unit: !115, retainedNodes: !137)
// CHECK:STDOUT: !135 = !DILocalVariable(arg: 1, scope: !134, type: !21)
// CHECK:STDOUT: !136 = !DILocation(line: 8, column: 3, scope: !134)
// CHECK:STDOUT: !137 = !{!135}
// CHECK:STDOUT: !138 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.90798694d9a177e6:core.Destroy.Core", scope: null, file: !114, line: 8, type: !23, spFlags: DISPFlagDefinition, unit: !115, retainedNodes: !141)
// CHECK:STDOUT: !139 = !DILocalVariable(arg: 1, scope: !138, type: !21)
// CHECK:STDOUT: !140 = !DILocation(line: 8, column: 3, scope: !138)
// CHECK:STDOUT: !141 = !{!139}
// CHECK:STDOUT: !142 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.fb4d9eb13d71ce13:core.Destroy.Core", scope: null, file: !114, line: 8, type: !23, spFlags: DISPFlagDefinition, unit: !115, retainedNodes: !145)
// CHECK:STDOUT: !143 = !DILocalVariable(arg: 1, scope: !142, type: !21)
// CHECK:STDOUT: !144 = !DILocation(line: 8, column: 3, scope: !142)
// CHECK:STDOUT: !145 = !{!143}
// CHECK:STDOUT: !146 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.fb4d9eb13d71ce13:core.Destroy.Core", scope: null, file: !114, line: 8, type: !23, spFlags: DISPFlagDefinition, unit: !115, retainedNodes: !149)
// CHECK:STDOUT: !147 = !DILocalVariable(arg: 1, scope: !146, type: !21)
// CHECK:STDOUT: !148 = !DILocation(line: 8, column: 3, scope: !146)
// CHECK:STDOUT: !149 = !{!147}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'insert_padding.carbon'
// CHECK:STDOUT: source_filename = "insert_padding.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !152 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %insert_padding.var = alloca <{ <{ { i32, i32, i32 }, [4 x i8] }>, ptr, i32 }>, align 8, !dbg !153
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %insert_padding.var), !dbg !153
// CHECK:STDOUT:   %.loc10.b = getelementptr inbounds nuw <{ <{ { i32, i32, i32 }, [4 x i8] }>, ptr, i32 }>, ptr %insert_padding.var, i32 0, i32 1, !dbg !155
// CHECK:STDOUT:   call void @_CG.Main(ptr %.loc10.b), !dbg !157
// CHECK:STDOUT:   call void @"_CSelfDestruct.b496992173693e26:core.Destroy.Core"(ptr %insert_padding.var), !dbg !153
// CHECK:STDOUT:   ret void, !dbg !158
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !159 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !161
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.cc12b768126bed5c:core.Destroy.Core"(ptr %self) #0 !dbg !163 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_3.1.z = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 2, !dbg !165
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc9_3.1.z), !dbg !165
// CHECK:STDOUT:   %.loc9_3.2.y = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 1, !dbg !165
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc9_3.2.y), !dbg !165
// CHECK:STDOUT:   %.loc9_3.3.x = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 0, !dbg !165
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc9_3.3.x), !dbg !165
// CHECK:STDOUT:   ret void, !dbg !165
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cc12b768126bed5c:core.Destroy.Core"(ptr %self) #0 !dbg !167 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.cc12b768126bed5c:core.Destroy.Core"(ptr %self), !dbg !169
// CHECK:STDOUT:   ret void, !dbg !169
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.8436f795915f0543:core.Destroy.Core"(ptr %self) #0 !dbg !171 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !173
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.b496992173693e26:core.Destroy.Core"(ptr %self) #0 !dbg !175 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_3.1.c = getelementptr inbounds nuw <{ <{ { i32, i32, i32 }, [4 x i8] }>, ptr, i32 }>, ptr %self, i32 0, i32 2, !dbg !177
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %.loc9_3.1.c), !dbg !177
// CHECK:STDOUT:   %.loc9_3.2.b = getelementptr inbounds nuw <{ <{ { i32, i32, i32 }, [4 x i8] }>, ptr, i32 }>, ptr %self, i32 0, i32 1, !dbg !177
// CHECK:STDOUT:   call void @"_CSelfDestruct.8436f795915f0543:core.Destroy.Core"(ptr %.loc9_3.2.b), !dbg !177
// CHECK:STDOUT:   %.loc9_3.3.a = getelementptr inbounds nuw <{ <{ { i32, i32, i32 }, [4 x i8] }>, ptr, i32 }>, ptr %self, i32 0, i32 0, !dbg !177
// CHECK:STDOUT:   call void @"_CSelfDestruct.cc12b768126bed5c:core.Destroy.Core"(ptr %.loc9_3.3.a), !dbg !177
// CHECK:STDOUT:   ret void, !dbg !177
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.b496992173693e26:core.Destroy.Core"(ptr %self) #0 !dbg !179 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.b496992173693e26:core.Destroy.Core"(ptr %self), !dbg !181
// CHECK:STDOUT:   ret void, !dbg !181
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!151}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{null, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !{null}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{null, !20}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !150 = !DIFile(filename: "insert_padding.carbon", directory: "")
// CHECK:STDOUT: !151 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !150, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !152 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !150, line: 5, type: !25, spFlags: DISPFlagDefinition, unit: !151)
// CHECK:STDOUT: !153 = !DILocation(line: 9, column: 3, scope: !152)
// CHECK:STDOUT: !155 = !DILocation(line: 10, column: 6, scope: !152)
// CHECK:STDOUT: !157 = !DILocation(line: 10, column: 3, scope: !152)
// CHECK:STDOUT: !158 = !DILocation(line: 5, column: 1, scope: !152)
// CHECK:STDOUT: !159 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !150, line: 9, type: !27, spFlags: DISPFlagDefinition, unit: !151, retainedNodes: !162)
// CHECK:STDOUT: !160 = !DILocalVariable(arg: 1, scope: !159, type: !20)
// CHECK:STDOUT: !161 = !DILocation(line: 9, column: 3, scope: !159)
// CHECK:STDOUT: !162 = !{!160}
// CHECK:STDOUT: !163 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.cc12b768126bed5c:core.Destroy.Core", scope: null, file: !150, line: 9, type: !23, spFlags: DISPFlagDefinition, unit: !151, retainedNodes: !166)
// CHECK:STDOUT: !164 = !DILocalVariable(arg: 1, scope: !163, type: !21)
// CHECK:STDOUT: !165 = !DILocation(line: 9, column: 3, scope: !163)
// CHECK:STDOUT: !166 = !{!164}
// CHECK:STDOUT: !167 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cc12b768126bed5c:core.Destroy.Core", scope: null, file: !150, line: 9, type: !23, spFlags: DISPFlagDefinition, unit: !151, retainedNodes: !170)
// CHECK:STDOUT: !168 = !DILocalVariable(arg: 1, scope: !167, type: !21)
// CHECK:STDOUT: !169 = !DILocation(line: 9, column: 3, scope: !167)
// CHECK:STDOUT: !170 = !{!168}
// CHECK:STDOUT: !171 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.8436f795915f0543:core.Destroy.Core", scope: null, file: !150, line: 9, type: !23, spFlags: DISPFlagDefinition, unit: !151, retainedNodes: !174)
// CHECK:STDOUT: !172 = !DILocalVariable(arg: 1, scope: !171, type: !21)
// CHECK:STDOUT: !173 = !DILocation(line: 9, column: 3, scope: !171)
// CHECK:STDOUT: !174 = !{!172}
// CHECK:STDOUT: !175 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.b496992173693e26:core.Destroy.Core", scope: null, file: !150, line: 9, type: !23, spFlags: DISPFlagDefinition, unit: !151, retainedNodes: !178)
// CHECK:STDOUT: !176 = !DILocalVariable(arg: 1, scope: !175, type: !21)
// CHECK:STDOUT: !177 = !DILocation(line: 9, column: 3, scope: !175)
// CHECK:STDOUT: !178 = !{!176}
// CHECK:STDOUT: !179 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.b496992173693e26:core.Destroy.Core", scope: null, file: !150, line: 9, type: !23, spFlags: DISPFlagDefinition, unit: !151, retainedNodes: !182)
// CHECK:STDOUT: !180 = !DILocalVariable(arg: 1, scope: !179, type: !21)
// CHECK:STDOUT: !181 = !DILocation(line: 9, column: 3, scope: !179)
// CHECK:STDOUT: !182 = !{!180}
// CHECK:STDOUT:
