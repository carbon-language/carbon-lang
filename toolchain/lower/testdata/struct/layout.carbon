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
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b1f928fab120a252(ptr %no_padding.var), !dbg !27
// CHECK:STDOUT:   ret void, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.b1f928fab120a252(ptr %t) #0 !dbg !39 {
// CHECK:STDOUT:   ret void, !dbg !44
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
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !18 = !DIFile(filename: "no_padding.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
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
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.b1f928fab120a252", scope: null, file: !10, line: 30, type: !23, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !45)
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !39, type: !21)
// CHECK:STDOUT: !44 = !DILocation(line: 30, column: 1, scope: !39)
// CHECK:STDOUT: !45 = !{!40}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'avoid_tail_padding.carbon'
// CHECK:STDOUT: source_filename = "avoid_tail_padding.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %avoid_tail_padding.var = alloca <{ ptr, i32 }>, align 8, !dbg !49
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %avoid_tail_padding.var), !dbg !49
// CHECK:STDOUT:   %.loc8.y = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %avoid_tail_padding.var, i32 0, i32 1, !dbg !51
// CHECK:STDOUT:   call void @_CG.Main(ptr %.loc8.y), !dbg !53
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.daab1ff9fbee3990(ptr %avoid_tail_padding.var), !dbg !49
// CHECK:STDOUT:   ret void, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.daab1ff9fbee3990(ptr %t) #0 !dbg !55 {
// CHECK:STDOUT:   ret void, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!47}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{null, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !{null}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !46 = !DIFile(filename: "avoid_tail_padding.carbon", directory: "")
// CHECK:STDOUT: !47 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !46, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !46, line: 5, type: !25, spFlags: DISPFlagDefinition, unit: !47)
// CHECK:STDOUT: !49 = !DILocation(line: 7, column: 3, scope: !48)
// CHECK:STDOUT: !51 = !DILocation(line: 8, column: 6, scope: !48)
// CHECK:STDOUT: !53 = !DILocation(line: 8, column: 3, scope: !48)
// CHECK:STDOUT: !54 = !DILocation(line: 5, column: 1, scope: !48)
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.daab1ff9fbee3990", scope: null, file: !10, line: 30, type: !23, spFlags: DISPFlagDefinition, unit: !47, retainedNodes: !61)
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !55, type: !21)
// CHECK:STDOUT: !60 = !DILocation(line: 30, column: 1, scope: !55)
// CHECK:STDOUT: !61 = !{!56}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'use_tail_padding.carbon'
// CHECK:STDOUT: source_filename = "use_tail_padding.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !64 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %use_tail_padding.var = alloca { <{ ptr, i32 }>, i32 }, align 8, !dbg !65
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %use_tail_padding.var), !dbg !65
// CHECK:STDOUT:   %.loc9_22.a = getelementptr inbounds nuw { <{ ptr, i32 }>, i32 }, ptr %use_tail_padding.var, i32 0, i32 0, !dbg !67
// CHECK:STDOUT:   %.loc9_24.y = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %.loc9_22.a, i32 0, i32 1, !dbg !67
// CHECK:STDOUT:   call void @_CG.Main(ptr %.loc9_24.y), !dbg !69
// CHECK:STDOUT:   %.loc10.b = getelementptr inbounds nuw { <{ ptr, i32 }>, i32 }, ptr %use_tail_padding.var, i32 0, i32 1, !dbg !70
// CHECK:STDOUT:   call void @_CG.Main(ptr %.loc10.b), !dbg !72
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3935d231ededa6f4(ptr %use_tail_padding.var), !dbg !65
// CHECK:STDOUT:   ret void, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.3935d231ededa6f4(ptr %t) #0 !dbg !74 {
// CHECK:STDOUT:   ret void, !dbg !79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!63}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{null, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !{null}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !62 = !DIFile(filename: "use_tail_padding.carbon", directory: "")
// CHECK:STDOUT: !63 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !62, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !62, line: 5, type: !25, spFlags: DISPFlagDefinition, unit: !63)
// CHECK:STDOUT: !65 = !DILocation(line: 8, column: 3, scope: !64)
// CHECK:STDOUT: !67 = !DILocation(line: 9, column: 6, scope: !64)
// CHECK:STDOUT: !69 = !DILocation(line: 9, column: 3, scope: !64)
// CHECK:STDOUT: !70 = !DILocation(line: 10, column: 6, scope: !64)
// CHECK:STDOUT: !72 = !DILocation(line: 10, column: 3, scope: !64)
// CHECK:STDOUT: !73 = !DILocation(line: 5, column: 1, scope: !64)
// CHECK:STDOUT: !74 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.3935d231ededa6f4", scope: null, file: !10, line: 30, type: !23, spFlags: DISPFlagDefinition, unit: !63, retainedNodes: !80)
// CHECK:STDOUT: !75 = !DILocalVariable(arg: 1, scope: !74, type: !21)
// CHECK:STDOUT: !79 = !DILocation(line: 30, column: 1, scope: !74)
// CHECK:STDOUT: !80 = !{!75}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'implicit_padding.carbon'
// CHECK:STDOUT: source_filename = "implicit_padding.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !83 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %implicit_padding.var = alloca { <{ ptr, i32 }>, ptr }, align 8, !dbg !84
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %implicit_padding.var), !dbg !84
// CHECK:STDOUT:   %.loc9_22.a = getelementptr inbounds nuw { <{ ptr, i32 }>, ptr }, ptr %implicit_padding.var, i32 0, i32 0, !dbg !86
// CHECK:STDOUT:   %.loc9_24.x = getelementptr inbounds nuw <{ ptr, i32 }>, ptr %.loc9_22.a, i32 0, i32 0, !dbg !86
// CHECK:STDOUT:   call void @_CG.Main(ptr %.loc9_24.x), !dbg !88
// CHECK:STDOUT:   %.loc10.b = getelementptr inbounds nuw { <{ ptr, i32 }>, ptr }, ptr %implicit_padding.var, i32 0, i32 1, !dbg !89
// CHECK:STDOUT:   call void @_CG.Main(ptr %.loc10.b), !dbg !91
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.40acc59ba774dea6(ptr %implicit_padding.var), !dbg !84
// CHECK:STDOUT:   ret void, !dbg !92
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.40acc59ba774dea6(ptr %t) #0 !dbg !93 {
// CHECK:STDOUT:   ret void, !dbg !98
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!82}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{null, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !{null}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !81 = !DIFile(filename: "implicit_padding.carbon", directory: "")
// CHECK:STDOUT: !82 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !81, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !83 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !81, line: 5, type: !25, spFlags: DISPFlagDefinition, unit: !82)
// CHECK:STDOUT: !84 = !DILocation(line: 8, column: 3, scope: !83)
// CHECK:STDOUT: !86 = !DILocation(line: 9, column: 6, scope: !83)
// CHECK:STDOUT: !88 = !DILocation(line: 9, column: 3, scope: !83)
// CHECK:STDOUT: !89 = !DILocation(line: 10, column: 6, scope: !83)
// CHECK:STDOUT: !91 = !DILocation(line: 10, column: 3, scope: !83)
// CHECK:STDOUT: !92 = !DILocation(line: 5, column: 1, scope: !83)
// CHECK:STDOUT: !93 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.40acc59ba774dea6", scope: null, file: !10, line: 30, type: !23, spFlags: DISPFlagDefinition, unit: !82, retainedNodes: !99)
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 1, scope: !93, type: !21)
// CHECK:STDOUT: !98 = !DILocation(line: 30, column: 1, scope: !93)
// CHECK:STDOUT: !99 = !{!94}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'insert_padding.carbon'
// CHECK:STDOUT: source_filename = "insert_padding.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !102 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %insert_padding.var = alloca <{ <{ { i32, i32, i32 }, [4 x i8] }>, ptr, i32 }>, align 8, !dbg !103
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %insert_padding.var), !dbg !103
// CHECK:STDOUT:   %.loc10.b = getelementptr inbounds nuw <{ <{ { i32, i32, i32 }, [4 x i8] }>, ptr, i32 }>, ptr %insert_padding.var, i32 0, i32 1, !dbg !105
// CHECK:STDOUT:   call void @_CG.Main(ptr %.loc10.b), !dbg !107
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3e2081f51c152a5d(ptr %insert_padding.var), !dbg !103
// CHECK:STDOUT:   ret void, !dbg !108
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.3e2081f51c152a5d(ptr %t) #0 !dbg !109 {
// CHECK:STDOUT:   ret void, !dbg !114
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!101}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{null, !21}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = !{null}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !100 = !DIFile(filename: "insert_padding.carbon", directory: "")
// CHECK:STDOUT: !101 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !100, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !102 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !100, line: 5, type: !25, spFlags: DISPFlagDefinition, unit: !101)
// CHECK:STDOUT: !103 = !DILocation(line: 9, column: 3, scope: !102)
// CHECK:STDOUT: !105 = !DILocation(line: 10, column: 6, scope: !102)
// CHECK:STDOUT: !107 = !DILocation(line: 10, column: 3, scope: !102)
// CHECK:STDOUT: !108 = !DILocation(line: 5, column: 1, scope: !102)
// CHECK:STDOUT: !109 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.3e2081f51c152a5d", scope: null, file: !10, line: 30, type: !23, spFlags: DISPFlagDefinition, unit: !101, retainedNodes: !115)
// CHECK:STDOUT: !110 = !DILocalVariable(arg: 1, scope: !109, type: !21)
// CHECK:STDOUT: !114 = !DILocation(line: 30, column: 1, scope: !109)
// CHECK:STDOUT: !115 = !{!110}
// CHECK:STDOUT:
