// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This is an integration test for the Core.Slice in the prelude, so use the
// real prelude.
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
// EXTRA-ARGS: --target=x86_64-linux-gnu
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/primitives/slice.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/primitives/slice.carbon

// --- make.carbon
library "[[@TEST_NAME]]";

fn MakeSlice(p: i32*, n: u64) -> Core.Slice(i32) {
  return Core.Slice(i32).Make(p, n);
}

fn GetSize(s: Core.Slice(i32)) -> u64 {
  return s.Size();
}

// --- copy.carbon
library "[[@TEST_NAME]]";

fn CopySlice(s: Core.Slice(f64)) -> Core.Slice(f64) {
  return s;
}

// --- unformed.carbon
library "[[@TEST_NAME]]";

fn Unformed(p: i32*, n: u64) -> Core.Slice(i32) {
  var s: Core.Slice(i32);
  s = Core.Slice(i32).Make(p, n);
  return s;
}

// --- iterate.carbon
library "[[@TEST_NAME]]";

fn IterateSlice(s: Core.Slice(i32)) -> i32 {
  var sum: i32 = 0;
  for (x: i32 in s) {
    sum += x;
  }
  return sum;
}


// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'make.carbon'
// CHECK:STDOUT: source_filename = "make.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakeSlice.Main(ptr sret({ ptr, i64 }) %return, ptr %p, i64 %n) #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CMake.Slice.Core.55456748a8a2eb70(ptr %return, ptr %p, i64 %n), !dbg !21
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CGetSize.Main(ptr %s) #0 !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Slice.Size.call = call i64 @_CSize.Slice.Core.55456748a8a2eb70(ptr %s), !dbg !29
// CHECK:STDOUT:   ret i64 %Slice.Size.call, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CMake.Slice.Core.55456748a8a2eb70(ptr sret({ ptr, i64 }) %return, ptr %ptr, i64 %size) #0 !dbg !34 {
// CHECK:STDOUT:   %ptr1 = getelementptr inbounds nuw { ptr, i64 }, ptr %return, i32 0, i32 0, !dbg !43
// CHECK:STDOUT:   store ptr %ptr, ptr %ptr1, align 8, !dbg !43
// CHECK:STDOUT:   %size2 = getelementptr inbounds nuw { ptr, i64 }, ptr %return, i32 0, i32 1, !dbg !43
// CHECK:STDOUT:   store i64 %size, ptr %size2, align 8, !dbg !43
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i64 @_CSize.Slice.Core.55456748a8a2eb70(ptr %self) #0 !dbg !46 {
// CHECK:STDOUT:   %size = getelementptr inbounds nuw { ptr, i64 }, ptr %self, i32 0, i32 1, !dbg !51
// CHECK:STDOUT:   %1 = load i64, ptr %size, align 8, !dbg !51
// CHECK:STDOUT:   ret i64 %1, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "make.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 64, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!7, !7, !6}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !10 = !{!6, !7}
// CHECK:STDOUT: !11 = !DISubroutineType(types: !10)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "MakeSlice", linkageName: "_CMakeSlice.Main", scope: null, file: !4, line: 3, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !23)
// CHECK:STDOUT: !13 = !DILocalVariable(arg: 1, scope: !12, type: !7)
// CHECK:STDOUT: !14 = !DILocalVariable(arg: 2, scope: !12, type: !6)
// CHECK:STDOUT: !21 = !DILocation(line: 4, column: 10, scope: !12)
// CHECK:STDOUT: !22 = !DILocation(line: 4, column: 3, scope: !12)
// CHECK:STDOUT: !23 = !{!13, !14}
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "GetSize", linkageName: "_CGetSize.Main", scope: null, file: !4, line: 7, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !31)
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !24, type: !7)
// CHECK:STDOUT: !29 = !DILocation(line: 8, column: 10, scope: !24)
// CHECK:STDOUT: !30 = !DILocation(line: 8, column: 3, scope: !24)
// CHECK:STDOUT: !31 = !{!25}
// CHECK:STDOUT: !33 = !DIFile(filename: "{{.*}}/prelude/types/slice.carbon", directory: "")
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Slice.Core.55456748a8a2eb70", scope: null, file: !33, line: 21, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !45)
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !34, type: !7)
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 2, scope: !34, type: !6)
// CHECK:STDOUT: !43 = !DILocation(line: 22, column: 12, scope: !34)
// CHECK:STDOUT: !44 = !DILocation(line: 22, column: 5, scope: !34)
// CHECK:STDOUT: !45 = !{!35, !36}
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "Size", linkageName: "_CSize.Slice.Core.55456748a8a2eb70", scope: null, file: !33, line: 25, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !53)
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !46, type: !7)
// CHECK:STDOUT: !51 = !DILocation(line: 25, column: 33, scope: !46)
// CHECK:STDOUT: !52 = !DILocation(line: 25, column: 26, scope: !46)
// CHECK:STDOUT: !53 = !{!47}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'copy.carbon'
// CHECK:STDOUT: source_filename = "copy.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCopySlice.Main(ptr sret({ ptr, i64 }) %return, ptr %s) #0 !dbg !58 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.Slice.eb057aa32837c84e.Core:Copy.Core.b709b3b9ccbae883"(ptr %return, ptr %s), !dbg !63
// CHECK:STDOUT:   ret void, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.Slice.eb057aa32837c84e.Core:Copy.Core.b709b3b9ccbae883"(ptr sret({ ptr, i64 }) %return, ptr %self) #0 !dbg !66 {
// CHECK:STDOUT:   %ptr = getelementptr inbounds nuw { ptr, i64 }, ptr %self, i32 0, i32 0, !dbg !71
// CHECK:STDOUT:   %1 = load ptr, ptr %ptr, align 8, !dbg !71
// CHECK:STDOUT:   %size = getelementptr inbounds nuw { ptr, i64 }, ptr %self, i32 0, i32 1, !dbg !72
// CHECK:STDOUT:   %2 = load i64, ptr %size, align 8, !dbg !72
// CHECK:STDOUT:   %ptr1 = getelementptr inbounds nuw { ptr, i64 }, ptr %return, i32 0, i32 0, !dbg !73
// CHECK:STDOUT:   store ptr %1, ptr %ptr1, align 8, !dbg !73
// CHECK:STDOUT:   %size2 = getelementptr inbounds nuw { ptr, i64 }, ptr %return, i32 0, i32 1, !dbg !73
// CHECK:STDOUT:   store i64 %2, ptr %size2, align 8, !dbg !73
// CHECK:STDOUT:   ret void, !dbg !74
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!55}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !33 = !DIFile(filename: "{{.*}}/prelude/types/slice.carbon", directory: "")
// CHECK:STDOUT: !54 = !DIFile(filename: "copy.carbon", directory: "")
// CHECK:STDOUT: !55 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !54, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !56 = !{!7, !7}
// CHECK:STDOUT: !57 = !DISubroutineType(types: !56)
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "CopySlice", linkageName: "_CCopySlice.Main", scope: null, file: !54, line: 3, type: !57, spFlags: DISPFlagDefinition, unit: !55, retainedNodes: !65)
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !58, type: !7)
// CHECK:STDOUT: !63 = !DILocation(line: 4, column: 10, scope: !58)
// CHECK:STDOUT: !64 = !DILocation(line: 4, column: 3, scope: !58)
// CHECK:STDOUT: !65 = !{!59}
// CHECK:STDOUT: !66 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Slice.eb057aa32837c84e.Core:Copy.Core.b709b3b9ccbae883", scope: null, file: !33, line: 33, type: !57, spFlags: DISPFlagDefinition, unit: !55, retainedNodes: !75)
// CHECK:STDOUT: !67 = !DILocalVariable(arg: 1, scope: !66, type: !7)
// CHECK:STDOUT: !71 = !DILocation(line: 33, column: 42, scope: !66)
// CHECK:STDOUT: !72 = !DILocation(line: 33, column: 60, scope: !66)
// CHECK:STDOUT: !73 = !DILocation(line: 33, column: 34, scope: !66)
// CHECK:STDOUT: !74 = !DILocation(line: 33, column: 27, scope: !66)
// CHECK:STDOUT: !75 = !{!67}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'unformed.carbon'
// CHECK:STDOUT: source_filename = "unformed.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CUnformed.Main(ptr sret({ ptr, i64 }) %return, ptr %p, i64 %n) #0 !dbg !82 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %s.var = alloca { ptr, i64 }, align 8, !dbg !89
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %s.var), !dbg !89
// CHECK:STDOUT:   call void @_CMake.Slice.Core.55456748a8a2eb70(ptr %s.var, ptr %p, i64 %n), !dbg !94
// CHECK:STDOUT:   call void @"_COp.Slice.eb057aa32837c84e.Core:Copy.Core.55456748a8a2eb70"(ptr %return, ptr %s.var), !dbg !95
// CHECK:STDOUT:   call void @"_CSelfDestruct.5aff42c49e706c3f:core.Destroy.Core"(ptr %s.var), !dbg !89
// CHECK:STDOUT:   ret void, !dbg !96
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.a17e1ddde5b22223:core.Destroy.Core"(ptr %self) #0 !dbg !98 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !100
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.92d33a575b583425:core.Destroy.Core"(ptr %self) #0 !dbg !102 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !104
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.e2458c4221aa7f57:core.Destroy.Core"(ptr %self) #0 !dbg !106 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc4_3.1.size = getelementptr inbounds nuw { ptr, i64 }, ptr %self, i32 0, i32 1, !dbg !108
// CHECK:STDOUT:   call void @"_CSelfDestruct.92d33a575b583425:core.Destroy.Core"(ptr %.loc4_3.1.size), !dbg !108
// CHECK:STDOUT:   %.loc4_3.2.ptr = getelementptr inbounds nuw { ptr, i64 }, ptr %self, i32 0, i32 0, !dbg !108
// CHECK:STDOUT:   call void @"_CSelfDestruct.a17e1ddde5b22223:core.Destroy.Core"(ptr %.loc4_3.2.ptr), !dbg !108
// CHECK:STDOUT:   ret void, !dbg !108
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.e2458c4221aa7f57:core.Destroy.Core"(ptr %self) #0 !dbg !110 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.e2458c4221aa7f57:core.Destroy.Core"(ptr %self), !dbg !112
// CHECK:STDOUT:   ret void, !dbg !112
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.5aff42c49e706c3f:core.Destroy.Core"(ptr %self) #0 !dbg !114 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc4_3.1.size = getelementptr inbounds nuw { ptr, i64 }, ptr %self, i32 0, i32 1, !dbg !116
// CHECK:STDOUT:   call void @"_CSelfDestruct.92d33a575b583425:core.Destroy.Core"(ptr %.loc4_3.1.size), !dbg !116
// CHECK:STDOUT:   %.loc4_3.2.ptr = getelementptr inbounds nuw { ptr, i64 }, ptr %self, i32 0, i32 0, !dbg !116
// CHECK:STDOUT:   call void @"_CSelfDestruct.a17e1ddde5b22223:core.Destroy.Core"(ptr %.loc4_3.2.ptr), !dbg !116
// CHECK:STDOUT:   ret void, !dbg !116
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.5aff42c49e706c3f:core.Destroy.Core"(ptr %self) #0 !dbg !118 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.5aff42c49e706c3f:core.Destroy.Core"(ptr %self), !dbg !120
// CHECK:STDOUT:   ret void, !dbg !120
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CMake.Slice.Core.55456748a8a2eb70(ptr sret({ ptr, i64 }) %return, ptr %ptr, i64 %size) #0 !dbg !122 {
// CHECK:STDOUT:   %ptr1 = getelementptr inbounds nuw { ptr, i64 }, ptr %return, i32 0, i32 0, !dbg !131
// CHECK:STDOUT:   store ptr %ptr, ptr %ptr1, align 8, !dbg !131
// CHECK:STDOUT:   %size2 = getelementptr inbounds nuw { ptr, i64 }, ptr %return, i32 0, i32 1, !dbg !131
// CHECK:STDOUT:   store i64 %size, ptr %size2, align 8, !dbg !131
// CHECK:STDOUT:   ret void, !dbg !132
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.Slice.eb057aa32837c84e.Core:Copy.Core.55456748a8a2eb70"(ptr sret({ ptr, i64 }) %return, ptr %self) #0 !dbg !134 {
// CHECK:STDOUT:   %ptr = getelementptr inbounds nuw { ptr, i64 }, ptr %self, i32 0, i32 0, !dbg !139
// CHECK:STDOUT:   %1 = load ptr, ptr %ptr, align 8, !dbg !139
// CHECK:STDOUT:   %size = getelementptr inbounds nuw { ptr, i64 }, ptr %self, i32 0, i32 1, !dbg !140
// CHECK:STDOUT:   %2 = load i64, ptr %size, align 8, !dbg !140
// CHECK:STDOUT:   %ptr1 = getelementptr inbounds nuw { ptr, i64 }, ptr %return, i32 0, i32 0, !dbg !141
// CHECK:STDOUT:   store ptr %1, ptr %ptr1, align 8, !dbg !141
// CHECK:STDOUT:   %size2 = getelementptr inbounds nuw { ptr, i64 }, ptr %return, i32 0, i32 1, !dbg !141
// CHECK:STDOUT:   store i64 %2, ptr %size2, align 8, !dbg !141
// CHECK:STDOUT:   ret void, !dbg !142
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!77}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 64, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!7, !7, !6}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !33 = !DIFile(filename: "{{.*}}/prelude/types/slice.carbon", directory: "")
// CHECK:STDOUT: !56 = !{!7, !7}
// CHECK:STDOUT: !57 = !DISubroutineType(types: !56)
// CHECK:STDOUT: !76 = !DIFile(filename: "unformed.carbon", directory: "")
// CHECK:STDOUT: !77 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !76, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !78 = !{null, !7}
// CHECK:STDOUT: !79 = !DISubroutineType(types: !78)
// CHECK:STDOUT: !80 = !{null, !6}
// CHECK:STDOUT: !81 = !DISubroutineType(types: !80)
// CHECK:STDOUT: !82 = distinct !DISubprogram(name: "Unformed", linkageName: "_CUnformed.Main", scope: null, file: !76, line: 3, type: !9, spFlags: DISPFlagDefinition, unit: !77, retainedNodes: !97)
// CHECK:STDOUT: !83 = !DILocalVariable(arg: 1, scope: !82, type: !7)
// CHECK:STDOUT: !84 = !DILocalVariable(arg: 2, scope: !82, type: !6)
// CHECK:STDOUT: !89 = !DILocation(line: 4, column: 3, scope: !82)
// CHECK:STDOUT: !94 = !DILocation(line: 5, column: 7, scope: !82)
// CHECK:STDOUT: !95 = !DILocation(line: 6, column: 10, scope: !82)
// CHECK:STDOUT: !96 = !DILocation(line: 6, column: 3, scope: !82)
// CHECK:STDOUT: !97 = !{!83, !84}
// CHECK:STDOUT: !98 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.a17e1ddde5b22223:core.Destroy.Core", scope: null, file: !76, line: 4, type: !79, spFlags: DISPFlagDefinition, unit: !77, retainedNodes: !101)
// CHECK:STDOUT: !99 = !DILocalVariable(arg: 1, scope: !98, type: !7)
// CHECK:STDOUT: !100 = !DILocation(line: 4, column: 3, scope: !98)
// CHECK:STDOUT: !101 = !{!99}
// CHECK:STDOUT: !102 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.92d33a575b583425:core.Destroy.Core", scope: null, file: !76, line: 4, type: !81, spFlags: DISPFlagDefinition, unit: !77, retainedNodes: !105)
// CHECK:STDOUT: !103 = !DILocalVariable(arg: 1, scope: !102, type: !6)
// CHECK:STDOUT: !104 = !DILocation(line: 4, column: 3, scope: !102)
// CHECK:STDOUT: !105 = !{!103}
// CHECK:STDOUT: !106 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.e2458c4221aa7f57:core.Destroy.Core", scope: null, file: !76, line: 4, type: !79, spFlags: DISPFlagDefinition, unit: !77, retainedNodes: !109)
// CHECK:STDOUT: !107 = !DILocalVariable(arg: 1, scope: !106, type: !7)
// CHECK:STDOUT: !108 = !DILocation(line: 4, column: 3, scope: !106)
// CHECK:STDOUT: !109 = !{!107}
// CHECK:STDOUT: !110 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.e2458c4221aa7f57:core.Destroy.Core", scope: null, file: !76, line: 4, type: !79, spFlags: DISPFlagDefinition, unit: !77, retainedNodes: !113)
// CHECK:STDOUT: !111 = !DILocalVariable(arg: 1, scope: !110, type: !7)
// CHECK:STDOUT: !112 = !DILocation(line: 4, column: 3, scope: !110)
// CHECK:STDOUT: !113 = !{!111}
// CHECK:STDOUT: !114 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.5aff42c49e706c3f:core.Destroy.Core", scope: null, file: !76, line: 4, type: !79, spFlags: DISPFlagDefinition, unit: !77, retainedNodes: !117)
// CHECK:STDOUT: !115 = !DILocalVariable(arg: 1, scope: !114, type: !7)
// CHECK:STDOUT: !116 = !DILocation(line: 4, column: 3, scope: !114)
// CHECK:STDOUT: !117 = !{!115}
// CHECK:STDOUT: !118 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.5aff42c49e706c3f:core.Destroy.Core", scope: null, file: !76, line: 4, type: !79, spFlags: DISPFlagDefinition, unit: !77, retainedNodes: !121)
// CHECK:STDOUT: !119 = !DILocalVariable(arg: 1, scope: !118, type: !7)
// CHECK:STDOUT: !120 = !DILocation(line: 4, column: 3, scope: !118)
// CHECK:STDOUT: !121 = !{!119}
// CHECK:STDOUT: !122 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Slice.Core.55456748a8a2eb70", scope: null, file: !33, line: 21, type: !9, spFlags: DISPFlagDefinition, unit: !77, retainedNodes: !133)
// CHECK:STDOUT: !123 = !DILocalVariable(arg: 1, scope: !122, type: !7)
// CHECK:STDOUT: !124 = !DILocalVariable(arg: 2, scope: !122, type: !6)
// CHECK:STDOUT: !131 = !DILocation(line: 22, column: 12, scope: !122)
// CHECK:STDOUT: !132 = !DILocation(line: 22, column: 5, scope: !122)
// CHECK:STDOUT: !133 = !{!123, !124}
// CHECK:STDOUT: !134 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Slice.eb057aa32837c84e.Core:Copy.Core.55456748a8a2eb70", scope: null, file: !33, line: 33, type: !57, spFlags: DISPFlagDefinition, unit: !77, retainedNodes: !143)
// CHECK:STDOUT: !135 = !DILocalVariable(arg: 1, scope: !134, type: !7)
// CHECK:STDOUT: !139 = !DILocation(line: 33, column: 42, scope: !134)
// CHECK:STDOUT: !140 = !DILocation(line: 33, column: 60, scope: !134)
// CHECK:STDOUT: !141 = !DILocation(line: 33, column: 34, scope: !134)
// CHECK:STDOUT: !142 = !DILocation(line: 33, column: 27, scope: !134)
// CHECK:STDOUT: !143 = !{!135}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'iterate.carbon'
// CHECK:STDOUT: source_filename = "iterate.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CIterateSlice.Main(ptr %s) #0 !dbg !150 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %sum.var = alloca i32, align 4, !dbg !155
// CHECK:STDOUT:   %var = alloca i64, align 8, !dbg !159
// CHECK:STDOUT:   %.loc5_19.5.temp = alloca <{ i32, i1 }>, align 4, !dbg !159
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %sum.var), !dbg !155
// CHECK:STDOUT:   store i32 0, ptr %sum.var, align 4, !dbg !155
// CHECK:STDOUT:   %Slice.as.Iterate.impl.NewCursor.call = call i64 @"_CNewCursor.Slice.929611b8ac21773a.Core:Iterate.Core.d86245b043c956e8"(ptr %s), !dbg !159
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %var), !dbg !159
// CHECK:STDOUT:   store i64 %Slice.as.Iterate.impl.NewCursor.call, ptr %var, align 8, !dbg !159
// CHECK:STDOUT:   br label %for.next, !dbg !159
// CHECK:STDOUT:
// CHECK:STDOUT: for.next:                                         ; preds = %for.body, %entry
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc5_19.5.temp), !dbg !159
// CHECK:STDOUT:   call void @"_CNext.Slice.929611b8ac21773a.Core:Iterate.Core.d86245b043c956e8"(ptr %.loc5_19.5.temp, ptr %s, ptr %var), !dbg !159
// CHECK:STDOUT:   %Optional.HasValue.call = call i1 @_CHasValue.Optional.Core.ebd99c5786039774(ptr %.loc5_19.5.temp), !dbg !159
// CHECK:STDOUT:   br i1 %Optional.HasValue.call, label %for.body, label %for.done, !dbg !159
// CHECK:STDOUT:
// CHECK:STDOUT: for.body:                                         ; preds = %for.next
// CHECK:STDOUT:   %Optional.Get.call = call i32 @_CGet.Optional.Core.ebd99c5786039774(ptr %.loc5_19.5.temp), !dbg !159
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.call = load i32, ptr %sum.var, align 4, !dbg !160
// CHECK:STDOUT:   %Int.as.AddAssignWith.impl.Op.call1 = add i32 %Int.as.AddAssignWith.impl.Op.call, %Optional.Get.call, !dbg !160
// CHECK:STDOUT:   store i32 %Int.as.AddAssignWith.impl.Op.call1, ptr %sum.var, align 4, !dbg !160
// CHECK:STDOUT:   call void @"_CSelfDestruct.bf93366da8df6472:core.Destroy.Core"(ptr %.loc5_19.5.temp), !dbg !159
// CHECK:STDOUT:   br label %for.next, !dbg !162
// CHECK:STDOUT:
// CHECK:STDOUT: for.done:                                         ; preds = %for.next
// CHECK:STDOUT:   call void @"_CSelfDestruct.bf93366da8df6472:core.Destroy.Core"(ptr %.loc5_19.5.temp), !dbg !159
// CHECK:STDOUT:   call void @"_CSelfDestruct.92d33a575b583425:core.Destroy.Core"(ptr %var), !dbg !159
// CHECK:STDOUT:   %.loc8 = load i32, ptr %sum.var, align 4, !dbg !163
// CHECK:STDOUT:   call void @"_CSelfDestruct.72395be3dbbe8946:core.Destroy.Core"(ptr %sum.var), !dbg !155
// CHECK:STDOUT:   ret i32 %.loc8, !dbg !164
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CSelfDestruct.72395be3dbbe8946:core.Destroy.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CSelfDestruct.92d33a575b583425:core.Destroy.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.b3126fe3b97d1b85:core.Destroy.Core"(ptr %self) #0 !dbg !166 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !168
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.b3126fe3b97d1b85:core.Destroy.Core"(ptr %self) #0 !dbg !170 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.b3126fe3b97d1b85:core.Destroy.Core"(ptr %self), !dbg !172
// CHECK:STDOUT:   ret void, !dbg !172
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.993995cdb1dad067:core.Destroy.Core"(ptr %self) #0 !dbg !174 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !176
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.48e0adf639fe7af4:core.Destroy.Core"(ptr %self) #0 !dbg !178 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !180
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.2e0c223a4e2f3d14:core.Destroy.Core"(ptr %self) #0 !dbg !182 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc5_19.1.has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %self, i32 0, i32 1, !dbg !184
// CHECK:STDOUT:   call void @"_CSelfDestruct.48e0adf639fe7af4:core.Destroy.Core"(ptr %.loc5_19.1.has_value), !dbg !184
// CHECK:STDOUT:   %.loc5_19.2.value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %self, i32 0, i32 0, !dbg !184
// CHECK:STDOUT:   call void @"_CSelfDestruct.993995cdb1dad067:core.Destroy.Core"(ptr %.loc5_19.2.value), !dbg !184
// CHECK:STDOUT:   ret void, !dbg !184
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.2e0c223a4e2f3d14:core.Destroy.Core"(ptr %self) #0 !dbg !186 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.2e0c223a4e2f3d14:core.Destroy.Core"(ptr %self), !dbg !188
// CHECK:STDOUT:   ret void, !dbg !188
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.d2ff3a86a5bbb967:core.Destroy.Core"(ptr %self) #0 !dbg !190 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc5_19.1.has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %self, i32 0, i32 1, !dbg !192
// CHECK:STDOUT:   call void @"_CSelfDestruct.48e0adf639fe7af4:core.Destroy.Core"(ptr %.loc5_19.1.has_value), !dbg !192
// CHECK:STDOUT:   %.loc5_19.2.value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %self, i32 0, i32 0, !dbg !192
// CHECK:STDOUT:   call void @"_CSelfDestruct.993995cdb1dad067:core.Destroy.Core"(ptr %.loc5_19.2.value), !dbg !192
// CHECK:STDOUT:   ret void, !dbg !192
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.d2ff3a86a5bbb967:core.Destroy.Core"(ptr %self) #0 !dbg !194 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.d2ff3a86a5bbb967:core.Destroy.Core"(ptr %self), !dbg !196
// CHECK:STDOUT:   ret void, !dbg !196
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.bf93366da8df6472:core.Destroy.Core"(ptr %self) #0 !dbg !198 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !200
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i64 @"_CNewCursor.Slice.929611b8ac21773a.Core:Iterate.Core.d86245b043c956e8"(ptr %self) #0 !dbg !203 {
// CHECK:STDOUT:   ret i64 0, !dbg !208
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNext.Slice.929611b8ac21773a.Core:Iterate.Core.d86245b043c956e8"(ptr sret(<{ i32, i1 }>) %return, ptr %self, ptr %cursor) #0 !dbg !210 {
// CHECK:STDOUT:   %1 = alloca i64, align 8, !dbg !220
// CHECK:STDOUT:   %2 = call i64 @_CSize.Slice.Core.55456748a8a2eb70(ptr %self), !dbg !218
// CHECK:STDOUT:   %3 = load i64, ptr %cursor, align 8, !dbg !217
// CHECK:STDOUT:   %4 = icmp ult i64 %3, %2, !dbg !217
// CHECK:STDOUT:   br i1 %4, label %5, label %10, !dbg !219
// CHECK:STDOUT:
// CHECK:STDOUT: 5:                                                ; preds = %0
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !220
// CHECK:STDOUT:   %6 = load i64, ptr %cursor, align 8, !dbg !221
// CHECK:STDOUT:   store i64 %6, ptr %1, align 8, !dbg !220
// CHECK:STDOUT:   call void @"_COp.UInt.9452f4c51951679b.Core:Inc.Core.9025b11fc94ef1f2"(ptr %cursor), !dbg !224
// CHECK:STDOUT:   %7 = call ptr @_CPtr.Slice.Core.55456748a8a2eb70(ptr %self), !dbg !225
// CHECK:STDOUT:   %8 = load i64, ptr %1, align 8, !dbg !226
// CHECK:STDOUT:   %pointer.offset = getelementptr inbounds i32, ptr %7, i64 %8, !dbg !227
// CHECK:STDOUT:   %9 = load i32, ptr %pointer.offset, align 4, !dbg !228
// CHECK:STDOUT:   call void @_CSome.Optional.Core.ebd99c5786039774(ptr %return, i32 %9), !dbg !229
// CHECK:STDOUT:   call void @"_CSelfDestruct.92d33a575b583425:core.Destroy.Core"(ptr %1), !dbg !220
// CHECK:STDOUT:   ret void, !dbg !232
// CHECK:STDOUT:
// CHECK:STDOUT: 10:                                               ; preds = %0
// CHECK:STDOUT:   call void @_CNone.Optional.Core.ebd99c5786039774(ptr %return), !dbg !233
// CHECK:STDOUT:   ret void, !dbg !236
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.ebd99c5786039774(ptr %self) #0 !dbg !239 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.d86245b043c956e8"(ptr %self), !dbg !245
// CHECK:STDOUT:   ret i1 %1, !dbg !246
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CGet.Optional.Core.ebd99c5786039774(ptr %self) #0 !dbg !248 {
// CHECK:STDOUT:   %1 = call i32 @"_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.d86245b043c956e8"(ptr %self), !dbg !254
// CHECK:STDOUT:   ret i32 %1, !dbg !255
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i64 @_CSize.Slice.Core.55456748a8a2eb70(ptr %self) #0 !dbg !257 {
// CHECK:STDOUT:   %size = getelementptr inbounds nuw { ptr, i64 }, ptr %self, i32 0, i32 1, !dbg !262
// CHECK:STDOUT:   %1 = load i64, ptr %size, align 8, !dbg !262
// CHECK:STDOUT:   ret i64 %1, !dbg !263
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.UInt.9452f4c51951679b.Core:Inc.Core.9025b11fc94ef1f2"(ptr %self) #0 !dbg !266 {
// CHECK:STDOUT:   call void @"_COp:thunk:AddAssignWith.03689a112aff8ff9.Core:UInt.9452f4c51951679b.Core:AddAssignWith.1d47ea6b06d1da31.Core.6c4979b2d311a38e"(ptr %self, i64 1), !dbg !269
// CHECK:STDOUT:   ret void, !dbg !272
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CPtr.Slice.Core.55456748a8a2eb70(ptr %self) #0 !dbg !274 {
// CHECK:STDOUT:   %ptr = getelementptr inbounds nuw { ptr, i64 }, ptr %self, i32 0, i32 0, !dbg !279
// CHECK:STDOUT:   %1 = load ptr, ptr %ptr, align 8, !dbg !279
// CHECK:STDOUT:   ret ptr %1, !dbg !280
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSome.Optional.Core.ebd99c5786039774(ptr sret(<{ i32, i1 }>) %return, i32 %value) #0 !dbg !282 {
// CHECK:STDOUT:   call void @"_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.d86245b043c956e8"(ptr %return, i32 %value), !dbg !287
// CHECK:STDOUT:   ret void, !dbg !288
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.ebd99c5786039774(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !290 {
// CHECK:STDOUT:   call void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.d86245b043c956e8"(ptr %return), !dbg !293
// CHECK:STDOUT:   ret void, !dbg !294
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.d86245b043c956e8"(ptr %value) #0 !dbg !295 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 1, !dbg !300
// CHECK:STDOUT:   %1 = load i8, ptr %has_value, align 1, !dbg !300
// CHECK:STDOUT:   %2 = trunc i8 %1 to i1, !dbg !300
// CHECK:STDOUT:   ret i1 %2, !dbg !301
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.d86245b043c956e8"(ptr %value) #0 !dbg !303 {
// CHECK:STDOUT:   %value1 = getelementptr inbounds nuw <{ i32, i1 }>, ptr %value, i32 0, i32 0, !dbg !308
// CHECK:STDOUT:   %1 = load i32, ptr %value1, align 4, !dbg !308
// CHECK:STDOUT:   ret i32 %1, !dbg !309
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk:AddAssignWith.03689a112aff8ff9.Core:UInt.9452f4c51951679b.Core:AddAssignWith.1d47ea6b06d1da31.Core.6c4979b2d311a38e"(ptr %self, i64 %other) #2 !dbg !311 {
// CHECK:STDOUT:   %1 = call i64 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.3260578ee2895055.Core.89b00b8a376d67f1"(i64 %other), !dbg !315
// CHECK:STDOUT:   %2 = load i64, ptr %self, align 8, !dbg !315
// CHECK:STDOUT:   %3 = add i64 %2, %1, !dbg !315
// CHECK:STDOUT:   store i64 %3, ptr %self, align 8, !dbg !315
// CHECK:STDOUT:   ret void, !dbg !315
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.d86245b043c956e8"(ptr sret(<{ i32, i1 }>) %return, i32 %self) #0 !dbg !319 {
// CHECK:STDOUT:   %value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 0, !dbg !328
// CHECK:STDOUT:   store i32 %self, ptr %value, align 4, !dbg !328
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !330
// CHECK:STDOUT:   store i8 1, ptr %has_value, align 1, !dbg !330
// CHECK:STDOUT:   ret void, !dbg !331
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.d86245b043c956e8"(ptr sret(<{ i32, i1 }>) %return) #0 !dbg !333 {
// CHECK:STDOUT:   %has_value = getelementptr inbounds nuw <{ i32, i1 }>, ptr %return, i32 0, i32 1, !dbg !340
// CHECK:STDOUT:   store i8 0, ptr %has_value, align 1, !dbg !340
// CHECK:STDOUT:   ret void, !dbg !341
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i64 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.3260578ee2895055.Core.89b00b8a376d67f1"(i64 %self) #0 !dbg !342 {
// CHECK:STDOUT:   %1 = call i64 @"_CConvert.UInt.9452f4c51951679b.Core:FromUInt.180681b028ad46dc.Core.eebbceaf184830c4"(i64 %self), !dbg !348
// CHECK:STDOUT:   ret i64 %1, !dbg !349
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i64 @"_CConvert.UInt.9452f4c51951679b.Core:FromUInt.180681b028ad46dc.Core.eebbceaf184830c4"(i64 %from) #0 !dbg !351 {
// CHECK:STDOUT:   %1 = call i64 @"_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.9025b11fc94ef1f2"(i64 %from), !dbg !356
// CHECK:STDOUT:   ret i64 %1, !dbg !358
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i64 @"_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.9025b11fc94ef1f2"(i64 %self) #0 !dbg !360 {
// CHECK:STDOUT:   ret i64 %self, !dbg !366
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_CSelfDestruct.bf93366da8df6472:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 3, 2, 1 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!145}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 64, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!7, !7, !6}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !10 = !{!6, !7}
// CHECK:STDOUT: !11 = !DISubroutineType(types: !10)
// CHECK:STDOUT: !32 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !33 = !DIFile(filename: "{{.*}}/prelude/types/slice.carbon", directory: "")
// CHECK:STDOUT: !56 = !{!7, !7}
// CHECK:STDOUT: !57 = !DISubroutineType(types: !56)
// CHECK:STDOUT: !78 = !{null, !7}
// CHECK:STDOUT: !79 = !DISubroutineType(types: !78)
// CHECK:STDOUT: !80 = !{null, !6}
// CHECK:STDOUT: !81 = !DISubroutineType(types: !80)
// CHECK:STDOUT: !144 = !DIFile(filename: "iterate.carbon", directory: "")
// CHECK:STDOUT: !145 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !144, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !146 = !{!32, !7}
// CHECK:STDOUT: !147 = !DISubroutineType(types: !146)
// CHECK:STDOUT: !148 = !{null, !32}
// CHECK:STDOUT: !149 = !DISubroutineType(types: !148)
// CHECK:STDOUT: !150 = distinct !DISubprogram(name: "IterateSlice", linkageName: "_CIterateSlice.Main", scope: null, file: !144, line: 3, type: !147, spFlags: DISPFlagDefinition, unit: !145, retainedNodes: !165)
// CHECK:STDOUT: !151 = !DILocalVariable(arg: 1, scope: !150, type: !7)
// CHECK:STDOUT: !155 = !DILocation(line: 4, column: 3, scope: !150)
// CHECK:STDOUT: !159 = !DILocation(line: 5, column: 7, scope: !150)
// CHECK:STDOUT: !160 = !DILocation(line: 6, column: 5, scope: !150)
// CHECK:STDOUT: !162 = !DILocation(line: 5, column: 3, scope: !150)
// CHECK:STDOUT: !163 = !DILocation(line: 8, column: 10, scope: !150)
// CHECK:STDOUT: !164 = !DILocation(line: 8, column: 3, scope: !150)
// CHECK:STDOUT: !165 = !{!151}
// CHECK:STDOUT: !166 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.b3126fe3b97d1b85:core.Destroy.Core", scope: null, file: !144, line: 5, type: !79, spFlags: DISPFlagDefinition, unit: !145, retainedNodes: !169)
// CHECK:STDOUT: !167 = !DILocalVariable(arg: 1, scope: !166, type: !7)
// CHECK:STDOUT: !168 = !DILocation(line: 5, column: 7, scope: !166)
// CHECK:STDOUT: !169 = !{!167}
// CHECK:STDOUT: !170 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.b3126fe3b97d1b85:core.Destroy.Core", scope: null, file: !144, line: 5, type: !79, spFlags: DISPFlagDefinition, unit: !145, retainedNodes: !173)
// CHECK:STDOUT: !171 = !DILocalVariable(arg: 1, scope: !170, type: !7)
// CHECK:STDOUT: !172 = !DILocation(line: 5, column: 7, scope: !170)
// CHECK:STDOUT: !173 = !{!171}
// CHECK:STDOUT: !174 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.993995cdb1dad067:core.Destroy.Core", scope: null, file: !144, line: 5, type: !149, spFlags: DISPFlagDefinition, unit: !145, retainedNodes: !177)
// CHECK:STDOUT: !175 = !DILocalVariable(arg: 1, scope: !174, type: !32)
// CHECK:STDOUT: !176 = !DILocation(line: 5, column: 7, scope: !174)
// CHECK:STDOUT: !177 = !{!175}
// CHECK:STDOUT: !178 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.48e0adf639fe7af4:core.Destroy.Core", scope: null, file: !144, line: 5, type: !79, spFlags: DISPFlagDefinition, unit: !145, retainedNodes: !181)
// CHECK:STDOUT: !179 = !DILocalVariable(arg: 1, scope: !178, type: !7)
// CHECK:STDOUT: !180 = !DILocation(line: 5, column: 7, scope: !178)
// CHECK:STDOUT: !181 = !{!179}
// CHECK:STDOUT: !182 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.2e0c223a4e2f3d14:core.Destroy.Core", scope: null, file: !144, line: 5, type: !79, spFlags: DISPFlagDefinition, unit: !145, retainedNodes: !185)
// CHECK:STDOUT: !183 = !DILocalVariable(arg: 1, scope: !182, type: !7)
// CHECK:STDOUT: !184 = !DILocation(line: 5, column: 7, scope: !182)
// CHECK:STDOUT: !185 = !{!183}
// CHECK:STDOUT: !186 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.2e0c223a4e2f3d14:core.Destroy.Core", scope: null, file: !144, line: 5, type: !79, spFlags: DISPFlagDefinition, unit: !145, retainedNodes: !189)
// CHECK:STDOUT: !187 = !DILocalVariable(arg: 1, scope: !186, type: !7)
// CHECK:STDOUT: !188 = !DILocation(line: 5, column: 7, scope: !186)
// CHECK:STDOUT: !189 = !{!187}
// CHECK:STDOUT: !190 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.d2ff3a86a5bbb967:core.Destroy.Core", scope: null, file: !144, line: 5, type: !79, spFlags: DISPFlagDefinition, unit: !145, retainedNodes: !193)
// CHECK:STDOUT: !191 = !DILocalVariable(arg: 1, scope: !190, type: !7)
// CHECK:STDOUT: !192 = !DILocation(line: 5, column: 7, scope: !190)
// CHECK:STDOUT: !193 = !{!191}
// CHECK:STDOUT: !194 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.d2ff3a86a5bbb967:core.Destroy.Core", scope: null, file: !144, line: 5, type: !79, spFlags: DISPFlagDefinition, unit: !145, retainedNodes: !197)
// CHECK:STDOUT: !195 = !DILocalVariable(arg: 1, scope: !194, type: !7)
// CHECK:STDOUT: !196 = !DILocation(line: 5, column: 7, scope: !194)
// CHECK:STDOUT: !197 = !{!195}
// CHECK:STDOUT: !198 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.bf93366da8df6472:core.Destroy.Core", scope: null, file: !144, line: 5, type: !79, spFlags: DISPFlagDefinition, unit: !145, retainedNodes: !201)
// CHECK:STDOUT: !199 = !DILocalVariable(arg: 1, scope: !198, type: !7)
// CHECK:STDOUT: !200 = !DILocation(line: 5, column: 7, scope: !198)
// CHECK:STDOUT: !201 = !{!199}
// CHECK:STDOUT: !202 = !DIFile(filename: "{{.*}}/prelude/iterate.carbon", directory: "")
// CHECK:STDOUT: !203 = distinct !DISubprogram(name: "NewCursor", linkageName: "_CNewCursor.Slice.929611b8ac21773a.Core:Iterate.Core.d86245b043c956e8", scope: null, file: !202, line: 42, type: !11, spFlags: DISPFlagDefinition, unit: !145, retainedNodes: !209)
// CHECK:STDOUT: !204 = !DILocalVariable(arg: 1, scope: !203, type: !7)
// CHECK:STDOUT: !208 = !DILocation(line: 42, column: 38, scope: !203)
// CHECK:STDOUT: !209 = !{!204}
// CHECK:STDOUT: !210 = distinct !DISubprogram(name: "Next", linkageName: "_CNext.Slice.929611b8ac21773a.Core:Iterate.Core.d86245b043c956e8", scope: null, file: !202, line: 43, type: !9, spFlags: DISPFlagDefinition, unit: !145, retainedNodes: !237)
// CHECK:STDOUT: !211 = !DILocalVariable(arg: 1, scope: !210, type: !7)
// CHECK:STDOUT: !212 = !DILocalVariable(arg: 2, scope: !210, type: !6)
// CHECK:STDOUT: !217 = !DILocation(line: 44, column: 9, scope: !210)
// CHECK:STDOUT: !218 = !DILocation(line: 44, column: 18, scope: !210)
// CHECK:STDOUT: !219 = !DILocation(line: 44, column: 8, scope: !210)
// CHECK:STDOUT: !220 = !DILocation(line: 45, column: 7, scope: !210)
// CHECK:STDOUT: !221 = !DILocation(line: 45, column: 25, scope: !210)
// CHECK:STDOUT: !224 = !DILocation(line: 46, column: 7, scope: !210)
// CHECK:STDOUT: !225 = !DILocation(line: 47, column: 46, scope: !210)
// CHECK:STDOUT: !226 = !DILocation(line: 47, column: 58, scope: !210)
// CHECK:STDOUT: !227 = !DILocation(line: 47, column: 32, scope: !210)
// CHECK:STDOUT: !228 = !DILocation(line: 47, column: 31, scope: !210)
// CHECK:STDOUT: !229 = !DILocation(line: 47, column: 14, scope: !210)
// CHECK:STDOUT: !230 = !{!7, !32}
// CHECK:STDOUT: !231 = !DISubroutineType(types: !230)
// CHECK:STDOUT: !232 = !DILocation(line: 47, column: 7, scope: !210)
// CHECK:STDOUT: !233 = !DILocation(line: 49, column: 14, scope: !210)
// CHECK:STDOUT: !234 = !{!7}
// CHECK:STDOUT: !235 = !DISubroutineType(types: !234)
// CHECK:STDOUT: !236 = !DILocation(line: 49, column: 7, scope: !210)
// CHECK:STDOUT: !237 = !{!211, !212}
// CHECK:STDOUT: !238 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !239 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.ebd99c5786039774", scope: null, file: !238, line: 36, type: !57, spFlags: DISPFlagDefinition, unit: !145, retainedNodes: !247)
// CHECK:STDOUT: !240 = !DILocalVariable(arg: 1, scope: !239, type: !7)
// CHECK:STDOUT: !245 = !DILocation(line: 37, column: 12, scope: !239)
// CHECK:STDOUT: !246 = !DILocation(line: 37, column: 5, scope: !239)
// CHECK:STDOUT: !247 = !{!240}
// CHECK:STDOUT: !248 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.ebd99c5786039774", scope: null, file: !238, line: 39, type: !147, spFlags: DISPFlagDefinition, unit: !145, retainedNodes: !256)
// CHECK:STDOUT: !249 = !DILocalVariable(arg: 1, scope: !248, type: !7)
// CHECK:STDOUT: !254 = !DILocation(line: 40, column: 12, scope: !248)
// CHECK:STDOUT: !255 = !DILocation(line: 40, column: 5, scope: !248)
// CHECK:STDOUT: !256 = !{!249}
// CHECK:STDOUT: !257 = distinct !DISubprogram(name: "Size", linkageName: "_CSize.Slice.Core.55456748a8a2eb70", scope: null, file: !33, line: 25, type: !11, spFlags: DISPFlagDefinition, unit: !145, retainedNodes: !264)
// CHECK:STDOUT: !258 = !DILocalVariable(arg: 1, scope: !257, type: !7)
// CHECK:STDOUT: !262 = !DILocation(line: 25, column: 33, scope: !257)
// CHECK:STDOUT: !263 = !DILocation(line: 25, column: 26, scope: !257)
// CHECK:STDOUT: !264 = !{!258}
// CHECK:STDOUT: !265 = !DIFile(filename: "{{.*}}/prelude/types/uint.carbon", directory: "")
// CHECK:STDOUT: !266 = distinct !DISubprogram(name: "Op", linkageName: "_COp.UInt.9452f4c51951679b.Core:Inc.Core.9025b11fc94ef1f2", scope: null, file: !265, line: 458, type: !81, spFlags: DISPFlagDefinition, unit: !145, retainedNodes: !273)
// CHECK:STDOUT: !267 = !DILocalVariable(arg: 1, scope: !266, type: !6)
// CHECK:STDOUT: !269 = !DILocation(line: 460, column: 5, scope: !266)
// CHECK:STDOUT: !270 = !{null, !6, !6}
// CHECK:STDOUT: !271 = !DISubroutineType(types: !270)
// CHECK:STDOUT: !272 = !DILocation(line: 458, column: 3, scope: !266)
// CHECK:STDOUT: !273 = !{!267}
// CHECK:STDOUT: !274 = distinct !DISubprogram(name: "Ptr", linkageName: "_CPtr.Slice.Core.55456748a8a2eb70", scope: null, file: !33, line: 26, type: !57, spFlags: DISPFlagDefinition, unit: !145, retainedNodes: !281)
// CHECK:STDOUT: !275 = !DILocalVariable(arg: 1, scope: !274, type: !7)
// CHECK:STDOUT: !279 = !DILocation(line: 26, column: 31, scope: !274)
// CHECK:STDOUT: !280 = !DILocation(line: 26, column: 24, scope: !274)
// CHECK:STDOUT: !281 = !{!275}
// CHECK:STDOUT: !282 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.ebd99c5786039774", scope: null, file: !238, line: 33, type: !231, spFlags: DISPFlagDefinition, unit: !145, retainedNodes: !289)
// CHECK:STDOUT: !283 = !DILocalVariable(arg: 1, scope: !282, type: !32)
// CHECK:STDOUT: !287 = !DILocation(line: 34, column: 12, scope: !282)
// CHECK:STDOUT: !288 = !DILocation(line: 34, column: 5, scope: !282)
// CHECK:STDOUT: !289 = !{!283}
// CHECK:STDOUT: !290 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.ebd99c5786039774", scope: null, file: !238, line: 30, type: !235, spFlags: DISPFlagDefinition, unit: !145)
// CHECK:STDOUT: !293 = !DILocation(line: 31, column: 12, scope: !290)
// CHECK:STDOUT: !294 = !DILocation(line: 31, column: 5, scope: !290)
// CHECK:STDOUT: !295 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.5d6f3f3a4b44a8f1:OptionalStorage.Core.d86245b043c956e8", scope: null, file: !238, line: 141, type: !57, spFlags: DISPFlagDefinition, unit: !145, retainedNodes: !302)
// CHECK:STDOUT: !296 = !DILocalVariable(arg: 1, scope: !295, type: !7)
// CHECK:STDOUT: !300 = !DILocation(line: 142, column: 12, scope: !295)
// CHECK:STDOUT: !301 = !DILocation(line: 142, column: 5, scope: !295)
// CHECK:STDOUT: !302 = !{!296}
// CHECK:STDOUT: !303 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.5d6f3f3a4b44a8f1:OptionalStorage.Core.d86245b043c956e8", scope: null, file: !238, line: 144, type: !147, spFlags: DISPFlagDefinition, unit: !145, retainedNodes: !310)
// CHECK:STDOUT: !304 = !DILocalVariable(arg: 1, scope: !303, type: !7)
// CHECK:STDOUT: !308 = !DILocation(line: 145, column: 12, scope: !303)
// CHECK:STDOUT: !309 = !DILocation(line: 145, column: 5, scope: !303)
// CHECK:STDOUT: !310 = !{!304}
// CHECK:STDOUT: !311 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:AddAssignWith.03689a112aff8ff9.Core:UInt.9452f4c51951679b.Core:AddAssignWith.1d47ea6b06d1da31.Core.6c4979b2d311a38e", scope: null, file: !265, line: 393, type: !271, spFlags: DISPFlagDefinition, unit: !145, retainedNodes: !318)
// CHECK:STDOUT: !312 = !DILocalVariable(arg: 1, scope: !311, type: !6)
// CHECK:STDOUT: !313 = !DILocalVariable(arg: 2, scope: !311, type: !6)
// CHECK:STDOUT: !315 = !DILocation(line: 393, column: 3, scope: !311)
// CHECK:STDOUT: !316 = !{!6, !6}
// CHECK:STDOUT: !317 = !DISubroutineType(types: !316)
// CHECK:STDOUT: !318 = !{!312, !313}
// CHECK:STDOUT: !319 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.5d6f3f3a4b44a8f1:OptionalStorage.Core.d86245b043c956e8", scope: null, file: !238, line: 131, type: !231, spFlags: DISPFlagDefinition, unit: !145, retainedNodes: !332)
// CHECK:STDOUT: !320 = !DILocalVariable(arg: 1, scope: !319, type: !32)
// CHECK:STDOUT: !328 = !DILocation(line: 137, column: 5, scope: !319)
// CHECK:STDOUT: !330 = !DILocation(line: 138, column: 5, scope: !319)
// CHECK:STDOUT: !331 = !DILocation(line: 139, column: 5, scope: !319)
// CHECK:STDOUT: !332 = !{!320}
// CHECK:STDOUT: !333 = distinct !DISubprogram(name: "None", linkageName: "_CNone.5d6f3f3a4b44a8f1:OptionalStorage.Core.d86245b043c956e8", scope: null, file: !238, line: 125, type: !235, spFlags: DISPFlagDefinition, unit: !145)
// CHECK:STDOUT: !340 = !DILocation(line: 128, column: 5, scope: !333)
// CHECK:STDOUT: !341 = !DILocation(line: 129, column: 5, scope: !333)
// CHECK:STDOUT: !342 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.3260578ee2895055.Core.89b00b8a376d67f1", scope: null, file: !265, line: 88, type: !317, spFlags: DISPFlagDefinition, unit: !145, retainedNodes: !350)
// CHECK:STDOUT: !343 = !DILocalVariable(arg: 1, scope: !342, type: !6)
// CHECK:STDOUT: !348 = !DILocation(line: 89, column: 12, scope: !342)
// CHECK:STDOUT: !349 = !DILocation(line: 89, column: 5, scope: !342)
// CHECK:STDOUT: !350 = !{!343}
// CHECK:STDOUT: !351 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.UInt.9452f4c51951679b.Core:FromUInt.180681b028ad46dc.Core.eebbceaf184830c4", scope: null, file: !265, line: 72, type: !317, spFlags: DISPFlagDefinition, unit: !145, retainedNodes: !359)
// CHECK:STDOUT: !352 = !DILocalVariable(arg: 1, scope: !351, type: !6)
// CHECK:STDOUT: !356 = !DILocation(line: 74, column: 17, scope: !351)
// CHECK:STDOUT: !358 = !DILocation(line: 74, column: 5, scope: !351)
// CHECK:STDOUT: !359 = !{!352}
// CHECK:STDOUT: !360 = distinct !DISubprogram(name: "AsUInt", linkageName: "_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.9025b11fc94ef1f2", scope: null, file: !265, line: 60, type: !317, spFlags: DISPFlagDefinition, unit: !145, retainedNodes: !367)
// CHECK:STDOUT: !361 = !DILocalVariable(arg: 1, scope: !360, type: !6)
// CHECK:STDOUT: !366 = !DILocation(line: 60, column: 29, scope: !360)
// CHECK:STDOUT: !367 = !{!361}
// CHECK:STDOUT:
