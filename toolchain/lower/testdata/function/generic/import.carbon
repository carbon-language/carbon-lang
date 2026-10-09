// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/import.carbon

// Test that we can import and call generic functions across libraries, both
// when the generic is defined in a directly imported library and when it's
// defined in an indirectly imported library.

// --- indirectly_imported.carbon

library "[[@TEST_NAME]]";

fn IndirectDeclared();

fn IndirectDefined() {}

fn IndirectGeneric(generic T: type, x: T*) -> T* {
  IndirectDeclared();
  IndirectDefined();
  return x;
}

// --- directly_imported.carbon

library "[[@TEST_NAME]]";

import library "indirectly_imported";

fn DirectDeclared();

fn DirectDefined() {}

fn DirectGeneric(generic T: type, y: T*) -> T* {
  DirectDeclared();
  DirectDefined();
  return IndirectGeneric(T, y);
}

// --- use.carbon

library "[[@TEST_NAME]]";

import library "directly_imported";

fn Call() -> i32 {
  var n: i32 = 0;
  var p: i32* = DirectGeneric(i32, &n);
  return *p;
}

// --- return_var.carbon

library "[[@TEST_NAME]]";

class C {
  var data: i32;
  var next: C*;
}

fn Make(unused generic T: type) -> C {
  returned var c: C;
  c = {.data = 42, .next = &c };
  return var;
}

// --- use_return_var.carbon

library "[[@TEST_NAME]]";

import library "return_var";

fn Run() {
  Make(());
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'indirectly_imported.carbon'
// CHECK:STDOUT: source_filename = "indirectly_imported.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CIndirectDeclared.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CIndirectDefined.Main() #0 !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "indirectly_imported.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !{null}
// CHECK:STDOUT: !21 = !DISubroutineType(types: !20)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "IndirectDefined", linkageName: "_CIndirectDefined.Main", scope: null, file: !18, line: 6, type: !21, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !23 = !DILocation(line: 6, column: 1, scope: !22)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'directly_imported.carbon'
// CHECK:STDOUT: source_filename = "directly_imported.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CDirectDeclared.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CDirectDefined.Main() #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!25}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !{null}
// CHECK:STDOUT: !21 = !DISubroutineType(types: !20)
// CHECK:STDOUT: !24 = !DIFile(filename: "directly_imported.carbon", directory: "")
// CHECK:STDOUT: !25 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !24, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "DirectDefined", linkageName: "_CDirectDefined.Main", scope: null, file: !24, line: 8, type: !21, spFlags: DISPFlagDefinition, unit: !25)
// CHECK:STDOUT: !27 = !DILocation(line: 8, column: 1, scope: !26)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'use.carbon'
// CHECK:STDOUT: source_filename = "use.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CCall.Main() #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !41
// CHECK:STDOUT:   %p.var = alloca ptr, align 8, !dbg !43
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !41
// CHECK:STDOUT:   store i32 0, ptr %n.var, align 4, !dbg !41
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %p.var), !dbg !43
// CHECK:STDOUT:   %DirectGeneric.call = call ptr @_CDirectGeneric.Main.6d93ae89cb2e7b0f(ptr %n.var), !dbg !46
// CHECK:STDOUT:   store ptr %DirectGeneric.call, ptr %p.var, align 8, !dbg !43
// CHECK:STDOUT:   %.loc9_11 = load ptr, ptr %p.var, align 8, !dbg !50
// CHECK:STDOUT:   %.loc9_10.2 = load i32, ptr %.loc9_11, align 4, !dbg !51
// CHECK:STDOUT:   call void @"_CSelfDestruct.1feff81df232a20a:core.Destroy.Core"(ptr %p.var), !dbg !43
// CHECK:STDOUT:   call void @"_CSelfDestruct.76db170ab64c05aa:core.Destroy.Core"(ptr %n.var), !dbg !41
// CHECK:STDOUT:   ret i32 %.loc9_10.2, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.1feff81df232a20a:core.Destroy.Core"(ptr %self) #0 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.76db170ab64c05aa:core.Destroy.Core"(ptr %self) #0 !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CDirectGeneric.Main.6d93ae89cb2e7b0f(ptr %y) #0 !dbg !61 {
// CHECK:STDOUT:   call void @_CDirectDeclared.Main(), !dbg !66
// CHECK:STDOUT:   call void @_CDirectDefined.Main(), !dbg !67
// CHECK:STDOUT:   %1 = call ptr @_CIndirectGeneric.Main.6d93ae89cb2e7b0f(ptr %y), !dbg !69
// CHECK:STDOUT:   ret ptr %1, !dbg !70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CDirectDeclared.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CDirectDefined.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CIndirectGeneric.Main.6d93ae89cb2e7b0f(ptr %x) #0 !dbg !72 {
// CHECK:STDOUT:   call void @_CIndirectDeclared.Main(), !dbg !77
// CHECK:STDOUT:   call void @_CIndirectDefined.Main(), !dbg !78
// CHECK:STDOUT:   ret ptr %x, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CIndirectDeclared.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CIndirectDefined.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!29}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "indirectly_imported.carbon", directory: "")
// CHECK:STDOUT: !24 = !DIFile(filename: "directly_imported.carbon", directory: "")
// CHECK:STDOUT: !28 = !DIFile(filename: "use.carbon", directory: "")
// CHECK:STDOUT: !29 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !28, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !30 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !31 = !{!30}
// CHECK:STDOUT: !32 = !DISubroutineType(types: !31)
// CHECK:STDOUT: !33 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !34 = !{null, !33}
// CHECK:STDOUT: !35 = !DISubroutineType(types: !34)
// CHECK:STDOUT: !36 = !{null, !30}
// CHECK:STDOUT: !37 = !DISubroutineType(types: !36)
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !28, line: 6, type: !32, spFlags: DISPFlagDefinition, unit: !29)
// CHECK:STDOUT: !41 = !DILocation(line: 7, column: 3, scope: !38)
// CHECK:STDOUT: !43 = !DILocation(line: 8, column: 3, scope: !38)
// CHECK:STDOUT: !46 = !DILocation(line: 8, column: 17, scope: !38)
// CHECK:STDOUT: !47 = !{!33, !33}
// CHECK:STDOUT: !48 = !DISubroutineType(types: !47)
// CHECK:STDOUT: !50 = !DILocation(line: 9, column: 11, scope: !38)
// CHECK:STDOUT: !51 = !DILocation(line: 9, column: 10, scope: !38)
// CHECK:STDOUT: !52 = !DILocation(line: 9, column: 3, scope: !38)
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.1feff81df232a20a:core.Destroy.Core", scope: null, file: !28, line: 8, type: !35, spFlags: DISPFlagDefinition, unit: !29, retainedNodes: !56)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !53, type: !33)
// CHECK:STDOUT: !55 = !DILocation(line: 8, column: 3, scope: !53)
// CHECK:STDOUT: !56 = !{!54}
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.76db170ab64c05aa:core.Destroy.Core", scope: null, file: !28, line: 7, type: !37, spFlags: DISPFlagDefinition, unit: !29, retainedNodes: !60)
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 1, scope: !57, type: !30)
// CHECK:STDOUT: !59 = !DILocation(line: 7, column: 3, scope: !57)
// CHECK:STDOUT: !60 = !{!58}
// CHECK:STDOUT: !61 = distinct !DISubprogram(name: "DirectGeneric", linkageName: "_CDirectGeneric.Main.6d93ae89cb2e7b0f", scope: null, file: !24, line: 10, type: !48, spFlags: DISPFlagDefinition, unit: !29, retainedNodes: !71)
// CHECK:STDOUT: !62 = !DILocalVariable(arg: 1, scope: !61, type: !33)
// CHECK:STDOUT: !66 = !DILocation(line: 11, column: 3, scope: !61)
// CHECK:STDOUT: !67 = !DILocation(line: 12, column: 3, scope: !61)
// CHECK:STDOUT: !69 = !DILocation(line: 13, column: 10, scope: !61)
// CHECK:STDOUT: !70 = !DILocation(line: 13, column: 3, scope: !61)
// CHECK:STDOUT: !71 = !{!62}
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "IndirectGeneric", linkageName: "_CIndirectGeneric.Main.6d93ae89cb2e7b0f", scope: null, file: !18, line: 8, type: !48, spFlags: DISPFlagDefinition, unit: !29, retainedNodes: !81)
// CHECK:STDOUT: !73 = !DILocalVariable(arg: 1, scope: !72, type: !33)
// CHECK:STDOUT: !77 = !DILocation(line: 9, column: 3, scope: !72)
// CHECK:STDOUT: !78 = !DILocation(line: 10, column: 3, scope: !72)
// CHECK:STDOUT: !80 = !DILocation(line: 11, column: 3, scope: !72)
// CHECK:STDOUT: !81 = !{!73}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'return_var.carbon'
// CHECK:STDOUT: source_filename = "return_var.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!83}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !82 = !DIFile(filename: "return_var.carbon", directory: "")
// CHECK:STDOUT: !83 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !82, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'use_return_var.carbon'
// CHECK:STDOUT: source_filename = "use_return_var.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !86 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_10.2.temp = alloca { i32, ptr }, align 8, !dbg !87
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_10.2.temp), !dbg !87
// CHECK:STDOUT:   call void @_CMake.Main.e43630e9a6c38c3f(ptr %.loc7_10.2.temp), !dbg !87
// CHECK:STDOUT:   call void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %.loc7_10.2.temp), !dbg !87
// CHECK:STDOUT:   ret i32 0, !dbg !90
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.76db170ab64c05aa:core.Destroy.Core"(ptr %self) #0 !dbg !91 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6b633cf020343e69:core.Destroy.Core"(ptr %self) #0 !dbg !95 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.a856f1753aa61283:core.Destroy.Core"(ptr %self) #0 !dbg !99 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_10.1.next = getelementptr inbounds nuw { i32, ptr }, ptr %self, i32 0, i32 1, !dbg !101
// CHECK:STDOUT:   call void @"_CSelfDestruct.6b633cf020343e69:core.Destroy.Core"(ptr %.loc7_10.1.next), !dbg !101
// CHECK:STDOUT:   %.loc7_10.2.data = getelementptr inbounds nuw { i32, ptr }, ptr %self, i32 0, i32 0, !dbg !101
// CHECK:STDOUT:   call void @"_CSelfDestruct.76db170ab64c05aa:core.Destroy.Core"(ptr %.loc7_10.2.data), !dbg !101
// CHECK:STDOUT:   ret void, !dbg !101
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.a856f1753aa61283:core.Destroy.Core"(ptr %self) #0 !dbg !103 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.a856f1753aa61283:core.Destroy.Core"(ptr %self), !dbg !105
// CHECK:STDOUT:   ret void, !dbg !105
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core"(ptr %self) #0 !dbg !107 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_10.1.next = getelementptr inbounds nuw { i32, ptr }, ptr %self, i32 0, i32 1, !dbg !109
// CHECK:STDOUT:   call void @"_CSelfDestruct.6b633cf020343e69:core.Destroy.Core"(ptr %.loc7_10.1.next), !dbg !109
// CHECK:STDOUT:   %.loc7_10.2.data = getelementptr inbounds nuw { i32, ptr }, ptr %self, i32 0, i32 0, !dbg !109
// CHECK:STDOUT:   call void @"_CSelfDestruct.76db170ab64c05aa:core.Destroy.Core"(ptr %.loc7_10.2.data), !dbg !109
// CHECK:STDOUT:   ret void, !dbg !109
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core"(ptr %self) #0 !dbg !111 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core"(ptr %self), !dbg !113
// CHECK:STDOUT:   ret void, !dbg !113
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CMake.Main.e43630e9a6c38c3f(ptr sret({ i32, ptr }) %return) #0 !dbg !115 {
// CHECK:STDOUT:   %data = getelementptr inbounds nuw { i32, ptr }, ptr %return, i32 0, i32 0, !dbg !124
// CHECK:STDOUT:   %next = getelementptr inbounds nuw { i32, ptr }, ptr %return, i32 0, i32 1, !dbg !124
// CHECK:STDOUT:   store ptr %return, ptr %next, align 8, !dbg !124
// CHECK:STDOUT:   store i32 42, ptr %data, align 4, !dbg !124
// CHECK:STDOUT:   ret void, !dbg !125
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!85}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !30 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !31 = !{!30}
// CHECK:STDOUT: !32 = !DISubroutineType(types: !31)
// CHECK:STDOUT: !33 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !34 = !{null, !33}
// CHECK:STDOUT: !35 = !DISubroutineType(types: !34)
// CHECK:STDOUT: !36 = !{null, !30}
// CHECK:STDOUT: !37 = !DISubroutineType(types: !36)
// CHECK:STDOUT: !82 = !DIFile(filename: "return_var.carbon", directory: "")
// CHECK:STDOUT: !84 = !DIFile(filename: "use_return_var.carbon", directory: "")
// CHECK:STDOUT: !85 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !84, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !86 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !84, line: 6, type: !32, spFlags: DISPFlagDefinition, unit: !85)
// CHECK:STDOUT: !87 = !DILocation(line: 7, column: 3, scope: !86)
// CHECK:STDOUT: !88 = !{!33}
// CHECK:STDOUT: !89 = !DISubroutineType(types: !88)
// CHECK:STDOUT: !90 = !DILocation(line: 6, column: 1, scope: !86)
// CHECK:STDOUT: !91 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.76db170ab64c05aa:core.Destroy.Core", scope: null, file: !84, line: 7, type: !37, spFlags: DISPFlagDefinition, unit: !85, retainedNodes: !94)
// CHECK:STDOUT: !92 = !DILocalVariable(arg: 1, scope: !91, type: !30)
// CHECK:STDOUT: !93 = !DILocation(line: 7, column: 3, scope: !91)
// CHECK:STDOUT: !94 = !{!92}
// CHECK:STDOUT: !95 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6b633cf020343e69:core.Destroy.Core", scope: null, file: !84, line: 7, type: !35, spFlags: DISPFlagDefinition, unit: !85, retainedNodes: !98)
// CHECK:STDOUT: !96 = !DILocalVariable(arg: 1, scope: !95, type: !33)
// CHECK:STDOUT: !97 = !DILocation(line: 7, column: 3, scope: !95)
// CHECK:STDOUT: !98 = !{!96}
// CHECK:STDOUT: !99 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.a856f1753aa61283:core.Destroy.Core", scope: null, file: !84, line: 7, type: !35, spFlags: DISPFlagDefinition, unit: !85, retainedNodes: !102)
// CHECK:STDOUT: !100 = !DILocalVariable(arg: 1, scope: !99, type: !33)
// CHECK:STDOUT: !101 = !DILocation(line: 7, column: 3, scope: !99)
// CHECK:STDOUT: !102 = !{!100}
// CHECK:STDOUT: !103 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.a856f1753aa61283:core.Destroy.Core", scope: null, file: !84, line: 7, type: !35, spFlags: DISPFlagDefinition, unit: !85, retainedNodes: !106)
// CHECK:STDOUT: !104 = !DILocalVariable(arg: 1, scope: !103, type: !33)
// CHECK:STDOUT: !105 = !DILocation(line: 7, column: 3, scope: !103)
// CHECK:STDOUT: !106 = !{!104}
// CHECK:STDOUT: !107 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.08d9e38b928ba575:core.Destroy.Core", scope: null, file: !84, line: 7, type: !35, spFlags: DISPFlagDefinition, unit: !85, retainedNodes: !110)
// CHECK:STDOUT: !108 = !DILocalVariable(arg: 1, scope: !107, type: !33)
// CHECK:STDOUT: !109 = !DILocation(line: 7, column: 3, scope: !107)
// CHECK:STDOUT: !110 = !{!108}
// CHECK:STDOUT: !111 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.08d9e38b928ba575:core.Destroy.Core", scope: null, file: !84, line: 7, type: !35, spFlags: DISPFlagDefinition, unit: !85, retainedNodes: !114)
// CHECK:STDOUT: !112 = !DILocalVariable(arg: 1, scope: !111, type: !33)
// CHECK:STDOUT: !113 = !DILocation(line: 7, column: 3, scope: !111)
// CHECK:STDOUT: !114 = !{!112}
// CHECK:STDOUT: !115 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Main.e43630e9a6c38c3f", scope: null, file: !82, line: 9, type: !89, spFlags: DISPFlagDefinition, unit: !85)
// CHECK:STDOUT: !124 = !DILocation(line: 11, column: 7, scope: !115)
// CHECK:STDOUT: !125 = !DILocation(line: 12, column: 3, scope: !115)
// CHECK:STDOUT:
