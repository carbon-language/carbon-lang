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
// CHECK:STDOUT: define void @_CIndirectDefined.Main() #0 !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIFile(filename: "indirectly_imported.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !22 = !{null}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "IndirectDefined", linkageName: "_CIndirectDefined.Main", scope: null, file: !20, line: 6, type: !23, spFlags: DISPFlagDefinition, unit: !21)
// CHECK:STDOUT: !25 = !DILocation(line: 6, column: 1, scope: !24)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'directly_imported.carbon'
// CHECK:STDOUT: source_filename = "directly_imported.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CDirectDeclared.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CDirectDefined.Main() #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!27}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !22 = !{null}
// CHECK:STDOUT: !23 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !26 = !DIFile(filename: "directly_imported.carbon", directory: "")
// CHECK:STDOUT: !27 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !26, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "DirectDefined", linkageName: "_CDirectDefined.Main", scope: null, file: !26, line: 8, type: !23, spFlags: DISPFlagDefinition, unit: !27)
// CHECK:STDOUT: !29 = !DILocation(line: 8, column: 1, scope: !28)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'use.carbon'
// CHECK:STDOUT: source_filename = "use.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CCall.Main() #0 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !38
// CHECK:STDOUT:   %p.var = alloca ptr, align 8, !dbg !40
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !38
// CHECK:STDOUT:   store i32 0, ptr %n.var, align 4, !dbg !38
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %p.var), !dbg !40
// CHECK:STDOUT:   %DirectGeneric.call = call ptr @_CDirectGeneric.Main.55456748a8a2eb70(ptr %n.var), !dbg !43
// CHECK:STDOUT:   store ptr %DirectGeneric.call, ptr %p.var, align 8, !dbg !40
// CHECK:STDOUT:   %.loc9_11 = load ptr, ptr %p.var, align 8, !dbg !48
// CHECK:STDOUT:   %.loc9_10.2 = load i32, ptr %.loc9_11, align 4, !dbg !49
// CHECK:STDOUT:   ret i32 %.loc9_10.2, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CDirectGeneric.Main.55456748a8a2eb70(ptr %y) #0 !dbg !51 {
// CHECK:STDOUT:   call void @_CDirectDeclared.Main(), !dbg !56
// CHECK:STDOUT:   call void @_CDirectDefined.Main(), !dbg !57
// CHECK:STDOUT:   %1 = call ptr @_CIndirectGeneric.Main.55456748a8a2eb70(ptr %y), !dbg !59
// CHECK:STDOUT:   ret ptr %1, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CDirectDeclared.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CDirectDefined.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CIndirectGeneric.Main.55456748a8a2eb70(ptr %x) #0 !dbg !62 {
// CHECK:STDOUT:   call void @_CIndirectDeclared.Main(), !dbg !67
// CHECK:STDOUT:   call void @_CIndirectDefined.Main(), !dbg !68
// CHECK:STDOUT:   ret ptr %x, !dbg !70
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!31}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIFile(filename: "indirectly_imported.carbon", directory: "")
// CHECK:STDOUT: !26 = !DIFile(filename: "directly_imported.carbon", directory: "")
// CHECK:STDOUT: !30 = !DIFile(filename: "use.carbon", directory: "")
// CHECK:STDOUT: !31 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !30, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !32 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !33 = !{!32}
// CHECK:STDOUT: !34 = !DISubroutineType(types: !33)
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !30, line: 6, type: !34, spFlags: DISPFlagDefinition, unit: !31)
// CHECK:STDOUT: !38 = !DILocation(line: 7, column: 3, scope: !35)
// CHECK:STDOUT: !40 = !DILocation(line: 8, column: 3, scope: !35)
// CHECK:STDOUT: !43 = !DILocation(line: 8, column: 17, scope: !35)
// CHECK:STDOUT: !44 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !45 = !{!44, !44}
// CHECK:STDOUT: !46 = !DISubroutineType(types: !45)
// CHECK:STDOUT: !48 = !DILocation(line: 9, column: 11, scope: !35)
// CHECK:STDOUT: !49 = !DILocation(line: 9, column: 10, scope: !35)
// CHECK:STDOUT: !50 = !DILocation(line: 9, column: 3, scope: !35)
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "DirectGeneric", linkageName: "_CDirectGeneric.Main.55456748a8a2eb70", scope: null, file: !26, line: 10, type: !46, spFlags: DISPFlagDefinition, unit: !31, retainedNodes: !61)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 1, scope: !51, type: !44)
// CHECK:STDOUT: !56 = !DILocation(line: 11, column: 3, scope: !51)
// CHECK:STDOUT: !57 = !DILocation(line: 12, column: 3, scope: !51)
// CHECK:STDOUT: !59 = !DILocation(line: 13, column: 10, scope: !51)
// CHECK:STDOUT: !60 = !DILocation(line: 13, column: 3, scope: !51)
// CHECK:STDOUT: !61 = !{!52}
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "IndirectGeneric", linkageName: "_CIndirectGeneric.Main.55456748a8a2eb70", scope: null, file: !20, line: 8, type: !46, spFlags: DISPFlagDefinition, unit: !31, retainedNodes: !71)
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !62, type: !44)
// CHECK:STDOUT: !67 = !DILocation(line: 9, column: 3, scope: !62)
// CHECK:STDOUT: !68 = !DILocation(line: 10, column: 3, scope: !62)
// CHECK:STDOUT: !70 = !DILocation(line: 11, column: 3, scope: !62)
// CHECK:STDOUT: !71 = !{!63}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'return_var.carbon'
// CHECK:STDOUT: source_filename = "return_var.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!73}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !72 = !DIFile(filename: "return_var.carbon", directory: "")
// CHECK:STDOUT: !73 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !72, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'use_return_var.carbon'
// CHECK:STDOUT: source_filename = "use_return_var.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !76 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_10.2.temp = alloca { i32, ptr }, align 8, !dbg !77
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_10.2.temp), !dbg !77
// CHECK:STDOUT:   call void @_CMake.Main.e43630e9a6c38c3f(ptr %.loc7_10.2.temp), !dbg !77
// CHECK:STDOUT:   ret i32 0, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CMake.Main.e43630e9a6c38c3f(ptr sret({ i32, ptr }) %return) #0 !dbg !81 {
// CHECK:STDOUT:   %data = getelementptr inbounds nuw { i32, ptr }, ptr %return, i32 0, i32 0, !dbg !90
// CHECK:STDOUT:   %next = getelementptr inbounds nuw { i32, ptr }, ptr %return, i32 0, i32 1, !dbg !90
// CHECK:STDOUT:   store ptr %return, ptr %next, align 8, !dbg !90
// CHECK:STDOUT:   store i32 42, ptr %data, align 4, !dbg !90
// CHECK:STDOUT:   ret void, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!75}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !32 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !33 = !{!32}
// CHECK:STDOUT: !34 = !DISubroutineType(types: !33)
// CHECK:STDOUT: !44 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !72 = !DIFile(filename: "return_var.carbon", directory: "")
// CHECK:STDOUT: !74 = !DIFile(filename: "use_return_var.carbon", directory: "")
// CHECK:STDOUT: !75 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !74, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !76 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !74, line: 6, type: !34, spFlags: DISPFlagDefinition, unit: !75)
// CHECK:STDOUT: !77 = !DILocation(line: 7, column: 3, scope: !76)
// CHECK:STDOUT: !78 = !{!44}
// CHECK:STDOUT: !79 = !DISubroutineType(types: !78)
// CHECK:STDOUT: !80 = !DILocation(line: 6, column: 1, scope: !76)
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Main.e43630e9a6c38c3f", scope: null, file: !72, line: 9, type: !79, spFlags: DISPFlagDefinition, unit: !75)
// CHECK:STDOUT: !90 = !DILocation(line: 11, column: 7, scope: !81)
// CHECK:STDOUT: !91 = !DILocation(line: 12, column: 3, scope: !81)
// CHECK:STDOUT:
