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
// CHECK:STDOUT: define i32 @_CCall.Main() #0 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !36
// CHECK:STDOUT:   %p.var = alloca ptr, align 8, !dbg !38
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !36
// CHECK:STDOUT:   store i32 0, ptr %n.var, align 4, !dbg !36
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %p.var), !dbg !38
// CHECK:STDOUT:   %DirectGeneric.call = call ptr @_CDirectGeneric.Main.55456748a8a2eb70(ptr %n.var), !dbg !41
// CHECK:STDOUT:   store ptr %DirectGeneric.call, ptr %p.var, align 8, !dbg !38
// CHECK:STDOUT:   %.loc9_11 = load ptr, ptr %p.var, align 8, !dbg !46
// CHECK:STDOUT:   %.loc9_10.2 = load i32, ptr %.loc9_11, align 4, !dbg !47
// CHECK:STDOUT:   ret i32 %.loc9_10.2, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CDirectGeneric.Main.55456748a8a2eb70(ptr %y) #0 !dbg !49 {
// CHECK:STDOUT:   call void @_CDirectDeclared.Main(), !dbg !54
// CHECK:STDOUT:   call void @_CDirectDefined.Main(), !dbg !55
// CHECK:STDOUT:   %1 = call ptr @_CIndirectGeneric.Main.55456748a8a2eb70(ptr %y), !dbg !57
// CHECK:STDOUT:   ret ptr %1, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CDirectDeclared.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CDirectDefined.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CIndirectGeneric.Main.55456748a8a2eb70(ptr %x) #0 !dbg !60 {
// CHECK:STDOUT:   call void @_CIndirectDeclared.Main(), !dbg !65
// CHECK:STDOUT:   call void @_CIndirectDefined.Main(), !dbg !66
// CHECK:STDOUT:   ret ptr %x, !dbg !68
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
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !28, line: 6, type: !32, spFlags: DISPFlagDefinition, unit: !29)
// CHECK:STDOUT: !36 = !DILocation(line: 7, column: 3, scope: !33)
// CHECK:STDOUT: !38 = !DILocation(line: 8, column: 3, scope: !33)
// CHECK:STDOUT: !41 = !DILocation(line: 8, column: 17, scope: !33)
// CHECK:STDOUT: !42 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !43 = !{!42, !42}
// CHECK:STDOUT: !44 = !DISubroutineType(types: !43)
// CHECK:STDOUT: !46 = !DILocation(line: 9, column: 11, scope: !33)
// CHECK:STDOUT: !47 = !DILocation(line: 9, column: 10, scope: !33)
// CHECK:STDOUT: !48 = !DILocation(line: 9, column: 3, scope: !33)
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "DirectGeneric", linkageName: "_CDirectGeneric.Main.55456748a8a2eb70", scope: null, file: !24, line: 10, type: !44, spFlags: DISPFlagDefinition, unit: !29, retainedNodes: !59)
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 1, scope: !49, type: !42)
// CHECK:STDOUT: !54 = !DILocation(line: 11, column: 3, scope: !49)
// CHECK:STDOUT: !55 = !DILocation(line: 12, column: 3, scope: !49)
// CHECK:STDOUT: !57 = !DILocation(line: 13, column: 10, scope: !49)
// CHECK:STDOUT: !58 = !DILocation(line: 13, column: 3, scope: !49)
// CHECK:STDOUT: !59 = !{!50}
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "IndirectGeneric", linkageName: "_CIndirectGeneric.Main.55456748a8a2eb70", scope: null, file: !18, line: 8, type: !44, spFlags: DISPFlagDefinition, unit: !29, retainedNodes: !69)
// CHECK:STDOUT: !61 = !DILocalVariable(arg: 1, scope: !60, type: !42)
// CHECK:STDOUT: !65 = !DILocation(line: 9, column: 3, scope: !60)
// CHECK:STDOUT: !66 = !DILocation(line: 10, column: 3, scope: !60)
// CHECK:STDOUT: !68 = !DILocation(line: 11, column: 3, scope: !60)
// CHECK:STDOUT: !69 = !{!61}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'return_var.carbon'
// CHECK:STDOUT: source_filename = "return_var.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!71}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !70 = !DIFile(filename: "return_var.carbon", directory: "")
// CHECK:STDOUT: !71 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !70, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'use_return_var.carbon'
// CHECK:STDOUT: source_filename = "use_return_var.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !74 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_10.2.temp = alloca { i32, ptr }, align 8, !dbg !75
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_10.2.temp), !dbg !75
// CHECK:STDOUT:   call void @_CMake.Main.e43630e9a6c38c3f(ptr %.loc7_10.2.temp), !dbg !75
// CHECK:STDOUT:   ret i32 0, !dbg !78
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CMake.Main.e43630e9a6c38c3f(ptr sret({ i32, ptr }) %return) #0 !dbg !97 {
// CHECK:STDOUT:   %data = getelementptr inbounds nuw i8, ptr %return, i64 0, !dbg !106
// CHECK:STDOUT:   %next = getelementptr inbounds nuw i8, ptr %return, i64 8, !dbg !106
// CHECK:STDOUT:   store ptr %return, ptr %next, align 8, !dbg !106
// CHECK:STDOUT:   store i32 42, ptr %data, align 4, !dbg !106
// CHECK:STDOUT:   ret void, !dbg !107
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.356af3b6aef18564(ptr %t) #0 !dbg !108 {
// CHECK:STDOUT:   ret void, !dbg !113
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!73}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !30 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !31 = !{!30}
// CHECK:STDOUT: !32 = !DISubroutineType(types: !31)
// CHECK:STDOUT: !42 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !70 = !DIFile(filename: "return_var.carbon", directory: "")
// CHECK:STDOUT: !72 = !DIFile(filename: "use_return_var.carbon", directory: "")
// CHECK:STDOUT: !73 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !72, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !74 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !72, line: 6, type: !32, spFlags: DISPFlagDefinition, unit: !73)
// CHECK:STDOUT: !75 = !DILocation(line: 7, column: 3, scope: !74)
// CHECK:STDOUT: !76 = !{!42}
// CHECK:STDOUT: !77 = !DISubroutineType(types: !76)
// CHECK:STDOUT: !78 = !DILocation(line: 6, column: 1, scope: !74)
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Main.e43630e9a6c38c3f", scope: null, file: !70, line: 9, type: !77, spFlags: DISPFlagDefinition, unit: !73)
// CHECK:STDOUT: !88 = !DILocation(line: 11, column: 7, scope: !79)
// CHECK:STDOUT: !89 = !DILocation(line: 12, column: 3, scope: !79)
// CHECK:STDOUT:
