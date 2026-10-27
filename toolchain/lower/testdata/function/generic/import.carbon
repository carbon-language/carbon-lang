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
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.4f29d15a20894e9b(ptr %p.var), !dbg !38
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %n.var), !dbg !36
// CHECK:STDOUT:   ret i32 %.loc9_10.2, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CDirectGeneric.Main.55456748a8a2eb70(ptr %y) #0 !dbg !53 {
// CHECK:STDOUT:   call void @_CDirectDeclared.Main(), !dbg !58
// CHECK:STDOUT:   call void @_CDirectDefined.Main(), !dbg !59
// CHECK:STDOUT:   %1 = call ptr @_CIndirectGeneric.Main.55456748a8a2eb70(ptr %y), !dbg !61
// CHECK:STDOUT:   ret ptr %1, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.4f29d15a20894e9b(ptr %t) #0 !dbg !64 {
// CHECK:STDOUT:   ret void, !dbg !69
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f127f30143696d51(ptr %t) #0 !dbg !71 {
// CHECK:STDOUT:   ret void, !dbg !76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CDirectDeclared.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CDirectDefined.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CIndirectGeneric.Main.55456748a8a2eb70(ptr %x) #0 !dbg !78 {
// CHECK:STDOUT:   call void @_CIndirectDeclared.Main(), !dbg !83
// CHECK:STDOUT:   call void @_CIndirectDefined.Main(), !dbg !84
// CHECK:STDOUT:   ret ptr %x, !dbg !86
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
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
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
// CHECK:STDOUT: !48 = !{null, !42}
// CHECK:STDOUT: !49 = !DISubroutineType(types: !48)
// CHECK:STDOUT: !50 = !{null, !30}
// CHECK:STDOUT: !51 = !DISubroutineType(types: !50)
// CHECK:STDOUT: !52 = !DILocation(line: 9, column: 3, scope: !33)
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "DirectGeneric", linkageName: "_CDirectGeneric.Main.55456748a8a2eb70", scope: null, file: !24, line: 10, type: !44, spFlags: DISPFlagDefinition, unit: !29, retainedNodes: !63)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !53, type: !42)
// CHECK:STDOUT: !58 = !DILocation(line: 11, column: 3, scope: !53)
// CHECK:STDOUT: !59 = !DILocation(line: 12, column: 3, scope: !53)
// CHECK:STDOUT: !61 = !DILocation(line: 13, column: 10, scope: !53)
// CHECK:STDOUT: !62 = !DILocation(line: 13, column: 3, scope: !53)
// CHECK:STDOUT: !63 = !{!54}
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.4f29d15a20894e9b", scope: null, file: !10, line: 32, type: !49, spFlags: DISPFlagDefinition, unit: !29, retainedNodes: !70)
// CHECK:STDOUT: !65 = !DILocalVariable(arg: 1, scope: !64, type: !42)
// CHECK:STDOUT: !69 = !DILocation(line: 32, column: 1, scope: !64)
// CHECK:STDOUT: !70 = !{!65}
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f127f30143696d51", scope: null, file: !10, line: 32, type: !51, spFlags: DISPFlagDefinition, unit: !29, retainedNodes: !77)
// CHECK:STDOUT: !72 = !DILocalVariable(arg: 1, scope: !71, type: !30)
// CHECK:STDOUT: !76 = !DILocation(line: 32, column: 1, scope: !71)
// CHECK:STDOUT: !77 = !{!72}
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "IndirectGeneric", linkageName: "_CIndirectGeneric.Main.55456748a8a2eb70", scope: null, file: !18, line: 8, type: !44, spFlags: DISPFlagDefinition, unit: !29, retainedNodes: !87)
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 1, scope: !78, type: !42)
// CHECK:STDOUT: !83 = !DILocation(line: 9, column: 3, scope: !78)
// CHECK:STDOUT: !84 = !DILocation(line: 10, column: 3, scope: !78)
// CHECK:STDOUT: !86 = !DILocation(line: 11, column: 3, scope: !78)
// CHECK:STDOUT: !87 = !{!79}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'return_var.carbon'
// CHECK:STDOUT: source_filename = "return_var.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!89}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !88 = !DIFile(filename: "return_var.carbon", directory: "")
// CHECK:STDOUT: !89 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !88, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'use_return_var.carbon'
// CHECK:STDOUT: source_filename = "use_return_var.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !92 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_10.2.temp = alloca { i32, ptr }, align 8, !dbg !93
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_10.2.temp), !dbg !93
// CHECK:STDOUT:   call void @_CMake.Main.e43630e9a6c38c3f(ptr %.loc7_10.2.temp), !dbg !93
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.356af3b6aef18564(ptr %.loc7_10.2.temp), !dbg !93
// CHECK:STDOUT:   ret i32 0, !dbg !96
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CMake.Main.e43630e9a6c38c3f(ptr sret({ i32, ptr }) %return) #0 !dbg !97 {
// CHECK:STDOUT:   %data = getelementptr inbounds nuw { i32, ptr }, ptr %return, i32 0, i32 0, !dbg !106
// CHECK:STDOUT:   %next = getelementptr inbounds nuw { i32, ptr }, ptr %return, i32 0, i32 1, !dbg !106
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!91}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !30 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !31 = !{!30}
// CHECK:STDOUT: !32 = !DISubroutineType(types: !31)
// CHECK:STDOUT: !42 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !48 = !{null, !42}
// CHECK:STDOUT: !49 = !DISubroutineType(types: !48)
// CHECK:STDOUT: !88 = !DIFile(filename: "return_var.carbon", directory: "")
// CHECK:STDOUT: !90 = !DIFile(filename: "use_return_var.carbon", directory: "")
// CHECK:STDOUT: !91 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !90, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !92 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !90, line: 6, type: !32, spFlags: DISPFlagDefinition, unit: !91)
// CHECK:STDOUT: !93 = !DILocation(line: 7, column: 3, scope: !92)
// CHECK:STDOUT: !94 = !{!42}
// CHECK:STDOUT: !95 = !DISubroutineType(types: !94)
// CHECK:STDOUT: !96 = !DILocation(line: 6, column: 1, scope: !92)
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Main.e43630e9a6c38c3f", scope: null, file: !88, line: 9, type: !95, spFlags: DISPFlagDefinition, unit: !91)
// CHECK:STDOUT: !106 = !DILocation(line: 11, column: 7, scope: !97)
// CHECK:STDOUT: !107 = !DILocation(line: 12, column: 3, scope: !97)
// CHECK:STDOUT: !108 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.356af3b6aef18564", scope: null, file: !10, line: 32, type: !49, spFlags: DISPFlagDefinition, unit: !91, retainedNodes: !114)
// CHECK:STDOUT: !109 = !DILocalVariable(arg: 1, scope: !108, type: !42)
// CHECK:STDOUT: !113 = !DILocation(line: 32, column: 1, scope: !108)
// CHECK:STDOUT: !114 = !{!109}
// CHECK:STDOUT:
