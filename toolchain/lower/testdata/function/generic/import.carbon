// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/import.carbon

// --- f.carbon

library "[[@TEST_NAME]]";

fn FDeclared();

fn FDefined() {}

fn F(T:! type, x: T*) -> T* {
  FDeclared();
  FDefined();
  return x;
}

// --- g.carbon

library "[[@TEST_NAME]]";

import library "f";

fn GDeclared();

fn GDefined() {}

fn G(T:! type, y: T*) -> T* {
  GDeclared();
  GDefined();
  return F(T, y);
}

// --- h.carbon

library "[[@TEST_NAME]]";

import library "g";

fn H() -> i32 {
  var n: i32 = 0;
  var p: i32* = G(i32, &n);
  return *p;
}

// CHECK:STDOUT: ; ModuleID = 'f.carbon'
// CHECK:STDOUT: source_filename = "f.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CFDeclared.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CFDefined.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "f.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "FDefined", linkageName: "_CFDefined.Main", scope: null, file: !3, line: 6, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 6, column: 1, scope: !4)
// CHECK:STDOUT: ; ModuleID = 'g.carbon'
// CHECK:STDOUT: source_filename = "g.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CGDeclared.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CGDefined.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "g.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "GDefined", linkageName: "_CGDefined.Main", scope: null, file: !3, line: 8, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 8, column: 1, scope: !4)
// CHECK:STDOUT: ; ModuleID = 'h.carbon'
// CHECK:STDOUT: source_filename = "h.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CH.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !7
// CHECK:STDOUT:   %p.var = alloca ptr, align 8, !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 4, ptr %n.var), !dbg !7
// CHECK:STDOUT:   store i32 0, ptr %n.var, align 4, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %p.var), !dbg !8
// CHECK:STDOUT:   %G.call = call ptr @_CG.Main.b88d1103f417c6d4(ptr %n.var), !dbg !9
// CHECK:STDOUT:   store ptr %G.call, ptr %p.var, align 8, !dbg !8
// CHECK:STDOUT:   %.loc9_11 = load ptr, ptr %p.var, align 8, !dbg !10
// CHECK:STDOUT:   %.loc9_10.2 = load i32, ptr %.loc9_11, align 4, !dbg !11
// CHECK:STDOUT:   ret i32 %.loc9_10.2, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr ptr @_CG.Main.b88d1103f417c6d4(ptr %y) !dbg !13 {
// CHECK:STDOUT:   call void @_CGDeclared.Main(), !dbg !15
// CHECK:STDOUT:   call void @_CGDefined.Main(), !dbg !16
// CHECK:STDOUT:   %1 = call ptr @_CF.Main.b88d1103f417c6d4(ptr %y), !dbg !17
// CHECK:STDOUT:   ret ptr %1, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CGDeclared.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CGDefined.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr ptr @_CF.Main.b88d1103f417c6d4(ptr %x) !dbg !19 {
// CHECK:STDOUT:   call void @_CFDeclared.Main(), !dbg !21
// CHECK:STDOUT:   call void @_CFDefined.Main(), !dbg !22
// CHECK:STDOUT:   ret ptr %x, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CFDeclared.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CFDefined.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "h.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main", scope: null, file: !3, line: 6, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 7, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 8, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 8, column: 17, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 9, column: 11, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 9, column: 10, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 9, column: 3, scope: !4)
// CHECK:STDOUT: !13 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.b88d1103f417c6d4", scope: null, file: !14, line: 10, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !14 = !DIFile(filename: "g.carbon", directory: "")
// CHECK:STDOUT: !15 = !DILocation(line: 11, column: 3, scope: !13)
// CHECK:STDOUT: !16 = !DILocation(line: 12, column: 3, scope: !13)
// CHECK:STDOUT: !17 = !DILocation(line: 13, column: 10, scope: !13)
// CHECK:STDOUT: !18 = !DILocation(line: 13, column: 3, scope: !13)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.b88d1103f417c6d4", scope: null, file: !20, line: 8, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !20 = !DIFile(filename: "f.carbon", directory: "")
// CHECK:STDOUT: !21 = !DILocation(line: 9, column: 3, scope: !19)
// CHECK:STDOUT: !22 = !DILocation(line: 10, column: 3, scope: !19)
// CHECK:STDOUT: !23 = !DILocation(line: 11, column: 3, scope: !19)
