// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/var/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/var/import.carbon

// --- simple.carbon

library "[[@TEST_NAME]]";

var v: i32 = 42;

// --- import_simple.carbon

library "[[@TEST_NAME]]";

import library "simple";

fn Run() -> i32 {
  return v;
}

// --- tuple_pattern.carbon

library "[[@TEST_NAME]]";

var (v: i32, w: i32) = (1, 2);

let (x: i32, var y: i32) = (3, 4);

var (_: i32, z: i32) = (5, 6);

// --- import_tuple_pattern.carbon

library "[[@TEST_NAME]]";

import library "tuple_pattern";

fn V() -> i32 { return v; }
fn W() -> i32 { return w; }
// TODO: Also test `x`. Right now, lowering a reference to an imported `let` asserts.
fn Y() -> i32 { return y; }
fn Z() -> i32 { return z; }

// CHECK:STDOUT: ; ModuleID = 'simple.carbon'
// CHECK:STDOUT: source_filename = "simple.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @_Cv.Main = global i32 0
// CHECK:STDOUT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @_C__global_init.Main, ptr null }]
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_C__global_init.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   store i32 42, ptr @_Cv.Main, align 4, !dbg !7
// CHECK:STDOUT:   ret void, !dbg !8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "simple.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !3, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 4, column: 1, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 0, scope: !4)
// CHECK:STDOUT: ; ModuleID = 'import_simple.carbon'
// CHECK:STDOUT: source_filename = "import_simple.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @_Cv.Main = external global i32
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7 = load i32, ptr @_Cv.Main, align 4, !dbg !7
// CHECK:STDOUT:   ret i32 %.loc7, !dbg !8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "import_simple.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !3, line: 6, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 7, column: 10, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 7, column: 3, scope: !4)
// CHECK:STDOUT: ; ModuleID = 'tuple_pattern.carbon'
// CHECK:STDOUT: source_filename = "tuple_pattern.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @_Cv.Main = global { i32, i32 } zeroinitializer
// CHECK:STDOUT: @_Cy.Main = global i32 0
// CHECK:STDOUT: @_Cz.Main = global { i32, i32 } zeroinitializer
// CHECK:STDOUT: @tuple.21c.loc4_1 = internal constant { i32, i32 } { i32 1, i32 2 }
// CHECK:STDOUT: @tuple.092.loc8_1 = internal constant { i32, i32 } { i32 5, i32 6 }
// CHECK:STDOUT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @_C__global_init.Main, ptr null }]
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_C__global_init.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 @_Cv.Main, ptr align 4 @tuple.21c.loc4_1, i64 8, i1 false), !dbg !7
// CHECK:STDOUT:   store i32 4, ptr @_Cy.Main, align 4, !dbg !8
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 @_Cz.Main, ptr align 4 @tuple.092.loc8_1, i64 8, i1 false), !dbg !9
// CHECK:STDOUT:   ret void, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #0
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "tuple_pattern.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !3, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 4, column: 1, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 6, column: 14, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 8, column: 1, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 0, scope: !4)
// CHECK:STDOUT: ; ModuleID = 'import_tuple_pattern.carbon'
// CHECK:STDOUT: source_filename = "import_tuple_pattern.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @_Cv.Main = external global { i32, i32 }
// CHECK:STDOUT: @_Cy.Main = external global i32
// CHECK:STDOUT: @_Cz.Main = external global { i32, i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CV.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc6 = load i32, ptr @_Cv.Main, align 4, !dbg !7
// CHECK:STDOUT:   ret i32 %.loc6, !dbg !8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CW.Main() !dbg !9 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7 = load i32, ptr getelementptr inbounds nuw ({ i32, i32 }, ptr @_Cv.Main, i32 0, i32 1), align 4, !dbg !10
// CHECK:STDOUT:   ret i32 %.loc7, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CY.Main() !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9 = load i32, ptr @_Cy.Main, align 4, !dbg !13
// CHECK:STDOUT:   ret i32 %.loc9, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CZ.Main() !dbg !15 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc10 = load i32, ptr getelementptr inbounds nuw ({ i32, i32 }, ptr @_Cz.Main, i32 0, i32 1), align 4, !dbg !16
// CHECK:STDOUT:   ret i32 %.loc10, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_Cv.Main, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "import_tuple_pattern.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "V", linkageName: "_CV.Main", scope: null, file: !3, line: 6, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 6, column: 24, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 6, column: 17, scope: !4)
// CHECK:STDOUT: !9 = distinct !DISubprogram(name: "W", linkageName: "_CW.Main", scope: null, file: !3, line: 7, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !10 = !DILocation(line: 7, column: 24, scope: !9)
// CHECK:STDOUT: !11 = !DILocation(line: 7, column: 17, scope: !9)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Y", linkageName: "_CY.Main", scope: null, file: !3, line: 9, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !13 = !DILocation(line: 9, column: 24, scope: !12)
// CHECK:STDOUT: !14 = !DILocation(line: 9, column: 17, scope: !12)
// CHECK:STDOUT: !15 = distinct !DISubprogram(name: "Z", linkageName: "_CZ.Main", scope: null, file: !3, line: 10, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !16 = !DILocation(line: 10, column: 24, scope: !15)
// CHECK:STDOUT: !17 = !DILocation(line: 10, column: 17, scope: !15)
