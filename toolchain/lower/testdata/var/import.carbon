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

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'simple.carbon'
// CHECK:STDOUT: source_filename = "simple.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @_Cv.Main = global i32 0
// CHECK:STDOUT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @_C__global_init.Main, ptr null }]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Main() #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   store i32 42, ptr @_Cv.Main, align 4, !dbg !24
// CHECK:STDOUT:   ret void, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "simple.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !21 = !{null}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !18, type: !22, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !24 = !DILocation(line: 4, column: 1, scope: !23)
// CHECK:STDOUT: !25 = !DILocation(line: 0, scope: !23)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'import_simple.carbon'
// CHECK:STDOUT: source_filename = "import_simple.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @_Cv.Main = external global i32
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7 = load i32, ptr @_Cv.Main, align 4, !dbg !33
// CHECK:STDOUT:   ret i32 %.loc7, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!27}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !26 = !DIFile(filename: "import_simple.carbon", directory: "")
// CHECK:STDOUT: !27 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !26, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !28 = !{!20}
// CHECK:STDOUT: !29 = !DISubroutineType(types: !28)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !26, line: 6, type: !29, spFlags: DISPFlagDefinition, unit: !27)
// CHECK:STDOUT: !33 = !DILocation(line: 7, column: 10, scope: !30)
// CHECK:STDOUT: !34 = !DILocation(line: 7, column: 3, scope: !30)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'tuple_pattern.carbon'
// CHECK:STDOUT: source_filename = "tuple_pattern.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @_Cv.Main = global { i32, i32 } zeroinitializer
// CHECK:STDOUT: @_Cy.Main = global i32 0
// CHECK:STDOUT: @_Cz.Main = global { i32, i32 } zeroinitializer
// CHECK:STDOUT: @tuple.102.loc4_1 = internal constant { i32, i32 } { i32 1, i32 2 }
// CHECK:STDOUT: @tuple.54c.loc8_1 = internal constant { i32, i32 } { i32 5, i32 6 }
// CHECK:STDOUT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @_C__global_init.Main, ptr null }]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Main() #0 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 @_Cv.Main, ptr align 4 @tuple.102.loc4_1, i64 8, i1 false), !dbg !38
// CHECK:STDOUT:   store i32 4, ptr @_Cy.Main, align 4, !dbg !39
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 @_Cz.Main, ptr align 4 @tuple.54c.loc8_1, i64 8, i1 false), !dbg !40
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!36}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !21 = !{null}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !35 = !DIFile(filename: "tuple_pattern.carbon", directory: "")
// CHECK:STDOUT: !36 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !35, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !35, type: !22, spFlags: DISPFlagDefinition, unit: !36)
// CHECK:STDOUT: !38 = !DILocation(line: 4, column: 1, scope: !37)
// CHECK:STDOUT: !39 = !DILocation(line: 6, column: 14, scope: !37)
// CHECK:STDOUT: !40 = !DILocation(line: 8, column: 1, scope: !37)
// CHECK:STDOUT: !41 = !DILocation(line: 0, scope: !37)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'import_tuple_pattern.carbon'
// CHECK:STDOUT: source_filename = "import_tuple_pattern.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @_Cv.Main = external global { i32, i32 }
// CHECK:STDOUT: @_Cy.Main = external global i32
// CHECK:STDOUT: @_Cz.Main = external global { i32, i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CV.Main() #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc6_24 = load i32, ptr @_Cv.Main, align 4, !dbg !47
// CHECK:STDOUT:   ret i32 %.loc6_24, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CW.Main() #0 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_24 = load i32, ptr getelementptr inbounds nuw ({ i32, i32 }, ptr @_Cv.Main, i32 0, i32 1), align 4, !dbg !52
// CHECK:STDOUT:   ret i32 %.loc7_24, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CY.Main() #0 !dbg !54 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_24 = load i32, ptr @_Cy.Main, align 4, !dbg !57
// CHECK:STDOUT:   ret i32 %.loc9_24, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CZ.Main() #0 !dbg !59 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc10_24 = load i32, ptr getelementptr inbounds nuw ({ i32, i32 }, ptr @_Cz.Main, i32 0, i32 1), align 4, !dbg !62
// CHECK:STDOUT:   ret i32 %.loc10_24, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_Cv.Main, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!43}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !28 = !{!20}
// CHECK:STDOUT: !29 = !DISubroutineType(types: !28)
// CHECK:STDOUT: !42 = !DIFile(filename: "import_tuple_pattern.carbon", directory: "")
// CHECK:STDOUT: !43 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !42, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "V", linkageName: "_CV.Main", scope: null, file: !42, line: 6, type: !29, spFlags: DISPFlagDefinition, unit: !43)
// CHECK:STDOUT: !47 = !DILocation(line: 6, column: 24, scope: !44)
// CHECK:STDOUT: !48 = !DILocation(line: 6, column: 17, scope: !44)
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "W", linkageName: "_CW.Main", scope: null, file: !42, line: 7, type: !29, spFlags: DISPFlagDefinition, unit: !43)
// CHECK:STDOUT: !52 = !DILocation(line: 7, column: 24, scope: !49)
// CHECK:STDOUT: !53 = !DILocation(line: 7, column: 17, scope: !49)
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "Y", linkageName: "_CY.Main", scope: null, file: !42, line: 9, type: !29, spFlags: DISPFlagDefinition, unit: !43)
// CHECK:STDOUT: !57 = !DILocation(line: 9, column: 24, scope: !54)
// CHECK:STDOUT: !58 = !DILocation(line: 9, column: 17, scope: !54)
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "Z", linkageName: "_CZ.Main", scope: null, file: !42, line: 10, type: !29, spFlags: DISPFlagDefinition, unit: !43)
// CHECK:STDOUT: !62 = !DILocation(line: 10, column: 24, scope: !59)
// CHECK:STDOUT: !63 = !DILocation(line: 10, column: 17, scope: !59)
// CHECK:STDOUT:
