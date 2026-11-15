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
fn X() -> i32 { return x; }
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
// CHECK:STDOUT: define internal void @_C__global_init.Main() #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   store i32 42, ptr @_Cv.Main, align 4, !dbg !26
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIFile(filename: "simple.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !23 = !{null}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !20, type: !24, spFlags: DISPFlagDefinition, unit: !21)
// CHECK:STDOUT: !26 = !DILocation(line: 4, column: 1, scope: !25)
// CHECK:STDOUT: !27 = !DILocation(line: 0, scope: !25)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'import_simple.carbon'
// CHECK:STDOUT: source_filename = "import_simple.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @_Cv.Main = external global i32
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7 = load i32, ptr @_Cv.Main, align 4, !dbg !35
// CHECK:STDOUT:   ret i32 %.loc7, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!29}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !28 = !DIFile(filename: "import_simple.carbon", directory: "")
// CHECK:STDOUT: !29 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !28, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !30 = !{!22}
// CHECK:STDOUT: !31 = !DISubroutineType(types: !30)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !28, line: 6, type: !31, spFlags: DISPFlagDefinition, unit: !29)
// CHECK:STDOUT: !35 = !DILocation(line: 7, column: 10, scope: !32)
// CHECK:STDOUT: !36 = !DILocation(line: 7, column: 3, scope: !32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'tuple_pattern.carbon'
// CHECK:STDOUT: source_filename = "tuple_pattern.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @_Cv.Main = global { i32, i32 } zeroinitializer
// CHECK:STDOUT: @_Cy.Main = global i32 0
// CHECK:STDOUT: @_Cz.Main = global { i32, i32 } zeroinitializer
// CHECK:STDOUT: @_Cx.Main = constant i32 3
// CHECK:STDOUT: @tuple.380.loc4_1 = internal constant { i32, i32 } { i32 1, i32 2 }
// CHECK:STDOUT: @tuple.da0.loc8_1 = internal constant { i32, i32 } { i32 5, i32 6 }
// CHECK:STDOUT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @_C__global_init.Main, ptr null }]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Main() #0 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 @_Cv.Main, ptr align 4 @tuple.380.loc4_1, i64 8, i1 false), !dbg !40
// CHECK:STDOUT:   store i32 4, ptr @_Cy.Main, align 4, !dbg !41
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 @_Cz.Main, ptr align 4 @tuple.da0.loc8_1, i64 8, i1 false), !dbg !42
// CHECK:STDOUT:   ret void, !dbg !43
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!38}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !23 = !{null}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !37 = !DIFile(filename: "tuple_pattern.carbon", directory: "")
// CHECK:STDOUT: !38 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !37, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !37, type: !24, spFlags: DISPFlagDefinition, unit: !38)
// CHECK:STDOUT: !40 = !DILocation(line: 4, column: 1, scope: !39)
// CHECK:STDOUT: !41 = !DILocation(line: 6, column: 14, scope: !39)
// CHECK:STDOUT: !42 = !DILocation(line: 8, column: 1, scope: !39)
// CHECK:STDOUT: !43 = !DILocation(line: 0, scope: !39)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'import_tuple_pattern.carbon'
// CHECK:STDOUT: source_filename = "import_tuple_pattern.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @_Cv.Main = external global { i32, i32 }
// CHECK:STDOUT: @_Cy.Main = external global i32
// CHECK:STDOUT: @_Cz.Main = external global { i32, i32 }
// CHECK:STDOUT: @_Cx.Main = external global i32
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CV.Main() #0 !dbg !46 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc6_24 = load i32, ptr @_Cv.Main, align 4, !dbg !49
// CHECK:STDOUT:   ret i32 %.loc6_24, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CW.Main() #0 !dbg !51 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_24 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_Cv.Main, i64 4), align 4, !dbg !54
// CHECK:STDOUT:   ret i32 %.loc7_24, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CX.Main() #0 !dbg !56 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.global_ref = load i32, ptr @_Cx.Main, align 4, !dbg !59
// CHECK:STDOUT:   ret i32 %x.global_ref, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CY.Main() #0 !dbg !61 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_24 = load i32, ptr @_Cy.Main, align 4, !dbg !64
// CHECK:STDOUT:   ret i32 %.loc9_24, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CZ.Main() #0 !dbg !66 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc10_24 = load i32, ptr getelementptr inbounds nuw (i8, ptr @_Cz.Main, i64 4), align 4, !dbg !69
// CHECK:STDOUT:   ret i32 %.loc10_24, !dbg !70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_Cv.Main, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!45}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !30 = !{!22}
// CHECK:STDOUT: !31 = !DISubroutineType(types: !30)
// CHECK:STDOUT: !44 = !DIFile(filename: "import_tuple_pattern.carbon", directory: "")
// CHECK:STDOUT: !45 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !44, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "V", linkageName: "_CV.Main", scope: null, file: !44, line: 6, type: !31, spFlags: DISPFlagDefinition, unit: !45)
// CHECK:STDOUT: !49 = !DILocation(line: 6, column: 24, scope: !46)
// CHECK:STDOUT: !50 = !DILocation(line: 6, column: 17, scope: !46)
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "W", linkageName: "_CW.Main", scope: null, file: !44, line: 7, type: !31, spFlags: DISPFlagDefinition, unit: !45)
// CHECK:STDOUT: !54 = !DILocation(line: 7, column: 24, scope: !51)
// CHECK:STDOUT: !55 = !DILocation(line: 7, column: 17, scope: !51)
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "X", linkageName: "_CX.Main", scope: null, file: !44, line: 8, type: !31, spFlags: DISPFlagDefinition, unit: !45)
// CHECK:STDOUT: !59 = !DILocation(line: 8, column: 24, scope: !56)
// CHECK:STDOUT: !60 = !DILocation(line: 8, column: 17, scope: !56)
// CHECK:STDOUT: !61 = distinct !DISubprogram(name: "Y", linkageName: "_CY.Main", scope: null, file: !44, line: 9, type: !31, spFlags: DISPFlagDefinition, unit: !45)
// CHECK:STDOUT: !64 = !DILocation(line: 9, column: 24, scope: !61)
// CHECK:STDOUT: !65 = !DILocation(line: 9, column: 17, scope: !61)
// CHECK:STDOUT: !66 = distinct !DISubprogram(name: "Z", linkageName: "_CZ.Main", scope: null, file: !44, line: 10, type: !31, spFlags: DISPFlagDefinition, unit: !45)
// CHECK:STDOUT: !69 = !DILocation(line: 10, column: 24, scope: !66)
// CHECK:STDOUT: !70 = !DILocation(line: 10, column: 17, scope: !66)
// CHECK:STDOUT:
