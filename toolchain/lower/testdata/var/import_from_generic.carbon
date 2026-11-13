// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/var/import_from_generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/var/import_from_generic.carbon

// --- lib.carbon

library "[[@TEST_NAME]]";

var g: i32 = 1;

fn Get[T: type](_: T) -> i32 { return g; }

// --- use.carbon

library "[[@TEST_NAME]]";

import library "lib";

// The direct reference to `g` and the reference from the specific `Get(())`
// should both refer to the same global.
fn F() -> i32 { return g; }
fn G() -> i32 { return Get(()); }

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'lib.carbon'
// CHECK:STDOUT: source_filename = "lib.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @_Cg.Main = global i32 0
// CHECK:STDOUT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @_C__global_init.Main, ptr null }]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Main() #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   store i32 1, ptr @_Cg.Main, align 4, !dbg !24
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
// CHECK:STDOUT: !18 = !DIFile(filename: "lib.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !21 = !{null}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !18, type: !22, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !24 = !DILocation(line: 4, column: 1, scope: !23)
// CHECK:STDOUT: !25 = !DILocation(line: 0, scope: !23)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'use.carbon'
// CHECK:STDOUT: source_filename = "use.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @_Cg.Main = external global i32
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CF.Main() #0 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc8_24 = load i32, ptr @_Cg.Main, align 4, !dbg !33
// CHECK:STDOUT:   ret i32 %.loc8_24, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CG.Main() #0 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Get.call = call i32 @_CGet.Main.e43630e9a6c38c3f(), !dbg !38
// CHECK:STDOUT:   ret i32 %Get.call, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CGet.Main.e43630e9a6c38c3f() #0 !dbg !40 {
// CHECK:STDOUT:   %1 = load i32, ptr @_Cg.Main, align 4, !dbg !44
// CHECK:STDOUT:   ret i32 %1, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!27}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "lib.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !26 = !DIFile(filename: "use.carbon", directory: "")
// CHECK:STDOUT: !27 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !26, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !28 = !{!20}
// CHECK:STDOUT: !29 = !DISubroutineType(types: !28)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !26, line: 8, type: !29, spFlags: DISPFlagDefinition, unit: !27)
// CHECK:STDOUT: !33 = !DILocation(line: 8, column: 24, scope: !30)
// CHECK:STDOUT: !34 = !DILocation(line: 8, column: 17, scope: !30)
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !26, line: 9, type: !29, spFlags: DISPFlagDefinition, unit: !27)
// CHECK:STDOUT: !38 = !DILocation(line: 9, column: 24, scope: !35)
// CHECK:STDOUT: !39 = !DILocation(line: 9, column: 17, scope: !35)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Main.e43630e9a6c38c3f", scope: null, file: !18, line: 6, type: !29, spFlags: DISPFlagDefinition, unit: !27)
// CHECK:STDOUT: !44 = !DILocation(line: 6, column: 39, scope: !40)
// CHECK:STDOUT: !45 = !DILocation(line: 6, column: 32, scope: !40)
// CHECK:STDOUT:
