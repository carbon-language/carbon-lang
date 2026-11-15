// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/global/simple_with_fun.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/global/simple_with_fun.carbon

fn test_a() -> i32 {
  return 0;
}

var a: i32 = test_a();

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'simple_with_fun.carbon'
// CHECK:STDOUT: source_filename = "simple_with_fun.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @_Ca.Main = global i32 0
// CHECK:STDOUT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @_C__global_init.Main, ptr null }]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_Ctest_a.Main() #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 0, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Main() #0 !dbg !31 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %test_a.call = call i32 @_Ctest_a.Main(), !dbg !32
// CHECK:STDOUT:   store i32 %test_a.call, ptr @_Ca.Main, align 4, !dbg !33
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIFile(filename: "simple_with_fun.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !23 = !{!22}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "test_a", linkageName: "_Ctest_a.Main", scope: null, file: !20, line: 13, type: !24, spFlags: DISPFlagDefinition, unit: !21)
// CHECK:STDOUT: !28 = !DILocation(line: 14, column: 3, scope: !25)
// CHECK:STDOUT: !29 = !{null}
// CHECK:STDOUT: !30 = !DISubroutineType(types: !29)
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !20, type: !30, spFlags: DISPFlagDefinition, unit: !21)
// CHECK:STDOUT: !32 = !DILocation(line: 17, column: 14, scope: !31)
// CHECK:STDOUT: !33 = !DILocation(line: 17, column: 1, scope: !31)
// CHECK:STDOUT: !34 = !DILocation(line: 0, scope: !31)
// CHECK:STDOUT:
