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
// CHECK:STDOUT: define i32 @_Ctest_a.Main() #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 0, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Main() #0 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %test_a.call = call i32 @_Ctest_a.Main(), !dbg !30
// CHECK:STDOUT:   store i32 %test_a.call, ptr @_Ca.Main, align 4, !dbg !31
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "simple_with_fun.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{!20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "test_a", linkageName: "_Ctest_a.Main", scope: null, file: !18, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !26 = !DILocation(line: 14, column: 3, scope: !23)
// CHECK:STDOUT: !27 = !{null}
// CHECK:STDOUT: !28 = !DISubroutineType(types: !27)
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !18, type: !28, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !30 = !DILocation(line: 17, column: 14, scope: !29)
// CHECK:STDOUT: !31 = !DILocation(line: 17, column: 1, scope: !29)
// CHECK:STDOUT: !32 = !DILocation(line: 0, scope: !29)
// CHECK:STDOUT:
