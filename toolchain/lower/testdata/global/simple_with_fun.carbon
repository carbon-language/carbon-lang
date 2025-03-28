// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

// CHECK:STDOUT: ; ModuleID = 'simple_with_fun.carbon'
// CHECK:STDOUT: source_filename = "simple_with_fun.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @a = internal global i32 0
// CHECK:STDOUT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @_C__global_init.Main, ptr null }]
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_Ctest_a.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 0, !dbg !7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_C__global_init.Main() !dbg !8 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %test_a.call = call i32 @_Ctest_a.Main(), !dbg !9
// CHECK:STDOUT:   store i32 %test_a.call, ptr @a, align 4, !dbg !10
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder i32 0, { 1, 0, 2 }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "simple_with_fun.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "test_a", linkageName: "_Ctest_a.Main", scope: null, file: !3, line: 11, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 12, column: 3, scope: !4)
// CHECK:STDOUT: !8 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !3, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !9 = !DILocation(line: 15, column: 14, scope: !8)
// CHECK:STDOUT: !10 = !DILocation(line: 15, column: 1, scope: !8)
// CHECK:STDOUT: !11 = !DILocation(line: 0, scope: !8)
