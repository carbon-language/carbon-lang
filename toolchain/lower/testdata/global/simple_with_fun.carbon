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
// CHECK:STDOUT: @a = internal global i32
// CHECK:STDOUT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @__global_init, ptr null }]
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @test_a() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @__global_init() !dbg !7 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %test_a.call = call i32 @test_a()
// CHECK:STDOUT:   store i32 %test_a.call, ptr @a, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder i32 0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "simple_with_fun.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "test_a", scope: null, file: !3, line: 11, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "__global_init", scope: null, file: !3, line: 4294967295, type: !5, spFlags: DISPFlagDefinition, unit: !2)
