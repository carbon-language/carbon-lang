// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/global/use.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/global/use.carbon
var a: i32 = 5;

var b: i32 = a;

fn F() -> i32 {
  return a;
}

// CHECK:STDOUT: ; ModuleID = 'use.carbon'
// CHECK:STDOUT: source_filename = "use.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @_Ca.Main = global i32 0
// CHECK:STDOUT: @_Cb.Main = global i32 0
// CHECK:STDOUT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @_C__global_init.Main, ptr null }]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CF.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17 = load i32, ptr @_Ca.Main, align 4, !dbg !8
// CHECK:STDOUT:   ret i32 %.loc17, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Main() #0 !dbg !10 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   store i32 5, ptr @_Ca.Main, align 4, !dbg !13
// CHECK:STDOUT:   %.loc14 = load i32, ptr @_Ca.Main, align 4, !dbg !14
// CHECK:STDOUT:   store i32 %.loc14, ptr @_Cb.Main, align 4, !dbg !15
// CHECK:STDOUT:   ret void, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "use.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !3, line: 16, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7}
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !DILocation(line: 17, column: 10, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 17, column: 3, scope: !4)
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !3, type: !11, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !11 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !12 = !{null}
// CHECK:STDOUT: !13 = !DILocation(line: 12, column: 1, scope: !10)
// CHECK:STDOUT: !14 = !DILocation(line: 14, column: 14, scope: !10)
// CHECK:STDOUT: !15 = !DILocation(line: 14, column: 1, scope: !10)
// CHECK:STDOUT: !16 = !DILocation(line: 0, scope: !10)
