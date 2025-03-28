// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/declaration/simple.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/declaration/simple.carbon

fn F(n: i32);

fn G(n: i32) { F(n); }

class A;
fn H(n: A);
fn I() -> A;

// CHECK:STDOUT: ; ModuleID = 'simple.carbon'
// CHECK:STDOUT: source_filename = "simple.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Main(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CG.Main(i32 %n) !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main(i32 %n), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CH.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CI.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "simple.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !3, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 13, column: 16, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 13, column: 1, scope: !4)
