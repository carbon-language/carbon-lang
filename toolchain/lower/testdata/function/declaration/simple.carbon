// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
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

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'simple.carbon'
// CHECK:STDOUT: source_filename = "simple.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Main(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main(i32 %n) #0 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main(i32 %n), !dbg !31
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CH.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CI.Main()
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
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !23 = !{null, !22}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !20, line: 15, type: !24, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !33)
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 1, scope: !27, type: !22)
// CHECK:STDOUT: !31 = !DILocation(line: 15, column: 16, scope: !27)
// CHECK:STDOUT: !32 = !DILocation(line: 15, column: 1, scope: !27)
// CHECK:STDOUT: !33 = !{!28}
// CHECK:STDOUT:
