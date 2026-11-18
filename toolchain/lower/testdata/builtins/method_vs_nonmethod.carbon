// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/builtins/method_vs_nonmethod.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/builtins/method_vs_nonmethod.carbon

fn AddNonmethod(a: i32, b: i32) -> i32 = "int.sadd";
fn AddMethod(self: i32, b: i32) -> i32 = "int.sadd";

fn TestAddNonmethod(a: i32, b: i32) -> i32 { return AddNonmethod(a, b); }
fn TestAddMethod(a: i32, b: i32) -> i32 { return a.(AddMethod)(b); }

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'method_vs_nonmethod.carbon'
// CHECK:STDOUT: source_filename = "method_vs_nonmethod.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestAddNonmethod.Main(i32 %a, i32 %b) #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %AddNonmethod.call = add i32 %a, %b, !dbg !32
// CHECK:STDOUT:   ret i32 %AddNonmethod.call, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestAddMethod.Main(i32 %a, i32 %b) #0 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %AddMethod.call = add i32 %a, %b, !dbg !42
// CHECK:STDOUT:   ret i32 %AddMethod.call, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "method_vs_nonmethod.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{!20, !20, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "TestAddNonmethod", linkageName: "_CTestAddNonmethod.Main", scope: null, file: !18, line: 16, type: !22, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !34)
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !23, type: !20)
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 2, scope: !23, type: !20)
// CHECK:STDOUT: !32 = !DILocation(line: 16, column: 53, scope: !23)
// CHECK:STDOUT: !33 = !DILocation(line: 16, column: 46, scope: !23)
// CHECK:STDOUT: !34 = !{!24, !25}
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "TestAddMethod", linkageName: "_CTestAddMethod.Main", scope: null, file: !18, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !45)
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !35, type: !20)
// CHECK:STDOUT: !37 = !DILocalVariable(arg: 2, scope: !35, type: !20)
// CHECK:STDOUT: !42 = !DILocation(line: 17, column: 50, scope: !35)
// CHECK:STDOUT: !44 = !DILocation(line: 17, column: 43, scope: !35)
// CHECK:STDOUT: !45 = !{!36, !37}
// CHECK:STDOUT:
