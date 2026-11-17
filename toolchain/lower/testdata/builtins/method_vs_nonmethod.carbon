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
// CHECK:STDOUT: define i32 @_CTestAddNonmethod.Main(i32 %a, i32 %b) #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %AddNonmethod.call = add i32 %a, %b, !dbg !34
// CHECK:STDOUT:   ret i32 %AddNonmethod.call, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestAddMethod.Main(i32 %a, i32 %b) #0 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %AddMethod.call = add i32 %a, %b, !dbg !44
// CHECK:STDOUT:   ret i32 %AddMethod.call, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIFile(filename: "method_vs_nonmethod.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !23 = !{!22, !22, !22}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "TestAddNonmethod", linkageName: "_CTestAddNonmethod.Main", scope: null, file: !20, line: 16, type: !24, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !36)
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 1, scope: !25, type: !22)
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 2, scope: !25, type: !22)
// CHECK:STDOUT: !34 = !DILocation(line: 16, column: 53, scope: !25)
// CHECK:STDOUT: !35 = !DILocation(line: 16, column: 46, scope: !25)
// CHECK:STDOUT: !36 = !{!26, !27}
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "TestAddMethod", linkageName: "_CTestAddMethod.Main", scope: null, file: !20, line: 17, type: !24, spFlags: DISPFlagDefinition, unit: !21, retainedNodes: !47)
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !37, type: !22)
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 2, scope: !37, type: !22)
// CHECK:STDOUT: !44 = !DILocation(line: 17, column: 50, scope: !37)
// CHECK:STDOUT: !46 = !DILocation(line: 17, column: 43, scope: !37)
// CHECK:STDOUT: !47 = !{!38, !39}
// CHECK:STDOUT:
