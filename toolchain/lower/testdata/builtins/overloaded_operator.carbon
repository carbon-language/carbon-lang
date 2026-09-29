// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/builtins/overloaded_operator.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/builtins/overloaded_operator.carbon

fn AddThreeIntegers(a: i32, b: i32, c: i32) -> i32 {
  return a + b + c;
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'overloaded_operator.carbon'
// CHECK:STDOUT: source_filename = "overloaded_operator.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CAddThreeIntegers.Main(i32 %a, i32 %b, i32 %c) #0 !dbg !9 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call.loc14_12 = add i32 %a, %b, !dbg !18
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call.loc14_16 = add i32 %Int.as.AddWith.impl.Op.call.loc14_12, %c, !dbg !18
// CHECK:STDOUT:   ret i32 %Int.as.AddWith.impl.Op.call.loc14_16, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "overloaded_operator.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !{!6, !6, !6, !6}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !9 = distinct !DISubprogram(name: "AddThreeIntegers", linkageName: "_CAddThreeIntegers.Main", scope: null, file: !4, line: 13, type: !8, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !22)
// CHECK:STDOUT: !10 = !DILocalVariable(arg: 1, scope: !9, type: !6)
// CHECK:STDOUT: !11 = !DILocalVariable(arg: 2, scope: !9, type: !6)
// CHECK:STDOUT: !12 = !DILocalVariable(arg: 3, scope: !9, type: !6)
// CHECK:STDOUT: !18 = !DILocation(line: 14, column: 10, scope: !9)
// CHECK:STDOUT: !21 = !DILocation(line: 14, column: 3, scope: !9)
// CHECK:STDOUT: !22 = !{!10, !11, !12}
// CHECK:STDOUT:
