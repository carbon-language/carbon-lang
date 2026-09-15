// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/bool.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/operators/or_empty_block.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/operators/or_empty_block.carbon

fn Or(b: bool, c: bool) -> bool {
  // Note that in this case, we must generate an empty block so that the `phi`
  // has two distinct predecessors.
  return b or c;
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'or_empty_block.carbon'
// CHECK:STDOUT: source_filename = "or_empty_block.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_COr.Main(i1 %b, i1 %c) #0 !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc16_12.1 = xor i1 %b, true, !dbg !28
// CHECK:STDOUT:   br i1 %.loc16_12.1, label %or.rhs, label %or.result, !dbg !28
// CHECK:STDOUT:
// CHECK:STDOUT: or.rhs:                                           ; preds = %entry
// CHECK:STDOUT:   br label %or.result, !dbg !28
// CHECK:STDOUT:
// CHECK:STDOUT: or.result:                                        ; preds = %or.rhs, %entry
// CHECK:STDOUT:   %0 = phi i1 [ true, %entry ], [ %c, %or.rhs ]
// CHECK:STDOUT:   ret i1 %0, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!17}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !16 = !DIFile(filename: "or_empty_block.carbon", directory: "")
// CHECK:STDOUT: !17 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !16, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !19 = !{!18, !18, !18}
// CHECK:STDOUT: !20 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "Or", linkageName: "_COr.Main", scope: null, file: !16, line: 13, type: !20, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !31)
// CHECK:STDOUT: !22 = !DILocalVariable(arg: 1, scope: !21, type: !18)
// CHECK:STDOUT: !23 = !DILocalVariable(arg: 2, scope: !21, type: !18)
// CHECK:STDOUT: !28 = !DILocation(line: 16, column: 10, scope: !21)
// CHECK:STDOUT: !30 = !DILocation(line: 16, column: 3, scope: !21)
// CHECK:STDOUT: !31 = !{!22, !23}
// CHECK:STDOUT:
