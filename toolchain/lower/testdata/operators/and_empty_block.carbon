// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/bool.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/operators/and_empty_block.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/operators/and_empty_block.carbon

fn And(b: bool, c: bool) -> bool {
  // Note that in this case, we must generate an empty block so that the `phi`
  // has two distinct predecessors.
  return b and c;
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'and_empty_block.carbon'
// CHECK:STDOUT: source_filename = "and_empty_block.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CAnd.Main(i1 %b, i1 %c) #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   br i1 %b, label %and.rhs, label %and.result, !dbg !30
// CHECK:STDOUT:
// CHECK:STDOUT: and.rhs:                                          ; preds = %entry
// CHECK:STDOUT:   br label %and.result, !dbg !30
// CHECK:STDOUT:
// CHECK:STDOUT: and.result:                                       ; preds = %and.rhs, %entry
// CHECK:STDOUT:   %0 = phi i1 [ false, %entry ], [ %c, %and.rhs ]
// CHECK:STDOUT:   ret i1 %0, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "and_empty_block.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{!20, !20, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "And", linkageName: "_CAnd.Main", scope: null, file: !18, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !33)
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !23, type: !20)
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 2, scope: !23, type: !20)
// CHECK:STDOUT: !30 = !DILocation(line: 16, column: 10, scope: !23)
// CHECK:STDOUT: !32 = !DILocation(line: 16, column: 3, scope: !23)
// CHECK:STDOUT: !33 = !{!24, !25}
// CHECK:STDOUT:
