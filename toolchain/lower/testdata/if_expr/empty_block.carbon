// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/if_expr/empty_block.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/if_expr/empty_block.carbon

fn Select(b: bool, c: bool, d: bool) -> i32 {
  return if b then if c then 1 as i32 else 2 else if d then 3 as i32 else 4;
}

// CHECK:STDOUT: ; ModuleID = 'empty_block.carbon'
// CHECK:STDOUT: source_filename = "empty_block.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CSelect.Main(i1 %b, i1 %c, i1 %d) !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   br i1 %b, label %if.expr.then.loc12_10, label %if.expr.else.loc12_10, !dbg !7
// CHECK:STDOUT:
// CHECK:STDOUT: if.expr.then.loc12_10:                            ; preds = %entry
// CHECK:STDOUT:   br i1 %c, label %if.expr.then.loc12_20, label %if.expr.else.loc12_20, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: if.expr.then.loc12_20:                            ; preds = %if.expr.then.loc12_10
// CHECK:STDOUT:   br label %if.expr.result.loc12_20, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: if.expr.else.loc12_20:                            ; preds = %if.expr.then.loc12_10
// CHECK:STDOUT:   br label %if.expr.result.loc12_20, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: if.expr.result.loc12_20:                          ; preds = %if.expr.else.loc12_20, %if.expr.then.loc12_20
// CHECK:STDOUT:   %0 = phi i32 [ 1, %if.expr.then.loc12_20 ], [ 2, %if.expr.else.loc12_20 ]
// CHECK:STDOUT:   br label %if.expr.result.loc12_10, !dbg !7
// CHECK:STDOUT:
// CHECK:STDOUT: if.expr.else.loc12_10:                            ; preds = %entry
// CHECK:STDOUT:   br i1 %d, label %if.expr.then.loc12_51, label %if.expr.else.loc12_51, !dbg !9
// CHECK:STDOUT:
// CHECK:STDOUT: if.expr.then.loc12_51:                            ; preds = %if.expr.else.loc12_10
// CHECK:STDOUT:   br label %if.expr.result.loc12_51, !dbg !9
// CHECK:STDOUT:
// CHECK:STDOUT: if.expr.else.loc12_51:                            ; preds = %if.expr.else.loc12_10
// CHECK:STDOUT:   br label %if.expr.result.loc12_51, !dbg !9
// CHECK:STDOUT:
// CHECK:STDOUT: if.expr.result.loc12_51:                          ; preds = %if.expr.else.loc12_51, %if.expr.then.loc12_51
// CHECK:STDOUT:   %1 = phi i32 [ 3, %if.expr.then.loc12_51 ], [ 4, %if.expr.else.loc12_51 ]
// CHECK:STDOUT:   br label %if.expr.result.loc12_10, !dbg !7
// CHECK:STDOUT:
// CHECK:STDOUT: if.expr.result.loc12_10:                          ; preds = %if.expr.result.loc12_51, %if.expr.result.loc12_20
// CHECK:STDOUT:   %2 = phi i32 [ %0, %if.expr.result.loc12_20 ], [ %1, %if.expr.result.loc12_51 ]
// CHECK:STDOUT:   ret i32 %2, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "empty_block.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Select", linkageName: "_CSelect.Main", scope: null, file: !3, line: 11, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 12, column: 10, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 12, column: 20, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 12, column: 51, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 12, column: 3, scope: !4)
