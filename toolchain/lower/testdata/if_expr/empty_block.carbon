// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/if_expr/empty_block.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/if_expr/empty_block.carbon

fn Select(b: bool, c: bool, d: bool) -> i32 {
  return if b then if c then 1 as i32 else 2 else if d then 3 as i32 else 4;
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'empty_block.carbon'
// CHECK:STDOUT: source_filename = "empty_block.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CSelect.Main(i1 %b, i1 %c, i1 %d) #0 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   br i1 %b, label %if.expr.then.loc14_10, label %if.expr.else.loc14_10, !dbg !50
// CHECK:STDOUT:
// CHECK:STDOUT: if.expr.then.loc14_10:                            ; preds = %entry
// CHECK:STDOUT:   br i1 %c, label %if.expr.then.loc14_20, label %if.expr.else.loc14_20, !dbg !52
// CHECK:STDOUT:
// CHECK:STDOUT: if.expr.then.loc14_20:                            ; preds = %if.expr.then.loc14_10
// CHECK:STDOUT:   br label %if.expr.result.loc14_20, !dbg !52
// CHECK:STDOUT:
// CHECK:STDOUT: if.expr.else.loc14_20:                            ; preds = %if.expr.then.loc14_10
// CHECK:STDOUT:   br label %if.expr.result.loc14_20, !dbg !52
// CHECK:STDOUT:
// CHECK:STDOUT: if.expr.result.loc14_20:                          ; preds = %if.expr.else.loc14_20, %if.expr.then.loc14_20
// CHECK:STDOUT:   %0 = phi i32 [ 1, %if.expr.then.loc14_20 ], [ 2, %if.expr.else.loc14_20 ]
// CHECK:STDOUT:   br label %if.expr.result.loc14_10, !dbg !50
// CHECK:STDOUT:
// CHECK:STDOUT: if.expr.else.loc14_10:                            ; preds = %entry
// CHECK:STDOUT:   br i1 %d, label %if.expr.then.loc14_51, label %if.expr.else.loc14_51, !dbg !54
// CHECK:STDOUT:
// CHECK:STDOUT: if.expr.then.loc14_51:                            ; preds = %if.expr.else.loc14_10
// CHECK:STDOUT:   br label %if.expr.result.loc14_51, !dbg !54
// CHECK:STDOUT:
// CHECK:STDOUT: if.expr.else.loc14_51:                            ; preds = %if.expr.else.loc14_10
// CHECK:STDOUT:   br label %if.expr.result.loc14_51, !dbg !54
// CHECK:STDOUT:
// CHECK:STDOUT: if.expr.result.loc14_51:                          ; preds = %if.expr.else.loc14_51, %if.expr.then.loc14_51
// CHECK:STDOUT:   %1 = phi i32 [ 3, %if.expr.then.loc14_51 ], [ 4, %if.expr.else.loc14_51 ]
// CHECK:STDOUT:   br label %if.expr.result.loc14_10, !dbg !50
// CHECK:STDOUT:
// CHECK:STDOUT: if.expr.result.loc14_10:                          ; preds = %if.expr.result.loc14_51, %if.expr.result.loc14_20
// CHECK:STDOUT:   %2 = phi i32 [ %0, %if.expr.result.loc14_20 ], [ %1, %if.expr.result.loc14_51 ]
// CHECK:STDOUT:   ret i32 %2, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!35}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !34 = !DIFile(filename: "empty_block.carbon", directory: "")
// CHECK:STDOUT: !35 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !34, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !37 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !38 = !{!36, !37, !37, !37}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "Select", linkageName: "_CSelect.Main", scope: null, file: !34, line: 13, type: !39, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !56)
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !40, type: !37)
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 2, scope: !40, type: !37)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 3, scope: !40, type: !37)
// CHECK:STDOUT: !50 = !DILocation(line: 14, column: 10, scope: !40)
// CHECK:STDOUT: !52 = !DILocation(line: 14, column: 20, scope: !40)
// CHECK:STDOUT: !54 = !DILocation(line: 14, column: 51, scope: !40)
// CHECK:STDOUT: !55 = !DILocation(line: 14, column: 3, scope: !40)
// CHECK:STDOUT: !56 = !{!41, !42, !43}
// CHECK:STDOUT:
