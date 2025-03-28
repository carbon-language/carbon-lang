// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/if_expr/basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/if_expr/basic.carbon

fn F() -> i32 { return 1; }
fn G() -> i32 { return 2; }

fn Select(b: bool) -> i32 {
  return if b then F() else G();
}

// CHECK:STDOUT: ; ModuleID = 'basic.carbon'
// CHECK:STDOUT: source_filename = "basic.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CF.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 1, !dbg !7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CG.Main() !dbg !8 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 2, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CSelect.Main(i1 %b) !dbg !10 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   br i1 %b, label %if.expr.then, label %if.expr.else, !dbg !11
// CHECK:STDOUT:
// CHECK:STDOUT: if.expr.then:                                     ; preds = %entry
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main(), !dbg !12
// CHECK:STDOUT:   br label %if.expr.result, !dbg !11
// CHECK:STDOUT:
// CHECK:STDOUT: if.expr.else:                                     ; preds = %entry
// CHECK:STDOUT:   %G.call = call i32 @_CG.Main(), !dbg !13
// CHECK:STDOUT:   br label %if.expr.result, !dbg !11
// CHECK:STDOUT:
// CHECK:STDOUT: if.expr.result:                                   ; preds = %if.expr.else, %if.expr.then
// CHECK:STDOUT:   %0 = phi i32 [ %F.call, %if.expr.then ], [ %G.call, %if.expr.else ]
// CHECK:STDOUT:   ret i32 %0, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "basic.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !3, line: 11, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 11, column: 17, scope: !4)
// CHECK:STDOUT: !8 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !3, line: 12, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !9 = !DILocation(line: 12, column: 17, scope: !8)
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "Select", linkageName: "_CSelect.Main", scope: null, file: !3, line: 14, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !11 = !DILocation(line: 15, column: 10, scope: !10)
// CHECK:STDOUT: !12 = !DILocation(line: 15, column: 20, scope: !10)
// CHECK:STDOUT: !13 = !DILocation(line: 15, column: 29, scope: !10)
// CHECK:STDOUT: !14 = !DILocation(line: 15, column: 3, scope: !10)
