// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/while/while.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/while/while.carbon

fn Cond() -> bool;

fn F();
fn G();
fn H();

fn While() {
  F();
  while (Cond()) {
    G();
  }
  H();
}

// CHECK:STDOUT: ; ModuleID = 'while.carbon'
// CHECK:STDOUT: source_filename = "while.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare i1 @_CCond.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CH.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CWhile.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main(), !dbg !7
// CHECK:STDOUT:   br label %while.cond, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: while.cond:                                       ; preds = %while.body, %entry
// CHECK:STDOUT:   %Cond.call = call i1 @_CCond.Main(), !dbg !9
// CHECK:STDOUT:   br i1 %Cond.call, label %while.body, label %while.done, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: while.body:                                       ; preds = %while.cond
// CHECK:STDOUT:   call void @_CG.Main(), !dbg !10
// CHECK:STDOUT:   br label %while.cond, !dbg !11
// CHECK:STDOUT:
// CHECK:STDOUT: while.done:                                       ; preds = %while.cond
// CHECK:STDOUT:   call void @_CH.Main(), !dbg !12
// CHECK:STDOUT:   ret void, !dbg !13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "while.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "While", linkageName: "_CWhile.Main", scope: null, file: !3, line: 17, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 18, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 19, column: 9, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 19, column: 10, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 20, column: 5, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 19, column: 3, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 22, column: 3, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 17, column: 1, scope: !4)
