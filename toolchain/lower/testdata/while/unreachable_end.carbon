// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/bool.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/while/unreachable_end.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/while/unreachable_end.carbon

fn Cond() -> bool;

fn F();
fn G();
fn H();

fn While() {
  F();
  while (Cond()) {
    G();
    return;
  }
  H();
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'unreachable_end.carbon'
// CHECK:STDOUT: source_filename = "unreachable_end.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare i1 @_CCond.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CH.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CWhile.Main() #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main(), !dbg !26
// CHECK:STDOUT:   br label %while.cond, !dbg !27
// CHECK:STDOUT:
// CHECK:STDOUT: while.cond:                                       ; preds = %entry
// CHECK:STDOUT:   %Cond.call = call i1 @_CCond.Main(), !dbg !28
// CHECK:STDOUT:   br i1 %Cond.call, label %while.body, label %while.done, !dbg !27
// CHECK:STDOUT:
// CHECK:STDOUT: while.body:                                       ; preds = %while.cond
// CHECK:STDOUT:   call void @_CG.Main(), !dbg !29
// CHECK:STDOUT:   ret void, !dbg !30
// CHECK:STDOUT:
// CHECK:STDOUT: while.done:                                       ; preds = %while.cond
// CHECK:STDOUT:   call void @_CH.Main(), !dbg !31
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "unreachable_end.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !23 = !{null}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "While", linkageName: "_CWhile.Main", scope: null, file: !18, line: 19, type: !24, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !26 = !DILocation(line: 20, column: 3, scope: !25)
// CHECK:STDOUT: !27 = !DILocation(line: 21, column: 9, scope: !25)
// CHECK:STDOUT: !28 = !DILocation(line: 21, column: 10, scope: !25)
// CHECK:STDOUT: !29 = !DILocation(line: 22, column: 5, scope: !25)
// CHECK:STDOUT: !30 = !DILocation(line: 23, column: 5, scope: !25)
// CHECK:STDOUT: !31 = !DILocation(line: 25, column: 3, scope: !25)
// CHECK:STDOUT: !32 = !DILocation(line: 19, column: 1, scope: !25)
// CHECK:STDOUT:
