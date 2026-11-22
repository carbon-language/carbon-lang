// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/bool.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/while/preheader.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/while/preheader.carbon

fn Cond() -> bool;

fn F();
fn G();

// TODO: It would be better to preserve the loop preheader blocks from SemIR
// into LLVM IR. If we don't, LLVM will recreate them for us as part of loop
// canonicalization.
fn While() {
  while (Cond()) {
    F();
  }

  if (Cond()) {
    while (Cond()) {
      G();
    }
  }
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'preheader.carbon'
// CHECK:STDOUT: source_filename = "preheader.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare i1 @_CCond.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CG.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CWhile.Main() #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   br label %while.cond.loc22
// CHECK:STDOUT:
// CHECK:STDOUT: while.cond.loc22:                                 ; preds = %entry, %while.body.loc22
// CHECK:STDOUT:   %Cond.call.loc22 = call i1 @_CCond.Main(), !dbg !25
// CHECK:STDOUT:   br i1 %Cond.call.loc22, label %while.body.loc22, label %while.done.loc22, !dbg !24
// CHECK:STDOUT:
// CHECK:STDOUT: while.body.loc22:                                 ; preds = %while.cond.loc22
// CHECK:STDOUT:   call void @_CF.Main(), !dbg !26
// CHECK:STDOUT:   br label %while.cond.loc22, !dbg !27
// CHECK:STDOUT:
// CHECK:STDOUT: while.done.loc22:                                 ; preds = %while.cond.loc22
// CHECK:STDOUT:   %Cond.call.loc26 = call i1 @_CCond.Main(), !dbg !28
// CHECK:STDOUT:   br i1 %Cond.call.loc26, label %while.cond.loc27, label %if.else, !dbg !29
// CHECK:STDOUT:
// CHECK:STDOUT: while.cond.loc27:                                 ; preds = %while.body.loc27, %while.done.loc22
// CHECK:STDOUT:   %Cond.call.loc27 = call i1 @_CCond.Main(), !dbg !31
// CHECK:STDOUT:   br i1 %Cond.call.loc27, label %while.body.loc27, label %while.done.loc27, !dbg !30
// CHECK:STDOUT:
// CHECK:STDOUT: while.body.loc27:                                 ; preds = %while.cond.loc27
// CHECK:STDOUT:   call void @_CG.Main(), !dbg !32
// CHECK:STDOUT:   br label %while.cond.loc27, !dbg !33
// CHECK:STDOUT:
// CHECK:STDOUT: while.done.loc27:                                 ; preds = %while.cond.loc27
// CHECK:STDOUT:   br label %if.else, !dbg !34
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %while.done.loc27, %while.done.loc22
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT:   uselistorder label %while.cond.loc22, { 1, 0 }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!17}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !16 = !DIFile(filename: "preheader.carbon", directory: "")
// CHECK:STDOUT: !17 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !16, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !21 = !{null}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "While", linkageName: "_CWhile.Main", scope: null, file: !16, line: 21, type: !22, spFlags: DISPFlagDefinition, unit: !17)
// CHECK:STDOUT: !24 = !DILocation(line: 22, column: 9, scope: !23)
// CHECK:STDOUT: !25 = !DILocation(line: 22, column: 10, scope: !23)
// CHECK:STDOUT: !26 = !DILocation(line: 23, column: 5, scope: !23)
// CHECK:STDOUT: !27 = !DILocation(line: 22, column: 3, scope: !23)
// CHECK:STDOUT: !28 = !DILocation(line: 26, column: 7, scope: !23)
// CHECK:STDOUT: !29 = !DILocation(line: 26, column: 6, scope: !23)
// CHECK:STDOUT: !30 = !DILocation(line: 27, column: 11, scope: !23)
// CHECK:STDOUT: !31 = !DILocation(line: 27, column: 12, scope: !23)
// CHECK:STDOUT: !32 = !DILocation(line: 28, column: 7, scope: !23)
// CHECK:STDOUT: !33 = !DILocation(line: 27, column: 5, scope: !23)
// CHECK:STDOUT: !34 = !DILocation(line: 26, column: 3, scope: !23)
// CHECK:STDOUT: !35 = !DILocation(line: 21, column: 1, scope: !23)
// CHECK:STDOUT:
