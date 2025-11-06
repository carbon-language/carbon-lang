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
// CHECK:STDOUT: define void @_CWhile.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   br label %while.cond.loc22
// CHECK:STDOUT:
// CHECK:STDOUT: while.cond.loc22:                                 ; preds = %entry, %while.body.loc22
// CHECK:STDOUT:   %Cond.call.loc22 = call i1 @_CCond.Main(), !dbg !7
// CHECK:STDOUT:   br i1 %Cond.call.loc22, label %while.body.loc22, label %while.done.loc22, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: while.body.loc22:                                 ; preds = %while.cond.loc22
// CHECK:STDOUT:   call void @_CF.Main(), !dbg !9
// CHECK:STDOUT:   br label %while.cond.loc22, !dbg !10
// CHECK:STDOUT:
// CHECK:STDOUT: while.done.loc22:                                 ; preds = %while.cond.loc22
// CHECK:STDOUT:   %Cond.call.loc26 = call i1 @_CCond.Main(), !dbg !11
// CHECK:STDOUT:   br i1 %Cond.call.loc26, label %while.cond.loc27, label %if.else, !dbg !12
// CHECK:STDOUT:
// CHECK:STDOUT: while.cond.loc27:                                 ; preds = %while.body.loc27, %while.done.loc22
// CHECK:STDOUT:   %Cond.call.loc27 = call i1 @_CCond.Main(), !dbg !13
// CHECK:STDOUT:   br i1 %Cond.call.loc27, label %while.body.loc27, label %while.done.loc27, !dbg !14
// CHECK:STDOUT:
// CHECK:STDOUT: while.body.loc27:                                 ; preds = %while.cond.loc27
// CHECK:STDOUT:   call void @_CG.Main(), !dbg !15
// CHECK:STDOUT:   br label %while.cond.loc27, !dbg !16
// CHECK:STDOUT:
// CHECK:STDOUT: while.done.loc27:                                 ; preds = %while.cond.loc27
// CHECK:STDOUT:   br label %if.else, !dbg !17
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %while.done.loc27, %while.done.loc22
// CHECK:STDOUT:   ret void, !dbg !18
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT:   uselistorder label %while.cond.loc22, { 1, 0 }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "preheader.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "While", linkageName: "_CWhile.Main", scope: null, file: !3, line: 21, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 22, column: 10, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 22, column: 9, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 23, column: 5, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 22, column: 3, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 26, column: 7, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 26, column: 6, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 27, column: 12, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 27, column: 11, scope: !4)
// CHECK:STDOUT: !15 = !DILocation(line: 28, column: 7, scope: !4)
// CHECK:STDOUT: !16 = !DILocation(line: 27, column: 5, scope: !4)
// CHECK:STDOUT: !17 = !DILocation(line: 26, column: 3, scope: !4)
// CHECK:STDOUT: !18 = !DILocation(line: 21, column: 1, scope: !4)
