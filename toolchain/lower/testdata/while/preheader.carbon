// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// CHECK:STDOUT: define void @_CWhile.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   br label %while.cond.loc20
// CHECK:STDOUT:
// CHECK:STDOUT: while.cond.loc20:                                 ; preds = %entry, %while.body.loc20
// CHECK:STDOUT:   %Cond.call.loc20 = call i1 @_CCond.Main(), !dbg !7
// CHECK:STDOUT:   br i1 %Cond.call.loc20, label %while.body.loc20, label %while.done.loc20, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: while.body.loc20:                                 ; preds = %while.cond.loc20
// CHECK:STDOUT:   call void @_CF.Main(), !dbg !9
// CHECK:STDOUT:   br label %while.cond.loc20, !dbg !10
// CHECK:STDOUT:
// CHECK:STDOUT: while.done.loc20:                                 ; preds = %while.cond.loc20
// CHECK:STDOUT:   %Cond.call.loc24 = call i1 @_CCond.Main(), !dbg !11
// CHECK:STDOUT:   br i1 %Cond.call.loc24, label %while.cond.loc25, label %if.else, !dbg !12
// CHECK:STDOUT:
// CHECK:STDOUT: while.cond.loc25:                                 ; preds = %while.body.loc25, %while.done.loc20
// CHECK:STDOUT:   %Cond.call.loc25 = call i1 @_CCond.Main(), !dbg !13
// CHECK:STDOUT:   br i1 %Cond.call.loc25, label %while.body.loc25, label %while.done.loc25, !dbg !14
// CHECK:STDOUT:
// CHECK:STDOUT: while.body.loc25:                                 ; preds = %while.cond.loc25
// CHECK:STDOUT:   call void @_CG.Main(), !dbg !15
// CHECK:STDOUT:   br label %while.cond.loc25, !dbg !16
// CHECK:STDOUT:
// CHECK:STDOUT: while.done.loc25:                                 ; preds = %while.cond.loc25
// CHECK:STDOUT:   br label %if.else, !dbg !17
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %while.done.loc25, %while.done.loc20
// CHECK:STDOUT:   ret void, !dbg !18
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT:   uselistorder label %while.cond.loc20, { 1, 0 }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "preheader.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "While", linkageName: "_CWhile.Main", scope: null, file: !3, line: 19, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 20, column: 10, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 20, column: 9, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 21, column: 5, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 20, column: 3, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 24, column: 7, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 24, column: 6, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 25, column: 12, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 25, column: 11, scope: !4)
// CHECK:STDOUT: !15 = !DILocation(line: 26, column: 7, scope: !4)
// CHECK:STDOUT: !16 = !DILocation(line: 25, column: 5, scope: !4)
// CHECK:STDOUT: !17 = !DILocation(line: 24, column: 3, scope: !4)
// CHECK:STDOUT: !18 = !DILocation(line: 19, column: 1, scope: !4)
