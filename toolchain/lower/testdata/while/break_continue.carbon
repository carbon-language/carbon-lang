// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/while/break_continue.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/while/break_continue.carbon

fn A() -> bool;
fn B() -> bool;
fn C() -> bool;

fn While() {
  while (A()) {
    if (B()) { continue; }
    if (C()) { break; }
  }
}

// CHECK:STDOUT: ; ModuleID = 'break_continue.carbon'
// CHECK:STDOUT: source_filename = "break_continue.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare i1 @_CA.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: declare i1 @_CB.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: declare i1 @_CC.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CWhile.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   br label %while.cond
// CHECK:STDOUT:
// CHECK:STDOUT: while.cond:                                       ; preds = %entry, %if.else.loc18, %if.then.loc17
// CHECK:STDOUT:   %A.call = call i1 @_CA.Main(), !dbg !7
// CHECK:STDOUT:   br i1 %A.call, label %while.body, label %while.done, !dbg !8
// CHECK:STDOUT:
// CHECK:STDOUT: while.body:                                       ; preds = %while.cond
// CHECK:STDOUT:   %B.call = call i1 @_CB.Main(), !dbg !9
// CHECK:STDOUT:   br i1 %B.call, label %if.then.loc17, label %if.else.loc17, !dbg !10
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc17:                                    ; preds = %while.body
// CHECK:STDOUT:   br label %while.cond, !dbg !11
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc17:                                    ; preds = %while.body
// CHECK:STDOUT:   %C.call = call i1 @_CC.Main(), !dbg !12
// CHECK:STDOUT:   br i1 %C.call, label %if.then.loc18, label %if.else.loc18, !dbg !13
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc18:                                    ; preds = %if.else.loc17
// CHECK:STDOUT:   br label %while.done, !dbg !14
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc18:                                    ; preds = %if.else.loc17
// CHECK:STDOUT:   br label %while.cond, !dbg !15
// CHECK:STDOUT:
// CHECK:STDOUT: while.done:                                       ; preds = %if.then.loc18, %while.cond
// CHECK:STDOUT:   ret void, !dbg !16
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT:   uselistorder label %while.cond, { 1, 2, 0 }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "break_continue.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "While", linkageName: "_CWhile.Main", scope: null, file: !3, line: 15, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 16, column: 10, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 16, column: 9, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 17, column: 9, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 17, column: 8, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 17, column: 16, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 18, column: 9, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 18, column: 8, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 18, column: 16, scope: !4)
// CHECK:STDOUT: !15 = !DILocation(line: 16, column: 3, scope: !4)
// CHECK:STDOUT: !16 = !DILocation(line: 15, column: 1, scope: !4)
