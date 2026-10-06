// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/bool.carbon
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

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'break_continue.carbon'
// CHECK:STDOUT: source_filename = "break_continue.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare i1 @_CA.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: declare i1 @_CB.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: declare i1 @_CC.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CWhile.Main() #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   br label %while.cond
// CHECK:STDOUT:
// CHECK:STDOUT: while.cond:                                       ; preds = %entry, %if.else.loc20, %if.then.loc19
// CHECK:STDOUT:   %A.call = call i1 @_CA.Main(), !dbg !25
// CHECK:STDOUT:   br i1 %A.call, label %while.body, label %while.done, !dbg !24
// CHECK:STDOUT:
// CHECK:STDOUT: while.body:                                       ; preds = %while.cond
// CHECK:STDOUT:   %B.call = call i1 @_CB.Main(), !dbg !26
// CHECK:STDOUT:   br i1 %B.call, label %if.then.loc19, label %if.else.loc19, !dbg !27
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc19:                                    ; preds = %while.body
// CHECK:STDOUT:   br label %while.cond, !dbg !28
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc19:                                    ; preds = %while.body
// CHECK:STDOUT:   %C.call = call i1 @_CC.Main(), !dbg !29
// CHECK:STDOUT:   br i1 %C.call, label %if.then.loc20, label %if.else.loc20, !dbg !30
// CHECK:STDOUT:
// CHECK:STDOUT: if.then.loc20:                                    ; preds = %if.else.loc19
// CHECK:STDOUT:   br label %while.done, !dbg !31
// CHECK:STDOUT:
// CHECK:STDOUT: if.else.loc20:                                    ; preds = %if.else.loc19
// CHECK:STDOUT:   br label %while.cond, !dbg !32
// CHECK:STDOUT:
// CHECK:STDOUT: while.done:                                       ; preds = %if.then.loc20, %while.cond
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT:   uselistorder label %while.cond, { 1, 2, 0 }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!17}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !16 = !DIFile(filename: "break_continue.carbon", directory: "")
// CHECK:STDOUT: !17 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !16, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !21 = !{null}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "While", linkageName: "_CWhile.Main", scope: null, file: !16, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !17)
// CHECK:STDOUT: !24 = !DILocation(line: 18, column: 9, scope: !23)
// CHECK:STDOUT: !25 = !DILocation(line: 18, column: 10, scope: !23)
// CHECK:STDOUT: !26 = !DILocation(line: 19, column: 9, scope: !23)
// CHECK:STDOUT: !27 = !DILocation(line: 19, column: 8, scope: !23)
// CHECK:STDOUT: !28 = !DILocation(line: 19, column: 16, scope: !23)
// CHECK:STDOUT: !29 = !DILocation(line: 20, column: 9, scope: !23)
// CHECK:STDOUT: !30 = !DILocation(line: 20, column: 8, scope: !23)
// CHECK:STDOUT: !31 = !DILocation(line: 20, column: 16, scope: !23)
// CHECK:STDOUT: !32 = !DILocation(line: 18, column: 3, scope: !23)
// CHECK:STDOUT: !33 = !DILocation(line: 17, column: 1, scope: !23)
// CHECK:STDOUT:
