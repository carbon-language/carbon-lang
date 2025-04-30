// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

// CHECK:STDOUT: ; ModuleID = 'unreachable_end.carbon'
// CHECK:STDOUT: source_filename = "unreachable_end.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare !dbg !4 i1 @_CCond.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: declare !dbg !7 void @_CF.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: declare !dbg !8 void @_CG.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: declare !dbg !9 void @_CH.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CWhile.Main() !dbg !10 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main(), !dbg !11
// CHECK:STDOUT:   br label %while.cond, !dbg !12
// CHECK:STDOUT:
// CHECK:STDOUT: while.cond:                                       ; preds = %entry
// CHECK:STDOUT:   %Cond.call = call i1 @_CCond.Main(), !dbg !13
// CHECK:STDOUT:   br i1 %Cond.call, label %while.body, label %while.done, !dbg !12
// CHECK:STDOUT:
// CHECK:STDOUT: while.body:                                       ; preds = %while.cond
// CHECK:STDOUT:   call void @_CG.Main(), !dbg !14
// CHECK:STDOUT:   ret void, !dbg !15
// CHECK:STDOUT:
// CHECK:STDOUT: while.done:                                       ; preds = %while.cond
// CHECK:STDOUT:   call void @_CH.Main(), !dbg !16
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "unreachable_end.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Cond", linkageName: "_CCond.Main", scope: null, file: !3, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !3, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !8 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !3, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !9 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main", scope: null, file: !3, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "While", linkageName: "_CWhile.Main", scope: null, file: !3, line: 17, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !11 = !DILocation(line: 18, column: 3, scope: !10)
// CHECK:STDOUT: !12 = !DILocation(line: 19, column: 9, scope: !10)
// CHECK:STDOUT: !13 = !DILocation(line: 19, column: 10, scope: !10)
// CHECK:STDOUT: !14 = !DILocation(line: 20, column: 5, scope: !10)
// CHECK:STDOUT: !15 = !DILocation(line: 21, column: 5, scope: !10)
// CHECK:STDOUT: !16 = !DILocation(line: 23, column: 3, scope: !10)
// CHECK:STDOUT: !17 = !DILocation(line: 17, column: 1, scope: !10)
