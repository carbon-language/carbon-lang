// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --include-carbon-core
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/builtins/print_read.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/builtins/print_read.carbon

import Core library "io";

fn PrintChar(c: i8) -> i32 = "print.char";
fn ReadChar() -> i32 = "read.char";

fn Main() {
  Core.Print(1);

  let EOF: i32 = -1;
  while (ReadChar() != EOF) {
    // "Hi"
    if (PrintChar(0x48) != EOF) {
      PrintChar(0x69);
    }
  }
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'print_read.carbon'
// CHECK:STDOUT: source_filename = "print_read.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @printf.int.format = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMain.Main() #0 !dbg !10 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 1), !dbg !11
// CHECK:STDOUT:   br label %while.cond, !dbg !14
// CHECK:STDOUT:
// CHECK:STDOUT: while.cond:                                       ; preds = %if.else, %entry
// CHECK:STDOUT:   %ReadChar.call = call i32 @getchar(), !dbg !15
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.call.loc23 = icmp ne i32 %ReadChar.call, -1, !dbg !15
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.NotEqual.call.loc23, label %while.body, label %while.done, !dbg !14
// CHECK:STDOUT:
// CHECK:STDOUT: while.body:                                       ; preds = %while.cond
// CHECK:STDOUT:   %PrintChar.call.loc25 = call i32 @putchar(i32 72), !dbg !17
// CHECK:STDOUT:   %Int.as.EqWith.impl.NotEqual.call.loc25 = icmp ne i32 %PrintChar.call.loc25, -1, !dbg !17
// CHECK:STDOUT:   br i1 %Int.as.EqWith.impl.NotEqual.call.loc25, label %if.then, label %if.else, !dbg !19
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %while.body
// CHECK:STDOUT:   %PrintChar.call.loc26 = call i32 @putchar(i32 105), !dbg !20
// CHECK:STDOUT:   br label %if.else, !dbg !21
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %while.body
// CHECK:STDOUT:   br label %while.cond, !dbg !22
// CHECK:STDOUT:
// CHECK:STDOUT: while.done:                                       ; preds = %while.cond
// CHECK:STDOUT:   ret void, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @printf(ptr, ...)
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @getchar()
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @putchar(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @putchar, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "print_read.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !8 = !{null}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "Main", linkageName: "_CMain.Main", scope: null, file: !4, line: 19, type: !9, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !11 = !DILocation(line: 20, column: 3, scope: !10)
// CHECK:STDOUT: !14 = !DILocation(line: 23, column: 9, scope: !10)
// CHECK:STDOUT: !15 = !DILocation(line: 23, column: 10, scope: !10)
// CHECK:STDOUT: !17 = !DILocation(line: 25, column: 9, scope: !10)
// CHECK:STDOUT: !19 = !DILocation(line: 25, column: 8, scope: !10)
// CHECK:STDOUT: !20 = !DILocation(line: 26, column: 7, scope: !10)
// CHECK:STDOUT: !21 = !DILocation(line: 25, column: 5, scope: !10)
// CHECK:STDOUT: !22 = !DILocation(line: 23, column: 3, scope: !10)
// CHECK:STDOUT: !23 = !DILocation(line: 19, column: 1, scope: !10)
// CHECK:STDOUT:
