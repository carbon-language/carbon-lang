// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

  let EOF: i32 = -(1 as i32);
  while (ReadChar() != EOF) {
    // "Hi"
    if (PrintChar(0x48) != EOF) {
      PrintChar(0x69);
    }
  }
}

// CHECK:STDOUT: ; ModuleID = 'print_read.carbon'
// CHECK:STDOUT: source_filename = "print_read.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @printf.int.format = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CMain.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %print.int = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 1), !dbg !7
// CHECK:STDOUT:   %.loc19_18.1.temp = alloca i32, align 4, !dbg !8
// CHECK:STDOUT:   store i32 -1, ptr %.loc19_18.1.temp, align 4, !dbg !8
// CHECK:STDOUT:   br label %while.cond, !dbg !9
// CHECK:STDOUT:
// CHECK:STDOUT: while.cond:                                       ; preds = %if.else, %entry
// CHECK:STDOUT:   %read.char = call i32 @getchar(), !dbg !10
// CHECK:STDOUT:   %.loc20_24 = load i32, ptr %.loc19_18.1.temp, align 4, !dbg !11
// CHECK:STDOUT:   %int.neq.loc20 = icmp ne i32 %read.char, %.loc20_24, !dbg !10
// CHECK:STDOUT:   br i1 %int.neq.loc20, label %while.body, label %while.done, !dbg !9
// CHECK:STDOUT:
// CHECK:STDOUT: while.body:                                       ; preds = %while.cond
// CHECK:STDOUT:   %print.char.loc22 = call i32 @putchar(i32 72), !dbg !12
// CHECK:STDOUT:   %.loc22_28 = load i32, ptr %.loc19_18.1.temp, align 4, !dbg !13
// CHECK:STDOUT:   %int.neq.loc22 = icmp ne i32 %print.char.loc22, %.loc22_28, !dbg !12
// CHECK:STDOUT:   br i1 %int.neq.loc22, label %if.then, label %if.else, !dbg !14
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %while.body
// CHECK:STDOUT:   %print.char.loc23 = call i32 @putchar(i32 105), !dbg !15
// CHECK:STDOUT:   br label %if.else, !dbg !16
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %while.body
// CHECK:STDOUT:   br label %while.cond, !dbg !17
// CHECK:STDOUT:
// CHECK:STDOUT: while.done:                                       ; preds = %while.cond
// CHECK:STDOUT:   ret void, !dbg !18
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
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "print_read.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Main", linkageName: "_CMain.Main", scope: null, file: !3, line: 16, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 17, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 19, column: 18, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 20, column: 9, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 20, column: 10, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 20, column: 24, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 22, column: 9, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 22, column: 28, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 22, column: 8, scope: !4)
// CHECK:STDOUT: !15 = !DILocation(line: 23, column: 7, scope: !4)
// CHECK:STDOUT: !16 = !DILocation(line: 22, column: 5, scope: !4)
// CHECK:STDOUT: !17 = !DILocation(line: 20, column: 3, scope: !4)
// CHECK:STDOUT: !18 = !DILocation(line: 16, column: 1, scope: !4)
