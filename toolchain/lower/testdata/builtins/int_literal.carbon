// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/builtins/int_literal.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/builtins/int_literal.carbon

fn Make() -> type = "int_literal.make_type";

fn Copy(x: Make()) -> Make() {
  return x;
}

fn MinusOne() -> i32 {
  return -1;
}

fn IntMax() -> i32 {
  return 0x1_0000_0000_0000_0000 / 0x2_0000_0000 - 1;
}

fn IntMin() -> i32 {
  return -0x8000_0000;
}

// CHECK:STDOUT: ; ModuleID = 'int_literal.carbon'
// CHECK:STDOUT: source_filename = "int_literal.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define {} @_CCopy.Main({} %x) !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret {} %x, !dbg !7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CMinusOne.Main() !dbg !8 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 -1, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CIntMax.Main() !dbg !10 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 2147483647, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CIntMin.Main() !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 -2147483648, !dbg !13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "int_literal.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Copy", linkageName: "_CCopy.Main", scope: null, file: !3, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 14, column: 3, scope: !4)
// CHECK:STDOUT: !8 = distinct !DISubprogram(name: "MinusOne", linkageName: "_CMinusOne.Main", scope: null, file: !3, line: 17, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !9 = !DILocation(line: 18, column: 3, scope: !8)
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "IntMax", linkageName: "_CIntMax.Main", scope: null, file: !3, line: 21, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !11 = !DILocation(line: 22, column: 3, scope: !10)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "IntMin", linkageName: "_CIntMin.Main", scope: null, file: !3, line: 25, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !13 = !DILocation(line: 26, column: 3, scope: !12)
