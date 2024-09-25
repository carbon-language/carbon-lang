// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/basics/type_values.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/basics/type_values.carbon

// TODO: It would be nice to treat these functions as returning `void` instead
// of `{}`.

fn I32() -> type {
  return i32;
}

fn I48() -> type {
  return i48;
}

fn F64() -> type {
  return f64;
}

// CHECK:STDOUT: ; ModuleID = 'type_values.carbon'
// CHECK:STDOUT: source_filename = "type_values.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: %type = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: define %type @_CI32.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret %type zeroinitializer, !dbg !7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define %type @_CI48.Main() !dbg !8 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret %type zeroinitializer, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define %type @_CF64.Main() !dbg !10 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret %type zeroinitializer, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "type_values.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "I32", linkageName: "_CI32.Main", scope: null, file: !3, line: 14, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 15, column: 3, scope: !4)
// CHECK:STDOUT: !8 = distinct !DISubprogram(name: "I48", linkageName: "_CI48.Main", scope: null, file: !3, line: 18, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !9 = !DILocation(line: 19, column: 3, scope: !8)
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "F64", linkageName: "_CF64.Main", scope: null, file: !3, line: 22, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !11 = !DILocation(line: 23, column: 3, scope: !10)
