// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/basics/int_types.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/basics/int_types.carbon

fn F_i8(a: i8) -> i8 { return a; }
fn F_u16(a: u16) -> u16 { return a; }
fn F_i64(a: i64) -> i64 { return a; }
fn F_u65536(a: u65536) -> u65536 { return a; }

// CHECK:STDOUT: ; ModuleID = 'int_types.carbon'
// CHECK:STDOUT: source_filename = "int_types.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i8 @_CF_i8.Main(i8 %a) !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i8 %a, !dbg !7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i16 @_CF_u16.Main(i16 %a) !dbg !8 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i16 %a, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i64 @_CF_i64.Main(i64 %a) !dbg !10 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i64 %a, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i65536 @_CF_u65536.Main(i65536 %a) !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i65536 %a, !dbg !13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "int_types.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F_i8", linkageName: "_CF_i8.Main", scope: null, file: !3, line: 11, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 11, column: 24, scope: !4)
// CHECK:STDOUT: !8 = distinct !DISubprogram(name: "F_u16", linkageName: "_CF_u16.Main", scope: null, file: !3, line: 12, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !9 = !DILocation(line: 12, column: 27, scope: !8)
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "F_i64", linkageName: "_CF_i64.Main", scope: null, file: !3, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !11 = !DILocation(line: 13, column: 27, scope: !10)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "F_u65536", linkageName: "_CF_u65536.Main", scope: null, file: !3, line: 14, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !13 = !DILocation(line: 14, column: 36, scope: !12)
