// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/packages/cross_package_call.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/packages/cross_package_call.carbon

// --- a.carbon

package A;
namespace NS;

fn NS.F() {}

fn NS.G(x: i32) -> i32 { return x; }

// --- b.carbon

import A;

fn F() { A.NS.F(); }

fn G() -> i32 { return A.NS.G(1); }

// CHECK:STDOUT: ; ModuleID = 'a.carbon'
// CHECK:STDOUT: source_filename = "a.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CF.NS.A() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CG.NS.A(i32 %x) !dbg !8 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "a.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.NS.A", scope: null, file: !3, line: 5, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 5, column: 1, scope: !4)
// CHECK:STDOUT: !8 = distinct !DISubprogram(name: "G", linkageName: "_CG.NS.A", scope: null, file: !3, line: 7, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !9 = !DILocation(line: 7, column: 26, scope: !8)
// CHECK:STDOUT: ; ModuleID = 'b.carbon'
// CHECK:STDOUT: source_filename = "b.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CF.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.NS.A(), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.NS.A()
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CG.Main() !dbg !9 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call i32 @_CG.NS.A(i32 1), !dbg !10
// CHECK:STDOUT:   ret i32 %G.call, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @_CG.NS.A(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "b.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !3, line: 4, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 4, column: 10, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 4, column: 1, scope: !4)
// CHECK:STDOUT: !9 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !3, line: 6, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !10 = !DILocation(line: 6, column: 24, scope: !9)
// CHECK:STDOUT: !11 = !DILocation(line: 6, column: 17, scope: !9)
