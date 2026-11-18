// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
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

interface I {
  fn F();
}

class C {}

impl C as I {
  fn F() {}
}

// --- b.carbon

import A;

fn F() { A.NS.F(); }

fn G() -> i32 { return A.NS.G(1); }

fn CallCAsI() {
  (A.C as A.I).F();
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'a.carbon'
// CHECK:STDOUT: source_filename = "a.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.NS.A() #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CG.NS.A(i32 %x) #0 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %x, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.C.A:I.A"() #0 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "a.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{null}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !{!20, !20}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "F", linkageName: "_CF.NS.A", scope: null, file: !18, line: 5, type: !22, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !26 = !DILocation(line: 5, column: 1, scope: !25)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "G", linkageName: "_CG.NS.A", scope: null, file: !18, line: 7, type: !24, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !34)
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 1, scope: !27, type: !20)
// CHECK:STDOUT: !33 = !DILocation(line: 7, column: 26, scope: !27)
// CHECK:STDOUT: !34 = !{!28}
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "F", linkageName: "_CF.C.A:I.A", scope: null, file: !18, line: 16, type: !22, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !36 = !DILocation(line: 16, column: 3, scope: !35)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'b.carbon'
// CHECK:STDOUT: source_filename = "b.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.NS.A(), !dbg !42
// CHECK:STDOUT:   ret void, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.NS.A()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CG.Main() #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %G.call = call i32 @_CG.NS.A(i32 1), !dbg !47
// CHECK:STDOUT:   ret i32 %G.call, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @_CG.NS.A(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCallCAsI.Main() #0 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CF.C.A:I.A"(), !dbg !50
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_CF.C.A:I.A"()
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!38}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{null}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !37 = !DIFile(filename: "b.carbon", directory: "")
// CHECK:STDOUT: !38 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !37, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !39 = !{!20}
// CHECK:STDOUT: !40 = !DISubroutineType(types: !39)
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !37, line: 4, type: !22, spFlags: DISPFlagDefinition, unit: !38)
// CHECK:STDOUT: !42 = !DILocation(line: 4, column: 10, scope: !41)
// CHECK:STDOUT: !43 = !DILocation(line: 4, column: 1, scope: !41)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !37, line: 6, type: !40, spFlags: DISPFlagDefinition, unit: !38)
// CHECK:STDOUT: !47 = !DILocation(line: 6, column: 24, scope: !44)
// CHECK:STDOUT: !48 = !DILocation(line: 6, column: 17, scope: !44)
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "CallCAsI", linkageName: "_CCallCAsI.Main", scope: null, file: !37, line: 8, type: !22, spFlags: DISPFlagDefinition, unit: !38)
// CHECK:STDOUT: !50 = !DILocation(line: 9, column: 3, scope: !49)
// CHECK:STDOUT: !51 = !DILocation(line: 8, column: 1, scope: !49)
// CHECK:STDOUT:
