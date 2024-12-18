// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/builtins/no_prelude/bool.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/builtins/no_prelude/bool.carbon

fn Bool() -> type = "bool.make_type";

fn Eq(a: Bool(), b: Bool()) -> Bool() = "bool.eq";
fn TestEq(a: Bool(), b: Bool()) -> Bool() { return Eq(a, b); }

fn Neq(a: Bool(), b: Bool()) -> Bool() = "bool.neq";
fn TestNeq(a: Bool(), b: Bool()) -> Bool() { return Neq(a, b); }

fn IfEq(a: Bool(), b: Bool()) -> Bool() {
  if (Eq(a, b)) { return true; }
  return false;
}

// CHECK:STDOUT: ; ModuleID = 'bool.carbon'
// CHECK:STDOUT: source_filename = "bool.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @_CTestEq.Main(i1 %a, i1 %b) !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %bool.eq = icmp eq i1 %a, %b, !dbg !7
// CHECK:STDOUT:   ret i1 %bool.eq, !dbg !8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @_CTestNeq.Main(i1 %a, i1 %b) !dbg !9 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %bool.neq = icmp ne i1 %a, %b, !dbg !10
// CHECK:STDOUT:   ret i1 %bool.neq, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @_CIfEq.Main(i1 %a, i1 %b) !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %bool.eq = icmp eq i1 %a, %b, !dbg !13
// CHECK:STDOUT:   br i1 %bool.eq, label %if.then, label %if.else, !dbg !14
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i1 true, !dbg !15
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   ret i1 false, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "bool.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "TestEq", linkageName: "_CTestEq.Main", scope: null, file: !3, line: 14, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 14, column: 52, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 14, column: 45, scope: !4)
// CHECK:STDOUT: !9 = distinct !DISubprogram(name: "TestNeq", linkageName: "_CTestNeq.Main", scope: null, file: !3, line: 17, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !10 = !DILocation(line: 17, column: 53, scope: !9)
// CHECK:STDOUT: !11 = !DILocation(line: 17, column: 46, scope: !9)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "IfEq", linkageName: "_CIfEq.Main", scope: null, file: !3, line: 19, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !13 = !DILocation(line: 20, column: 7, scope: !12)
// CHECK:STDOUT: !14 = !DILocation(line: 20, column: 6, scope: !12)
// CHECK:STDOUT: !15 = !DILocation(line: 20, column: 19, scope: !12)
// CHECK:STDOUT: !16 = !DILocation(line: 21, column: 3, scope: !12)
