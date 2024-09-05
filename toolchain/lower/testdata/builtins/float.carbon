// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/builtins/float.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/builtins/float.carbon

fn Negate(a: f64) -> f64 = "float.negate";
fn TestNegate(a: f64) -> f64 { return Negate(a); }

fn Add(a: f64, b: f64) -> f64 = "float.add";
fn TestAdd(a: f64, b: f64) -> f64 { return Add(a, b); }

fn Sub(a: f64, b: f64) -> f64 = "float.sub";
fn TestSub(a: f64, b: f64) -> f64 { return Sub(a, b); }

fn Mul(a: f64, b: f64) -> f64 = "float.mul";
fn TestMul(a: f64, b: f64) -> f64 { return Mul(a, b); }

fn Div(a: f64, b: f64) -> f64 = "float.div";
fn TestDiv(a: f64, b: f64) -> f64 { return Div(a, b); }

fn Eq(a: f64, b: f64) -> bool = "float.eq";
fn TestEq(a: f64, b: f64) -> bool { return Eq(a, b); }

fn Neq(a: f64, b: f64) -> bool = "float.neq";
fn TestNeq(a: f64, b: f64) -> bool { return Neq(a, b); }

fn Less(a: f64, b: f64) -> bool = "float.less";
fn TestLess(a: f64, b: f64) -> bool { return Less(a, b); }

fn LessEq(a: f64, b: f64) -> bool = "float.less_eq";
fn TestLessEq(a: f64, b: f64) -> bool { return LessEq(a, b); }

fn Greater(a: f64, b: f64) -> bool = "float.greater";
fn TestGreater(a: f64, b: f64) -> bool { return Greater(a, b); }

fn GreaterEq(a: f64, b: f64) -> bool = "float.greater_eq";
fn TestGreaterEq(a: f64, b: f64) -> bool { return GreaterEq(a, b); }

// CHECK:STDOUT: ; ModuleID = 'float.carbon'
// CHECK:STDOUT: source_filename = "float.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define double @_CTestNegate.Main(double %a) !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %float.negate = fneg double %a, !dbg !7
// CHECK:STDOUT:   ret double %float.negate, !dbg !8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define double @_CTestAdd.Main(double %a, double %b) !dbg !9 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %float.add = fadd double %a, %b, !dbg !10
// CHECK:STDOUT:   ret double %float.add, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define double @_CTestSub.Main(double %a, double %b) !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %float.sub = fsub double %a, %b, !dbg !13
// CHECK:STDOUT:   ret double %float.sub, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define double @_CTestMul.Main(double %a, double %b) !dbg !15 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %float.mul = fmul double %a, %b, !dbg !16
// CHECK:STDOUT:   ret double %float.mul, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define double @_CTestDiv.Main(double %a, double %b) !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %float.div = fdiv double %a, %b, !dbg !19
// CHECK:STDOUT:   ret double %float.div, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @_CTestEq.Main(double %a, double %b) !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %float.eq = fcmp oeq double %a, %b, !dbg !22
// CHECK:STDOUT:   ret i1 %float.eq, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @_CTestNeq.Main(double %a, double %b) !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %float.neq = fcmp one double %a, %b, !dbg !25
// CHECK:STDOUT:   ret i1 %float.neq, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @_CTestLess.Main(double %a, double %b) !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %float.less = fcmp olt double %a, %b, !dbg !28
// CHECK:STDOUT:   ret i1 %float.less, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @_CTestLessEq.Main(double %a, double %b) !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %float.less_eq = fcmp ole double %a, %b, !dbg !31
// CHECK:STDOUT:   ret i1 %float.less_eq, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @_CTestGreater.Main(double %a, double %b) !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %float.greater = fcmp ogt double %a, %b, !dbg !34
// CHECK:STDOUT:   ret i1 %float.greater, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @_CTestGreaterEq.Main(double %a, double %b) !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %float.greater_eq = fcmp oge double %a, %b, !dbg !37
// CHECK:STDOUT:   ret i1 %float.greater_eq, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "float.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "TestNegate", linkageName: "_CTestNegate.Main", scope: null, file: !3, line: 12, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 12, column: 39, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 12, column: 32, scope: !4)
// CHECK:STDOUT: !9 = distinct !DISubprogram(name: "TestAdd", linkageName: "_CTestAdd.Main", scope: null, file: !3, line: 15, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !10 = !DILocation(line: 15, column: 44, scope: !9)
// CHECK:STDOUT: !11 = !DILocation(line: 15, column: 37, scope: !9)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "TestSub", linkageName: "_CTestSub.Main", scope: null, file: !3, line: 18, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !13 = !DILocation(line: 18, column: 44, scope: !12)
// CHECK:STDOUT: !14 = !DILocation(line: 18, column: 37, scope: !12)
// CHECK:STDOUT: !15 = distinct !DISubprogram(name: "TestMul", linkageName: "_CTestMul.Main", scope: null, file: !3, line: 21, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !16 = !DILocation(line: 21, column: 44, scope: !15)
// CHECK:STDOUT: !17 = !DILocation(line: 21, column: 37, scope: !15)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "TestDiv", linkageName: "_CTestDiv.Main", scope: null, file: !3, line: 24, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !19 = !DILocation(line: 24, column: 44, scope: !18)
// CHECK:STDOUT: !20 = !DILocation(line: 24, column: 37, scope: !18)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "TestEq", linkageName: "_CTestEq.Main", scope: null, file: !3, line: 27, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !22 = !DILocation(line: 27, column: 44, scope: !21)
// CHECK:STDOUT: !23 = !DILocation(line: 27, column: 37, scope: !21)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "TestNeq", linkageName: "_CTestNeq.Main", scope: null, file: !3, line: 30, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !25 = !DILocation(line: 30, column: 45, scope: !24)
// CHECK:STDOUT: !26 = !DILocation(line: 30, column: 38, scope: !24)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "TestLess", linkageName: "_CTestLess.Main", scope: null, file: !3, line: 33, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !28 = !DILocation(line: 33, column: 46, scope: !27)
// CHECK:STDOUT: !29 = !DILocation(line: 33, column: 39, scope: !27)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "TestLessEq", linkageName: "_CTestLessEq.Main", scope: null, file: !3, line: 36, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !31 = !DILocation(line: 36, column: 48, scope: !30)
// CHECK:STDOUT: !32 = !DILocation(line: 36, column: 41, scope: !30)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "TestGreater", linkageName: "_CTestGreater.Main", scope: null, file: !3, line: 39, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !34 = !DILocation(line: 39, column: 49, scope: !33)
// CHECK:STDOUT: !35 = !DILocation(line: 39, column: 42, scope: !33)
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "TestGreaterEq", linkageName: "_CTestGreaterEq.Main", scope: null, file: !3, line: 42, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !37 = !DILocation(line: 42, column: 51, scope: !36)
// CHECK:STDOUT: !38 = !DILocation(line: 42, column: 44, scope: !36)
