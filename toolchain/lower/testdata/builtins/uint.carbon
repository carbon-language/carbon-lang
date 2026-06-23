// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/builtins/uint.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/builtins/uint.carbon

fn Negate(a: u64) -> u64 = "int.unegate";
fn TestNegate(a: u64) -> u64 { return Negate(a); }

fn Add(a: u64, b: u64) -> u64 = "int.uadd";
fn TestAdd(a: u64, b: u64) -> u64 { return Add(a, b); }

fn Sub(a: u64, b: u64) -> u64 = "int.usub";
fn TestSub(a: u64, b: u64) -> u64 { return Sub(a, b); }

fn Mul(a: u64, b: u64) -> u64 = "int.umul";
fn TestMul(a: u64, b: u64) -> u64 { return Mul(a, b); }

fn Div(a: u64, b: u64) -> u64 = "int.udiv";
fn TestDiv(a: u64, b: u64) -> u64 { return Div(a, b); }

fn Mod(a: u64, b: u64) -> u64 = "int.umod";
fn TestMod(a: u64, b: u64) -> u64 { return Mod(a, b); }

fn Complement(a: u64) -> u64 = "int.complement";
fn TestComplement(a: u64) -> u64 { return Complement(a); }

fn And(a: u64, b: u64) -> u64 = "int.and";
fn TestAnd(a: u64, b: u64) -> u64 { return And(a, b); }

fn Or(a: u64, b: u64) -> u64 = "int.or";
fn TestOr(a: u64, b: u64) -> u64 { return Or(a, b); }

fn Xor(a: u64, b: u64) -> u64 = "int.xor";
fn TestXor(a: u64, b: u64) -> u64 { return Xor(a, b); }

fn LeftShift(a: u64, b: u64) -> u64 = "int.left_shift";
fn TestLeftShift(a: u64, b: u64) -> u64 { return LeftShift(a, b); }

fn RightShift(a: u64, b: u64) -> u64 = "int.right_shift";
fn TestRightShift(a: u64, b: u64) -> u64 { return RightShift(a, b); }

fn Eq(a: u64, b: u64) -> bool = "int.eq";
fn TestEq(a: u64, b: u64) -> bool { return Eq(a, b); }

fn Neq(a: u64, b: u64) -> bool = "int.neq";
fn TestNeq(a: u64, b: u64) -> bool { return Neq(a, b); }

fn Less(a: u64, b: u64) -> bool = "int.less";
fn TestLess(a: u64, b: u64) -> bool { return Less(a, b); }

fn LessEq(a: u64, b: u64) -> bool = "int.less_eq";
fn TestLessEq(a: u64, b: u64) -> bool { return LessEq(a, b); }

fn Greater(a: u64, b: u64) -> bool = "int.greater";
fn TestGreater(a: u64, b: u64) -> bool { return Greater(a, b); }

fn GreaterEq(a: u64, b: u64) -> bool = "int.greater_eq";
fn TestGreaterEq(a: u64, b: u64) -> bool { return GreaterEq(a, b); }

// CHECK:STDOUT: ; ModuleID = 'uint.carbon'
// CHECK:STDOUT: source_filename = "uint.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestNegate.Main(i64 %a) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Negate.call = sub i64 0, %a, !dbg !10
// CHECK:STDOUT:   ret i64 %Negate.call, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestAdd.Main(i64 %a, i64 %b) #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Add.call = add i64 %a, %b, !dbg !18
// CHECK:STDOUT:   ret i64 %Add.call, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestSub.Main(i64 %a, i64 %b) #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Sub.call = sub i64 %a, %b, !dbg !24
// CHECK:STDOUT:   ret i64 %Sub.call, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestMul.Main(i64 %a, i64 %b) #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Mul.call = mul i64 %a, %b, !dbg !30
// CHECK:STDOUT:   ret i64 %Mul.call, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestDiv.Main(i64 %a, i64 %b) #0 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Div.call = udiv i64 %a, %b, !dbg !36
// CHECK:STDOUT:   ret i64 %Div.call, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestMod.Main(i64 %a, i64 %b) #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Mod.call = urem i64 %a, %b, !dbg !42
// CHECK:STDOUT:   ret i64 %Mod.call, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestComplement.Main(i64 %a) #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Complement.call = xor i64 -1, %a, !dbg !47
// CHECK:STDOUT:   ret i64 %Complement.call, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestAnd.Main(i64 %a, i64 %b) #0 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %And.call = and i64 %a, %b, !dbg !53
// CHECK:STDOUT:   ret i64 %And.call, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestOr.Main(i64 %a, i64 %b) #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Or.call = or i64 %a, %b, !dbg !59
// CHECK:STDOUT:   ret i64 %Or.call, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestXor.Main(i64 %a, i64 %b) #0 !dbg !61 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Xor.call = xor i64 %a, %b, !dbg !65
// CHECK:STDOUT:   ret i64 %Xor.call, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestLeftShift.Main(i64 %a, i64 %b) #0 !dbg !67 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %LeftShift.call = shl i64 %a, %b, !dbg !71
// CHECK:STDOUT:   ret i64 %LeftShift.call, !dbg !72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestRightShift.Main(i64 %a, i64 %b) #0 !dbg !73 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %RightShift.call = lshr i64 %a, %b, !dbg !77
// CHECK:STDOUT:   ret i64 %RightShift.call, !dbg !78
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestEq.Main(i64 %a, i64 %b) #0 !dbg !79 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Eq.call = icmp eq i64 %a, %b, !dbg !86
// CHECK:STDOUT:   ret i1 %Eq.call, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestNeq.Main(i64 %a, i64 %b) #0 !dbg !88 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Neq.call = icmp ne i64 %a, %b, !dbg !92
// CHECK:STDOUT:   ret i1 %Neq.call, !dbg !93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestLess.Main(i64 %a, i64 %b) #0 !dbg !94 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Less.call = icmp ult i64 %a, %b, !dbg !98
// CHECK:STDOUT:   ret i1 %Less.call, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestLessEq.Main(i64 %a, i64 %b) #0 !dbg !100 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %LessEq.call = icmp ule i64 %a, %b, !dbg !104
// CHECK:STDOUT:   ret i1 %LessEq.call, !dbg !105
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestGreater.Main(i64 %a, i64 %b) #0 !dbg !106 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Greater.call = icmp ugt i64 %a, %b, !dbg !110
// CHECK:STDOUT:   ret i1 %Greater.call, !dbg !111
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestGreaterEq.Main(i64 %a, i64 %b) #0 !dbg !112 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %GreaterEq.call = icmp uge i64 %a, %b, !dbg !116
// CHECK:STDOUT:   ret i1 %GreaterEq.call, !dbg !117
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "uint.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "TestNegate", linkageName: "_CTestNegate.Main", scope: null, file: !1, line: 14, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !7}
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 64, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 14, column: 39, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 14, column: 32, scope: !4)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "TestAdd", linkageName: "_CTestAdd.Main", scope: null, file: !1, line: 17, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !15)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{!7, !7, !7}
// CHECK:STDOUT: !15 = !{!16, !17}
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 1, scope: !12, type: !7)
// CHECK:STDOUT: !17 = !DILocalVariable(arg: 2, scope: !12, type: !7)
// CHECK:STDOUT: !18 = !DILocation(line: 17, column: 44, scope: !12)
// CHECK:STDOUT: !19 = !DILocation(line: 17, column: 37, scope: !12)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "TestSub", linkageName: "_CTestSub.Main", scope: null, file: !1, line: 20, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !21)
// CHECK:STDOUT: !21 = !{!22, !23}
// CHECK:STDOUT: !22 = !DILocalVariable(arg: 1, scope: !20, type: !7)
// CHECK:STDOUT: !23 = !DILocalVariable(arg: 2, scope: !20, type: !7)
// CHECK:STDOUT: !24 = !DILocation(line: 20, column: 44, scope: !20)
// CHECK:STDOUT: !25 = !DILocation(line: 20, column: 37, scope: !20)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "TestMul", linkageName: "_CTestMul.Main", scope: null, file: !1, line: 23, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !27)
// CHECK:STDOUT: !27 = !{!28, !29}
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 1, scope: !26, type: !7)
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 2, scope: !26, type: !7)
// CHECK:STDOUT: !30 = !DILocation(line: 23, column: 44, scope: !26)
// CHECK:STDOUT: !31 = !DILocation(line: 23, column: 37, scope: !26)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "TestDiv", linkageName: "_CTestDiv.Main", scope: null, file: !1, line: 26, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !33)
// CHECK:STDOUT: !33 = !{!34, !35}
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 1, scope: !32, type: !7)
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 2, scope: !32, type: !7)
// CHECK:STDOUT: !36 = !DILocation(line: 26, column: 44, scope: !32)
// CHECK:STDOUT: !37 = !DILocation(line: 26, column: 37, scope: !32)
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "TestMod", linkageName: "_CTestMod.Main", scope: null, file: !1, line: 29, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !39)
// CHECK:STDOUT: !39 = !{!40, !41}
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !38, type: !7)
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 2, scope: !38, type: !7)
// CHECK:STDOUT: !42 = !DILocation(line: 29, column: 44, scope: !38)
// CHECK:STDOUT: !43 = !DILocation(line: 29, column: 37, scope: !38)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "TestComplement", linkageName: "_CTestComplement.Main", scope: null, file: !1, line: 32, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !45)
// CHECK:STDOUT: !45 = !{!46}
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 1, scope: !44, type: !7)
// CHECK:STDOUT: !47 = !DILocation(line: 32, column: 43, scope: !44)
// CHECK:STDOUT: !48 = !DILocation(line: 32, column: 36, scope: !44)
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "TestAnd", linkageName: "_CTestAnd.Main", scope: null, file: !1, line: 35, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !50)
// CHECK:STDOUT: !50 = !{!51, !52}
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !49, type: !7)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 2, scope: !49, type: !7)
// CHECK:STDOUT: !53 = !DILocation(line: 35, column: 44, scope: !49)
// CHECK:STDOUT: !54 = !DILocation(line: 35, column: 37, scope: !49)
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "TestOr", linkageName: "_CTestOr.Main", scope: null, file: !1, line: 38, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !56)
// CHECK:STDOUT: !56 = !{!57, !58}
// CHECK:STDOUT: !57 = !DILocalVariable(arg: 1, scope: !55, type: !7)
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 2, scope: !55, type: !7)
// CHECK:STDOUT: !59 = !DILocation(line: 38, column: 43, scope: !55)
// CHECK:STDOUT: !60 = !DILocation(line: 38, column: 36, scope: !55)
// CHECK:STDOUT: !61 = distinct !DISubprogram(name: "TestXor", linkageName: "_CTestXor.Main", scope: null, file: !1, line: 41, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !62)
// CHECK:STDOUT: !62 = !{!63, !64}
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !61, type: !7)
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 2, scope: !61, type: !7)
// CHECK:STDOUT: !65 = !DILocation(line: 41, column: 44, scope: !61)
// CHECK:STDOUT: !66 = !DILocation(line: 41, column: 37, scope: !61)
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "TestLeftShift", linkageName: "_CTestLeftShift.Main", scope: null, file: !1, line: 44, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !68)
// CHECK:STDOUT: !68 = !{!69, !70}
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 1, scope: !67, type: !7)
// CHECK:STDOUT: !70 = !DILocalVariable(arg: 2, scope: !67, type: !7)
// CHECK:STDOUT: !71 = !DILocation(line: 44, column: 50, scope: !67)
// CHECK:STDOUT: !72 = !DILocation(line: 44, column: 43, scope: !67)
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "TestRightShift", linkageName: "_CTestRightShift.Main", scope: null, file: !1, line: 47, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !74)
// CHECK:STDOUT: !74 = !{!75, !76}
// CHECK:STDOUT: !75 = !DILocalVariable(arg: 1, scope: !73, type: !7)
// CHECK:STDOUT: !76 = !DILocalVariable(arg: 2, scope: !73, type: !7)
// CHECK:STDOUT: !77 = !DILocation(line: 47, column: 51, scope: !73)
// CHECK:STDOUT: !78 = !DILocation(line: 47, column: 44, scope: !73)
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "TestEq", linkageName: "_CTestEq.Main", scope: null, file: !1, line: 50, type: !80, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !83)
// CHECK:STDOUT: !80 = !DISubroutineType(types: !81)
// CHECK:STDOUT: !81 = !{!82, !7, !7}
// CHECK:STDOUT: !82 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !83 = !{!84, !85}
// CHECK:STDOUT: !84 = !DILocalVariable(arg: 1, scope: !79, type: !7)
// CHECK:STDOUT: !85 = !DILocalVariable(arg: 2, scope: !79, type: !7)
// CHECK:STDOUT: !86 = !DILocation(line: 50, column: 44, scope: !79)
// CHECK:STDOUT: !87 = !DILocation(line: 50, column: 37, scope: !79)
// CHECK:STDOUT: !88 = distinct !DISubprogram(name: "TestNeq", linkageName: "_CTestNeq.Main", scope: null, file: !1, line: 53, type: !80, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !89)
// CHECK:STDOUT: !89 = !{!90, !91}
// CHECK:STDOUT: !90 = !DILocalVariable(arg: 1, scope: !88, type: !7)
// CHECK:STDOUT: !91 = !DILocalVariable(arg: 2, scope: !88, type: !7)
// CHECK:STDOUT: !92 = !DILocation(line: 53, column: 45, scope: !88)
// CHECK:STDOUT: !93 = !DILocation(line: 53, column: 38, scope: !88)
// CHECK:STDOUT: !94 = distinct !DISubprogram(name: "TestLess", linkageName: "_CTestLess.Main", scope: null, file: !1, line: 56, type: !80, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !95)
// CHECK:STDOUT: !95 = !{!96, !97}
// CHECK:STDOUT: !96 = !DILocalVariable(arg: 1, scope: !94, type: !7)
// CHECK:STDOUT: !97 = !DILocalVariable(arg: 2, scope: !94, type: !7)
// CHECK:STDOUT: !98 = !DILocation(line: 56, column: 46, scope: !94)
// CHECK:STDOUT: !99 = !DILocation(line: 56, column: 39, scope: !94)
// CHECK:STDOUT: !100 = distinct !DISubprogram(name: "TestLessEq", linkageName: "_CTestLessEq.Main", scope: null, file: !1, line: 59, type: !80, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !101)
// CHECK:STDOUT: !101 = !{!102, !103}
// CHECK:STDOUT: !102 = !DILocalVariable(arg: 1, scope: !100, type: !7)
// CHECK:STDOUT: !103 = !DILocalVariable(arg: 2, scope: !100, type: !7)
// CHECK:STDOUT: !104 = !DILocation(line: 59, column: 48, scope: !100)
// CHECK:STDOUT: !105 = !DILocation(line: 59, column: 41, scope: !100)
// CHECK:STDOUT: !106 = distinct !DISubprogram(name: "TestGreater", linkageName: "_CTestGreater.Main", scope: null, file: !1, line: 62, type: !80, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !107)
// CHECK:STDOUT: !107 = !{!108, !109}
// CHECK:STDOUT: !108 = !DILocalVariable(arg: 1, scope: !106, type: !7)
// CHECK:STDOUT: !109 = !DILocalVariable(arg: 2, scope: !106, type: !7)
// CHECK:STDOUT: !110 = !DILocation(line: 62, column: 49, scope: !106)
// CHECK:STDOUT: !111 = !DILocation(line: 62, column: 42, scope: !106)
// CHECK:STDOUT: !112 = distinct !DISubprogram(name: "TestGreaterEq", linkageName: "_CTestGreaterEq.Main", scope: null, file: !1, line: 65, type: !80, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !113)
// CHECK:STDOUT: !113 = !{!114, !115}
// CHECK:STDOUT: !114 = !DILocalVariable(arg: 1, scope: !112, type: !7)
// CHECK:STDOUT: !115 = !DILocalVariable(arg: 2, scope: !112, type: !7)
// CHECK:STDOUT: !116 = !DILocation(line: 65, column: 51, scope: !112)
// CHECK:STDOUT: !117 = !DILocation(line: 65, column: 44, scope: !112)
