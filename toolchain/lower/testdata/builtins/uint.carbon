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

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'uint.carbon'
// CHECK:STDOUT: source_filename = "uint.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestNegate.Main(i64 %a) #0 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Negate.call = sub i64 0, %a, !dbg !47
// CHECK:STDOUT:   ret i64 %Negate.call, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestAdd.Main(i64 %a, i64 %b) #0 !dbg !50 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Add.call = add i64 %a, %b, !dbg !59
// CHECK:STDOUT:   ret i64 %Add.call, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestSub.Main(i64 %a, i64 %b) #0 !dbg !62 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Sub.call = sub i64 %a, %b, !dbg !71
// CHECK:STDOUT:   ret i64 %Sub.call, !dbg !72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestMul.Main(i64 %a, i64 %b) #0 !dbg !74 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Mul.call = mul i64 %a, %b, !dbg !83
// CHECK:STDOUT:   ret i64 %Mul.call, !dbg !84
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestDiv.Main(i64 %a, i64 %b) #0 !dbg !86 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Div.call = udiv i64 %a, %b, !dbg !95
// CHECK:STDOUT:   ret i64 %Div.call, !dbg !96
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestMod.Main(i64 %a, i64 %b) #0 !dbg !98 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Mod.call = urem i64 %a, %b, !dbg !107
// CHECK:STDOUT:   ret i64 %Mod.call, !dbg !108
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestComplement.Main(i64 %a) #0 !dbg !110 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Complement.call = xor i64 -1, %a, !dbg !116
// CHECK:STDOUT:   ret i64 %Complement.call, !dbg !117
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestAnd.Main(i64 %a, i64 %b) #0 !dbg !119 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %And.call = and i64 %a, %b, !dbg !128
// CHECK:STDOUT:   ret i64 %And.call, !dbg !129
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestOr.Main(i64 %a, i64 %b) #0 !dbg !131 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Or.call = or i64 %a, %b, !dbg !140
// CHECK:STDOUT:   ret i64 %Or.call, !dbg !141
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestXor.Main(i64 %a, i64 %b) #0 !dbg !143 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Xor.call = xor i64 %a, %b, !dbg !152
// CHECK:STDOUT:   ret i64 %Xor.call, !dbg !153
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestLeftShift.Main(i64 %a, i64 %b) #0 !dbg !155 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %LeftShift.call = shl i64 %a, %b, !dbg !164
// CHECK:STDOUT:   ret i64 %LeftShift.call, !dbg !165
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestRightShift.Main(i64 %a, i64 %b) #0 !dbg !167 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %RightShift.call = lshr i64 %a, %b, !dbg !176
// CHECK:STDOUT:   ret i64 %RightShift.call, !dbg !177
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestEq.Main(i64 %a, i64 %b) #0 !dbg !179 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Eq.call = icmp eq i64 %a, %b, !dbg !188
// CHECK:STDOUT:   ret i1 %Eq.call, !dbg !189
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestNeq.Main(i64 %a, i64 %b) #0 !dbg !191 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Neq.call = icmp ne i64 %a, %b, !dbg !200
// CHECK:STDOUT:   ret i1 %Neq.call, !dbg !201
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestLess.Main(i64 %a, i64 %b) #0 !dbg !203 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Less.call = icmp ult i64 %a, %b, !dbg !212
// CHECK:STDOUT:   ret i1 %Less.call, !dbg !213
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestLessEq.Main(i64 %a, i64 %b) #0 !dbg !215 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %LessEq.call = icmp ule i64 %a, %b, !dbg !224
// CHECK:STDOUT:   ret i1 %LessEq.call, !dbg !225
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestGreater.Main(i64 %a, i64 %b) #0 !dbg !227 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Greater.call = icmp ugt i64 %a, %b, !dbg !236
// CHECK:STDOUT:   ret i1 %Greater.call, !dbg !237
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestGreaterEq.Main(i64 %a, i64 %b) #0 !dbg !239 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %GreaterEq.call = icmp uge i64 %a, %b, !dbg !248
// CHECK:STDOUT:   ret i1 %GreaterEq.call, !dbg !249
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!33}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !17 = !DIBasicType(name: "int", size: 64, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !32 = !DIFile(filename: "uint.carbon", directory: "")
// CHECK:STDOUT: !33 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !32, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !34 = !{!17, !17}
// CHECK:STDOUT: !35 = !DISubroutineType(types: !34)
// CHECK:STDOUT: !36 = !{!17, !17, !17}
// CHECK:STDOUT: !37 = !DISubroutineType(types: !36)
// CHECK:STDOUT: !38 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !39 = !{!38, !17, !17}
// CHECK:STDOUT: !40 = !DISubroutineType(types: !39)
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "TestNegate", linkageName: "_CTestNegate.Main", scope: null, file: !32, line: 14, type: !35, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !49)
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 1, scope: !41, type: !17)
// CHECK:STDOUT: !47 = !DILocation(line: 14, column: 39, scope: !41)
// CHECK:STDOUT: !48 = !DILocation(line: 14, column: 32, scope: !41)
// CHECK:STDOUT: !49 = !{!42}
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "TestAdd", linkageName: "_CTestAdd.Main", scope: null, file: !32, line: 17, type: !37, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !61)
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !50, type: !17)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 2, scope: !50, type: !17)
// CHECK:STDOUT: !59 = !DILocation(line: 17, column: 44, scope: !50)
// CHECK:STDOUT: !60 = !DILocation(line: 17, column: 37, scope: !50)
// CHECK:STDOUT: !61 = !{!51, !52}
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "TestSub", linkageName: "_CTestSub.Main", scope: null, file: !32, line: 20, type: !37, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !73)
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !62, type: !17)
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 2, scope: !62, type: !17)
// CHECK:STDOUT: !71 = !DILocation(line: 20, column: 44, scope: !62)
// CHECK:STDOUT: !72 = !DILocation(line: 20, column: 37, scope: !62)
// CHECK:STDOUT: !73 = !{!63, !64}
// CHECK:STDOUT: !74 = distinct !DISubprogram(name: "TestMul", linkageName: "_CTestMul.Main", scope: null, file: !32, line: 23, type: !37, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !85)
// CHECK:STDOUT: !75 = !DILocalVariable(arg: 1, scope: !74, type: !17)
// CHECK:STDOUT: !76 = !DILocalVariable(arg: 2, scope: !74, type: !17)
// CHECK:STDOUT: !83 = !DILocation(line: 23, column: 44, scope: !74)
// CHECK:STDOUT: !84 = !DILocation(line: 23, column: 37, scope: !74)
// CHECK:STDOUT: !85 = !{!75, !76}
// CHECK:STDOUT: !86 = distinct !DISubprogram(name: "TestDiv", linkageName: "_CTestDiv.Main", scope: null, file: !32, line: 26, type: !37, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !97)
// CHECK:STDOUT: !87 = !DILocalVariable(arg: 1, scope: !86, type: !17)
// CHECK:STDOUT: !88 = !DILocalVariable(arg: 2, scope: !86, type: !17)
// CHECK:STDOUT: !95 = !DILocation(line: 26, column: 44, scope: !86)
// CHECK:STDOUT: !96 = !DILocation(line: 26, column: 37, scope: !86)
// CHECK:STDOUT: !97 = !{!87, !88}
// CHECK:STDOUT: !98 = distinct !DISubprogram(name: "TestMod", linkageName: "_CTestMod.Main", scope: null, file: !32, line: 29, type: !37, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !109)
// CHECK:STDOUT: !99 = !DILocalVariable(arg: 1, scope: !98, type: !17)
// CHECK:STDOUT: !100 = !DILocalVariable(arg: 2, scope: !98, type: !17)
// CHECK:STDOUT: !107 = !DILocation(line: 29, column: 44, scope: !98)
// CHECK:STDOUT: !108 = !DILocation(line: 29, column: 37, scope: !98)
// CHECK:STDOUT: !109 = !{!99, !100}
// CHECK:STDOUT: !110 = distinct !DISubprogram(name: "TestComplement", linkageName: "_CTestComplement.Main", scope: null, file: !32, line: 32, type: !35, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !118)
// CHECK:STDOUT: !111 = !DILocalVariable(arg: 1, scope: !110, type: !17)
// CHECK:STDOUT: !116 = !DILocation(line: 32, column: 43, scope: !110)
// CHECK:STDOUT: !117 = !DILocation(line: 32, column: 36, scope: !110)
// CHECK:STDOUT: !118 = !{!111}
// CHECK:STDOUT: !119 = distinct !DISubprogram(name: "TestAnd", linkageName: "_CTestAnd.Main", scope: null, file: !32, line: 35, type: !37, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !130)
// CHECK:STDOUT: !120 = !DILocalVariable(arg: 1, scope: !119, type: !17)
// CHECK:STDOUT: !121 = !DILocalVariable(arg: 2, scope: !119, type: !17)
// CHECK:STDOUT: !128 = !DILocation(line: 35, column: 44, scope: !119)
// CHECK:STDOUT: !129 = !DILocation(line: 35, column: 37, scope: !119)
// CHECK:STDOUT: !130 = !{!120, !121}
// CHECK:STDOUT: !131 = distinct !DISubprogram(name: "TestOr", linkageName: "_CTestOr.Main", scope: null, file: !32, line: 38, type: !37, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !142)
// CHECK:STDOUT: !132 = !DILocalVariable(arg: 1, scope: !131, type: !17)
// CHECK:STDOUT: !133 = !DILocalVariable(arg: 2, scope: !131, type: !17)
// CHECK:STDOUT: !140 = !DILocation(line: 38, column: 43, scope: !131)
// CHECK:STDOUT: !141 = !DILocation(line: 38, column: 36, scope: !131)
// CHECK:STDOUT: !142 = !{!132, !133}
// CHECK:STDOUT: !143 = distinct !DISubprogram(name: "TestXor", linkageName: "_CTestXor.Main", scope: null, file: !32, line: 41, type: !37, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !154)
// CHECK:STDOUT: !144 = !DILocalVariable(arg: 1, scope: !143, type: !17)
// CHECK:STDOUT: !145 = !DILocalVariable(arg: 2, scope: !143, type: !17)
// CHECK:STDOUT: !152 = !DILocation(line: 41, column: 44, scope: !143)
// CHECK:STDOUT: !153 = !DILocation(line: 41, column: 37, scope: !143)
// CHECK:STDOUT: !154 = !{!144, !145}
// CHECK:STDOUT: !155 = distinct !DISubprogram(name: "TestLeftShift", linkageName: "_CTestLeftShift.Main", scope: null, file: !32, line: 44, type: !37, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !166)
// CHECK:STDOUT: !156 = !DILocalVariable(arg: 1, scope: !155, type: !17)
// CHECK:STDOUT: !157 = !DILocalVariable(arg: 2, scope: !155, type: !17)
// CHECK:STDOUT: !164 = !DILocation(line: 44, column: 50, scope: !155)
// CHECK:STDOUT: !165 = !DILocation(line: 44, column: 43, scope: !155)
// CHECK:STDOUT: !166 = !{!156, !157}
// CHECK:STDOUT: !167 = distinct !DISubprogram(name: "TestRightShift", linkageName: "_CTestRightShift.Main", scope: null, file: !32, line: 47, type: !37, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !178)
// CHECK:STDOUT: !168 = !DILocalVariable(arg: 1, scope: !167, type: !17)
// CHECK:STDOUT: !169 = !DILocalVariable(arg: 2, scope: !167, type: !17)
// CHECK:STDOUT: !176 = !DILocation(line: 47, column: 51, scope: !167)
// CHECK:STDOUT: !177 = !DILocation(line: 47, column: 44, scope: !167)
// CHECK:STDOUT: !178 = !{!168, !169}
// CHECK:STDOUT: !179 = distinct !DISubprogram(name: "TestEq", linkageName: "_CTestEq.Main", scope: null, file: !32, line: 50, type: !40, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !190)
// CHECK:STDOUT: !180 = !DILocalVariable(arg: 1, scope: !179, type: !17)
// CHECK:STDOUT: !181 = !DILocalVariable(arg: 2, scope: !179, type: !17)
// CHECK:STDOUT: !188 = !DILocation(line: 50, column: 44, scope: !179)
// CHECK:STDOUT: !189 = !DILocation(line: 50, column: 37, scope: !179)
// CHECK:STDOUT: !190 = !{!180, !181}
// CHECK:STDOUT: !191 = distinct !DISubprogram(name: "TestNeq", linkageName: "_CTestNeq.Main", scope: null, file: !32, line: 53, type: !40, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !202)
// CHECK:STDOUT: !192 = !DILocalVariable(arg: 1, scope: !191, type: !17)
// CHECK:STDOUT: !193 = !DILocalVariable(arg: 2, scope: !191, type: !17)
// CHECK:STDOUT: !200 = !DILocation(line: 53, column: 45, scope: !191)
// CHECK:STDOUT: !201 = !DILocation(line: 53, column: 38, scope: !191)
// CHECK:STDOUT: !202 = !{!192, !193}
// CHECK:STDOUT: !203 = distinct !DISubprogram(name: "TestLess", linkageName: "_CTestLess.Main", scope: null, file: !32, line: 56, type: !40, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !214)
// CHECK:STDOUT: !204 = !DILocalVariable(arg: 1, scope: !203, type: !17)
// CHECK:STDOUT: !205 = !DILocalVariable(arg: 2, scope: !203, type: !17)
// CHECK:STDOUT: !212 = !DILocation(line: 56, column: 46, scope: !203)
// CHECK:STDOUT: !213 = !DILocation(line: 56, column: 39, scope: !203)
// CHECK:STDOUT: !214 = !{!204, !205}
// CHECK:STDOUT: !215 = distinct !DISubprogram(name: "TestLessEq", linkageName: "_CTestLessEq.Main", scope: null, file: !32, line: 59, type: !40, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !226)
// CHECK:STDOUT: !216 = !DILocalVariable(arg: 1, scope: !215, type: !17)
// CHECK:STDOUT: !217 = !DILocalVariable(arg: 2, scope: !215, type: !17)
// CHECK:STDOUT: !224 = !DILocation(line: 59, column: 48, scope: !215)
// CHECK:STDOUT: !225 = !DILocation(line: 59, column: 41, scope: !215)
// CHECK:STDOUT: !226 = !{!216, !217}
// CHECK:STDOUT: !227 = distinct !DISubprogram(name: "TestGreater", linkageName: "_CTestGreater.Main", scope: null, file: !32, line: 62, type: !40, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !238)
// CHECK:STDOUT: !228 = !DILocalVariable(arg: 1, scope: !227, type: !17)
// CHECK:STDOUT: !229 = !DILocalVariable(arg: 2, scope: !227, type: !17)
// CHECK:STDOUT: !236 = !DILocation(line: 62, column: 49, scope: !227)
// CHECK:STDOUT: !237 = !DILocation(line: 62, column: 42, scope: !227)
// CHECK:STDOUT: !238 = !{!228, !229}
// CHECK:STDOUT: !239 = distinct !DISubprogram(name: "TestGreaterEq", linkageName: "_CTestGreaterEq.Main", scope: null, file: !32, line: 65, type: !40, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !250)
// CHECK:STDOUT: !240 = !DILocalVariable(arg: 1, scope: !239, type: !17)
// CHECK:STDOUT: !241 = !DILocalVariable(arg: 2, scope: !239, type: !17)
// CHECK:STDOUT: !248 = !DILocation(line: 65, column: 51, scope: !239)
// CHECK:STDOUT: !249 = !DILocation(line: 65, column: 44, scope: !239)
// CHECK:STDOUT: !250 = !{!240, !241}
// CHECK:STDOUT:
