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
// CHECK:STDOUT: define i64 @_CTestNegate.Main(i64 %a) #0 !dbg !43 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Negate.call = sub i64 0, %a, !dbg !49
// CHECK:STDOUT:   ret i64 %Negate.call, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestAdd.Main(i64 %a, i64 %b) #0 !dbg !52 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Add.call = add i64 %a, %b, !dbg !61
// CHECK:STDOUT:   ret i64 %Add.call, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestSub.Main(i64 %a, i64 %b) #0 !dbg !64 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Sub.call = sub i64 %a, %b, !dbg !73
// CHECK:STDOUT:   ret i64 %Sub.call, !dbg !74
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestMul.Main(i64 %a, i64 %b) #0 !dbg !76 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Mul.call = mul i64 %a, %b, !dbg !85
// CHECK:STDOUT:   ret i64 %Mul.call, !dbg !86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestDiv.Main(i64 %a, i64 %b) #0 !dbg !88 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Div.call = udiv i64 %a, %b, !dbg !97
// CHECK:STDOUT:   ret i64 %Div.call, !dbg !98
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestMod.Main(i64 %a, i64 %b) #0 !dbg !100 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Mod.call = urem i64 %a, %b, !dbg !109
// CHECK:STDOUT:   ret i64 %Mod.call, !dbg !110
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestComplement.Main(i64 %a) #0 !dbg !112 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Complement.call = xor i64 -1, %a, !dbg !118
// CHECK:STDOUT:   ret i64 %Complement.call, !dbg !119
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestAnd.Main(i64 %a, i64 %b) #0 !dbg !121 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %And.call = and i64 %a, %b, !dbg !130
// CHECK:STDOUT:   ret i64 %And.call, !dbg !131
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestOr.Main(i64 %a, i64 %b) #0 !dbg !133 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Or.call = or i64 %a, %b, !dbg !142
// CHECK:STDOUT:   ret i64 %Or.call, !dbg !143
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestXor.Main(i64 %a, i64 %b) #0 !dbg !145 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Xor.call = xor i64 %a, %b, !dbg !154
// CHECK:STDOUT:   ret i64 %Xor.call, !dbg !155
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestLeftShift.Main(i64 %a, i64 %b) #0 !dbg !157 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %LeftShift.call = shl i64 %a, %b, !dbg !166
// CHECK:STDOUT:   ret i64 %LeftShift.call, !dbg !167
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestRightShift.Main(i64 %a, i64 %b) #0 !dbg !169 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %RightShift.call = lshr i64 %a, %b, !dbg !178
// CHECK:STDOUT:   ret i64 %RightShift.call, !dbg !179
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestEq.Main(i64 %a, i64 %b) #0 !dbg !181 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Eq.call = icmp eq i64 %a, %b, !dbg !190
// CHECK:STDOUT:   ret i1 %Eq.call, !dbg !191
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestNeq.Main(i64 %a, i64 %b) #0 !dbg !193 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Neq.call = icmp ne i64 %a, %b, !dbg !202
// CHECK:STDOUT:   ret i1 %Neq.call, !dbg !203
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestLess.Main(i64 %a, i64 %b) #0 !dbg !205 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Less.call = icmp ult i64 %a, %b, !dbg !214
// CHECK:STDOUT:   ret i1 %Less.call, !dbg !215
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestLessEq.Main(i64 %a, i64 %b) #0 !dbg !217 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %LessEq.call = icmp ule i64 %a, %b, !dbg !226
// CHECK:STDOUT:   ret i1 %LessEq.call, !dbg !227
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestGreater.Main(i64 %a, i64 %b) #0 !dbg !229 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Greater.call = icmp ugt i64 %a, %b, !dbg !238
// CHECK:STDOUT:   ret i1 %Greater.call, !dbg !239
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestGreaterEq.Main(i64 %a, i64 %b) #0 !dbg !241 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %GreaterEq.call = icmp uge i64 %a, %b, !dbg !250
// CHECK:STDOUT:   ret i1 %GreaterEq.call, !dbg !251
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!35}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !19 = !DIBasicType(name: "int", size: 64, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !34 = !DIFile(filename: "uint.carbon", directory: "")
// CHECK:STDOUT: !35 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !34, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !36 = !{!19, !19}
// CHECK:STDOUT: !37 = !DISubroutineType(types: !36)
// CHECK:STDOUT: !38 = !{!19, !19, !19}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !40 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !41 = !{!40, !19, !19}
// CHECK:STDOUT: !42 = !DISubroutineType(types: !41)
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "TestNegate", linkageName: "_CTestNegate.Main", scope: null, file: !34, line: 14, type: !37, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !51)
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 1, scope: !43, type: !19)
// CHECK:STDOUT: !49 = !DILocation(line: 14, column: 39, scope: !43)
// CHECK:STDOUT: !50 = !DILocation(line: 14, column: 32, scope: !43)
// CHECK:STDOUT: !51 = !{!44}
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "TestAdd", linkageName: "_CTestAdd.Main", scope: null, file: !34, line: 17, type: !39, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !63)
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 1, scope: !52, type: !19)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 2, scope: !52, type: !19)
// CHECK:STDOUT: !61 = !DILocation(line: 17, column: 44, scope: !52)
// CHECK:STDOUT: !62 = !DILocation(line: 17, column: 37, scope: !52)
// CHECK:STDOUT: !63 = !{!53, !54}
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "TestSub", linkageName: "_CTestSub.Main", scope: null, file: !34, line: 20, type: !39, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !75)
// CHECK:STDOUT: !65 = !DILocalVariable(arg: 1, scope: !64, type: !19)
// CHECK:STDOUT: !66 = !DILocalVariable(arg: 2, scope: !64, type: !19)
// CHECK:STDOUT: !73 = !DILocation(line: 20, column: 44, scope: !64)
// CHECK:STDOUT: !74 = !DILocation(line: 20, column: 37, scope: !64)
// CHECK:STDOUT: !75 = !{!65, !66}
// CHECK:STDOUT: !76 = distinct !DISubprogram(name: "TestMul", linkageName: "_CTestMul.Main", scope: null, file: !34, line: 23, type: !39, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !87)
// CHECK:STDOUT: !77 = !DILocalVariable(arg: 1, scope: !76, type: !19)
// CHECK:STDOUT: !78 = !DILocalVariable(arg: 2, scope: !76, type: !19)
// CHECK:STDOUT: !85 = !DILocation(line: 23, column: 44, scope: !76)
// CHECK:STDOUT: !86 = !DILocation(line: 23, column: 37, scope: !76)
// CHECK:STDOUT: !87 = !{!77, !78}
// CHECK:STDOUT: !88 = distinct !DISubprogram(name: "TestDiv", linkageName: "_CTestDiv.Main", scope: null, file: !34, line: 26, type: !39, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !99)
// CHECK:STDOUT: !89 = !DILocalVariable(arg: 1, scope: !88, type: !19)
// CHECK:STDOUT: !90 = !DILocalVariable(arg: 2, scope: !88, type: !19)
// CHECK:STDOUT: !97 = !DILocation(line: 26, column: 44, scope: !88)
// CHECK:STDOUT: !98 = !DILocation(line: 26, column: 37, scope: !88)
// CHECK:STDOUT: !99 = !{!89, !90}
// CHECK:STDOUT: !100 = distinct !DISubprogram(name: "TestMod", linkageName: "_CTestMod.Main", scope: null, file: !34, line: 29, type: !39, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !111)
// CHECK:STDOUT: !101 = !DILocalVariable(arg: 1, scope: !100, type: !19)
// CHECK:STDOUT: !102 = !DILocalVariable(arg: 2, scope: !100, type: !19)
// CHECK:STDOUT: !109 = !DILocation(line: 29, column: 44, scope: !100)
// CHECK:STDOUT: !110 = !DILocation(line: 29, column: 37, scope: !100)
// CHECK:STDOUT: !111 = !{!101, !102}
// CHECK:STDOUT: !112 = distinct !DISubprogram(name: "TestComplement", linkageName: "_CTestComplement.Main", scope: null, file: !34, line: 32, type: !37, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !120)
// CHECK:STDOUT: !113 = !DILocalVariable(arg: 1, scope: !112, type: !19)
// CHECK:STDOUT: !118 = !DILocation(line: 32, column: 43, scope: !112)
// CHECK:STDOUT: !119 = !DILocation(line: 32, column: 36, scope: !112)
// CHECK:STDOUT: !120 = !{!113}
// CHECK:STDOUT: !121 = distinct !DISubprogram(name: "TestAnd", linkageName: "_CTestAnd.Main", scope: null, file: !34, line: 35, type: !39, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !132)
// CHECK:STDOUT: !122 = !DILocalVariable(arg: 1, scope: !121, type: !19)
// CHECK:STDOUT: !123 = !DILocalVariable(arg: 2, scope: !121, type: !19)
// CHECK:STDOUT: !130 = !DILocation(line: 35, column: 44, scope: !121)
// CHECK:STDOUT: !131 = !DILocation(line: 35, column: 37, scope: !121)
// CHECK:STDOUT: !132 = !{!122, !123}
// CHECK:STDOUT: !133 = distinct !DISubprogram(name: "TestOr", linkageName: "_CTestOr.Main", scope: null, file: !34, line: 38, type: !39, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !144)
// CHECK:STDOUT: !134 = !DILocalVariable(arg: 1, scope: !133, type: !19)
// CHECK:STDOUT: !135 = !DILocalVariable(arg: 2, scope: !133, type: !19)
// CHECK:STDOUT: !142 = !DILocation(line: 38, column: 43, scope: !133)
// CHECK:STDOUT: !143 = !DILocation(line: 38, column: 36, scope: !133)
// CHECK:STDOUT: !144 = !{!134, !135}
// CHECK:STDOUT: !145 = distinct !DISubprogram(name: "TestXor", linkageName: "_CTestXor.Main", scope: null, file: !34, line: 41, type: !39, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !156)
// CHECK:STDOUT: !146 = !DILocalVariable(arg: 1, scope: !145, type: !19)
// CHECK:STDOUT: !147 = !DILocalVariable(arg: 2, scope: !145, type: !19)
// CHECK:STDOUT: !154 = !DILocation(line: 41, column: 44, scope: !145)
// CHECK:STDOUT: !155 = !DILocation(line: 41, column: 37, scope: !145)
// CHECK:STDOUT: !156 = !{!146, !147}
// CHECK:STDOUT: !157 = distinct !DISubprogram(name: "TestLeftShift", linkageName: "_CTestLeftShift.Main", scope: null, file: !34, line: 44, type: !39, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !168)
// CHECK:STDOUT: !158 = !DILocalVariable(arg: 1, scope: !157, type: !19)
// CHECK:STDOUT: !159 = !DILocalVariable(arg: 2, scope: !157, type: !19)
// CHECK:STDOUT: !166 = !DILocation(line: 44, column: 50, scope: !157)
// CHECK:STDOUT: !167 = !DILocation(line: 44, column: 43, scope: !157)
// CHECK:STDOUT: !168 = !{!158, !159}
// CHECK:STDOUT: !169 = distinct !DISubprogram(name: "TestRightShift", linkageName: "_CTestRightShift.Main", scope: null, file: !34, line: 47, type: !39, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !180)
// CHECK:STDOUT: !170 = !DILocalVariable(arg: 1, scope: !169, type: !19)
// CHECK:STDOUT: !171 = !DILocalVariable(arg: 2, scope: !169, type: !19)
// CHECK:STDOUT: !178 = !DILocation(line: 47, column: 51, scope: !169)
// CHECK:STDOUT: !179 = !DILocation(line: 47, column: 44, scope: !169)
// CHECK:STDOUT: !180 = !{!170, !171}
// CHECK:STDOUT: !181 = distinct !DISubprogram(name: "TestEq", linkageName: "_CTestEq.Main", scope: null, file: !34, line: 50, type: !42, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !192)
// CHECK:STDOUT: !182 = !DILocalVariable(arg: 1, scope: !181, type: !19)
// CHECK:STDOUT: !183 = !DILocalVariable(arg: 2, scope: !181, type: !19)
// CHECK:STDOUT: !190 = !DILocation(line: 50, column: 44, scope: !181)
// CHECK:STDOUT: !191 = !DILocation(line: 50, column: 37, scope: !181)
// CHECK:STDOUT: !192 = !{!182, !183}
// CHECK:STDOUT: !193 = distinct !DISubprogram(name: "TestNeq", linkageName: "_CTestNeq.Main", scope: null, file: !34, line: 53, type: !42, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !204)
// CHECK:STDOUT: !194 = !DILocalVariable(arg: 1, scope: !193, type: !19)
// CHECK:STDOUT: !195 = !DILocalVariable(arg: 2, scope: !193, type: !19)
// CHECK:STDOUT: !202 = !DILocation(line: 53, column: 45, scope: !193)
// CHECK:STDOUT: !203 = !DILocation(line: 53, column: 38, scope: !193)
// CHECK:STDOUT: !204 = !{!194, !195}
// CHECK:STDOUT: !205 = distinct !DISubprogram(name: "TestLess", linkageName: "_CTestLess.Main", scope: null, file: !34, line: 56, type: !42, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !216)
// CHECK:STDOUT: !206 = !DILocalVariable(arg: 1, scope: !205, type: !19)
// CHECK:STDOUT: !207 = !DILocalVariable(arg: 2, scope: !205, type: !19)
// CHECK:STDOUT: !214 = !DILocation(line: 56, column: 46, scope: !205)
// CHECK:STDOUT: !215 = !DILocation(line: 56, column: 39, scope: !205)
// CHECK:STDOUT: !216 = !{!206, !207}
// CHECK:STDOUT: !217 = distinct !DISubprogram(name: "TestLessEq", linkageName: "_CTestLessEq.Main", scope: null, file: !34, line: 59, type: !42, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !228)
// CHECK:STDOUT: !218 = !DILocalVariable(arg: 1, scope: !217, type: !19)
// CHECK:STDOUT: !219 = !DILocalVariable(arg: 2, scope: !217, type: !19)
// CHECK:STDOUT: !226 = !DILocation(line: 59, column: 48, scope: !217)
// CHECK:STDOUT: !227 = !DILocation(line: 59, column: 41, scope: !217)
// CHECK:STDOUT: !228 = !{!218, !219}
// CHECK:STDOUT: !229 = distinct !DISubprogram(name: "TestGreater", linkageName: "_CTestGreater.Main", scope: null, file: !34, line: 62, type: !42, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !240)
// CHECK:STDOUT: !230 = !DILocalVariable(arg: 1, scope: !229, type: !19)
// CHECK:STDOUT: !231 = !DILocalVariable(arg: 2, scope: !229, type: !19)
// CHECK:STDOUT: !238 = !DILocation(line: 62, column: 49, scope: !229)
// CHECK:STDOUT: !239 = !DILocation(line: 62, column: 42, scope: !229)
// CHECK:STDOUT: !240 = !{!230, !231}
// CHECK:STDOUT: !241 = distinct !DISubprogram(name: "TestGreaterEq", linkageName: "_CTestGreaterEq.Main", scope: null, file: !34, line: 65, type: !42, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !252)
// CHECK:STDOUT: !242 = !DILocalVariable(arg: 1, scope: !241, type: !19)
// CHECK:STDOUT: !243 = !DILocalVariable(arg: 2, scope: !241, type: !19)
// CHECK:STDOUT: !250 = !DILocation(line: 65, column: 51, scope: !241)
// CHECK:STDOUT: !251 = !DILocation(line: 65, column: 44, scope: !241)
// CHECK:STDOUT: !252 = !{!242, !243}
// CHECK:STDOUT:
