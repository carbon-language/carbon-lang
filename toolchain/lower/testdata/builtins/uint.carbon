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
// CHECK:STDOUT: define i64 @_CTestNegate.Main(i64 %a) #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Negate.call = sub i64 0, %a, !dbg !50
// CHECK:STDOUT:   ret i64 %Negate.call, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestAdd.Main(i64 %a, i64 %b) #0 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Add.call = add i64 %a, %b, !dbg !62
// CHECK:STDOUT:   ret i64 %Add.call, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestSub.Main(i64 %a, i64 %b) #0 !dbg !65 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Sub.call = sub i64 %a, %b, !dbg !74
// CHECK:STDOUT:   ret i64 %Sub.call, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestMul.Main(i64 %a, i64 %b) #0 !dbg !77 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Mul.call = mul i64 %a, %b, !dbg !86
// CHECK:STDOUT:   ret i64 %Mul.call, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestDiv.Main(i64 %a, i64 %b) #0 !dbg !89 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Div.call = udiv i64 %a, %b, !dbg !98
// CHECK:STDOUT:   ret i64 %Div.call, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestMod.Main(i64 %a, i64 %b) #0 !dbg !101 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Mod.call = urem i64 %a, %b, !dbg !110
// CHECK:STDOUT:   ret i64 %Mod.call, !dbg !111
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestComplement.Main(i64 %a) #0 !dbg !113 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Complement.call = xor i64 -1, %a, !dbg !119
// CHECK:STDOUT:   ret i64 %Complement.call, !dbg !120
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestAnd.Main(i64 %a, i64 %b) #0 !dbg !122 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %And.call = and i64 %a, %b, !dbg !131
// CHECK:STDOUT:   ret i64 %And.call, !dbg !132
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestOr.Main(i64 %a, i64 %b) #0 !dbg !134 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Or.call = or i64 %a, %b, !dbg !143
// CHECK:STDOUT:   ret i64 %Or.call, !dbg !144
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestXor.Main(i64 %a, i64 %b) #0 !dbg !146 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Xor.call = xor i64 %a, %b, !dbg !155
// CHECK:STDOUT:   ret i64 %Xor.call, !dbg !156
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestLeftShift.Main(i64 %a, i64 %b) #0 !dbg !158 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %LeftShift.call = shl i64 %a, %b, !dbg !167
// CHECK:STDOUT:   ret i64 %LeftShift.call, !dbg !168
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestRightShift.Main(i64 %a, i64 %b) #0 !dbg !170 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %RightShift.call = lshr i64 %a, %b, !dbg !179
// CHECK:STDOUT:   ret i64 %RightShift.call, !dbg !180
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestEq.Main(i64 %a, i64 %b) #0 !dbg !182 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Eq.call = icmp eq i64 %a, %b, !dbg !191
// CHECK:STDOUT:   ret i1 %Eq.call, !dbg !192
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestNeq.Main(i64 %a, i64 %b) #0 !dbg !194 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Neq.call = icmp ne i64 %a, %b, !dbg !203
// CHECK:STDOUT:   ret i1 %Neq.call, !dbg !204
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestLess.Main(i64 %a, i64 %b) #0 !dbg !206 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Less.call = icmp ult i64 %a, %b, !dbg !215
// CHECK:STDOUT:   ret i1 %Less.call, !dbg !216
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestLessEq.Main(i64 %a, i64 %b) #0 !dbg !218 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %LessEq.call = icmp ule i64 %a, %b, !dbg !227
// CHECK:STDOUT:   ret i1 %LessEq.call, !dbg !228
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestGreater.Main(i64 %a, i64 %b) #0 !dbg !230 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Greater.call = icmp ugt i64 %a, %b, !dbg !239
// CHECK:STDOUT:   ret i1 %Greater.call, !dbg !240
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestGreaterEq.Main(i64 %a, i64 %b) #0 !dbg !242 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %GreaterEq.call = icmp uge i64 %a, %b, !dbg !251
// CHECK:STDOUT:   ret i1 %GreaterEq.call, !dbg !252
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!36}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 64, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !35 = !DIFile(filename: "uint.carbon", directory: "")
// CHECK:STDOUT: !36 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !35, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !37 = !{!20, !20}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = !{!20, !20, !20}
// CHECK:STDOUT: !40 = !DISubroutineType(types: !39)
// CHECK:STDOUT: !41 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !42 = !{!41, !20, !20}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "TestNegate", linkageName: "_CTestNegate.Main", scope: null, file: !35, line: 14, type: !38, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !52)
// CHECK:STDOUT: !45 = !DILocalVariable(arg: 1, scope: !44, type: !20)
// CHECK:STDOUT: !50 = !DILocation(line: 14, column: 39, scope: !44)
// CHECK:STDOUT: !51 = !DILocation(line: 14, column: 32, scope: !44)
// CHECK:STDOUT: !52 = !{!45}
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "TestAdd", linkageName: "_CTestAdd.Main", scope: null, file: !35, line: 17, type: !40, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !64)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !53, type: !20)
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 2, scope: !53, type: !20)
// CHECK:STDOUT: !62 = !DILocation(line: 17, column: 44, scope: !53)
// CHECK:STDOUT: !63 = !DILocation(line: 17, column: 37, scope: !53)
// CHECK:STDOUT: !64 = !{!54, !55}
// CHECK:STDOUT: !65 = distinct !DISubprogram(name: "TestSub", linkageName: "_CTestSub.Main", scope: null, file: !35, line: 20, type: !40, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !76)
// CHECK:STDOUT: !66 = !DILocalVariable(arg: 1, scope: !65, type: !20)
// CHECK:STDOUT: !67 = !DILocalVariable(arg: 2, scope: !65, type: !20)
// CHECK:STDOUT: !74 = !DILocation(line: 20, column: 44, scope: !65)
// CHECK:STDOUT: !75 = !DILocation(line: 20, column: 37, scope: !65)
// CHECK:STDOUT: !76 = !{!66, !67}
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "TestMul", linkageName: "_CTestMul.Main", scope: null, file: !35, line: 23, type: !40, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !88)
// CHECK:STDOUT: !78 = !DILocalVariable(arg: 1, scope: !77, type: !20)
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 2, scope: !77, type: !20)
// CHECK:STDOUT: !86 = !DILocation(line: 23, column: 44, scope: !77)
// CHECK:STDOUT: !87 = !DILocation(line: 23, column: 37, scope: !77)
// CHECK:STDOUT: !88 = !{!78, !79}
// CHECK:STDOUT: !89 = distinct !DISubprogram(name: "TestDiv", linkageName: "_CTestDiv.Main", scope: null, file: !35, line: 26, type: !40, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !100)
// CHECK:STDOUT: !90 = !DILocalVariable(arg: 1, scope: !89, type: !20)
// CHECK:STDOUT: !91 = !DILocalVariable(arg: 2, scope: !89, type: !20)
// CHECK:STDOUT: !98 = !DILocation(line: 26, column: 44, scope: !89)
// CHECK:STDOUT: !99 = !DILocation(line: 26, column: 37, scope: !89)
// CHECK:STDOUT: !100 = !{!90, !91}
// CHECK:STDOUT: !101 = distinct !DISubprogram(name: "TestMod", linkageName: "_CTestMod.Main", scope: null, file: !35, line: 29, type: !40, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !112)
// CHECK:STDOUT: !102 = !DILocalVariable(arg: 1, scope: !101, type: !20)
// CHECK:STDOUT: !103 = !DILocalVariable(arg: 2, scope: !101, type: !20)
// CHECK:STDOUT: !110 = !DILocation(line: 29, column: 44, scope: !101)
// CHECK:STDOUT: !111 = !DILocation(line: 29, column: 37, scope: !101)
// CHECK:STDOUT: !112 = !{!102, !103}
// CHECK:STDOUT: !113 = distinct !DISubprogram(name: "TestComplement", linkageName: "_CTestComplement.Main", scope: null, file: !35, line: 32, type: !38, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !121)
// CHECK:STDOUT: !114 = !DILocalVariable(arg: 1, scope: !113, type: !20)
// CHECK:STDOUT: !119 = !DILocation(line: 32, column: 43, scope: !113)
// CHECK:STDOUT: !120 = !DILocation(line: 32, column: 36, scope: !113)
// CHECK:STDOUT: !121 = !{!114}
// CHECK:STDOUT: !122 = distinct !DISubprogram(name: "TestAnd", linkageName: "_CTestAnd.Main", scope: null, file: !35, line: 35, type: !40, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !133)
// CHECK:STDOUT: !123 = !DILocalVariable(arg: 1, scope: !122, type: !20)
// CHECK:STDOUT: !124 = !DILocalVariable(arg: 2, scope: !122, type: !20)
// CHECK:STDOUT: !131 = !DILocation(line: 35, column: 44, scope: !122)
// CHECK:STDOUT: !132 = !DILocation(line: 35, column: 37, scope: !122)
// CHECK:STDOUT: !133 = !{!123, !124}
// CHECK:STDOUT: !134 = distinct !DISubprogram(name: "TestOr", linkageName: "_CTestOr.Main", scope: null, file: !35, line: 38, type: !40, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !145)
// CHECK:STDOUT: !135 = !DILocalVariable(arg: 1, scope: !134, type: !20)
// CHECK:STDOUT: !136 = !DILocalVariable(arg: 2, scope: !134, type: !20)
// CHECK:STDOUT: !143 = !DILocation(line: 38, column: 43, scope: !134)
// CHECK:STDOUT: !144 = !DILocation(line: 38, column: 36, scope: !134)
// CHECK:STDOUT: !145 = !{!135, !136}
// CHECK:STDOUT: !146 = distinct !DISubprogram(name: "TestXor", linkageName: "_CTestXor.Main", scope: null, file: !35, line: 41, type: !40, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !157)
// CHECK:STDOUT: !147 = !DILocalVariable(arg: 1, scope: !146, type: !20)
// CHECK:STDOUT: !148 = !DILocalVariable(arg: 2, scope: !146, type: !20)
// CHECK:STDOUT: !155 = !DILocation(line: 41, column: 44, scope: !146)
// CHECK:STDOUT: !156 = !DILocation(line: 41, column: 37, scope: !146)
// CHECK:STDOUT: !157 = !{!147, !148}
// CHECK:STDOUT: !158 = distinct !DISubprogram(name: "TestLeftShift", linkageName: "_CTestLeftShift.Main", scope: null, file: !35, line: 44, type: !40, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !169)
// CHECK:STDOUT: !159 = !DILocalVariable(arg: 1, scope: !158, type: !20)
// CHECK:STDOUT: !160 = !DILocalVariable(arg: 2, scope: !158, type: !20)
// CHECK:STDOUT: !167 = !DILocation(line: 44, column: 50, scope: !158)
// CHECK:STDOUT: !168 = !DILocation(line: 44, column: 43, scope: !158)
// CHECK:STDOUT: !169 = !{!159, !160}
// CHECK:STDOUT: !170 = distinct !DISubprogram(name: "TestRightShift", linkageName: "_CTestRightShift.Main", scope: null, file: !35, line: 47, type: !40, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !181)
// CHECK:STDOUT: !171 = !DILocalVariable(arg: 1, scope: !170, type: !20)
// CHECK:STDOUT: !172 = !DILocalVariable(arg: 2, scope: !170, type: !20)
// CHECK:STDOUT: !179 = !DILocation(line: 47, column: 51, scope: !170)
// CHECK:STDOUT: !180 = !DILocation(line: 47, column: 44, scope: !170)
// CHECK:STDOUT: !181 = !{!171, !172}
// CHECK:STDOUT: !182 = distinct !DISubprogram(name: "TestEq", linkageName: "_CTestEq.Main", scope: null, file: !35, line: 50, type: !43, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !193)
// CHECK:STDOUT: !183 = !DILocalVariable(arg: 1, scope: !182, type: !20)
// CHECK:STDOUT: !184 = !DILocalVariable(arg: 2, scope: !182, type: !20)
// CHECK:STDOUT: !191 = !DILocation(line: 50, column: 44, scope: !182)
// CHECK:STDOUT: !192 = !DILocation(line: 50, column: 37, scope: !182)
// CHECK:STDOUT: !193 = !{!183, !184}
// CHECK:STDOUT: !194 = distinct !DISubprogram(name: "TestNeq", linkageName: "_CTestNeq.Main", scope: null, file: !35, line: 53, type: !43, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !205)
// CHECK:STDOUT: !195 = !DILocalVariable(arg: 1, scope: !194, type: !20)
// CHECK:STDOUT: !196 = !DILocalVariable(arg: 2, scope: !194, type: !20)
// CHECK:STDOUT: !203 = !DILocation(line: 53, column: 45, scope: !194)
// CHECK:STDOUT: !204 = !DILocation(line: 53, column: 38, scope: !194)
// CHECK:STDOUT: !205 = !{!195, !196}
// CHECK:STDOUT: !206 = distinct !DISubprogram(name: "TestLess", linkageName: "_CTestLess.Main", scope: null, file: !35, line: 56, type: !43, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !217)
// CHECK:STDOUT: !207 = !DILocalVariable(arg: 1, scope: !206, type: !20)
// CHECK:STDOUT: !208 = !DILocalVariable(arg: 2, scope: !206, type: !20)
// CHECK:STDOUT: !215 = !DILocation(line: 56, column: 46, scope: !206)
// CHECK:STDOUT: !216 = !DILocation(line: 56, column: 39, scope: !206)
// CHECK:STDOUT: !217 = !{!207, !208}
// CHECK:STDOUT: !218 = distinct !DISubprogram(name: "TestLessEq", linkageName: "_CTestLessEq.Main", scope: null, file: !35, line: 59, type: !43, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !229)
// CHECK:STDOUT: !219 = !DILocalVariable(arg: 1, scope: !218, type: !20)
// CHECK:STDOUT: !220 = !DILocalVariable(arg: 2, scope: !218, type: !20)
// CHECK:STDOUT: !227 = !DILocation(line: 59, column: 48, scope: !218)
// CHECK:STDOUT: !228 = !DILocation(line: 59, column: 41, scope: !218)
// CHECK:STDOUT: !229 = !{!219, !220}
// CHECK:STDOUT: !230 = distinct !DISubprogram(name: "TestGreater", linkageName: "_CTestGreater.Main", scope: null, file: !35, line: 62, type: !43, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !241)
// CHECK:STDOUT: !231 = !DILocalVariable(arg: 1, scope: !230, type: !20)
// CHECK:STDOUT: !232 = !DILocalVariable(arg: 2, scope: !230, type: !20)
// CHECK:STDOUT: !239 = !DILocation(line: 62, column: 49, scope: !230)
// CHECK:STDOUT: !240 = !DILocation(line: 62, column: 42, scope: !230)
// CHECK:STDOUT: !241 = !{!231, !232}
// CHECK:STDOUT: !242 = distinct !DISubprogram(name: "TestGreaterEq", linkageName: "_CTestGreaterEq.Main", scope: null, file: !35, line: 65, type: !43, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !253)
// CHECK:STDOUT: !243 = !DILocalVariable(arg: 1, scope: !242, type: !20)
// CHECK:STDOUT: !244 = !DILocalVariable(arg: 2, scope: !242, type: !20)
// CHECK:STDOUT: !251 = !DILocation(line: 65, column: 51, scope: !242)
// CHECK:STDOUT: !252 = !DILocation(line: 65, column: 44, scope: !242)
// CHECK:STDOUT: !253 = !{!243, !244}
// CHECK:STDOUT:
