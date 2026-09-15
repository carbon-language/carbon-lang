// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/builtins/int.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/builtins/int.carbon

// --- basic.carbon

library "[[@TEST_NAME]]";

fn Negate(a: i32) -> i32 = "int.snegate";
fn TestNegate(a: i32) -> i32 { return Negate(a); }

fn Add(a: i32, b: i32) -> i32 = "int.sadd";
fn TestAdd(a: i32, b: i32) -> i32 { return Add(a, b); }

fn Sub(a: i32, b: i32) -> i32 = "int.ssub";
fn TestSub(a: i32, b: i32) -> i32 { return Sub(a, b); }

fn Mul(a: i32, b: i32) -> i32 = "int.smul";
fn TestMul(a: i32, b: i32) -> i32 { return Mul(a, b); }

fn Div(a: i32, b: i32) -> i32 = "int.sdiv";
fn TestDiv(a: i32, b: i32) -> i32 { return Div(a, b); }

fn Mod(a: i32, b: i32) -> i32 = "int.smod";
fn TestMod(a: i32, b: i32) -> i32 { return Mod(a, b); }

fn Complement(a: i32) -> i32 = "int.complement";
fn TestComplement(a: i32) -> i32 { return Complement(a); }

fn And(a: i32, b: i32) -> i32 = "int.and";
fn TestAnd(a: i32, b: i32) -> i32 { return And(a, b); }

fn Or(a: i32, b: i32) -> i32 = "int.or";
fn TestOr(a: i32, b: i32) -> i32 { return Or(a, b); }

fn Xor(a: i32, b: i32) -> i32 = "int.xor";
fn TestXor(a: i32, b: i32) -> i32 { return Xor(a, b); }

fn LeftShift(a: i32, b: i32) -> i32 = "int.left_shift";
fn TestLeftShift(a: i32, b: i32) -> i32 { return LeftShift(a, b); }

fn ArithmeticRightShift(a: i32, b: i32) -> i32 = "int.right_shift";
fn TestArithmeticRightShift(a: i32, b: i32) -> i32 { return ArithmeticRightShift(a, b); }

fn LogicalRightShift(a: u32, b: u32) -> u32 = "int.right_shift";
fn TestLogicalRightShift(a: u32, b: u32) -> u32 { return LogicalRightShift(a, b); }

fn Eq(a: i32, b: i32) -> bool = "int.eq";
fn TestEq(a: i32, b: i32) -> bool { return Eq(a, b); }

fn Neq(a: i32, b: i32) -> bool = "int.neq";
fn TestNeq(a: i32, b: i32) -> bool { return Neq(a, b); }

fn Less(a: i32, b: i32) -> bool = "int.less";
fn TestLess(a: i32, b: i32) -> bool { return Less(a, b); }

fn LessEq(a: i32, b: i32) -> bool = "int.less_eq";
fn TestLessEq(a: i32, b: i32) -> bool { return LessEq(a, b); }

fn Greater(a: i32, b: i32) -> bool = "int.greater";
fn TestGreater(a: i32, b: i32) -> bool { return Greater(a, b); }

fn GreaterEq(a: i32, b: i32) -> bool = "int.greater_eq";
fn TestGreaterEq(a: i32, b: i32) -> bool { return GreaterEq(a, b); }

// --- compound_assign.carbon

library "[[@TEST_NAME]]";

fn SAdd(ref a: i32, b: i32) = "int.sadd_assign";
fn TestSAdd(ref a: i32, b: i32) { SAdd(ref a, b); }

fn SSub(ref a: i32, b: i32) = "int.ssub_assign";
fn TestSSub(ref a: i32, b: i32) { SSub(ref a, b); }

fn SMul(ref a: i32, b: i32) = "int.smul_assign";
fn TestSMul(ref a: i32, b: i32) { SMul(ref a, b); }

fn SDiv(ref a: i32, b: i32) = "int.sdiv_assign";
fn TestSDiv(ref a: i32, b: i32) { SDiv(ref a, b); }

fn SMod(ref a: i32, b: i32) = "int.smod_assign";
fn TestSMod(ref a: i32, b: i32) { SMod(ref a, b); }

fn UAdd(ref a: i32, b: i32) = "int.uadd_assign";
fn TestUAdd(ref a: i32, b: i32) { UAdd(ref a, b); }

fn USub(ref a: i32, b: i32) = "int.usub_assign";
fn TestUSub(ref a: i32, b: i32) { USub(ref a, b); }

fn UMul(ref a: i32, b: i32) = "int.umul_assign";
fn TestUMul(ref a: i32, b: i32) { UMul(ref a, b); }

fn UDiv(ref a: i32, b: i32) = "int.udiv_assign";
fn TestUDiv(ref a: i32, b: i32) { UDiv(ref a, b); }

fn UMod(ref a: i32, b: i32) = "int.umod_assign";
fn TestUMod(ref a: i32, b: i32) { UMod(ref a, b); }

fn And(ref a: i32, b: i32) = "int.and_assign";
fn TestAnd(ref a: i32, b: i32) { And(ref a, b); }

fn Or(ref a: i32, b: i32) = "int.or_assign";
fn TestOr(ref a: i32, b: i32) { Or(ref a, b); }

fn Xor(ref a: i32, b: i32) = "int.xor_assign";
fn TestXor(ref a: i32, b: i32) { Xor(ref a, b); }

fn LeftShift(ref a: i32, b: i32) = "int.left_shift_assign";
fn TestLeftShift(ref a: i32, b: i32) { LeftShift(ref a, b); }

fn ArithmeticRightShift(ref a: i32, b: u32) = "int.right_shift_assign";
fn TestArithmeticRightShift(ref a: i32, b: u32) { ArithmeticRightShift(ref a, b); }

fn LogicalRightShift(ref a: u32, b: u32) = "int.right_shift_assign";
fn TestLogicalRightShift(ref a: u32, b: u32) { LogicalRightShift(ref a, b); }

// --- mixed_shift.carbon

library "[[@TEST_NAME]]";

fn LeftShiftSmaller(a: i32, b: i16) -> i32 = "int.left_shift";
fn TestLeftShiftSmaller(a: i32, b: i16) -> i32 { return LeftShiftSmaller(a, b); }

fn RightShiftSmaller(a: i32, b: i16) -> i32 = "int.right_shift";
fn TestRightShiftSmaller(a: i32, b: i16) -> i32 { return RightShiftSmaller(a, b); }

fn LeftShiftLargerII(a: i16, b: i32) -> i16 = "int.left_shift";
fn TestLeftShiftLargerII(a: i16, b: i32) -> i16 { return LeftShiftLargerII(a, b); }

fn RightShiftLargerII(a: i16, b: i32) -> i16 = "int.right_shift";
fn TestRightShiftLargerII(a: i16, b: i32) -> i16 { return RightShiftLargerII(a, b); }

fn LeftShiftLargerIU(a: i16, b: u32) -> i16 = "int.left_shift";
fn TestLeftShiftLargerIU(a: i16, b: u32) -> i16 { return LeftShiftLargerIU(a, b); }

fn RightShiftLargerIU(a: i16, b: u32) -> i16 = "int.right_shift";
fn TestRightShiftLargerIU(a: i16, b: u32) -> i16 { return RightShiftLargerIU(a, b); }

fn LeftShiftLargerUI(a: u16, b: i32) -> u16 = "int.left_shift";
fn TestLeftShiftLargerUI(a: u16, b: i32) -> u16 { return LeftShiftLargerUI(a, b); }

fn RightShiftLargerUI(a: u16, b: i32) -> u16 = "int.right_shift";
fn TestRightShiftLargerUI(a: u16, b: i32) -> u16 { return RightShiftLargerUI(a, b); }

fn LeftShiftLargerUU(a: u16, b: u32) -> u16 = "int.left_shift";
fn TestLeftShiftLargerUU(a: u16, b: u32) -> u16 { return LeftShiftLargerUU(a, b); }

fn RightShiftLargerUU(a: u16, b: u32) -> u16 = "int.right_shift";
fn TestRightShiftLargerUU(a: u16, b: u32) -> u16 { return RightShiftLargerUU(a, b); }

// --- mixed_compare.carbon

library "[[@TEST_NAME]]";

fn Eq_u16_u32(a: u16, b: u32) -> bool = "int.eq";
fn Eq_i16_u32(a: i16, b: u32) -> bool = "int.eq";
fn Eq_u16_i32(a: u16, b: i32) -> bool = "int.eq";
fn Eq_i16_i32(a: i16, b: i32) -> bool = "int.eq";
fn Eq_i32_u32(a: i32, b: u32) -> bool = "int.eq";

fn TestEq_u16_u32(a: u16, b: u32) -> bool { return Eq_u16_u32(a, b); }
fn TestEq_i16_u32(a: i16, b: u32) -> bool { return Eq_i16_u32(a, b); }
fn TestEq_u16_i32(a: u16, b: i32) -> bool { return Eq_u16_i32(a, b); }
fn TestEq_i16_i32(a: i16, b: i32) -> bool { return Eq_i16_i32(a, b); }
fn TestEq_i32_u32(a: i32, b: u32) -> bool { return Eq_i32_u32(a, b); }

fn Less_u16_u32(a: u16, b: u32) -> bool = "int.less";
fn Less_i16_u32(a: i16, b: u32) -> bool = "int.less";
fn Less_u16_i32(a: u16, b: i32) -> bool = "int.less";
fn Less_i16_i32(a: i16, b: i32) -> bool = "int.less";
fn Less_i32_u32(a: i32, b: u32) -> bool = "int.less";

fn TestLess_u16_u32(a: u16, b: u32) -> bool { return Less_u16_u32(a, b); }
fn TestLess_i16_u32(a: i16, b: u32) -> bool { return Less_i16_u32(a, b); }
fn TestLess_u16_i32(a: u16, b: i32) -> bool { return Less_u16_i32(a, b); }
fn TestLess_i16_i32(a: i16, b: i32) -> bool { return Less_i16_i32(a, b); }
fn TestLess_i32_u32(a: i32, b: u32) -> bool { return Less_i32_u32(a, b); }

// --- convert.carbon

library "[[@TEST_NAME]]";

// Size preserving
fn Int32ToInt32(a: i32) -> i32 = "int.convert";
fn Int32ToUint32(a: i32) -> u32 = "int.convert";
fn Uint32ToInt32(a: u32) -> i32 = "int.convert";
fn Uint32ToUint32(a: u32) -> u32 = "int.convert";

fn TestInt32ToInt32(a: i32) -> i32 { return Int32ToInt32(a); }
fn TestInt32ToUint32(a: i32) -> u32 { return Int32ToUint32(a); }
fn TestUint32ToInt32(a: u32) -> i32 { return Uint32ToInt32(a); }
fn TestUint32ToUint32(a: u32) -> u32 { return Uint32ToUint32(a); }

// Narrowing
fn Int32ToInt16(a: i32) -> i16 = "int.convert";
fn Int32ToUint16(a: i32) -> u16 = "int.convert";
fn Uint32ToInt16(a: u32) -> i16 = "int.convert";
fn Uint32ToUint16(a: u32) -> u16 = "int.convert";

fn TestInt32ToInt16(a: i32) -> i16 { return Int32ToInt16(a); }
fn TestInt32ToUint16(a: i32) -> u16 { return Int32ToUint16(a); }
fn TestUint32ToInt16(a: u32) -> i16 { return Uint32ToInt16(a); }
fn TestUint32ToUint16(a: u32) -> u16 { return Uint32ToUint16(a); }

// Widening
fn Int32ToInt64(a: i32) -> i64 = "int.convert";
fn Int32ToUint64(a: i32) -> u64 = "int.convert";
fn Uint32ToInt64(a: u32) -> i64 = "int.convert";
fn Uint32ToUint64(a: u32) -> u64 = "int.convert";

fn TestInt32ToInt64(a: i32) -> i64 { return Int32ToInt64(a); }
fn TestInt32ToUint64(a: i32) -> u64 { return Int32ToUint64(a); }
fn TestUint32ToInt64(a: u32) -> i64 { return Uint32ToInt64(a); }
fn TestUint32ToUint64(a: u32) -> u64 { return Uint32ToUint64(a); }

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'basic.carbon'
// CHECK:STDOUT: source_filename = "basic.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestNegate.Main(i32 %a) #0 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Negate.call = sub i32 0, %a, !dbg !51
// CHECK:STDOUT:   ret i32 %Negate.call, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestAdd.Main(i32 %a, i32 %b) #0 !dbg !54 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Add.call = add i32 %a, %b, !dbg !63
// CHECK:STDOUT:   ret i32 %Add.call, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestSub.Main(i32 %a, i32 %b) #0 !dbg !66 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Sub.call = sub i32 %a, %b, !dbg !75
// CHECK:STDOUT:   ret i32 %Sub.call, !dbg !76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestMul.Main(i32 %a, i32 %b) #0 !dbg !78 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Mul.call = mul i32 %a, %b, !dbg !87
// CHECK:STDOUT:   ret i32 %Mul.call, !dbg !88
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestDiv.Main(i32 %a, i32 %b) #0 !dbg !90 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Div.call = sdiv i32 %a, %b, !dbg !99
// CHECK:STDOUT:   ret i32 %Div.call, !dbg !100
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestMod.Main(i32 %a, i32 %b) #0 !dbg !102 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Mod.call = srem i32 %a, %b, !dbg !111
// CHECK:STDOUT:   ret i32 %Mod.call, !dbg !112
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestComplement.Main(i32 %a) #0 !dbg !114 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Complement.call = xor i32 -1, %a, !dbg !120
// CHECK:STDOUT:   ret i32 %Complement.call, !dbg !121
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestAnd.Main(i32 %a, i32 %b) #0 !dbg !123 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %And.call = and i32 %a, %b, !dbg !132
// CHECK:STDOUT:   ret i32 %And.call, !dbg !133
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestOr.Main(i32 %a, i32 %b) #0 !dbg !135 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Or.call = or i32 %a, %b, !dbg !144
// CHECK:STDOUT:   ret i32 %Or.call, !dbg !145
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestXor.Main(i32 %a, i32 %b) #0 !dbg !147 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Xor.call = xor i32 %a, %b, !dbg !156
// CHECK:STDOUT:   ret i32 %Xor.call, !dbg !157
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestLeftShift.Main(i32 %a, i32 %b) #0 !dbg !159 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %LeftShift.call = shl i32 %a, %b, !dbg !168
// CHECK:STDOUT:   ret i32 %LeftShift.call, !dbg !169
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestArithmeticRightShift.Main(i32 %a, i32 %b) #0 !dbg !171 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ArithmeticRightShift.call = ashr i32 %a, %b, !dbg !180
// CHECK:STDOUT:   ret i32 %ArithmeticRightShift.call, !dbg !181
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestLogicalRightShift.Main(i32 %a, i32 %b) #0 !dbg !183 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %LogicalRightShift.call = lshr i32 %a, %b, !dbg !192
// CHECK:STDOUT:   ret i32 %LogicalRightShift.call, !dbg !193
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestEq.Main(i32 %a, i32 %b) #0 !dbg !195 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Eq.call = icmp eq i32 %a, %b, !dbg !204
// CHECK:STDOUT:   ret i1 %Eq.call, !dbg !205
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestNeq.Main(i32 %a, i32 %b) #0 !dbg !207 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Neq.call = icmp ne i32 %a, %b, !dbg !216
// CHECK:STDOUT:   ret i1 %Neq.call, !dbg !217
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestLess.Main(i32 %a, i32 %b) #0 !dbg !219 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Less.call = icmp slt i32 %a, %b, !dbg !228
// CHECK:STDOUT:   ret i1 %Less.call, !dbg !229
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestLessEq.Main(i32 %a, i32 %b) #0 !dbg !231 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %LessEq.call = icmp sle i32 %a, %b, !dbg !240
// CHECK:STDOUT:   ret i1 %LessEq.call, !dbg !241
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestGreater.Main(i32 %a, i32 %b) #0 !dbg !243 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Greater.call = icmp sgt i32 %a, %b, !dbg !252
// CHECK:STDOUT:   ret i1 %Greater.call, !dbg !253
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestGreaterEq.Main(i32 %a, i32 %b) #0 !dbg !255 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %GreaterEq.call = icmp sge i32 %a, %b, !dbg !264
// CHECK:STDOUT:   ret i1 %GreaterEq.call, !dbg !265
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!33}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !32 = !DIFile(filename: "basic.carbon", directory: "")
// CHECK:STDOUT: !33 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !32, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !34 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !35 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !36 = !{!34, !34}
// CHECK:STDOUT: !37 = !DISubroutineType(types: !36)
// CHECK:STDOUT: !38 = !{!34, !34, !34}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !40 = !{!35, !35, !35}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !42 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !43 = !{!42, !34, !34}
// CHECK:STDOUT: !44 = !DISubroutineType(types: !43)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "TestNegate", linkageName: "_CTestNegate.Main", scope: null, file: !32, line: 5, type: !37, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !53)
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 1, scope: !45, type: !34)
// CHECK:STDOUT: !51 = !DILocation(line: 5, column: 39, scope: !45)
// CHECK:STDOUT: !52 = !DILocation(line: 5, column: 32, scope: !45)
// CHECK:STDOUT: !53 = !{!46}
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "TestAdd", linkageName: "_CTestAdd.Main", scope: null, file: !32, line: 8, type: !39, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !65)
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 1, scope: !54, type: !34)
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 2, scope: !54, type: !34)
// CHECK:STDOUT: !63 = !DILocation(line: 8, column: 44, scope: !54)
// CHECK:STDOUT: !64 = !DILocation(line: 8, column: 37, scope: !54)
// CHECK:STDOUT: !65 = !{!55, !56}
// CHECK:STDOUT: !66 = distinct !DISubprogram(name: "TestSub", linkageName: "_CTestSub.Main", scope: null, file: !32, line: 11, type: !39, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !77)
// CHECK:STDOUT: !67 = !DILocalVariable(arg: 1, scope: !66, type: !34)
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 2, scope: !66, type: !34)
// CHECK:STDOUT: !75 = !DILocation(line: 11, column: 44, scope: !66)
// CHECK:STDOUT: !76 = !DILocation(line: 11, column: 37, scope: !66)
// CHECK:STDOUT: !77 = !{!67, !68}
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "TestMul", linkageName: "_CTestMul.Main", scope: null, file: !32, line: 14, type: !39, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !89)
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 1, scope: !78, type: !34)
// CHECK:STDOUT: !80 = !DILocalVariable(arg: 2, scope: !78, type: !34)
// CHECK:STDOUT: !87 = !DILocation(line: 14, column: 44, scope: !78)
// CHECK:STDOUT: !88 = !DILocation(line: 14, column: 37, scope: !78)
// CHECK:STDOUT: !89 = !{!79, !80}
// CHECK:STDOUT: !90 = distinct !DISubprogram(name: "TestDiv", linkageName: "_CTestDiv.Main", scope: null, file: !32, line: 17, type: !39, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !101)
// CHECK:STDOUT: !91 = !DILocalVariable(arg: 1, scope: !90, type: !34)
// CHECK:STDOUT: !92 = !DILocalVariable(arg: 2, scope: !90, type: !34)
// CHECK:STDOUT: !99 = !DILocation(line: 17, column: 44, scope: !90)
// CHECK:STDOUT: !100 = !DILocation(line: 17, column: 37, scope: !90)
// CHECK:STDOUT: !101 = !{!91, !92}
// CHECK:STDOUT: !102 = distinct !DISubprogram(name: "TestMod", linkageName: "_CTestMod.Main", scope: null, file: !32, line: 20, type: !39, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !113)
// CHECK:STDOUT: !103 = !DILocalVariable(arg: 1, scope: !102, type: !34)
// CHECK:STDOUT: !104 = !DILocalVariable(arg: 2, scope: !102, type: !34)
// CHECK:STDOUT: !111 = !DILocation(line: 20, column: 44, scope: !102)
// CHECK:STDOUT: !112 = !DILocation(line: 20, column: 37, scope: !102)
// CHECK:STDOUT: !113 = !{!103, !104}
// CHECK:STDOUT: !114 = distinct !DISubprogram(name: "TestComplement", linkageName: "_CTestComplement.Main", scope: null, file: !32, line: 23, type: !37, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !122)
// CHECK:STDOUT: !115 = !DILocalVariable(arg: 1, scope: !114, type: !34)
// CHECK:STDOUT: !120 = !DILocation(line: 23, column: 43, scope: !114)
// CHECK:STDOUT: !121 = !DILocation(line: 23, column: 36, scope: !114)
// CHECK:STDOUT: !122 = !{!115}
// CHECK:STDOUT: !123 = distinct !DISubprogram(name: "TestAnd", linkageName: "_CTestAnd.Main", scope: null, file: !32, line: 26, type: !39, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !134)
// CHECK:STDOUT: !124 = !DILocalVariable(arg: 1, scope: !123, type: !34)
// CHECK:STDOUT: !125 = !DILocalVariable(arg: 2, scope: !123, type: !34)
// CHECK:STDOUT: !132 = !DILocation(line: 26, column: 44, scope: !123)
// CHECK:STDOUT: !133 = !DILocation(line: 26, column: 37, scope: !123)
// CHECK:STDOUT: !134 = !{!124, !125}
// CHECK:STDOUT: !135 = distinct !DISubprogram(name: "TestOr", linkageName: "_CTestOr.Main", scope: null, file: !32, line: 29, type: !39, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !146)
// CHECK:STDOUT: !136 = !DILocalVariable(arg: 1, scope: !135, type: !34)
// CHECK:STDOUT: !137 = !DILocalVariable(arg: 2, scope: !135, type: !34)
// CHECK:STDOUT: !144 = !DILocation(line: 29, column: 43, scope: !135)
// CHECK:STDOUT: !145 = !DILocation(line: 29, column: 36, scope: !135)
// CHECK:STDOUT: !146 = !{!136, !137}
// CHECK:STDOUT: !147 = distinct !DISubprogram(name: "TestXor", linkageName: "_CTestXor.Main", scope: null, file: !32, line: 32, type: !39, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !158)
// CHECK:STDOUT: !148 = !DILocalVariable(arg: 1, scope: !147, type: !34)
// CHECK:STDOUT: !149 = !DILocalVariable(arg: 2, scope: !147, type: !34)
// CHECK:STDOUT: !156 = !DILocation(line: 32, column: 44, scope: !147)
// CHECK:STDOUT: !157 = !DILocation(line: 32, column: 37, scope: !147)
// CHECK:STDOUT: !158 = !{!148, !149}
// CHECK:STDOUT: !159 = distinct !DISubprogram(name: "TestLeftShift", linkageName: "_CTestLeftShift.Main", scope: null, file: !32, line: 35, type: !39, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !170)
// CHECK:STDOUT: !160 = !DILocalVariable(arg: 1, scope: !159, type: !34)
// CHECK:STDOUT: !161 = !DILocalVariable(arg: 2, scope: !159, type: !34)
// CHECK:STDOUT: !168 = !DILocation(line: 35, column: 50, scope: !159)
// CHECK:STDOUT: !169 = !DILocation(line: 35, column: 43, scope: !159)
// CHECK:STDOUT: !170 = !{!160, !161}
// CHECK:STDOUT: !171 = distinct !DISubprogram(name: "TestArithmeticRightShift", linkageName: "_CTestArithmeticRightShift.Main", scope: null, file: !32, line: 38, type: !39, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !182)
// CHECK:STDOUT: !172 = !DILocalVariable(arg: 1, scope: !171, type: !34)
// CHECK:STDOUT: !173 = !DILocalVariable(arg: 2, scope: !171, type: !34)
// CHECK:STDOUT: !180 = !DILocation(line: 38, column: 61, scope: !171)
// CHECK:STDOUT: !181 = !DILocation(line: 38, column: 54, scope: !171)
// CHECK:STDOUT: !182 = !{!172, !173}
// CHECK:STDOUT: !183 = distinct !DISubprogram(name: "TestLogicalRightShift", linkageName: "_CTestLogicalRightShift.Main", scope: null, file: !32, line: 41, type: !41, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !194)
// CHECK:STDOUT: !184 = !DILocalVariable(arg: 1, scope: !183, type: !35)
// CHECK:STDOUT: !185 = !DILocalVariable(arg: 2, scope: !183, type: !35)
// CHECK:STDOUT: !192 = !DILocation(line: 41, column: 58, scope: !183)
// CHECK:STDOUT: !193 = !DILocation(line: 41, column: 51, scope: !183)
// CHECK:STDOUT: !194 = !{!184, !185}
// CHECK:STDOUT: !195 = distinct !DISubprogram(name: "TestEq", linkageName: "_CTestEq.Main", scope: null, file: !32, line: 44, type: !44, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !206)
// CHECK:STDOUT: !196 = !DILocalVariable(arg: 1, scope: !195, type: !34)
// CHECK:STDOUT: !197 = !DILocalVariable(arg: 2, scope: !195, type: !34)
// CHECK:STDOUT: !204 = !DILocation(line: 44, column: 44, scope: !195)
// CHECK:STDOUT: !205 = !DILocation(line: 44, column: 37, scope: !195)
// CHECK:STDOUT: !206 = !{!196, !197}
// CHECK:STDOUT: !207 = distinct !DISubprogram(name: "TestNeq", linkageName: "_CTestNeq.Main", scope: null, file: !32, line: 47, type: !44, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !218)
// CHECK:STDOUT: !208 = !DILocalVariable(arg: 1, scope: !207, type: !34)
// CHECK:STDOUT: !209 = !DILocalVariable(arg: 2, scope: !207, type: !34)
// CHECK:STDOUT: !216 = !DILocation(line: 47, column: 45, scope: !207)
// CHECK:STDOUT: !217 = !DILocation(line: 47, column: 38, scope: !207)
// CHECK:STDOUT: !218 = !{!208, !209}
// CHECK:STDOUT: !219 = distinct !DISubprogram(name: "TestLess", linkageName: "_CTestLess.Main", scope: null, file: !32, line: 50, type: !44, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !230)
// CHECK:STDOUT: !220 = !DILocalVariable(arg: 1, scope: !219, type: !34)
// CHECK:STDOUT: !221 = !DILocalVariable(arg: 2, scope: !219, type: !34)
// CHECK:STDOUT: !228 = !DILocation(line: 50, column: 46, scope: !219)
// CHECK:STDOUT: !229 = !DILocation(line: 50, column: 39, scope: !219)
// CHECK:STDOUT: !230 = !{!220, !221}
// CHECK:STDOUT: !231 = distinct !DISubprogram(name: "TestLessEq", linkageName: "_CTestLessEq.Main", scope: null, file: !32, line: 53, type: !44, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !242)
// CHECK:STDOUT: !232 = !DILocalVariable(arg: 1, scope: !231, type: !34)
// CHECK:STDOUT: !233 = !DILocalVariable(arg: 2, scope: !231, type: !34)
// CHECK:STDOUT: !240 = !DILocation(line: 53, column: 48, scope: !231)
// CHECK:STDOUT: !241 = !DILocation(line: 53, column: 41, scope: !231)
// CHECK:STDOUT: !242 = !{!232, !233}
// CHECK:STDOUT: !243 = distinct !DISubprogram(name: "TestGreater", linkageName: "_CTestGreater.Main", scope: null, file: !32, line: 56, type: !44, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !254)
// CHECK:STDOUT: !244 = !DILocalVariable(arg: 1, scope: !243, type: !34)
// CHECK:STDOUT: !245 = !DILocalVariable(arg: 2, scope: !243, type: !34)
// CHECK:STDOUT: !252 = !DILocation(line: 56, column: 49, scope: !243)
// CHECK:STDOUT: !253 = !DILocation(line: 56, column: 42, scope: !243)
// CHECK:STDOUT: !254 = !{!244, !245}
// CHECK:STDOUT: !255 = distinct !DISubprogram(name: "TestGreaterEq", linkageName: "_CTestGreaterEq.Main", scope: null, file: !32, line: 59, type: !44, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !266)
// CHECK:STDOUT: !256 = !DILocalVariable(arg: 1, scope: !255, type: !34)
// CHECK:STDOUT: !257 = !DILocalVariable(arg: 2, scope: !255, type: !34)
// CHECK:STDOUT: !264 = !DILocation(line: 59, column: 51, scope: !255)
// CHECK:STDOUT: !265 = !DILocation(line: 59, column: 44, scope: !255)
// CHECK:STDOUT: !266 = !{!256, !257}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'compound_assign.carbon'
// CHECK:STDOUT: source_filename = "compound_assign.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestSAdd.Main(ptr %a, i32 %b) #0 !dbg !275 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %SAdd.call = load i32, ptr %a, align 4, !dbg !283
// CHECK:STDOUT:   %SAdd.call1 = add i32 %SAdd.call, %b, !dbg !283
// CHECK:STDOUT:   store i32 %SAdd.call1, ptr %a, align 4, !dbg !283
// CHECK:STDOUT:   ret void, !dbg !284
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestSSub.Main(ptr %a, i32 %b) #0 !dbg !286 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %SSub.call = load i32, ptr %a, align 4, !dbg !294
// CHECK:STDOUT:   %SSub.call1 = sub i32 %SSub.call, %b, !dbg !294
// CHECK:STDOUT:   store i32 %SSub.call1, ptr %a, align 4, !dbg !294
// CHECK:STDOUT:   ret void, !dbg !295
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestSMul.Main(ptr %a, i32 %b) #0 !dbg !297 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %SMul.call = load i32, ptr %a, align 4, !dbg !305
// CHECK:STDOUT:   %SMul.call1 = mul i32 %SMul.call, %b, !dbg !305
// CHECK:STDOUT:   store i32 %SMul.call1, ptr %a, align 4, !dbg !305
// CHECK:STDOUT:   ret void, !dbg !306
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestSDiv.Main(ptr %a, i32 %b) #0 !dbg !308 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %SDiv.call = load i32, ptr %a, align 4, !dbg !316
// CHECK:STDOUT:   %SDiv.call1 = sdiv i32 %SDiv.call, %b, !dbg !316
// CHECK:STDOUT:   store i32 %SDiv.call1, ptr %a, align 4, !dbg !316
// CHECK:STDOUT:   ret void, !dbg !317
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestSMod.Main(ptr %a, i32 %b) #0 !dbg !319 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %SMod.call = load i32, ptr %a, align 4, !dbg !327
// CHECK:STDOUT:   %SMod.call1 = srem i32 %SMod.call, %b, !dbg !327
// CHECK:STDOUT:   store i32 %SMod.call1, ptr %a, align 4, !dbg !327
// CHECK:STDOUT:   ret void, !dbg !328
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestUAdd.Main(ptr %a, i32 %b) #0 !dbg !330 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %UAdd.call = load i32, ptr %a, align 4, !dbg !338
// CHECK:STDOUT:   %UAdd.call1 = add i32 %UAdd.call, %b, !dbg !338
// CHECK:STDOUT:   store i32 %UAdd.call1, ptr %a, align 4, !dbg !338
// CHECK:STDOUT:   ret void, !dbg !339
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestUSub.Main(ptr %a, i32 %b) #0 !dbg !341 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %USub.call = load i32, ptr %a, align 4, !dbg !349
// CHECK:STDOUT:   %USub.call1 = sub i32 %USub.call, %b, !dbg !349
// CHECK:STDOUT:   store i32 %USub.call1, ptr %a, align 4, !dbg !349
// CHECK:STDOUT:   ret void, !dbg !350
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestUMul.Main(ptr %a, i32 %b) #0 !dbg !352 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %UMul.call = load i32, ptr %a, align 4, !dbg !360
// CHECK:STDOUT:   %UMul.call1 = mul i32 %UMul.call, %b, !dbg !360
// CHECK:STDOUT:   store i32 %UMul.call1, ptr %a, align 4, !dbg !360
// CHECK:STDOUT:   ret void, !dbg !361
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestUDiv.Main(ptr %a, i32 %b) #0 !dbg !363 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %UDiv.call = load i32, ptr %a, align 4, !dbg !371
// CHECK:STDOUT:   %UDiv.call1 = udiv i32 %UDiv.call, %b, !dbg !371
// CHECK:STDOUT:   store i32 %UDiv.call1, ptr %a, align 4, !dbg !371
// CHECK:STDOUT:   ret void, !dbg !372
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestUMod.Main(ptr %a, i32 %b) #0 !dbg !374 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %UMod.call = load i32, ptr %a, align 4, !dbg !382
// CHECK:STDOUT:   %UMod.call1 = urem i32 %UMod.call, %b, !dbg !382
// CHECK:STDOUT:   store i32 %UMod.call1, ptr %a, align 4, !dbg !382
// CHECK:STDOUT:   ret void, !dbg !383
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestAnd.Main(ptr %a, i32 %b) #0 !dbg !385 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %And.call = load i32, ptr %a, align 4, !dbg !393
// CHECK:STDOUT:   %And.call1 = and i32 %And.call, %b, !dbg !393
// CHECK:STDOUT:   store i32 %And.call1, ptr %a, align 4, !dbg !393
// CHECK:STDOUT:   ret void, !dbg !394
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestOr.Main(ptr %a, i32 %b) #0 !dbg !396 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Or.call = load i32, ptr %a, align 4, !dbg !404
// CHECK:STDOUT:   %Or.call1 = or i32 %Or.call, %b, !dbg !404
// CHECK:STDOUT:   store i32 %Or.call1, ptr %a, align 4, !dbg !404
// CHECK:STDOUT:   ret void, !dbg !405
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestXor.Main(ptr %a, i32 %b) #0 !dbg !407 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Xor.call = load i32, ptr %a, align 4, !dbg !415
// CHECK:STDOUT:   %Xor.call1 = xor i32 %Xor.call, %b, !dbg !415
// CHECK:STDOUT:   store i32 %Xor.call1, ptr %a, align 4, !dbg !415
// CHECK:STDOUT:   ret void, !dbg !416
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestLeftShift.Main(ptr %a, i32 %b) #0 !dbg !418 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %LeftShift.call = load i32, ptr %a, align 4, !dbg !426
// CHECK:STDOUT:   %LeftShift.call1 = shl i32 %LeftShift.call, %b, !dbg !426
// CHECK:STDOUT:   store i32 %LeftShift.call1, ptr %a, align 4, !dbg !426
// CHECK:STDOUT:   ret void, !dbg !427
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestArithmeticRightShift.Main(ptr %a, i32 %b) #0 !dbg !429 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ArithmeticRightShift.call = load i32, ptr %a, align 4, !dbg !437
// CHECK:STDOUT:   %ArithmeticRightShift.call1 = ashr i32 %ArithmeticRightShift.call, %b, !dbg !437
// CHECK:STDOUT:   store i32 %ArithmeticRightShift.call1, ptr %a, align 4, !dbg !437
// CHECK:STDOUT:   ret void, !dbg !438
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestLogicalRightShift.Main(ptr %a, i32 %b) #0 !dbg !440 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %LogicalRightShift.call = load i32, ptr %a, align 4, !dbg !448
// CHECK:STDOUT:   %LogicalRightShift.call1 = lshr i32 %LogicalRightShift.call, %b, !dbg !448
// CHECK:STDOUT:   store i32 %LogicalRightShift.call1, ptr %a, align 4, !dbg !448
// CHECK:STDOUT:   ret void, !dbg !449
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!268}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !34 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !35 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !267 = !DIFile(filename: "compound_assign.carbon", directory: "")
// CHECK:STDOUT: !268 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !267, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !269 = !{null, !34, !34}
// CHECK:STDOUT: !270 = !DISubroutineType(types: !269)
// CHECK:STDOUT: !271 = !{null, !34, !35}
// CHECK:STDOUT: !272 = !DISubroutineType(types: !271)
// CHECK:STDOUT: !273 = !{null, !35, !35}
// CHECK:STDOUT: !274 = !DISubroutineType(types: !273)
// CHECK:STDOUT: !275 = distinct !DISubprogram(name: "TestSAdd", linkageName: "_CTestSAdd.Main", scope: null, file: !267, line: 5, type: !270, spFlags: DISPFlagDefinition, unit: !268, retainedNodes: !285)
// CHECK:STDOUT: !276 = !DILocalVariable(arg: 1, scope: !275, type: !34)
// CHECK:STDOUT: !277 = !DILocalVariable(arg: 2, scope: !275, type: !34)
// CHECK:STDOUT: !283 = !DILocation(line: 5, column: 35, scope: !275)
// CHECK:STDOUT: !284 = !DILocation(line: 5, column: 1, scope: !275)
// CHECK:STDOUT: !285 = !{!276, !277}
// CHECK:STDOUT: !286 = distinct !DISubprogram(name: "TestSSub", linkageName: "_CTestSSub.Main", scope: null, file: !267, line: 8, type: !270, spFlags: DISPFlagDefinition, unit: !268, retainedNodes: !296)
// CHECK:STDOUT: !287 = !DILocalVariable(arg: 1, scope: !286, type: !34)
// CHECK:STDOUT: !288 = !DILocalVariable(arg: 2, scope: !286, type: !34)
// CHECK:STDOUT: !294 = !DILocation(line: 8, column: 35, scope: !286)
// CHECK:STDOUT: !295 = !DILocation(line: 8, column: 1, scope: !286)
// CHECK:STDOUT: !296 = !{!287, !288}
// CHECK:STDOUT: !297 = distinct !DISubprogram(name: "TestSMul", linkageName: "_CTestSMul.Main", scope: null, file: !267, line: 11, type: !270, spFlags: DISPFlagDefinition, unit: !268, retainedNodes: !307)
// CHECK:STDOUT: !298 = !DILocalVariable(arg: 1, scope: !297, type: !34)
// CHECK:STDOUT: !299 = !DILocalVariable(arg: 2, scope: !297, type: !34)
// CHECK:STDOUT: !305 = !DILocation(line: 11, column: 35, scope: !297)
// CHECK:STDOUT: !306 = !DILocation(line: 11, column: 1, scope: !297)
// CHECK:STDOUT: !307 = !{!298, !299}
// CHECK:STDOUT: !308 = distinct !DISubprogram(name: "TestSDiv", linkageName: "_CTestSDiv.Main", scope: null, file: !267, line: 14, type: !270, spFlags: DISPFlagDefinition, unit: !268, retainedNodes: !318)
// CHECK:STDOUT: !309 = !DILocalVariable(arg: 1, scope: !308, type: !34)
// CHECK:STDOUT: !310 = !DILocalVariable(arg: 2, scope: !308, type: !34)
// CHECK:STDOUT: !316 = !DILocation(line: 14, column: 35, scope: !308)
// CHECK:STDOUT: !317 = !DILocation(line: 14, column: 1, scope: !308)
// CHECK:STDOUT: !318 = !{!309, !310}
// CHECK:STDOUT: !319 = distinct !DISubprogram(name: "TestSMod", linkageName: "_CTestSMod.Main", scope: null, file: !267, line: 17, type: !270, spFlags: DISPFlagDefinition, unit: !268, retainedNodes: !329)
// CHECK:STDOUT: !320 = !DILocalVariable(arg: 1, scope: !319, type: !34)
// CHECK:STDOUT: !321 = !DILocalVariable(arg: 2, scope: !319, type: !34)
// CHECK:STDOUT: !327 = !DILocation(line: 17, column: 35, scope: !319)
// CHECK:STDOUT: !328 = !DILocation(line: 17, column: 1, scope: !319)
// CHECK:STDOUT: !329 = !{!320, !321}
// CHECK:STDOUT: !330 = distinct !DISubprogram(name: "TestUAdd", linkageName: "_CTestUAdd.Main", scope: null, file: !267, line: 20, type: !270, spFlags: DISPFlagDefinition, unit: !268, retainedNodes: !340)
// CHECK:STDOUT: !331 = !DILocalVariable(arg: 1, scope: !330, type: !34)
// CHECK:STDOUT: !332 = !DILocalVariable(arg: 2, scope: !330, type: !34)
// CHECK:STDOUT: !338 = !DILocation(line: 20, column: 35, scope: !330)
// CHECK:STDOUT: !339 = !DILocation(line: 20, column: 1, scope: !330)
// CHECK:STDOUT: !340 = !{!331, !332}
// CHECK:STDOUT: !341 = distinct !DISubprogram(name: "TestUSub", linkageName: "_CTestUSub.Main", scope: null, file: !267, line: 23, type: !270, spFlags: DISPFlagDefinition, unit: !268, retainedNodes: !351)
// CHECK:STDOUT: !342 = !DILocalVariable(arg: 1, scope: !341, type: !34)
// CHECK:STDOUT: !343 = !DILocalVariable(arg: 2, scope: !341, type: !34)
// CHECK:STDOUT: !349 = !DILocation(line: 23, column: 35, scope: !341)
// CHECK:STDOUT: !350 = !DILocation(line: 23, column: 1, scope: !341)
// CHECK:STDOUT: !351 = !{!342, !343}
// CHECK:STDOUT: !352 = distinct !DISubprogram(name: "TestUMul", linkageName: "_CTestUMul.Main", scope: null, file: !267, line: 26, type: !270, spFlags: DISPFlagDefinition, unit: !268, retainedNodes: !362)
// CHECK:STDOUT: !353 = !DILocalVariable(arg: 1, scope: !352, type: !34)
// CHECK:STDOUT: !354 = !DILocalVariable(arg: 2, scope: !352, type: !34)
// CHECK:STDOUT: !360 = !DILocation(line: 26, column: 35, scope: !352)
// CHECK:STDOUT: !361 = !DILocation(line: 26, column: 1, scope: !352)
// CHECK:STDOUT: !362 = !{!353, !354}
// CHECK:STDOUT: !363 = distinct !DISubprogram(name: "TestUDiv", linkageName: "_CTestUDiv.Main", scope: null, file: !267, line: 29, type: !270, spFlags: DISPFlagDefinition, unit: !268, retainedNodes: !373)
// CHECK:STDOUT: !364 = !DILocalVariable(arg: 1, scope: !363, type: !34)
// CHECK:STDOUT: !365 = !DILocalVariable(arg: 2, scope: !363, type: !34)
// CHECK:STDOUT: !371 = !DILocation(line: 29, column: 35, scope: !363)
// CHECK:STDOUT: !372 = !DILocation(line: 29, column: 1, scope: !363)
// CHECK:STDOUT: !373 = !{!364, !365}
// CHECK:STDOUT: !374 = distinct !DISubprogram(name: "TestUMod", linkageName: "_CTestUMod.Main", scope: null, file: !267, line: 32, type: !270, spFlags: DISPFlagDefinition, unit: !268, retainedNodes: !384)
// CHECK:STDOUT: !375 = !DILocalVariable(arg: 1, scope: !374, type: !34)
// CHECK:STDOUT: !376 = !DILocalVariable(arg: 2, scope: !374, type: !34)
// CHECK:STDOUT: !382 = !DILocation(line: 32, column: 35, scope: !374)
// CHECK:STDOUT: !383 = !DILocation(line: 32, column: 1, scope: !374)
// CHECK:STDOUT: !384 = !{!375, !376}
// CHECK:STDOUT: !385 = distinct !DISubprogram(name: "TestAnd", linkageName: "_CTestAnd.Main", scope: null, file: !267, line: 35, type: !270, spFlags: DISPFlagDefinition, unit: !268, retainedNodes: !395)
// CHECK:STDOUT: !386 = !DILocalVariable(arg: 1, scope: !385, type: !34)
// CHECK:STDOUT: !387 = !DILocalVariable(arg: 2, scope: !385, type: !34)
// CHECK:STDOUT: !393 = !DILocation(line: 35, column: 34, scope: !385)
// CHECK:STDOUT: !394 = !DILocation(line: 35, column: 1, scope: !385)
// CHECK:STDOUT: !395 = !{!386, !387}
// CHECK:STDOUT: !396 = distinct !DISubprogram(name: "TestOr", linkageName: "_CTestOr.Main", scope: null, file: !267, line: 38, type: !270, spFlags: DISPFlagDefinition, unit: !268, retainedNodes: !406)
// CHECK:STDOUT: !397 = !DILocalVariable(arg: 1, scope: !396, type: !34)
// CHECK:STDOUT: !398 = !DILocalVariable(arg: 2, scope: !396, type: !34)
// CHECK:STDOUT: !404 = !DILocation(line: 38, column: 33, scope: !396)
// CHECK:STDOUT: !405 = !DILocation(line: 38, column: 1, scope: !396)
// CHECK:STDOUT: !406 = !{!397, !398}
// CHECK:STDOUT: !407 = distinct !DISubprogram(name: "TestXor", linkageName: "_CTestXor.Main", scope: null, file: !267, line: 41, type: !270, spFlags: DISPFlagDefinition, unit: !268, retainedNodes: !417)
// CHECK:STDOUT: !408 = !DILocalVariable(arg: 1, scope: !407, type: !34)
// CHECK:STDOUT: !409 = !DILocalVariable(arg: 2, scope: !407, type: !34)
// CHECK:STDOUT: !415 = !DILocation(line: 41, column: 34, scope: !407)
// CHECK:STDOUT: !416 = !DILocation(line: 41, column: 1, scope: !407)
// CHECK:STDOUT: !417 = !{!408, !409}
// CHECK:STDOUT: !418 = distinct !DISubprogram(name: "TestLeftShift", linkageName: "_CTestLeftShift.Main", scope: null, file: !267, line: 44, type: !270, spFlags: DISPFlagDefinition, unit: !268, retainedNodes: !428)
// CHECK:STDOUT: !419 = !DILocalVariable(arg: 1, scope: !418, type: !34)
// CHECK:STDOUT: !420 = !DILocalVariable(arg: 2, scope: !418, type: !34)
// CHECK:STDOUT: !426 = !DILocation(line: 44, column: 40, scope: !418)
// CHECK:STDOUT: !427 = !DILocation(line: 44, column: 1, scope: !418)
// CHECK:STDOUT: !428 = !{!419, !420}
// CHECK:STDOUT: !429 = distinct !DISubprogram(name: "TestArithmeticRightShift", linkageName: "_CTestArithmeticRightShift.Main", scope: null, file: !267, line: 47, type: !272, spFlags: DISPFlagDefinition, unit: !268, retainedNodes: !439)
// CHECK:STDOUT: !430 = !DILocalVariable(arg: 1, scope: !429, type: !34)
// CHECK:STDOUT: !431 = !DILocalVariable(arg: 2, scope: !429, type: !35)
// CHECK:STDOUT: !437 = !DILocation(line: 47, column: 51, scope: !429)
// CHECK:STDOUT: !438 = !DILocation(line: 47, column: 1, scope: !429)
// CHECK:STDOUT: !439 = !{!430, !431}
// CHECK:STDOUT: !440 = distinct !DISubprogram(name: "TestLogicalRightShift", linkageName: "_CTestLogicalRightShift.Main", scope: null, file: !267, line: 50, type: !274, spFlags: DISPFlagDefinition, unit: !268, retainedNodes: !450)
// CHECK:STDOUT: !441 = !DILocalVariable(arg: 1, scope: !440, type: !35)
// CHECK:STDOUT: !442 = !DILocalVariable(arg: 2, scope: !440, type: !35)
// CHECK:STDOUT: !448 = !DILocation(line: 50, column: 48, scope: !440)
// CHECK:STDOUT: !449 = !DILocation(line: 50, column: 1, scope: !440)
// CHECK:STDOUT: !450 = !{!441, !442}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'mixed_shift.carbon'
// CHECK:STDOUT: source_filename = "mixed_shift.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestLeftShiftSmaller.Main(i32 %a, i16 %b) #0 !dbg !465 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %LeftShiftSmaller.call.rhs = zext i16 %b to i32, !dbg !474
// CHECK:STDOUT:   %LeftShiftSmaller.call = shl i32 %a, %LeftShiftSmaller.call.rhs, !dbg !474
// CHECK:STDOUT:   ret i32 %LeftShiftSmaller.call, !dbg !475
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestRightShiftSmaller.Main(i32 %a, i16 %b) #0 !dbg !477 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %RightShiftSmaller.call.rhs = zext i16 %b to i32, !dbg !486
// CHECK:STDOUT:   %RightShiftSmaller.call = ashr i32 %a, %RightShiftSmaller.call.rhs, !dbg !486
// CHECK:STDOUT:   ret i32 %RightShiftSmaller.call, !dbg !487
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i16 @_CTestLeftShiftLargerII.Main(i16 %a, i32 %b) #0 !dbg !489 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %LeftShiftLargerII.call.rhs = trunc i32 %b to i16, !dbg !498
// CHECK:STDOUT:   %LeftShiftLargerII.call = shl i16 %a, %LeftShiftLargerII.call.rhs, !dbg !498
// CHECK:STDOUT:   ret i16 %LeftShiftLargerII.call, !dbg !499
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i16 @_CTestRightShiftLargerII.Main(i16 %a, i32 %b) #0 !dbg !501 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %RightShiftLargerII.call.rhs = trunc i32 %b to i16, !dbg !510
// CHECK:STDOUT:   %RightShiftLargerII.call = ashr i16 %a, %RightShiftLargerII.call.rhs, !dbg !510
// CHECK:STDOUT:   ret i16 %RightShiftLargerII.call, !dbg !511
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i16 @_CTestLeftShiftLargerIU.Main(i16 %a, i32 %b) #0 !dbg !513 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %LeftShiftLargerIU.call.rhs = trunc i32 %b to i16, !dbg !522
// CHECK:STDOUT:   %LeftShiftLargerIU.call = shl i16 %a, %LeftShiftLargerIU.call.rhs, !dbg !522
// CHECK:STDOUT:   ret i16 %LeftShiftLargerIU.call, !dbg !523
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i16 @_CTestRightShiftLargerIU.Main(i16 %a, i32 %b) #0 !dbg !525 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %RightShiftLargerIU.call.rhs = trunc i32 %b to i16, !dbg !534
// CHECK:STDOUT:   %RightShiftLargerIU.call = ashr i16 %a, %RightShiftLargerIU.call.rhs, !dbg !534
// CHECK:STDOUT:   ret i16 %RightShiftLargerIU.call, !dbg !535
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i16 @_CTestLeftShiftLargerUI.Main(i16 %a, i32 %b) #0 !dbg !537 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %LeftShiftLargerUI.call.rhs = trunc i32 %b to i16, !dbg !546
// CHECK:STDOUT:   %LeftShiftLargerUI.call = shl i16 %a, %LeftShiftLargerUI.call.rhs, !dbg !546
// CHECK:STDOUT:   ret i16 %LeftShiftLargerUI.call, !dbg !547
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i16 @_CTestRightShiftLargerUI.Main(i16 %a, i32 %b) #0 !dbg !549 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %RightShiftLargerUI.call.rhs = trunc i32 %b to i16, !dbg !558
// CHECK:STDOUT:   %RightShiftLargerUI.call = lshr i16 %a, %RightShiftLargerUI.call.rhs, !dbg !558
// CHECK:STDOUT:   ret i16 %RightShiftLargerUI.call, !dbg !559
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i16 @_CTestLeftShiftLargerUU.Main(i16 %a, i32 %b) #0 !dbg !561 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %LeftShiftLargerUU.call.rhs = trunc i32 %b to i16, !dbg !570
// CHECK:STDOUT:   %LeftShiftLargerUU.call = shl i16 %a, %LeftShiftLargerUU.call.rhs, !dbg !570
// CHECK:STDOUT:   ret i16 %LeftShiftLargerUU.call, !dbg !571
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i16 @_CTestRightShiftLargerUU.Main(i16 %a, i32 %b) #0 !dbg !573 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %RightShiftLargerUU.call.rhs = trunc i32 %b to i16, !dbg !582
// CHECK:STDOUT:   %RightShiftLargerUU.call = lshr i16 %a, %RightShiftLargerUU.call.rhs, !dbg !582
// CHECK:STDOUT:   ret i16 %RightShiftLargerUU.call, !dbg !583
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!452}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !34 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !35 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !451 = !DIFile(filename: "mixed_shift.carbon", directory: "")
// CHECK:STDOUT: !452 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !451, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !453 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !454 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !455 = !{!34, !34, !453}
// CHECK:STDOUT: !456 = !DISubroutineType(types: !455)
// CHECK:STDOUT: !457 = !{!453, !453, !34}
// CHECK:STDOUT: !458 = !DISubroutineType(types: !457)
// CHECK:STDOUT: !459 = !{!453, !453, !35}
// CHECK:STDOUT: !460 = !DISubroutineType(types: !459)
// CHECK:STDOUT: !461 = !{!454, !454, !34}
// CHECK:STDOUT: !462 = !DISubroutineType(types: !461)
// CHECK:STDOUT: !463 = !{!454, !454, !35}
// CHECK:STDOUT: !464 = !DISubroutineType(types: !463)
// CHECK:STDOUT: !465 = distinct !DISubprogram(name: "TestLeftShiftSmaller", linkageName: "_CTestLeftShiftSmaller.Main", scope: null, file: !451, line: 5, type: !456, spFlags: DISPFlagDefinition, unit: !452, retainedNodes: !476)
// CHECK:STDOUT: !466 = !DILocalVariable(arg: 1, scope: !465, type: !34)
// CHECK:STDOUT: !467 = !DILocalVariable(arg: 2, scope: !465, type: !453)
// CHECK:STDOUT: !474 = !DILocation(line: 5, column: 57, scope: !465)
// CHECK:STDOUT: !475 = !DILocation(line: 5, column: 50, scope: !465)
// CHECK:STDOUT: !476 = !{!466, !467}
// CHECK:STDOUT: !477 = distinct !DISubprogram(name: "TestRightShiftSmaller", linkageName: "_CTestRightShiftSmaller.Main", scope: null, file: !451, line: 8, type: !456, spFlags: DISPFlagDefinition, unit: !452, retainedNodes: !488)
// CHECK:STDOUT: !478 = !DILocalVariable(arg: 1, scope: !477, type: !34)
// CHECK:STDOUT: !479 = !DILocalVariable(arg: 2, scope: !477, type: !453)
// CHECK:STDOUT: !486 = !DILocation(line: 8, column: 58, scope: !477)
// CHECK:STDOUT: !487 = !DILocation(line: 8, column: 51, scope: !477)
// CHECK:STDOUT: !488 = !{!478, !479}
// CHECK:STDOUT: !489 = distinct !DISubprogram(name: "TestLeftShiftLargerII", linkageName: "_CTestLeftShiftLargerII.Main", scope: null, file: !451, line: 11, type: !458, spFlags: DISPFlagDefinition, unit: !452, retainedNodes: !500)
// CHECK:STDOUT: !490 = !DILocalVariable(arg: 1, scope: !489, type: !453)
// CHECK:STDOUT: !491 = !DILocalVariable(arg: 2, scope: !489, type: !34)
// CHECK:STDOUT: !498 = !DILocation(line: 11, column: 58, scope: !489)
// CHECK:STDOUT: !499 = !DILocation(line: 11, column: 51, scope: !489)
// CHECK:STDOUT: !500 = !{!490, !491}
// CHECK:STDOUT: !501 = distinct !DISubprogram(name: "TestRightShiftLargerII", linkageName: "_CTestRightShiftLargerII.Main", scope: null, file: !451, line: 14, type: !458, spFlags: DISPFlagDefinition, unit: !452, retainedNodes: !512)
// CHECK:STDOUT: !502 = !DILocalVariable(arg: 1, scope: !501, type: !453)
// CHECK:STDOUT: !503 = !DILocalVariable(arg: 2, scope: !501, type: !34)
// CHECK:STDOUT: !510 = !DILocation(line: 14, column: 59, scope: !501)
// CHECK:STDOUT: !511 = !DILocation(line: 14, column: 52, scope: !501)
// CHECK:STDOUT: !512 = !{!502, !503}
// CHECK:STDOUT: !513 = distinct !DISubprogram(name: "TestLeftShiftLargerIU", linkageName: "_CTestLeftShiftLargerIU.Main", scope: null, file: !451, line: 17, type: !460, spFlags: DISPFlagDefinition, unit: !452, retainedNodes: !524)
// CHECK:STDOUT: !514 = !DILocalVariable(arg: 1, scope: !513, type: !453)
// CHECK:STDOUT: !515 = !DILocalVariable(arg: 2, scope: !513, type: !35)
// CHECK:STDOUT: !522 = !DILocation(line: 17, column: 58, scope: !513)
// CHECK:STDOUT: !523 = !DILocation(line: 17, column: 51, scope: !513)
// CHECK:STDOUT: !524 = !{!514, !515}
// CHECK:STDOUT: !525 = distinct !DISubprogram(name: "TestRightShiftLargerIU", linkageName: "_CTestRightShiftLargerIU.Main", scope: null, file: !451, line: 20, type: !460, spFlags: DISPFlagDefinition, unit: !452, retainedNodes: !536)
// CHECK:STDOUT: !526 = !DILocalVariable(arg: 1, scope: !525, type: !453)
// CHECK:STDOUT: !527 = !DILocalVariable(arg: 2, scope: !525, type: !35)
// CHECK:STDOUT: !534 = !DILocation(line: 20, column: 59, scope: !525)
// CHECK:STDOUT: !535 = !DILocation(line: 20, column: 52, scope: !525)
// CHECK:STDOUT: !536 = !{!526, !527}
// CHECK:STDOUT: !537 = distinct !DISubprogram(name: "TestLeftShiftLargerUI", linkageName: "_CTestLeftShiftLargerUI.Main", scope: null, file: !451, line: 23, type: !462, spFlags: DISPFlagDefinition, unit: !452, retainedNodes: !548)
// CHECK:STDOUT: !538 = !DILocalVariable(arg: 1, scope: !537, type: !454)
// CHECK:STDOUT: !539 = !DILocalVariable(arg: 2, scope: !537, type: !34)
// CHECK:STDOUT: !546 = !DILocation(line: 23, column: 58, scope: !537)
// CHECK:STDOUT: !547 = !DILocation(line: 23, column: 51, scope: !537)
// CHECK:STDOUT: !548 = !{!538, !539}
// CHECK:STDOUT: !549 = distinct !DISubprogram(name: "TestRightShiftLargerUI", linkageName: "_CTestRightShiftLargerUI.Main", scope: null, file: !451, line: 26, type: !462, spFlags: DISPFlagDefinition, unit: !452, retainedNodes: !560)
// CHECK:STDOUT: !550 = !DILocalVariable(arg: 1, scope: !549, type: !454)
// CHECK:STDOUT: !551 = !DILocalVariable(arg: 2, scope: !549, type: !34)
// CHECK:STDOUT: !558 = !DILocation(line: 26, column: 59, scope: !549)
// CHECK:STDOUT: !559 = !DILocation(line: 26, column: 52, scope: !549)
// CHECK:STDOUT: !560 = !{!550, !551}
// CHECK:STDOUT: !561 = distinct !DISubprogram(name: "TestLeftShiftLargerUU", linkageName: "_CTestLeftShiftLargerUU.Main", scope: null, file: !451, line: 29, type: !464, spFlags: DISPFlagDefinition, unit: !452, retainedNodes: !572)
// CHECK:STDOUT: !562 = !DILocalVariable(arg: 1, scope: !561, type: !454)
// CHECK:STDOUT: !563 = !DILocalVariable(arg: 2, scope: !561, type: !35)
// CHECK:STDOUT: !570 = !DILocation(line: 29, column: 58, scope: !561)
// CHECK:STDOUT: !571 = !DILocation(line: 29, column: 51, scope: !561)
// CHECK:STDOUT: !572 = !{!562, !563}
// CHECK:STDOUT: !573 = distinct !DISubprogram(name: "TestRightShiftLargerUU", linkageName: "_CTestRightShiftLargerUU.Main", scope: null, file: !451, line: 32, type: !464, spFlags: DISPFlagDefinition, unit: !452, retainedNodes: !584)
// CHECK:STDOUT: !574 = !DILocalVariable(arg: 1, scope: !573, type: !454)
// CHECK:STDOUT: !575 = !DILocalVariable(arg: 2, scope: !573, type: !35)
// CHECK:STDOUT: !582 = !DILocation(line: 32, column: 59, scope: !573)
// CHECK:STDOUT: !583 = !DILocation(line: 32, column: 52, scope: !573)
// CHECK:STDOUT: !584 = !{!574, !575}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'mixed_compare.carbon'
// CHECK:STDOUT: source_filename = "mixed_compare.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestEq_u16_u32.Main(i16 %a, i32 %b) #0 !dbg !597 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Eq_u16_u32.call.lhs = zext i16 %a to i32, !dbg !606
// CHECK:STDOUT:   %Eq_u16_u32.call = icmp eq i32 %Eq_u16_u32.call.lhs, %b, !dbg !606
// CHECK:STDOUT:   ret i1 %Eq_u16_u32.call, !dbg !607
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestEq_i16_u32.Main(i16 %a, i32 %b) #0 !dbg !609 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Eq_i16_u32.call.lhs = sext i16 %a to i33, !dbg !618
// CHECK:STDOUT:   %Eq_i16_u32.call.rhs = zext i32 %b to i33, !dbg !618
// CHECK:STDOUT:   %Eq_i16_u32.call = icmp eq i33 %Eq_i16_u32.call.lhs, %Eq_i16_u32.call.rhs, !dbg !618
// CHECK:STDOUT:   ret i1 %Eq_i16_u32.call, !dbg !619
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestEq_u16_i32.Main(i16 %a, i32 %b) #0 !dbg !621 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Eq_u16_i32.call.lhs = zext i16 %a to i32, !dbg !630
// CHECK:STDOUT:   %Eq_u16_i32.call = icmp eq i32 %Eq_u16_i32.call.lhs, %b, !dbg !630
// CHECK:STDOUT:   ret i1 %Eq_u16_i32.call, !dbg !631
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestEq_i16_i32.Main(i16 %a, i32 %b) #0 !dbg !633 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Eq_i16_i32.call.lhs = sext i16 %a to i32, !dbg !642
// CHECK:STDOUT:   %Eq_i16_i32.call = icmp eq i32 %Eq_i16_i32.call.lhs, %b, !dbg !642
// CHECK:STDOUT:   ret i1 %Eq_i16_i32.call, !dbg !643
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestEq_i32_u32.Main(i32 %a, i32 %b) #0 !dbg !645 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Eq_i32_u32.call.lhs = sext i32 %a to i33, !dbg !654
// CHECK:STDOUT:   %Eq_i32_u32.call.rhs = zext i32 %b to i33, !dbg !654
// CHECK:STDOUT:   %Eq_i32_u32.call = icmp eq i33 %Eq_i32_u32.call.lhs, %Eq_i32_u32.call.rhs, !dbg !654
// CHECK:STDOUT:   ret i1 %Eq_i32_u32.call, !dbg !655
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestLess_u16_u32.Main(i16 %a, i32 %b) #0 !dbg !657 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Less_u16_u32.call.lhs = zext i16 %a to i32, !dbg !666
// CHECK:STDOUT:   %Less_u16_u32.call = icmp ult i32 %Less_u16_u32.call.lhs, %b, !dbg !666
// CHECK:STDOUT:   ret i1 %Less_u16_u32.call, !dbg !667
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestLess_i16_u32.Main(i16 %a, i32 %b) #0 !dbg !669 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Less_i16_u32.call.lhs = sext i16 %a to i33, !dbg !678
// CHECK:STDOUT:   %Less_i16_u32.call.rhs = zext i32 %b to i33, !dbg !678
// CHECK:STDOUT:   %Less_i16_u32.call = icmp slt i33 %Less_i16_u32.call.lhs, %Less_i16_u32.call.rhs, !dbg !678
// CHECK:STDOUT:   ret i1 %Less_i16_u32.call, !dbg !679
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestLess_u16_i32.Main(i16 %a, i32 %b) #0 !dbg !681 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Less_u16_i32.call.lhs = zext i16 %a to i32, !dbg !690
// CHECK:STDOUT:   %Less_u16_i32.call = icmp slt i32 %Less_u16_i32.call.lhs, %b, !dbg !690
// CHECK:STDOUT:   ret i1 %Less_u16_i32.call, !dbg !691
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestLess_i16_i32.Main(i16 %a, i32 %b) #0 !dbg !693 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Less_i16_i32.call.lhs = sext i16 %a to i32, !dbg !702
// CHECK:STDOUT:   %Less_i16_i32.call = icmp slt i32 %Less_i16_i32.call.lhs, %b, !dbg !702
// CHECK:STDOUT:   ret i1 %Less_i16_i32.call, !dbg !703
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestLess_i32_u32.Main(i32 %a, i32 %b) #0 !dbg !705 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Less_i32_u32.call.lhs = sext i32 %a to i33, !dbg !714
// CHECK:STDOUT:   %Less_i32_u32.call.rhs = zext i32 %b to i33, !dbg !714
// CHECK:STDOUT:   %Less_i32_u32.call = icmp slt i33 %Less_i32_u32.call.lhs, %Less_i32_u32.call.rhs, !dbg !714
// CHECK:STDOUT:   ret i1 %Less_i32_u32.call, !dbg !715
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!586}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !34 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !35 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !42 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !453 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !454 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !585 = !DIFile(filename: "mixed_compare.carbon", directory: "")
// CHECK:STDOUT: !586 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !585, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !587 = !{!42, !454, !35}
// CHECK:STDOUT: !588 = !DISubroutineType(types: !587)
// CHECK:STDOUT: !589 = !{!42, !453, !35}
// CHECK:STDOUT: !590 = !DISubroutineType(types: !589)
// CHECK:STDOUT: !591 = !{!42, !454, !34}
// CHECK:STDOUT: !592 = !DISubroutineType(types: !591)
// CHECK:STDOUT: !593 = !{!42, !453, !34}
// CHECK:STDOUT: !594 = !DISubroutineType(types: !593)
// CHECK:STDOUT: !595 = !{!42, !34, !35}
// CHECK:STDOUT: !596 = !DISubroutineType(types: !595)
// CHECK:STDOUT: !597 = distinct !DISubprogram(name: "TestEq_u16_u32", linkageName: "_CTestEq_u16_u32.Main", scope: null, file: !585, line: 10, type: !588, spFlags: DISPFlagDefinition, unit: !586, retainedNodes: !608)
// CHECK:STDOUT: !598 = !DILocalVariable(arg: 1, scope: !597, type: !454)
// CHECK:STDOUT: !599 = !DILocalVariable(arg: 2, scope: !597, type: !35)
// CHECK:STDOUT: !606 = !DILocation(line: 10, column: 52, scope: !597)
// CHECK:STDOUT: !607 = !DILocation(line: 10, column: 45, scope: !597)
// CHECK:STDOUT: !608 = !{!598, !599}
// CHECK:STDOUT: !609 = distinct !DISubprogram(name: "TestEq_i16_u32", linkageName: "_CTestEq_i16_u32.Main", scope: null, file: !585, line: 11, type: !590, spFlags: DISPFlagDefinition, unit: !586, retainedNodes: !620)
// CHECK:STDOUT: !610 = !DILocalVariable(arg: 1, scope: !609, type: !453)
// CHECK:STDOUT: !611 = !DILocalVariable(arg: 2, scope: !609, type: !35)
// CHECK:STDOUT: !618 = !DILocation(line: 11, column: 52, scope: !609)
// CHECK:STDOUT: !619 = !DILocation(line: 11, column: 45, scope: !609)
// CHECK:STDOUT: !620 = !{!610, !611}
// CHECK:STDOUT: !621 = distinct !DISubprogram(name: "TestEq_u16_i32", linkageName: "_CTestEq_u16_i32.Main", scope: null, file: !585, line: 12, type: !592, spFlags: DISPFlagDefinition, unit: !586, retainedNodes: !632)
// CHECK:STDOUT: !622 = !DILocalVariable(arg: 1, scope: !621, type: !454)
// CHECK:STDOUT: !623 = !DILocalVariable(arg: 2, scope: !621, type: !34)
// CHECK:STDOUT: !630 = !DILocation(line: 12, column: 52, scope: !621)
// CHECK:STDOUT: !631 = !DILocation(line: 12, column: 45, scope: !621)
// CHECK:STDOUT: !632 = !{!622, !623}
// CHECK:STDOUT: !633 = distinct !DISubprogram(name: "TestEq_i16_i32", linkageName: "_CTestEq_i16_i32.Main", scope: null, file: !585, line: 13, type: !594, spFlags: DISPFlagDefinition, unit: !586, retainedNodes: !644)
// CHECK:STDOUT: !634 = !DILocalVariable(arg: 1, scope: !633, type: !453)
// CHECK:STDOUT: !635 = !DILocalVariable(arg: 2, scope: !633, type: !34)
// CHECK:STDOUT: !642 = !DILocation(line: 13, column: 52, scope: !633)
// CHECK:STDOUT: !643 = !DILocation(line: 13, column: 45, scope: !633)
// CHECK:STDOUT: !644 = !{!634, !635}
// CHECK:STDOUT: !645 = distinct !DISubprogram(name: "TestEq_i32_u32", linkageName: "_CTestEq_i32_u32.Main", scope: null, file: !585, line: 14, type: !596, spFlags: DISPFlagDefinition, unit: !586, retainedNodes: !656)
// CHECK:STDOUT: !646 = !DILocalVariable(arg: 1, scope: !645, type: !34)
// CHECK:STDOUT: !647 = !DILocalVariable(arg: 2, scope: !645, type: !35)
// CHECK:STDOUT: !654 = !DILocation(line: 14, column: 52, scope: !645)
// CHECK:STDOUT: !655 = !DILocation(line: 14, column: 45, scope: !645)
// CHECK:STDOUT: !656 = !{!646, !647}
// CHECK:STDOUT: !657 = distinct !DISubprogram(name: "TestLess_u16_u32", linkageName: "_CTestLess_u16_u32.Main", scope: null, file: !585, line: 22, type: !588, spFlags: DISPFlagDefinition, unit: !586, retainedNodes: !668)
// CHECK:STDOUT: !658 = !DILocalVariable(arg: 1, scope: !657, type: !454)
// CHECK:STDOUT: !659 = !DILocalVariable(arg: 2, scope: !657, type: !35)
// CHECK:STDOUT: !666 = !DILocation(line: 22, column: 54, scope: !657)
// CHECK:STDOUT: !667 = !DILocation(line: 22, column: 47, scope: !657)
// CHECK:STDOUT: !668 = !{!658, !659}
// CHECK:STDOUT: !669 = distinct !DISubprogram(name: "TestLess_i16_u32", linkageName: "_CTestLess_i16_u32.Main", scope: null, file: !585, line: 23, type: !590, spFlags: DISPFlagDefinition, unit: !586, retainedNodes: !680)
// CHECK:STDOUT: !670 = !DILocalVariable(arg: 1, scope: !669, type: !453)
// CHECK:STDOUT: !671 = !DILocalVariable(arg: 2, scope: !669, type: !35)
// CHECK:STDOUT: !678 = !DILocation(line: 23, column: 54, scope: !669)
// CHECK:STDOUT: !679 = !DILocation(line: 23, column: 47, scope: !669)
// CHECK:STDOUT: !680 = !{!670, !671}
// CHECK:STDOUT: !681 = distinct !DISubprogram(name: "TestLess_u16_i32", linkageName: "_CTestLess_u16_i32.Main", scope: null, file: !585, line: 24, type: !592, spFlags: DISPFlagDefinition, unit: !586, retainedNodes: !692)
// CHECK:STDOUT: !682 = !DILocalVariable(arg: 1, scope: !681, type: !454)
// CHECK:STDOUT: !683 = !DILocalVariable(arg: 2, scope: !681, type: !34)
// CHECK:STDOUT: !690 = !DILocation(line: 24, column: 54, scope: !681)
// CHECK:STDOUT: !691 = !DILocation(line: 24, column: 47, scope: !681)
// CHECK:STDOUT: !692 = !{!682, !683}
// CHECK:STDOUT: !693 = distinct !DISubprogram(name: "TestLess_i16_i32", linkageName: "_CTestLess_i16_i32.Main", scope: null, file: !585, line: 25, type: !594, spFlags: DISPFlagDefinition, unit: !586, retainedNodes: !704)
// CHECK:STDOUT: !694 = !DILocalVariable(arg: 1, scope: !693, type: !453)
// CHECK:STDOUT: !695 = !DILocalVariable(arg: 2, scope: !693, type: !34)
// CHECK:STDOUT: !702 = !DILocation(line: 25, column: 54, scope: !693)
// CHECK:STDOUT: !703 = !DILocation(line: 25, column: 47, scope: !693)
// CHECK:STDOUT: !704 = !{!694, !695}
// CHECK:STDOUT: !705 = distinct !DISubprogram(name: "TestLess_i32_u32", linkageName: "_CTestLess_i32_u32.Main", scope: null, file: !585, line: 26, type: !596, spFlags: DISPFlagDefinition, unit: !586, retainedNodes: !716)
// CHECK:STDOUT: !706 = !DILocalVariable(arg: 1, scope: !705, type: !34)
// CHECK:STDOUT: !707 = !DILocalVariable(arg: 2, scope: !705, type: !35)
// CHECK:STDOUT: !714 = !DILocation(line: 26, column: 54, scope: !705)
// CHECK:STDOUT: !715 = !DILocation(line: 26, column: 47, scope: !705)
// CHECK:STDOUT: !716 = !{!706, !707}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'convert.carbon'
// CHECK:STDOUT: source_filename = "convert.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestInt32ToInt32.Main(i32 %a) #0 !dbg !742 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %a, !dbg !749
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestInt32ToUint32.Main(i32 %a) #0 !dbg !751 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %a, !dbg !758
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestUint32ToInt32.Main(i32 %a) #0 !dbg !760 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %a, !dbg !767
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestUint32ToUint32.Main(i32 %a) #0 !dbg !769 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %a, !dbg !776
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i16 @_CTestInt32ToInt16.Main(i32 %a) #0 !dbg !778 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int32ToInt16.call = trunc i32 %a to i16, !dbg !784
// CHECK:STDOUT:   ret i16 %Int32ToInt16.call, !dbg !785
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i16 @_CTestInt32ToUint16.Main(i32 %a) #0 !dbg !787 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int32ToUint16.call = trunc i32 %a to i16, !dbg !793
// CHECK:STDOUT:   ret i16 %Int32ToUint16.call, !dbg !794
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i16 @_CTestUint32ToInt16.Main(i32 %a) #0 !dbg !796 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Uint32ToInt16.call = trunc i32 %a to i16, !dbg !802
// CHECK:STDOUT:   ret i16 %Uint32ToInt16.call, !dbg !803
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i16 @_CTestUint32ToUint16.Main(i32 %a) #0 !dbg !805 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Uint32ToUint16.call = trunc i32 %a to i16, !dbg !811
// CHECK:STDOUT:   ret i16 %Uint32ToUint16.call, !dbg !812
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestInt32ToInt64.Main(i32 %a) #0 !dbg !814 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int32ToInt64.call = sext i32 %a to i64, !dbg !820
// CHECK:STDOUT:   ret i64 %Int32ToInt64.call, !dbg !821
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestInt32ToUint64.Main(i32 %a) #0 !dbg !823 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int32ToUint64.call = sext i32 %a to i64, !dbg !829
// CHECK:STDOUT:   ret i64 %Int32ToUint64.call, !dbg !830
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestUint32ToInt64.Main(i32 %a) #0 !dbg !832 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Uint32ToInt64.call = zext i32 %a to i64, !dbg !838
// CHECK:STDOUT:   ret i64 %Uint32ToInt64.call, !dbg !839
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestUint32ToUint64.Main(i32 %a) #0 !dbg !841 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Uint32ToUint64.call = zext i32 %a to i64, !dbg !847
// CHECK:STDOUT:   ret i64 %Uint32ToUint64.call, !dbg !848
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!718}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !17 = !DIBasicType(name: "int", size: 64, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !34 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !35 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !36 = !{!34, !34}
// CHECK:STDOUT: !37 = !DISubroutineType(types: !36)
// CHECK:STDOUT: !453 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !454 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !717 = !DIFile(filename: "convert.carbon", directory: "")
// CHECK:STDOUT: !718 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !717, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !719 = !DIBasicType(name: "int", size: 64, encoding: DW_ATE_signed)
// CHECK:STDOUT: !720 = !{!35, !34}
// CHECK:STDOUT: !721 = !DISubroutineType(types: !720)
// CHECK:STDOUT: !722 = !{!34, !35}
// CHECK:STDOUT: !723 = !DISubroutineType(types: !722)
// CHECK:STDOUT: !724 = !{!35, !35}
// CHECK:STDOUT: !725 = !DISubroutineType(types: !724)
// CHECK:STDOUT: !726 = !{!453, !34}
// CHECK:STDOUT: !727 = !DISubroutineType(types: !726)
// CHECK:STDOUT: !728 = !{!454, !34}
// CHECK:STDOUT: !729 = !DISubroutineType(types: !728)
// CHECK:STDOUT: !730 = !{!453, !35}
// CHECK:STDOUT: !731 = !DISubroutineType(types: !730)
// CHECK:STDOUT: !732 = !{!454, !35}
// CHECK:STDOUT: !733 = !DISubroutineType(types: !732)
// CHECK:STDOUT: !734 = !{!719, !34}
// CHECK:STDOUT: !735 = !DISubroutineType(types: !734)
// CHECK:STDOUT: !736 = !{!17, !34}
// CHECK:STDOUT: !737 = !DISubroutineType(types: !736)
// CHECK:STDOUT: !738 = !{!719, !35}
// CHECK:STDOUT: !739 = !DISubroutineType(types: !738)
// CHECK:STDOUT: !740 = !{!17, !35}
// CHECK:STDOUT: !741 = !DISubroutineType(types: !740)
// CHECK:STDOUT: !742 = distinct !DISubprogram(name: "TestInt32ToInt32", linkageName: "_CTestInt32ToInt32.Main", scope: null, file: !717, line: 10, type: !37, spFlags: DISPFlagDefinition, unit: !718, retainedNodes: !750)
// CHECK:STDOUT: !743 = !DILocalVariable(arg: 1, scope: !742, type: !34)
// CHECK:STDOUT: !749 = !DILocation(line: 10, column: 38, scope: !742)
// CHECK:STDOUT: !750 = !{!743}
// CHECK:STDOUT: !751 = distinct !DISubprogram(name: "TestInt32ToUint32", linkageName: "_CTestInt32ToUint32.Main", scope: null, file: !717, line: 11, type: !721, spFlags: DISPFlagDefinition, unit: !718, retainedNodes: !759)
// CHECK:STDOUT: !752 = !DILocalVariable(arg: 1, scope: !751, type: !34)
// CHECK:STDOUT: !758 = !DILocation(line: 11, column: 39, scope: !751)
// CHECK:STDOUT: !759 = !{!752}
// CHECK:STDOUT: !760 = distinct !DISubprogram(name: "TestUint32ToInt32", linkageName: "_CTestUint32ToInt32.Main", scope: null, file: !717, line: 12, type: !723, spFlags: DISPFlagDefinition, unit: !718, retainedNodes: !768)
// CHECK:STDOUT: !761 = !DILocalVariable(arg: 1, scope: !760, type: !35)
// CHECK:STDOUT: !767 = !DILocation(line: 12, column: 39, scope: !760)
// CHECK:STDOUT: !768 = !{!761}
// CHECK:STDOUT: !769 = distinct !DISubprogram(name: "TestUint32ToUint32", linkageName: "_CTestUint32ToUint32.Main", scope: null, file: !717, line: 13, type: !725, spFlags: DISPFlagDefinition, unit: !718, retainedNodes: !777)
// CHECK:STDOUT: !770 = !DILocalVariable(arg: 1, scope: !769, type: !35)
// CHECK:STDOUT: !776 = !DILocation(line: 13, column: 40, scope: !769)
// CHECK:STDOUT: !777 = !{!770}
// CHECK:STDOUT: !778 = distinct !DISubprogram(name: "TestInt32ToInt16", linkageName: "_CTestInt32ToInt16.Main", scope: null, file: !717, line: 21, type: !727, spFlags: DISPFlagDefinition, unit: !718, retainedNodes: !786)
// CHECK:STDOUT: !779 = !DILocalVariable(arg: 1, scope: !778, type: !34)
// CHECK:STDOUT: !784 = !DILocation(line: 21, column: 45, scope: !778)
// CHECK:STDOUT: !785 = !DILocation(line: 21, column: 38, scope: !778)
// CHECK:STDOUT: !786 = !{!779}
// CHECK:STDOUT: !787 = distinct !DISubprogram(name: "TestInt32ToUint16", linkageName: "_CTestInt32ToUint16.Main", scope: null, file: !717, line: 22, type: !729, spFlags: DISPFlagDefinition, unit: !718, retainedNodes: !795)
// CHECK:STDOUT: !788 = !DILocalVariable(arg: 1, scope: !787, type: !34)
// CHECK:STDOUT: !793 = !DILocation(line: 22, column: 46, scope: !787)
// CHECK:STDOUT: !794 = !DILocation(line: 22, column: 39, scope: !787)
// CHECK:STDOUT: !795 = !{!788}
// CHECK:STDOUT: !796 = distinct !DISubprogram(name: "TestUint32ToInt16", linkageName: "_CTestUint32ToInt16.Main", scope: null, file: !717, line: 23, type: !731, spFlags: DISPFlagDefinition, unit: !718, retainedNodes: !804)
// CHECK:STDOUT: !797 = !DILocalVariable(arg: 1, scope: !796, type: !35)
// CHECK:STDOUT: !802 = !DILocation(line: 23, column: 46, scope: !796)
// CHECK:STDOUT: !803 = !DILocation(line: 23, column: 39, scope: !796)
// CHECK:STDOUT: !804 = !{!797}
// CHECK:STDOUT: !805 = distinct !DISubprogram(name: "TestUint32ToUint16", linkageName: "_CTestUint32ToUint16.Main", scope: null, file: !717, line: 24, type: !733, spFlags: DISPFlagDefinition, unit: !718, retainedNodes: !813)
// CHECK:STDOUT: !806 = !DILocalVariable(arg: 1, scope: !805, type: !35)
// CHECK:STDOUT: !811 = !DILocation(line: 24, column: 47, scope: !805)
// CHECK:STDOUT: !812 = !DILocation(line: 24, column: 40, scope: !805)
// CHECK:STDOUT: !813 = !{!806}
// CHECK:STDOUT: !814 = distinct !DISubprogram(name: "TestInt32ToInt64", linkageName: "_CTestInt32ToInt64.Main", scope: null, file: !717, line: 32, type: !735, spFlags: DISPFlagDefinition, unit: !718, retainedNodes: !822)
// CHECK:STDOUT: !815 = !DILocalVariable(arg: 1, scope: !814, type: !34)
// CHECK:STDOUT: !820 = !DILocation(line: 32, column: 45, scope: !814)
// CHECK:STDOUT: !821 = !DILocation(line: 32, column: 38, scope: !814)
// CHECK:STDOUT: !822 = !{!815}
// CHECK:STDOUT: !823 = distinct !DISubprogram(name: "TestInt32ToUint64", linkageName: "_CTestInt32ToUint64.Main", scope: null, file: !717, line: 33, type: !737, spFlags: DISPFlagDefinition, unit: !718, retainedNodes: !831)
// CHECK:STDOUT: !824 = !DILocalVariable(arg: 1, scope: !823, type: !34)
// CHECK:STDOUT: !829 = !DILocation(line: 33, column: 46, scope: !823)
// CHECK:STDOUT: !830 = !DILocation(line: 33, column: 39, scope: !823)
// CHECK:STDOUT: !831 = !{!824}
// CHECK:STDOUT: !832 = distinct !DISubprogram(name: "TestUint32ToInt64", linkageName: "_CTestUint32ToInt64.Main", scope: null, file: !717, line: 34, type: !739, spFlags: DISPFlagDefinition, unit: !718, retainedNodes: !840)
// CHECK:STDOUT: !833 = !DILocalVariable(arg: 1, scope: !832, type: !35)
// CHECK:STDOUT: !838 = !DILocation(line: 34, column: 46, scope: !832)
// CHECK:STDOUT: !839 = !DILocation(line: 34, column: 39, scope: !832)
// CHECK:STDOUT: !840 = !{!833}
// CHECK:STDOUT: !841 = distinct !DISubprogram(name: "TestUint32ToUint64", linkageName: "_CTestUint32ToUint64.Main", scope: null, file: !717, line: 35, type: !741, spFlags: DISPFlagDefinition, unit: !718, retainedNodes: !849)
// CHECK:STDOUT: !842 = !DILocalVariable(arg: 1, scope: !841, type: !35)
// CHECK:STDOUT: !847 = !DILocation(line: 35, column: 47, scope: !841)
// CHECK:STDOUT: !848 = !DILocation(line: 35, column: 40, scope: !841)
// CHECK:STDOUT: !849 = !{!842}
// CHECK:STDOUT:
