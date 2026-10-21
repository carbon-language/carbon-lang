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
// CHECK:STDOUT: define i32 @_CTestNegate.Main(i32 %a) #0 !dbg !47 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Negate.call = sub i32 0, %a, !dbg !53
// CHECK:STDOUT:   ret i32 %Negate.call, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestAdd.Main(i32 %a, i32 %b) #0 !dbg !56 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Add.call = add i32 %a, %b, !dbg !65
// CHECK:STDOUT:   ret i32 %Add.call, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestSub.Main(i32 %a, i32 %b) #0 !dbg !68 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Sub.call = sub i32 %a, %b, !dbg !77
// CHECK:STDOUT:   ret i32 %Sub.call, !dbg !78
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestMul.Main(i32 %a, i32 %b) #0 !dbg !80 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Mul.call = mul i32 %a, %b, !dbg !89
// CHECK:STDOUT:   ret i32 %Mul.call, !dbg !90
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestDiv.Main(i32 %a, i32 %b) #0 !dbg !92 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Div.call = sdiv i32 %a, %b, !dbg !101
// CHECK:STDOUT:   ret i32 %Div.call, !dbg !102
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestMod.Main(i32 %a, i32 %b) #0 !dbg !104 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Mod.call = srem i32 %a, %b, !dbg !113
// CHECK:STDOUT:   ret i32 %Mod.call, !dbg !114
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestComplement.Main(i32 %a) #0 !dbg !116 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Complement.call = xor i32 -1, %a, !dbg !122
// CHECK:STDOUT:   ret i32 %Complement.call, !dbg !123
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestAnd.Main(i32 %a, i32 %b) #0 !dbg !125 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %And.call = and i32 %a, %b, !dbg !134
// CHECK:STDOUT:   ret i32 %And.call, !dbg !135
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestOr.Main(i32 %a, i32 %b) #0 !dbg !137 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Or.call = or i32 %a, %b, !dbg !146
// CHECK:STDOUT:   ret i32 %Or.call, !dbg !147
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestXor.Main(i32 %a, i32 %b) #0 !dbg !149 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Xor.call = xor i32 %a, %b, !dbg !158
// CHECK:STDOUT:   ret i32 %Xor.call, !dbg !159
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestLeftShift.Main(i32 %a, i32 %b) #0 !dbg !161 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %LeftShift.call = shl i32 %a, %b, !dbg !170
// CHECK:STDOUT:   ret i32 %LeftShift.call, !dbg !171
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestArithmeticRightShift.Main(i32 %a, i32 %b) #0 !dbg !173 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ArithmeticRightShift.call = ashr i32 %a, %b, !dbg !182
// CHECK:STDOUT:   ret i32 %ArithmeticRightShift.call, !dbg !183
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestLogicalRightShift.Main(i32 %a, i32 %b) #0 !dbg !185 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %LogicalRightShift.call = lshr i32 %a, %b, !dbg !194
// CHECK:STDOUT:   ret i32 %LogicalRightShift.call, !dbg !195
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestEq.Main(i32 %a, i32 %b) #0 !dbg !197 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Eq.call = icmp eq i32 %a, %b, !dbg !206
// CHECK:STDOUT:   ret i1 %Eq.call, !dbg !207
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestNeq.Main(i32 %a, i32 %b) #0 !dbg !209 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Neq.call = icmp ne i32 %a, %b, !dbg !218
// CHECK:STDOUT:   ret i1 %Neq.call, !dbg !219
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestLess.Main(i32 %a, i32 %b) #0 !dbg !221 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Less.call = icmp slt i32 %a, %b, !dbg !230
// CHECK:STDOUT:   ret i1 %Less.call, !dbg !231
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestLessEq.Main(i32 %a, i32 %b) #0 !dbg !233 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %LessEq.call = icmp sle i32 %a, %b, !dbg !242
// CHECK:STDOUT:   ret i1 %LessEq.call, !dbg !243
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestGreater.Main(i32 %a, i32 %b) #0 !dbg !245 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Greater.call = icmp sgt i32 %a, %b, !dbg !254
// CHECK:STDOUT:   ret i1 %Greater.call, !dbg !255
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestGreaterEq.Main(i32 %a, i32 %b) #0 !dbg !257 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %GreaterEq.call = icmp sge i32 %a, %b, !dbg !266
// CHECK:STDOUT:   ret i1 %GreaterEq.call, !dbg !267
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!35}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !34 = !DIFile(filename: "basic.carbon", directory: "")
// CHECK:STDOUT: !35 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !34, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !37 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !38 = !{!36, !36}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !40 = !{!36, !36, !36}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !42 = !{!37, !37, !37}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !44 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !45 = !{!44, !36, !36}
// CHECK:STDOUT: !46 = !DISubroutineType(types: !45)
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "TestNegate", linkageName: "_CTestNegate.Main", scope: null, file: !34, line: 5, type: !39, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !55)
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 1, scope: !47, type: !36)
// CHECK:STDOUT: !53 = !DILocation(line: 5, column: 39, scope: !47)
// CHECK:STDOUT: !54 = !DILocation(line: 5, column: 32, scope: !47)
// CHECK:STDOUT: !55 = !{!48}
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "TestAdd", linkageName: "_CTestAdd.Main", scope: null, file: !34, line: 8, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !67)
// CHECK:STDOUT: !57 = !DILocalVariable(arg: 1, scope: !56, type: !36)
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 2, scope: !56, type: !36)
// CHECK:STDOUT: !65 = !DILocation(line: 8, column: 44, scope: !56)
// CHECK:STDOUT: !66 = !DILocation(line: 8, column: 37, scope: !56)
// CHECK:STDOUT: !67 = !{!57, !58}
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "TestSub", linkageName: "_CTestSub.Main", scope: null, file: !34, line: 11, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !79)
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 1, scope: !68, type: !36)
// CHECK:STDOUT: !70 = !DILocalVariable(arg: 2, scope: !68, type: !36)
// CHECK:STDOUT: !77 = !DILocation(line: 11, column: 44, scope: !68)
// CHECK:STDOUT: !78 = !DILocation(line: 11, column: 37, scope: !68)
// CHECK:STDOUT: !79 = !{!69, !70}
// CHECK:STDOUT: !80 = distinct !DISubprogram(name: "TestMul", linkageName: "_CTestMul.Main", scope: null, file: !34, line: 14, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !91)
// CHECK:STDOUT: !81 = !DILocalVariable(arg: 1, scope: !80, type: !36)
// CHECK:STDOUT: !82 = !DILocalVariable(arg: 2, scope: !80, type: !36)
// CHECK:STDOUT: !89 = !DILocation(line: 14, column: 44, scope: !80)
// CHECK:STDOUT: !90 = !DILocation(line: 14, column: 37, scope: !80)
// CHECK:STDOUT: !91 = !{!81, !82}
// CHECK:STDOUT: !92 = distinct !DISubprogram(name: "TestDiv", linkageName: "_CTestDiv.Main", scope: null, file: !34, line: 17, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !103)
// CHECK:STDOUT: !93 = !DILocalVariable(arg: 1, scope: !92, type: !36)
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 2, scope: !92, type: !36)
// CHECK:STDOUT: !101 = !DILocation(line: 17, column: 44, scope: !92)
// CHECK:STDOUT: !102 = !DILocation(line: 17, column: 37, scope: !92)
// CHECK:STDOUT: !103 = !{!93, !94}
// CHECK:STDOUT: !104 = distinct !DISubprogram(name: "TestMod", linkageName: "_CTestMod.Main", scope: null, file: !34, line: 20, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !115)
// CHECK:STDOUT: !105 = !DILocalVariable(arg: 1, scope: !104, type: !36)
// CHECK:STDOUT: !106 = !DILocalVariable(arg: 2, scope: !104, type: !36)
// CHECK:STDOUT: !113 = !DILocation(line: 20, column: 44, scope: !104)
// CHECK:STDOUT: !114 = !DILocation(line: 20, column: 37, scope: !104)
// CHECK:STDOUT: !115 = !{!105, !106}
// CHECK:STDOUT: !116 = distinct !DISubprogram(name: "TestComplement", linkageName: "_CTestComplement.Main", scope: null, file: !34, line: 23, type: !39, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !124)
// CHECK:STDOUT: !117 = !DILocalVariable(arg: 1, scope: !116, type: !36)
// CHECK:STDOUT: !122 = !DILocation(line: 23, column: 43, scope: !116)
// CHECK:STDOUT: !123 = !DILocation(line: 23, column: 36, scope: !116)
// CHECK:STDOUT: !124 = !{!117}
// CHECK:STDOUT: !125 = distinct !DISubprogram(name: "TestAnd", linkageName: "_CTestAnd.Main", scope: null, file: !34, line: 26, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !136)
// CHECK:STDOUT: !126 = !DILocalVariable(arg: 1, scope: !125, type: !36)
// CHECK:STDOUT: !127 = !DILocalVariable(arg: 2, scope: !125, type: !36)
// CHECK:STDOUT: !134 = !DILocation(line: 26, column: 44, scope: !125)
// CHECK:STDOUT: !135 = !DILocation(line: 26, column: 37, scope: !125)
// CHECK:STDOUT: !136 = !{!126, !127}
// CHECK:STDOUT: !137 = distinct !DISubprogram(name: "TestOr", linkageName: "_CTestOr.Main", scope: null, file: !34, line: 29, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !148)
// CHECK:STDOUT: !138 = !DILocalVariable(arg: 1, scope: !137, type: !36)
// CHECK:STDOUT: !139 = !DILocalVariable(arg: 2, scope: !137, type: !36)
// CHECK:STDOUT: !146 = !DILocation(line: 29, column: 43, scope: !137)
// CHECK:STDOUT: !147 = !DILocation(line: 29, column: 36, scope: !137)
// CHECK:STDOUT: !148 = !{!138, !139}
// CHECK:STDOUT: !149 = distinct !DISubprogram(name: "TestXor", linkageName: "_CTestXor.Main", scope: null, file: !34, line: 32, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !160)
// CHECK:STDOUT: !150 = !DILocalVariable(arg: 1, scope: !149, type: !36)
// CHECK:STDOUT: !151 = !DILocalVariable(arg: 2, scope: !149, type: !36)
// CHECK:STDOUT: !158 = !DILocation(line: 32, column: 44, scope: !149)
// CHECK:STDOUT: !159 = !DILocation(line: 32, column: 37, scope: !149)
// CHECK:STDOUT: !160 = !{!150, !151}
// CHECK:STDOUT: !161 = distinct !DISubprogram(name: "TestLeftShift", linkageName: "_CTestLeftShift.Main", scope: null, file: !34, line: 35, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !172)
// CHECK:STDOUT: !162 = !DILocalVariable(arg: 1, scope: !161, type: !36)
// CHECK:STDOUT: !163 = !DILocalVariable(arg: 2, scope: !161, type: !36)
// CHECK:STDOUT: !170 = !DILocation(line: 35, column: 50, scope: !161)
// CHECK:STDOUT: !171 = !DILocation(line: 35, column: 43, scope: !161)
// CHECK:STDOUT: !172 = !{!162, !163}
// CHECK:STDOUT: !173 = distinct !DISubprogram(name: "TestArithmeticRightShift", linkageName: "_CTestArithmeticRightShift.Main", scope: null, file: !34, line: 38, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !184)
// CHECK:STDOUT: !174 = !DILocalVariable(arg: 1, scope: !173, type: !36)
// CHECK:STDOUT: !175 = !DILocalVariable(arg: 2, scope: !173, type: !36)
// CHECK:STDOUT: !182 = !DILocation(line: 38, column: 61, scope: !173)
// CHECK:STDOUT: !183 = !DILocation(line: 38, column: 54, scope: !173)
// CHECK:STDOUT: !184 = !{!174, !175}
// CHECK:STDOUT: !185 = distinct !DISubprogram(name: "TestLogicalRightShift", linkageName: "_CTestLogicalRightShift.Main", scope: null, file: !34, line: 41, type: !43, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !196)
// CHECK:STDOUT: !186 = !DILocalVariable(arg: 1, scope: !185, type: !37)
// CHECK:STDOUT: !187 = !DILocalVariable(arg: 2, scope: !185, type: !37)
// CHECK:STDOUT: !194 = !DILocation(line: 41, column: 58, scope: !185)
// CHECK:STDOUT: !195 = !DILocation(line: 41, column: 51, scope: !185)
// CHECK:STDOUT: !196 = !{!186, !187}
// CHECK:STDOUT: !197 = distinct !DISubprogram(name: "TestEq", linkageName: "_CTestEq.Main", scope: null, file: !34, line: 44, type: !46, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !208)
// CHECK:STDOUT: !198 = !DILocalVariable(arg: 1, scope: !197, type: !36)
// CHECK:STDOUT: !199 = !DILocalVariable(arg: 2, scope: !197, type: !36)
// CHECK:STDOUT: !206 = !DILocation(line: 44, column: 44, scope: !197)
// CHECK:STDOUT: !207 = !DILocation(line: 44, column: 37, scope: !197)
// CHECK:STDOUT: !208 = !{!198, !199}
// CHECK:STDOUT: !209 = distinct !DISubprogram(name: "TestNeq", linkageName: "_CTestNeq.Main", scope: null, file: !34, line: 47, type: !46, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !220)
// CHECK:STDOUT: !210 = !DILocalVariable(arg: 1, scope: !209, type: !36)
// CHECK:STDOUT: !211 = !DILocalVariable(arg: 2, scope: !209, type: !36)
// CHECK:STDOUT: !218 = !DILocation(line: 47, column: 45, scope: !209)
// CHECK:STDOUT: !219 = !DILocation(line: 47, column: 38, scope: !209)
// CHECK:STDOUT: !220 = !{!210, !211}
// CHECK:STDOUT: !221 = distinct !DISubprogram(name: "TestLess", linkageName: "_CTestLess.Main", scope: null, file: !34, line: 50, type: !46, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !232)
// CHECK:STDOUT: !222 = !DILocalVariable(arg: 1, scope: !221, type: !36)
// CHECK:STDOUT: !223 = !DILocalVariable(arg: 2, scope: !221, type: !36)
// CHECK:STDOUT: !230 = !DILocation(line: 50, column: 46, scope: !221)
// CHECK:STDOUT: !231 = !DILocation(line: 50, column: 39, scope: !221)
// CHECK:STDOUT: !232 = !{!222, !223}
// CHECK:STDOUT: !233 = distinct !DISubprogram(name: "TestLessEq", linkageName: "_CTestLessEq.Main", scope: null, file: !34, line: 53, type: !46, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !244)
// CHECK:STDOUT: !234 = !DILocalVariable(arg: 1, scope: !233, type: !36)
// CHECK:STDOUT: !235 = !DILocalVariable(arg: 2, scope: !233, type: !36)
// CHECK:STDOUT: !242 = !DILocation(line: 53, column: 48, scope: !233)
// CHECK:STDOUT: !243 = !DILocation(line: 53, column: 41, scope: !233)
// CHECK:STDOUT: !244 = !{!234, !235}
// CHECK:STDOUT: !245 = distinct !DISubprogram(name: "TestGreater", linkageName: "_CTestGreater.Main", scope: null, file: !34, line: 56, type: !46, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !256)
// CHECK:STDOUT: !246 = !DILocalVariable(arg: 1, scope: !245, type: !36)
// CHECK:STDOUT: !247 = !DILocalVariable(arg: 2, scope: !245, type: !36)
// CHECK:STDOUT: !254 = !DILocation(line: 56, column: 49, scope: !245)
// CHECK:STDOUT: !255 = !DILocation(line: 56, column: 42, scope: !245)
// CHECK:STDOUT: !256 = !{!246, !247}
// CHECK:STDOUT: !257 = distinct !DISubprogram(name: "TestGreaterEq", linkageName: "_CTestGreaterEq.Main", scope: null, file: !34, line: 59, type: !46, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !268)
// CHECK:STDOUT: !258 = !DILocalVariable(arg: 1, scope: !257, type: !36)
// CHECK:STDOUT: !259 = !DILocalVariable(arg: 2, scope: !257, type: !36)
// CHECK:STDOUT: !266 = !DILocation(line: 59, column: 51, scope: !257)
// CHECK:STDOUT: !267 = !DILocation(line: 59, column: 44, scope: !257)
// CHECK:STDOUT: !268 = !{!258, !259}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'compound_assign.carbon'
// CHECK:STDOUT: source_filename = "compound_assign.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestSAdd.Main(ptr %a, i32 %b) #0 !dbg !277 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %SAdd.call = load i32, ptr %a, align 4, !dbg !285
// CHECK:STDOUT:   %SAdd.call1 = add i32 %SAdd.call, %b, !dbg !285
// CHECK:STDOUT:   store i32 %SAdd.call1, ptr %a, align 4, !dbg !285
// CHECK:STDOUT:   ret void, !dbg !286
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestSSub.Main(ptr %a, i32 %b) #0 !dbg !288 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %SSub.call = load i32, ptr %a, align 4, !dbg !296
// CHECK:STDOUT:   %SSub.call1 = sub i32 %SSub.call, %b, !dbg !296
// CHECK:STDOUT:   store i32 %SSub.call1, ptr %a, align 4, !dbg !296
// CHECK:STDOUT:   ret void, !dbg !297
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestSMul.Main(ptr %a, i32 %b) #0 !dbg !299 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %SMul.call = load i32, ptr %a, align 4, !dbg !307
// CHECK:STDOUT:   %SMul.call1 = mul i32 %SMul.call, %b, !dbg !307
// CHECK:STDOUT:   store i32 %SMul.call1, ptr %a, align 4, !dbg !307
// CHECK:STDOUT:   ret void, !dbg !308
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestSDiv.Main(ptr %a, i32 %b) #0 !dbg !310 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %SDiv.call = load i32, ptr %a, align 4, !dbg !318
// CHECK:STDOUT:   %SDiv.call1 = sdiv i32 %SDiv.call, %b, !dbg !318
// CHECK:STDOUT:   store i32 %SDiv.call1, ptr %a, align 4, !dbg !318
// CHECK:STDOUT:   ret void, !dbg !319
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestSMod.Main(ptr %a, i32 %b) #0 !dbg !321 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %SMod.call = load i32, ptr %a, align 4, !dbg !329
// CHECK:STDOUT:   %SMod.call1 = srem i32 %SMod.call, %b, !dbg !329
// CHECK:STDOUT:   store i32 %SMod.call1, ptr %a, align 4, !dbg !329
// CHECK:STDOUT:   ret void, !dbg !330
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestUAdd.Main(ptr %a, i32 %b) #0 !dbg !332 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %UAdd.call = load i32, ptr %a, align 4, !dbg !340
// CHECK:STDOUT:   %UAdd.call1 = add i32 %UAdd.call, %b, !dbg !340
// CHECK:STDOUT:   store i32 %UAdd.call1, ptr %a, align 4, !dbg !340
// CHECK:STDOUT:   ret void, !dbg !341
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestUSub.Main(ptr %a, i32 %b) #0 !dbg !343 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %USub.call = load i32, ptr %a, align 4, !dbg !351
// CHECK:STDOUT:   %USub.call1 = sub i32 %USub.call, %b, !dbg !351
// CHECK:STDOUT:   store i32 %USub.call1, ptr %a, align 4, !dbg !351
// CHECK:STDOUT:   ret void, !dbg !352
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestUMul.Main(ptr %a, i32 %b) #0 !dbg !354 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %UMul.call = load i32, ptr %a, align 4, !dbg !362
// CHECK:STDOUT:   %UMul.call1 = mul i32 %UMul.call, %b, !dbg !362
// CHECK:STDOUT:   store i32 %UMul.call1, ptr %a, align 4, !dbg !362
// CHECK:STDOUT:   ret void, !dbg !363
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestUDiv.Main(ptr %a, i32 %b) #0 !dbg !365 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %UDiv.call = load i32, ptr %a, align 4, !dbg !373
// CHECK:STDOUT:   %UDiv.call1 = udiv i32 %UDiv.call, %b, !dbg !373
// CHECK:STDOUT:   store i32 %UDiv.call1, ptr %a, align 4, !dbg !373
// CHECK:STDOUT:   ret void, !dbg !374
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestUMod.Main(ptr %a, i32 %b) #0 !dbg !376 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %UMod.call = load i32, ptr %a, align 4, !dbg !384
// CHECK:STDOUT:   %UMod.call1 = urem i32 %UMod.call, %b, !dbg !384
// CHECK:STDOUT:   store i32 %UMod.call1, ptr %a, align 4, !dbg !384
// CHECK:STDOUT:   ret void, !dbg !385
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestAnd.Main(ptr %a, i32 %b) #0 !dbg !387 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %And.call = load i32, ptr %a, align 4, !dbg !395
// CHECK:STDOUT:   %And.call1 = and i32 %And.call, %b, !dbg !395
// CHECK:STDOUT:   store i32 %And.call1, ptr %a, align 4, !dbg !395
// CHECK:STDOUT:   ret void, !dbg !396
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestOr.Main(ptr %a, i32 %b) #0 !dbg !398 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Or.call = load i32, ptr %a, align 4, !dbg !406
// CHECK:STDOUT:   %Or.call1 = or i32 %Or.call, %b, !dbg !406
// CHECK:STDOUT:   store i32 %Or.call1, ptr %a, align 4, !dbg !406
// CHECK:STDOUT:   ret void, !dbg !407
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestXor.Main(ptr %a, i32 %b) #0 !dbg !409 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Xor.call = load i32, ptr %a, align 4, !dbg !417
// CHECK:STDOUT:   %Xor.call1 = xor i32 %Xor.call, %b, !dbg !417
// CHECK:STDOUT:   store i32 %Xor.call1, ptr %a, align 4, !dbg !417
// CHECK:STDOUT:   ret void, !dbg !418
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestLeftShift.Main(ptr %a, i32 %b) #0 !dbg !420 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %LeftShift.call = load i32, ptr %a, align 4, !dbg !428
// CHECK:STDOUT:   %LeftShift.call1 = shl i32 %LeftShift.call, %b, !dbg !428
// CHECK:STDOUT:   store i32 %LeftShift.call1, ptr %a, align 4, !dbg !428
// CHECK:STDOUT:   ret void, !dbg !429
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestArithmeticRightShift.Main(ptr %a, i32 %b) #0 !dbg !431 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ArithmeticRightShift.call = load i32, ptr %a, align 4, !dbg !439
// CHECK:STDOUT:   %ArithmeticRightShift.call1 = ashr i32 %ArithmeticRightShift.call, %b, !dbg !439
// CHECK:STDOUT:   store i32 %ArithmeticRightShift.call1, ptr %a, align 4, !dbg !439
// CHECK:STDOUT:   ret void, !dbg !440
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestLogicalRightShift.Main(ptr %a, i32 %b) #0 !dbg !442 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %LogicalRightShift.call = load i32, ptr %a, align 4, !dbg !450
// CHECK:STDOUT:   %LogicalRightShift.call1 = lshr i32 %LogicalRightShift.call, %b, !dbg !450
// CHECK:STDOUT:   store i32 %LogicalRightShift.call1, ptr %a, align 4, !dbg !450
// CHECK:STDOUT:   ret void, !dbg !451
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!270}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !37 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !269 = !DIFile(filename: "compound_assign.carbon", directory: "")
// CHECK:STDOUT: !270 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !269, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !271 = !{null, !36, !36}
// CHECK:STDOUT: !272 = !DISubroutineType(types: !271)
// CHECK:STDOUT: !273 = !{null, !36, !37}
// CHECK:STDOUT: !274 = !DISubroutineType(types: !273)
// CHECK:STDOUT: !275 = !{null, !37, !37}
// CHECK:STDOUT: !276 = !DISubroutineType(types: !275)
// CHECK:STDOUT: !277 = distinct !DISubprogram(name: "TestSAdd", linkageName: "_CTestSAdd.Main", scope: null, file: !269, line: 5, type: !272, spFlags: DISPFlagDefinition, unit: !270, retainedNodes: !287)
// CHECK:STDOUT: !278 = !DILocalVariable(arg: 1, scope: !277, type: !36)
// CHECK:STDOUT: !279 = !DILocalVariable(arg: 2, scope: !277, type: !36)
// CHECK:STDOUT: !285 = !DILocation(line: 5, column: 35, scope: !277)
// CHECK:STDOUT: !286 = !DILocation(line: 5, column: 1, scope: !277)
// CHECK:STDOUT: !287 = !{!278, !279}
// CHECK:STDOUT: !288 = distinct !DISubprogram(name: "TestSSub", linkageName: "_CTestSSub.Main", scope: null, file: !269, line: 8, type: !272, spFlags: DISPFlagDefinition, unit: !270, retainedNodes: !298)
// CHECK:STDOUT: !289 = !DILocalVariable(arg: 1, scope: !288, type: !36)
// CHECK:STDOUT: !290 = !DILocalVariable(arg: 2, scope: !288, type: !36)
// CHECK:STDOUT: !296 = !DILocation(line: 8, column: 35, scope: !288)
// CHECK:STDOUT: !297 = !DILocation(line: 8, column: 1, scope: !288)
// CHECK:STDOUT: !298 = !{!289, !290}
// CHECK:STDOUT: !299 = distinct !DISubprogram(name: "TestSMul", linkageName: "_CTestSMul.Main", scope: null, file: !269, line: 11, type: !272, spFlags: DISPFlagDefinition, unit: !270, retainedNodes: !309)
// CHECK:STDOUT: !300 = !DILocalVariable(arg: 1, scope: !299, type: !36)
// CHECK:STDOUT: !301 = !DILocalVariable(arg: 2, scope: !299, type: !36)
// CHECK:STDOUT: !307 = !DILocation(line: 11, column: 35, scope: !299)
// CHECK:STDOUT: !308 = !DILocation(line: 11, column: 1, scope: !299)
// CHECK:STDOUT: !309 = !{!300, !301}
// CHECK:STDOUT: !310 = distinct !DISubprogram(name: "TestSDiv", linkageName: "_CTestSDiv.Main", scope: null, file: !269, line: 14, type: !272, spFlags: DISPFlagDefinition, unit: !270, retainedNodes: !320)
// CHECK:STDOUT: !311 = !DILocalVariable(arg: 1, scope: !310, type: !36)
// CHECK:STDOUT: !312 = !DILocalVariable(arg: 2, scope: !310, type: !36)
// CHECK:STDOUT: !318 = !DILocation(line: 14, column: 35, scope: !310)
// CHECK:STDOUT: !319 = !DILocation(line: 14, column: 1, scope: !310)
// CHECK:STDOUT: !320 = !{!311, !312}
// CHECK:STDOUT: !321 = distinct !DISubprogram(name: "TestSMod", linkageName: "_CTestSMod.Main", scope: null, file: !269, line: 17, type: !272, spFlags: DISPFlagDefinition, unit: !270, retainedNodes: !331)
// CHECK:STDOUT: !322 = !DILocalVariable(arg: 1, scope: !321, type: !36)
// CHECK:STDOUT: !323 = !DILocalVariable(arg: 2, scope: !321, type: !36)
// CHECK:STDOUT: !329 = !DILocation(line: 17, column: 35, scope: !321)
// CHECK:STDOUT: !330 = !DILocation(line: 17, column: 1, scope: !321)
// CHECK:STDOUT: !331 = !{!322, !323}
// CHECK:STDOUT: !332 = distinct !DISubprogram(name: "TestUAdd", linkageName: "_CTestUAdd.Main", scope: null, file: !269, line: 20, type: !272, spFlags: DISPFlagDefinition, unit: !270, retainedNodes: !342)
// CHECK:STDOUT: !333 = !DILocalVariable(arg: 1, scope: !332, type: !36)
// CHECK:STDOUT: !334 = !DILocalVariable(arg: 2, scope: !332, type: !36)
// CHECK:STDOUT: !340 = !DILocation(line: 20, column: 35, scope: !332)
// CHECK:STDOUT: !341 = !DILocation(line: 20, column: 1, scope: !332)
// CHECK:STDOUT: !342 = !{!333, !334}
// CHECK:STDOUT: !343 = distinct !DISubprogram(name: "TestUSub", linkageName: "_CTestUSub.Main", scope: null, file: !269, line: 23, type: !272, spFlags: DISPFlagDefinition, unit: !270, retainedNodes: !353)
// CHECK:STDOUT: !344 = !DILocalVariable(arg: 1, scope: !343, type: !36)
// CHECK:STDOUT: !345 = !DILocalVariable(arg: 2, scope: !343, type: !36)
// CHECK:STDOUT: !351 = !DILocation(line: 23, column: 35, scope: !343)
// CHECK:STDOUT: !352 = !DILocation(line: 23, column: 1, scope: !343)
// CHECK:STDOUT: !353 = !{!344, !345}
// CHECK:STDOUT: !354 = distinct !DISubprogram(name: "TestUMul", linkageName: "_CTestUMul.Main", scope: null, file: !269, line: 26, type: !272, spFlags: DISPFlagDefinition, unit: !270, retainedNodes: !364)
// CHECK:STDOUT: !355 = !DILocalVariable(arg: 1, scope: !354, type: !36)
// CHECK:STDOUT: !356 = !DILocalVariable(arg: 2, scope: !354, type: !36)
// CHECK:STDOUT: !362 = !DILocation(line: 26, column: 35, scope: !354)
// CHECK:STDOUT: !363 = !DILocation(line: 26, column: 1, scope: !354)
// CHECK:STDOUT: !364 = !{!355, !356}
// CHECK:STDOUT: !365 = distinct !DISubprogram(name: "TestUDiv", linkageName: "_CTestUDiv.Main", scope: null, file: !269, line: 29, type: !272, spFlags: DISPFlagDefinition, unit: !270, retainedNodes: !375)
// CHECK:STDOUT: !366 = !DILocalVariable(arg: 1, scope: !365, type: !36)
// CHECK:STDOUT: !367 = !DILocalVariable(arg: 2, scope: !365, type: !36)
// CHECK:STDOUT: !373 = !DILocation(line: 29, column: 35, scope: !365)
// CHECK:STDOUT: !374 = !DILocation(line: 29, column: 1, scope: !365)
// CHECK:STDOUT: !375 = !{!366, !367}
// CHECK:STDOUT: !376 = distinct !DISubprogram(name: "TestUMod", linkageName: "_CTestUMod.Main", scope: null, file: !269, line: 32, type: !272, spFlags: DISPFlagDefinition, unit: !270, retainedNodes: !386)
// CHECK:STDOUT: !377 = !DILocalVariable(arg: 1, scope: !376, type: !36)
// CHECK:STDOUT: !378 = !DILocalVariable(arg: 2, scope: !376, type: !36)
// CHECK:STDOUT: !384 = !DILocation(line: 32, column: 35, scope: !376)
// CHECK:STDOUT: !385 = !DILocation(line: 32, column: 1, scope: !376)
// CHECK:STDOUT: !386 = !{!377, !378}
// CHECK:STDOUT: !387 = distinct !DISubprogram(name: "TestAnd", linkageName: "_CTestAnd.Main", scope: null, file: !269, line: 35, type: !272, spFlags: DISPFlagDefinition, unit: !270, retainedNodes: !397)
// CHECK:STDOUT: !388 = !DILocalVariable(arg: 1, scope: !387, type: !36)
// CHECK:STDOUT: !389 = !DILocalVariable(arg: 2, scope: !387, type: !36)
// CHECK:STDOUT: !395 = !DILocation(line: 35, column: 34, scope: !387)
// CHECK:STDOUT: !396 = !DILocation(line: 35, column: 1, scope: !387)
// CHECK:STDOUT: !397 = !{!388, !389}
// CHECK:STDOUT: !398 = distinct !DISubprogram(name: "TestOr", linkageName: "_CTestOr.Main", scope: null, file: !269, line: 38, type: !272, spFlags: DISPFlagDefinition, unit: !270, retainedNodes: !408)
// CHECK:STDOUT: !399 = !DILocalVariable(arg: 1, scope: !398, type: !36)
// CHECK:STDOUT: !400 = !DILocalVariable(arg: 2, scope: !398, type: !36)
// CHECK:STDOUT: !406 = !DILocation(line: 38, column: 33, scope: !398)
// CHECK:STDOUT: !407 = !DILocation(line: 38, column: 1, scope: !398)
// CHECK:STDOUT: !408 = !{!399, !400}
// CHECK:STDOUT: !409 = distinct !DISubprogram(name: "TestXor", linkageName: "_CTestXor.Main", scope: null, file: !269, line: 41, type: !272, spFlags: DISPFlagDefinition, unit: !270, retainedNodes: !419)
// CHECK:STDOUT: !410 = !DILocalVariable(arg: 1, scope: !409, type: !36)
// CHECK:STDOUT: !411 = !DILocalVariable(arg: 2, scope: !409, type: !36)
// CHECK:STDOUT: !417 = !DILocation(line: 41, column: 34, scope: !409)
// CHECK:STDOUT: !418 = !DILocation(line: 41, column: 1, scope: !409)
// CHECK:STDOUT: !419 = !{!410, !411}
// CHECK:STDOUT: !420 = distinct !DISubprogram(name: "TestLeftShift", linkageName: "_CTestLeftShift.Main", scope: null, file: !269, line: 44, type: !272, spFlags: DISPFlagDefinition, unit: !270, retainedNodes: !430)
// CHECK:STDOUT: !421 = !DILocalVariable(arg: 1, scope: !420, type: !36)
// CHECK:STDOUT: !422 = !DILocalVariable(arg: 2, scope: !420, type: !36)
// CHECK:STDOUT: !428 = !DILocation(line: 44, column: 40, scope: !420)
// CHECK:STDOUT: !429 = !DILocation(line: 44, column: 1, scope: !420)
// CHECK:STDOUT: !430 = !{!421, !422}
// CHECK:STDOUT: !431 = distinct !DISubprogram(name: "TestArithmeticRightShift", linkageName: "_CTestArithmeticRightShift.Main", scope: null, file: !269, line: 47, type: !274, spFlags: DISPFlagDefinition, unit: !270, retainedNodes: !441)
// CHECK:STDOUT: !432 = !DILocalVariable(arg: 1, scope: !431, type: !36)
// CHECK:STDOUT: !433 = !DILocalVariable(arg: 2, scope: !431, type: !37)
// CHECK:STDOUT: !439 = !DILocation(line: 47, column: 51, scope: !431)
// CHECK:STDOUT: !440 = !DILocation(line: 47, column: 1, scope: !431)
// CHECK:STDOUT: !441 = !{!432, !433}
// CHECK:STDOUT: !442 = distinct !DISubprogram(name: "TestLogicalRightShift", linkageName: "_CTestLogicalRightShift.Main", scope: null, file: !269, line: 50, type: !276, spFlags: DISPFlagDefinition, unit: !270, retainedNodes: !452)
// CHECK:STDOUT: !443 = !DILocalVariable(arg: 1, scope: !442, type: !37)
// CHECK:STDOUT: !444 = !DILocalVariable(arg: 2, scope: !442, type: !37)
// CHECK:STDOUT: !450 = !DILocation(line: 50, column: 48, scope: !442)
// CHECK:STDOUT: !451 = !DILocation(line: 50, column: 1, scope: !442)
// CHECK:STDOUT: !452 = !{!443, !444}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'mixed_shift.carbon'
// CHECK:STDOUT: source_filename = "mixed_shift.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestLeftShiftSmaller.Main(i32 %a, i16 %b) #0 !dbg !467 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %LeftShiftSmaller.call.rhs = zext i16 %b to i32, !dbg !476
// CHECK:STDOUT:   %LeftShiftSmaller.call = shl i32 %a, %LeftShiftSmaller.call.rhs, !dbg !476
// CHECK:STDOUT:   ret i32 %LeftShiftSmaller.call, !dbg !477
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestRightShiftSmaller.Main(i32 %a, i16 %b) #0 !dbg !479 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %RightShiftSmaller.call.rhs = zext i16 %b to i32, !dbg !488
// CHECK:STDOUT:   %RightShiftSmaller.call = ashr i32 %a, %RightShiftSmaller.call.rhs, !dbg !488
// CHECK:STDOUT:   ret i32 %RightShiftSmaller.call, !dbg !489
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i16 @_CTestLeftShiftLargerII.Main(i16 %a, i32 %b) #0 !dbg !491 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %LeftShiftLargerII.call.rhs = trunc i32 %b to i16, !dbg !500
// CHECK:STDOUT:   %LeftShiftLargerII.call = shl i16 %a, %LeftShiftLargerII.call.rhs, !dbg !500
// CHECK:STDOUT:   ret i16 %LeftShiftLargerII.call, !dbg !501
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i16 @_CTestRightShiftLargerII.Main(i16 %a, i32 %b) #0 !dbg !503 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %RightShiftLargerII.call.rhs = trunc i32 %b to i16, !dbg !512
// CHECK:STDOUT:   %RightShiftLargerII.call = ashr i16 %a, %RightShiftLargerII.call.rhs, !dbg !512
// CHECK:STDOUT:   ret i16 %RightShiftLargerII.call, !dbg !513
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i16 @_CTestLeftShiftLargerIU.Main(i16 %a, i32 %b) #0 !dbg !515 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %LeftShiftLargerIU.call.rhs = trunc i32 %b to i16, !dbg !524
// CHECK:STDOUT:   %LeftShiftLargerIU.call = shl i16 %a, %LeftShiftLargerIU.call.rhs, !dbg !524
// CHECK:STDOUT:   ret i16 %LeftShiftLargerIU.call, !dbg !525
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i16 @_CTestRightShiftLargerIU.Main(i16 %a, i32 %b) #0 !dbg !527 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %RightShiftLargerIU.call.rhs = trunc i32 %b to i16, !dbg !536
// CHECK:STDOUT:   %RightShiftLargerIU.call = ashr i16 %a, %RightShiftLargerIU.call.rhs, !dbg !536
// CHECK:STDOUT:   ret i16 %RightShiftLargerIU.call, !dbg !537
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i16 @_CTestLeftShiftLargerUI.Main(i16 %a, i32 %b) #0 !dbg !539 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %LeftShiftLargerUI.call.rhs = trunc i32 %b to i16, !dbg !548
// CHECK:STDOUT:   %LeftShiftLargerUI.call = shl i16 %a, %LeftShiftLargerUI.call.rhs, !dbg !548
// CHECK:STDOUT:   ret i16 %LeftShiftLargerUI.call, !dbg !549
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i16 @_CTestRightShiftLargerUI.Main(i16 %a, i32 %b) #0 !dbg !551 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %RightShiftLargerUI.call.rhs = trunc i32 %b to i16, !dbg !560
// CHECK:STDOUT:   %RightShiftLargerUI.call = lshr i16 %a, %RightShiftLargerUI.call.rhs, !dbg !560
// CHECK:STDOUT:   ret i16 %RightShiftLargerUI.call, !dbg !561
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i16 @_CTestLeftShiftLargerUU.Main(i16 %a, i32 %b) #0 !dbg !563 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %LeftShiftLargerUU.call.rhs = trunc i32 %b to i16, !dbg !572
// CHECK:STDOUT:   %LeftShiftLargerUU.call = shl i16 %a, %LeftShiftLargerUU.call.rhs, !dbg !572
// CHECK:STDOUT:   ret i16 %LeftShiftLargerUU.call, !dbg !573
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i16 @_CTestRightShiftLargerUU.Main(i16 %a, i32 %b) #0 !dbg !575 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %RightShiftLargerUU.call.rhs = trunc i32 %b to i16, !dbg !584
// CHECK:STDOUT:   %RightShiftLargerUU.call = lshr i16 %a, %RightShiftLargerUU.call.rhs, !dbg !584
// CHECK:STDOUT:   ret i16 %RightShiftLargerUU.call, !dbg !585
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!454}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !37 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !453 = !DIFile(filename: "mixed_shift.carbon", directory: "")
// CHECK:STDOUT: !454 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !453, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !455 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !456 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !457 = !{!36, !36, !455}
// CHECK:STDOUT: !458 = !DISubroutineType(types: !457)
// CHECK:STDOUT: !459 = !{!455, !455, !36}
// CHECK:STDOUT: !460 = !DISubroutineType(types: !459)
// CHECK:STDOUT: !461 = !{!455, !455, !37}
// CHECK:STDOUT: !462 = !DISubroutineType(types: !461)
// CHECK:STDOUT: !463 = !{!456, !456, !36}
// CHECK:STDOUT: !464 = !DISubroutineType(types: !463)
// CHECK:STDOUT: !465 = !{!456, !456, !37}
// CHECK:STDOUT: !466 = !DISubroutineType(types: !465)
// CHECK:STDOUT: !467 = distinct !DISubprogram(name: "TestLeftShiftSmaller", linkageName: "_CTestLeftShiftSmaller.Main", scope: null, file: !453, line: 5, type: !458, spFlags: DISPFlagDefinition, unit: !454, retainedNodes: !478)
// CHECK:STDOUT: !468 = !DILocalVariable(arg: 1, scope: !467, type: !36)
// CHECK:STDOUT: !469 = !DILocalVariable(arg: 2, scope: !467, type: !455)
// CHECK:STDOUT: !476 = !DILocation(line: 5, column: 57, scope: !467)
// CHECK:STDOUT: !477 = !DILocation(line: 5, column: 50, scope: !467)
// CHECK:STDOUT: !478 = !{!468, !469}
// CHECK:STDOUT: !479 = distinct !DISubprogram(name: "TestRightShiftSmaller", linkageName: "_CTestRightShiftSmaller.Main", scope: null, file: !453, line: 8, type: !458, spFlags: DISPFlagDefinition, unit: !454, retainedNodes: !490)
// CHECK:STDOUT: !480 = !DILocalVariable(arg: 1, scope: !479, type: !36)
// CHECK:STDOUT: !481 = !DILocalVariable(arg: 2, scope: !479, type: !455)
// CHECK:STDOUT: !488 = !DILocation(line: 8, column: 58, scope: !479)
// CHECK:STDOUT: !489 = !DILocation(line: 8, column: 51, scope: !479)
// CHECK:STDOUT: !490 = !{!480, !481}
// CHECK:STDOUT: !491 = distinct !DISubprogram(name: "TestLeftShiftLargerII", linkageName: "_CTestLeftShiftLargerII.Main", scope: null, file: !453, line: 11, type: !460, spFlags: DISPFlagDefinition, unit: !454, retainedNodes: !502)
// CHECK:STDOUT: !492 = !DILocalVariable(arg: 1, scope: !491, type: !455)
// CHECK:STDOUT: !493 = !DILocalVariable(arg: 2, scope: !491, type: !36)
// CHECK:STDOUT: !500 = !DILocation(line: 11, column: 58, scope: !491)
// CHECK:STDOUT: !501 = !DILocation(line: 11, column: 51, scope: !491)
// CHECK:STDOUT: !502 = !{!492, !493}
// CHECK:STDOUT: !503 = distinct !DISubprogram(name: "TestRightShiftLargerII", linkageName: "_CTestRightShiftLargerII.Main", scope: null, file: !453, line: 14, type: !460, spFlags: DISPFlagDefinition, unit: !454, retainedNodes: !514)
// CHECK:STDOUT: !504 = !DILocalVariable(arg: 1, scope: !503, type: !455)
// CHECK:STDOUT: !505 = !DILocalVariable(arg: 2, scope: !503, type: !36)
// CHECK:STDOUT: !512 = !DILocation(line: 14, column: 59, scope: !503)
// CHECK:STDOUT: !513 = !DILocation(line: 14, column: 52, scope: !503)
// CHECK:STDOUT: !514 = !{!504, !505}
// CHECK:STDOUT: !515 = distinct !DISubprogram(name: "TestLeftShiftLargerIU", linkageName: "_CTestLeftShiftLargerIU.Main", scope: null, file: !453, line: 17, type: !462, spFlags: DISPFlagDefinition, unit: !454, retainedNodes: !526)
// CHECK:STDOUT: !516 = !DILocalVariable(arg: 1, scope: !515, type: !455)
// CHECK:STDOUT: !517 = !DILocalVariable(arg: 2, scope: !515, type: !37)
// CHECK:STDOUT: !524 = !DILocation(line: 17, column: 58, scope: !515)
// CHECK:STDOUT: !525 = !DILocation(line: 17, column: 51, scope: !515)
// CHECK:STDOUT: !526 = !{!516, !517}
// CHECK:STDOUT: !527 = distinct !DISubprogram(name: "TestRightShiftLargerIU", linkageName: "_CTestRightShiftLargerIU.Main", scope: null, file: !453, line: 20, type: !462, spFlags: DISPFlagDefinition, unit: !454, retainedNodes: !538)
// CHECK:STDOUT: !528 = !DILocalVariable(arg: 1, scope: !527, type: !455)
// CHECK:STDOUT: !529 = !DILocalVariable(arg: 2, scope: !527, type: !37)
// CHECK:STDOUT: !536 = !DILocation(line: 20, column: 59, scope: !527)
// CHECK:STDOUT: !537 = !DILocation(line: 20, column: 52, scope: !527)
// CHECK:STDOUT: !538 = !{!528, !529}
// CHECK:STDOUT: !539 = distinct !DISubprogram(name: "TestLeftShiftLargerUI", linkageName: "_CTestLeftShiftLargerUI.Main", scope: null, file: !453, line: 23, type: !464, spFlags: DISPFlagDefinition, unit: !454, retainedNodes: !550)
// CHECK:STDOUT: !540 = !DILocalVariable(arg: 1, scope: !539, type: !456)
// CHECK:STDOUT: !541 = !DILocalVariable(arg: 2, scope: !539, type: !36)
// CHECK:STDOUT: !548 = !DILocation(line: 23, column: 58, scope: !539)
// CHECK:STDOUT: !549 = !DILocation(line: 23, column: 51, scope: !539)
// CHECK:STDOUT: !550 = !{!540, !541}
// CHECK:STDOUT: !551 = distinct !DISubprogram(name: "TestRightShiftLargerUI", linkageName: "_CTestRightShiftLargerUI.Main", scope: null, file: !453, line: 26, type: !464, spFlags: DISPFlagDefinition, unit: !454, retainedNodes: !562)
// CHECK:STDOUT: !552 = !DILocalVariable(arg: 1, scope: !551, type: !456)
// CHECK:STDOUT: !553 = !DILocalVariable(arg: 2, scope: !551, type: !36)
// CHECK:STDOUT: !560 = !DILocation(line: 26, column: 59, scope: !551)
// CHECK:STDOUT: !561 = !DILocation(line: 26, column: 52, scope: !551)
// CHECK:STDOUT: !562 = !{!552, !553}
// CHECK:STDOUT: !563 = distinct !DISubprogram(name: "TestLeftShiftLargerUU", linkageName: "_CTestLeftShiftLargerUU.Main", scope: null, file: !453, line: 29, type: !466, spFlags: DISPFlagDefinition, unit: !454, retainedNodes: !574)
// CHECK:STDOUT: !564 = !DILocalVariable(arg: 1, scope: !563, type: !456)
// CHECK:STDOUT: !565 = !DILocalVariable(arg: 2, scope: !563, type: !37)
// CHECK:STDOUT: !572 = !DILocation(line: 29, column: 58, scope: !563)
// CHECK:STDOUT: !573 = !DILocation(line: 29, column: 51, scope: !563)
// CHECK:STDOUT: !574 = !{!564, !565}
// CHECK:STDOUT: !575 = distinct !DISubprogram(name: "TestRightShiftLargerUU", linkageName: "_CTestRightShiftLargerUU.Main", scope: null, file: !453, line: 32, type: !466, spFlags: DISPFlagDefinition, unit: !454, retainedNodes: !586)
// CHECK:STDOUT: !576 = !DILocalVariable(arg: 1, scope: !575, type: !456)
// CHECK:STDOUT: !577 = !DILocalVariable(arg: 2, scope: !575, type: !37)
// CHECK:STDOUT: !584 = !DILocation(line: 32, column: 59, scope: !575)
// CHECK:STDOUT: !585 = !DILocation(line: 32, column: 52, scope: !575)
// CHECK:STDOUT: !586 = !{!576, !577}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'mixed_compare.carbon'
// CHECK:STDOUT: source_filename = "mixed_compare.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestEq_u16_u32.Main(i16 %a, i32 %b) #0 !dbg !599 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Eq_u16_u32.call.lhs = zext i16 %a to i32, !dbg !608
// CHECK:STDOUT:   %Eq_u16_u32.call = icmp eq i32 %Eq_u16_u32.call.lhs, %b, !dbg !608
// CHECK:STDOUT:   ret i1 %Eq_u16_u32.call, !dbg !609
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestEq_i16_u32.Main(i16 %a, i32 %b) #0 !dbg !611 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Eq_i16_u32.call.lhs = sext i16 %a to i33, !dbg !620
// CHECK:STDOUT:   %Eq_i16_u32.call.rhs = zext i32 %b to i33, !dbg !620
// CHECK:STDOUT:   %Eq_i16_u32.call = icmp eq i33 %Eq_i16_u32.call.lhs, %Eq_i16_u32.call.rhs, !dbg !620
// CHECK:STDOUT:   ret i1 %Eq_i16_u32.call, !dbg !621
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestEq_u16_i32.Main(i16 %a, i32 %b) #0 !dbg !623 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Eq_u16_i32.call.lhs = zext i16 %a to i32, !dbg !632
// CHECK:STDOUT:   %Eq_u16_i32.call = icmp eq i32 %Eq_u16_i32.call.lhs, %b, !dbg !632
// CHECK:STDOUT:   ret i1 %Eq_u16_i32.call, !dbg !633
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestEq_i16_i32.Main(i16 %a, i32 %b) #0 !dbg !635 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Eq_i16_i32.call.lhs = sext i16 %a to i32, !dbg !644
// CHECK:STDOUT:   %Eq_i16_i32.call = icmp eq i32 %Eq_i16_i32.call.lhs, %b, !dbg !644
// CHECK:STDOUT:   ret i1 %Eq_i16_i32.call, !dbg !645
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestEq_i32_u32.Main(i32 %a, i32 %b) #0 !dbg !647 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Eq_i32_u32.call.lhs = sext i32 %a to i33, !dbg !656
// CHECK:STDOUT:   %Eq_i32_u32.call.rhs = zext i32 %b to i33, !dbg !656
// CHECK:STDOUT:   %Eq_i32_u32.call = icmp eq i33 %Eq_i32_u32.call.lhs, %Eq_i32_u32.call.rhs, !dbg !656
// CHECK:STDOUT:   ret i1 %Eq_i32_u32.call, !dbg !657
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestLess_u16_u32.Main(i16 %a, i32 %b) #0 !dbg !659 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Less_u16_u32.call.lhs = zext i16 %a to i32, !dbg !668
// CHECK:STDOUT:   %Less_u16_u32.call = icmp ult i32 %Less_u16_u32.call.lhs, %b, !dbg !668
// CHECK:STDOUT:   ret i1 %Less_u16_u32.call, !dbg !669
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestLess_i16_u32.Main(i16 %a, i32 %b) #0 !dbg !671 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Less_i16_u32.call.lhs = sext i16 %a to i33, !dbg !680
// CHECK:STDOUT:   %Less_i16_u32.call.rhs = zext i32 %b to i33, !dbg !680
// CHECK:STDOUT:   %Less_i16_u32.call = icmp slt i33 %Less_i16_u32.call.lhs, %Less_i16_u32.call.rhs, !dbg !680
// CHECK:STDOUT:   ret i1 %Less_i16_u32.call, !dbg !681
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestLess_u16_i32.Main(i16 %a, i32 %b) #0 !dbg !683 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Less_u16_i32.call.lhs = zext i16 %a to i32, !dbg !692
// CHECK:STDOUT:   %Less_u16_i32.call = icmp slt i32 %Less_u16_i32.call.lhs, %b, !dbg !692
// CHECK:STDOUT:   ret i1 %Less_u16_i32.call, !dbg !693
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestLess_i16_i32.Main(i16 %a, i32 %b) #0 !dbg !695 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Less_i16_i32.call.lhs = sext i16 %a to i32, !dbg !704
// CHECK:STDOUT:   %Less_i16_i32.call = icmp slt i32 %Less_i16_i32.call.lhs, %b, !dbg !704
// CHECK:STDOUT:   ret i1 %Less_i16_i32.call, !dbg !705
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CTestLess_i32_u32.Main(i32 %a, i32 %b) #0 !dbg !707 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Less_i32_u32.call.lhs = sext i32 %a to i33, !dbg !716
// CHECK:STDOUT:   %Less_i32_u32.call.rhs = zext i32 %b to i33, !dbg !716
// CHECK:STDOUT:   %Less_i32_u32.call = icmp slt i33 %Less_i32_u32.call.lhs, %Less_i32_u32.call.rhs, !dbg !716
// CHECK:STDOUT:   ret i1 %Less_i32_u32.call, !dbg !717
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!588}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !37 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !44 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !455 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !456 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !587 = !DIFile(filename: "mixed_compare.carbon", directory: "")
// CHECK:STDOUT: !588 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !587, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !589 = !{!44, !456, !37}
// CHECK:STDOUT: !590 = !DISubroutineType(types: !589)
// CHECK:STDOUT: !591 = !{!44, !455, !37}
// CHECK:STDOUT: !592 = !DISubroutineType(types: !591)
// CHECK:STDOUT: !593 = !{!44, !456, !36}
// CHECK:STDOUT: !594 = !DISubroutineType(types: !593)
// CHECK:STDOUT: !595 = !{!44, !455, !36}
// CHECK:STDOUT: !596 = !DISubroutineType(types: !595)
// CHECK:STDOUT: !597 = !{!44, !36, !37}
// CHECK:STDOUT: !598 = !DISubroutineType(types: !597)
// CHECK:STDOUT: !599 = distinct !DISubprogram(name: "TestEq_u16_u32", linkageName: "_CTestEq_u16_u32.Main", scope: null, file: !587, line: 10, type: !590, spFlags: DISPFlagDefinition, unit: !588, retainedNodes: !610)
// CHECK:STDOUT: !600 = !DILocalVariable(arg: 1, scope: !599, type: !456)
// CHECK:STDOUT: !601 = !DILocalVariable(arg: 2, scope: !599, type: !37)
// CHECK:STDOUT: !608 = !DILocation(line: 10, column: 52, scope: !599)
// CHECK:STDOUT: !609 = !DILocation(line: 10, column: 45, scope: !599)
// CHECK:STDOUT: !610 = !{!600, !601}
// CHECK:STDOUT: !611 = distinct !DISubprogram(name: "TestEq_i16_u32", linkageName: "_CTestEq_i16_u32.Main", scope: null, file: !587, line: 11, type: !592, spFlags: DISPFlagDefinition, unit: !588, retainedNodes: !622)
// CHECK:STDOUT: !612 = !DILocalVariable(arg: 1, scope: !611, type: !455)
// CHECK:STDOUT: !613 = !DILocalVariable(arg: 2, scope: !611, type: !37)
// CHECK:STDOUT: !620 = !DILocation(line: 11, column: 52, scope: !611)
// CHECK:STDOUT: !621 = !DILocation(line: 11, column: 45, scope: !611)
// CHECK:STDOUT: !622 = !{!612, !613}
// CHECK:STDOUT: !623 = distinct !DISubprogram(name: "TestEq_u16_i32", linkageName: "_CTestEq_u16_i32.Main", scope: null, file: !587, line: 12, type: !594, spFlags: DISPFlagDefinition, unit: !588, retainedNodes: !634)
// CHECK:STDOUT: !624 = !DILocalVariable(arg: 1, scope: !623, type: !456)
// CHECK:STDOUT: !625 = !DILocalVariable(arg: 2, scope: !623, type: !36)
// CHECK:STDOUT: !632 = !DILocation(line: 12, column: 52, scope: !623)
// CHECK:STDOUT: !633 = !DILocation(line: 12, column: 45, scope: !623)
// CHECK:STDOUT: !634 = !{!624, !625}
// CHECK:STDOUT: !635 = distinct !DISubprogram(name: "TestEq_i16_i32", linkageName: "_CTestEq_i16_i32.Main", scope: null, file: !587, line: 13, type: !596, spFlags: DISPFlagDefinition, unit: !588, retainedNodes: !646)
// CHECK:STDOUT: !636 = !DILocalVariable(arg: 1, scope: !635, type: !455)
// CHECK:STDOUT: !637 = !DILocalVariable(arg: 2, scope: !635, type: !36)
// CHECK:STDOUT: !644 = !DILocation(line: 13, column: 52, scope: !635)
// CHECK:STDOUT: !645 = !DILocation(line: 13, column: 45, scope: !635)
// CHECK:STDOUT: !646 = !{!636, !637}
// CHECK:STDOUT: !647 = distinct !DISubprogram(name: "TestEq_i32_u32", linkageName: "_CTestEq_i32_u32.Main", scope: null, file: !587, line: 14, type: !598, spFlags: DISPFlagDefinition, unit: !588, retainedNodes: !658)
// CHECK:STDOUT: !648 = !DILocalVariable(arg: 1, scope: !647, type: !36)
// CHECK:STDOUT: !649 = !DILocalVariable(arg: 2, scope: !647, type: !37)
// CHECK:STDOUT: !656 = !DILocation(line: 14, column: 52, scope: !647)
// CHECK:STDOUT: !657 = !DILocation(line: 14, column: 45, scope: !647)
// CHECK:STDOUT: !658 = !{!648, !649}
// CHECK:STDOUT: !659 = distinct !DISubprogram(name: "TestLess_u16_u32", linkageName: "_CTestLess_u16_u32.Main", scope: null, file: !587, line: 22, type: !590, spFlags: DISPFlagDefinition, unit: !588, retainedNodes: !670)
// CHECK:STDOUT: !660 = !DILocalVariable(arg: 1, scope: !659, type: !456)
// CHECK:STDOUT: !661 = !DILocalVariable(arg: 2, scope: !659, type: !37)
// CHECK:STDOUT: !668 = !DILocation(line: 22, column: 54, scope: !659)
// CHECK:STDOUT: !669 = !DILocation(line: 22, column: 47, scope: !659)
// CHECK:STDOUT: !670 = !{!660, !661}
// CHECK:STDOUT: !671 = distinct !DISubprogram(name: "TestLess_i16_u32", linkageName: "_CTestLess_i16_u32.Main", scope: null, file: !587, line: 23, type: !592, spFlags: DISPFlagDefinition, unit: !588, retainedNodes: !682)
// CHECK:STDOUT: !672 = !DILocalVariable(arg: 1, scope: !671, type: !455)
// CHECK:STDOUT: !673 = !DILocalVariable(arg: 2, scope: !671, type: !37)
// CHECK:STDOUT: !680 = !DILocation(line: 23, column: 54, scope: !671)
// CHECK:STDOUT: !681 = !DILocation(line: 23, column: 47, scope: !671)
// CHECK:STDOUT: !682 = !{!672, !673}
// CHECK:STDOUT: !683 = distinct !DISubprogram(name: "TestLess_u16_i32", linkageName: "_CTestLess_u16_i32.Main", scope: null, file: !587, line: 24, type: !594, spFlags: DISPFlagDefinition, unit: !588, retainedNodes: !694)
// CHECK:STDOUT: !684 = !DILocalVariable(arg: 1, scope: !683, type: !456)
// CHECK:STDOUT: !685 = !DILocalVariable(arg: 2, scope: !683, type: !36)
// CHECK:STDOUT: !692 = !DILocation(line: 24, column: 54, scope: !683)
// CHECK:STDOUT: !693 = !DILocation(line: 24, column: 47, scope: !683)
// CHECK:STDOUT: !694 = !{!684, !685}
// CHECK:STDOUT: !695 = distinct !DISubprogram(name: "TestLess_i16_i32", linkageName: "_CTestLess_i16_i32.Main", scope: null, file: !587, line: 25, type: !596, spFlags: DISPFlagDefinition, unit: !588, retainedNodes: !706)
// CHECK:STDOUT: !696 = !DILocalVariable(arg: 1, scope: !695, type: !455)
// CHECK:STDOUT: !697 = !DILocalVariable(arg: 2, scope: !695, type: !36)
// CHECK:STDOUT: !704 = !DILocation(line: 25, column: 54, scope: !695)
// CHECK:STDOUT: !705 = !DILocation(line: 25, column: 47, scope: !695)
// CHECK:STDOUT: !706 = !{!696, !697}
// CHECK:STDOUT: !707 = distinct !DISubprogram(name: "TestLess_i32_u32", linkageName: "_CTestLess_i32_u32.Main", scope: null, file: !587, line: 26, type: !598, spFlags: DISPFlagDefinition, unit: !588, retainedNodes: !718)
// CHECK:STDOUT: !708 = !DILocalVariable(arg: 1, scope: !707, type: !36)
// CHECK:STDOUT: !709 = !DILocalVariable(arg: 2, scope: !707, type: !37)
// CHECK:STDOUT: !716 = !DILocation(line: 26, column: 54, scope: !707)
// CHECK:STDOUT: !717 = !DILocation(line: 26, column: 47, scope: !707)
// CHECK:STDOUT: !718 = !{!708, !709}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'convert.carbon'
// CHECK:STDOUT: source_filename = "convert.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestInt32ToInt32.Main(i32 %a) #0 !dbg !744 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %a, !dbg !751
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestInt32ToUint32.Main(i32 %a) #0 !dbg !753 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %a, !dbg !760
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestUint32ToInt32.Main(i32 %a) #0 !dbg !762 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %a, !dbg !769
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestUint32ToUint32.Main(i32 %a) #0 !dbg !771 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %a, !dbg !778
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i16 @_CTestInt32ToInt16.Main(i32 %a) #0 !dbg !780 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int32ToInt16.call = trunc i32 %a to i16, !dbg !786
// CHECK:STDOUT:   ret i16 %Int32ToInt16.call, !dbg !787
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i16 @_CTestInt32ToUint16.Main(i32 %a) #0 !dbg !789 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int32ToUint16.call = trunc i32 %a to i16, !dbg !795
// CHECK:STDOUT:   ret i16 %Int32ToUint16.call, !dbg !796
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i16 @_CTestUint32ToInt16.Main(i32 %a) #0 !dbg !798 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Uint32ToInt16.call = trunc i32 %a to i16, !dbg !804
// CHECK:STDOUT:   ret i16 %Uint32ToInt16.call, !dbg !805
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i16 @_CTestUint32ToUint16.Main(i32 %a) #0 !dbg !807 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Uint32ToUint16.call = trunc i32 %a to i16, !dbg !813
// CHECK:STDOUT:   ret i16 %Uint32ToUint16.call, !dbg !814
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestInt32ToInt64.Main(i32 %a) #0 !dbg !816 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int32ToInt64.call = sext i32 %a to i64, !dbg !822
// CHECK:STDOUT:   ret i64 %Int32ToInt64.call, !dbg !823
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestInt32ToUint64.Main(i32 %a) #0 !dbg !825 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int32ToUint64.call = sext i32 %a to i64, !dbg !831
// CHECK:STDOUT:   ret i64 %Int32ToUint64.call, !dbg !832
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestUint32ToInt64.Main(i32 %a) #0 !dbg !834 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Uint32ToInt64.call = zext i32 %a to i64, !dbg !840
// CHECK:STDOUT:   ret i64 %Uint32ToInt64.call, !dbg !841
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestUint32ToUint64.Main(i32 %a) #0 !dbg !843 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Uint32ToUint64.call = zext i32 %a to i64, !dbg !849
// CHECK:STDOUT:   ret i64 %Uint32ToUint64.call, !dbg !850
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!720}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !19 = !DIBasicType(name: "int", size: 64, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !37 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !38 = !{!36, !36}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !455 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !456 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !719 = !DIFile(filename: "convert.carbon", directory: "")
// CHECK:STDOUT: !720 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !719, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !721 = !DIBasicType(name: "int", size: 64, encoding: DW_ATE_signed)
// CHECK:STDOUT: !722 = !{!37, !36}
// CHECK:STDOUT: !723 = !DISubroutineType(types: !722)
// CHECK:STDOUT: !724 = !{!36, !37}
// CHECK:STDOUT: !725 = !DISubroutineType(types: !724)
// CHECK:STDOUT: !726 = !{!37, !37}
// CHECK:STDOUT: !727 = !DISubroutineType(types: !726)
// CHECK:STDOUT: !728 = !{!455, !36}
// CHECK:STDOUT: !729 = !DISubroutineType(types: !728)
// CHECK:STDOUT: !730 = !{!456, !36}
// CHECK:STDOUT: !731 = !DISubroutineType(types: !730)
// CHECK:STDOUT: !732 = !{!455, !37}
// CHECK:STDOUT: !733 = !DISubroutineType(types: !732)
// CHECK:STDOUT: !734 = !{!456, !37}
// CHECK:STDOUT: !735 = !DISubroutineType(types: !734)
// CHECK:STDOUT: !736 = !{!721, !36}
// CHECK:STDOUT: !737 = !DISubroutineType(types: !736)
// CHECK:STDOUT: !738 = !{!19, !36}
// CHECK:STDOUT: !739 = !DISubroutineType(types: !738)
// CHECK:STDOUT: !740 = !{!721, !37}
// CHECK:STDOUT: !741 = !DISubroutineType(types: !740)
// CHECK:STDOUT: !742 = !{!19, !37}
// CHECK:STDOUT: !743 = !DISubroutineType(types: !742)
// CHECK:STDOUT: !744 = distinct !DISubprogram(name: "TestInt32ToInt32", linkageName: "_CTestInt32ToInt32.Main", scope: null, file: !719, line: 10, type: !39, spFlags: DISPFlagDefinition, unit: !720, retainedNodes: !752)
// CHECK:STDOUT: !745 = !DILocalVariable(arg: 1, scope: !744, type: !36)
// CHECK:STDOUT: !751 = !DILocation(line: 10, column: 38, scope: !744)
// CHECK:STDOUT: !752 = !{!745}
// CHECK:STDOUT: !753 = distinct !DISubprogram(name: "TestInt32ToUint32", linkageName: "_CTestInt32ToUint32.Main", scope: null, file: !719, line: 11, type: !723, spFlags: DISPFlagDefinition, unit: !720, retainedNodes: !761)
// CHECK:STDOUT: !754 = !DILocalVariable(arg: 1, scope: !753, type: !36)
// CHECK:STDOUT: !760 = !DILocation(line: 11, column: 39, scope: !753)
// CHECK:STDOUT: !761 = !{!754}
// CHECK:STDOUT: !762 = distinct !DISubprogram(name: "TestUint32ToInt32", linkageName: "_CTestUint32ToInt32.Main", scope: null, file: !719, line: 12, type: !725, spFlags: DISPFlagDefinition, unit: !720, retainedNodes: !770)
// CHECK:STDOUT: !763 = !DILocalVariable(arg: 1, scope: !762, type: !37)
// CHECK:STDOUT: !769 = !DILocation(line: 12, column: 39, scope: !762)
// CHECK:STDOUT: !770 = !{!763}
// CHECK:STDOUT: !771 = distinct !DISubprogram(name: "TestUint32ToUint32", linkageName: "_CTestUint32ToUint32.Main", scope: null, file: !719, line: 13, type: !727, spFlags: DISPFlagDefinition, unit: !720, retainedNodes: !779)
// CHECK:STDOUT: !772 = !DILocalVariable(arg: 1, scope: !771, type: !37)
// CHECK:STDOUT: !778 = !DILocation(line: 13, column: 40, scope: !771)
// CHECK:STDOUT: !779 = !{!772}
// CHECK:STDOUT: !780 = distinct !DISubprogram(name: "TestInt32ToInt16", linkageName: "_CTestInt32ToInt16.Main", scope: null, file: !719, line: 21, type: !729, spFlags: DISPFlagDefinition, unit: !720, retainedNodes: !788)
// CHECK:STDOUT: !781 = !DILocalVariable(arg: 1, scope: !780, type: !36)
// CHECK:STDOUT: !786 = !DILocation(line: 21, column: 45, scope: !780)
// CHECK:STDOUT: !787 = !DILocation(line: 21, column: 38, scope: !780)
// CHECK:STDOUT: !788 = !{!781}
// CHECK:STDOUT: !789 = distinct !DISubprogram(name: "TestInt32ToUint16", linkageName: "_CTestInt32ToUint16.Main", scope: null, file: !719, line: 22, type: !731, spFlags: DISPFlagDefinition, unit: !720, retainedNodes: !797)
// CHECK:STDOUT: !790 = !DILocalVariable(arg: 1, scope: !789, type: !36)
// CHECK:STDOUT: !795 = !DILocation(line: 22, column: 46, scope: !789)
// CHECK:STDOUT: !796 = !DILocation(line: 22, column: 39, scope: !789)
// CHECK:STDOUT: !797 = !{!790}
// CHECK:STDOUT: !798 = distinct !DISubprogram(name: "TestUint32ToInt16", linkageName: "_CTestUint32ToInt16.Main", scope: null, file: !719, line: 23, type: !733, spFlags: DISPFlagDefinition, unit: !720, retainedNodes: !806)
// CHECK:STDOUT: !799 = !DILocalVariable(arg: 1, scope: !798, type: !37)
// CHECK:STDOUT: !804 = !DILocation(line: 23, column: 46, scope: !798)
// CHECK:STDOUT: !805 = !DILocation(line: 23, column: 39, scope: !798)
// CHECK:STDOUT: !806 = !{!799}
// CHECK:STDOUT: !807 = distinct !DISubprogram(name: "TestUint32ToUint16", linkageName: "_CTestUint32ToUint16.Main", scope: null, file: !719, line: 24, type: !735, spFlags: DISPFlagDefinition, unit: !720, retainedNodes: !815)
// CHECK:STDOUT: !808 = !DILocalVariable(arg: 1, scope: !807, type: !37)
// CHECK:STDOUT: !813 = !DILocation(line: 24, column: 47, scope: !807)
// CHECK:STDOUT: !814 = !DILocation(line: 24, column: 40, scope: !807)
// CHECK:STDOUT: !815 = !{!808}
// CHECK:STDOUT: !816 = distinct !DISubprogram(name: "TestInt32ToInt64", linkageName: "_CTestInt32ToInt64.Main", scope: null, file: !719, line: 32, type: !737, spFlags: DISPFlagDefinition, unit: !720, retainedNodes: !824)
// CHECK:STDOUT: !817 = !DILocalVariable(arg: 1, scope: !816, type: !36)
// CHECK:STDOUT: !822 = !DILocation(line: 32, column: 45, scope: !816)
// CHECK:STDOUT: !823 = !DILocation(line: 32, column: 38, scope: !816)
// CHECK:STDOUT: !824 = !{!817}
// CHECK:STDOUT: !825 = distinct !DISubprogram(name: "TestInt32ToUint64", linkageName: "_CTestInt32ToUint64.Main", scope: null, file: !719, line: 33, type: !739, spFlags: DISPFlagDefinition, unit: !720, retainedNodes: !833)
// CHECK:STDOUT: !826 = !DILocalVariable(arg: 1, scope: !825, type: !36)
// CHECK:STDOUT: !831 = !DILocation(line: 33, column: 46, scope: !825)
// CHECK:STDOUT: !832 = !DILocation(line: 33, column: 39, scope: !825)
// CHECK:STDOUT: !833 = !{!826}
// CHECK:STDOUT: !834 = distinct !DISubprogram(name: "TestUint32ToInt64", linkageName: "_CTestUint32ToInt64.Main", scope: null, file: !719, line: 34, type: !741, spFlags: DISPFlagDefinition, unit: !720, retainedNodes: !842)
// CHECK:STDOUT: !835 = !DILocalVariable(arg: 1, scope: !834, type: !37)
// CHECK:STDOUT: !840 = !DILocation(line: 34, column: 46, scope: !834)
// CHECK:STDOUT: !841 = !DILocation(line: 34, column: 39, scope: !834)
// CHECK:STDOUT: !842 = !{!835}
// CHECK:STDOUT: !843 = distinct !DISubprogram(name: "TestUint32ToUint64", linkageName: "_CTestUint32ToUint64.Main", scope: null, file: !719, line: 35, type: !743, spFlags: DISPFlagDefinition, unit: !720, retainedNodes: !851)
// CHECK:STDOUT: !844 = !DILocalVariable(arg: 1, scope: !843, type: !37)
// CHECK:STDOUT: !849 = !DILocation(line: 35, column: 47, scope: !843)
// CHECK:STDOUT: !850 = !DILocation(line: 35, column: 40, scope: !843)
// CHECK:STDOUT: !851 = !{!844}
// CHECK:STDOUT:
