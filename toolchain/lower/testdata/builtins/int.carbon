// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

// CHECK:STDOUT: ; ModuleID = 'basic.carbon'
// CHECK:STDOUT: source_filename = "basic.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CTestNegate.Main(i32 %a) !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.snegate = sub i32 0, %a, !dbg !7
// CHECK:STDOUT:   ret i32 %int.snegate, !dbg !8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CTestAdd.Main(i32 %a, i32 %b) !dbg !9 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.sadd = add i32 %a, %b, !dbg !10
// CHECK:STDOUT:   ret i32 %int.sadd, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CTestSub.Main(i32 %a, i32 %b) !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.ssub = sub i32 %a, %b, !dbg !13
// CHECK:STDOUT:   ret i32 %int.ssub, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CTestMul.Main(i32 %a, i32 %b) !dbg !15 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.smul = mul i32 %a, %b, !dbg !16
// CHECK:STDOUT:   ret i32 %int.smul, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CTestDiv.Main(i32 %a, i32 %b) !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.sdiv = sdiv i32 %a, %b, !dbg !19
// CHECK:STDOUT:   ret i32 %int.sdiv, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CTestMod.Main(i32 %a, i32 %b) !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.smod = srem i32 %a, %b, !dbg !22
// CHECK:STDOUT:   ret i32 %int.smod, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CTestComplement.Main(i32 %a) !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.complement = xor i32 -1, %a, !dbg !25
// CHECK:STDOUT:   ret i32 %int.complement, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CTestAnd.Main(i32 %a, i32 %b) !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.and = and i32 %a, %b, !dbg !28
// CHECK:STDOUT:   ret i32 %int.and, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CTestOr.Main(i32 %a, i32 %b) !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.or = or i32 %a, %b, !dbg !31
// CHECK:STDOUT:   ret i32 %int.or, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CTestXor.Main(i32 %a, i32 %b) !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.xor = xor i32 %a, %b, !dbg !34
// CHECK:STDOUT:   ret i32 %int.xor, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CTestLeftShift.Main(i32 %a, i32 %b) !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.left_shift = shl i32 %a, %b, !dbg !37
// CHECK:STDOUT:   ret i32 %int.left_shift, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CTestArithmeticRightShift.Main(i32 %a, i32 %b) !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.right_shift = ashr i32 %a, %b, !dbg !40
// CHECK:STDOUT:   ret i32 %int.right_shift, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CTestLogicalRightShift.Main(i32 %a, i32 %b) !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.right_shift = lshr i32 %a, %b, !dbg !43
// CHECK:STDOUT:   ret i32 %int.right_shift, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @_CTestEq.Main(i32 %a, i32 %b) !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.eq = icmp eq i32 %a, %b, !dbg !46
// CHECK:STDOUT:   ret i1 %int.eq, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @_CTestNeq.Main(i32 %a, i32 %b) !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.neq = icmp ne i32 %a, %b, !dbg !49
// CHECK:STDOUT:   ret i1 %int.neq, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @_CTestLess.Main(i32 %a, i32 %b) !dbg !51 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.less = icmp slt i32 %a, %b, !dbg !52
// CHECK:STDOUT:   ret i1 %int.less, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @_CTestLessEq.Main(i32 %a, i32 %b) !dbg !54 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.less_eq = icmp sle i32 %a, %b, !dbg !55
// CHECK:STDOUT:   ret i1 %int.less_eq, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @_CTestGreater.Main(i32 %a, i32 %b) !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.greater = icmp sgt i32 %a, %b, !dbg !58
// CHECK:STDOUT:   ret i1 %int.greater, !dbg !59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @_CTestGreaterEq.Main(i32 %a, i32 %b) !dbg !60 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.greater_eq = icmp sge i32 %a, %b, !dbg !61
// CHECK:STDOUT:   ret i1 %int.greater_eq, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "basic.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "TestNegate", linkageName: "_CTestNegate.Main", scope: null, file: !3, line: 5, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 5, column: 39, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 5, column: 32, scope: !4)
// CHECK:STDOUT: !9 = distinct !DISubprogram(name: "TestAdd", linkageName: "_CTestAdd.Main", scope: null, file: !3, line: 8, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !10 = !DILocation(line: 8, column: 44, scope: !9)
// CHECK:STDOUT: !11 = !DILocation(line: 8, column: 37, scope: !9)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "TestSub", linkageName: "_CTestSub.Main", scope: null, file: !3, line: 11, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !13 = !DILocation(line: 11, column: 44, scope: !12)
// CHECK:STDOUT: !14 = !DILocation(line: 11, column: 37, scope: !12)
// CHECK:STDOUT: !15 = distinct !DISubprogram(name: "TestMul", linkageName: "_CTestMul.Main", scope: null, file: !3, line: 14, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !16 = !DILocation(line: 14, column: 44, scope: !15)
// CHECK:STDOUT: !17 = !DILocation(line: 14, column: 37, scope: !15)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "TestDiv", linkageName: "_CTestDiv.Main", scope: null, file: !3, line: 17, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !19 = !DILocation(line: 17, column: 44, scope: !18)
// CHECK:STDOUT: !20 = !DILocation(line: 17, column: 37, scope: !18)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "TestMod", linkageName: "_CTestMod.Main", scope: null, file: !3, line: 20, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !22 = !DILocation(line: 20, column: 44, scope: !21)
// CHECK:STDOUT: !23 = !DILocation(line: 20, column: 37, scope: !21)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "TestComplement", linkageName: "_CTestComplement.Main", scope: null, file: !3, line: 23, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !25 = !DILocation(line: 23, column: 43, scope: !24)
// CHECK:STDOUT: !26 = !DILocation(line: 23, column: 36, scope: !24)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "TestAnd", linkageName: "_CTestAnd.Main", scope: null, file: !3, line: 26, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !28 = !DILocation(line: 26, column: 44, scope: !27)
// CHECK:STDOUT: !29 = !DILocation(line: 26, column: 37, scope: !27)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "TestOr", linkageName: "_CTestOr.Main", scope: null, file: !3, line: 29, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !31 = !DILocation(line: 29, column: 43, scope: !30)
// CHECK:STDOUT: !32 = !DILocation(line: 29, column: 36, scope: !30)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "TestXor", linkageName: "_CTestXor.Main", scope: null, file: !3, line: 32, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !34 = !DILocation(line: 32, column: 44, scope: !33)
// CHECK:STDOUT: !35 = !DILocation(line: 32, column: 37, scope: !33)
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "TestLeftShift", linkageName: "_CTestLeftShift.Main", scope: null, file: !3, line: 35, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !37 = !DILocation(line: 35, column: 50, scope: !36)
// CHECK:STDOUT: !38 = !DILocation(line: 35, column: 43, scope: !36)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "TestArithmeticRightShift", linkageName: "_CTestArithmeticRightShift.Main", scope: null, file: !3, line: 38, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !40 = !DILocation(line: 38, column: 61, scope: !39)
// CHECK:STDOUT: !41 = !DILocation(line: 38, column: 54, scope: !39)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "TestLogicalRightShift", linkageName: "_CTestLogicalRightShift.Main", scope: null, file: !3, line: 41, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !43 = !DILocation(line: 41, column: 58, scope: !42)
// CHECK:STDOUT: !44 = !DILocation(line: 41, column: 51, scope: !42)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "TestEq", linkageName: "_CTestEq.Main", scope: null, file: !3, line: 44, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !46 = !DILocation(line: 44, column: 44, scope: !45)
// CHECK:STDOUT: !47 = !DILocation(line: 44, column: 37, scope: !45)
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "TestNeq", linkageName: "_CTestNeq.Main", scope: null, file: !3, line: 47, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !49 = !DILocation(line: 47, column: 45, scope: !48)
// CHECK:STDOUT: !50 = !DILocation(line: 47, column: 38, scope: !48)
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "TestLess", linkageName: "_CTestLess.Main", scope: null, file: !3, line: 50, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !52 = !DILocation(line: 50, column: 46, scope: !51)
// CHECK:STDOUT: !53 = !DILocation(line: 50, column: 39, scope: !51)
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "TestLessEq", linkageName: "_CTestLessEq.Main", scope: null, file: !3, line: 53, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !55 = !DILocation(line: 53, column: 48, scope: !54)
// CHECK:STDOUT: !56 = !DILocation(line: 53, column: 41, scope: !54)
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "TestGreater", linkageName: "_CTestGreater.Main", scope: null, file: !3, line: 56, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !58 = !DILocation(line: 56, column: 49, scope: !57)
// CHECK:STDOUT: !59 = !DILocation(line: 56, column: 42, scope: !57)
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "TestGreaterEq", linkageName: "_CTestGreaterEq.Main", scope: null, file: !3, line: 59, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !61 = !DILocation(line: 59, column: 51, scope: !60)
// CHECK:STDOUT: !62 = !DILocation(line: 59, column: 44, scope: !60)
// CHECK:STDOUT: ; ModuleID = 'mixed_shift.carbon'
// CHECK:STDOUT: source_filename = "mixed_shift.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CTestLeftShiftSmaller.Main(i32 %a, i16 %b) !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.left_shift.rhs = zext i16 %b to i32, !dbg !7
// CHECK:STDOUT:   %int.left_shift = shl i32 %a, %int.left_shift.rhs, !dbg !7
// CHECK:STDOUT:   ret i32 %int.left_shift, !dbg !8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CTestRightShiftSmaller.Main(i32 %a, i16 %b) !dbg !9 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.right_shift.rhs = zext i16 %b to i32, !dbg !10
// CHECK:STDOUT:   %int.right_shift = ashr i32 %a, %int.right_shift.rhs, !dbg !10
// CHECK:STDOUT:   ret i32 %int.right_shift, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i16 @_CTestLeftShiftLargerII.Main(i16 %a, i32 %b) !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.left_shift.rhs = trunc i32 %b to i16, !dbg !13
// CHECK:STDOUT:   %int.left_shift = shl i16 %a, %int.left_shift.rhs, !dbg !13
// CHECK:STDOUT:   ret i16 %int.left_shift, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i16 @_CTestRightShiftLargerII.Main(i16 %a, i32 %b) !dbg !15 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.right_shift.rhs = trunc i32 %b to i16, !dbg !16
// CHECK:STDOUT:   %int.right_shift = ashr i16 %a, %int.right_shift.rhs, !dbg !16
// CHECK:STDOUT:   ret i16 %int.right_shift, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i16 @_CTestLeftShiftLargerIU.Main(i16 %a, i32 %b) !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.left_shift.rhs = trunc i32 %b to i16, !dbg !19
// CHECK:STDOUT:   %int.left_shift = shl i16 %a, %int.left_shift.rhs, !dbg !19
// CHECK:STDOUT:   ret i16 %int.left_shift, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i16 @_CTestRightShiftLargerIU.Main(i16 %a, i32 %b) !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.right_shift.rhs = trunc i32 %b to i16, !dbg !22
// CHECK:STDOUT:   %int.right_shift = ashr i16 %a, %int.right_shift.rhs, !dbg !22
// CHECK:STDOUT:   ret i16 %int.right_shift, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i16 @_CTestLeftShiftLargerUI.Main(i16 %a, i32 %b) !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.left_shift.rhs = trunc i32 %b to i16, !dbg !25
// CHECK:STDOUT:   %int.left_shift = shl i16 %a, %int.left_shift.rhs, !dbg !25
// CHECK:STDOUT:   ret i16 %int.left_shift, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i16 @_CTestRightShiftLargerUI.Main(i16 %a, i32 %b) !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.right_shift.rhs = trunc i32 %b to i16, !dbg !28
// CHECK:STDOUT:   %int.right_shift = lshr i16 %a, %int.right_shift.rhs, !dbg !28
// CHECK:STDOUT:   ret i16 %int.right_shift, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i16 @_CTestLeftShiftLargerUU.Main(i16 %a, i32 %b) !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.left_shift.rhs = trunc i32 %b to i16, !dbg !31
// CHECK:STDOUT:   %int.left_shift = shl i16 %a, %int.left_shift.rhs, !dbg !31
// CHECK:STDOUT:   ret i16 %int.left_shift, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i16 @_CTestRightShiftLargerUU.Main(i16 %a, i32 %b) !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.right_shift.rhs = trunc i32 %b to i16, !dbg !34
// CHECK:STDOUT:   %int.right_shift = lshr i16 %a, %int.right_shift.rhs, !dbg !34
// CHECK:STDOUT:   ret i16 %int.right_shift, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "mixed_shift.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "TestLeftShiftSmaller", linkageName: "_CTestLeftShiftSmaller.Main", scope: null, file: !3, line: 5, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 5, column: 57, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 5, column: 50, scope: !4)
// CHECK:STDOUT: !9 = distinct !DISubprogram(name: "TestRightShiftSmaller", linkageName: "_CTestRightShiftSmaller.Main", scope: null, file: !3, line: 8, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !10 = !DILocation(line: 8, column: 58, scope: !9)
// CHECK:STDOUT: !11 = !DILocation(line: 8, column: 51, scope: !9)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "TestLeftShiftLargerII", linkageName: "_CTestLeftShiftLargerII.Main", scope: null, file: !3, line: 11, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !13 = !DILocation(line: 11, column: 58, scope: !12)
// CHECK:STDOUT: !14 = !DILocation(line: 11, column: 51, scope: !12)
// CHECK:STDOUT: !15 = distinct !DISubprogram(name: "TestRightShiftLargerII", linkageName: "_CTestRightShiftLargerII.Main", scope: null, file: !3, line: 14, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !16 = !DILocation(line: 14, column: 59, scope: !15)
// CHECK:STDOUT: !17 = !DILocation(line: 14, column: 52, scope: !15)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "TestLeftShiftLargerIU", linkageName: "_CTestLeftShiftLargerIU.Main", scope: null, file: !3, line: 17, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !19 = !DILocation(line: 17, column: 58, scope: !18)
// CHECK:STDOUT: !20 = !DILocation(line: 17, column: 51, scope: !18)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "TestRightShiftLargerIU", linkageName: "_CTestRightShiftLargerIU.Main", scope: null, file: !3, line: 20, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !22 = !DILocation(line: 20, column: 59, scope: !21)
// CHECK:STDOUT: !23 = !DILocation(line: 20, column: 52, scope: !21)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "TestLeftShiftLargerUI", linkageName: "_CTestLeftShiftLargerUI.Main", scope: null, file: !3, line: 23, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !25 = !DILocation(line: 23, column: 58, scope: !24)
// CHECK:STDOUT: !26 = !DILocation(line: 23, column: 51, scope: !24)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "TestRightShiftLargerUI", linkageName: "_CTestRightShiftLargerUI.Main", scope: null, file: !3, line: 26, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !28 = !DILocation(line: 26, column: 59, scope: !27)
// CHECK:STDOUT: !29 = !DILocation(line: 26, column: 52, scope: !27)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "TestLeftShiftLargerUU", linkageName: "_CTestLeftShiftLargerUU.Main", scope: null, file: !3, line: 29, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !31 = !DILocation(line: 29, column: 58, scope: !30)
// CHECK:STDOUT: !32 = !DILocation(line: 29, column: 51, scope: !30)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "TestRightShiftLargerUU", linkageName: "_CTestRightShiftLargerUU.Main", scope: null, file: !3, line: 32, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !34 = !DILocation(line: 32, column: 59, scope: !33)
// CHECK:STDOUT: !35 = !DILocation(line: 32, column: 52, scope: !33)
// CHECK:STDOUT: ; ModuleID = 'mixed_compare.carbon'
// CHECK:STDOUT: source_filename = "mixed_compare.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @_CTestEq_u16_u32.Main(i16 %a, i32 %b) !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.eq.lhs = zext i16 %a to i32, !dbg !7
// CHECK:STDOUT:   %int.eq = icmp eq i32 %int.eq.lhs, %b, !dbg !7
// CHECK:STDOUT:   ret i1 %int.eq, !dbg !8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @_CTestEq_i16_u32.Main(i16 %a, i32 %b) !dbg !9 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.eq.lhs = sext i16 %a to i33, !dbg !10
// CHECK:STDOUT:   %int.eq.rhs = zext i32 %b to i33, !dbg !10
// CHECK:STDOUT:   %int.eq = icmp eq i33 %int.eq.lhs, %int.eq.rhs, !dbg !10
// CHECK:STDOUT:   ret i1 %int.eq, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @_CTestEq_u16_i32.Main(i16 %a, i32 %b) !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.eq.lhs = zext i16 %a to i32, !dbg !13
// CHECK:STDOUT:   %int.eq = icmp eq i32 %int.eq.lhs, %b, !dbg !13
// CHECK:STDOUT:   ret i1 %int.eq, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @_CTestEq_i16_i32.Main(i16 %a, i32 %b) !dbg !15 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.eq.lhs = sext i16 %a to i32, !dbg !16
// CHECK:STDOUT:   %int.eq = icmp eq i32 %int.eq.lhs, %b, !dbg !16
// CHECK:STDOUT:   ret i1 %int.eq, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @_CTestEq_i32_u32.Main(i32 %a, i32 %b) !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.eq.lhs = sext i32 %a to i33, !dbg !19
// CHECK:STDOUT:   %int.eq.rhs = zext i32 %b to i33, !dbg !19
// CHECK:STDOUT:   %int.eq = icmp eq i33 %int.eq.lhs, %int.eq.rhs, !dbg !19
// CHECK:STDOUT:   ret i1 %int.eq, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @_CTestLess_u16_u32.Main(i16 %a, i32 %b) !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.less.lhs = zext i16 %a to i32, !dbg !22
// CHECK:STDOUT:   %int.less = icmp ult i32 %int.less.lhs, %b, !dbg !22
// CHECK:STDOUT:   ret i1 %int.less, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @_CTestLess_i16_u32.Main(i16 %a, i32 %b) !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.less.lhs = sext i16 %a to i33, !dbg !25
// CHECK:STDOUT:   %int.less.rhs = zext i32 %b to i33, !dbg !25
// CHECK:STDOUT:   %int.less = icmp slt i33 %int.less.lhs, %int.less.rhs, !dbg !25
// CHECK:STDOUT:   ret i1 %int.less, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @_CTestLess_u16_i32.Main(i16 %a, i32 %b) !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.less.lhs = zext i16 %a to i32, !dbg !28
// CHECK:STDOUT:   %int.less = icmp slt i32 %int.less.lhs, %b, !dbg !28
// CHECK:STDOUT:   ret i1 %int.less, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @_CTestLess_i16_i32.Main(i16 %a, i32 %b) !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.less.lhs = sext i16 %a to i32, !dbg !31
// CHECK:STDOUT:   %int.less = icmp slt i32 %int.less.lhs, %b, !dbg !31
// CHECK:STDOUT:   ret i1 %int.less, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @_CTestLess_i32_u32.Main(i32 %a, i32 %b) !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.less.lhs = sext i32 %a to i33, !dbg !34
// CHECK:STDOUT:   %int.less.rhs = zext i32 %b to i33, !dbg !34
// CHECK:STDOUT:   %int.less = icmp slt i33 %int.less.lhs, %int.less.rhs, !dbg !34
// CHECK:STDOUT:   ret i1 %int.less, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "mixed_compare.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "TestEq_u16_u32", linkageName: "_CTestEq_u16_u32.Main", scope: null, file: !3, line: 10, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 10, column: 52, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 10, column: 45, scope: !4)
// CHECK:STDOUT: !9 = distinct !DISubprogram(name: "TestEq_i16_u32", linkageName: "_CTestEq_i16_u32.Main", scope: null, file: !3, line: 11, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !10 = !DILocation(line: 11, column: 52, scope: !9)
// CHECK:STDOUT: !11 = !DILocation(line: 11, column: 45, scope: !9)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "TestEq_u16_i32", linkageName: "_CTestEq_u16_i32.Main", scope: null, file: !3, line: 12, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !13 = !DILocation(line: 12, column: 52, scope: !12)
// CHECK:STDOUT: !14 = !DILocation(line: 12, column: 45, scope: !12)
// CHECK:STDOUT: !15 = distinct !DISubprogram(name: "TestEq_i16_i32", linkageName: "_CTestEq_i16_i32.Main", scope: null, file: !3, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !16 = !DILocation(line: 13, column: 52, scope: !15)
// CHECK:STDOUT: !17 = !DILocation(line: 13, column: 45, scope: !15)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "TestEq_i32_u32", linkageName: "_CTestEq_i32_u32.Main", scope: null, file: !3, line: 14, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !19 = !DILocation(line: 14, column: 52, scope: !18)
// CHECK:STDOUT: !20 = !DILocation(line: 14, column: 45, scope: !18)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "TestLess_u16_u32", linkageName: "_CTestLess_u16_u32.Main", scope: null, file: !3, line: 22, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !22 = !DILocation(line: 22, column: 54, scope: !21)
// CHECK:STDOUT: !23 = !DILocation(line: 22, column: 47, scope: !21)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "TestLess_i16_u32", linkageName: "_CTestLess_i16_u32.Main", scope: null, file: !3, line: 23, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !25 = !DILocation(line: 23, column: 54, scope: !24)
// CHECK:STDOUT: !26 = !DILocation(line: 23, column: 47, scope: !24)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "TestLess_u16_i32", linkageName: "_CTestLess_u16_i32.Main", scope: null, file: !3, line: 24, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !28 = !DILocation(line: 24, column: 54, scope: !27)
// CHECK:STDOUT: !29 = !DILocation(line: 24, column: 47, scope: !27)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "TestLess_i16_i32", linkageName: "_CTestLess_i16_i32.Main", scope: null, file: !3, line: 25, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !31 = !DILocation(line: 25, column: 54, scope: !30)
// CHECK:STDOUT: !32 = !DILocation(line: 25, column: 47, scope: !30)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "TestLess_i32_u32", linkageName: "_CTestLess_i32_u32.Main", scope: null, file: !3, line: 26, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !34 = !DILocation(line: 26, column: 54, scope: !33)
// CHECK:STDOUT: !35 = !DILocation(line: 26, column: 47, scope: !33)
// CHECK:STDOUT: ; ModuleID = 'convert.carbon'
// CHECK:STDOUT: source_filename = "convert.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CTestInt32ToInt32.Main(i32 %a) !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %a, !dbg !7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CTestInt32ToUint32.Main(i32 %a) !dbg !8 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %a, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CTestUint32ToInt32.Main(i32 %a) !dbg !10 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %a, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CTestUint32ToUint32.Main(i32 %a) !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 %a, !dbg !13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i16 @_CTestInt32ToInt16.Main(i32 %a) !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.convert = trunc i32 %a to i16, !dbg !15
// CHECK:STDOUT:   ret i16 %int.convert, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i16 @_CTestInt32ToUint16.Main(i32 %a) !dbg !17 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.convert = trunc i32 %a to i16, !dbg !18
// CHECK:STDOUT:   ret i16 %int.convert, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i16 @_CTestUint32ToInt16.Main(i32 %a) !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.convert = trunc i32 %a to i16, !dbg !21
// CHECK:STDOUT:   ret i16 %int.convert, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i16 @_CTestUint32ToUint16.Main(i32 %a) !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.convert = trunc i32 %a to i16, !dbg !24
// CHECK:STDOUT:   ret i16 %int.convert, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i64 @_CTestInt32ToInt64.Main(i32 %a) !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.convert = sext i32 %a to i64, !dbg !27
// CHECK:STDOUT:   ret i64 %int.convert, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i64 @_CTestInt32ToUint64.Main(i32 %a) !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.convert = sext i32 %a to i64, !dbg !30
// CHECK:STDOUT:   ret i64 %int.convert, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i64 @_CTestUint32ToInt64.Main(i32 %a) !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.convert = zext i32 %a to i64, !dbg !33
// CHECK:STDOUT:   ret i64 %int.convert, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i64 @_CTestUint32ToUint64.Main(i32 %a) !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.convert = zext i32 %a to i64, !dbg !36
// CHECK:STDOUT:   ret i64 %int.convert, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "convert.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "TestInt32ToInt32", linkageName: "_CTestInt32ToInt32.Main", scope: null, file: !3, line: 10, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 10, column: 38, scope: !4)
// CHECK:STDOUT: !8 = distinct !DISubprogram(name: "TestInt32ToUint32", linkageName: "_CTestInt32ToUint32.Main", scope: null, file: !3, line: 11, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !9 = !DILocation(line: 11, column: 39, scope: !8)
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "TestUint32ToInt32", linkageName: "_CTestUint32ToInt32.Main", scope: null, file: !3, line: 12, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !11 = !DILocation(line: 12, column: 39, scope: !10)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "TestUint32ToUint32", linkageName: "_CTestUint32ToUint32.Main", scope: null, file: !3, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !13 = !DILocation(line: 13, column: 40, scope: !12)
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "TestInt32ToInt16", linkageName: "_CTestInt32ToInt16.Main", scope: null, file: !3, line: 21, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !15 = !DILocation(line: 21, column: 45, scope: !14)
// CHECK:STDOUT: !16 = !DILocation(line: 21, column: 38, scope: !14)
// CHECK:STDOUT: !17 = distinct !DISubprogram(name: "TestInt32ToUint16", linkageName: "_CTestInt32ToUint16.Main", scope: null, file: !3, line: 22, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !18 = !DILocation(line: 22, column: 46, scope: !17)
// CHECK:STDOUT: !19 = !DILocation(line: 22, column: 39, scope: !17)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "TestUint32ToInt16", linkageName: "_CTestUint32ToInt16.Main", scope: null, file: !3, line: 23, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !21 = !DILocation(line: 23, column: 46, scope: !20)
// CHECK:STDOUT: !22 = !DILocation(line: 23, column: 39, scope: !20)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "TestUint32ToUint16", linkageName: "_CTestUint32ToUint16.Main", scope: null, file: !3, line: 24, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !24 = !DILocation(line: 24, column: 47, scope: !23)
// CHECK:STDOUT: !25 = !DILocation(line: 24, column: 40, scope: !23)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "TestInt32ToInt64", linkageName: "_CTestInt32ToInt64.Main", scope: null, file: !3, line: 32, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !27 = !DILocation(line: 32, column: 45, scope: !26)
// CHECK:STDOUT: !28 = !DILocation(line: 32, column: 38, scope: !26)
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "TestInt32ToUint64", linkageName: "_CTestInt32ToUint64.Main", scope: null, file: !3, line: 33, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !30 = !DILocation(line: 33, column: 46, scope: !29)
// CHECK:STDOUT: !31 = !DILocation(line: 33, column: 39, scope: !29)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "TestUint32ToInt64", linkageName: "_CTestUint32ToInt64.Main", scope: null, file: !3, line: 34, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !33 = !DILocation(line: 34, column: 46, scope: !32)
// CHECK:STDOUT: !34 = !DILocation(line: 34, column: 39, scope: !32)
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "TestUint32ToUint64", linkageName: "_CTestUint32ToUint64.Main", scope: null, file: !3, line: 35, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !36 = !DILocation(line: 35, column: 47, scope: !35)
// CHECK:STDOUT: !37 = !DILocation(line: 35, column: 40, scope: !35)
