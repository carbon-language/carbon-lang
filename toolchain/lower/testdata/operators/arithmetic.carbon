// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/operators/arithmetic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/operators/arithmetic.carbon

// Test each of the arithmetic operations, signed and unsigned values may lower
// differently.

fn add_i32(a: i32, b: i32) -> i32 { return a + b; }
fn add_u32(a: u32, b: u32) -> u32 { return a + b; }

fn div_i32(a: i32, b: i32) -> i32 { return a / b; }
fn div_u32(a: u32, b: u32) -> u32 { return a / b; }

fn mod_i32(a: i32, b: i32) -> i32 { return a % b; }
fn mod_u32(a: u32, b: u32) -> u32 { return a % b; }

fn mul_i32(a: i32, b: i32) -> i32 { return a * b; }
fn mul_u32(a: u32, b: u32) -> u32 { return a * b; }

fn neg_i32(a: i32) -> i32 { return -a; }
fn neg_u32(a: u32) -> u32 { return -a; }

fn sub_i32(a: i32, b: i32) -> i32 { return a - b; }
fn sub_u32(a: u32, b: u32) -> u32 { return a - b; }

// One test for non-32 bit, verify the size is correct.

fn div_i16(a: i16, b: i16) -> i16 { return a / b; }
fn div_u16(a: u16, b: u16) -> u16 { return a / b; }

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'arithmetic.carbon'
// CHECK:STDOUT: source_filename = "arithmetic.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_Cadd_i32.Main(i32 %a, i32 %b) #0 !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %a, %b, !dbg !29
// CHECK:STDOUT:   ret i32 %Int.as.AddWith.impl.Op.call, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_Cadd_u32.Main(i32 %a, i32 %b) #0 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %UInt.as.AddWith.impl.Op.call = add i32 %a, %b, !dbg !40
// CHECK:STDOUT:   ret i32 %UInt.as.AddWith.impl.Op.call, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_Cdiv_i32.Main(i32 %a, i32 %b) #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.DivWith.impl.Op.call = sdiv i32 %a, %b, !dbg !51
// CHECK:STDOUT:   ret i32 %Int.as.DivWith.impl.Op.call, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_Cdiv_u32.Main(i32 %a, i32 %b) #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %UInt.as.DivWith.impl.Op.call = udiv i32 %a, %b, !dbg !62
// CHECK:STDOUT:   ret i32 %UInt.as.DivWith.impl.Op.call, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_Cmod_i32.Main(i32 %a, i32 %b) #0 !dbg !66 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %a, %b, !dbg !73
// CHECK:STDOUT:   ret i32 %Int.as.ModWith.impl.Op.call, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_Cmod_u32.Main(i32 %a, i32 %b) #0 !dbg !77 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %UInt.as.ModWith.impl.Op.call = urem i32 %a, %b, !dbg !84
// CHECK:STDOUT:   ret i32 %UInt.as.ModWith.impl.Op.call, !dbg !86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_Cmul_i32.Main(i32 %a, i32 %b) #0 !dbg !88 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.MulWith.impl.Op.call = mul i32 %a, %b, !dbg !95
// CHECK:STDOUT:   ret i32 %Int.as.MulWith.impl.Op.call, !dbg !97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_Cmul_u32.Main(i32 %a, i32 %b) #0 !dbg !99 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %UInt.as.MulWith.impl.Op.call = mul i32 %a, %b, !dbg !106
// CHECK:STDOUT:   ret i32 %UInt.as.MulWith.impl.Op.call, !dbg !108
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_Cneg_i32.Main(i32 %a) #0 !dbg !110 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.call = sub i32 0, %a, !dbg !116
// CHECK:STDOUT:   ret i32 %Int.as.Negate.impl.Op.call, !dbg !117
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_Cneg_u32.Main(i32 %a) #0 !dbg !119 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %UInt.as.Negate.impl.Op.call = sub i32 0, %a, !dbg !125
// CHECK:STDOUT:   ret i32 %UInt.as.Negate.impl.Op.call, !dbg !126
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_Csub_i32.Main(i32 %a, i32 %b) #0 !dbg !128 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.SubWith.impl.Op.call = sub i32 %a, %b, !dbg !135
// CHECK:STDOUT:   ret i32 %Int.as.SubWith.impl.Op.call, !dbg !137
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_Csub_u32.Main(i32 %a, i32 %b) #0 !dbg !139 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %UInt.as.SubWith.impl.Op.call = sub i32 %a, %b, !dbg !146
// CHECK:STDOUT:   ret i32 %UInt.as.SubWith.impl.Op.call, !dbg !148
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i16 @_Cdiv_i16.Main(i16 %a, i16 %b) #0 !dbg !150 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.DivWith.impl.Op.call = sdiv i16 %a, %b, !dbg !157
// CHECK:STDOUT:   ret i16 %Int.as.DivWith.impl.Op.call, !dbg !159
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i16 @_Cdiv_u16.Main(i16 %a, i16 %b) #0 !dbg !161 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %UInt.as.DivWith.impl.Op.call = udiv i16 %a, %b, !dbg !168
// CHECK:STDOUT:   ret i16 %UInt.as.DivWith.impl.Op.call, !dbg !170
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "arithmetic.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !8 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !9 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !10 = !{!6, !6, !6}
// CHECK:STDOUT: !11 = !DISubroutineType(types: !10)
// CHECK:STDOUT: !12 = !{!7, !7, !7}
// CHECK:STDOUT: !13 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !14 = !{!6, !6}
// CHECK:STDOUT: !15 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !16 = !{!7, !7}
// CHECK:STDOUT: !17 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !18 = !{!8, !8, !8}
// CHECK:STDOUT: !19 = !DISubroutineType(types: !18)
// CHECK:STDOUT: !20 = !{!9, !9, !9}
// CHECK:STDOUT: !21 = !DISubroutineType(types: !20)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "add_i32", linkageName: "_Cadd_i32.Main", scope: null, file: !4, line: 16, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !32)
// CHECK:STDOUT: !23 = !DILocalVariable(arg: 1, scope: !22, type: !6)
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 2, scope: !22, type: !6)
// CHECK:STDOUT: !29 = !DILocation(line: 16, column: 44, scope: !22)
// CHECK:STDOUT: !31 = !DILocation(line: 16, column: 37, scope: !22)
// CHECK:STDOUT: !32 = !{!23, !24}
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "add_u32", linkageName: "_Cadd_u32.Main", scope: null, file: !4, line: 17, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !43)
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 1, scope: !33, type: !7)
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 2, scope: !33, type: !7)
// CHECK:STDOUT: !40 = !DILocation(line: 17, column: 44, scope: !33)
// CHECK:STDOUT: !42 = !DILocation(line: 17, column: 37, scope: !33)
// CHECK:STDOUT: !43 = !{!34, !35}
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "div_i32", linkageName: "_Cdiv_i32.Main", scope: null, file: !4, line: 19, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !54)
// CHECK:STDOUT: !45 = !DILocalVariable(arg: 1, scope: !44, type: !6)
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 2, scope: !44, type: !6)
// CHECK:STDOUT: !51 = !DILocation(line: 19, column: 44, scope: !44)
// CHECK:STDOUT: !53 = !DILocation(line: 19, column: 37, scope: !44)
// CHECK:STDOUT: !54 = !{!45, !46}
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "div_u32", linkageName: "_Cdiv_u32.Main", scope: null, file: !4, line: 20, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !65)
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !55, type: !7)
// CHECK:STDOUT: !57 = !DILocalVariable(arg: 2, scope: !55, type: !7)
// CHECK:STDOUT: !62 = !DILocation(line: 20, column: 44, scope: !55)
// CHECK:STDOUT: !64 = !DILocation(line: 20, column: 37, scope: !55)
// CHECK:STDOUT: !65 = !{!56, !57}
// CHECK:STDOUT: !66 = distinct !DISubprogram(name: "mod_i32", linkageName: "_Cmod_i32.Main", scope: null, file: !4, line: 22, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !76)
// CHECK:STDOUT: !67 = !DILocalVariable(arg: 1, scope: !66, type: !6)
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 2, scope: !66, type: !6)
// CHECK:STDOUT: !73 = !DILocation(line: 22, column: 44, scope: !66)
// CHECK:STDOUT: !75 = !DILocation(line: 22, column: 37, scope: !66)
// CHECK:STDOUT: !76 = !{!67, !68}
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "mod_u32", linkageName: "_Cmod_u32.Main", scope: null, file: !4, line: 23, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !87)
// CHECK:STDOUT: !78 = !DILocalVariable(arg: 1, scope: !77, type: !7)
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 2, scope: !77, type: !7)
// CHECK:STDOUT: !84 = !DILocation(line: 23, column: 44, scope: !77)
// CHECK:STDOUT: !86 = !DILocation(line: 23, column: 37, scope: !77)
// CHECK:STDOUT: !87 = !{!78, !79}
// CHECK:STDOUT: !88 = distinct !DISubprogram(name: "mul_i32", linkageName: "_Cmul_i32.Main", scope: null, file: !4, line: 25, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !98)
// CHECK:STDOUT: !89 = !DILocalVariable(arg: 1, scope: !88, type: !6)
// CHECK:STDOUT: !90 = !DILocalVariable(arg: 2, scope: !88, type: !6)
// CHECK:STDOUT: !95 = !DILocation(line: 25, column: 44, scope: !88)
// CHECK:STDOUT: !97 = !DILocation(line: 25, column: 37, scope: !88)
// CHECK:STDOUT: !98 = !{!89, !90}
// CHECK:STDOUT: !99 = distinct !DISubprogram(name: "mul_u32", linkageName: "_Cmul_u32.Main", scope: null, file: !4, line: 26, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !109)
// CHECK:STDOUT: !100 = !DILocalVariable(arg: 1, scope: !99, type: !7)
// CHECK:STDOUT: !101 = !DILocalVariable(arg: 2, scope: !99, type: !7)
// CHECK:STDOUT: !106 = !DILocation(line: 26, column: 44, scope: !99)
// CHECK:STDOUT: !108 = !DILocation(line: 26, column: 37, scope: !99)
// CHECK:STDOUT: !109 = !{!100, !101}
// CHECK:STDOUT: !110 = distinct !DISubprogram(name: "neg_i32", linkageName: "_Cneg_i32.Main", scope: null, file: !4, line: 28, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !118)
// CHECK:STDOUT: !111 = !DILocalVariable(arg: 1, scope: !110, type: !6)
// CHECK:STDOUT: !116 = !DILocation(line: 28, column: 36, scope: !110)
// CHECK:STDOUT: !117 = !DILocation(line: 28, column: 29, scope: !110)
// CHECK:STDOUT: !118 = !{!111}
// CHECK:STDOUT: !119 = distinct !DISubprogram(name: "neg_u32", linkageName: "_Cneg_u32.Main", scope: null, file: !4, line: 29, type: !17, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !127)
// CHECK:STDOUT: !120 = !DILocalVariable(arg: 1, scope: !119, type: !7)
// CHECK:STDOUT: !125 = !DILocation(line: 29, column: 36, scope: !119)
// CHECK:STDOUT: !126 = !DILocation(line: 29, column: 29, scope: !119)
// CHECK:STDOUT: !127 = !{!120}
// CHECK:STDOUT: !128 = distinct !DISubprogram(name: "sub_i32", linkageName: "_Csub_i32.Main", scope: null, file: !4, line: 31, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !138)
// CHECK:STDOUT: !129 = !DILocalVariable(arg: 1, scope: !128, type: !6)
// CHECK:STDOUT: !130 = !DILocalVariable(arg: 2, scope: !128, type: !6)
// CHECK:STDOUT: !135 = !DILocation(line: 31, column: 44, scope: !128)
// CHECK:STDOUT: !137 = !DILocation(line: 31, column: 37, scope: !128)
// CHECK:STDOUT: !138 = !{!129, !130}
// CHECK:STDOUT: !139 = distinct !DISubprogram(name: "sub_u32", linkageName: "_Csub_u32.Main", scope: null, file: !4, line: 32, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !149)
// CHECK:STDOUT: !140 = !DILocalVariable(arg: 1, scope: !139, type: !7)
// CHECK:STDOUT: !141 = !DILocalVariable(arg: 2, scope: !139, type: !7)
// CHECK:STDOUT: !146 = !DILocation(line: 32, column: 44, scope: !139)
// CHECK:STDOUT: !148 = !DILocation(line: 32, column: 37, scope: !139)
// CHECK:STDOUT: !149 = !{!140, !141}
// CHECK:STDOUT: !150 = distinct !DISubprogram(name: "div_i16", linkageName: "_Cdiv_i16.Main", scope: null, file: !4, line: 36, type: !19, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !160)
// CHECK:STDOUT: !151 = !DILocalVariable(arg: 1, scope: !150, type: !8)
// CHECK:STDOUT: !152 = !DILocalVariable(arg: 2, scope: !150, type: !8)
// CHECK:STDOUT: !157 = !DILocation(line: 36, column: 44, scope: !150)
// CHECK:STDOUT: !159 = !DILocation(line: 36, column: 37, scope: !150)
// CHECK:STDOUT: !160 = !{!151, !152}
// CHECK:STDOUT: !161 = distinct !DISubprogram(name: "div_u16", linkageName: "_Cdiv_u16.Main", scope: null, file: !4, line: 37, type: !21, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !171)
// CHECK:STDOUT: !162 = !DILocalVariable(arg: 1, scope: !161, type: !9)
// CHECK:STDOUT: !163 = !DILocalVariable(arg: 2, scope: !161, type: !9)
// CHECK:STDOUT: !168 = !DILocation(line: 37, column: 44, scope: !161)
// CHECK:STDOUT: !170 = !DILocation(line: 37, column: 37, scope: !161)
// CHECK:STDOUT: !171 = !{!162, !163}
// CHECK:STDOUT:
