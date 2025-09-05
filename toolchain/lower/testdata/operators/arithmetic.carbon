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

// CHECK:STDOUT: ; ModuleID = 'arithmetic.carbon'
// CHECK:STDOUT: source_filename = "arithmetic.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_Cadd_i32.Main(i32 %a, i32 %b) !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %a, %b, !dbg !7
// CHECK:STDOUT:   ret i32 %Int.as.AddWith.impl.Op.call, !dbg !8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_Cadd_u32.Main(i32 %a, i32 %b) !dbg !9 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %UInt.as.AddWith.impl.Op.call = add i32 %a, %b, !dbg !10
// CHECK:STDOUT:   ret i32 %UInt.as.AddWith.impl.Op.call, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_Cdiv_i32.Main(i32 %a, i32 %b) !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.DivWith.impl.Op.call = sdiv i32 %a, %b, !dbg !13
// CHECK:STDOUT:   ret i32 %Int.as.DivWith.impl.Op.call, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_Cdiv_u32.Main(i32 %a, i32 %b) !dbg !15 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %UInt.as.DivWith.impl.Op.call = udiv i32 %a, %b, !dbg !16
// CHECK:STDOUT:   ret i32 %UInt.as.DivWith.impl.Op.call, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_Cmod_i32.Main(i32 %a, i32 %b) !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %a, %b, !dbg !19
// CHECK:STDOUT:   ret i32 %Int.as.ModWith.impl.Op.call, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_Cmod_u32.Main(i32 %a, i32 %b) !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %UInt.as.ModWith.impl.Op.call = urem i32 %a, %b, !dbg !22
// CHECK:STDOUT:   ret i32 %UInt.as.ModWith.impl.Op.call, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_Cmul_i32.Main(i32 %a, i32 %b) !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.MulWith.impl.Op.call = mul i32 %a, %b, !dbg !25
// CHECK:STDOUT:   ret i32 %Int.as.MulWith.impl.Op.call, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_Cmul_u32.Main(i32 %a, i32 %b) !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %UInt.as.MulWith.impl.Op.call = mul i32 %a, %b, !dbg !28
// CHECK:STDOUT:   ret i32 %UInt.as.MulWith.impl.Op.call, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_Cneg_i32.Main(i32 %a) !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.call = sub i32 0, %a, !dbg !31
// CHECK:STDOUT:   ret i32 %Int.as.Negate.impl.Op.call, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_Cneg_u32.Main(i32 %a) !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %UInt.as.Negate.impl.Op.call = sub i32 0, %a, !dbg !34
// CHECK:STDOUT:   ret i32 %UInt.as.Negate.impl.Op.call, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_Csub_i32.Main(i32 %a, i32 %b) !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.SubWith.impl.Op.call = sub i32 %a, %b, !dbg !37
// CHECK:STDOUT:   ret i32 %Int.as.SubWith.impl.Op.call, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_Csub_u32.Main(i32 %a, i32 %b) !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %UInt.as.SubWith.impl.Op.call = sub i32 %a, %b, !dbg !40
// CHECK:STDOUT:   ret i32 %UInt.as.SubWith.impl.Op.call, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i16 @_Cdiv_i16.Main(i16 %a, i16 %b) !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.DivWith.impl.Op.call = sdiv i16 %a, %b, !dbg !43
// CHECK:STDOUT:   ret i16 %Int.as.DivWith.impl.Op.call, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i16 @_Cdiv_u16.Main(i16 %a, i16 %b) !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %UInt.as.DivWith.impl.Op.call = udiv i16 %a, %b, !dbg !46
// CHECK:STDOUT:   ret i16 %UInt.as.DivWith.impl.Op.call, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "arithmetic.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "add_i32", linkageName: "_Cadd_i32.Main", scope: null, file: !3, line: 16, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 16, column: 44, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 16, column: 37, scope: !4)
// CHECK:STDOUT: !9 = distinct !DISubprogram(name: "add_u32", linkageName: "_Cadd_u32.Main", scope: null, file: !3, line: 17, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !10 = !DILocation(line: 17, column: 44, scope: !9)
// CHECK:STDOUT: !11 = !DILocation(line: 17, column: 37, scope: !9)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "div_i32", linkageName: "_Cdiv_i32.Main", scope: null, file: !3, line: 19, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !13 = !DILocation(line: 19, column: 44, scope: !12)
// CHECK:STDOUT: !14 = !DILocation(line: 19, column: 37, scope: !12)
// CHECK:STDOUT: !15 = distinct !DISubprogram(name: "div_u32", linkageName: "_Cdiv_u32.Main", scope: null, file: !3, line: 20, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !16 = !DILocation(line: 20, column: 44, scope: !15)
// CHECK:STDOUT: !17 = !DILocation(line: 20, column: 37, scope: !15)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "mod_i32", linkageName: "_Cmod_i32.Main", scope: null, file: !3, line: 22, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !19 = !DILocation(line: 22, column: 44, scope: !18)
// CHECK:STDOUT: !20 = !DILocation(line: 22, column: 37, scope: !18)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "mod_u32", linkageName: "_Cmod_u32.Main", scope: null, file: !3, line: 23, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !22 = !DILocation(line: 23, column: 44, scope: !21)
// CHECK:STDOUT: !23 = !DILocation(line: 23, column: 37, scope: !21)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "mul_i32", linkageName: "_Cmul_i32.Main", scope: null, file: !3, line: 25, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !25 = !DILocation(line: 25, column: 44, scope: !24)
// CHECK:STDOUT: !26 = !DILocation(line: 25, column: 37, scope: !24)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "mul_u32", linkageName: "_Cmul_u32.Main", scope: null, file: !3, line: 26, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !28 = !DILocation(line: 26, column: 44, scope: !27)
// CHECK:STDOUT: !29 = !DILocation(line: 26, column: 37, scope: !27)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "neg_i32", linkageName: "_Cneg_i32.Main", scope: null, file: !3, line: 28, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !31 = !DILocation(line: 28, column: 36, scope: !30)
// CHECK:STDOUT: !32 = !DILocation(line: 28, column: 29, scope: !30)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "neg_u32", linkageName: "_Cneg_u32.Main", scope: null, file: !3, line: 29, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !34 = !DILocation(line: 29, column: 36, scope: !33)
// CHECK:STDOUT: !35 = !DILocation(line: 29, column: 29, scope: !33)
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "sub_i32", linkageName: "_Csub_i32.Main", scope: null, file: !3, line: 31, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !37 = !DILocation(line: 31, column: 44, scope: !36)
// CHECK:STDOUT: !38 = !DILocation(line: 31, column: 37, scope: !36)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "sub_u32", linkageName: "_Csub_u32.Main", scope: null, file: !3, line: 32, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !40 = !DILocation(line: 32, column: 44, scope: !39)
// CHECK:STDOUT: !41 = !DILocation(line: 32, column: 37, scope: !39)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "div_i16", linkageName: "_Cdiv_i16.Main", scope: null, file: !3, line: 36, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !43 = !DILocation(line: 36, column: 44, scope: !42)
// CHECK:STDOUT: !44 = !DILocation(line: 36, column: 37, scope: !42)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "div_u16", linkageName: "_Cdiv_u16.Main", scope: null, file: !3, line: 37, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !46 = !DILocation(line: 37, column: 44, scope: !45)
// CHECK:STDOUT: !47 = !DILocation(line: 37, column: 37, scope: !45)
