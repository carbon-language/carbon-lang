// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// CHECK:STDOUT:   %int.sadd = add i32 %a, %b, !dbg !7
// CHECK:STDOUT:   ret i32 %int.sadd, !dbg !8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_Cadd_u32.Main(i32 %a, i32 %b) !dbg !9 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.uadd = add i32 %a, %b, !dbg !10
// CHECK:STDOUT:   ret i32 %int.uadd, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_Cdiv_i32.Main(i32 %a, i32 %b) !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.sdiv = sdiv i32 %a, %b, !dbg !13
// CHECK:STDOUT:   ret i32 %int.sdiv, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_Cdiv_u32.Main(i32 %a, i32 %b) !dbg !15 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.udiv = udiv i32 %a, %b, !dbg !16
// CHECK:STDOUT:   ret i32 %int.udiv, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_Cmod_i32.Main(i32 %a, i32 %b) !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.smod = srem i32 %a, %b, !dbg !19
// CHECK:STDOUT:   ret i32 %int.smod, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_Cmod_u32.Main(i32 %a, i32 %b) !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.umod = urem i32 %a, %b, !dbg !22
// CHECK:STDOUT:   ret i32 %int.umod, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_Cmul_i32.Main(i32 %a, i32 %b) !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.smul = mul i32 %a, %b, !dbg !25
// CHECK:STDOUT:   ret i32 %int.smul, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_Cmul_u32.Main(i32 %a, i32 %b) !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.umul = mul i32 %a, %b, !dbg !28
// CHECK:STDOUT:   ret i32 %int.umul, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_Cneg_i32.Main(i32 %a) !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.snegate = sub i32 0, %a, !dbg !31
// CHECK:STDOUT:   ret i32 %int.snegate, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_Cneg_u32.Main(i32 %a) !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.unegate = sub i32 0, %a, !dbg !34
// CHECK:STDOUT:   ret i32 %int.unegate, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_Csub_i32.Main(i32 %a, i32 %b) !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.ssub = sub i32 %a, %b, !dbg !37
// CHECK:STDOUT:   ret i32 %int.ssub, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_Csub_u32.Main(i32 %a, i32 %b) !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.usub = sub i32 %a, %b, !dbg !40
// CHECK:STDOUT:   ret i32 %int.usub, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i16 @_Cdiv_i16.Main(i16 %a, i16 %b) !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.sdiv = sdiv i16 %a, %b, !dbg !43
// CHECK:STDOUT:   ret i16 %int.sdiv, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i16 @_Cdiv_u16.Main(i16 %a, i16 %b) !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.udiv = udiv i16 %a, %b, !dbg !46
// CHECK:STDOUT:   ret i16 %int.udiv, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "arithmetic.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "add_i32", linkageName: "_Cadd_i32.Main", scope: null, file: !3, line: 14, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 14, column: 44, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 14, column: 37, scope: !4)
// CHECK:STDOUT: !9 = distinct !DISubprogram(name: "add_u32", linkageName: "_Cadd_u32.Main", scope: null, file: !3, line: 15, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !10 = !DILocation(line: 15, column: 44, scope: !9)
// CHECK:STDOUT: !11 = !DILocation(line: 15, column: 37, scope: !9)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "div_i32", linkageName: "_Cdiv_i32.Main", scope: null, file: !3, line: 17, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !13 = !DILocation(line: 17, column: 44, scope: !12)
// CHECK:STDOUT: !14 = !DILocation(line: 17, column: 37, scope: !12)
// CHECK:STDOUT: !15 = distinct !DISubprogram(name: "div_u32", linkageName: "_Cdiv_u32.Main", scope: null, file: !3, line: 18, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !16 = !DILocation(line: 18, column: 44, scope: !15)
// CHECK:STDOUT: !17 = !DILocation(line: 18, column: 37, scope: !15)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "mod_i32", linkageName: "_Cmod_i32.Main", scope: null, file: !3, line: 20, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !19 = !DILocation(line: 20, column: 44, scope: !18)
// CHECK:STDOUT: !20 = !DILocation(line: 20, column: 37, scope: !18)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "mod_u32", linkageName: "_Cmod_u32.Main", scope: null, file: !3, line: 21, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !22 = !DILocation(line: 21, column: 44, scope: !21)
// CHECK:STDOUT: !23 = !DILocation(line: 21, column: 37, scope: !21)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "mul_i32", linkageName: "_Cmul_i32.Main", scope: null, file: !3, line: 23, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !25 = !DILocation(line: 23, column: 44, scope: !24)
// CHECK:STDOUT: !26 = !DILocation(line: 23, column: 37, scope: !24)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "mul_u32", linkageName: "_Cmul_u32.Main", scope: null, file: !3, line: 24, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !28 = !DILocation(line: 24, column: 44, scope: !27)
// CHECK:STDOUT: !29 = !DILocation(line: 24, column: 37, scope: !27)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "neg_i32", linkageName: "_Cneg_i32.Main", scope: null, file: !3, line: 26, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !31 = !DILocation(line: 26, column: 36, scope: !30)
// CHECK:STDOUT: !32 = !DILocation(line: 26, column: 29, scope: !30)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "neg_u32", linkageName: "_Cneg_u32.Main", scope: null, file: !3, line: 27, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !34 = !DILocation(line: 27, column: 36, scope: !33)
// CHECK:STDOUT: !35 = !DILocation(line: 27, column: 29, scope: !33)
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "sub_i32", linkageName: "_Csub_i32.Main", scope: null, file: !3, line: 29, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !37 = !DILocation(line: 29, column: 44, scope: !36)
// CHECK:STDOUT: !38 = !DILocation(line: 29, column: 37, scope: !36)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "sub_u32", linkageName: "_Csub_u32.Main", scope: null, file: !3, line: 30, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !40 = !DILocation(line: 30, column: 44, scope: !39)
// CHECK:STDOUT: !41 = !DILocation(line: 30, column: 37, scope: !39)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "div_i16", linkageName: "_Cdiv_i16.Main", scope: null, file: !3, line: 34, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !43 = !DILocation(line: 34, column: 44, scope: !42)
// CHECK:STDOUT: !44 = !DILocation(line: 34, column: 37, scope: !42)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "div_u16", linkageName: "_Cdiv_u16.Main", scope: null, file: !3, line: 35, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !46 = !DILocation(line: 35, column: 44, scope: !45)
// CHECK:STDOUT: !47 = !DILocation(line: 35, column: 37, scope: !45)
