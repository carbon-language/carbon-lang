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
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_Cadd_i32.Main(i32 %a, i32 %b) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %a, %b, !dbg !11
// CHECK:STDOUT:   ret i32 %Int.as.AddWith.impl.Op.call, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_Cadd_u32.Main(i32 %a, i32 %b) #0 !dbg !13 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %UInt.as.AddWith.impl.Op.call = add i32 %a, %b, !dbg !20
// CHECK:STDOUT:   ret i32 %UInt.as.AddWith.impl.Op.call, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_Cdiv_i32.Main(i32 %a, i32 %b) #0 !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.DivWith.impl.Op.call = sdiv i32 %a, %b, !dbg !26
// CHECK:STDOUT:   ret i32 %Int.as.DivWith.impl.Op.call, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_Cdiv_u32.Main(i32 %a, i32 %b) #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %UInt.as.DivWith.impl.Op.call = udiv i32 %a, %b, !dbg !32
// CHECK:STDOUT:   ret i32 %UInt.as.DivWith.impl.Op.call, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_Cmod_i32.Main(i32 %a, i32 %b) #0 !dbg !34 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %a, %b, !dbg !38
// CHECK:STDOUT:   ret i32 %Int.as.ModWith.impl.Op.call, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_Cmod_u32.Main(i32 %a, i32 %b) #0 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %UInt.as.ModWith.impl.Op.call = urem i32 %a, %b, !dbg !44
// CHECK:STDOUT:   ret i32 %UInt.as.ModWith.impl.Op.call, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_Cmul_i32.Main(i32 %a, i32 %b) #0 !dbg !46 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.MulWith.impl.Op.call = mul i32 %a, %b, !dbg !50
// CHECK:STDOUT:   ret i32 %Int.as.MulWith.impl.Op.call, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_Cmul_u32.Main(i32 %a, i32 %b) #0 !dbg !52 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %UInt.as.MulWith.impl.Op.call = mul i32 %a, %b, !dbg !56
// CHECK:STDOUT:   ret i32 %UInt.as.MulWith.impl.Op.call, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_Cneg_i32.Main(i32 %a) #0 !dbg !58 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.call = sub i32 0, %a, !dbg !63
// CHECK:STDOUT:   ret i32 %Int.as.Negate.impl.Op.call, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_Cneg_u32.Main(i32 %a) #0 !dbg !65 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %UInt.as.Negate.impl.Op.call = sub i32 0, %a, !dbg !70
// CHECK:STDOUT:   ret i32 %UInt.as.Negate.impl.Op.call, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_Csub_i32.Main(i32 %a, i32 %b) #0 !dbg !72 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.SubWith.impl.Op.call = sub i32 %a, %b, !dbg !76
// CHECK:STDOUT:   ret i32 %Int.as.SubWith.impl.Op.call, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_Csub_u32.Main(i32 %a, i32 %b) #0 !dbg !78 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %UInt.as.SubWith.impl.Op.call = sub i32 %a, %b, !dbg !82
// CHECK:STDOUT:   ret i32 %UInt.as.SubWith.impl.Op.call, !dbg !83
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i16 @_Cdiv_i16.Main(i16 %a, i16 %b) #0 !dbg !84 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.DivWith.impl.Op.call = sdiv i16 %a, %b, !dbg !91
// CHECK:STDOUT:   ret i16 %Int.as.DivWith.impl.Op.call, !dbg !92
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i16 @_Cdiv_u16.Main(i16 %a, i16 %b) #0 !dbg !93 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %UInt.as.DivWith.impl.Op.call = udiv i16 %a, %b, !dbg !100
// CHECK:STDOUT:   ret i16 %UInt.as.DivWith.impl.Op.call, !dbg !101
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "arithmetic.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "add_i32", linkageName: "_Cadd_i32.Main", scope: null, file: !3, line: 16, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !7, !7}
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !{!9, !10}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocalVariable(arg: 2, scope: !4, type: !7)
// CHECK:STDOUT: !11 = !DILocation(line: 16, column: 44, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 16, column: 37, scope: !4)
// CHECK:STDOUT: !13 = distinct !DISubprogram(name: "add_u32", linkageName: "_Cadd_u32.Main", scope: null, file: !3, line: 17, type: !14, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !17)
// CHECK:STDOUT: !14 = !DISubroutineType(types: !15)
// CHECK:STDOUT: !15 = !{!16, !16, !16}
// CHECK:STDOUT: !16 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !17 = !{!18, !19}
// CHECK:STDOUT: !18 = !DILocalVariable(arg: 1, scope: !13, type: !16)
// CHECK:STDOUT: !19 = !DILocalVariable(arg: 2, scope: !13, type: !16)
// CHECK:STDOUT: !20 = !DILocation(line: 17, column: 44, scope: !13)
// CHECK:STDOUT: !21 = !DILocation(line: 17, column: 37, scope: !13)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "div_i32", linkageName: "_Cdiv_i32.Main", scope: null, file: !3, line: 19, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !23)
// CHECK:STDOUT: !23 = !{!24, !25}
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !22, type: !7)
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 2, scope: !22, type: !7)
// CHECK:STDOUT: !26 = !DILocation(line: 19, column: 44, scope: !22)
// CHECK:STDOUT: !27 = !DILocation(line: 19, column: 37, scope: !22)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "div_u32", linkageName: "_Cdiv_u32.Main", scope: null, file: !3, line: 20, type: !14, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !29)
// CHECK:STDOUT: !29 = !{!30, !31}
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 1, scope: !28, type: !16)
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 2, scope: !28, type: !16)
// CHECK:STDOUT: !32 = !DILocation(line: 20, column: 44, scope: !28)
// CHECK:STDOUT: !33 = !DILocation(line: 20, column: 37, scope: !28)
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "mod_i32", linkageName: "_Cmod_i32.Main", scope: null, file: !3, line: 22, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !35)
// CHECK:STDOUT: !35 = !{!36, !37}
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !34, type: !7)
// CHECK:STDOUT: !37 = !DILocalVariable(arg: 2, scope: !34, type: !7)
// CHECK:STDOUT: !38 = !DILocation(line: 22, column: 44, scope: !34)
// CHECK:STDOUT: !39 = !DILocation(line: 22, column: 37, scope: !34)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "mod_u32", linkageName: "_Cmod_u32.Main", scope: null, file: !3, line: 23, type: !14, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !41)
// CHECK:STDOUT: !41 = !{!42, !43}
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 1, scope: !40, type: !16)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 2, scope: !40, type: !16)
// CHECK:STDOUT: !44 = !DILocation(line: 23, column: 44, scope: !40)
// CHECK:STDOUT: !45 = !DILocation(line: 23, column: 37, scope: !40)
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "mul_i32", linkageName: "_Cmul_i32.Main", scope: null, file: !3, line: 25, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !47)
// CHECK:STDOUT: !47 = !{!48, !49}
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 1, scope: !46, type: !7)
// CHECK:STDOUT: !49 = !DILocalVariable(arg: 2, scope: !46, type: !7)
// CHECK:STDOUT: !50 = !DILocation(line: 25, column: 44, scope: !46)
// CHECK:STDOUT: !51 = !DILocation(line: 25, column: 37, scope: !46)
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "mul_u32", linkageName: "_Cmul_u32.Main", scope: null, file: !3, line: 26, type: !14, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !53)
// CHECK:STDOUT: !53 = !{!54, !55}
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !52, type: !16)
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 2, scope: !52, type: !16)
// CHECK:STDOUT: !56 = !DILocation(line: 26, column: 44, scope: !52)
// CHECK:STDOUT: !57 = !DILocation(line: 26, column: 37, scope: !52)
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "neg_i32", linkageName: "_Cneg_i32.Main", scope: null, file: !3, line: 28, type: !59, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !61)
// CHECK:STDOUT: !59 = !DISubroutineType(types: !60)
// CHECK:STDOUT: !60 = !{!7, !7}
// CHECK:STDOUT: !61 = !{!62}
// CHECK:STDOUT: !62 = !DILocalVariable(arg: 1, scope: !58, type: !7)
// CHECK:STDOUT: !63 = !DILocation(line: 28, column: 36, scope: !58)
// CHECK:STDOUT: !64 = !DILocation(line: 28, column: 29, scope: !58)
// CHECK:STDOUT: !65 = distinct !DISubprogram(name: "neg_u32", linkageName: "_Cneg_u32.Main", scope: null, file: !3, line: 29, type: !66, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !68)
// CHECK:STDOUT: !66 = !DISubroutineType(types: !67)
// CHECK:STDOUT: !67 = !{!16, !16}
// CHECK:STDOUT: !68 = !{!69}
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 1, scope: !65, type: !16)
// CHECK:STDOUT: !70 = !DILocation(line: 29, column: 36, scope: !65)
// CHECK:STDOUT: !71 = !DILocation(line: 29, column: 29, scope: !65)
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "sub_i32", linkageName: "_Csub_i32.Main", scope: null, file: !3, line: 31, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !73)
// CHECK:STDOUT: !73 = !{!74, !75}
// CHECK:STDOUT: !74 = !DILocalVariable(arg: 1, scope: !72, type: !7)
// CHECK:STDOUT: !75 = !DILocalVariable(arg: 2, scope: !72, type: !7)
// CHECK:STDOUT: !76 = !DILocation(line: 31, column: 44, scope: !72)
// CHECK:STDOUT: !77 = !DILocation(line: 31, column: 37, scope: !72)
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "sub_u32", linkageName: "_Csub_u32.Main", scope: null, file: !3, line: 32, type: !14, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !79)
// CHECK:STDOUT: !79 = !{!80, !81}
// CHECK:STDOUT: !80 = !DILocalVariable(arg: 1, scope: !78, type: !16)
// CHECK:STDOUT: !81 = !DILocalVariable(arg: 2, scope: !78, type: !16)
// CHECK:STDOUT: !82 = !DILocation(line: 32, column: 44, scope: !78)
// CHECK:STDOUT: !83 = !DILocation(line: 32, column: 37, scope: !78)
// CHECK:STDOUT: !84 = distinct !DISubprogram(name: "div_i16", linkageName: "_Cdiv_i16.Main", scope: null, file: !3, line: 36, type: !85, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !88)
// CHECK:STDOUT: !85 = !DISubroutineType(types: !86)
// CHECK:STDOUT: !86 = !{!87, !87, !87}
// CHECK:STDOUT: !87 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !88 = !{!89, !90}
// CHECK:STDOUT: !89 = !DILocalVariable(arg: 1, scope: !84, type: !87)
// CHECK:STDOUT: !90 = !DILocalVariable(arg: 2, scope: !84, type: !87)
// CHECK:STDOUT: !91 = !DILocation(line: 36, column: 44, scope: !84)
// CHECK:STDOUT: !92 = !DILocation(line: 36, column: 37, scope: !84)
// CHECK:STDOUT: !93 = distinct !DISubprogram(name: "div_u16", linkageName: "_Cdiv_u16.Main", scope: null, file: !3, line: 37, type: !94, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !97)
// CHECK:STDOUT: !94 = !DISubroutineType(types: !95)
// CHECK:STDOUT: !95 = !{!96, !96, !96}
// CHECK:STDOUT: !96 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !97 = !{!98, !99}
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 1, scope: !93, type: !96)
// CHECK:STDOUT: !99 = !DILocalVariable(arg: 2, scope: !93, type: !96)
// CHECK:STDOUT: !100 = !DILocation(line: 37, column: 44, scope: !93)
// CHECK:STDOUT: !101 = !DILocation(line: 37, column: 37, scope: !93)
