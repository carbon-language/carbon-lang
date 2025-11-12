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
// CHECK:STDOUT:   %UInt.as.AddWith.impl.Op.call = add i32 %a, %b, !dbg !17
// CHECK:STDOUT:   ret i32 %UInt.as.AddWith.impl.Op.call, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_Cdiv_i32.Main(i32 %a, i32 %b) #0 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.DivWith.impl.Op.call = sdiv i32 %a, %b, !dbg !23
// CHECK:STDOUT:   ret i32 %Int.as.DivWith.impl.Op.call, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_Cdiv_u32.Main(i32 %a, i32 %b) #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %UInt.as.DivWith.impl.Op.call = udiv i32 %a, %b, !dbg !29
// CHECK:STDOUT:   ret i32 %UInt.as.DivWith.impl.Op.call, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_Cmod_i32.Main(i32 %a, i32 %b) #0 !dbg !31 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.ModWith.impl.Op.call = srem i32 %a, %b, !dbg !35
// CHECK:STDOUT:   ret i32 %Int.as.ModWith.impl.Op.call, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_Cmod_u32.Main(i32 %a, i32 %b) #0 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %UInt.as.ModWith.impl.Op.call = urem i32 %a, %b, !dbg !41
// CHECK:STDOUT:   ret i32 %UInt.as.ModWith.impl.Op.call, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_Cmul_i32.Main(i32 %a, i32 %b) #0 !dbg !43 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.MulWith.impl.Op.call = mul i32 %a, %b, !dbg !47
// CHECK:STDOUT:   ret i32 %Int.as.MulWith.impl.Op.call, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_Cmul_u32.Main(i32 %a, i32 %b) #0 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %UInt.as.MulWith.impl.Op.call = mul i32 %a, %b, !dbg !53
// CHECK:STDOUT:   ret i32 %UInt.as.MulWith.impl.Op.call, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_Cneg_i32.Main(i32 %a) #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.Negate.impl.Op.call = sub i32 0, %a, !dbg !60
// CHECK:STDOUT:   ret i32 %Int.as.Negate.impl.Op.call, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_Cneg_u32.Main(i32 %a) #0 !dbg !62 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %UInt.as.Negate.impl.Op.call = sub i32 0, %a, !dbg !65
// CHECK:STDOUT:   ret i32 %UInt.as.Negate.impl.Op.call, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_Csub_i32.Main(i32 %a, i32 %b) #0 !dbg !67 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.SubWith.impl.Op.call = sub i32 %a, %b, !dbg !71
// CHECK:STDOUT:   ret i32 %Int.as.SubWith.impl.Op.call, !dbg !72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_Csub_u32.Main(i32 %a, i32 %b) #0 !dbg !73 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %UInt.as.SubWith.impl.Op.call = sub i32 %a, %b, !dbg !77
// CHECK:STDOUT:   ret i32 %UInt.as.SubWith.impl.Op.call, !dbg !78
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i16 @_Cdiv_i16.Main(i16 %a, i16 %b) #0 !dbg !79 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.DivWith.impl.Op.call = sdiv i16 %a, %b, !dbg !86
// CHECK:STDOUT:   ret i16 %Int.as.DivWith.impl.Op.call, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i16 @_Cdiv_u16.Main(i16 %a, i16 %b) #0 !dbg !88 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %UInt.as.DivWith.impl.Op.call = udiv i16 %a, %b, !dbg !92
// CHECK:STDOUT:   ret i16 %UInt.as.DivWith.impl.Op.call, !dbg !93
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
// CHECK:STDOUT: !13 = distinct !DISubprogram(name: "add_u32", linkageName: "_Cadd_u32.Main", scope: null, file: !3, line: 17, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !14)
// CHECK:STDOUT: !14 = !{!15, !16}
// CHECK:STDOUT: !15 = !DILocalVariable(arg: 1, scope: !13, type: !7)
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 2, scope: !13, type: !7)
// CHECK:STDOUT: !17 = !DILocation(line: 17, column: 44, scope: !13)
// CHECK:STDOUT: !18 = !DILocation(line: 17, column: 37, scope: !13)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "div_i32", linkageName: "_Cdiv_i32.Main", scope: null, file: !3, line: 19, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !20)
// CHECK:STDOUT: !20 = !{!21, !22}
// CHECK:STDOUT: !21 = !DILocalVariable(arg: 1, scope: !19, type: !7)
// CHECK:STDOUT: !22 = !DILocalVariable(arg: 2, scope: !19, type: !7)
// CHECK:STDOUT: !23 = !DILocation(line: 19, column: 44, scope: !19)
// CHECK:STDOUT: !24 = !DILocation(line: 19, column: 37, scope: !19)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "div_u32", linkageName: "_Cdiv_u32.Main", scope: null, file: !3, line: 20, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !26)
// CHECK:STDOUT: !26 = !{!27, !28}
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !25, type: !7)
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 2, scope: !25, type: !7)
// CHECK:STDOUT: !29 = !DILocation(line: 20, column: 44, scope: !25)
// CHECK:STDOUT: !30 = !DILocation(line: 20, column: 37, scope: !25)
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "mod_i32", linkageName: "_Cmod_i32.Main", scope: null, file: !3, line: 22, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !32)
// CHECK:STDOUT: !32 = !{!33, !34}
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 1, scope: !31, type: !7)
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 2, scope: !31, type: !7)
// CHECK:STDOUT: !35 = !DILocation(line: 22, column: 44, scope: !31)
// CHECK:STDOUT: !36 = !DILocation(line: 22, column: 37, scope: !31)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "mod_u32", linkageName: "_Cmod_u32.Main", scope: null, file: !3, line: 23, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !38)
// CHECK:STDOUT: !38 = !{!39, !40}
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !37, type: !7)
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 2, scope: !37, type: !7)
// CHECK:STDOUT: !41 = !DILocation(line: 23, column: 44, scope: !37)
// CHECK:STDOUT: !42 = !DILocation(line: 23, column: 37, scope: !37)
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "mul_i32", linkageName: "_Cmul_i32.Main", scope: null, file: !3, line: 25, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !44)
// CHECK:STDOUT: !44 = !{!45, !46}
// CHECK:STDOUT: !45 = !DILocalVariable(arg: 1, scope: !43, type: !7)
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 2, scope: !43, type: !7)
// CHECK:STDOUT: !47 = !DILocation(line: 25, column: 44, scope: !43)
// CHECK:STDOUT: !48 = !DILocation(line: 25, column: 37, scope: !43)
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "mul_u32", linkageName: "_Cmul_u32.Main", scope: null, file: !3, line: 26, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !50)
// CHECK:STDOUT: !50 = !{!51, !52}
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !49, type: !7)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 2, scope: !49, type: !7)
// CHECK:STDOUT: !53 = !DILocation(line: 26, column: 44, scope: !49)
// CHECK:STDOUT: !54 = !DILocation(line: 26, column: 37, scope: !49)
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "neg_i32", linkageName: "_Cneg_i32.Main", scope: null, file: !3, line: 28, type: !56, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !58)
// CHECK:STDOUT: !56 = !DISubroutineType(types: !57)
// CHECK:STDOUT: !57 = !{!7, !7}
// CHECK:STDOUT: !58 = !{!59}
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !55, type: !7)
// CHECK:STDOUT: !60 = !DILocation(line: 28, column: 36, scope: !55)
// CHECK:STDOUT: !61 = !DILocation(line: 28, column: 29, scope: !55)
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "neg_u32", linkageName: "_Cneg_u32.Main", scope: null, file: !3, line: 29, type: !56, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !63)
// CHECK:STDOUT: !63 = !{!64}
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !62, type: !7)
// CHECK:STDOUT: !65 = !DILocation(line: 29, column: 36, scope: !62)
// CHECK:STDOUT: !66 = !DILocation(line: 29, column: 29, scope: !62)
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "sub_i32", linkageName: "_Csub_i32.Main", scope: null, file: !3, line: 31, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !68)
// CHECK:STDOUT: !68 = !{!69, !70}
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 1, scope: !67, type: !7)
// CHECK:STDOUT: !70 = !DILocalVariable(arg: 2, scope: !67, type: !7)
// CHECK:STDOUT: !71 = !DILocation(line: 31, column: 44, scope: !67)
// CHECK:STDOUT: !72 = !DILocation(line: 31, column: 37, scope: !67)
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "sub_u32", linkageName: "_Csub_u32.Main", scope: null, file: !3, line: 32, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !74)
// CHECK:STDOUT: !74 = !{!75, !76}
// CHECK:STDOUT: !75 = !DILocalVariable(arg: 1, scope: !73, type: !7)
// CHECK:STDOUT: !76 = !DILocalVariable(arg: 2, scope: !73, type: !7)
// CHECK:STDOUT: !77 = !DILocation(line: 32, column: 44, scope: !73)
// CHECK:STDOUT: !78 = !DILocation(line: 32, column: 37, scope: !73)
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "div_i16", linkageName: "_Cdiv_i16.Main", scope: null, file: !3, line: 36, type: !80, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !83)
// CHECK:STDOUT: !80 = !DISubroutineType(types: !81)
// CHECK:STDOUT: !81 = !{!82, !82, !82}
// CHECK:STDOUT: !82 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !83 = !{!84, !85}
// CHECK:STDOUT: !84 = !DILocalVariable(arg: 1, scope: !79, type: !82)
// CHECK:STDOUT: !85 = !DILocalVariable(arg: 2, scope: !79, type: !82)
// CHECK:STDOUT: !86 = !DILocation(line: 36, column: 44, scope: !79)
// CHECK:STDOUT: !87 = !DILocation(line: 36, column: 37, scope: !79)
// CHECK:STDOUT: !88 = distinct !DISubprogram(name: "div_u16", linkageName: "_Cdiv_u16.Main", scope: null, file: !3, line: 37, type: !80, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !89)
// CHECK:STDOUT: !89 = !{!90, !91}
// CHECK:STDOUT: !90 = !DILocalVariable(arg: 1, scope: !88, type: !82)
// CHECK:STDOUT: !91 = !DILocalVariable(arg: 2, scope: !88, type: !82)
// CHECK:STDOUT: !92 = !DILocation(line: 37, column: 44, scope: !88)
// CHECK:STDOUT: !93 = !DILocation(line: 37, column: 37, scope: !88)
