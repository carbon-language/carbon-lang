// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/builtins/int_to_float.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/builtins/int_to_float.carbon

// --- int_to_float.carbon
library "[[@TEST_NAME]]";

fn ConvertI32ToF32(a: i32) -> f32 = "int.convert_float";
fn TestConvertI32ToF32(a: i32) -> f32 { return ConvertI32ToF32(a); }

fn ConvertU32ToF32(a: u32) -> f32 = "int.convert_float";
fn TestConvertU32ToF32(a: u32) -> f32 { return ConvertU32ToF32(a); }

fn ConvertI32ToF64(a: i32) -> f64 = "int.convert_float";
fn TestConvertI32ToF64(a: i32) -> f64 { return ConvertI32ToF64(a); }

fn ConvertU32ToF64(a: u32) -> f64 = "int.convert_float";
fn TestConvertU32ToF64(a: u32) -> f64 { return ConvertU32ToF64(a); }

fn ConvertI64ToF128(a: i64) -> f128 = "int.convert_float";
fn TestConvertI64ToF128(a: i64) -> f128 { return ConvertI64ToF128(a); }

fn ConvertU64ToF128(a: u64) -> f128 = "int.convert_float";
fn TestConvertU64ToF128(a: u64) -> f128 { return ConvertU64ToF128(a); }

// CHECK:STDOUT: ; ModuleID = 'int_to_float.carbon'
// CHECK:STDOUT: source_filename = "int_to_float.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define float @_CTestConvertI32ToF32.Main(i32 %a) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ConvertI32ToF32.call = sitofp i32 %a to float, !dbg !11
// CHECK:STDOUT:   ret float %ConvertI32ToF32.call, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define float @_CTestConvertU32ToF32.Main(i32 %a) #0 !dbg !13 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ConvertU32ToF32.call = uitofp i32 %a to float, !dbg !19
// CHECK:STDOUT:   ret float %ConvertU32ToF32.call, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define double @_CTestConvertI32ToF64.Main(i32 %a) #0 !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ConvertI32ToF64.call = sitofp i32 %a to double, !dbg !24
// CHECK:STDOUT:   ret double %ConvertI32ToF64.call, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define double @_CTestConvertU32ToF64.Main(i32 %a) #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ConvertU32ToF64.call = uitofp i32 %a to double, !dbg !29
// CHECK:STDOUT:   ret double %ConvertU32ToF64.call, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define fp128 @_CTestConvertI64ToF128.Main(i64 %a) #0 !dbg !31 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ConvertI64ToF128.call = sitofp i64 %a to fp128, !dbg !37
// CHECK:STDOUT:   ret fp128 %ConvertI64ToF128.call, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define fp128 @_CTestConvertU64ToF128.Main(i64 %a) #0 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ConvertU64ToF128.call = uitofp i64 %a to fp128, !dbg !45
// CHECK:STDOUT:   ret fp128 %ConvertU64ToF128.call, !dbg !46
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
// CHECK:STDOUT: !3 = !DIFile(filename: "int_to_float.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "TestConvertI32ToF32", linkageName: "_CTestConvertI32ToF32.Main", scope: null, file: !3, line: 4, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !9)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !8}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !9 = !{!10}
// CHECK:STDOUT: !10 = !DILocalVariable(arg: 1, scope: !4, type: !8)
// CHECK:STDOUT: !11 = !DILocation(line: 4, column: 48, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 4, column: 41, scope: !4)
// CHECK:STDOUT: !13 = distinct !DISubprogram(name: "TestConvertU32ToF32", linkageName: "_CTestConvertU32ToF32.Main", scope: null, file: !3, line: 7, type: !14, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !17)
// CHECK:STDOUT: !14 = !DISubroutineType(types: !15)
// CHECK:STDOUT: !15 = !{!7, !16}
// CHECK:STDOUT: !16 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !17 = !{!18}
// CHECK:STDOUT: !18 = !DILocalVariable(arg: 1, scope: !13, type: !16)
// CHECK:STDOUT: !19 = !DILocation(line: 7, column: 48, scope: !13)
// CHECK:STDOUT: !20 = !DILocation(line: 7, column: 41, scope: !13)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "TestConvertI32ToF64", linkageName: "_CTestConvertI32ToF64.Main", scope: null, file: !3, line: 10, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !22)
// CHECK:STDOUT: !22 = !{!23}
// CHECK:STDOUT: !23 = !DILocalVariable(arg: 1, scope: !21, type: !8)
// CHECK:STDOUT: !24 = !DILocation(line: 10, column: 48, scope: !21)
// CHECK:STDOUT: !25 = !DILocation(line: 10, column: 41, scope: !21)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "TestConvertU32ToF64", linkageName: "_CTestConvertU32ToF64.Main", scope: null, file: !3, line: 13, type: !14, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !27)
// CHECK:STDOUT: !27 = !{!28}
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 1, scope: !26, type: !16)
// CHECK:STDOUT: !29 = !DILocation(line: 13, column: 48, scope: !26)
// CHECK:STDOUT: !30 = !DILocation(line: 13, column: 41, scope: !26)
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "TestConvertI64ToF128", linkageName: "_CTestConvertI64ToF128.Main", scope: null, file: !3, line: 16, type: !32, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !35)
// CHECK:STDOUT: !32 = !DISubroutineType(types: !33)
// CHECK:STDOUT: !33 = !{!7, !34}
// CHECK:STDOUT: !34 = !DIBasicType(name: "int", size: 64, encoding: DW_ATE_signed)
// CHECK:STDOUT: !35 = !{!36}
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !31, type: !34)
// CHECK:STDOUT: !37 = !DILocation(line: 16, column: 50, scope: !31)
// CHECK:STDOUT: !38 = !DILocation(line: 16, column: 43, scope: !31)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "TestConvertU64ToF128", linkageName: "_CTestConvertU64ToF128.Main", scope: null, file: !3, line: 19, type: !40, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !43)
// CHECK:STDOUT: !40 = !DISubroutineType(types: !41)
// CHECK:STDOUT: !41 = !{!7, !42}
// CHECK:STDOUT: !42 = !DIBasicType(name: "int", size: 64, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !43 = !{!44}
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 1, scope: !39, type: !42)
// CHECK:STDOUT: !45 = !DILocation(line: 19, column: 50, scope: !39)
// CHECK:STDOUT: !46 = !DILocation(line: 19, column: 43, scope: !39)
