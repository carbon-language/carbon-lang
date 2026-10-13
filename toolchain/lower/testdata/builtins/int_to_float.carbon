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

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'int_to_float.carbon'
// CHECK:STDOUT: source_filename = "int_to_float.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define float @_CTestConvertI32ToF32.Main(i32 %a) #0 !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ConvertI32ToF32.call = sitofp i32 %a to float, !dbg !54
// CHECK:STDOUT:   ret float %ConvertI32ToF32.call, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define float @_CTestConvertU32ToF32.Main(i32 %a) #0 !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ConvertU32ToF32.call = uitofp i32 %a to float, !dbg !63
// CHECK:STDOUT:   ret float %ConvertU32ToF32.call, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define double @_CTestConvertI32ToF64.Main(i32 %a) #0 !dbg !66 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ConvertI32ToF64.call = sitofp i32 %a to double, !dbg !72
// CHECK:STDOUT:   ret double %ConvertI32ToF64.call, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define double @_CTestConvertU32ToF64.Main(i32 %a) #0 !dbg !75 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ConvertU32ToF64.call = uitofp i32 %a to double, !dbg !81
// CHECK:STDOUT:   ret double %ConvertU32ToF64.call, !dbg !82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define fp128 @_CTestConvertI64ToF128.Main(i64 %a) #0 !dbg !84 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ConvertI64ToF128.call = sitofp i64 %a to fp128, !dbg !90
// CHECK:STDOUT:   ret fp128 %ConvertI64ToF128.call, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define fp128 @_CTestConvertU64ToF128.Main(i64 %a) #0 !dbg !93 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ConvertU64ToF128.call = uitofp i64 %a to fp128, !dbg !99
// CHECK:STDOUT:   ret fp128 %ConvertU64ToF128.call, !dbg !100
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!35}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !19 = !DIBasicType(name: "int", size: 64, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !34 = !DIFile(filename: "int_to_float.carbon", directory: "")
// CHECK:STDOUT: !35 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !34, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !37 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !38 = !DIBasicType(name: "int", size: 64, encoding: DW_ATE_signed)
// CHECK:STDOUT: !39 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !40 = !{!39, !36}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !42 = !{!39, !37}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !44 = !{!39, !38}
// CHECK:STDOUT: !45 = !DISubroutineType(types: !44)
// CHECK:STDOUT: !46 = !{!39, !19}
// CHECK:STDOUT: !47 = !DISubroutineType(types: !46)
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "TestConvertI32ToF32", linkageName: "_CTestConvertI32ToF32.Main", scope: null, file: !34, line: 4, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !56)
// CHECK:STDOUT: !49 = !DILocalVariable(arg: 1, scope: !48, type: !36)
// CHECK:STDOUT: !54 = !DILocation(line: 4, column: 48, scope: !48)
// CHECK:STDOUT: !55 = !DILocation(line: 4, column: 41, scope: !48)
// CHECK:STDOUT: !56 = !{!49}
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "TestConvertU32ToF32", linkageName: "_CTestConvertU32ToF32.Main", scope: null, file: !34, line: 7, type: !43, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !65)
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 1, scope: !57, type: !37)
// CHECK:STDOUT: !63 = !DILocation(line: 7, column: 48, scope: !57)
// CHECK:STDOUT: !64 = !DILocation(line: 7, column: 41, scope: !57)
// CHECK:STDOUT: !65 = !{!58}
// CHECK:STDOUT: !66 = distinct !DISubprogram(name: "TestConvertI32ToF64", linkageName: "_CTestConvertI32ToF64.Main", scope: null, file: !34, line: 10, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !74)
// CHECK:STDOUT: !67 = !DILocalVariable(arg: 1, scope: !66, type: !36)
// CHECK:STDOUT: !72 = !DILocation(line: 10, column: 48, scope: !66)
// CHECK:STDOUT: !73 = !DILocation(line: 10, column: 41, scope: !66)
// CHECK:STDOUT: !74 = !{!67}
// CHECK:STDOUT: !75 = distinct !DISubprogram(name: "TestConvertU32ToF64", linkageName: "_CTestConvertU32ToF64.Main", scope: null, file: !34, line: 13, type: !43, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !83)
// CHECK:STDOUT: !76 = !DILocalVariable(arg: 1, scope: !75, type: !37)
// CHECK:STDOUT: !81 = !DILocation(line: 13, column: 48, scope: !75)
// CHECK:STDOUT: !82 = !DILocation(line: 13, column: 41, scope: !75)
// CHECK:STDOUT: !83 = !{!76}
// CHECK:STDOUT: !84 = distinct !DISubprogram(name: "TestConvertI64ToF128", linkageName: "_CTestConvertI64ToF128.Main", scope: null, file: !34, line: 16, type: !45, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !92)
// CHECK:STDOUT: !85 = !DILocalVariable(arg: 1, scope: !84, type: !38)
// CHECK:STDOUT: !90 = !DILocation(line: 16, column: 50, scope: !84)
// CHECK:STDOUT: !91 = !DILocation(line: 16, column: 43, scope: !84)
// CHECK:STDOUT: !92 = !{!85}
// CHECK:STDOUT: !93 = distinct !DISubprogram(name: "TestConvertU64ToF128", linkageName: "_CTestConvertU64ToF128.Main", scope: null, file: !34, line: 19, type: !47, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !101)
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 1, scope: !93, type: !19)
// CHECK:STDOUT: !99 = !DILocation(line: 19, column: 50, scope: !93)
// CHECK:STDOUT: !100 = !DILocation(line: 19, column: 43, scope: !93)
// CHECK:STDOUT: !101 = !{!94}
// CHECK:STDOUT:
