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
// CHECK:STDOUT: define float @_CTestConvertI32ToF32.Main(i32 %a) #0 !dbg !46 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ConvertI32ToF32.call = sitofp i32 %a to float, !dbg !52
// CHECK:STDOUT:   ret float %ConvertI32ToF32.call, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define float @_CTestConvertU32ToF32.Main(i32 %a) #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ConvertU32ToF32.call = uitofp i32 %a to float, !dbg !61
// CHECK:STDOUT:   ret float %ConvertU32ToF32.call, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define double @_CTestConvertI32ToF64.Main(i32 %a) #0 !dbg !64 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ConvertI32ToF64.call = sitofp i32 %a to double, !dbg !70
// CHECK:STDOUT:   ret double %ConvertI32ToF64.call, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define double @_CTestConvertU32ToF64.Main(i32 %a) #0 !dbg !73 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ConvertU32ToF64.call = uitofp i32 %a to double, !dbg !79
// CHECK:STDOUT:   ret double %ConvertU32ToF64.call, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define fp128 @_CTestConvertI64ToF128.Main(i64 %a) #0 !dbg !82 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ConvertI64ToF128.call = sitofp i64 %a to fp128, !dbg !88
// CHECK:STDOUT:   ret fp128 %ConvertI64ToF128.call, !dbg !89
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define fp128 @_CTestConvertU64ToF128.Main(i64 %a) #0 !dbg !91 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ConvertU64ToF128.call = uitofp i64 %a to fp128, !dbg !97
// CHECK:STDOUT:   ret fp128 %ConvertU64ToF128.call, !dbg !98
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!33}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !17 = !DIBasicType(name: "int", size: 64, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !32 = !DIFile(filename: "int_to_float.carbon", directory: "")
// CHECK:STDOUT: !33 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !32, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !34 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !35 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 64, encoding: DW_ATE_signed)
// CHECK:STDOUT: !37 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !38 = !{!37, !34}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !40 = !{!37, !35}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !42 = !{!37, !36}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !44 = !{!37, !17}
// CHECK:STDOUT: !45 = !DISubroutineType(types: !44)
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "TestConvertI32ToF32", linkageName: "_CTestConvertI32ToF32.Main", scope: null, file: !32, line: 4, type: !39, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !54)
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !46, type: !34)
// CHECK:STDOUT: !52 = !DILocation(line: 4, column: 48, scope: !46)
// CHECK:STDOUT: !53 = !DILocation(line: 4, column: 41, scope: !46)
// CHECK:STDOUT: !54 = !{!47}
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "TestConvertU32ToF32", linkageName: "_CTestConvertU32ToF32.Main", scope: null, file: !32, line: 7, type: !41, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !63)
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !55, type: !35)
// CHECK:STDOUT: !61 = !DILocation(line: 7, column: 48, scope: !55)
// CHECK:STDOUT: !62 = !DILocation(line: 7, column: 41, scope: !55)
// CHECK:STDOUT: !63 = !{!56}
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "TestConvertI32ToF64", linkageName: "_CTestConvertI32ToF64.Main", scope: null, file: !32, line: 10, type: !39, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !72)
// CHECK:STDOUT: !65 = !DILocalVariable(arg: 1, scope: !64, type: !34)
// CHECK:STDOUT: !70 = !DILocation(line: 10, column: 48, scope: !64)
// CHECK:STDOUT: !71 = !DILocation(line: 10, column: 41, scope: !64)
// CHECK:STDOUT: !72 = !{!65}
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "TestConvertU32ToF64", linkageName: "_CTestConvertU32ToF64.Main", scope: null, file: !32, line: 13, type: !41, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !81)
// CHECK:STDOUT: !74 = !DILocalVariable(arg: 1, scope: !73, type: !35)
// CHECK:STDOUT: !79 = !DILocation(line: 13, column: 48, scope: !73)
// CHECK:STDOUT: !80 = !DILocation(line: 13, column: 41, scope: !73)
// CHECK:STDOUT: !81 = !{!74}
// CHECK:STDOUT: !82 = distinct !DISubprogram(name: "TestConvertI64ToF128", linkageName: "_CTestConvertI64ToF128.Main", scope: null, file: !32, line: 16, type: !43, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !90)
// CHECK:STDOUT: !83 = !DILocalVariable(arg: 1, scope: !82, type: !36)
// CHECK:STDOUT: !88 = !DILocation(line: 16, column: 50, scope: !82)
// CHECK:STDOUT: !89 = !DILocation(line: 16, column: 43, scope: !82)
// CHECK:STDOUT: !90 = !{!83}
// CHECK:STDOUT: !91 = distinct !DISubprogram(name: "TestConvertU64ToF128", linkageName: "_CTestConvertU64ToF128.Main", scope: null, file: !32, line: 19, type: !45, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !99)
// CHECK:STDOUT: !92 = !DILocalVariable(arg: 1, scope: !91, type: !17)
// CHECK:STDOUT: !97 = !DILocation(line: 19, column: 50, scope: !91)
// CHECK:STDOUT: !98 = !DILocation(line: 19, column: 43, scope: !91)
// CHECK:STDOUT: !99 = !{!92}
// CHECK:STDOUT:
