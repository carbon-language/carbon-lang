// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/builtins/float_to_int.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/builtins/float_to_int.carbon

// --- float_to_int.carbon
library "[[@TEST_NAME]]";

fn ConvertF32ToI32(a: f32) -> i32 = "float.convert_int";
fn TestConvertF32ToI32(a: f32) -> i32 { return ConvertF32ToI32(a); }

fn ConvertF32ToU32(a: f32) -> u32 = "float.convert_int";
fn TestConvertF32ToU32(a: f32) -> u32 { return ConvertF32ToU32(a); }

fn ConvertF64ToI32(a: f64) -> i32 = "float.convert_int";
fn TestConvertF64ToI32(a: f64) -> i32 { return ConvertF64ToI32(a); }

fn ConvertF64ToU32(a: f64) -> u32 = "float.convert_int";
fn TestConvertF64ToU32(a: f64) -> u32 { return ConvertF64ToU32(a); }

fn ConvertF128ToI64(a: f128) -> i64 = "float.convert_int";
fn TestConvertF128ToI64(a: f128) -> i64 { return ConvertF128ToI64(a); }

fn ConvertF128ToU64(a: f128) -> u64 = "float.convert_int";
fn TestConvertF128ToU64(a: f128) -> u64 { return ConvertF128ToU64(a); }

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'float_to_int.carbon'
// CHECK:STDOUT: source_filename = "float_to_int.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestConvertF32ToI32.Main(float %a) #0 !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ConvertF32ToI32.call = fptosi float %a to i32, !dbg !54
// CHECK:STDOUT:   ret i32 %ConvertF32ToI32.call, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestConvertF32ToU32.Main(float %a) #0 !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ConvertF32ToU32.call = fptoui float %a to i32, !dbg !63
// CHECK:STDOUT:   ret i32 %ConvertF32ToU32.call, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestConvertF64ToI32.Main(double %a) #0 !dbg !66 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ConvertF64ToI32.call = fptosi double %a to i32, !dbg !72
// CHECK:STDOUT:   ret i32 %ConvertF64ToI32.call, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestConvertF64ToU32.Main(double %a) #0 !dbg !75 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ConvertF64ToU32.call = fptoui double %a to i32, !dbg !81
// CHECK:STDOUT:   ret i32 %ConvertF64ToU32.call, !dbg !82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestConvertF128ToI64.Main(fp128 %a) #0 !dbg !84 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ConvertF128ToI64.call = fptosi fp128 %a to i64, !dbg !90
// CHECK:STDOUT:   ret i64 %ConvertF128ToI64.call, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestConvertF128ToU64.Main(fp128 %a) #0 !dbg !93 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ConvertF128ToU64.call = fptoui fp128 %a to i64, !dbg !99
// CHECK:STDOUT:   ret i64 %ConvertF128ToU64.call, !dbg !100
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
// CHECK:STDOUT: !34 = !DIFile(filename: "float_to_int.carbon", directory: "")
// CHECK:STDOUT: !35 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !34, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !36 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !37 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !38 = !DIBasicType(name: "int", size: 64, encoding: DW_ATE_signed)
// CHECK:STDOUT: !39 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !40 = !{!36, !39}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !42 = !{!37, !39}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !44 = !{!38, !39}
// CHECK:STDOUT: !45 = !DISubroutineType(types: !44)
// CHECK:STDOUT: !46 = !{!19, !39}
// CHECK:STDOUT: !47 = !DISubroutineType(types: !46)
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "TestConvertF32ToI32", linkageName: "_CTestConvertF32ToI32.Main", scope: null, file: !34, line: 4, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !56)
// CHECK:STDOUT: !49 = !DILocalVariable(arg: 1, scope: !48, type: !39)
// CHECK:STDOUT: !54 = !DILocation(line: 4, column: 48, scope: !48)
// CHECK:STDOUT: !55 = !DILocation(line: 4, column: 41, scope: !48)
// CHECK:STDOUT: !56 = !{!49}
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "TestConvertF32ToU32", linkageName: "_CTestConvertF32ToU32.Main", scope: null, file: !34, line: 7, type: !43, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !65)
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 1, scope: !57, type: !39)
// CHECK:STDOUT: !63 = !DILocation(line: 7, column: 48, scope: !57)
// CHECK:STDOUT: !64 = !DILocation(line: 7, column: 41, scope: !57)
// CHECK:STDOUT: !65 = !{!58}
// CHECK:STDOUT: !66 = distinct !DISubprogram(name: "TestConvertF64ToI32", linkageName: "_CTestConvertF64ToI32.Main", scope: null, file: !34, line: 10, type: !41, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !74)
// CHECK:STDOUT: !67 = !DILocalVariable(arg: 1, scope: !66, type: !39)
// CHECK:STDOUT: !72 = !DILocation(line: 10, column: 48, scope: !66)
// CHECK:STDOUT: !73 = !DILocation(line: 10, column: 41, scope: !66)
// CHECK:STDOUT: !74 = !{!67}
// CHECK:STDOUT: !75 = distinct !DISubprogram(name: "TestConvertF64ToU32", linkageName: "_CTestConvertF64ToU32.Main", scope: null, file: !34, line: 13, type: !43, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !83)
// CHECK:STDOUT: !76 = !DILocalVariable(arg: 1, scope: !75, type: !39)
// CHECK:STDOUT: !81 = !DILocation(line: 13, column: 48, scope: !75)
// CHECK:STDOUT: !82 = !DILocation(line: 13, column: 41, scope: !75)
// CHECK:STDOUT: !83 = !{!76}
// CHECK:STDOUT: !84 = distinct !DISubprogram(name: "TestConvertF128ToI64", linkageName: "_CTestConvertF128ToI64.Main", scope: null, file: !34, line: 16, type: !45, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !92)
// CHECK:STDOUT: !85 = !DILocalVariable(arg: 1, scope: !84, type: !39)
// CHECK:STDOUT: !90 = !DILocation(line: 16, column: 50, scope: !84)
// CHECK:STDOUT: !91 = !DILocation(line: 16, column: 43, scope: !84)
// CHECK:STDOUT: !92 = !{!85}
// CHECK:STDOUT: !93 = distinct !DISubprogram(name: "TestConvertF128ToU64", linkageName: "_CTestConvertF128ToU64.Main", scope: null, file: !34, line: 19, type: !47, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !101)
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 1, scope: !93, type: !39)
// CHECK:STDOUT: !99 = !DILocation(line: 19, column: 50, scope: !93)
// CHECK:STDOUT: !100 = !DILocation(line: 19, column: 43, scope: !93)
// CHECK:STDOUT: !101 = !{!94}
// CHECK:STDOUT:
