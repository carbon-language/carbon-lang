// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/builtins/float_to_float.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/builtins/float_to_float.carbon

// --- float_to_float.carbon
library "[[@TEST_NAME]]";

fn ConvertF32ToF64(a: f32) -> f64 = "float.convert";
fn TestConvertF32ToF64(a: f32) -> f64 { return ConvertF32ToF64(a); }

fn ConvertF64ToF32(a: f64) -> f32 = "float.convert";
fn TestConvertF64ToF32(a: f64) -> f32 { return ConvertF64ToF32(a); }

fn ConvertF16ToF128(a: f16) -> f128 = "float.convert";
fn TestConvertF16ToF128(a: f16) -> f128 { return ConvertF16ToF128(a); }

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'float_to_float.carbon'
// CHECK:STDOUT: source_filename = "float_to_float.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define double @_CTestConvertF32ToF64.Main(float %a) #0 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ConvertF32ToF64.call = fpext float %a to double, !dbg !43
// CHECK:STDOUT:   ret double %ConvertF32ToF64.call, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define float @_CTestConvertF64ToF32.Main(double %a) #0 !dbg !46 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ConvertF64ToF32.call = fptrunc double %a to float, !dbg !52
// CHECK:STDOUT:   ret float %ConvertF64ToF32.call, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define fp128 @_CTestConvertF16ToF128.Main(half %a) #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ConvertF16ToF128.call = fpext half %a to fp128, !dbg !61
// CHECK:STDOUT:   ret fp128 %ConvertF16ToF128.call, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!33}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !32 = !DIFile(filename: "float_to_float.carbon", directory: "")
// CHECK:STDOUT: !33 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !32, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !34 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !35 = !{!34, !34}
// CHECK:STDOUT: !36 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "TestConvertF32ToF64", linkageName: "_CTestConvertF32ToF64.Main", scope: null, file: !32, line: 4, type: !36, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !45)
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !37, type: !34)
// CHECK:STDOUT: !43 = !DILocation(line: 4, column: 48, scope: !37)
// CHECK:STDOUT: !44 = !DILocation(line: 4, column: 41, scope: !37)
// CHECK:STDOUT: !45 = !{!38}
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "TestConvertF64ToF32", linkageName: "_CTestConvertF64ToF32.Main", scope: null, file: !32, line: 7, type: !36, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !54)
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !46, type: !34)
// CHECK:STDOUT: !52 = !DILocation(line: 7, column: 48, scope: !46)
// CHECK:STDOUT: !53 = !DILocation(line: 7, column: 41, scope: !46)
// CHECK:STDOUT: !54 = !{!47}
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "TestConvertF16ToF128", linkageName: "_CTestConvertF16ToF128.Main", scope: null, file: !32, line: 10, type: !36, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !63)
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !55, type: !34)
// CHECK:STDOUT: !61 = !DILocation(line: 10, column: 50, scope: !55)
// CHECK:STDOUT: !62 = !DILocation(line: 10, column: 43, scope: !55)
// CHECK:STDOUT: !63 = !{!56}
// CHECK:STDOUT:
