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

// CHECK:STDOUT: ; ModuleID = 'float_to_float.carbon'
// CHECK:STDOUT: source_filename = "float_to_float.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define double @_CTestConvertF32ToF64.Main(float %a) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ConvertF32ToF64.call = fpext float %a to double, !dbg !10
// CHECK:STDOUT:   ret double %ConvertF32ToF64.call, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define float @_CTestConvertF64ToF32.Main(double %a) #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ConvertF64ToF32.call = fptrunc double %a to float, !dbg !15
// CHECK:STDOUT:   ret float %ConvertF64ToF32.call, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define fp128 @_CTestConvertF16ToF128.Main(half %a) #0 !dbg !17 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ConvertF16ToF128.call = fpext half %a to fp128, !dbg !20
// CHECK:STDOUT:   ret fp128 %ConvertF16ToF128.call, !dbg !21
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
// CHECK:STDOUT: !3 = !DIFile(filename: "float_to_float.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "TestConvertF32ToF64", linkageName: "_CTestConvertF32ToF64.Main", scope: null, file: !3, line: 4, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 4, column: 48, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 4, column: 41, scope: !4)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "TestConvertF64ToF32", linkageName: "_CTestConvertF64ToF32.Main", scope: null, file: !3, line: 7, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !13)
// CHECK:STDOUT: !13 = !{!14}
// CHECK:STDOUT: !14 = !DILocalVariable(arg: 1, scope: !12, type: !7)
// CHECK:STDOUT: !15 = !DILocation(line: 7, column: 48, scope: !12)
// CHECK:STDOUT: !16 = !DILocation(line: 7, column: 41, scope: !12)
// CHECK:STDOUT: !17 = distinct !DISubprogram(name: "TestConvertF16ToF128", linkageName: "_CTestConvertF16ToF128.Main", scope: null, file: !3, line: 10, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !18)
// CHECK:STDOUT: !18 = !{!19}
// CHECK:STDOUT: !19 = !DILocalVariable(arg: 1, scope: !17, type: !7)
// CHECK:STDOUT: !20 = !DILocation(line: 10, column: 50, scope: !17)
// CHECK:STDOUT: !21 = !DILocation(line: 10, column: 43, scope: !17)
