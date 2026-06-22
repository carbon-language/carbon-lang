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

// CHECK:STDOUT: ; ModuleID = 'float_to_int.carbon'
// CHECK:STDOUT: source_filename = "float_to_int.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestConvertF32ToI32.Main(float %a) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ConvertF32ToI32.call = fptosi float %a to i32, !dbg !11
// CHECK:STDOUT:   ret i32 %ConvertF32ToI32.call, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestConvertF32ToU32.Main(float %a) #0 !dbg !13 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ConvertF32ToU32.call = fptoui float %a to i32, !dbg !19
// CHECK:STDOUT:   ret i32 %ConvertF32ToU32.call, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestConvertF64ToI32.Main(double %a) #0 !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ConvertF64ToI32.call = fptosi double %a to i32, !dbg !24
// CHECK:STDOUT:   ret i32 %ConvertF64ToI32.call, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestConvertF64ToU32.Main(double %a) #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ConvertF64ToU32.call = fptoui double %a to i32, !dbg !29
// CHECK:STDOUT:   ret i32 %ConvertF64ToU32.call, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestConvertF128ToI64.Main(fp128 %a) #0 !dbg !31 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ConvertF128ToI64.call = fptosi fp128 %a to i64, !dbg !37
// CHECK:STDOUT:   ret i64 %ConvertF128ToI64.call, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CTestConvertF128ToU64.Main(fp128 %a) #0 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ConvertF128ToU64.call = fptoui fp128 %a to i64, !dbg !45
// CHECK:STDOUT:   ret i64 %ConvertF128ToU64.call, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "float_to_int.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "TestConvertF32ToI32", linkageName: "_CTestConvertF32ToI32.Main", scope: null, file: !1, line: 4, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !9)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !8}
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !9 = !{!10}
// CHECK:STDOUT: !10 = !DILocalVariable(arg: 1, scope: !4, type: !8)
// CHECK:STDOUT: !11 = !DILocation(line: 4, column: 48, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 4, column: 41, scope: !4)
// CHECK:STDOUT: !13 = distinct !DISubprogram(name: "TestConvertF32ToU32", linkageName: "_CTestConvertF32ToU32.Main", scope: null, file: !1, line: 7, type: !14, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !17)
// CHECK:STDOUT: !14 = !DISubroutineType(types: !15)
// CHECK:STDOUT: !15 = !{!16, !8}
// CHECK:STDOUT: !16 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !17 = !{!18}
// CHECK:STDOUT: !18 = !DILocalVariable(arg: 1, scope: !13, type: !8)
// CHECK:STDOUT: !19 = !DILocation(line: 7, column: 48, scope: !13)
// CHECK:STDOUT: !20 = !DILocation(line: 7, column: 41, scope: !13)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "TestConvertF64ToI32", linkageName: "_CTestConvertF64ToI32.Main", scope: null, file: !1, line: 10, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !22)
// CHECK:STDOUT: !22 = !{!23}
// CHECK:STDOUT: !23 = !DILocalVariable(arg: 1, scope: !21, type: !8)
// CHECK:STDOUT: !24 = !DILocation(line: 10, column: 48, scope: !21)
// CHECK:STDOUT: !25 = !DILocation(line: 10, column: 41, scope: !21)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "TestConvertF64ToU32", linkageName: "_CTestConvertF64ToU32.Main", scope: null, file: !1, line: 13, type: !14, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !27)
// CHECK:STDOUT: !27 = !{!28}
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 1, scope: !26, type: !8)
// CHECK:STDOUT: !29 = !DILocation(line: 13, column: 48, scope: !26)
// CHECK:STDOUT: !30 = !DILocation(line: 13, column: 41, scope: !26)
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "TestConvertF128ToI64", linkageName: "_CTestConvertF128ToI64.Main", scope: null, file: !1, line: 16, type: !32, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !35)
// CHECK:STDOUT: !32 = !DISubroutineType(types: !33)
// CHECK:STDOUT: !33 = !{!34, !8}
// CHECK:STDOUT: !34 = !DIBasicType(name: "int", size: 64, encoding: DW_ATE_signed)
// CHECK:STDOUT: !35 = !{!36}
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !31, type: !8)
// CHECK:STDOUT: !37 = !DILocation(line: 16, column: 50, scope: !31)
// CHECK:STDOUT: !38 = !DILocation(line: 16, column: 43, scope: !31)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "TestConvertF128ToU64", linkageName: "_CTestConvertF128ToU64.Main", scope: null, file: !1, line: 19, type: !40, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !43)
// CHECK:STDOUT: !40 = !DISubroutineType(types: !41)
// CHECK:STDOUT: !41 = !{!42, !8}
// CHECK:STDOUT: !42 = !DIBasicType(name: "int", size: 64, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !43 = !{!44}
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 1, scope: !39, type: !8)
// CHECK:STDOUT: !45 = !DILocation(line: 19, column: 50, scope: !39)
// CHECK:STDOUT: !46 = !DILocation(line: 19, column: 43, scope: !39)
