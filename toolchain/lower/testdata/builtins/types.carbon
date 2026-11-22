// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/builtins/types.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/builtins/types.carbon

fn MakeIntLiteral() -> type = "int_literal.make_type";
alias IntLiteral = MakeIntLiteral();
fn MakeFloatLiteral() -> type = "float_literal.make_type";
alias FloatLiteral = MakeFloatLiteral();
fn Int(size: IntLiteral) -> type = "int.make_type_signed";
fn Float(size: IntLiteral) -> type = "float.make_type";
fn Bool() -> type = "bool.make_type";
fn AsI32(n: IntLiteral) -> Int(32) = "int.convert_checked";
fn AsF16(f: FloatLiteral) -> Float(16) = "float.convert_checked";
fn AsF32(f: FloatLiteral) -> Float(32) = "float.convert_checked";
fn AsF64(f: FloatLiteral) -> Float(64) = "float.convert_checked";
fn AsF128(f: FloatLiteral) -> Float(128) = "float.convert_checked";

fn IntParam(i: Int(32));
fn FloatParam(e: Float(16), f: Float(32), g: Float(64), h: Float(128));
fn BoolParam(b: Bool());

fn Uses() {
  IntParam(AsI32(1));
  FloatParam(AsF16(1.0), AsF32(1.0), AsF64(1.0), AsF128(1.0));
  BoolParam(false);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'types.carbon'
// CHECK:STDOUT: source_filename = "types.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CIntParam.Main(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CFloatParam.Main(half, float, double, fp128)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CBoolParam.Main(i1)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CUses.Main() #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CIntParam.Main(i32 1), !dbg !27
// CHECK:STDOUT:   call void @_CFloatParam.Main(half 1.000000e+00, float 1.000000e+00, double 1.000000e+00, fp128 1.000000e+00), !dbg !28
// CHECK:STDOUT:   call void @_CBoolParam.Main(i1 false), !dbg !29
// CHECK:STDOUT:   ret void, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!15}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "types.carbon", directory: "")
// CHECK:STDOUT: !15 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !14, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !24 = !{null}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "Uses", linkageName: "_CUses.Main", scope: null, file: !14, line: 30, type: !25, spFlags: DISPFlagDefinition, unit: !15)
// CHECK:STDOUT: !27 = !DILocation(line: 31, column: 3, scope: !26)
// CHECK:STDOUT: !28 = !DILocation(line: 32, column: 3, scope: !26)
// CHECK:STDOUT: !29 = !DILocation(line: 33, column: 3, scope: !26)
// CHECK:STDOUT: !30 = !DILocation(line: 30, column: 1, scope: !26)
// CHECK:STDOUT:
