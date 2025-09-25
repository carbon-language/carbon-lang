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

fn IntLiteral() -> type = "int_literal.make_type";
fn FloatLiteral() -> type = "float_literal.make_type";
fn Int(size: IntLiteral()) -> type = "int.make_type_signed";
fn Float(size: IntLiteral()) -> type = "float.make_type";
fn Bool() -> type = "bool.make_type";
fn AsI32(n: IntLiteral()) -> Int(32) = "int.convert_checked";
fn AsF16(f: FloatLiteral()) -> Float(16) = "float.convert_checked";
fn AsF32(f: FloatLiteral()) -> Float(32) = "float.convert_checked";
fn AsF64(f: FloatLiteral()) -> Float(64) = "float.convert_checked";
fn AsF128(f: FloatLiteral()) -> Float(128) = "float.convert_checked";

fn IntParam(i: Int(32));
fn FloatParam(e: Float(16), f: Float(32), g: Float(64), h: Float(128));
fn BoolParam(b: Bool());

fn Uses() {
  IntParam(AsI32(1));
  FloatParam(AsF16(1.0), AsF32(1.0), AsF64(1.0), AsF128(1.0));
  BoolParam(false);
}

// CHECK:STDOUT: ; ModuleID = 'types.carbon'
// CHECK:STDOUT: source_filename = "types.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CIntParam.Main(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CFloatParam.Main(half, float, double, fp128)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CBoolParam.Main(i1)
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CUses.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CIntParam.Main(i32 1), !dbg !7
// CHECK:STDOUT:   call void @_CFloatParam.Main(half 0xH3C00, float 1.000000e+00, double 1.000000e+00, fp128 0xL00000000000000003FFF000000000000), !dbg !8
// CHECK:STDOUT:   call void @_CBoolParam.Main(i1 false), !dbg !9
// CHECK:STDOUT:   ret void, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "types.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Uses", linkageName: "_CUses.Main", scope: null, file: !3, line: 28, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 29, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 30, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 31, column: 3, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 28, column: 1, scope: !4)
