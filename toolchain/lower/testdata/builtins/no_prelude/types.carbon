// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/builtins/no_prelude/types.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/builtins/no_prelude/types.carbon

fn IntLiteral() -> type = "int_literal.make_type";
fn Int(size: IntLiteral()) -> type = "int.make_type_signed";
fn Float(size: IntLiteral()) -> type = "float.make_type";
fn Bool() -> type = "bool.make_type";
fn AsI32(n: IntLiteral()) -> Int(32) = "int.convert_checked";

fn F() {
  var i: Int(32) = AsI32(0);
  var f: Float(64) = 0.0;
  var b: Bool() = false;
}

// CHECK:STDOUT: ; ModuleID = 'types.carbon'
// CHECK:STDOUT: source_filename = "types.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CF.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %i.var = alloca i32, align 4, !dbg !7
// CHECK:STDOUT:   %f.var = alloca double, align 8, !dbg !7
// CHECK:STDOUT:   %b.var = alloca i1, align 1, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 4, ptr %i.var), !dbg !7
// CHECK:STDOUT:   store i32 0, ptr %i.var, align 4, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %f.var), !dbg !7
// CHECK:STDOUT:   store double 0.000000e+00, ptr %f.var, align 8, !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 1, ptr %b.var), !dbg !7
// CHECK:STDOUT:   store i1 false, ptr %b.var, align 1, !dbg !9
// CHECK:STDOUT:   ret void, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "types.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !3, line: 17, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 18, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 19, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 20, column: 3, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 17, column: 1, scope: !4)
