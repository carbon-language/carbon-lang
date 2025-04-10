// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/call_different_impls.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/call_different_impls.carbon

import Core library "io";

interface I {
  fn F();
}
class X {}
impl X as I {
  fn F() { Core.Print(1); }
}
class Y {}
impl Y as I {
  fn F() { Core.Print(2); }
}

// Cannot coalesce the lowering for G specifics, as they call different functions.
// Check different functions are still emitted when trying to deduplicate emitted definitons.
fn G(T:! I) { T.F(); }

fn Run() {
  G(X);
  G(Y);
}

// CHECK:STDOUT: ; ModuleID = 'call_different_impls.carbon'
// CHECK:STDOUT: source_filename = "call_different_impls.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @printf.int.format = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1
// CHECK:STDOUT:
// CHECK:STDOUT: define void @"_CF.X.Main:I.Main"() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %print.int = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 1), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @"_CF.Y.Main:I.Main"() !dbg !9 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %print.int = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 2), !dbg !10
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @main() !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CG.Main.562d5cdd5cdbd5c5(), !dbg !13
// CHECK:STDOUT:   call void @_CG.Main.386ac59331346fc4(), !dbg !14
// CHECK:STDOUT:   ret void, !dbg !15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @printf(ptr, ...)
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CG.Main.562d5cdd5cdbd5c5() !dbg !16 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CF.X.Main:I.Main"(), !dbg !17
// CHECK:STDOUT:   ret void, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CG.Main.386ac59331346fc4() !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CF.Y.Main:I.Main"(), !dbg !20
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @printf, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "call_different_impls.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.X.Main:I.Main", scope: null, file: !3, line: 18, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 18, column: 12, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 18, column: 3, scope: !4)
// CHECK:STDOUT: !9 = distinct !DISubprogram(name: "F", linkageName: "_CF.Y.Main:I.Main", scope: null, file: !3, line: 22, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !10 = !DILocation(line: 22, column: 12, scope: !9)
// CHECK:STDOUT: !11 = !DILocation(line: 22, column: 3, scope: !9)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !3, line: 29, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !13 = !DILocation(line: 30, column: 3, scope: !12)
// CHECK:STDOUT: !14 = !DILocation(line: 31, column: 3, scope: !12)
// CHECK:STDOUT: !15 = !DILocation(line: 29, column: 1, scope: !12)
// CHECK:STDOUT: !16 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.562d5cdd5cdbd5c5", scope: null, file: !3, line: 27, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !17 = !DILocation(line: 27, column: 15, scope: !16)
// CHECK:STDOUT: !18 = !DILocation(line: 27, column: 1, scope: !16)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.386ac59331346fc4", scope: null, file: !3, line: 27, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !20 = !DILocation(line: 27, column: 15, scope: !19)
// CHECK:STDOUT: !21 = !DILocation(line: 27, column: 1, scope: !19)
