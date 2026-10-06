// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --include-carbon-core
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
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
fn G(generic T: I) { T.F(); }

fn Run() {
  G(X);
  G(Y);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_different_impls.carbon'
// CHECK:STDOUT: source_filename = "call_different_impls.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @printf.int.format = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.X.Main:I.Main"() #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 1), !dbg !12
// CHECK:STDOUT:   ret void, !dbg !13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.Y.Main:I.Main"() #0 !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Print.call = call i32 (ptr, ...) @printf(ptr @printf.int.format, i32 2), !dbg !15
// CHECK:STDOUT:   ret void, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !17 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CG.Main.d0cc85b6c89517e3(), !dbg !18
// CHECK:STDOUT:   call void @_CG.Main.e8d4d639bae9ba40(), !dbg !19
// CHECK:STDOUT:   ret i32 0, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @printf(ptr, ...)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CG.Main.d0cc85b6c89517e3() #0 !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CF.X.Main:I.Main"(), !dbg !22
// CHECK:STDOUT:   ret void, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CG.Main.e8d4d639bae9ba40() #0 !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CF.Y.Main:I.Main"(), !dbg !25
// CHECK:STDOUT:   ret void, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @printf, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "call_different_impls.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !{null}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !9 = !{!6}
// CHECK:STDOUT: !10 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "F", linkageName: "_CF.X.Main:I.Main", scope: null, file: !4, line: 21, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !12 = !DILocation(line: 21, column: 12, scope: !11)
// CHECK:STDOUT: !13 = !DILocation(line: 21, column: 3, scope: !11)
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "F", linkageName: "_CF.Y.Main:I.Main", scope: null, file: !4, line: 25, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !15 = !DILocation(line: 25, column: 12, scope: !14)
// CHECK:STDOUT: !16 = !DILocation(line: 25, column: 3, scope: !14)
// CHECK:STDOUT: !17 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !4, line: 32, type: !10, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !18 = !DILocation(line: 33, column: 3, scope: !17)
// CHECK:STDOUT: !19 = !DILocation(line: 34, column: 3, scope: !17)
// CHECK:STDOUT: !20 = !DILocation(line: 32, column: 1, scope: !17)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.d0cc85b6c89517e3", scope: null, file: !4, line: 30, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !22 = !DILocation(line: 30, column: 22, scope: !21)
// CHECK:STDOUT: !23 = !DILocation(line: 30, column: 1, scope: !21)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main.e8d4d639bae9ba40", scope: null, file: !4, line: 30, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !25 = !DILocation(line: 30, column: 22, scope: !24)
// CHECK:STDOUT: !26 = !DILocation(line: 30, column: 1, scope: !24)
// CHECK:STDOUT:
