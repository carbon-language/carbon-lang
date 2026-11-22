// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/return/no_return_from_run.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/return/no_return_from_run.carbon

// We expect this to lower to LLVM IR that returns i32 0.
fn Run() {}

// LLVM IR for calls to Run should reflect the i32 call type of the lowered Run function.
fn Foo() {
  Run();
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'no_return_from_run.carbon'
// CHECK:STDOUT: source_filename = "no_return_from_run.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #0 !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 0, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CFoo.Main() #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Run.call = call i32 @main(), !dbg !24
// CHECK:STDOUT:   ret void, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!15}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "no_return_from_run.carbon", directory: "")
// CHECK:STDOUT: !15 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !14, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !16 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !17 = !{!16}
// CHECK:STDOUT: !18 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !19 = !{null}
// CHECK:STDOUT: !20 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !14, line: 14, type: !18, spFlags: DISPFlagDefinition, unit: !15)
// CHECK:STDOUT: !22 = !DILocation(line: 14, column: 1, scope: !21)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "Foo", linkageName: "_CFoo.Main", scope: null, file: !14, line: 17, type: !20, spFlags: DISPFlagDefinition, unit: !15)
// CHECK:STDOUT: !24 = !DILocation(line: 18, column: 3, scope: !23)
// CHECK:STDOUT: !25 = !DILocation(line: 17, column: 1, scope: !23)
// CHECK:STDOUT:
