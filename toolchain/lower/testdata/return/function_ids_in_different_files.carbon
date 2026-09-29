// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/return/function_ids_in_different_files.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/return/function_ids_in_different_files.carbon

// --- a.carbon
library "[[@TEST_NAME]]";
fn F(unused generic T: type) {
  // Explicit no-argument `return` in a generic function.
  return;
}

// --- b.carbon
library "[[@TEST_NAME]]";
import library "a";
fn G() {
  // Create specific in a different file.
  F({});
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'a.carbon'
// CHECK:STDOUT: source_filename = "a.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!15}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "a.carbon", directory: "")
// CHECK:STDOUT: !15 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !14, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'b.carbon'
// CHECK:STDOUT: source_filename = "b.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main.6944efcfbb6a8360(), !dbg !21
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CF.Main.6944efcfbb6a8360() #0 !dbg !23 {
// CHECK:STDOUT:   ret void, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!17}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !14 = !DIFile(filename: "a.carbon", directory: "")
// CHECK:STDOUT: !16 = !DIFile(filename: "b.carbon", directory: "")
// CHECK:STDOUT: !17 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !16, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !18 = !{null}
// CHECK:STDOUT: !19 = !DISubroutineType(types: !18)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !16, line: 3, type: !19, spFlags: DISPFlagDefinition, unit: !17)
// CHECK:STDOUT: !21 = !DILocation(line: 5, column: 3, scope: !20)
// CHECK:STDOUT: !22 = !DILocation(line: 3, column: 1, scope: !20)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.6944efcfbb6a8360", scope: null, file: !14, line: 2, type: !19, spFlags: DISPFlagDefinition, unit: !17)
// CHECK:STDOUT: !24 = !DILocation(line: 4, column: 3, scope: !23)
// CHECK:STDOUT:
