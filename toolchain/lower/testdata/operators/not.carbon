// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/operators/not.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/operators/not.carbon

fn Not(b: bool) -> bool {
  return not b;
}

// CHECK:STDOUT: ; ModuleID = 'not.carbon'
// CHECK:STDOUT: source_filename = "not.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @_CNot.Main(i1 %b) !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12 = xor i1 %b, true, !dbg !7
// CHECK:STDOUT:   ret i1 %.loc12, !dbg !8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "not.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Not", linkageName: "_CNot.Main", scope: null, file: !3, line: 11, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 12, column: 10, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 12, column: 3, scope: !4)
