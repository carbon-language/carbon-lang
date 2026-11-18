// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/bool.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/operators/not.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/operators/not.carbon

fn Not(b: bool) -> bool {
  return not b;
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'not.carbon'
// CHECK:STDOUT: source_filename = "not.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CNot.Main(i1 %b) #0 !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc14 = xor i1 %b, true, !dbg !27
// CHECK:STDOUT:   ret i1 %.loc14, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!17}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !16 = !DIFile(filename: "not.carbon", directory: "")
// CHECK:STDOUT: !17 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !16, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !19 = !{!18, !18}
// CHECK:STDOUT: !20 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "Not", linkageName: "_CNot.Main", scope: null, file: !16, line: 13, type: !20, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !29)
// CHECK:STDOUT: !22 = !DILocalVariable(arg: 1, scope: !21, type: !18)
// CHECK:STDOUT: !27 = !DILocation(line: 14, column: 10, scope: !21)
// CHECK:STDOUT: !28 = !DILocation(line: 14, column: 3, scope: !21)
// CHECK:STDOUT: !29 = !{!22}
// CHECK:STDOUT:
