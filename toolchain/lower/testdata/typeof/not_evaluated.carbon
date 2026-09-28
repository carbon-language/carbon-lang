// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/bool.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/typeof/not_evaluated.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/typeof/not_evaluated.carbon

// The operand of `typeof` should not be evaluated at runtime.

fn Make() -> bool;

fn F(a: bool, b: bool) -> bool {
  let c: typeof(Make()) = a;
  let unused d: typeof(a and b) = c;
  return c;
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'not_evaluated.carbon'
// CHECK:STDOUT: source_filename = "not_evaluated.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare i1 @_CMake.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CF.Main(i1 %a, i1 %b) #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i1 %a, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!17}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !16 = !DIFile(filename: "not_evaluated.carbon", directory: "")
// CHECK:STDOUT: !17 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !16, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{!18, !18, !18}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !16, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !38)
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !23, type: !18)
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 2, scope: !23, type: !18)
// CHECK:STDOUT: !37 = !DILocation(line: 20, column: 3, scope: !23)
// CHECK:STDOUT: !38 = !{!24, !25}
// CHECK:STDOUT:
