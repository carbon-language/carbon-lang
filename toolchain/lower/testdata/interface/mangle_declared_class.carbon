// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interface/mangle_declared_class.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interface/mangle_declared_class.carbon

interface I {
  fn F();
}
class C;
impl C as I {
  fn F() {}
}

constraint N {
  extend require impls I;
}
class D;
impl D as N {
  fn F() {}
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'mangle_declared_class.carbon'
// CHECK:STDOUT: source_filename = "mangle_declared_class.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.C.Main:I.Main"() #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CF.D.Main:I.Main"() #0 !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!17}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !16 = !DIFile(filename: "mangle_declared_class.carbon", directory: "")
// CHECK:STDOUT: !17 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !16, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !18 = !{null}
// CHECK:STDOUT: !19 = !DISubroutineType(types: !18)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "F", linkageName: "_CF.C.Main:I.Main", scope: null, file: !16, line: 18, type: !19, spFlags: DISPFlagDefinition, unit: !17)
// CHECK:STDOUT: !21 = !DILocation(line: 18, column: 3, scope: !20)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "F", linkageName: "_CF.D.Main:I.Main", scope: null, file: !16, line: 26, type: !19, spFlags: DISPFlagDefinition, unit: !17)
// CHECK:STDOUT: !23 = !DILocation(line: 26, column: 3, scope: !22)
// CHECK:STDOUT:
