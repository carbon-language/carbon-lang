// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/bool.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/primitives/bool.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/primitives/bool.carbon

fn F() -> bool {
  return false;
}

fn T() -> bool {
  return true;
}

fn LoadStore(p: bool*, b: bool) -> bool {
  let v: bool = *p;
  *p = b;
  return v;
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'bool.carbon'
// CHECK:STDOUT: source_filename = "bool.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CF.Main() #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i1 false, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CT.Main() #0 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i1 true, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CLoadStore.Main(ptr %p, i1 %b) #0 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc22_17.2 = load i8, ptr %p, align 1, !dbg !41
// CHECK:STDOUT:   %.loc22_17.21 = trunc i8 %.loc22_17.2 to i1, !dbg !41
// CHECK:STDOUT:   %0 = zext i1 %b to i8, !dbg !45
// CHECK:STDOUT:   store i8 %0, ptr %p, align 1, !dbg !45
// CHECK:STDOUT:   ret i1 %.loc22_17.21, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "bool.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{!20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !{!20, !20, !20}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !18, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !28 = !DILocation(line: 14, column: 3, scope: !25)
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "T", linkageName: "_CT.Main", scope: null, file: !18, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !32 = !DILocation(line: 18, column: 3, scope: !29)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "LoadStore", linkageName: "_CLoadStore.Main", scope: null, file: !18, line: 21, type: !24, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !49)
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 1, scope: !33, type: !20)
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 2, scope: !33, type: !20)
// CHECK:STDOUT: !41 = !DILocation(line: 22, column: 17, scope: !33)
// CHECK:STDOUT: !45 = !DILocation(line: 23, column: 3, scope: !33)
// CHECK:STDOUT: !48 = !DILocation(line: 24, column: 3, scope: !33)
// CHECK:STDOUT: !49 = !{!34, !35}
// CHECK:STDOUT:
