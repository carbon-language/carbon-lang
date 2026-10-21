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
// CHECK:STDOUT: define i1 @_CF.Main() #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i1 false, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CT.Main() #0 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i1 true, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CLoadStore.Main(ptr %p, i1 %b) #0 !dbg !31 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc22_17.2 = load i8, ptr %p, align 1, !dbg !39
// CHECK:STDOUT:   %.loc22_17.21 = trunc i8 %.loc22_17.2 to i1, !dbg !39
// CHECK:STDOUT:   %0 = zext i1 %b to i8, !dbg !43
// CHECK:STDOUT:   store i8 %0, ptr %p, align 1, !dbg !43
// CHECK:STDOUT:   ret i1 %.loc22_17.21, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!17}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !16 = !DIFile(filename: "bool.carbon", directory: "")
// CHECK:STDOUT: !17 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !16, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !19 = !{!18}
// CHECK:STDOUT: !20 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !21 = !{!18, !18, !18}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !16, line: 13, type: !20, spFlags: DISPFlagDefinition, unit: !17)
// CHECK:STDOUT: !26 = !DILocation(line: 14, column: 3, scope: !23)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "T", linkageName: "_CT.Main", scope: null, file: !16, line: 17, type: !20, spFlags: DISPFlagDefinition, unit: !17)
// CHECK:STDOUT: !30 = !DILocation(line: 18, column: 3, scope: !27)
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "LoadStore", linkageName: "_CLoadStore.Main", scope: null, file: !16, line: 21, type: !22, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !47)
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !31, type: !18)
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 2, scope: !31, type: !18)
// CHECK:STDOUT: !39 = !DILocation(line: 22, column: 17, scope: !31)
// CHECK:STDOUT: !43 = !DILocation(line: 23, column: 3, scope: !31)
// CHECK:STDOUT: !46 = !DILocation(line: 24, column: 3, scope: !31)
// CHECK:STDOUT: !47 = !{!32, !33}
// CHECK:STDOUT:
