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

// CHECK:STDOUT: ; ModuleID = 'bool.carbon'
// CHECK:STDOUT: source_filename = "bool.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CF.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i1 false, !dbg !8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CT.Main() #0 !dbg !9 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i1 true, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CLoadStore.Main(ptr %p, i1 %b) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc22_17.2 = load i8, ptr %p, align 1, !dbg !17
// CHECK:STDOUT:   %.loc22_17.21 = trunc i8 %.loc22_17.2 to i1, !dbg !17
// CHECK:STDOUT:   %0 = zext i1 %b to i8, !dbg !18
// CHECK:STDOUT:   store i8 %0, ptr %p, align 1, !dbg !18
// CHECK:STDOUT:   ret i1 %.loc22_17.21, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "bool.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !3, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 8)
// CHECK:STDOUT: !8 = !DILocation(line: 14, column: 3, scope: !4)
// CHECK:STDOUT: !9 = distinct !DISubprogram(name: "T", linkageName: "_CT.Main", scope: null, file: !3, line: 17, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !10 = !DILocation(line: 18, column: 3, scope: !9)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "LoadStore", linkageName: "_CLoadStore.Main", scope: null, file: !3, line: 21, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !14)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{!7, !7, !7}
// CHECK:STDOUT: !14 = !{!15, !16}
// CHECK:STDOUT: !15 = !DILocalVariable(arg: 1, scope: !11, type: !7)
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 2, scope: !11, type: !7)
// CHECK:STDOUT: !17 = !DILocation(line: 22, column: 17, scope: !11)
// CHECK:STDOUT: !18 = !DILocation(line: 23, column: 3, scope: !11)
// CHECK:STDOUT: !19 = !DILocation(line: 24, column: 3, scope: !11)
