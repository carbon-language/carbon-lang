// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/bool.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/if/else.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/if/else.carbon

fn F() {}
fn G() {}
fn H() {}

fn If(b: bool) {
  if (b) {
    F();
  } else {
    G();
  }
  H();
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'else.carbon'
// CHECK:STDOUT: source_filename = "else.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CH.Main() #0 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CIf.Main(i1 %b) #0 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   br i1 %b, label %if.then, label %if.else, !dbg !33
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   call void @_CF.Main(), !dbg !34
// CHECK:STDOUT:   br label %if.done, !dbg !35
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   call void @_CG.Main(), !dbg !36
// CHECK:STDOUT:   br label %if.done, !dbg !35
// CHECK:STDOUT:
// CHECK:STDOUT: if.done:                                          ; preds = %if.else, %if.then
// CHECK:STDOUT:   call void @_CH.Main(), !dbg !37
// CHECK:STDOUT:   ret void, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!17}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !16 = !DIFile(filename: "else.carbon", directory: "")
// CHECK:STDOUT: !17 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !16, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !18 = !{null}
// CHECK:STDOUT: !19 = !DISubroutineType(types: !18)
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{null, !20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !16, line: 13, type: !19, spFlags: DISPFlagDefinition, unit: !17)
// CHECK:STDOUT: !24 = !DILocation(line: 13, column: 1, scope: !23)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !16, line: 14, type: !19, spFlags: DISPFlagDefinition, unit: !17)
// CHECK:STDOUT: !26 = !DILocation(line: 14, column: 1, scope: !25)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main", scope: null, file: !16, line: 15, type: !19, spFlags: DISPFlagDefinition, unit: !17)
// CHECK:STDOUT: !28 = !DILocation(line: 15, column: 1, scope: !27)
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "If", linkageName: "_CIf.Main", scope: null, file: !16, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !17, retainedNodes: !39)
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 1, scope: !29, type: !20)
// CHECK:STDOUT: !33 = !DILocation(line: 18, column: 6, scope: !29)
// CHECK:STDOUT: !34 = !DILocation(line: 19, column: 5, scope: !29)
// CHECK:STDOUT: !35 = !DILocation(line: 18, column: 3, scope: !29)
// CHECK:STDOUT: !36 = !DILocation(line: 21, column: 5, scope: !29)
// CHECK:STDOUT: !37 = !DILocation(line: 23, column: 3, scope: !29)
// CHECK:STDOUT: !38 = !DILocation(line: 17, column: 1, scope: !29)
// CHECK:STDOUT: !39 = !{!30}
// CHECK:STDOUT:
