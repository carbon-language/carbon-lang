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

// CHECK:STDOUT: ; ModuleID = 'else.carbon'
// CHECK:STDOUT: source_filename = "else.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !8 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CH.Main() #0 !dbg !10 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CIf.Main(i1 %b) #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   br i1 %b, label %if.then, label %if.else, !dbg !18
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   call void @_CF.Main(), !dbg !19
// CHECK:STDOUT:   br label %if.done, !dbg !20
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   call void @_CG.Main(), !dbg !21
// CHECK:STDOUT:   br label %if.done, !dbg !20
// CHECK:STDOUT:
// CHECK:STDOUT: if.done:                                          ; preds = %if.else, %if.then
// CHECK:STDOUT:   call void @_CH.Main(), !dbg !22
// CHECK:STDOUT:   ret void, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "else.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !1, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 13, column: 1, scope: !4)
// CHECK:STDOUT: !8 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !1, line: 14, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !9 = !DILocation(line: 14, column: 1, scope: !8)
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main", scope: null, file: !1, line: 15, type: !5, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !11 = !DILocation(line: 15, column: 1, scope: !10)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "If", linkageName: "_CIf.Main", scope: null, file: !1, line: 17, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !16)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{null, !15}
// CHECK:STDOUT: !15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !16 = !{!17}
// CHECK:STDOUT: !17 = !DILocalVariable(arg: 1, scope: !12, type: !15)
// CHECK:STDOUT: !18 = !DILocation(line: 18, column: 6, scope: !12)
// CHECK:STDOUT: !19 = !DILocation(line: 19, column: 5, scope: !12)
// CHECK:STDOUT: !20 = !DILocation(line: 18, column: 3, scope: !12)
// CHECK:STDOUT: !21 = !DILocation(line: 21, column: 5, scope: !12)
// CHECK:STDOUT: !22 = !DILocation(line: 23, column: 3, scope: !12)
// CHECK:STDOUT: !23 = !DILocation(line: 17, column: 1, scope: !12)
