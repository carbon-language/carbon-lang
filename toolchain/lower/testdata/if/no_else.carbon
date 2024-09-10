// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/if/no_else.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/if/no_else.carbon

fn F() {}
fn G() {}

fn If(b: bool) {
  if (b) {
    F();
  }
  G();
}

// CHECK:STDOUT: ; ModuleID = 'no_else.carbon'
// CHECK:STDOUT: source_filename = "no_else.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CF.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CG.Main() !dbg !8 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CIf.Main(i1 %b) !dbg !10 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   br i1 %b, label %if.then, label %if.else, !dbg !11
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   call void @_CF.Main(), !dbg !12
// CHECK:STDOUT:   br label %if.else, !dbg !13
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %if.then, %entry
// CHECK:STDOUT:   call void @_CG.Main(), !dbg !14
// CHECK:STDOUT:   ret void, !dbg !15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "no_else.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !3, line: 11, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 11, column: 1, scope: !4)
// CHECK:STDOUT: !8 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !3, line: 12, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !9 = !DILocation(line: 12, column: 1, scope: !8)
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "If", linkageName: "_CIf.Main", scope: null, file: !3, line: 14, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !11 = !DILocation(line: 15, column: 6, scope: !10)
// CHECK:STDOUT: !12 = !DILocation(line: 16, column: 5, scope: !10)
// CHECK:STDOUT: !13 = !DILocation(line: 15, column: 3, scope: !10)
// CHECK:STDOUT: !14 = !DILocation(line: 18, column: 3, scope: !10)
// CHECK:STDOUT: !15 = !DILocation(line: 14, column: 1, scope: !10)
