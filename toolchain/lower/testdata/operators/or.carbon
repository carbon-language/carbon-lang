// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/bool.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/operators/or.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/operators/or.carbon

fn F() -> bool { return true; }
fn G() -> bool { return true; }

fn Or() -> bool {
  return F() or G();
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'or.carbon'
// CHECK:STDOUT: source_filename = "or.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CF.Main() #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i1 true, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CG.Main() #0 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i1 true, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_COr.Main() #0 !dbg !31 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F.call = call i1 @_CF.Main(), !dbg !34
// CHECK:STDOUT:   %.loc17_14.3 = xor i1 %F.call, true, !dbg !34
// CHECK:STDOUT:   br i1 %.loc17_14.3, label %or.rhs, label %or.result, !dbg !34
// CHECK:STDOUT:
// CHECK:STDOUT: or.rhs:                                           ; preds = %entry
// CHECK:STDOUT:   %G.call = call i1 @_CG.Main(), !dbg !35
// CHECK:STDOUT:   br label %or.result, !dbg !34
// CHECK:STDOUT:
// CHECK:STDOUT: or.result:                                        ; preds = %or.rhs, %entry
// CHECK:STDOUT:   %0 = phi i1 [ true, %entry ], [ %G.call, %or.rhs ]
// CHECK:STDOUT:   ret i1 %0, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "or.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{!20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !18, line: 13, type: !22, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !26 = !DILocation(line: 13, column: 18, scope: !23)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !18, line: 14, type: !22, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !30 = !DILocation(line: 14, column: 18, scope: !27)
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "Or", linkageName: "_COr.Main", scope: null, file: !18, line: 16, type: !22, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !34 = !DILocation(line: 17, column: 10, scope: !31)
// CHECK:STDOUT: !35 = !DILocation(line: 17, column: 17, scope: !31)
// CHECK:STDOUT: !36 = !DILocation(line: 17, column: 3, scope: !31)
// CHECK:STDOUT:
