// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/operators/and.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/operators/and.carbon

fn F() -> bool { return true; }
fn G() -> bool { return true; }

fn And() -> bool {
  return F() and G();
}

// CHECK:STDOUT: ; ModuleID = 'and.carbon'
// CHECK:STDOUT: source_filename = "and.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @F() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i1 true
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @G() !dbg !7 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i1 true
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i1 @And() !dbg !8 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F.call = call i1 @F()
// CHECK:STDOUT:   br i1 %F.call, label %and.rhs, label %and.result
// CHECK:STDOUT:
// CHECK:STDOUT: and.rhs:                                          ; preds = %entry
// CHECK:STDOUT:   %G.call = call i1 @G()
// CHECK:STDOUT:   br label %and.result
// CHECK:STDOUT:
// CHECK:STDOUT: and.result:                                       ; preds = %and.rhs, %entry
// CHECK:STDOUT:   %0 = phi i1 [ false, %entry ], [ %G.call, %and.rhs ]
// CHECK:STDOUT:   ret i1 %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "and.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F", scope: null, file: !3, line: 11, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "G", scope: null, file: !3, line: 12, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !8 = distinct !DISubprogram(name: "And", scope: null, file: !3, line: 14, type: !5, spFlags: DISPFlagDefinition, unit: !2)
