// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/function/import/cpp_run.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/function/import/cpp_run.carbon

// --- carbon_run.carbon

import Cpp inline '''
namespace N {
  void Run() {}
}
''';

fn Run() {
  Cpp.N.Run();
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'carbon_run.carbon'
// CHECK:STDOUT: source_filename = "carbon_run.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define dso_local void @_ZN1N3RunEv() #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #1 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN1N3RunEv(), !dbg !26
// CHECK:STDOUT:   ret i32 0, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!28, !29, !30, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!35}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "carbon_run.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{!20}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !18, line: 8, type: !22, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !26 = !DILocation(line: 9, column: 3, scope: !25)
// CHECK:STDOUT: !27 = !DILocation(line: 8, column: 1, scope: !25)
// CHECK:STDOUT: !28 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !29 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !30 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !31 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !32 = !{!"omnipotent char", !31, i64 0}
// CHECK:STDOUT: !33 = !{!"int", !32, i64 0}
// CHECK:STDOUT: !34 = !{!"__libc_errno", !33, i64 0}
// CHECK:STDOUT: !35 = !{!34, !33, i64 0}
// CHECK:STDOUT:
