// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// EXTRA-ARGS: --clang-arg=-g
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/debug_info.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/debug_info.carbon

import Cpp;

inline Cpp '''
int f() { return 42; }
''';

fn Call() -> i32 {
  return Cpp.f();
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'debug_info.carbon'
// CHECK:STDOUT: source_filename = "debug_info.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define dso_local noundef i32 @_Z1fv() #0 !dbg !6 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 42, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CCall.Main() #1 !dbg !31 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %f.call = call i32 @_Z1fv(), !dbg !34
// CHECK:STDOUT:   ret i32 %f.call, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!1, !30}
// CHECK:STDOUT: !llvm.module.flags = !{!14, !15, !36, !37, !38}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!43}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !DIFile(filename: "<carbon Cpp imports>", directory: "", checksumkind: CSK_MD5, checksum: "9eecb7db59d16c80417c72d1e1f4fbf1")
// CHECK:STDOUT: !1 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !0, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
// CHECK:STDOUT: !2 = !DIFile(filename: "debug_info.carbon", directory: "")
// CHECK:STDOUT: !3 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !4 = !{!3}
// CHECK:STDOUT: !5 = !DISubroutineType(types: !4)
// CHECK:STDOUT: !6 = distinct !DISubprogram(name: "f", linkageName: "_Z1fv", scope: !2, file: !2, line: 18, type: !5, scopeLine: 18, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !1, keyInstructions: true)
// CHECK:STDOUT: !10 = !DILocation(line: 18, column: 11, scope: !6, atomGroup: 1, atomRank: 1)
// CHECK:STDOUT: !14 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !15 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !30 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !2, line: 21, type: !5, spFlags: DISPFlagDefinition, unit: !30)
// CHECK:STDOUT: !34 = !DILocation(line: 22, column: 10, scope: !31)
// CHECK:STDOUT: !35 = !DILocation(line: 22, column: 3, scope: !31)
// CHECK:STDOUT: !36 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !37 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !38 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !39 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !40 = !{!"omnipotent char", !39, i64 0}
// CHECK:STDOUT: !41 = !{!"int", !40, i64 0}
// CHECK:STDOUT: !42 = !{!"__libc_errno", !41, i64 0}
// CHECK:STDOUT: !43 = !{!42, !41, i64 0}
// CHECK:STDOUT:
