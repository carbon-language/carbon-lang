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

// CHECK:STDOUT: ; ModuleID = 'debug_info.carbon'
// CHECK:STDOUT: source_filename = "debug_info.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define dso_local noundef i32 @_Z1fv() #0 !dbg !13 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 42, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CCall.Main() #1 !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %f.call = call i32 @_Z1fv(), !dbg !19
// CHECK:STDOUT:   ret i32 %f.call, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0, !2}
// CHECK:STDOUT: !llvm.module.flags = !{!4, !5, !6, !7, !8}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!9}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
// CHECK:STDOUT: !1 = !DIFile(filename: "<carbon Cpp imports>", directory: "", checksumkind: CSK_MD5, checksum: "d41d8cd98f00b204e9800998ecf8427e")
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "debug_info.carbon", directory: "")
// CHECK:STDOUT: !4 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !5 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !7 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !8 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !9 = !{!10, !10, i64 0}
// CHECK:STDOUT: !10 = !{!"int", !11, i64 0}
// CHECK:STDOUT: !11 = !{!"omnipotent char", !12, i64 0}
// CHECK:STDOUT: !12 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !13 = distinct !DISubprogram(name: "f", linkageName: "_Z1fv", scope: !3, file: !3, line: 18, type: !14, scopeLine: 18, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, keyInstructions: true)
// CHECK:STDOUT: !14 = !DISubroutineType(types: !15)
// CHECK:STDOUT: !15 = !{!16}
// CHECK:STDOUT: !16 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !17 = !DILocation(line: 18, column: 11, scope: !13, atomGroup: 1, atomRank: 1)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !3, line: 21, type: !14, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !19 = !DILocation(line: 22, column: 10, scope: !18)
// CHECK:STDOUT: !20 = !DILocation(line: 22, column: 3, scope: !18)
