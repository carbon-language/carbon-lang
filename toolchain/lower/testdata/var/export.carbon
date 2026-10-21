// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/var/export.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/var/export.carbon

// --- var_export.carbon
library "[[@TEST_NAME]]";

import Cpp;

var x: i32;

inline Cpp '''
int Get() {
  return Carbon::x;
}
''';

fn Run() -> i32 {
  return Cpp.Get();
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'var_export.carbon'
// CHECK:STDOUT: source_filename = "var_export.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: @_Cx.Main = global i32 0, align 4
// CHECK:STDOUT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @_C__global_init.Main, ptr null }]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define dso_local noundef i32 @_Z3Getv() #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %0 = load i32, ptr @_Cx.Main, align 4, !tbaa !3
// CHECK:STDOUT:   ret i32 %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @main() #1 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Get.call = call i32 @_Z3Getv(), !dbg !30
// CHECK:STDOUT:   ret i32 %Get.call, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Main() #1 !dbg !34 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   store i32 poison, ptr @_Cx.Main, align 4, !dbg !35
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!23}
// CHECK:STDOUT: !llvm.module.flags = !{!37, !38, !39, !6, !7}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!41}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !1 = !{!"omnipotent char", !0, i64 0}
// CHECK:STDOUT: !2 = !{!"int", !1, i64 0}
// CHECK:STDOUT: !3 = !{!2, !2, i64 0}
// CHECK:STDOUT: !6 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !7 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !22 = !DIFile(filename: "var_export.carbon", directory: "")
// CHECK:STDOUT: !23 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !22, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !24 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !25 = !{!24}
// CHECK:STDOUT: !26 = !DISubroutineType(types: !25)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "Run", linkageName: "main", scope: null, file: !22, line: 13, type: !26, spFlags: DISPFlagDefinition, unit: !23)
// CHECK:STDOUT: !30 = !DILocation(line: 14, column: 10, scope: !27)
// CHECK:STDOUT: !31 = !DILocation(line: 14, column: 3, scope: !27)
// CHECK:STDOUT: !32 = !{null}
// CHECK:STDOUT: !33 = !DISubroutineType(types: !32)
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !22, type: !33, spFlags: DISPFlagDefinition, unit: !23)
// CHECK:STDOUT: !35 = !DILocation(line: 5, column: 1, scope: !34)
// CHECK:STDOUT: !36 = !DILocation(line: 0, scope: !34)
// CHECK:STDOUT: !37 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !38 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !39 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !40 = !{!"__libc_errno", !2, i64 0}
// CHECK:STDOUT: !41 = !{!40, !2, i64 0}
// CHECK:STDOUT:
