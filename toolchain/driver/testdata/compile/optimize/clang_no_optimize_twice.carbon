// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: compile --optimize=none foo.carbon --target=x86_64-unknown-linux-gnu --phase=optimize --dump-llvm-ir --exclude-dump-file-prefix=%{core} --clang-arg=-O3
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/driver/testdata/compile/optimize/clang_no_optimize_twice.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/driver/testdata/compile/optimize/clang_no_optimize_twice.carbon

// --- foo.carbon

import Cpp inline '''c++
// This should not be `optnone`, because Clang had `-O3`.
inline int f() { return 42; }

// The call to `f()` should not be inlined because Carbon had `--optimize=none`,
// so we don't run the inliner.
int g() { return f(); }
''';

// This should be emitted `optnone`.
fn Call() -> i32 {
  return Cpp.g();
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'foo.carbon'
// CHECK:STDOUT: source_filename = "foo.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: $_Z1fv = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress uwtable
// CHECK:STDOUT: define dso_local noundef i32 @_Z1gv() #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %call = call noundef i32 @_Z1fv()
// CHECK:STDOUT:   ret i32 %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local noundef i32 @_Z1fv() #1 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: noinline nounwind optnone
// CHECK:STDOUT: define i32 @_CCall.Main() #2 !dbg !5 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %g.call = call i32 @_Z1gv(), !dbg !8
// CHECK:STDOUT:   ret i32 %g.call, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { noinline nounwind optnone }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!1}
// CHECK:STDOUT: !llvm.module.flags = !{!10, !11, !12, !18, !19}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!17}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !DIFile(filename: "foo.carbon", directory: "")
// CHECK:STDOUT: !1 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !0, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !2 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !3 = !{!2}
// CHECK:STDOUT: !4 = !DISubroutineType(types: !3)
// CHECK:STDOUT: !5 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !0, line: 12, type: !4, spFlags: DISPFlagDefinition, unit: !1)
// CHECK:STDOUT: !8 = !DILocation(line: 13, column: 10, scope: !5)
// CHECK:STDOUT: !9 = !DILocation(line: 13, column: 3, scope: !5)
// CHECK:STDOUT: !10 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !11 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !12 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !13 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !14 = !{!"omnipotent char", !13, i64 0}
// CHECK:STDOUT: !15 = !{!"int", !14, i64 0}
// CHECK:STDOUT: !16 = !{!"__libc_errno", !15, i64 0}
// CHECK:STDOUT: !17 = !{!16, !15, i64 0}
// CHECK:STDOUT: !18 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !19 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT:
