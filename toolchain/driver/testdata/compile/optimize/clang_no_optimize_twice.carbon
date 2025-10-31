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

// CHECK:STDOUT: ; ModuleID = 'foo.carbon'
// CHECK:STDOUT: source_filename = "foo.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: $_Z1fv = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: noinline nounwind optnone
// CHECK:STDOUT: define i32 @_CCall.Main() #0 !dbg !7 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %g.call = call i32 @_Z1gv(), !dbg !10
// CHECK:STDOUT:   ret i32 %g.call, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress noinline optnone
// CHECK:STDOUT: define dso_local i32 @_Z1gv() #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %call = call i32 @_Z1fv()
// CHECK:STDOUT:   ret i32 %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress noinline nounwind optnone
// CHECK:STDOUT: define linkonce_odr dso_local i32 @_Z1fv() #2 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { noinline nounwind optnone }
// CHECK:STDOUT: attributes #1 = { mustprogress noinline optnone "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="0" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { mustprogress noinline nounwind optnone "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="0" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 1, !"wchar_size", i32 4}
// CHECK:STDOUT: !3 = !{i32 8, !"PIC Level", i32 0}
// CHECK:STDOUT: !4 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "foo.carbon", directory: "")
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !6, line: 12, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !8 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !9 = !{}
// CHECK:STDOUT: !10 = !DILocation(line: 13, column: 10, scope: !7)
// CHECK:STDOUT: !11 = !DILocation(line: 13, column: 3, scope: !7)
