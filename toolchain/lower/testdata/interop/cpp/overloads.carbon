// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/overloads.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/overloads.carbon

// ============================================================================
// overload set
// ============================================================================

// --- import_overload_set.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''

int foo() {
  return 0;
}

int foo(int a) {
  return a + 1;
}

''';

fn F() -> i32 {
  var n: i32 = 0;
  n = Cpp.foo(n);
  return n;
}

// CHECK:STDOUT: ; ModuleID = 'import_overload_set.carbon'
// CHECK:STDOUT: source_filename = "import_overload_set.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CF.Main() !dbg !7 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !10
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !10
// CHECK:STDOUT:   store i32 0, ptr %n.var, align 4, !dbg !10
// CHECK:STDOUT:   %.loc18 = load i32, ptr %n.var, align 4, !dbg !11
// CHECK:STDOUT:   %foo.call = call i32 @_Z3fooi(i32 %.loc18), !dbg !12
// CHECK:STDOUT:   store i32 %foo.call, ptr %n.var, align 4, !dbg !13
// CHECK:STDOUT:   %.loc19 = load i32, ptr %n.var, align 4, !dbg !14
// CHECK:STDOUT:   ret i32 %.loc19, !dbg !15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress noinline nounwind optnone
// CHECK:STDOUT: define dso_local i32 @_Z3fooi(i32 %a) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.addr = alloca i32, align 4
// CHECK:STDOUT:   store i32 %a, ptr %a.addr, align 4
// CHECK:STDOUT:   %0 = load i32, ptr %a.addr, align 4
// CHECK:STDOUT:   %add = add nsw i32 %0, 1
// CHECK:STDOUT:   ret i32 %add
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #1 = { mustprogress noinline nounwind optnone "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="0" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
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
// CHECK:STDOUT: !6 = !DIFile(filename: "import_overload_set.carbon", directory: "")
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !6, line: 16, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !8 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !9 = !{}
// CHECK:STDOUT: !10 = !DILocation(line: 17, column: 3, scope: !7)
// CHECK:STDOUT: !11 = !DILocation(line: 18, column: 15, scope: !7)
// CHECK:STDOUT: !12 = !DILocation(line: 18, column: 7, scope: !7)
// CHECK:STDOUT: !13 = !DILocation(line: 18, column: 3, scope: !7)
// CHECK:STDOUT: !14 = !DILocation(line: 19, column: 10, scope: !7)
// CHECK:STDOUT: !15 = !DILocation(line: 19, column: 3, scope: !7)
