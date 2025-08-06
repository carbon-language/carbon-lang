// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/thunk.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/thunk.carbon

// ============================================================================
// Parameter requires a thunk to be generated
// ============================================================================

// --- thunk.h

void foo(short x, int y, long z);

// --- import_thunk.carbon

library "[[@TEST_NAME]]";

import Cpp library "thunk.h";

fn MyF() {
  // TODO: Make sure we actually store the value 5 into `alloca`.
  // See https://github.com/carbon-language/carbon-lang/pull/5850/files#r2249030529
  Cpp.foo(5, 6, 7);
}

// CHECK:STDOUT: ; ModuleID = 'import_thunk.carbon'
// CHECK:STDOUT: source_filename = "import_thunk.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CMyF.Main() !dbg !7 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_18.1.temp = alloca i16, align 2, !dbg !10
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 2, ptr %.loc9_18.1.temp), !dbg !10
// CHECK:STDOUT:   call void @_Z3foosil.carbon_thunk(ptr %.loc9_18.1.temp, i32 6, i64 7), !dbg !10
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z3foosil(i16, i32, i64)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress
// CHECK:STDOUT: define dso_local void @_Z3foosil.carbon_thunk(ptr %x, i32 %y, i64 %z) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.addr = alloca ptr, align 8
// CHECK:STDOUT:   %y.addr = alloca i32, align 4
// CHECK:STDOUT:   %z.addr = alloca i64, align 8
// CHECK:STDOUT:   store ptr %x, ptr %x.addr, align 8
// CHECK:STDOUT:   store i32 %y, ptr %y.addr, align 4
// CHECK:STDOUT:   store i64 %z, ptr %z.addr, align 8
// CHECK:STDOUT:   %0 = load ptr, ptr %x.addr, align 8
// CHECK:STDOUT:   %1 = load i16, ptr %0, align 2
// CHECK:STDOUT:   %2 = load i32, ptr %y.addr, align 4
// CHECK:STDOUT:   %3 = load i64, ptr %z.addr, align 8
// CHECK:STDOUT:   call void @_Z3foosil(i16 signext %1, i32 %2, i64 %3)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="0" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
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
// CHECK:STDOUT: !6 = !DIFile(filename: "import_thunk.carbon", directory: "")
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "MyF", linkageName: "_CMyF.Main", scope: null, file: !6, line: 6, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !8 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !9 = !{}
// CHECK:STDOUT: !10 = !DILocation(line: 9, column: 3, scope: !7)
// CHECK:STDOUT: !11 = !DILocation(line: 6, column: 1, scope: !7)
