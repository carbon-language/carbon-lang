// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/class/export/mangle.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/class/export/mangle.carbon

// --- other.carbon
package Other;
import Cpp;

class Class1;

inline Cpp '''
void f0(Carbon::Class1*) {
}
''';

// --- namespace.carbon

library "[[@TEST_NAME]]";

import Other;
import Cpp;

class Class2;

inline Cpp '''

void f1(Carbon::Other::Class1*) {
}

void f2(Carbon::Class2*) {
}

''';

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'other.carbon'
// CHECK:STDOUT: source_filename = "other.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define dso_local void @_Z2f0PN6Carbon5Other6Class1E(ptr noundef %0) #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!12}
// CHECK:STDOUT: !llvm.module.flags = !{!13, !14, !15, !9, !10}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!18}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !1 = !{!"omnipotent char", !0, i64 0}
// CHECK:STDOUT: !2 = !{!"any pointer", !1, i64 0}
// CHECK:STDOUT: !3 = !{!"p1 _ZTSN6Carbon5Other6Class1E", !2, i64 0}
// CHECK:STDOUT: !4 = !{!3, !3, i64 0}
// CHECK:STDOUT: !9 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !10 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !11 = !DIFile(filename: "other.carbon", directory: "")
// CHECK:STDOUT: !12 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !11, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !13 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !14 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !15 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !16 = !{!"int", !1, i64 0}
// CHECK:STDOUT: !17 = !{!"__libc_errno", !16, i64 0}
// CHECK:STDOUT: !18 = !{!17, !16, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'namespace.carbon'
// CHECK:STDOUT: source_filename = "namespace.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define dso_local void @_Z2f1PN6Carbon5Other6Class1E(ptr noundef %0) #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define dso_local void @_Z2f2PN6Carbon4Main6Class2E(ptr noundef %0) #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !6
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!20}
// CHECK:STDOUT: !llvm.module.flags = !{!13, !14, !15, !9, !10}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!18}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !1 = !{!"omnipotent char", !0, i64 0}
// CHECK:STDOUT: !2 = !{!"any pointer", !1, i64 0}
// CHECK:STDOUT: !3 = !{!"p1 _ZTSN6Carbon5Other6Class1E", !2, i64 0}
// CHECK:STDOUT: !4 = !{!3, !3, i64 0}
// CHECK:STDOUT: !5 = !{!"p1 _ZTSN6Carbon4Main6Class2E", !2, i64 0}
// CHECK:STDOUT: !6 = !{!5, !5, i64 0}
// CHECK:STDOUT: !9 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !10 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !13 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !14 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !15 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !16 = !{!"int", !1, i64 0}
// CHECK:STDOUT: !17 = !{!"__libc_errno", !16, i64 0}
// CHECK:STDOUT: !18 = !{!17, !16, i64 0}
// CHECK:STDOUT: !19 = !DIFile(filename: "namespace.carbon", directory: "")
// CHECK:STDOUT: !20 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !19, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT:
