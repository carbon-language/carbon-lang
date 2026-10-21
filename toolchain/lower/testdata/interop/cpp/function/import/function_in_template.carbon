// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/function/import/function_in_template.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/function/import/function_in_template.carbon

// --- class_template.h

template<typename T> struct X {
  static void f(T t) {}
};

using Y = X<int>;

// --- use_class_template.carbon

library "[[@TEST_NAME]]";

import Cpp library "class_template.h";

fn F() {
  //@dump-sem-ir-begin
  Cpp.Y.f(42);
  //@dump-sem-ir-end
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'use_class_template.carbon'
// CHECK:STDOUT: source_filename = "use_class_template.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN1XIiE1fEi = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN1XIiE1fEi(i32 42), !dbg !26
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN1XIiE1fEi(i32 noundef %t) #1 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %t.addr = alloca i32, align 4
// CHECK:STDOUT:   store i32 %t, ptr %t.addr, align 4, !tbaa !31
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!32, !33, !34, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!36}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "use_class_template.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !21 = !{null}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !18, line: 6, type: !22, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !26 = !DILocation(line: 8, column: 3, scope: !25)
// CHECK:STDOUT: !27 = !DILocation(line: 6, column: 1, scope: !25)
// CHECK:STDOUT: !28 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !29 = !{!"omnipotent char", !28, i64 0}
// CHECK:STDOUT: !30 = !{!"int", !29, i64 0}
// CHECK:STDOUT: !31 = !{!30, !30, i64 0}
// CHECK:STDOUT: !32 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !33 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !34 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !35 = !{!"__libc_errno", !30, i64 0}
// CHECK:STDOUT: !36 = !{!35, !30, i64 0}
// CHECK:STDOUT:
