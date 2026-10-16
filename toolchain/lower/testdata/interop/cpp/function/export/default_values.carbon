// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/function/export/default_values.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/function/export/default_values.carbon

// --- other_function.carbon

package OtherFunction;
fn F(unused i: i32 = 23) {}

// --- basic.carbon

library "[[@TEST_NAME]]";

import OtherFunction;
import Cpp inline '''
void CallF() {
  Carbon::OtherFunction::F();
}
''';

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'other_function.carbon'
// CHECK:STDOUT: source_filename = "other_function.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.OtherFunction(i32 %i) #0 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!23}
// CHECK:STDOUT: !llvm.module.flags = !{!6, !7}
// CHECK:STDOUT:
// CHECK:STDOUT: !6 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !7 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !22 = !DIFile(filename: "other_function.carbon", directory: "")
// CHECK:STDOUT: !23 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !22, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !24 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !25 = !{null, !24}
// CHECK:STDOUT: !26 = !DISubroutineType(types: !25)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "F", linkageName: "_CF.OtherFunction", scope: null, file: !22, line: 3, type: !26, spFlags: DISPFlagDefinition, unit: !23, retainedNodes: !31)
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 1, scope: !27, type: !24)
// CHECK:STDOUT: !30 = !DILocation(line: 3, column: 1, scope: !27)
// CHECK:STDOUT: !31 = !{!28}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'basic.carbon'
// CHECK:STDOUT: source_filename = "basic.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: @_C__F__0__default_value.Main = external global i32, align 4
// CHECK:STDOUT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @_C__global_init.Main, ptr null }]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress uwtable
// CHECK:STDOUT: define dso_local void @_Z5CallFv() #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %0 = load i32, ptr @_C__F__0__default_value.Main, align 4, !tbaa !3
// CHECK:STDOUT:   call void @_ZN6Carbon13OtherFunctionL1FEi(i32 noundef %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN6Carbon13OtherFunctionL1FEi(i32 noundef %0) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !3
// CHECK:STDOUT:   call void @_CF__carbon_thunk.OtherFunction(ptr noundef nonnull align 4 dereferenceable(4) %.addr)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.OtherFunction(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF__carbon_thunk.OtherFunction(ptr %_) #2 !dbg !34 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.1 = load i32, ptr %_, align 4, !dbg !36
// CHECK:STDOUT:   call void @_CF.OtherFunction(i32 %.1), !dbg !36
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Main() #2 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   store i32 23, ptr @_C__F__0__default_value.Main, align 4
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!33}
// CHECK:STDOUT: !llvm.module.flags = !{!42, !43, !44, !6, !7}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!46}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !1 = !{!"omnipotent char", !0, i64 0}
// CHECK:STDOUT: !2 = !{!"int", !1, i64 0}
// CHECK:STDOUT: !3 = !{!2, !2, i64 0}
// CHECK:STDOUT: !6 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !7 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !22 = !DIFile(filename: "other_function.carbon", directory: "")
// CHECK:STDOUT: !24 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !25 = !{null, !24}
// CHECK:STDOUT: !26 = !DISubroutineType(types: !25)
// CHECK:STDOUT: !32 = !DIFile(filename: "basic.carbon", directory: "")
// CHECK:STDOUT: !33 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !32, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "F__carbon_thunk", linkageName: "_CF__carbon_thunk.OtherFunction", scope: null, file: !22, line: 3, type: !26, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !37)
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !34, type: !24)
// CHECK:STDOUT: !36 = !DILocation(line: 3, column: 1, scope: !34)
// CHECK:STDOUT: !37 = !{!35}
// CHECK:STDOUT: !38 = !{null}
// CHECK:STDOUT: !39 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !32, type: !39, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !41 = !DILocation(line: 0, scope: !40)
// CHECK:STDOUT: !42 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !43 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !44 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !45 = !{!"__libc_errno", !2, i64 0}
// CHECK:STDOUT: !46 = !{!45, !2, i64 0}
// CHECK:STDOUT:
