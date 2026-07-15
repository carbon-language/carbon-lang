// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/array.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/array.carbon

// --- array_value_param.carbon
library "[[@TEST_NAME]]";

import Cpp;

fn F(a: array(i32, 5)) -> i32 { return a[2]; }

inline Cpp '''
int G() {
  int arr[5];
  return Carbon::F(arr);
}

int H() {
  return Carbon::F({1, 2, 3, 4, 5});
}
''';

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'array_value_param.carbon'
// CHECK:STDOUT: source_filename = "array_value_param.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: @constinit = private constant [5 x i32] [i32 1, i32 2, i32 3, i32 4, i32 5], align 4
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress uwtable
// CHECK:STDOUT: define dso_local noundef i32 @_Z1Gv() #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %arr = alloca [5 x i32], align 16
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %arr) #3
// CHECK:STDOUT:   %call = call noundef i32 @_ZN6CarbonL1FERA5_Ki(ptr noundef nonnull align 4 dereferenceable(20) %arr)
// CHECK:STDOUT:   call void @llvm.lifetime.end.p0(ptr %arr) #3
// CHECK:STDOUT:   ret i32 %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal noundef i32 @_ZN6CarbonL1FERA5_Ki(ptr noundef nonnull align 4 dereferenceable(20) %0) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %retval = alloca i32, align 4
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !12
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !12, !nonnull !15, !align !16
// CHECK:STDOUT:   call void @_CF__carbon_thunk.Main(ptr noundef nonnull align 4 dereferenceable(20) %1, ptr noundef nonnull align 4 dereferenceable(4) %retval)
// CHECK:STDOUT:   %2 = load i32, ptr %retval, align 4
// CHECK:STDOUT:   ret i32 %2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.end.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress uwtable
// CHECK:STDOUT: define dso_local noundef i32 @_Z1Hv() #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ref.tmp = alloca [5 x i32], align 4
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ref.tmp) #3
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %ref.tmp, ptr align 4 @constinit, i64 20, i1 false), !tbaa.struct !17
// CHECK:STDOUT:   %call = call noundef i32 @_ZN6CarbonL1FERA5_Ki(ptr noundef nonnull align 4 dereferenceable(20) %ref.tmp)
// CHECK:STDOUT:   call void @llvm.lifetime.end.p0(ptr %ref.tmp) #3
// CHECK:STDOUT:   ret i32 %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CF.Main(ptr %a) #3 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc5_43.2.array.index = getelementptr inbounds [5 x i32], ptr %a, i32 0, i32 2, !dbg !26
// CHECK:STDOUT:   %.loc5_43.3 = load i32, ptr %.loc5_43.2.array.index, align 4, !dbg !26
// CHECK:STDOUT:   ret i32 %.loc5_43.3, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF__carbon_thunk.Main(ptr %_, ptr %_1) #3 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main(ptr %_), !dbg !34
// CHECK:STDOUT:   store i32 %F.call, ptr %_1, align 4, !dbg !34
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3, !4, !5, !6}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!7}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "array_value_param.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !3 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !5 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !6 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !{!8, !9, i64 0}
// CHECK:STDOUT: !8 = !{!"__libc_errno", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"int", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"omnipotent char", !11, i64 0}
// CHECK:STDOUT: !11 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !12 = !{!13, !13, i64 0}
// CHECK:STDOUT: !13 = !{!"p1 int", !14, i64 0}
// CHECK:STDOUT: !14 = !{!"any pointer", !10, i64 0}
// CHECK:STDOUT: !15 = !{}
// CHECK:STDOUT: !16 = !{i64 4}
// CHECK:STDOUT: !17 = !{i64 0, i64 20, !18}
// CHECK:STDOUT: !18 = !{!10, !10, i64 0}
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !1, line: 5, type: !20, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !24)
// CHECK:STDOUT: !20 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !21 = !{!22, !23}
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!25}
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !19, type: !23)
// CHECK:STDOUT: !26 = !DILocation(line: 5, column: 40, scope: !19)
// CHECK:STDOUT: !27 = !DILocation(line: 5, column: 33, scope: !19)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "F__carbon_thunk", linkageName: "_CF__carbon_thunk.Main", scope: null, file: !1, line: 5, type: !29, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !31)
// CHECK:STDOUT: !29 = !DISubroutineType(types: !30)
// CHECK:STDOUT: !30 = !{null, !23, !22}
// CHECK:STDOUT: !31 = !{!32, !33}
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !28, type: !23)
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 2, scope: !28, type: !22)
// CHECK:STDOUT: !34 = !DILocation(line: 5, column: 1, scope: !28)
// CHECK:STDOUT:
