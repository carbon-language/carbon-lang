// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/function/export/function.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/function/export/function.carbon

// --- other.carbon
package Other;
fn NoArgs() {}
fn BoolArg(a: bool) {
  a;
}
fn IntArg(a: i32) {
  a;
}
fn FloatArg(a: f32) {
  a;
}
fn IntReturn() -> i32 {
  return 123;
}
// --- function.carbon

library "[[@TEST_NAME]]";

import Other;
import Cpp inline '''
int G() {
  Carbon::Other::NoArgs();
  Carbon::Other::BoolArg(true);
  Carbon::Other::IntArg(123);
  Carbon::Other::FloatArg(1.5);
  return Carbon::Other::IntReturn();
}
''';

// --- single_file.carbon

library "[[@TEST_NAME]]";

import Cpp;

fn F() {}

inline Cpp '''
inline void G1() {
  Carbon::F();
}

// Call an inline function indirectly to ensure declarations are getting
// properly registered with the CodeGen consumer.
inline void G2() {
  G1();
}
''';

fn H() { Cpp.G2(); }

// CHECK:STDOUT: ; ModuleID = 'other.carbon'
// CHECK:STDOUT: source_filename = "other.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CNoArgs.Other() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CBoolArg.Other(i1 %a) #0 !dbg !8 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CIntArg.Other(i32 %a) #0 !dbg !15 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CFloatArg.Other(float %a) #0 !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CIntReturn.Other() #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 123, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "other.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "NoArgs", linkageName: "_CNoArgs.Other", scope: null, file: !3, line: 2, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 2, column: 1, scope: !4)
// CHECK:STDOUT: !8 = distinct !DISubprogram(name: "BoolArg", linkageName: "_CBoolArg.Other", scope: null, file: !3, line: 3, type: !9, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !12)
// CHECK:STDOUT: !9 = !DISubroutineType(types: !10)
// CHECK:STDOUT: !10 = !{null, !11}
// CHECK:STDOUT: !11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !12 = !{!13}
// CHECK:STDOUT: !13 = !DILocalVariable(arg: 1, scope: !8, type: !11)
// CHECK:STDOUT: !14 = !DILocation(line: 3, column: 1, scope: !8)
// CHECK:STDOUT: !15 = distinct !DISubprogram(name: "IntArg", linkageName: "_CIntArg.Other", scope: null, file: !3, line: 6, type: !16, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !19)
// CHECK:STDOUT: !16 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !17 = !{null, !18}
// CHECK:STDOUT: !18 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !19 = !{!20}
// CHECK:STDOUT: !20 = !DILocalVariable(arg: 1, scope: !15, type: !18)
// CHECK:STDOUT: !21 = !DILocation(line: 6, column: 1, scope: !15)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "FloatArg", linkageName: "_CFloatArg.Other", scope: null, file: !3, line: 9, type: !9, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !23)
// CHECK:STDOUT: !23 = !{!24}
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !22, type: !11)
// CHECK:STDOUT: !25 = !DILocation(line: 9, column: 1, scope: !22)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "IntReturn", linkageName: "_CIntReturn.Other", scope: null, file: !3, line: 12, type: !27, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !27 = !DISubroutineType(types: !28)
// CHECK:STDOUT: !28 = !{!18}
// CHECK:STDOUT: !29 = !DILocation(line: 13, column: 3, scope: !26)
// CHECK:STDOUT: ; ModuleID = 'function.carbon'
// CHECK:STDOUT: source_filename = "function.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress uwtable
// CHECK:STDOUT: define dso_local noundef i32 @_Z1Gv() #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN6Carbon5OtherL6NoArgsEv()
// CHECK:STDOUT:   call void @_ZN6Carbon5OtherL7BoolArgEb(i1 noundef zeroext true)
// CHECK:STDOUT:   call void @_ZN6Carbon5OtherL6IntArgEi(i32 noundef 123)
// CHECK:STDOUT:   call void @_ZN6Carbon5OtherL8FloatArgEf(float noundef 1.500000e+00)
// CHECK:STDOUT:   %call = call noundef i32 @_ZN6Carbon5OtherL9IntReturnEv()
// CHECK:STDOUT:   ret i32 %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN6Carbon5OtherL6NoArgsEv() #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CNoArgs__carbon_thunk.Other()
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN6Carbon5OtherL7BoolArgEb(i1 noundef zeroext %0) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca i8, align 1
// CHECK:STDOUT:   %storedv = zext i1 %0 to i8
// CHECK:STDOUT:   store i8 %storedv, ptr %.addr, align 1, !tbaa !11
// CHECK:STDOUT:   call void @_CBoolArg__carbon_thunk.Other(ptr noundef nonnull align 1 dereferenceable(1) %.addr)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN6Carbon5OtherL6IntArgEi(i32 noundef %0) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !7
// CHECK:STDOUT:   call void @_CIntArg__carbon_thunk.Other(ptr noundef nonnull align 4 dereferenceable(4) %.addr)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN6Carbon5OtherL8FloatArgEf(float noundef %0) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca float, align 4
// CHECK:STDOUT:   store float %0, ptr %.addr, align 4, !tbaa !13
// CHECK:STDOUT:   call void @_CFloatArg__carbon_thunk.Other(ptr noundef nonnull align 4 dereferenceable(4) %.addr)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal noundef i32 @_ZN6Carbon5OtherL9IntReturnEv() #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %retval = alloca i32, align 4
// CHECK:STDOUT:   call void @_CIntReturn__carbon_thunk.Other(ptr noundef nonnull align 4 dereferenceable(4) %retval)
// CHECK:STDOUT:   %0 = load i32, ptr %retval, align 4
// CHECK:STDOUT:   ret i32 %0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CNoArgs.Other()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CNoArgs__carbon_thunk.Other() #2 !dbg !15 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CNoArgs.Other(), !dbg !19
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CBoolArg.Other(i1)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CBoolArg__carbon_thunk.Other(ptr %_) #2 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.1 = load i8, ptr %_, align 1, !dbg !26
// CHECK:STDOUT:   %.11 = trunc i8 %.1 to i1, !dbg !26
// CHECK:STDOUT:   call void @_CBoolArg.Other(i1 %.11), !dbg !26
// CHECK:STDOUT:   ret void, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CIntArg.Other(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CIntArg__carbon_thunk.Other(ptr %_) #2 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.1 = load i32, ptr %_, align 4, !dbg !33
// CHECK:STDOUT:   call void @_CIntArg.Other(i32 %.1), !dbg !33
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CFloatArg.Other(float)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CFloatArg__carbon_thunk.Other(ptr %_) #2 !dbg !34 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.1 = load float, ptr %_, align 4, !dbg !37
// CHECK:STDOUT:   call void @_CFloatArg.Other(float %.1), !dbg !37
// CHECK:STDOUT:   ret void, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @_CIntReturn.Other()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CIntReturn__carbon_thunk.Other(ptr %_) #2 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %IntReturn.call = call i32 @_CIntReturn.Other(), !dbg !41
// CHECK:STDOUT:   store i32 %IntReturn.call, ptr %_, align 4, !dbg !41
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!7}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !3 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "function.carbon", directory: "")
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = !{!12, !12, i64 0}
// CHECK:STDOUT: !12 = !{!"bool", !9, i64 0}
// CHECK:STDOUT: !13 = !{!14, !14, i64 0}
// CHECK:STDOUT: !14 = !{!"float", !9, i64 0}
// CHECK:STDOUT: !15 = distinct !DISubprogram(name: "NoArgs__carbon_thunk", linkageName: "_CNoArgs__carbon_thunk.Other", scope: null, file: !16, line: 2, type: !17, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !16 = !DIFile(filename: "other.carbon", directory: "")
// CHECK:STDOUT: !17 = !DISubroutineType(types: !18)
// CHECK:STDOUT: !18 = !{null}
// CHECK:STDOUT: !19 = !DILocation(line: 2, column: 1, scope: !15)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "BoolArg__carbon_thunk", linkageName: "_CBoolArg__carbon_thunk.Other", scope: null, file: !16, line: 3, type: !21, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !24)
// CHECK:STDOUT: !21 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !22 = !{null, !23}
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!25}
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !20, type: !23)
// CHECK:STDOUT: !26 = !DILocation(line: 3, column: 1, scope: !20)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "IntArg__carbon_thunk", linkageName: "_CIntArg__carbon_thunk.Other", scope: null, file: !16, line: 6, type: !28, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !31)
// CHECK:STDOUT: !28 = !DISubroutineType(types: !29)
// CHECK:STDOUT: !29 = !{null, !30}
// CHECK:STDOUT: !30 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !31 = !{!32}
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !27, type: !30)
// CHECK:STDOUT: !33 = !DILocation(line: 6, column: 1, scope: !27)
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "FloatArg__carbon_thunk", linkageName: "_CFloatArg__carbon_thunk.Other", scope: null, file: !16, line: 9, type: !21, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !35)
// CHECK:STDOUT: !35 = !{!36}
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !34, type: !23)
// CHECK:STDOUT: !37 = !DILocation(line: 9, column: 1, scope: !34)
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "IntReturn__carbon_thunk", linkageName: "_CIntReturn__carbon_thunk.Other", scope: null, file: !16, line: 12, type: !28, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !39)
// CHECK:STDOUT: !39 = !{!40}
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !38, type: !30)
// CHECK:STDOUT: !41 = !DILocation(line: 12, column: 1, scope: !38)
// CHECK:STDOUT: ; ModuleID = 'single_file.carbon'
// CHECK:STDOUT: source_filename = "single_file.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: $_Z2G2v = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_Z2G1v = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF__carbon_thunk.Main() #0 !dbg !15 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Main(), !dbg !16
// CHECK:STDOUT:   ret void, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CH.Main() #0 !dbg !17 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z2G2v(), !dbg !18
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_Z2G2v() #1 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z2G1v()
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_Z2G1v() #1 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN6CarbonL1FEv()
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN6CarbonL1FEv() #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF__carbon_thunk.Main()
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { inlinehint mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!7}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !3 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "single_file.carbon", directory: "")
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !6, line: 6, type: !12, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{null}
// CHECK:STDOUT: !14 = !DILocation(line: 6, column: 1, scope: !11)
// CHECK:STDOUT: !15 = distinct !DISubprogram(name: "F__carbon_thunk", linkageName: "_CF__carbon_thunk.Main", scope: null, file: !6, line: 6, type: !12, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !16 = !DILocation(line: 6, column: 1, scope: !15)
// CHECK:STDOUT: !17 = distinct !DISubprogram(name: "H", linkageName: "_CH.Main", scope: null, file: !6, line: 20, type: !12, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !18 = !DILocation(line: 20, column: 10, scope: !17)
// CHECK:STDOUT: !19 = !DILocation(line: 20, column: 1, scope: !17)
