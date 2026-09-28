// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/function/export/constructor.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/function/export/constructor.carbon

// --- call_constructor.carbon
library "[[@TEST_NAME]]";

import Cpp;

class OneArg {
  fn OneArg(i: i32) -> Self;
}

inline Cpp '''
void f() {
  Carbon::OneArg a(42);
}
''';

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_constructor.carbon'
// CHECK:STDOUT: source_filename = "call_constructor.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %"class.Carbon::OneArg" = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN6Carbon6OneArgD2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress uwtable
// CHECK:STDOUT: define dso_local void @_Z1fv() #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a = alloca %"class.Carbon::OneArg", align 1
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a) #5
// CHECK:STDOUT:   call void @_ZN6Carbon6OneArgC2Ei(ptr noundef nonnull align 1 %a, i32 noundef 42)
// CHECK:STDOUT:   call void @_ZN6Carbon6OneArgD2Ev(ptr noundef nonnull align 1 %a) #5
// CHECK:STDOUT:   call void @llvm.lifetime.end.p0(ptr %a) #5
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN6Carbon6OneArgC2Ei(ptr noundef nonnull align 1 %this, i32 noundef %0) unnamed_addr #2 align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !40
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !42
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @_COneArg__carbon_thunk.OneArg.Main(ptr noundef nonnull align 1 %this1, ptr noundef nonnull align 4 dereferenceable(4) %.addr)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN6Carbon6OneArgD2Ev(ptr noundef nonnull align 1 %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !40
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @"_C__destroy_thunk:thunk.OneArg.Main"(ptr noundef nonnull align 1 %this1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.end.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_COneArg.OneArg.Main(ptr sret({}), i32)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_C__destroy_thunk:thunk.OneArg.Main"(ptr %self) #4 !dbg !15 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSelfDestruct.7040ef4e2429d520:core.Destroy.Core"(ptr %self), !dbg !17
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.cb8adced19392b99:core.Destroy.Core"(ptr %self) #5 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.7040ef4e2429d520:core.Destroy.Core"(ptr %self) #5 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.7040ef4e2429d520:core.Destroy.Core"(ptr %self) #5 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.7040ef4e2429d520:core.Destroy.Core"(ptr %self), !dbg !29
// CHECK:STDOUT:   ret void, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_COneArg__carbon_thunk.OneArg.Main(ptr %_, ptr %_1) #5 !dbg !31 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc6_28.2 = load i32, ptr %_1, align 4, !dbg !34
// CHECK:STDOUT:   call void @_COneArg.OneArg.Main(ptr %_, i32 %.loc6_28.2), !dbg !34
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #4 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #5 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!6}
// CHECK:STDOUT: !llvm.module.flags = !{!43, !44, !45, !3, !4}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!47}
// CHECK:STDOUT:
// CHECK:STDOUT: !3 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !4 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !5 = !DIFile(filename: "call_constructor.carbon", directory: "")
// CHECK:STDOUT: !6 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !5, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !11 = !{null, !8}
// CHECK:STDOUT: !12 = !DISubroutineType(types: !11)
// CHECK:STDOUT: !13 = !{null, !8, !7}
// CHECK:STDOUT: !14 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !15 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.OneArg.Main", scope: null, file: !5, line: 5, type: !12, spFlags: DISPFlagDefinition, unit: !6, retainedNodes: !18)
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 1, scope: !15, type: !8)
// CHECK:STDOUT: !17 = !DILocation(line: 5, column: 1, scope: !15)
// CHECK:STDOUT: !18 = !{!16}
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.cb8adced19392b99:core.Destroy.Core", scope: null, file: !5, line: 5, type: !12, spFlags: DISPFlagDefinition, unit: !6, retainedNodes: !22)
// CHECK:STDOUT: !20 = !DILocalVariable(arg: 1, scope: !19, type: !8)
// CHECK:STDOUT: !21 = !DILocation(line: 5, column: 1, scope: !19)
// CHECK:STDOUT: !22 = !{!20}
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.7040ef4e2429d520:core.Destroy.Core", scope: null, file: !5, line: 5, type: !12, spFlags: DISPFlagDefinition, unit: !6, retainedNodes: !26)
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !23, type: !8)
// CHECK:STDOUT: !25 = !DILocation(line: 5, column: 1, scope: !23)
// CHECK:STDOUT: !26 = !{!24}
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.7040ef4e2429d520:core.Destroy.Core", scope: null, file: !5, line: 5, type: !12, spFlags: DISPFlagDefinition, unit: !6, retainedNodes: !30)
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 1, scope: !27, type: !8)
// CHECK:STDOUT: !29 = !DILocation(line: 5, column: 1, scope: !27)
// CHECK:STDOUT: !30 = !{!28}
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "OneArg__carbon_thunk", linkageName: "_COneArg__carbon_thunk.OneArg.Main", scope: null, file: !5, line: 6, type: !14, spFlags: DISPFlagDefinition, unit: !6, retainedNodes: !35)
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !31, type: !8)
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 2, scope: !31, type: !7)
// CHECK:STDOUT: !34 = !DILocation(line: 6, column: 3, scope: !31)
// CHECK:STDOUT: !35 = !{!32, !33}
// CHECK:STDOUT: !36 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !37 = !{!"omnipotent char", !36, i64 0}
// CHECK:STDOUT: !38 = !{!"any pointer", !37, i64 0}
// CHECK:STDOUT: !39 = !{!"p1 _ZTSN6Carbon6OneArgE", !38, i64 0}
// CHECK:STDOUT: !40 = !{!39, !39, i64 0}
// CHECK:STDOUT: !41 = !{!"int", !37, i64 0}
// CHECK:STDOUT: !42 = !{!41, !41, i64 0}
// CHECK:STDOUT: !43 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !44 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !45 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !46 = !{!"__libc_errno", !41, i64 0}
// CHECK:STDOUT: !47 = !{!46, !41, i64 0}
// CHECK:STDOUT:
