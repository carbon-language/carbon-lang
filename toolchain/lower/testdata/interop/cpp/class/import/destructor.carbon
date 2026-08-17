// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
// EXTRA-ARGS: --clang-arg=-std=c++20
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/class/import/destructor.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/class/import/destructor.carbon

// --- call_destructor.carbon
library "[[@TEST_NAME]]";

import Cpp;

inline Cpp '''
struct Trivial {};
struct Nontrivial {
  ~Nontrivial();
};
struct Constexpr {
  constexpr ~Constexpr() {}
};
''';

fn Destroy() {
  var _: Cpp.Trivial = {};
  var _: Cpp.Nontrivial = {};
  var _: Cpp.Constexpr = {};
}

interface Make {
  fn Make() -> Self;
}
impl Cpp.Trivial as Make {
  fn Make() -> Self { return Cpp.Trivial.Trivial(); }
}
impl Cpp.Nontrivial as Make {
  fn Make() -> Self { return Cpp.Nontrivial.Nontrivial(); }
}
impl Cpp.Constexpr as Make {
  fn Make() -> Self { return Cpp.Constexpr.Constexpr(); }
}

fn DestroyIndirectly(generic T: Core.Destroy & Make) {
  var unused a: T = T.Make();
}

fn DestroyAIndirectly() {
  DestroyIndirectly(Cpp.Trivial);
  DestroyIndirectly(Cpp.Nontrivial);
  DestroyIndirectly(Cpp.Constexpr);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_destructor.carbon'
// CHECK:STDOUT: source_filename = "call_destructor.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN9ConstexprD2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: @Trivial.val.loc16_3 = internal constant {} zeroinitializer
// CHECK:STDOUT: @Nontrivial.val.loc17_3 = internal constant {} zeroinitializer
// CHECK:STDOUT: @Constexpr.val.loc18_3 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CDestroy.Main() #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var.loc16 = alloca {}, align 1, !dbg !15
// CHECK:STDOUT:   %_.var.loc17 = alloca {}, align 1, !dbg !16
// CHECK:STDOUT:   %_.var.loc18 = alloca {}, align 1, !dbg !17
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc16), !dbg !15
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %_.var.loc16, ptr align 1 @Trivial.val.loc16_3, i64 0, i1 false), !dbg !15
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc17), !dbg !16
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %_.var.loc17, ptr align 1 @Nontrivial.val.loc17_3, i64 0, i1 false), !dbg !16
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc18), !dbg !17
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %_.var.loc18, ptr align 1 @Constexpr.val.loc18_3, i64 0, i1 false), !dbg !17
// CHECK:STDOUT:   call void @_ZN9ConstexprD2Ev(ptr %_.var.loc18), !dbg !17
// CHECK:STDOUT:   call void @_ZN10NontrivialD1Ev(ptr %_.var.loc17), !dbg !16
// CHECK:STDOUT:   ret void, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN9ConstexprD2Ev(ptr noundef nonnull align 1 dereferenceable(1) %this) unnamed_addr #1 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !19
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Destroy.74d6bfd440841f14.Core:Constexpr.Cpp"(ptr %self) #2 !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN9ConstexprD2Ev(ptr %self), !dbg !28
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: declare void @_ZN10NontrivialD1Ev(ptr noundef nonnull align 1 dereferenceable(1)) unnamed_addr #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Destroy.470f41253d102356.Core:Trivial.Cpp"(ptr %self) #2 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CMake.Trivial.Cpp:Make.Main"(ptr sret({}) %return) #0 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN7TrivialC1Ev.carbon_thunk.(ptr %return), !dbg !36
// CHECK:STDOUT:   ret void, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN7TrivialC1Ev.carbon_thunk.(ptr noundef %return) #4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !38
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !38
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CMake.Nontrivial.Cpp:Make.Main"(ptr sret({}) %return) #0 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN10NontrivialC1Ev.carbon_thunk.(ptr %return), !dbg !41
// CHECK:STDOUT:   ret void, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN10NontrivialC1Ev.carbon_thunk.(ptr noundef %return) #4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !43
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !43
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CMake.Constexpr.Cpp:Make.Main"(ptr sret({}) %return) #0 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN9ConstexprC1Ev.carbon_thunk.(ptr %return), !dbg !46
// CHECK:STDOUT:   ret void, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN9ConstexprC1Ev.carbon_thunk.(ptr noundef %return) #4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !19
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !19
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CDestroyAIndirectly.Main() #0 !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CDestroyIndirectly.Main.2a28a415478a31f5(), !dbg !49
// CHECK:STDOUT:   call void @_CDestroyIndirectly.Main.a0f524e76111d8a1(), !dbg !50
// CHECK:STDOUT:   call void @_CDestroyIndirectly.Main.bf648cc7ac7f04db(), !dbg !51
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #5
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #5
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CDestroyIndirectly.Main.2a28a415478a31f5() #0 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !54
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !54
// CHECK:STDOUT:   call void @"_CMake.Trivial.Cpp:Make.Main"(ptr %a.var), !dbg !55
// CHECK:STDOUT:   call void @"_COp:thunk:Destroy.470f41253d102356.Core:Trivial.Cpp"(ptr %a.var), !dbg !54
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CDestroyIndirectly.Main.a0f524e76111d8a1() #0 !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !58
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !58
// CHECK:STDOUT:   call void @"_CMake.Nontrivial.Cpp:Make.Main"(ptr %a.var), !dbg !59
// CHECK:STDOUT:   call void @_ZN10NontrivialD1Ev(ptr %a.var), !dbg !58
// CHECK:STDOUT:   ret void, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CDestroyIndirectly.Main.bf648cc7ac7f04db() #0 !dbg !61 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !62
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !62
// CHECK:STDOUT:   call void @"_CMake.Constexpr.Cpp:Make.Main"(ptr %a.var), !dbg !63
// CHECK:STDOUT:   call void @"_COp:thunk:Destroy.74d6bfd440841f14.Core:Constexpr.Cpp"(ptr %a.var), !dbg !62
// CHECK:STDOUT:   ret void, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_ZN9ConstexprD2Ev, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 2, 5, 4, 3 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #3 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #4 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #5 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3, !4, !5, !6}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!7}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "call_destructor.carbon", directory: "")
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
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Destroy", linkageName: "_CDestroy.Main", scope: null, file: !1, line: 15, type: !13, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{null}
// CHECK:STDOUT: !15 = !DILocation(line: 16, column: 3, scope: !12)
// CHECK:STDOUT: !16 = !DILocation(line: 17, column: 3, scope: !12)
// CHECK:STDOUT: !17 = !DILocation(line: 18, column: 3, scope: !12)
// CHECK:STDOUT: !18 = !DILocation(line: 15, column: 1, scope: !12)
// CHECK:STDOUT: !19 = !{!20, !20, i64 0}
// CHECK:STDOUT: !20 = !{!"p1 _ZTS9Constexpr", !21, i64 0}
// CHECK:STDOUT: !21 = !{!"any pointer", !10, i64 0}
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Destroy.74d6bfd440841f14.Core:Constexpr.Cpp", scope: null, file: !1, line: 11, type: !23, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !26)
// CHECK:STDOUT: !23 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !24 = !{null, !25}
// CHECK:STDOUT: !25 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !26 = !{!27}
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !22, type: !25)
// CHECK:STDOUT: !28 = !DILocation(line: 11, column: 13, scope: !22)
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Destroy.470f41253d102356.Core:Trivial.Cpp", scope: null, file: !1, line: 6, type: !23, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !30)
// CHECK:STDOUT: !30 = !{!31}
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 1, scope: !29, type: !25)
// CHECK:STDOUT: !32 = !DILocation(line: 6, column: 8, scope: !29)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Trivial.Cpp:Make.Main", scope: null, file: !1, line: 25, type: !34, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !34 = !DISubroutineType(types: !35)
// CHECK:STDOUT: !35 = !{!25}
// CHECK:STDOUT: !36 = !DILocation(line: 25, column: 30, scope: !33)
// CHECK:STDOUT: !37 = !DILocation(line: 25, column: 23, scope: !33)
// CHECK:STDOUT: !38 = !{!39, !39, i64 0}
// CHECK:STDOUT: !39 = !{!"p1 _ZTS7Trivial", !21, i64 0}
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Nontrivial.Cpp:Make.Main", scope: null, file: !1, line: 28, type: !34, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !41 = !DILocation(line: 28, column: 30, scope: !40)
// CHECK:STDOUT: !42 = !DILocation(line: 28, column: 23, scope: !40)
// CHECK:STDOUT: !43 = !{!44, !44, i64 0}
// CHECK:STDOUT: !44 = !{!"p1 _ZTS10Nontrivial", !21, i64 0}
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Constexpr.Cpp:Make.Main", scope: null, file: !1, line: 31, type: !34, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !46 = !DILocation(line: 31, column: 30, scope: !45)
// CHECK:STDOUT: !47 = !DILocation(line: 31, column: 23, scope: !45)
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "DestroyAIndirectly", linkageName: "_CDestroyAIndirectly.Main", scope: null, file: !1, line: 38, type: !13, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !49 = !DILocation(line: 39, column: 3, scope: !48)
// CHECK:STDOUT: !50 = !DILocation(line: 40, column: 3, scope: !48)
// CHECK:STDOUT: !51 = !DILocation(line: 41, column: 3, scope: !48)
// CHECK:STDOUT: !52 = !DILocation(line: 38, column: 1, scope: !48)
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "DestroyIndirectly", linkageName: "_CDestroyIndirectly.Main.2a28a415478a31f5", scope: null, file: !1, line: 34, type: !13, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !54 = !DILocation(line: 35, column: 3, scope: !53)
// CHECK:STDOUT: !55 = !DILocation(line: 35, column: 21, scope: !53)
// CHECK:STDOUT: !56 = !DILocation(line: 34, column: 1, scope: !53)
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "DestroyIndirectly", linkageName: "_CDestroyIndirectly.Main.a0f524e76111d8a1", scope: null, file: !1, line: 34, type: !13, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !58 = !DILocation(line: 35, column: 3, scope: !57)
// CHECK:STDOUT: !59 = !DILocation(line: 35, column: 21, scope: !57)
// CHECK:STDOUT: !60 = !DILocation(line: 34, column: 1, scope: !57)
// CHECK:STDOUT: !61 = distinct !DISubprogram(name: "DestroyIndirectly", linkageName: "_CDestroyIndirectly.Main.bf648cc7ac7f04db", scope: null, file: !1, line: 34, type: !13, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !62 = !DILocation(line: 35, column: 3, scope: !61)
// CHECK:STDOUT: !63 = !DILocation(line: 35, column: 21, scope: !61)
// CHECK:STDOUT: !64 = !DILocation(line: 34, column: 1, scope: !61)
// CHECK:STDOUT:
