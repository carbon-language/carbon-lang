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
// CHECK:STDOUT: define void @_CDestroy.Main() #0 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var.loc16 = alloca {}, align 1, !dbg !42
// CHECK:STDOUT:   %_.var.loc17 = alloca {}, align 1, !dbg !44
// CHECK:STDOUT:   %_.var.loc18 = alloca {}, align 1, !dbg !46
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc16), !dbg !42
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %_.var.loc16, ptr align 1 @Trivial.val.loc16_3, i64 0, i1 false), !dbg !42
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc17), !dbg !44
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %_.var.loc17, ptr align 1 @Nontrivial.val.loc17_3, i64 0, i1 false), !dbg !44
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc18), !dbg !46
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %_.var.loc18, ptr align 1 @Constexpr.val.loc18_3, i64 0, i1 false), !dbg !46
// CHECK:STDOUT:   call void @_ZN9ConstexprD2Ev(ptr %_.var.loc18), !dbg !46
// CHECK:STDOUT:   call void @_ZN10NontrivialD1Ev(ptr %_.var.loc17), !dbg !44
// CHECK:STDOUT:   ret void, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN9ConstexprD2Ev(ptr noundef nonnull align 1 dereferenceable(1) %this) unnamed_addr #1 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !98
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Destroy.74d6bfd440841f14.Core:Constexpr.Cpp"(ptr %self) #2 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN9ConstexprD2Ev(ptr %self), !dbg !52
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: declare void @_ZN10NontrivialD1Ev(ptr noundef nonnull align 1 dereferenceable(1)) unnamed_addr #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Destroy.470f41253d102356.Core:Trivial.Cpp"(ptr %self) #2 !dbg !54 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CMake.Trivial.Cpp:Make.Main"(ptr sret({}) %return) #0 !dbg !59 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN7TrivialC1Ev.carbon_thunk.(ptr %return), !dbg !62
// CHECK:STDOUT:   ret void, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN7TrivialC1Ev.carbon_thunk.(ptr noundef %return) #4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !100
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !100
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CMake.Nontrivial.Cpp:Make.Main"(ptr sret({}) %return) #0 !dbg !64 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN10NontrivialC1Ev.carbon_thunk.(ptr %return), !dbg !67
// CHECK:STDOUT:   ret void, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN10NontrivialC1Ev.carbon_thunk.(ptr noundef %return) #4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !102
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !102
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CMake.Constexpr.Cpp:Make.Main"(ptr sret({}) %return) #0 !dbg !69 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN9ConstexprC1Ev.carbon_thunk.(ptr %return), !dbg !72
// CHECK:STDOUT:   ret void, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN9ConstexprC1Ev.carbon_thunk.(ptr noundef %return) #4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !98
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !98
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CDestroyAIndirectly.Main() #0 !dbg !74 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CDestroyIndirectly.Main.2a28a415478a31f5(), !dbg !75
// CHECK:STDOUT:   call void @_CDestroyIndirectly.Main.a0f524e76111d8a1(), !dbg !76
// CHECK:STDOUT:   call void @_CDestroyIndirectly.Main.bf648cc7ac7f04db(), !dbg !77
// CHECK:STDOUT:   ret void, !dbg !78
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #5
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #5
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CDestroyIndirectly.Main.2a28a415478a31f5() #0 !dbg !79 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !80
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !80
// CHECK:STDOUT:   call void @"_CMake.Trivial.Cpp:Make.Main"(ptr %a.var), !dbg !81
// CHECK:STDOUT:   call void @"_COp:thunk:Destroy.470f41253d102356.Core:Trivial.Cpp"(ptr %a.var), !dbg !80
// CHECK:STDOUT:   ret void, !dbg !83
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CDestroyIndirectly.Main.a0f524e76111d8a1() #0 !dbg !84 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !85
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !85
// CHECK:STDOUT:   call void @"_CMake.Nontrivial.Cpp:Make.Main"(ptr %a.var), !dbg !86
// CHECK:STDOUT:   call void @_ZN10NontrivialD1Ev(ptr %a.var), !dbg !85
// CHECK:STDOUT:   ret void, !dbg !88
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CDestroyIndirectly.Main.bf648cc7ac7f04db() #0 !dbg !89 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !90
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !90
// CHECK:STDOUT:   call void @"_CMake.Constexpr.Cpp:Make.Main"(ptr %a.var), !dbg !91
// CHECK:STDOUT:   call void @"_COp:thunk:Destroy.74d6bfd440841f14.Core:Constexpr.Cpp"(ptr %a.var), !dbg !90
// CHECK:STDOUT:   ret void, !dbg !93
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!33}
// CHECK:STDOUT: !llvm.module.flags = !{!103, !104, !105, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!108}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !32 = !DIFile(filename: "call_destructor.carbon", directory: "")
// CHECK:STDOUT: !33 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !32, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !34 = !{null}
// CHECK:STDOUT: !35 = !DISubroutineType(types: !34)
// CHECK:STDOUT: !36 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !37 = !{null, !36}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = !{!36}
// CHECK:STDOUT: !40 = !DISubroutineType(types: !39)
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "Destroy", linkageName: "_CDestroy.Main", scope: null, file: !32, line: 15, type: !35, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !42 = !DILocation(line: 16, column: 3, scope: !41)
// CHECK:STDOUT: !44 = !DILocation(line: 17, column: 3, scope: !41)
// CHECK:STDOUT: !46 = !DILocation(line: 18, column: 3, scope: !41)
// CHECK:STDOUT: !48 = !DILocation(line: 15, column: 1, scope: !41)
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Destroy.74d6bfd440841f14.Core:Constexpr.Cpp", scope: null, file: !32, line: 11, type: !38, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !53)
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 1, scope: !49, type: !36)
// CHECK:STDOUT: !52 = !DILocation(line: 11, column: 13, scope: !49)
// CHECK:STDOUT: !53 = !{!50}
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Destroy.470f41253d102356.Core:Trivial.Cpp", scope: null, file: !32, line: 6, type: !38, spFlags: DISPFlagDefinition, unit: !33, retainedNodes: !58)
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 1, scope: !54, type: !36)
// CHECK:STDOUT: !57 = !DILocation(line: 6, column: 8, scope: !54)
// CHECK:STDOUT: !58 = !{!55}
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Trivial.Cpp:Make.Main", scope: null, file: !32, line: 25, type: !40, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !62 = !DILocation(line: 25, column: 30, scope: !59)
// CHECK:STDOUT: !63 = !DILocation(line: 25, column: 23, scope: !59)
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Nontrivial.Cpp:Make.Main", scope: null, file: !32, line: 28, type: !40, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !67 = !DILocation(line: 28, column: 30, scope: !64)
// CHECK:STDOUT: !68 = !DILocation(line: 28, column: 23, scope: !64)
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Constexpr.Cpp:Make.Main", scope: null, file: !32, line: 31, type: !40, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !72 = !DILocation(line: 31, column: 30, scope: !69)
// CHECK:STDOUT: !73 = !DILocation(line: 31, column: 23, scope: !69)
// CHECK:STDOUT: !74 = distinct !DISubprogram(name: "DestroyAIndirectly", linkageName: "_CDestroyAIndirectly.Main", scope: null, file: !32, line: 38, type: !35, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !75 = !DILocation(line: 39, column: 3, scope: !74)
// CHECK:STDOUT: !76 = !DILocation(line: 40, column: 3, scope: !74)
// CHECK:STDOUT: !77 = !DILocation(line: 41, column: 3, scope: !74)
// CHECK:STDOUT: !78 = !DILocation(line: 38, column: 1, scope: !74)
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "DestroyIndirectly", linkageName: "_CDestroyIndirectly.Main.2a28a415478a31f5", scope: null, file: !32, line: 34, type: !35, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !80 = !DILocation(line: 35, column: 3, scope: !79)
// CHECK:STDOUT: !81 = !DILocation(line: 35, column: 21, scope: !79)
// CHECK:STDOUT: !83 = !DILocation(line: 34, column: 1, scope: !79)
// CHECK:STDOUT: !84 = distinct !DISubprogram(name: "DestroyIndirectly", linkageName: "_CDestroyIndirectly.Main.a0f524e76111d8a1", scope: null, file: !32, line: 34, type: !35, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !85 = !DILocation(line: 35, column: 3, scope: !84)
// CHECK:STDOUT: !86 = !DILocation(line: 35, column: 21, scope: !84)
// CHECK:STDOUT: !88 = !DILocation(line: 34, column: 1, scope: !84)
// CHECK:STDOUT: !89 = distinct !DISubprogram(name: "DestroyIndirectly", linkageName: "_CDestroyIndirectly.Main.bf648cc7ac7f04db", scope: null, file: !32, line: 34, type: !35, spFlags: DISPFlagDefinition, unit: !33)
// CHECK:STDOUT: !90 = !DILocation(line: 35, column: 3, scope: !89)
// CHECK:STDOUT: !91 = !DILocation(line: 35, column: 21, scope: !89)
// CHECK:STDOUT: !93 = !DILocation(line: 34, column: 1, scope: !89)
// CHECK:STDOUT: !94 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !95 = !{!"omnipotent char", !94, i64 0}
// CHECK:STDOUT: !96 = !{!"any pointer", !95, i64 0}
// CHECK:STDOUT: !97 = !{!"p1 _ZTS9Constexpr", !96, i64 0}
// CHECK:STDOUT: !98 = !{!97, !97, i64 0}
// CHECK:STDOUT: !99 = !{!"p1 _ZTS7Trivial", !96, i64 0}
// CHECK:STDOUT: !100 = !{!99, !99, i64 0}
// CHECK:STDOUT: !101 = !{!"p1 _ZTS10Nontrivial", !96, i64 0}
// CHECK:STDOUT: !102 = !{!101, !101, i64 0}
// CHECK:STDOUT: !103 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !104 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !105 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !106 = !{!"int", !95, i64 0}
// CHECK:STDOUT: !107 = !{!"__libc_errno", !106, i64 0}
// CHECK:STDOUT: !108 = !{!107, !106, i64 0}
// CHECK:STDOUT:
