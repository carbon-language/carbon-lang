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
// CHECK:STDOUT: @Trivial.val.loc16_3.1 = internal constant {} zeroinitializer
// CHECK:STDOUT: @Nontrivial.val.loc17_3.1 = internal constant {} zeroinitializer
// CHECK:STDOUT: @Constexpr.val.loc18_3.1 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CDestroy.Main() #0 !dbg !43 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var.loc16 = alloca {}, align 1, !dbg !44
// CHECK:STDOUT:   %_.var.loc17 = alloca {}, align 1, !dbg !46
// CHECK:STDOUT:   %_.var.loc18 = alloca {}, align 1, !dbg !48
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc16), !dbg !44
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %_.var.loc16, ptr align 1 @Trivial.val.loc16_3.1, i64 0, i1 false), !dbg !44
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc17), !dbg !46
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %_.var.loc17, ptr align 1 @Nontrivial.val.loc17_3.1, i64 0, i1 false), !dbg !46
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc18), !dbg !48
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %_.var.loc18, ptr align 1 @Constexpr.val.loc18_3.1, i64 0, i1 false), !dbg !48
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.8c85e43f6e9dad76(ptr %_.var.loc18), !dbg !48
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ed633dc4d3ef0663(ptr %_.var.loc17), !dbg !46
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.e75943c87eaa5f64(ptr %_.var.loc16), !dbg !44
// CHECK:STDOUT:   ret void, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN9ConstexprD2Ev(ptr noundef nonnull align 1 dereferenceable(1) %this) unnamed_addr #1 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !116
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:SubobjectDestroy.8a0a027716a70929.Core:Constexpr.Cpp"(ptr %self) #2 !dbg !51 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN9ConstexprD2Ev(ptr %self), !dbg !54
// CHECK:STDOUT:   ret void, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: declare void @_ZN10NontrivialD1Ev(ptr noundef nonnull align 1 dereferenceable(1)) unnamed_addr #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CMake.Trivial.Cpp:Make.Main"(ptr sret({}) %return) #0 !dbg !56 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN7TrivialC1Ev.carbon_thunk.(ptr %return), !dbg !59
// CHECK:STDOUT:   ret void, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN7TrivialC1Ev.carbon_thunk.(ptr noundef %return) #4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !118
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !118
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CMake.Nontrivial.Cpp:Make.Main"(ptr sret({}) %return) #0 !dbg !61 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN10NontrivialC1Ev.carbon_thunk.(ptr %return), !dbg !64
// CHECK:STDOUT:   ret void, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN10NontrivialC1Ev.carbon_thunk.(ptr noundef %return) #4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !120
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !120
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CMake.Constexpr.Cpp:Make.Main"(ptr sret({}) %return) #0 !dbg !66 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN9ConstexprC1Ev.carbon_thunk.(ptr %return), !dbg !69
// CHECK:STDOUT:   ret void, !dbg !70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN9ConstexprC1Ev.carbon_thunk.(ptr noundef %return) #4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !116
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !116
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CDestroyAIndirectly.Main() #0 !dbg !71 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CDestroyIndirectly.Main.d0a14ccc1d0e7f82(), !dbg !72
// CHECK:STDOUT:   call void @_CDestroyIndirectly.Main.b437c55fb4b99438(), !dbg !73
// CHECK:STDOUT:   call void @_CDestroyIndirectly.Main.587d6c5d61349179(), !dbg !74
// CHECK:STDOUT:   ret void, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #5
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #5
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.8c85e43f6e9dad76(ptr %t) #0 !dbg !76 {
// CHECK:STDOUT:   call void @"_COp:thunk:SubobjectDestroy.8a0a027716a70929.Core:Constexpr.Cpp"(ptr %t), !dbg !80
// CHECK:STDOUT:   ret void, !dbg !81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.ed633dc4d3ef0663(ptr %t) #0 !dbg !83 {
// CHECK:STDOUT:   call void @_ZN10NontrivialD1Ev(ptr %t), !dbg !87
// CHECK:STDOUT:   ret void, !dbg !88
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.e75943c87eaa5f64(ptr %t) #0 !dbg !90 {
// CHECK:STDOUT:   ret void, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CDestroyIndirectly.Main.d0a14ccc1d0e7f82() #0 !dbg !97 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !98
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !98
// CHECK:STDOUT:   call void @"_CMake.Trivial.Cpp:Make.Main"(ptr %a.var), !dbg !99
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.e75943c87eaa5f64(ptr %a.var), !dbg !98
// CHECK:STDOUT:   ret void, !dbg !101
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CDestroyIndirectly.Main.b437c55fb4b99438() #0 !dbg !102 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !103
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !103
// CHECK:STDOUT:   call void @"_CMake.Nontrivial.Cpp:Make.Main"(ptr %a.var), !dbg !104
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ed633dc4d3ef0663(ptr %a.var), !dbg !103
// CHECK:STDOUT:   ret void, !dbg !106
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CDestroyIndirectly.Main.587d6c5d61349179() #0 !dbg !107 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !108
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !108
// CHECK:STDOUT:   call void @"_CMake.Constexpr.Cpp:Make.Main"(ptr %a.var), !dbg !109
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.8c85e43f6e9dad76(ptr %a.var), !dbg !108
// CHECK:STDOUT:   ret void, !dbg !111
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!35}
// CHECK:STDOUT: !llvm.module.flags = !{!121, !122, !123, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!126}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !26 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !34 = !DIFile(filename: "call_destructor.carbon", directory: "")
// CHECK:STDOUT: !35 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !34, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !36 = !{null}
// CHECK:STDOUT: !37 = !DISubroutineType(types: !36)
// CHECK:STDOUT: !38 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !39 = !{null, !38}
// CHECK:STDOUT: !40 = !DISubroutineType(types: !39)
// CHECK:STDOUT: !41 = !{!38}
// CHECK:STDOUT: !42 = !DISubroutineType(types: !41)
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "Destroy", linkageName: "_CDestroy.Main", scope: null, file: !34, line: 15, type: !37, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !44 = !DILocation(line: 16, column: 3, scope: !43)
// CHECK:STDOUT: !46 = !DILocation(line: 17, column: 3, scope: !43)
// CHECK:STDOUT: !48 = !DILocation(line: 18, column: 3, scope: !43)
// CHECK:STDOUT: !50 = !DILocation(line: 15, column: 1, scope: !43)
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:SubobjectDestroy.8a0a027716a70929.Core:Constexpr.Cpp", scope: null, file: !34, line: 11, type: !40, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !55)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 1, scope: !51, type: !38)
// CHECK:STDOUT: !54 = !DILocation(line: 11, column: 13, scope: !51)
// CHECK:STDOUT: !55 = !{!52}
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Trivial.Cpp:Make.Main", scope: null, file: !34, line: 25, type: !42, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !59 = !DILocation(line: 25, column: 30, scope: !56)
// CHECK:STDOUT: !60 = !DILocation(line: 25, column: 23, scope: !56)
// CHECK:STDOUT: !61 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Nontrivial.Cpp:Make.Main", scope: null, file: !34, line: 28, type: !42, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !64 = !DILocation(line: 28, column: 30, scope: !61)
// CHECK:STDOUT: !65 = !DILocation(line: 28, column: 23, scope: !61)
// CHECK:STDOUT: !66 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Constexpr.Cpp:Make.Main", scope: null, file: !34, line: 31, type: !42, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !69 = !DILocation(line: 31, column: 30, scope: !66)
// CHECK:STDOUT: !70 = !DILocation(line: 31, column: 23, scope: !66)
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "DestroyAIndirectly", linkageName: "_CDestroyAIndirectly.Main", scope: null, file: !34, line: 38, type: !37, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !72 = !DILocation(line: 39, column: 3, scope: !71)
// CHECK:STDOUT: !73 = !DILocation(line: 40, column: 3, scope: !71)
// CHECK:STDOUT: !74 = !DILocation(line: 41, column: 3, scope: !71)
// CHECK:STDOUT: !75 = !DILocation(line: 38, column: 1, scope: !71)
// CHECK:STDOUT: !76 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.8c85e43f6e9dad76", scope: null, file: !26, line: 32, type: !40, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !82)
// CHECK:STDOUT: !77 = !DILocalVariable(arg: 1, scope: !76, type: !38)
// CHECK:STDOUT: !80 = !DILocation(line: 34, column: 5, scope: !76)
// CHECK:STDOUT: !81 = !DILocation(line: 32, column: 1, scope: !76)
// CHECK:STDOUT: !82 = !{!77}
// CHECK:STDOUT: !83 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.ed633dc4d3ef0663", scope: null, file: !26, line: 32, type: !40, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !89)
// CHECK:STDOUT: !84 = !DILocalVariable(arg: 1, scope: !83, type: !38)
// CHECK:STDOUT: !87 = !DILocation(line: 34, column: 5, scope: !83)
// CHECK:STDOUT: !88 = !DILocation(line: 32, column: 1, scope: !83)
// CHECK:STDOUT: !89 = !{!84}
// CHECK:STDOUT: !90 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.e75943c87eaa5f64", scope: null, file: !26, line: 32, type: !40, spFlags: DISPFlagDefinition, unit: !35, retainedNodes: !96)
// CHECK:STDOUT: !91 = !DILocalVariable(arg: 1, scope: !90, type: !38)
// CHECK:STDOUT: !95 = !DILocation(line: 32, column: 1, scope: !90)
// CHECK:STDOUT: !96 = !{!91}
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "DestroyIndirectly", linkageName: "_CDestroyIndirectly.Main.d0a14ccc1d0e7f82", scope: null, file: !34, line: 34, type: !37, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !98 = !DILocation(line: 35, column: 3, scope: !97)
// CHECK:STDOUT: !99 = !DILocation(line: 35, column: 21, scope: !97)
// CHECK:STDOUT: !101 = !DILocation(line: 34, column: 1, scope: !97)
// CHECK:STDOUT: !102 = distinct !DISubprogram(name: "DestroyIndirectly", linkageName: "_CDestroyIndirectly.Main.b437c55fb4b99438", scope: null, file: !34, line: 34, type: !37, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !103 = !DILocation(line: 35, column: 3, scope: !102)
// CHECK:STDOUT: !104 = !DILocation(line: 35, column: 21, scope: !102)
// CHECK:STDOUT: !106 = !DILocation(line: 34, column: 1, scope: !102)
// CHECK:STDOUT: !107 = distinct !DISubprogram(name: "DestroyIndirectly", linkageName: "_CDestroyIndirectly.Main.587d6c5d61349179", scope: null, file: !34, line: 34, type: !37, spFlags: DISPFlagDefinition, unit: !35)
// CHECK:STDOUT: !108 = !DILocation(line: 35, column: 3, scope: !107)
// CHECK:STDOUT: !109 = !DILocation(line: 35, column: 21, scope: !107)
// CHECK:STDOUT: !111 = !DILocation(line: 34, column: 1, scope: !107)
// CHECK:STDOUT: !112 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !113 = !{!"omnipotent char", !112, i64 0}
// CHECK:STDOUT: !114 = !{!"any pointer", !113, i64 0}
// CHECK:STDOUT: !115 = !{!"p1 _ZTS9Constexpr", !114, i64 0}
// CHECK:STDOUT: !116 = !{!115, !115, i64 0}
// CHECK:STDOUT: !117 = !{!"p1 _ZTS7Trivial", !114, i64 0}
// CHECK:STDOUT: !118 = !{!117, !117, i64 0}
// CHECK:STDOUT: !119 = !{!"p1 _ZTS10Nontrivial", !114, i64 0}
// CHECK:STDOUT: !120 = !{!119, !119, i64 0}
// CHECK:STDOUT: !121 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !122 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !123 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !124 = !{!"int", !113, i64 0}
// CHECK:STDOUT: !125 = !{!"__libc_errno", !124, i64 0}
// CHECK:STDOUT: !126 = !{!125, !124, i64 0}
// CHECK:STDOUT:
