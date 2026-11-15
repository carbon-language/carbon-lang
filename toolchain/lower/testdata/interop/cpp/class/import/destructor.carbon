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
// CHECK:STDOUT: define void @_CDestroy.Main() #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var.loc16 = alloca {}, align 1, !dbg !45
// CHECK:STDOUT:   %_.var.loc17 = alloca {}, align 1, !dbg !47
// CHECK:STDOUT:   %_.var.loc18 = alloca {}, align 1, !dbg !49
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc16), !dbg !45
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %_.var.loc16, ptr align 1 @Trivial.val.loc16_3, i64 0, i1 false), !dbg !45
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc17), !dbg !47
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %_.var.loc17, ptr align 1 @Nontrivial.val.loc17_3, i64 0, i1 false), !dbg !47
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc18), !dbg !49
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %_.var.loc18, ptr align 1 @Constexpr.val.loc18_3, i64 0, i1 false), !dbg !49
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN9ConstexprD2Ev(ptr noundef nonnull align 1 dereferenceable(1) %this) unnamed_addr #1 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !101
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:SubobjectDestroy.8a0a027716a70929.Core:Constexpr.Cpp"(ptr %self) #2 !dbg !52 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN9ConstexprD2Ev(ptr %self), !dbg !55
// CHECK:STDOUT:   ret void, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: declare void @_ZN10NontrivialD1Ev(ptr noundef nonnull align 1 dereferenceable(1)) unnamed_addr #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:SubobjectDestroy.ac48bb57359016da.Core:Trivial.Cpp"(ptr %self) #2 !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CMake.Trivial.Cpp:Make.Main"(ptr sret({}) %return) #0 !dbg !62 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN7TrivialC1Ev.carbon_thunk.(ptr %return), !dbg !65
// CHECK:STDOUT:   ret void, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN7TrivialC1Ev.carbon_thunk.(ptr noundef %return) #4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !103
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !103
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CMake.Nontrivial.Cpp:Make.Main"(ptr sret({}) %return) #0 !dbg !67 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN10NontrivialC1Ev.carbon_thunk.(ptr %return), !dbg !70
// CHECK:STDOUT:   ret void, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN10NontrivialC1Ev.carbon_thunk.(ptr noundef %return) #4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !105
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !105
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CMake.Constexpr.Cpp:Make.Main"(ptr sret({}) %return) #0 !dbg !72 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN9ConstexprC1Ev.carbon_thunk.(ptr %return), !dbg !75
// CHECK:STDOUT:   ret void, !dbg !76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN9ConstexprC1Ev.carbon_thunk.(ptr noundef %return) #4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !101
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !101
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CDestroyAIndirectly.Main() #0 !dbg !77 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CDestroyIndirectly.Main.1ebfd06ba26076ab(), !dbg !78
// CHECK:STDOUT:   call void @_CDestroyIndirectly.Main.2abe72164b371546(), !dbg !79
// CHECK:STDOUT:   call void @_CDestroyIndirectly.Main.19da2d9454902708(), !dbg !80
// CHECK:STDOUT:   ret void, !dbg !81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #5
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #5
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CDestroyIndirectly.Main.1ebfd06ba26076ab() #0 !dbg !82 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !83
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !83
// CHECK:STDOUT:   call void @"_CMake.Trivial.Cpp:Make.Main"(ptr %a.var), !dbg !84
// CHECK:STDOUT:   ret void, !dbg !86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CDestroyIndirectly.Main.2abe72164b371546() #0 !dbg !87 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !88
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !88
// CHECK:STDOUT:   call void @"_CMake.Nontrivial.Cpp:Make.Main"(ptr %a.var), !dbg !89
// CHECK:STDOUT:   ret void, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CDestroyIndirectly.Main.19da2d9454902708() #0 !dbg !92 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca {}, align 1, !dbg !93
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !93
// CHECK:STDOUT:   call void @"_CMake.Constexpr.Cpp:Make.Main"(ptr %a.var), !dbg !94
// CHECK:STDOUT:   ret void, !dbg !96
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!36}
// CHECK:STDOUT: !llvm.module.flags = !{!106, !107, !108, !109, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!112}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !35 = !DIFile(filename: "call_destructor.carbon", directory: "")
// CHECK:STDOUT: !36 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !35, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !37 = !{null}
// CHECK:STDOUT: !38 = !DISubroutineType(types: !37)
// CHECK:STDOUT: !39 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !40 = !{null, !39}
// CHECK:STDOUT: !41 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !42 = !{!39}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "Destroy", linkageName: "_CDestroy.Main", scope: null, file: !35, line: 15, type: !38, spFlags: DISPFlagDefinition, unit: !36)
// CHECK:STDOUT: !45 = !DILocation(line: 16, column: 3, scope: !44)
// CHECK:STDOUT: !47 = !DILocation(line: 17, column: 3, scope: !44)
// CHECK:STDOUT: !49 = !DILocation(line: 18, column: 3, scope: !44)
// CHECK:STDOUT: !51 = !DILocation(line: 15, column: 1, scope: !44)
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:SubobjectDestroy.8a0a027716a70929.Core:Constexpr.Cpp", scope: null, file: !35, line: 11, type: !41, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !56)
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 1, scope: !52, type: !39)
// CHECK:STDOUT: !55 = !DILocation(line: 11, column: 13, scope: !52)
// CHECK:STDOUT: !56 = !{!53}
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:SubobjectDestroy.ac48bb57359016da.Core:Trivial.Cpp", scope: null, file: !35, line: 6, type: !41, spFlags: DISPFlagDefinition, unit: !36, retainedNodes: !61)
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 1, scope: !57, type: !39)
// CHECK:STDOUT: !60 = !DILocation(line: 6, column: 8, scope: !57)
// CHECK:STDOUT: !61 = !{!58}
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Trivial.Cpp:Make.Main", scope: null, file: !35, line: 25, type: !43, spFlags: DISPFlagDefinition, unit: !36)
// CHECK:STDOUT: !65 = !DILocation(line: 25, column: 30, scope: !62)
// CHECK:STDOUT: !66 = !DILocation(line: 25, column: 23, scope: !62)
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Nontrivial.Cpp:Make.Main", scope: null, file: !35, line: 28, type: !43, spFlags: DISPFlagDefinition, unit: !36)
// CHECK:STDOUT: !70 = !DILocation(line: 28, column: 30, scope: !67)
// CHECK:STDOUT: !71 = !DILocation(line: 28, column: 23, scope: !67)
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Constexpr.Cpp:Make.Main", scope: null, file: !35, line: 31, type: !43, spFlags: DISPFlagDefinition, unit: !36)
// CHECK:STDOUT: !75 = !DILocation(line: 31, column: 30, scope: !72)
// CHECK:STDOUT: !76 = !DILocation(line: 31, column: 23, scope: !72)
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "DestroyAIndirectly", linkageName: "_CDestroyAIndirectly.Main", scope: null, file: !35, line: 38, type: !38, spFlags: DISPFlagDefinition, unit: !36)
// CHECK:STDOUT: !78 = !DILocation(line: 39, column: 3, scope: !77)
// CHECK:STDOUT: !79 = !DILocation(line: 40, column: 3, scope: !77)
// CHECK:STDOUT: !80 = !DILocation(line: 41, column: 3, scope: !77)
// CHECK:STDOUT: !81 = !DILocation(line: 38, column: 1, scope: !77)
// CHECK:STDOUT: !82 = distinct !DISubprogram(name: "DestroyIndirectly", linkageName: "_CDestroyIndirectly.Main.1ebfd06ba26076ab", scope: null, file: !35, line: 34, type: !38, spFlags: DISPFlagDefinition, unit: !36)
// CHECK:STDOUT: !83 = !DILocation(line: 35, column: 3, scope: !82)
// CHECK:STDOUT: !84 = !DILocation(line: 35, column: 21, scope: !82)
// CHECK:STDOUT: !86 = !DILocation(line: 34, column: 1, scope: !82)
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "DestroyIndirectly", linkageName: "_CDestroyIndirectly.Main.2abe72164b371546", scope: null, file: !35, line: 34, type: !38, spFlags: DISPFlagDefinition, unit: !36)
// CHECK:STDOUT: !88 = !DILocation(line: 35, column: 3, scope: !87)
// CHECK:STDOUT: !89 = !DILocation(line: 35, column: 21, scope: !87)
// CHECK:STDOUT: !91 = !DILocation(line: 34, column: 1, scope: !87)
// CHECK:STDOUT: !92 = distinct !DISubprogram(name: "DestroyIndirectly", linkageName: "_CDestroyIndirectly.Main.19da2d9454902708", scope: null, file: !35, line: 34, type: !38, spFlags: DISPFlagDefinition, unit: !36)
// CHECK:STDOUT: !93 = !DILocation(line: 35, column: 3, scope: !92)
// CHECK:STDOUT: !94 = !DILocation(line: 35, column: 21, scope: !92)
// CHECK:STDOUT: !96 = !DILocation(line: 34, column: 1, scope: !92)
// CHECK:STDOUT: !97 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !98 = !{!"omnipotent char", !97, i64 0}
// CHECK:STDOUT: !99 = !{!"any pointer", !98, i64 0}
// CHECK:STDOUT: !100 = !{!"p1 _ZTS9Constexpr", !99, i64 0}
// CHECK:STDOUT: !101 = !{!100, !100, i64 0}
// CHECK:STDOUT: !102 = !{!"p1 _ZTS7Trivial", !99, i64 0}
// CHECK:STDOUT: !103 = !{!102, !102, i64 0}
// CHECK:STDOUT: !104 = !{!"p1 _ZTS10Nontrivial", !99, i64 0}
// CHECK:STDOUT: !105 = !{!104, !104, i64 0}
// CHECK:STDOUT: !106 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !107 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !108 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !109 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !110 = !{!"int", !98, i64 0}
// CHECK:STDOUT: !111 = !{!"__libc_errno", !110, i64 0}
// CHECK:STDOUT: !112 = !{!111, !110, i64 0}
// CHECK:STDOUT:
