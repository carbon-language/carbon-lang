// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/class/import/dynamic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/class/import/dynamic.carbon

// --- dynamic_base_from_cpp.carbon

library "[[@TEST_NAME]]";

import Cpp;

inline Cpp '''

struct FurtherBase {
  virtual auto further_base_func() -> void;
};

struct Base: FurtherBase {
  int i;
  virtual auto func() & -> void;
};

__attribute__((optnone)) auto Use(Base& b) {
}

''';

base class Derived {
  extend base: Cpp.Base;
  virtual fn other_func(unused self) {
  }
  override fn func(unused ref self) {
  }
}

fn DoThing() {
  var d: Derived = {.base = Cpp.Base.Base()};
  Cpp.Use(ref d);
}

// CHECK:STDOUT: ; ModuleID = 'dynamic_base_from_cpp.carbon'
// CHECK:STDOUT: source_filename = "dynamic_base_from_cpp.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN4BaseC2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN11FurtherBaseC2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTVN6Carbon7DerivedE = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTIN6Carbon7DerivedE = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTSN6Carbon7DerivedE = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: @_ZTV4Base = available_externally constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr @_ZTI4Base, ptr @_ZN11FurtherBase17further_base_funcEv, ptr @_ZNR4Base4funcEv] }, align 8
// CHECK:STDOUT: @_ZTV11FurtherBase = available_externally constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI11FurtherBase, ptr @_ZN11FurtherBase17further_base_funcEv] }, align 8
// CHECK:STDOUT: @_ZTVN6Carbon7DerivedE = linkonce_odr dso_local constant { [5 x ptr] } { [5 x ptr] [ptr null, ptr @_ZTIN6Carbon7DerivedE, ptr @_ZN11FurtherBase17further_base_funcEv, ptr @_ZNR6Carbon7Derived4funcEv, ptr @_ZN6Carbon7Derived10other_funcEv] }, comdat, align 8
// CHECK:STDOUT: @_ZTI4Base = external constant ptr
// CHECK:STDOUT: @_ZTI11FurtherBase = external constant ptr
// CHECK:STDOUT: @_ZTIN6Carbon7DerivedE = linkonce_odr dso_local constant { ptr, ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2), ptr @_ZTSN6Carbon7DerivedE, ptr @_ZTI4Base }, comdat, align 8
// CHECK:STDOUT: @_ZTVN10__cxxabiv120__si_class_type_infoE = external global [0 x ptr]
// CHECK:STDOUT: @_ZTSN6Carbon7DerivedE = linkonce_odr dso_local constant [18 x i8] c"N6Carbon7DerivedE\00", comdat, align 1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress noinline nounwind optnone uwtable
// CHECK:STDOUT: define dso_local void @_Z3UseR4Base(ptr noundef nonnull align 8 dereferenceable(12) %b) #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %b.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %b, ptr %b.addr, align 8, !tbaa !11
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN11FurtherBase17further_base_funcEv(ptr noundef nonnull align 8 dereferenceable(8)) unnamed_addr #1
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZNR4Base4funcEv(ptr noundef nonnull align 8 dereferenceable(12)) unnamed_addr #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_Cother_func.Derived.Main(ptr %self) #2 !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_Cfunc.Derived.Main(ptr %self) #2 !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define void @"_Cfunc:thunk:Base.Cpp:Derived.Main"(ptr %self) #3 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Cfunc.Derived.Main(ptr %self), !dbg !28
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CDoThing.Main() #2 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %d.var = alloca { [16 x i8] }, align 8, !dbg !32
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %d.var), !dbg !32
// CHECK:STDOUT:   %.loc31_44.1.base = getelementptr inbounds nuw { [16 x i8] }, ptr %d.var, i32 0, i32 0, !dbg !33
// CHECK:STDOUT:   call void @_ZN4BaseC1Ev.carbon_thunk.(ptr %.loc31_44.1.base), !dbg !34
// CHECK:STDOUT:   %.loc31_44.7.base = getelementptr inbounds nuw { [16 x i8] }, ptr %d.var, i32 0, i32 0, !dbg !33
// CHECK:STDOUT:   %.loc31_44.8.base = getelementptr inbounds nuw [16 x i8], ptr %.loc31_44.7.base, i32 0, i32 0, !dbg !33
// CHECK:STDOUT:   %.loc31_44.9.vptr = getelementptr inbounds nuw [8 x i8], ptr %.loc31_44.8.base, i32 0, i32 0, !dbg !33
// CHECK:STDOUT:   store ptr getelementptr inbounds inrange(-16, 24) ({ [5 x ptr] }, ptr @_ZTVN6Carbon7DerivedE, i32 0, i32 0, i32 2), ptr %.loc31_44.9.vptr, align 8, !dbg !33
// CHECK:STDOUT:   %.loc32_11.3.base = getelementptr inbounds nuw [16 x i8], ptr %d.var, i32 0, i32 0, !dbg !35
// CHECK:STDOUT:   call void @_Z3UseR4Base(ptr %.loc32_11.3.base), !dbg !36
// CHECK:STDOUT:   call void @"_COp.8185e3504351afa8:core.Destroy.Core"(ptr %d.var), !dbg !32
// CHECK:STDOUT:   ret void, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN4BaseC1Ev.carbon_thunk.(ptr noundef %return) #4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !11
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !11
// CHECK:STDOUT:   call void @_ZN4BaseC2Ev(ptr noundef nonnull align 8 dereferenceable(12) %0) #2
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define void @"_C__destroy_thunk:thunk.Derived.Main"(ptr %self) #3 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.8185e3504351afa8:core.Destroy.Core"(ptr %self), !dbg !41
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.13b019890e86b6be:core.Destroy.Core"(ptr %self) #2 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.8185e3504351afa8:core.Destroy.Core"(ptr %self) #2 !dbg !46 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_Cfunc__carbon_thunk.Derived.Main(ptr %self) #2 !dbg !50 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Cfunc.Derived.Main(ptr %self), !dbg !53
// CHECK:STDOUT:   ret void, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_Cother_func__carbon_thunk.Derived.Main(ptr %self) #2 !dbg !54 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Derived.other_func.call.vtable = load ptr, ptr %self, align 8, !dbg !57
// CHECK:STDOUT:   %Derived.other_func.call = call ptr @llvm.load.relative.i32(ptr %Derived.other_func.call.vtable, i32 8), !dbg !57
// CHECK:STDOUT:   call void %Derived.other_func.call(ptr %self), !dbg !57
// CHECK:STDOUT:   ret void, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #5
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
// CHECK:STDOUT: declare ptr @llvm.load.relative.i32(ptr, i32) #6
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZNR6Carbon7Derived4funcEv(ptr noundef nonnull align 8 dereferenceable(12) %this) unnamed_addr #4 align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !58
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @_Cfunc__carbon_thunk.Derived.Main(ptr noundef nonnull align 8 dereferenceable(12) %this1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN6Carbon7Derived10other_funcEv(ptr noundef nonnull align 8 dereferenceable(12) %this) unnamed_addr #4 align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !58
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @_Cother_func__carbon_thunk.Derived.Main(ptr noundef nonnull align 8 dereferenceable(12) %this1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN4BaseC2Ev(ptr noundef nonnull align 8 dereferenceable(12) %this) unnamed_addr #7 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !11
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @_ZN11FurtherBaseC2Ev(ptr noundef nonnull align 8 dereferenceable(8) %this1) #2
// CHECK:STDOUT:   store ptr getelementptr inbounds inrange(-16, 16) ({ [4 x ptr] }, ptr @_ZTV4Base, i32 0, i32 0, i32 2), ptr %this1, align 8, !tbaa !60
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN11FurtherBaseC2Ev(ptr noundef nonnull align 8 dereferenceable(8) %this) unnamed_addr #7 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !62
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   store ptr getelementptr inbounds inrange(-16, 8) ({ [3 x ptr] }, ptr @_ZTV11FurtherBase, i32 0, i32 0, i32 2), ptr %this1, align 8, !tbaa !60
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_ZN11FurtherBase17further_base_funcEv, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_COp.8185e3504351afa8:core.Destroy.Core", { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress noinline nounwind optnone uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nounwind }
// CHECK:STDOUT: attributes #3 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #4 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #5 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #6 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
// CHECK:STDOUT: attributes #7 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3, !4, !5, !6}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!7}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "dynamic_base_from_cpp.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !3 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !5 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !6 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = !{!12, !12, i64 0}
// CHECK:STDOUT: !12 = !{!"p1 _ZTS4Base", !13, i64 0}
// CHECK:STDOUT: !13 = !{!"any pointer", !9, i64 0}
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "other_func", linkageName: "_Cother_func.Derived.Main", scope: null, file: !1, line: 24, type: !15, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !18)
// CHECK:STDOUT: !15 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !16 = !{null, !17}
// CHECK:STDOUT: !17 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !18 = !{!19}
// CHECK:STDOUT: !19 = !DILocalVariable(arg: 1, scope: !14, type: !17)
// CHECK:STDOUT: !20 = !DILocation(line: 24, column: 3, scope: !14)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "func", linkageName: "_Cfunc.Derived.Main", scope: null, file: !1, line: 26, type: !15, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !22)
// CHECK:STDOUT: !22 = !{!23}
// CHECK:STDOUT: !23 = !DILocalVariable(arg: 1, scope: !21, type: !17)
// CHECK:STDOUT: !24 = !DILocation(line: 26, column: 3, scope: !21)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "func", linkageName: "_Cfunc:thunk:Base.Cpp:Derived.Main", scope: null, file: !1, line: 26, type: !15, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !26)
// CHECK:STDOUT: !26 = !{!27}
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !25, type: !17)
// CHECK:STDOUT: !28 = !DILocation(line: 26, column: 3, scope: !25)
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "DoThing", linkageName: "_CDoThing.Main", scope: null, file: !1, line: 30, type: !30, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !30 = !DISubroutineType(types: !31)
// CHECK:STDOUT: !31 = !{null}
// CHECK:STDOUT: !32 = !DILocation(line: 31, column: 3, scope: !29)
// CHECK:STDOUT: !33 = !DILocation(line: 31, column: 20, scope: !29)
// CHECK:STDOUT: !34 = !DILocation(line: 31, column: 29, scope: !29)
// CHECK:STDOUT: !35 = !DILocation(line: 32, column: 11, scope: !29)
// CHECK:STDOUT: !36 = !DILocation(line: 32, column: 3, scope: !29)
// CHECK:STDOUT: !37 = !DILocation(line: 30, column: 1, scope: !29)
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.Derived.Main", scope: null, file: !1, line: 22, type: !15, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !39)
// CHECK:STDOUT: !39 = !{!40}
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !38, type: !17)
// CHECK:STDOUT: !41 = !DILocation(line: 22, column: 1, scope: !38)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "Op", linkageName: "_COp.13b019890e86b6be:core.Destroy.Core", scope: null, file: !1, line: 22, type: !15, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !43)
// CHECK:STDOUT: !43 = !{!44}
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 1, scope: !42, type: !17)
// CHECK:STDOUT: !45 = !DILocation(line: 22, column: 1, scope: !42)
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "Op", linkageName: "_COp.8185e3504351afa8:core.Destroy.Core", scope: null, file: !1, line: 22, type: !15, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !47)
// CHECK:STDOUT: !47 = !{!48}
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 1, scope: !46, type: !17)
// CHECK:STDOUT: !49 = !DILocation(line: 22, column: 1, scope: !46)
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "func__carbon_thunk", linkageName: "_Cfunc__carbon_thunk.Derived.Main", scope: null, file: !1, line: 26, type: !15, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !51)
// CHECK:STDOUT: !51 = !{!52}
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 1, scope: !50, type: !17)
// CHECK:STDOUT: !53 = !DILocation(line: 26, column: 3, scope: !50)
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "other_func__carbon_thunk", linkageName: "_Cother_func__carbon_thunk.Derived.Main", scope: null, file: !1, line: 24, type: !15, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !55)
// CHECK:STDOUT: !55 = !{!56}
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !54, type: !17)
// CHECK:STDOUT: !57 = !DILocation(line: 24, column: 3, scope: !54)
// CHECK:STDOUT: !58 = !{!59, !59, i64 0}
// CHECK:STDOUT: !59 = !{!"p1 _ZTSN6Carbon7DerivedE", !13, i64 0}
// CHECK:STDOUT: !60 = !{!61, !61, i64 0}
// CHECK:STDOUT: !61 = !{!"vtable pointer", !10, i64 0}
// CHECK:STDOUT: !62 = !{!63, !63, i64 0}
// CHECK:STDOUT: !63 = !{!"p1 _ZTS11FurtherBase", !13, i64 0}
