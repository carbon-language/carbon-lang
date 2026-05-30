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
  virtual fn other_func[unused self: Self]() {
  }
  override fn func[unused ref self: Self]() {
  }
}

inline Cpp '''

// TODO: Carbon lowering should request clang to generate the vtable.
void WorkaroundToEmitCtorAndVtable() {
  Carbon::Derived d;
}

// TODO: Carbon lowering should emit initialization that refers to the
// Clang-lowered vtable.
auto DoThing() -> void {
  Carbon::Derived d;
  Use(d);
}

''';

// CHECK:STDOUT: ; ModuleID = 'dynamic_base_from_cpp.carbon'
// CHECK:STDOUT: source_filename = "dynamic_base_from_cpp.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %"class.Carbon::Derived" = type { %struct.Base.base, [4 x i8] }
// CHECK:STDOUT: %struct.Base.base = type <{ %struct.FurtherBase, i32 }>
// CHECK:STDOUT: %struct.FurtherBase = type { ptr }
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN6Carbon7DerivedC2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN6Carbon7DerivedD2Ev = comdat any
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
// CHECK:STDOUT: @_ZTVN6Carbon7DerivedE = linkonce_odr dso_local unnamed_addr constant { [5 x ptr] } { [5 x ptr] [ptr null, ptr @_ZTIN6Carbon7DerivedE, ptr @_ZN11FurtherBase17further_base_funcEv, ptr @_ZNR6Carbon7Derived4funcEv, ptr @_ZN6Carbon7Derived10other_funcEv] }, comdat, align 8
// CHECK:STDOUT: @_ZTIN6Carbon7DerivedE = linkonce_odr dso_local constant { ptr, ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2), ptr @_ZTSN6Carbon7DerivedE, ptr @_ZTI4Base }, comdat, align 8
// CHECK:STDOUT: @_ZTVN10__cxxabiv120__si_class_type_infoE = external global [0 x ptr]
// CHECK:STDOUT: @_ZTSN6Carbon7DerivedE = linkonce_odr dso_local constant [18 x i8] c"N6Carbon7DerivedE\00", comdat, align 1
// CHECK:STDOUT: @_ZTI4Base = external constant ptr
// CHECK:STDOUT: @_ZTV4Base = available_externally unnamed_addr constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr @_ZTI4Base, ptr @_ZN11FurtherBase17further_base_funcEv, ptr @_ZNR4Base4funcEv] }, align 8
// CHECK:STDOUT: @_ZTV11FurtherBase = available_externally unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI11FurtherBase, ptr @_ZN11FurtherBase17further_base_funcEv] }, align 8
// CHECK:STDOUT: @_ZTI11FurtherBase = external constant ptr
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress noinline nounwind optnone uwtable
// CHECK:STDOUT: define dso_local void @_Z3UseR4Base(ptr noundef nonnull align 8 dereferenceable(12) %b) #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %b.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %b, ptr %b.addr, align 8, !tbaa !11
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define dso_local void @_Z29WorkaroundToEmitCtorAndVtablev() #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %d = alloca %"class.Carbon::Derived", align 8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %d) #5
// CHECK:STDOUT:   call void @_ZN6Carbon7DerivedC2Ev(ptr noundef nonnull align 8 dereferenceable(12) %d) #5
// CHECK:STDOUT:   call void @_ZN6Carbon7DerivedD2Ev(ptr noundef nonnull align 8 dead_on_return(12) dereferenceable(12) %d) #5
// CHECK:STDOUT:   call void @llvm.lifetime.end.p0(ptr %d) #5
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN6Carbon7DerivedC2Ev(ptr noundef nonnull align 8 dereferenceable(12) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !14
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @_ZN4BaseC2Ev(ptr noundef nonnull align 8 dereferenceable(12) %this1) #5
// CHECK:STDOUT:   store ptr getelementptr inbounds inrange(-16, 24) ({ [5 x ptr] }, ptr @_ZTVN6Carbon7DerivedE, i32 0, i32 0, i32 2), ptr %this1, align 8, !tbaa !16
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN6Carbon7DerivedD2Ev(ptr noundef nonnull align 8 dead_on_return(12) dereferenceable(12) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !14
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   store ptr getelementptr inbounds inrange(-16, 24) ({ [5 x ptr] }, ptr @_ZTVN6Carbon7DerivedE, i32 0, i32 0, i32 2), ptr %this1, align 8, !tbaa !16
// CHECK:STDOUT:   call void @"_C__destroy_thunk:thunk.Derived.Main"(ptr noundef nonnull align 8 dereferenceable(12) %this1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.end.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define dso_local void @_Z7DoThingv() #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %d = alloca %"class.Carbon::Derived", align 8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %d) #5
// CHECK:STDOUT:   call void @_ZN6Carbon7DerivedC2Ev(ptr noundef nonnull align 8 dereferenceable(12) %d) #5
// CHECK:STDOUT:   call void @_Z3UseR4Base(ptr noundef nonnull align 8 dereferenceable(12) %d)
// CHECK:STDOUT:   call void @_ZN6Carbon7DerivedD2Ev(ptr noundef nonnull align 8 dead_on_return(12) dereferenceable(12) %d) #5
// CHECK:STDOUT:   call void @llvm.lifetime.end.p0(ptr %d) #5
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN11FurtherBase17further_base_funcEv(ptr noundef nonnull align 8 dereferenceable(8)) unnamed_addr #4
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZNR4Base4funcEv(ptr noundef nonnull align 8 dereferenceable(12)) unnamed_addr #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_Cother_func.Derived.Main(ptr %self) #5 !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_Cfunc.Derived.Main(ptr %self) #5 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define void @"_C__destroy_thunk:thunk.Derived.Main"(ptr %self) #6 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.dc7802eff02bba82:core.Destroy.Core"(ptr %self), !dbg !32
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.47e6e98bb48502be:core.Destroy.Core"(ptr %self) #5 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.dc7802eff02bba82:core.Destroy.Core"(ptr %self) #5 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_Cfunc__carbon_thunk.Derived.Main(ptr %self) #5 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Derived.func.call.vtable = load ptr, ptr %self, align 8, !dbg !44
// CHECK:STDOUT:   %Derived.func.call = call ptr @llvm.load.relative.i32(ptr %Derived.func.call.vtable, i32 4), !dbg !44
// CHECK:STDOUT:   call void %Derived.func.call(ptr %self), !dbg !44
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_Cother_func__carbon_thunk.Derived.Main(ptr %self) #5 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Derived.other_func.call.vtable = load ptr, ptr %self, align 8, !dbg !48
// CHECK:STDOUT:   %Derived.other_func.call = call ptr @llvm.load.relative.i32(ptr %Derived.other_func.call.vtable, i32 8), !dbg !48
// CHECK:STDOUT:   call void %Derived.other_func.call(ptr %self), !dbg !48
// CHECK:STDOUT:   ret void, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
// CHECK:STDOUT: declare ptr @llvm.load.relative.i32(ptr, i32) #7
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN4BaseC2Ev(ptr noundef nonnull align 8 dereferenceable(12) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !11
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @_ZN11FurtherBaseC2Ev(ptr noundef nonnull align 8 dereferenceable(8) %this1) #5
// CHECK:STDOUT:   store ptr getelementptr inbounds inrange(-16, 16) ({ [4 x ptr] }, ptr @_ZTV4Base, i32 0, i32 0, i32 2), ptr %this1, align 8, !tbaa !16
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZNR6Carbon7Derived4funcEv(ptr noundef nonnull align 8 dereferenceable(12) %this) unnamed_addr #8 align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !14
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @_Cfunc__carbon_thunk.Derived.Main(ptr noundef nonnull align 8 dereferenceable(12) %this1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN6Carbon7Derived10other_funcEv(ptr noundef nonnull align 8 dereferenceable(12) %this) unnamed_addr #8 align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !14
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @_Cother_func__carbon_thunk.Derived.Main(ptr noundef nonnull align 8 dereferenceable(12) %this1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN11FurtherBaseC2Ev(ptr noundef nonnull align 8 dereferenceable(8) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !49
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   store ptr getelementptr inbounds inrange(-16, 8) ({ [3 x ptr] }, ptr @_ZTV11FurtherBase, i32 0, i32 0, i32 2), ptr %this1, align 8, !tbaa !16
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_ZN6Carbon7DerivedC2Ev, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_ZN6Carbon7DerivedD2Ev, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_ZN11FurtherBase17further_base_funcEv, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.load.relative.i32, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress noinline nounwind optnone uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #4 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #5 = { nounwind }
// CHECK:STDOUT: attributes #6 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #7 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
// CHECK:STDOUT: attributes #8 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
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
// CHECK:STDOUT: !6 = !DIFile(filename: "dynamic_base_from_cpp.carbon", directory: "")
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = !{!12, !12, i64 0}
// CHECK:STDOUT: !12 = !{!"p1 _ZTS4Base", !13, i64 0}
// CHECK:STDOUT: !13 = !{!"any pointer", !9, i64 0}
// CHECK:STDOUT: !14 = !{!15, !15, i64 0}
// CHECK:STDOUT: !15 = !{!"p1 _ZTSN6Carbon7DerivedE", !13, i64 0}
// CHECK:STDOUT: !16 = !{!17, !17, i64 0}
// CHECK:STDOUT: !17 = !{!"vtable pointer", !10, i64 0}
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "other_func", linkageName: "_Cother_func.Derived.Main", scope: null, file: !6, line: 24, type: !19, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !22)
// CHECK:STDOUT: !19 = !DISubroutineType(types: !20)
// CHECK:STDOUT: !20 = !{null, !21}
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!23}
// CHECK:STDOUT: !23 = !DILocalVariable(arg: 1, scope: !18, type: !21)
// CHECK:STDOUT: !24 = !DILocation(line: 24, column: 3, scope: !18)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "func", linkageName: "_Cfunc.Derived.Main", scope: null, file: !6, line: 26, type: !19, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !26)
// CHECK:STDOUT: !26 = !{!27}
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !25, type: !21)
// CHECK:STDOUT: !28 = !DILocation(line: 26, column: 3, scope: !25)
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.Derived.Main", scope: null, file: !6, line: 22, type: !19, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !30)
// CHECK:STDOUT: !30 = !{!31}
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 1, scope: !29, type: !21)
// CHECK:STDOUT: !32 = !DILocation(line: 22, column: 1, scope: !29)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "Op", linkageName: "_COp.47e6e98bb48502be:core.Destroy.Core", scope: null, file: !6, line: 22, type: !19, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !34)
// CHECK:STDOUT: !34 = !{!35}
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !33, type: !21)
// CHECK:STDOUT: !36 = !DILocation(line: 22, column: 1, scope: !33)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "Op", linkageName: "_COp.dc7802eff02bba82:core.Destroy.Core", scope: null, file: !6, line: 22, type: !19, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !38)
// CHECK:STDOUT: !38 = !{!39}
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !37, type: !21)
// CHECK:STDOUT: !40 = !DILocation(line: 22, column: 1, scope: !37)
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "func__carbon_thunk", linkageName: "_Cfunc__carbon_thunk.Derived.Main", scope: null, file: !6, line: 26, type: !19, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !42)
// CHECK:STDOUT: !42 = !{!43}
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !41, type: !21)
// CHECK:STDOUT: !44 = !DILocation(line: 26, column: 3, scope: !41)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "other_func__carbon_thunk", linkageName: "_Cother_func__carbon_thunk.Derived.Main", scope: null, file: !6, line: 24, type: !19, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !46)
// CHECK:STDOUT: !46 = !{!47}
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !45, type: !21)
// CHECK:STDOUT: !48 = !DILocation(line: 24, column: 3, scope: !45)
// CHECK:STDOUT: !49 = !{!50, !50, i64 0}
// CHECK:STDOUT: !50 = !{!"p1 _ZTS11FurtherBase", !13, i64 0}
