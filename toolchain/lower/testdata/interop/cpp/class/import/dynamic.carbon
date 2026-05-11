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
  virtual auto further_base_func() -> int {
    return 11;
  }
};

struct Base: FurtherBase {
  int i;
  virtual auto func() -> int {
    return 7;
  }
};

auto Use(Base& b) -> int {
  return b.func();
}

''';

base class Derived {
  extend base: Cpp.Base;
  // TODO: `other_func` should appear in the vtable (it currently
  // doesn't because the CXXRecordDecl we export has no CXXMethodDecl
  // children - those need to be added, and in the right order)
  virtual fn other_func[unused self: Self]() -> i32 {
    return 3;
  }
}

inline Cpp '''

// TODO: Carbon lowering should request clang to generate the vtable.
void WorkaroundToEmitCtorAndVtable() {
  Carbon::Derived d;
}

// TODO: Carbon lowering should emit initialization that refers to the
// Clang-lowered vtable.
auto DoThing() -> int {
  Carbon::Derived d;
  return Use(d);
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
// CHECK:STDOUT: $__clang_call_terminate = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN11FurtherBase17further_base_funcEv = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN4Base4funcEv = comdat any
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
// CHECK:STDOUT: $_ZTI4Base = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTS4Base = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTI11FurtherBase = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTS11FurtherBase = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTV4Base = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTV11FurtherBase = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: @"_CDerived.Main.$vtable" = unnamed_addr constant [3 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @_ZN11FurtherBase17further_base_funcEv to i64), i64 ptrtoint (ptr @"_CDerived.Main.$vtable" to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr @_ZN4Base4funcEv to i64), i64 ptrtoint (ptr @"_CDerived.Main.$vtable" to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr @_Cother_func.Derived.Main to i64), i64 ptrtoint (ptr @"_CDerived.Main.$vtable" to i64)) to i32)]
// CHECK:STDOUT: @_ZTVN6Carbon7DerivedE = linkonce_odr dso_local unnamed_addr constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr @_ZTIN6Carbon7DerivedE, ptr @_ZN11FurtherBase17further_base_funcEv, ptr @_ZN4Base4funcEv] }, comdat, align 8
// CHECK:STDOUT: @_ZTIN6Carbon7DerivedE = linkonce_odr dso_local constant { ptr, ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2), ptr @_ZTSN6Carbon7DerivedE, ptr @_ZTI4Base }, comdat, align 8
// CHECK:STDOUT: @_ZTVN10__cxxabiv120__si_class_type_infoE = external global [0 x ptr]
// CHECK:STDOUT: @_ZTSN6Carbon7DerivedE = linkonce_odr dso_local constant [18 x i8] c"N6Carbon7DerivedE\00", comdat, align 1
// CHECK:STDOUT: @_ZTI4Base = linkonce_odr dso_local constant { ptr, ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2), ptr @_ZTS4Base, ptr @_ZTI11FurtherBase }, comdat, align 8
// CHECK:STDOUT: @_ZTS4Base = linkonce_odr dso_local constant [6 x i8] c"4Base\00", comdat, align 1
// CHECK:STDOUT: @_ZTI11FurtherBase = linkonce_odr dso_local constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS11FurtherBase }, comdat, align 8
// CHECK:STDOUT: @_ZTVN10__cxxabiv117__class_type_infoE = external global [0 x ptr]
// CHECK:STDOUT: @_ZTS11FurtherBase = linkonce_odr dso_local constant [14 x i8] c"11FurtherBase\00", comdat, align 1
// CHECK:STDOUT: @_ZTV4Base = linkonce_odr dso_local unnamed_addr constant { [4 x ptr] } { [4 x ptr] [ptr null, ptr @_ZTI4Base, ptr @_ZN11FurtherBase17further_base_funcEv, ptr @_ZN4Base4funcEv] }, comdat, align 8
// CHECK:STDOUT: @_ZTV11FurtherBase = linkonce_odr dso_local unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr @_ZTI11FurtherBase, ptr @_ZN11FurtherBase17further_base_funcEv] }, comdat, align 8
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress uwtable
// CHECK:STDOUT: define dso_local noundef i32 @_Z3UseR4Base(ptr noundef nonnull align 8 dereferenceable(12) %b) #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %b.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %b, ptr %b.addr, align 8, !tbaa !11
// CHECK:STDOUT:   %0 = load ptr, ptr %b.addr, align 8, !tbaa !11, !nonnull !14, !align !15
// CHECK:STDOUT:   %vtable = load ptr, ptr %0, align 8, !tbaa !16
// CHECK:STDOUT:   %vfn = getelementptr inbounds ptr, ptr %vtable, i64 1
// CHECK:STDOUT:   %1 = load ptr, ptr %vfn, align 8
// CHECK:STDOUT:   %call = call noundef i32 %1(ptr noundef nonnull align 8 dereferenceable(12) %0)
// CHECK:STDOUT:   ret i32 %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress uwtable
// CHECK:STDOUT: define dso_local void @_Z29WorkaroundToEmitCtorAndVtablev() #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %d = alloca %"class.Carbon::Derived", align 8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %d) #5
// CHECK:STDOUT:   call void @_ZN6Carbon7DerivedC2Ev(ptr noundef nonnull align 8 dereferenceable(12) %d) #5
// CHECK:STDOUT:   call void @_ZN6Carbon7DerivedD2Ev(ptr noundef nonnull align 8 dead_on_return(12) dereferenceable(12) %d)
// CHECK:STDOUT:   call void @llvm.lifetime.end.p0(ptr %d) #5
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN6Carbon7DerivedC2Ev(ptr noundef nonnull align 8 dereferenceable(12) %this) unnamed_addr #2 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !18
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @_ZN4BaseC2Ev(ptr noundef nonnull align 8 dereferenceable(12) %this1) #5
// CHECK:STDOUT:   store ptr getelementptr inbounds inrange(-16, 16) ({ [4 x ptr] }, ptr @_ZTVN6Carbon7DerivedE, i32 0, i32 0, i32 2), ptr %this1, align 8, !tbaa !16
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN6Carbon7DerivedD2Ev(ptr noundef nonnull align 8 dead_on_return(12) dereferenceable(12) %this) unnamed_addr #2 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !18
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   store ptr getelementptr inbounds inrange(-16, 16) ({ [4 x ptr] }, ptr @_ZTVN6Carbon7DerivedE, i32 0, i32 0, i32 2), ptr %this1, align 8, !tbaa !16
// CHECK:STDOUT:   call void @"_C__destroy_thunk:thunk.Derived.Main"(ptr noundef nonnull align 8 dereferenceable(12) %this1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.end.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress uwtable
// CHECK:STDOUT: define dso_local noundef i32 @_Z7DoThingv() #0 personality ptr @__gxx_personality_v0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %d = alloca %"class.Carbon::Derived", align 8
// CHECK:STDOUT:   %exn.slot = alloca ptr, align 8
// CHECK:STDOUT:   %ehselector.slot = alloca i32, align 4
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %d) #5
// CHECK:STDOUT:   call void @_ZN6Carbon7DerivedC2Ev(ptr noundef nonnull align 8 dereferenceable(12) %d) #5
// CHECK:STDOUT:   %call = invoke noundef i32 @_Z3UseR4Base(ptr noundef nonnull align 8 dereferenceable(12) %d)
// CHECK:STDOUT:           to label %invoke.cont unwind label %lpad
// CHECK:STDOUT:
// CHECK:STDOUT: invoke.cont:                                      ; preds = %entry
// CHECK:STDOUT:   call void @_ZN6Carbon7DerivedD2Ev(ptr noundef nonnull align 8 dead_on_return(12) dereferenceable(12) %d)
// CHECK:STDOUT:   call void @llvm.lifetime.end.p0(ptr %d) #5
// CHECK:STDOUT:   ret i32 %call
// CHECK:STDOUT:
// CHECK:STDOUT: lpad:                                             ; preds = %entry
// CHECK:STDOUT:   %0 = landingpad { ptr, i32 }
// CHECK:STDOUT:           cleanup
// CHECK:STDOUT:   %1 = extractvalue { ptr, i32 } %0, 0
// CHECK:STDOUT:   store ptr %1, ptr %exn.slot, align 8
// CHECK:STDOUT:   %2 = extractvalue { ptr, i32 } %0, 1
// CHECK:STDOUT:   store i32 %2, ptr %ehselector.slot, align 4
// CHECK:STDOUT:   invoke void @_ZN6Carbon7DerivedD2Ev(ptr noundef nonnull align 8 dead_on_return(12) dereferenceable(12) %d)
// CHECK:STDOUT:           to label %invoke.cont1 unwind label %terminate.lpad
// CHECK:STDOUT:
// CHECK:STDOUT: invoke.cont1:                                     ; preds = %lpad
// CHECK:STDOUT:   call void @llvm.lifetime.end.p0(ptr %d) #5
// CHECK:STDOUT:   br label %eh.resume
// CHECK:STDOUT:
// CHECK:STDOUT: eh.resume:                                        ; preds = %invoke.cont1
// CHECK:STDOUT:   %exn = load ptr, ptr %exn.slot, align 8
// CHECK:STDOUT:   %sel = load i32, ptr %ehselector.slot, align 4
// CHECK:STDOUT:   %lpad.val = insertvalue { ptr, i32 } poison, ptr %exn, 0
// CHECK:STDOUT:   %lpad.val2 = insertvalue { ptr, i32 } %lpad.val, i32 %sel, 1
// CHECK:STDOUT:   resume { ptr, i32 } %lpad.val2
// CHECK:STDOUT:
// CHECK:STDOUT: terminate.lpad:                                   ; preds = %lpad
// CHECK:STDOUT:   %3 = landingpad { ptr, i32 }
// CHECK:STDOUT:           catch ptr null
// CHECK:STDOUT:   %4 = extractvalue { ptr, i32 } %3, 0
// CHECK:STDOUT:   call void @__clang_call_terminate(ptr %4) #7
// CHECK:STDOUT:   unreachable
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT:   uselistorder ptr %d, { 0, 2, 1, 3, 4, 5, 6 }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @__gxx_personality_v0(...)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: noinline noreturn nounwind uwtable
// CHECK:STDOUT: define linkonce_odr hidden void @__clang_call_terminate(ptr noundef %0) #3 comdat {
// CHECK:STDOUT:   %2 = call ptr @__cxa_begin_catch(ptr %0) #5
// CHECK:STDOUT:   call void @_ZSt9terminatev() #7
// CHECK:STDOUT:   unreachable
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare ptr @__cxa_begin_catch(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZSt9terminatev()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local noundef i32 @_ZN11FurtherBase17further_base_funcEv(ptr noundef nonnull align 8 dereferenceable(8) %this) unnamed_addr #4 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !20
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret i32 11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local noundef i32 @_ZN4Base4funcEv(ptr noundef nonnull align 8 dereferenceable(12) %this) unnamed_addr #4 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !11
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret i32 7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_Cother_func.Derived.Main(ptr %self) #5 !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i32 3, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define void @"_C__destroy_thunk:thunk.Derived.Main"(ptr %self) #6 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.12e0d0434542305a:core.Destroy.Core"(ptr %self), !dbg !35
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.bfe31446004e2b71:core.Destroy.Core"(ptr %self) #5 !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.12e0d0434542305a:core.Destroy.Core"(ptr %self) #5 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN4BaseC2Ev(ptr noundef nonnull align 8 dereferenceable(12) %this) unnamed_addr #2 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !11
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @_ZN11FurtherBaseC2Ev(ptr noundef nonnull align 8 dereferenceable(8) %this1) #5
// CHECK:STDOUT:   store ptr getelementptr inbounds inrange(-16, 16) ({ [4 x ptr] }, ptr @_ZTV4Base, i32 0, i32 0, i32 2), ptr %this1, align 8, !tbaa !16
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN11FurtherBaseC2Ev(ptr noundef nonnull align 8 dereferenceable(8) %this) unnamed_addr #2 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !20
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   store ptr getelementptr inbounds inrange(-16, 8) ({ [3 x ptr] }, ptr @_ZTV11FurtherBase, i32 0, i32 0, i32 2), ptr %this1, align 8, !tbaa !16
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2), { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_ZN6Carbon7DerivedC2Ev, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_ZN6Carbon7DerivedD2Ev, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_ZN11FurtherBase17further_base_funcEv, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_ZN4Base4funcEv, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { noinline noreturn nounwind uwtable "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #4 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #5 = { nounwind }
// CHECK:STDOUT: attributes #6 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #7 = { noreturn nounwind }
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
// CHECK:STDOUT: !14 = !{}
// CHECK:STDOUT: !15 = !{i64 8}
// CHECK:STDOUT: !16 = !{!17, !17, i64 0}
// CHECK:STDOUT: !17 = !{!"vtable pointer", !10, i64 0}
// CHECK:STDOUT: !18 = !{!19, !19, i64 0}
// CHECK:STDOUT: !19 = !{!"p1 _ZTSN6Carbon7DerivedE", !13, i64 0}
// CHECK:STDOUT: !20 = !{!21, !21, i64 0}
// CHECK:STDOUT: !21 = !{!"p1 _ZTS11FurtherBase", !13, i64 0}
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "other_func", linkageName: "_Cother_func.Derived.Main", scope: null, file: !6, line: 32, type: !23, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !27)
// CHECK:STDOUT: !23 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !24 = !{!25, !26}
// CHECK:STDOUT: !25 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !26 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !27 = !{!28}
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 1, scope: !22, type: !26)
// CHECK:STDOUT: !29 = !DILocation(line: 33, column: 5, scope: !22)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.Derived.Main", scope: null, file: !6, line: 27, type: !31, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !33)
// CHECK:STDOUT: !31 = !DISubroutineType(types: !32)
// CHECK:STDOUT: !32 = !{null, !26}
// CHECK:STDOUT: !33 = !{!34}
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 1, scope: !30, type: !26)
// CHECK:STDOUT: !35 = !DILocation(line: 27, column: 1, scope: !30)
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "Op", linkageName: "_COp.bfe31446004e2b71:core.Destroy.Core", scope: null, file: !6, line: 27, type: !31, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !37)
// CHECK:STDOUT: !37 = !{!38}
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !36, type: !26)
// CHECK:STDOUT: !39 = !DILocation(line: 27, column: 1, scope: !36)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "Op", linkageName: "_COp.12e0d0434542305a:core.Destroy.Core", scope: null, file: !6, line: 27, type: !31, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !41)
// CHECK:STDOUT: !41 = !{!42}
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 1, scope: !40, type: !26)
// CHECK:STDOUT: !43 = !DILocation(line: 27, column: 1, scope: !40)
