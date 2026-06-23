// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// EXTRA-ARGS: --clang-arg=-fno-exceptions
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/class/virtual_fn.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/class/virtual_fn.carbon

// --- override_virtual_dtor.carbon
library "[[@TEST_NAME]]";

import Cpp;

inline Cpp '''
struct A {
  virtual ~A() = 0;
  virtual void F() const;
};
''';

class Final {
  extend base: Cpp.A;
  override fn F(self);
}

base class Base {
  extend base: Cpp.A;
  override fn F(self);
}

inline Cpp '''
void delete_new_Final() {
  // vtable should refer to Carbon::Final's dtor.
  auto* p = new Carbon::Final;
  // Perform a devirtualized call to Final destructor.
  delete p;
}

void delete_new_Base() {
  // vtable should refer to Carbon::Base's dtor.
  auto* p = new Carbon::Base;
  // Perform a virtual call to destructor.
  delete p;
}
''';

// CHECK:STDOUT: ; ModuleID = 'override_virtual_dtor.carbon'
// CHECK:STDOUT: source_filename = "override_virtual_dtor.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN6Carbon5FinalC2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN6Carbon5FinalD2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN6Carbon4BaseC2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN6Carbon5FinalD0Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN6Carbon4BaseD2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN6Carbon4BaseD0Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN1AC2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTVN6Carbon5FinalE = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTVN6Carbon4BaseE = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTIN6Carbon5FinalE = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTSN6Carbon5FinalE = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTIN6Carbon4BaseE = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTSN6Carbon4BaseE = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: @_ZTV1A = available_externally unnamed_addr constant { [5 x ptr] } { [5 x ptr] [ptr null, ptr @_ZTI1A, ptr @__cxa_pure_virtual, ptr @__cxa_pure_virtual, ptr @_ZNK1A1FEv] }, align 8
// CHECK:STDOUT: @_ZTVN6Carbon5FinalE = linkonce_odr dso_local unnamed_addr constant { [6 x ptr] } { [6 x ptr] [ptr null, ptr @_ZTIN6Carbon5FinalE, ptr @_ZN6Carbon5FinalD2Ev, ptr @_ZN6Carbon5FinalD0Ev, ptr @_ZNK1A1FEv, ptr @_ZN6Carbon5Final1FEv] }, comdat, align 8
// CHECK:STDOUT: @_ZTVN6Carbon4BaseE = linkonce_odr dso_local unnamed_addr constant { [6 x ptr] } { [6 x ptr] [ptr null, ptr @_ZTIN6Carbon4BaseE, ptr @_ZN6Carbon4BaseD2Ev, ptr @_ZN6Carbon4BaseD0Ev, ptr @_ZNK1A1FEv, ptr @_ZN6Carbon4Base1FEv] }, comdat, align 8
// CHECK:STDOUT: @_ZTI1A = external constant ptr
// CHECK:STDOUT: @_ZTIN6Carbon5FinalE = linkonce_odr dso_local constant { ptr, ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2), ptr @_ZTSN6Carbon5FinalE, ptr @_ZTI1A }, comdat, align 8
// CHECK:STDOUT: @_ZTVN10__cxxabiv120__si_class_type_infoE = external global [0 x ptr]
// CHECK:STDOUT: @_ZTSN6Carbon5FinalE = linkonce_odr dso_local constant [16 x i8] c"N6Carbon5FinalE\00", comdat, align 1
// CHECK:STDOUT: @_ZTIN6Carbon4BaseE = linkonce_odr dso_local constant { ptr, ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2), ptr @_ZTSN6Carbon4BaseE, ptr @_ZTI1A }, comdat, align 8
// CHECK:STDOUT: @_ZTSN6Carbon4BaseE = linkonce_odr dso_local constant [15 x i8] c"N6Carbon4BaseE\00", comdat, align 1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define dso_local void @_Z16delete_new_Finalv() #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %p = alloca ptr, align 8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %p) #8
// CHECK:STDOUT:   %call = call noalias noundef nonnull ptr @_Znwm(i64 noundef 8) #10
// CHECK:STDOUT:   call void @_ZN6Carbon5FinalC2Ev(ptr noundef nonnull align 8 dereferenceable(8) %call) #8
// CHECK:STDOUT:   store ptr %call, ptr %p, align 8, !tbaa !11
// CHECK:STDOUT:   %0 = load ptr, ptr %p, align 8, !tbaa !11
// CHECK:STDOUT:   %isnull = icmp eq ptr %0, null
// CHECK:STDOUT:   br i1 %isnull, label %delete.end, label %delete.notnull
// CHECK:STDOUT:
// CHECK:STDOUT: delete.notnull:                                   ; preds = %entry
// CHECK:STDOUT:   call void @_ZN6Carbon5FinalD2Ev(ptr noundef nonnull align 8 dead_on_return(8) dereferenceable(8) %0) #8
// CHECK:STDOUT:   call void @_ZdlPvm(ptr noundef %0, i64 noundef 8) #11
// CHECK:STDOUT:   br label %delete.end
// CHECK:STDOUT:
// CHECK:STDOUT: delete.end:                                       ; preds = %delete.notnull, %entry
// CHECK:STDOUT:   call void @llvm.lifetime.end.p0(ptr %p) #8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nobuiltin allocsize(0)
// CHECK:STDOUT: declare noundef nonnull ptr @_Znwm(i64 noundef) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN6Carbon5FinalC2Ev(ptr noundef nonnull align 8 dereferenceable(8) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !11
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @_ZN1AC2Ev(ptr noundef nonnull align 8 dereferenceable(8) %this1) #8
// CHECK:STDOUT:   store ptr getelementptr inbounds inrange(-16, 32) ({ [6 x ptr] }, ptr @_ZTVN6Carbon5FinalE, i32 0, i32 0, i32 2), ptr %this1, align 8, !tbaa !14
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN6Carbon5FinalD2Ev(ptr noundef nonnull align 8 dead_on_return(8) dereferenceable(8) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !11
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @"_C__destroy_thunk:thunk.Final.Main"(ptr noundef nonnull align 8 dereferenceable(8) %this1)
// CHECK:STDOUT:   call void @_ZN1AD2Ev(ptr noundef nonnull align 8 dead_on_return(8) dereferenceable(8) %this1) #8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nobuiltin nounwind
// CHECK:STDOUT: declare void @_ZdlPvm(ptr noundef, i64 noundef) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.end.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define dso_local void @_Z15delete_new_Basev() #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %p = alloca ptr, align 8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %p) #8
// CHECK:STDOUT:   %call = call noalias noundef nonnull ptr @_Znwm(i64 noundef 8) #10
// CHECK:STDOUT:   call void @_ZN6Carbon4BaseC2Ev(ptr noundef nonnull align 8 dereferenceable(8) %call) #8
// CHECK:STDOUT:   store ptr %call, ptr %p, align 8, !tbaa !16
// CHECK:STDOUT:   %0 = load ptr, ptr %p, align 8, !tbaa !16
// CHECK:STDOUT:   %isnull = icmp eq ptr %0, null
// CHECK:STDOUT:   br i1 %isnull, label %delete.end, label %delete.notnull
// CHECK:STDOUT:
// CHECK:STDOUT: delete.notnull:                                   ; preds = %entry
// CHECK:STDOUT:   %vtable = load ptr, ptr %0, align 8, !tbaa !14
// CHECK:STDOUT:   %vfn = getelementptr inbounds ptr, ptr %vtable, i64 1
// CHECK:STDOUT:   %1 = load ptr, ptr %vfn, align 8
// CHECK:STDOUT:   call void %1(ptr noundef nonnull align 8 dereferenceable(8) %0) #8
// CHECK:STDOUT:   br label %delete.end
// CHECK:STDOUT:
// CHECK:STDOUT: delete.end:                                       ; preds = %delete.notnull, %entry
// CHECK:STDOUT:   call void @llvm.lifetime.end.p0(ptr %p) #8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN6Carbon4BaseC2Ev(ptr noundef nonnull align 8 dereferenceable(8) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !16
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @_ZN1AC2Ev(ptr noundef nonnull align 8 dereferenceable(8) %this1) #8
// CHECK:STDOUT:   store ptr getelementptr inbounds inrange(-16, 32) ({ [6 x ptr] }, ptr @_ZTVN6Carbon4BaseE, i32 0, i32 0, i32 2), ptr %this1, align 8, !tbaa !14
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZNK1A1FEv.carbon_thunk._(ptr noundef nonnull align 8 dereferenceable(8) %this) #5 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !18
// CHECK:STDOUT:   %0 = load ptr, ptr %this.addr, align 8, !tbaa !18, !nonnull !20, !align !21
// CHECK:STDOUT:   %vtable = load ptr, ptr %0, align 8, !tbaa !14
// CHECK:STDOUT:   %vfn = getelementptr inbounds ptr, ptr %vtable, i64 2
// CHECK:STDOUT:   %1 = load ptr, ptr %vfn, align 8
// CHECK:STDOUT:   call void %1(ptr noundef nonnull align 8 dereferenceable(8) %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Final.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define void @"_CF:thunk:A.Cpp:Final.Main"(ptr %self) #6 !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Final.Main(ptr %self), !dbg !28
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Base.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define void @"_CF:thunk:A.Cpp:Base.Main"(ptr %self) #6 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Base.Main(ptr %self), !dbg !32
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define void @"_C__destroy_thunk:thunk.Final.Main"(ptr %self) #6 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.6622c02ec7a967c3:core.Destroy.Core"(ptr %self), !dbg !36
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: declare void @_ZN1AD1Ev(ptr noundef nonnull align 8 dead_on_return(8) dereferenceable(8)) unnamed_addr #7
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.4fc2347fc019a955:core.Destroy.Core"(ptr %self) #8 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.6622c02ec7a967c3:core.Destroy.Core"(ptr %self) #8 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF__carbon_thunk.Final.Main(ptr %self) #8 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Final.Main(ptr %self), !dbg !48
// CHECK:STDOUT:   ret void, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define void @"_C__destroy_thunk:thunk.Base.Main"(ptr %self) #6 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.a45b619846c20518:core.Destroy.Core"(ptr %self), !dbg !52
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.a45b619846c20518:core.Destroy.Core"(ptr %self) #8 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF__carbon_thunk.Base.Main(ptr %self) #8 !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Base.Main(ptr %self), !dbg !60
// CHECK:STDOUT:   ret void, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @__cxa_pure_virtual() unnamed_addr
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZNK1A1FEv(ptr noundef nonnull align 8 dereferenceable(8)) unnamed_addr #9
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN6Carbon5FinalD0Ev(ptr noundef nonnull align 8 dereferenceable(8) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !11
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @_ZN6Carbon5FinalD2Ev(ptr noundef nonnull align 8 dead_on_return(8) dereferenceable(8) %this1) #8
// CHECK:STDOUT:   call void @_ZdlPvm(ptr noundef %this1, i64 noundef 8) #11
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN6Carbon5Final1FEv(ptr noundef nonnull align 8 dereferenceable(8) %this) unnamed_addr #5 align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !11
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @_CF__carbon_thunk.Final.Main(ptr noundef nonnull align 8 dereferenceable(8) %this1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN6Carbon4BaseD2Ev(ptr noundef nonnull align 8 dead_on_return(8) dereferenceable(8) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !16
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   store ptr getelementptr inbounds inrange(-16, 32) ({ [6 x ptr] }, ptr @_ZTVN6Carbon4BaseE, i32 0, i32 0, i32 2), ptr %this1, align 8, !tbaa !14
// CHECK:STDOUT:   call void @"_C__destroy_thunk:thunk.Base.Main"(ptr noundef nonnull align 8 dereferenceable(8) %this1)
// CHECK:STDOUT:   call void @_ZN1AD2Ev(ptr noundef nonnull align 8 dead_on_return(8) dereferenceable(8) %this1) #8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN6Carbon4BaseD0Ev(ptr noundef nonnull align 8 dereferenceable(8) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !16
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @_ZN6Carbon4BaseD2Ev(ptr noundef nonnull align 8 dead_on_return(8) dereferenceable(8) %this1) #8
// CHECK:STDOUT:   call void @_ZdlPvm(ptr noundef %this1, i64 noundef 8) #11
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN6Carbon4Base1FEv(ptr noundef nonnull align 8 dereferenceable(8) %this) unnamed_addr #5 align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !16
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @_CF__carbon_thunk.Base.Main(ptr noundef nonnull align 8 dereferenceable(8) %this1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN1AC2Ev(ptr noundef nonnull align 8 dereferenceable(8) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !18
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   store ptr getelementptr inbounds inrange(-16, 24) ({ [5 x ptr] }, ptr @_ZTV1A, i32 0, i32 0, i32 2), ptr %this1, align 8, !tbaa !14
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: declare void @_ZN1AD2Ev(ptr noundef nonnull align 8 dead_on_return(8) dereferenceable(8)) unnamed_addr #7
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_ZN6Carbon5FinalD2Ev, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @__cxa_pure_virtual, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_ZNK1A1FEv, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_ZN6Carbon4BaseD2Ev, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_ZN1AC2Ev, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_ZN1AD2Ev, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { nobuiltin allocsize(0) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #4 = { nobuiltin nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #5 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #6 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #7 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #8 = { nounwind }
// CHECK:STDOUT: attributes #9 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #10 = { builtin allocsize(0) }
// CHECK:STDOUT: attributes #11 = { builtin nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3, !4, !5, !6}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!7}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "override_virtual_dtor.carbon", directory: "")
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
// CHECK:STDOUT: !12 = !{!"p1 _ZTSN6Carbon5FinalE", !13, i64 0}
// CHECK:STDOUT: !13 = !{!"any pointer", !9, i64 0}
// CHECK:STDOUT: !14 = !{!15, !15, i64 0}
// CHECK:STDOUT: !15 = !{!"vtable pointer", !10, i64 0}
// CHECK:STDOUT: !16 = !{!17, !17, i64 0}
// CHECK:STDOUT: !17 = !{!"p1 _ZTSN6Carbon4BaseE", !13, i64 0}
// CHECK:STDOUT: !18 = !{!19, !19, i64 0}
// CHECK:STDOUT: !19 = !{!"p1 _ZTS1A", !13, i64 0}
// CHECK:STDOUT: !20 = !{}
// CHECK:STDOUT: !21 = !{i64 8}
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "F", linkageName: "_CF:thunk:A.Cpp:Final.Main", scope: null, file: !1, line: 14, type: !23, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !26)
// CHECK:STDOUT: !23 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !24 = !{null, !25}
// CHECK:STDOUT: !25 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !26 = !{!27}
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !22, type: !25)
// CHECK:STDOUT: !28 = !DILocation(line: 14, column: 3, scope: !22)
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "F", linkageName: "_CF:thunk:A.Cpp:Base.Main", scope: null, file: !1, line: 19, type: !23, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !30)
// CHECK:STDOUT: !30 = !{!31}
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 1, scope: !29, type: !25)
// CHECK:STDOUT: !32 = !DILocation(line: 19, column: 3, scope: !29)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.Final.Main", scope: null, file: !1, line: 12, type: !23, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !34)
// CHECK:STDOUT: !34 = !{!35}
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !33, type: !25)
// CHECK:STDOUT: !36 = !DILocation(line: 12, column: 1, scope: !33)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "Op", linkageName: "_COp.4fc2347fc019a955:core.Destroy.Core", scope: null, file: !1, line: 12, type: !23, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !38)
// CHECK:STDOUT: !38 = !{!39}
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !37, type: !25)
// CHECK:STDOUT: !40 = !DILocation(line: 12, column: 1, scope: !37)
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "Op", linkageName: "_COp.6622c02ec7a967c3:core.Destroy.Core", scope: null, file: !1, line: 12, type: !23, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !42)
// CHECK:STDOUT: !42 = !{!43}
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !41, type: !25)
// CHECK:STDOUT: !44 = !DILocation(line: 12, column: 1, scope: !41)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "F__carbon_thunk", linkageName: "_CF__carbon_thunk.Final.Main", scope: null, file: !1, line: 14, type: !23, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !46)
// CHECK:STDOUT: !46 = !{!47}
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !45, type: !25)
// CHECK:STDOUT: !48 = !DILocation(line: 14, column: 3, scope: !45)
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.Base.Main", scope: null, file: !1, line: 17, type: !23, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !50)
// CHECK:STDOUT: !50 = !{!51}
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !49, type: !25)
// CHECK:STDOUT: !52 = !DILocation(line: 17, column: 1, scope: !49)
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "Op", linkageName: "_COp.a45b619846c20518:core.Destroy.Core", scope: null, file: !1, line: 17, type: !23, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !54)
// CHECK:STDOUT: !54 = !{!55}
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 1, scope: !53, type: !25)
// CHECK:STDOUT: !56 = !DILocation(line: 17, column: 1, scope: !53)
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "F__carbon_thunk", linkageName: "_CF__carbon_thunk.Base.Main", scope: null, file: !1, line: 19, type: !23, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !58)
// CHECK:STDOUT: !58 = !{!59}
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !57, type: !25)
// CHECK:STDOUT: !60 = !DILocation(line: 19, column: 3, scope: !57)
