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

// CHECK:STDOUT: ; ---
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
// CHECK:STDOUT: @_ZTV1A = available_externally constant { [5 x ptr] } { [5 x ptr] [ptr null, ptr @_ZTI1A, ptr @__cxa_pure_virtual, ptr @__cxa_pure_virtual, ptr @_ZNK1A1FEv] }, align 8
// CHECK:STDOUT: @_ZTVN6Carbon5FinalE = linkonce_odr dso_local constant { [5 x ptr] } { [5 x ptr] [ptr null, ptr @_ZTIN6Carbon5FinalE, ptr @_ZN6Carbon5FinalD2Ev, ptr @_ZN6Carbon5FinalD0Ev, ptr @_ZNK6Carbon5Final1FEv] }, comdat, align 8
// CHECK:STDOUT: @_ZTVN6Carbon4BaseE = linkonce_odr dso_local constant { [5 x ptr] } { [5 x ptr] [ptr null, ptr @_ZTIN6Carbon4BaseE, ptr @_ZN6Carbon4BaseD2Ev, ptr @_ZN6Carbon4BaseD0Ev, ptr @_ZNK6Carbon4Base1FEv] }, comdat, align 8
// CHECK:STDOUT: @_ZTI1A = external constant ptr
// CHECK:STDOUT: @_ZTIN6Carbon5FinalE = linkonce_odr dso_local constant { ptr, ptr, ptr } { ptr getelementptr inbounds (i8, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 16), ptr @_ZTSN6Carbon5FinalE, ptr @_ZTI1A }, comdat, align 8
// CHECK:STDOUT: @_ZTVN10__cxxabiv120__si_class_type_infoE = external global [0 x ptr]
// CHECK:STDOUT: @_ZTSN6Carbon5FinalE = linkonce_odr dso_local constant [16 x i8] c"N6Carbon5FinalE\00", comdat, align 1
// CHECK:STDOUT: @_ZTIN6Carbon4BaseE = linkonce_odr dso_local constant { ptr, ptr, ptr } { ptr getelementptr inbounds (i8, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 16), ptr @_ZTSN6Carbon4BaseE, ptr @_ZTI1A }, comdat, align 8
// CHECK:STDOUT: @_ZTSN6Carbon4BaseE = linkonce_odr dso_local constant [15 x i8] c"N6Carbon4BaseE\00", comdat, align 1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define dso_local void @_Z16delete_new_Finalv() #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %p = alloca ptr, align 8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %p) #8
// CHECK:STDOUT:   %call = call noalias noundef nonnull ptr @_Znwm(i64 noundef 8) #11
// CHECK:STDOUT:   call void @_ZN6Carbon5FinalC2Ev(ptr noundef nonnull align 8 dereferenceable(8) %call) #8
// CHECK:STDOUT:   store ptr %call, ptr %p, align 8, !tbaa !4
// CHECK:STDOUT:   %0 = load ptr, ptr %p, align 8, !tbaa !4
// CHECK:STDOUT:   %isnull = icmp eq ptr %0, null
// CHECK:STDOUT:   br i1 %isnull, label %delete.end, label %delete.notnull
// CHECK:STDOUT:
// CHECK:STDOUT: delete.notnull:                                   ; preds = %entry
// CHECK:STDOUT:   call void @_ZN6Carbon5FinalD2Ev(ptr noundef nonnull align 8 dead_on_return(8) dereferenceable(8) %0) #8
// CHECK:STDOUT:   call void @_ZdlPvm(ptr noundef %0, i64 noundef 8) #12
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
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !4
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @_ZN1AC2Ev(ptr noundef nonnull align 8 dereferenceable(8) %this1) #8
// CHECK:STDOUT:   store ptr getelementptr inbounds inrange(-16, 24) (i8, ptr @_ZTVN6Carbon5FinalE, i64 16), ptr %this1, align 8, !tbaa !8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN6Carbon5FinalD2Ev(ptr noundef nonnull align 8 dead_on_return(8) dereferenceable(8) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !4
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
// CHECK:STDOUT:   %call = call noalias noundef nonnull ptr @_Znwm(i64 noundef 8) #11
// CHECK:STDOUT:   call void @_ZN6Carbon4BaseC2Ev(ptr noundef nonnull align 8 dereferenceable(8) %call) #8
// CHECK:STDOUT:   store ptr %call, ptr %p, align 8, !tbaa !6
// CHECK:STDOUT:   %0 = load ptr, ptr %p, align 8, !tbaa !6
// CHECK:STDOUT:   %isnull = icmp eq ptr %0, null
// CHECK:STDOUT:   br i1 %isnull, label %delete.end, label %delete.notnull
// CHECK:STDOUT:
// CHECK:STDOUT: delete.notnull:                                   ; preds = %entry
// CHECK:STDOUT:   %vtable = load ptr, ptr %0, align 8, !tbaa !8
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
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !6
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @_ZN1AC2Ev(ptr noundef nonnull align 8 dereferenceable(8) %this1) #8
// CHECK:STDOUT:   store ptr getelementptr inbounds inrange(-16, 24) (i8, ptr @_ZTVN6Carbon4BaseE, i64 16), ptr %this1, align 8, !tbaa !8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZNK1A1FEv.carbon_thunk._(ptr noundef nonnull align 8 dereferenceable(8) %this) #5 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !96
// CHECK:STDOUT:   %0 = load ptr, ptr %this.addr, align 8, !tbaa !96, !nonnull !99, !align !100
// CHECK:STDOUT:   %vtable = load ptr, ptr %0, align 8, !tbaa !8
// CHECK:STDOUT:   %vfn = getelementptr inbounds ptr, ptr %vtable, i64 2
// CHECK:STDOUT:   %1 = load ptr, ptr %vfn, align 8
// CHECK:STDOUT:   call void %1(ptr noundef nonnull align 8 dereferenceable(8) %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Final.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_CF:thunk:A.Cpp:Final.Main"(ptr %self) #6 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Final.Main(ptr %self), !dbg !35
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_C__destroy_thunk:thunk.Final.Main"(ptr %self) #6 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7b90814fbe6b5a2a(ptr %self), !dbg !39
// CHECK:STDOUT:   ret void, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:DestroyImpl.20eb678cb24c67a6.Core:DestroyImpl.Core"(ptr %self) #6 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: declare void @_ZN1AD1Ev(ptr noundef nonnull align 8 dead_on_return(8) dereferenceable(8)) unnamed_addr #7
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.e9cc9e95545cc62f:core.SubobjectDestroy.Core"(ptr %self) #8 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12_13.1.base = getelementptr inbounds nuw { [8 x i8] }, ptr %self, i32 0, i32 0, !dbg !47
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.6191dcb71c708f97(ptr %.loc12_13.1.base), !dbg !47
// CHECK:STDOUT:   ret void, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.28387b086eba9957:core.SubobjectDestroy.Core"(ptr %self) #8 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12_13.1.base = getelementptr inbounds nuw { [8 x i8] }, ptr %self, i32 0, i32 0, !dbg !51
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.6191dcb71c708f97(ptr %.loc12_13.1.base), !dbg !51
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF__carbon_thunk.Final.Main(ptr %self) #8 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Final.Main(ptr %self), !dbg !55
// CHECK:STDOUT:   ret void, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Base.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_CF:thunk:A.Cpp:Base.Main"(ptr %self) #6 !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Base.Main(ptr %self), !dbg !60
// CHECK:STDOUT:   ret void, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_C__destroy_thunk:thunk.Base.Main"(ptr %self) #6 !dbg !62 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.77bbe39226a4db33(ptr %self), !dbg !64
// CHECK:STDOUT:   ret void, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.294df737cd074d60:core.SubobjectDestroy.Core"(ptr %self) #8 !dbg !66 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17_17.1.base = getelementptr inbounds nuw { [8 x i8] }, ptr %self, i32 0, i32 0, !dbg !68
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.6191dcb71c708f97(ptr %.loc17_17.1.base), !dbg !68
// CHECK:STDOUT:   ret void, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF__carbon_thunk.Base.Main(ptr %self) #8 !dbg !70 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Base.Main(ptr %self), !dbg !72
// CHECK:STDOUT:   ret void, !dbg !72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.7b90814fbe6b5a2a(ptr %t) #8 !dbg !74 {
// CHECK:STDOUT:   call void @"_COp.28387b086eba9957:core.SubobjectDestroy.Core"(ptr %t), !dbg !78
// CHECK:STDOUT:   ret void, !dbg !79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.6191dcb71c708f97(ptr %t) #8 !dbg !81 {
// CHECK:STDOUT:   call void @"_COp:thunk:DestroyImpl.20eb678cb24c67a6.Core:DestroyImpl.Core"(ptr %t), !dbg !84
// CHECK:STDOUT:   %vtable = load ptr, ptr %t, align 8, !dbg !85
// CHECK:STDOUT:   %1 = call ptr @llvm.load.relative.i32(ptr %vtable, i32 0), !dbg !85
// CHECK:STDOUT:   call void %1(ptr %t), !dbg !85
// CHECK:STDOUT:   ret void, !dbg !86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.77bbe39226a4db33(ptr %t) #8 !dbg !88 {
// CHECK:STDOUT:   call void @"_COp.294df737cd074d60:core.SubobjectDestroy.Core"(ptr %t), !dbg !92
// CHECK:STDOUT:   ret void, !dbg !93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
// CHECK:STDOUT: declare ptr @llvm.load.relative.i32(ptr, i32) #9
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @__cxa_pure_virtual() unnamed_addr
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZNK1A1FEv(ptr noundef nonnull align 8 dereferenceable(8)) unnamed_addr #10
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN6Carbon5FinalD0Ev(ptr noundef nonnull align 8 dereferenceable(8) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !4
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @_ZN6Carbon5FinalD2Ev(ptr noundef nonnull align 8 dead_on_return(8) dereferenceable(8) %this1) #8
// CHECK:STDOUT:   call void @_ZdlPvm(ptr noundef %this1, i64 noundef 8) #12
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZNK6Carbon5Final1FEv(ptr noundef nonnull align 8 dereferenceable(8) %this) unnamed_addr #5 align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !4
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @_CF__carbon_thunk.Final.Main(ptr noundef nonnull align 8 dereferenceable(8) %this1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN6Carbon4BaseD2Ev(ptr noundef nonnull align 8 dead_on_return(8) dereferenceable(8) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !6
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   store ptr getelementptr inbounds inrange(-16, 24) (i8, ptr @_ZTVN6Carbon4BaseE, i64 16), ptr %this1, align 8, !tbaa !8
// CHECK:STDOUT:   call void @"_C__destroy_thunk:thunk.Base.Main"(ptr noundef nonnull align 8 dereferenceable(8) %this1)
// CHECK:STDOUT:   call void @_ZN1AD2Ev(ptr noundef nonnull align 8 dead_on_return(8) dereferenceable(8) %this1) #8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN6Carbon4BaseD0Ev(ptr noundef nonnull align 8 dereferenceable(8) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !6
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @_ZN6Carbon4BaseD2Ev(ptr noundef nonnull align 8 dead_on_return(8) dereferenceable(8) %this1) #8
// CHECK:STDOUT:   call void @_ZdlPvm(ptr noundef %this1, i64 noundef 8) #12
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZNK6Carbon4Base1FEv(ptr noundef nonnull align 8 dereferenceable(8) %this) unnamed_addr #5 align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !6
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @_CF__carbon_thunk.Base.Main(ptr noundef nonnull align 8 dereferenceable(8) %this1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN1AC2Ev(ptr noundef nonnull align 8 dereferenceable(8) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !96
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   store ptr getelementptr inbounds inrange(-16, 24) (i8, ptr @_ZTV1A, i64 16), ptr %this1, align 8, !tbaa !8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: declare void @_ZN1AD2Ev(ptr noundef nonnull align 8 dead_on_return(8) dereferenceable(8)) unnamed_addr #7
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_ZN6Carbon5FinalD2Ev, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.6191dcb71c708f97, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @__cxa_pure_virtual, { 1, 0 }
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
// CHECK:STDOUT: attributes #9 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
// CHECK:STDOUT: attributes #10 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #11 = { builtin allocsize(0) memory(inaccessiblemem: readwrite, errnomem: write) }
// CHECK:STDOUT: attributes #12 = { builtin nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!28}
// CHECK:STDOUT: !llvm.module.flags = !{!102, !103, !104, !11, !12}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!107}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !1 = !{!"omnipotent char", !0, i64 0}
// CHECK:STDOUT: !2 = !{!"any pointer", !1, i64 0}
// CHECK:STDOUT: !3 = !{!"p1 _ZTSN6Carbon5FinalE", !2, i64 0}
// CHECK:STDOUT: !4 = !{!3, !3, i64 0}
// CHECK:STDOUT: !5 = !{!"p1 _ZTSN6Carbon4BaseE", !2, i64 0}
// CHECK:STDOUT: !6 = !{!5, !5, i64 0}
// CHECK:STDOUT: !7 = !{!"vtable pointer", !0, i64 0}
// CHECK:STDOUT: !8 = !{!7, !7, i64 0}
// CHECK:STDOUT: !11 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !12 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !19 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !27 = !DIFile(filename: "override_virtual_dtor.carbon", directory: "")
// CHECK:STDOUT: !28 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !27, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !29 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !30 = !{null, !29}
// CHECK:STDOUT: !31 = !DISubroutineType(types: !30)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "F", linkageName: "_CF:thunk:A.Cpp:Final.Main", scope: null, file: !27, line: 14, type: !31, spFlags: DISPFlagDefinition, unit: !28, retainedNodes: !36)
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 1, scope: !32, type: !29)
// CHECK:STDOUT: !35 = !DILocation(line: 14, column: 3, scope: !32)
// CHECK:STDOUT: !36 = !{!33}
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.Final.Main", scope: null, file: !27, line: 12, type: !31, spFlags: DISPFlagDefinition, unit: !28, retainedNodes: !40)
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !37, type: !29)
// CHECK:STDOUT: !39 = !DILocation(line: 12, column: 1, scope: !37)
// CHECK:STDOUT: !40 = !{!38}
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:DestroyImpl.20eb678cb24c67a6.Core:DestroyImpl.Core", scope: null, file: !27, type: !31, spFlags: DISPFlagDefinition, unit: !28, retainedNodes: !44)
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 1, scope: !41, type: !29)
// CHECK:STDOUT: !43 = !DILocation(line: 0, scope: !41)
// CHECK:STDOUT: !44 = !{!42}
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "Op", linkageName: "_COp.e9cc9e95545cc62f:core.SubobjectDestroy.Core", scope: null, file: !27, line: 12, type: !31, spFlags: DISPFlagDefinition, unit: !28, retainedNodes: !48)
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 1, scope: !45, type: !29)
// CHECK:STDOUT: !47 = !DILocation(line: 12, column: 1, scope: !45)
// CHECK:STDOUT: !48 = !{!46}
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "Op", linkageName: "_COp.28387b086eba9957:core.SubobjectDestroy.Core", scope: null, file: !27, line: 12, type: !31, spFlags: DISPFlagDefinition, unit: !28, retainedNodes: !52)
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 1, scope: !49, type: !29)
// CHECK:STDOUT: !51 = !DILocation(line: 12, column: 1, scope: !49)
// CHECK:STDOUT: !52 = !{!50}
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "F__carbon_thunk", linkageName: "_CF__carbon_thunk.Final.Main", scope: null, file: !27, line: 14, type: !31, spFlags: DISPFlagDefinition, unit: !28, retainedNodes: !56)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !53, type: !29)
// CHECK:STDOUT: !55 = !DILocation(line: 14, column: 3, scope: !53)
// CHECK:STDOUT: !56 = !{!54}
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "F", linkageName: "_CF:thunk:A.Cpp:Base.Main", scope: null, file: !27, line: 19, type: !31, spFlags: DISPFlagDefinition, unit: !28, retainedNodes: !61)
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 1, scope: !57, type: !29)
// CHECK:STDOUT: !60 = !DILocation(line: 19, column: 3, scope: !57)
// CHECK:STDOUT: !61 = !{!58}
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.Base.Main", scope: null, file: !27, line: 17, type: !31, spFlags: DISPFlagDefinition, unit: !28, retainedNodes: !65)
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !62, type: !29)
// CHECK:STDOUT: !64 = !DILocation(line: 17, column: 1, scope: !62)
// CHECK:STDOUT: !65 = !{!63}
// CHECK:STDOUT: !66 = distinct !DISubprogram(name: "Op", linkageName: "_COp.294df737cd074d60:core.SubobjectDestroy.Core", scope: null, file: !27, line: 17, type: !31, spFlags: DISPFlagDefinition, unit: !28, retainedNodes: !69)
// CHECK:STDOUT: !67 = !DILocalVariable(arg: 1, scope: !66, type: !29)
// CHECK:STDOUT: !68 = !DILocation(line: 17, column: 1, scope: !66)
// CHECK:STDOUT: !69 = !{!67}
// CHECK:STDOUT: !70 = distinct !DISubprogram(name: "F__carbon_thunk", linkageName: "_CF__carbon_thunk.Base.Main", scope: null, file: !27, line: 19, type: !31, spFlags: DISPFlagDefinition, unit: !28, retainedNodes: !73)
// CHECK:STDOUT: !71 = !DILocalVariable(arg: 1, scope: !70, type: !29)
// CHECK:STDOUT: !72 = !DILocation(line: 19, column: 3, scope: !70)
// CHECK:STDOUT: !73 = !{!71}
// CHECK:STDOUT: !74 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.7b90814fbe6b5a2a", scope: null, file: !19, line: 26, type: !31, spFlags: DISPFlagDefinition, unit: !28, retainedNodes: !80)
// CHECK:STDOUT: !75 = !DILocalVariable(arg: 1, scope: !74, type: !29)
// CHECK:STDOUT: !78 = !DILocation(line: 28, column: 5, scope: !74)
// CHECK:STDOUT: !79 = !DILocation(line: 26, column: 1, scope: !74)
// CHECK:STDOUT: !80 = !{!75}
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.6191dcb71c708f97", scope: null, file: !19, line: 26, type: !31, spFlags: DISPFlagDefinition, unit: !28, retainedNodes: !87)
// CHECK:STDOUT: !82 = !DILocalVariable(arg: 1, scope: !81, type: !29)
// CHECK:STDOUT: !84 = !DILocation(line: 27, column: 5, scope: !81)
// CHECK:STDOUT: !85 = !DILocation(line: 28, column: 5, scope: !81)
// CHECK:STDOUT: !86 = !DILocation(line: 26, column: 1, scope: !81)
// CHECK:STDOUT: !87 = !{!82}
// CHECK:STDOUT: !88 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.77bbe39226a4db33", scope: null, file: !19, line: 26, type: !31, spFlags: DISPFlagDefinition, unit: !28, retainedNodes: !94)
// CHECK:STDOUT: !89 = !DILocalVariable(arg: 1, scope: !88, type: !29)
// CHECK:STDOUT: !92 = !DILocation(line: 28, column: 5, scope: !88)
// CHECK:STDOUT: !93 = !DILocation(line: 26, column: 1, scope: !88)
// CHECK:STDOUT: !94 = !{!89}
// CHECK:STDOUT: !95 = !{!"p1 _ZTS1A", !2, i64 0}
// CHECK:STDOUT: !96 = !{!95, !95, i64 0}
// CHECK:STDOUT: !99 = !{}
// CHECK:STDOUT: !100 = !{i64 8}
// CHECK:STDOUT: !102 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !103 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !104 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !105 = !{!"int", !1, i64 0}
// CHECK:STDOUT: !106 = !{!"__libc_errno", !105, i64 0}
// CHECK:STDOUT: !107 = !{!106, !105, i64 0}
// CHECK:STDOUT:
