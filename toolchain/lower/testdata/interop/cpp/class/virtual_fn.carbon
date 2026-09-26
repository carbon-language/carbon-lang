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
// CHECK:STDOUT:   store ptr %call, ptr %p, align 8, !tbaa !4
// CHECK:STDOUT:   %0 = load ptr, ptr %p, align 8, !tbaa !4
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
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !4
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @_ZN1AC2Ev(ptr noundef nonnull align 8 dereferenceable(8) %this1) #8
// CHECK:STDOUT:   store ptr getelementptr inbounds inrange(-16, 24) ({ [5 x ptr] }, ptr @_ZTVN6Carbon5FinalE, i32 0, i32 0, i32 2), ptr %this1, align 8, !tbaa !8
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
// CHECK:STDOUT:   %call = call noalias noundef nonnull ptr @_Znwm(i64 noundef 8) #10
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
// CHECK:STDOUT:   store ptr getelementptr inbounds inrange(-16, 24) ({ [5 x ptr] }, ptr @_ZTVN6Carbon4BaseE, i32 0, i32 0, i32 2), ptr %this1, align 8, !tbaa !8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZNK1A1FEv.carbon_thunk._(ptr noundef nonnull align 8 dereferenceable(8) %this) #5 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !87
// CHECK:STDOUT:   %0 = load ptr, ptr %this.addr, align 8, !tbaa !87, !nonnull !90, !align !91
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
// CHECK:STDOUT:   call void @"_CSelfDestruct.28387b086eba9957:core.Destroy.Core"(ptr %self), !dbg !39
// CHECK:STDOUT:   ret void, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: declare void @_ZN1AD1Ev(ptr noundef nonnull align 8 dead_on_return(8) dereferenceable(8)) unnamed_addr #7
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f2cd4feee22137a3:core.Destroy.Core"(ptr %self) #8 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %A.cpp_destructor.call.vtable = load ptr, ptr %self, align 8, !dbg !43
// CHECK:STDOUT:   %A.cpp_destructor.call = getelementptr ptr, ptr %A.cpp_destructor.call.vtable, i32 0, !dbg !43
// CHECK:STDOUT:   %A.cpp_destructor.call.memptr.virtualfn = load ptr, ptr %A.cpp_destructor.call, align 8, !dbg !43
// CHECK:STDOUT:   call void %A.cpp_destructor.call.memptr.virtualfn(ptr %self), !dbg !43
// CHECK:STDOUT:   ret void, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.e9cc9e95545cc62f:core.Destroy.Core"(ptr %self) #8 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12.base = getelementptr inbounds nuw { [8 x i8] }, ptr %self, i32 0, i32 0, !dbg !47
// CHECK:STDOUT:   call void @"_CSelfDestruct.f2cd4feee22137a3:core.Destroy.Core"(ptr %.loc12.base), !dbg !47
// CHECK:STDOUT:   ret void, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.e9cc9e95545cc62f:core.Destroy.Core"(ptr %self) #8 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.e9cc9e95545cc62f:core.Destroy.Core"(ptr %self), !dbg !51
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.28387b086eba9957:core.Destroy.Core"(ptr %self) #8 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12.base = getelementptr inbounds nuw { [8 x i8] }, ptr %self, i32 0, i32 0, !dbg !55
// CHECK:STDOUT:   call void @"_CSelfDestruct.f2cd4feee22137a3:core.Destroy.Core"(ptr %.loc12.base), !dbg !55
// CHECK:STDOUT:   ret void, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.28387b086eba9957:core.Destroy.Core"(ptr %self) #8 !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.28387b086eba9957:core.Destroy.Core"(ptr %self), !dbg !59
// CHECK:STDOUT:   ret void, !dbg !59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF__carbon_thunk.Final.Main(ptr %self) #8 !dbg !61 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Final.Main(ptr %self), !dbg !63
// CHECK:STDOUT:   ret void, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CF.Base.Main(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_CF:thunk:A.Cpp:Base.Main"(ptr %self) #6 !dbg !65 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Base.Main(ptr %self), !dbg !68
// CHECK:STDOUT:   ret void, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_C__destroy_thunk:thunk.Base.Main"(ptr %self) #6 !dbg !70 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSelfDestruct.294df737cd074d60:core.Destroy.Core"(ptr %self), !dbg !72
// CHECK:STDOUT:   ret void, !dbg !72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.294df737cd074d60:core.Destroy.Core"(ptr %self) #8 !dbg !74 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17.base = getelementptr inbounds nuw { [8 x i8] }, ptr %self, i32 0, i32 0, !dbg !76
// CHECK:STDOUT:   call void @"_CSelfDestruct.f2cd4feee22137a3:core.Destroy.Core"(ptr %.loc17.base), !dbg !76
// CHECK:STDOUT:   ret void, !dbg !76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.294df737cd074d60:core.Destroy.Core"(ptr %self) #8 !dbg !78 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.294df737cd074d60:core.Destroy.Core"(ptr %self), !dbg !80
// CHECK:STDOUT:   ret void, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF__carbon_thunk.Base.Main(ptr %self) #8 !dbg !82 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CF.Base.Main(ptr %self), !dbg !84
// CHECK:STDOUT:   ret void, !dbg !84
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
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !4
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @_ZN6Carbon5FinalD2Ev(ptr noundef nonnull align 8 dead_on_return(8) dereferenceable(8) %this1) #8
// CHECK:STDOUT:   call void @_ZdlPvm(ptr noundef %this1, i64 noundef 8) #11
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
// CHECK:STDOUT:   store ptr getelementptr inbounds inrange(-16, 24) ({ [5 x ptr] }, ptr @_ZTVN6Carbon4BaseE, i32 0, i32 0, i32 2), ptr %this1, align 8, !tbaa !8
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
// CHECK:STDOUT:   call void @_ZdlPvm(ptr noundef %this1, i64 noundef 8) #11
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
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !87
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   store ptr getelementptr inbounds inrange(-16, 24) ({ [5 x ptr] }, ptr @_ZTV1A, i32 0, i32 0, i32 2), ptr %this1, align 8, !tbaa !8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: declare void @_ZN1AD2Ev(ptr noundef nonnull align 8 dead_on_return(8) dereferenceable(8)) unnamed_addr #7
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_ZN6Carbon5FinalD2Ev, { 2, 1, 0 }
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
// CHECK:STDOUT: attributes #9 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #10 = { builtin allocsize(0) memory(inaccessiblemem: readwrite, errnomem: write) }
// CHECK:STDOUT: attributes #11 = { builtin nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!28}
// CHECK:STDOUT: !llvm.module.flags = !{!93, !94, !95, !11, !12}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!98}
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
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f2cd4feee22137a3:core.Destroy.Core", scope: null, file: !27, line: 12, type: !31, spFlags: DISPFlagDefinition, unit: !28, retainedNodes: !44)
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 1, scope: !41, type: !29)
// CHECK:STDOUT: !43 = !DILocation(line: 12, column: 1, scope: !41)
// CHECK:STDOUT: !44 = !{!42}
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.e9cc9e95545cc62f:core.Destroy.Core", scope: null, file: !27, line: 12, type: !31, spFlags: DISPFlagDefinition, unit: !28, retainedNodes: !48)
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 1, scope: !45, type: !29)
// CHECK:STDOUT: !47 = !DILocation(line: 12, column: 1, scope: !45)
// CHECK:STDOUT: !48 = !{!46}
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.e9cc9e95545cc62f:core.Destroy.Core", scope: null, file: !27, line: 12, type: !31, spFlags: DISPFlagDefinition, unit: !28, retainedNodes: !52)
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 1, scope: !49, type: !29)
// CHECK:STDOUT: !51 = !DILocation(line: 12, column: 1, scope: !49)
// CHECK:STDOUT: !52 = !{!50}
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.28387b086eba9957:core.Destroy.Core", scope: null, file: !27, line: 12, type: !31, spFlags: DISPFlagDefinition, unit: !28, retainedNodes: !56)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !53, type: !29)
// CHECK:STDOUT: !55 = !DILocation(line: 12, column: 1, scope: !53)
// CHECK:STDOUT: !56 = !{!54}
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.28387b086eba9957:core.Destroy.Core", scope: null, file: !27, line: 12, type: !31, spFlags: DISPFlagDefinition, unit: !28, retainedNodes: !60)
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 1, scope: !57, type: !29)
// CHECK:STDOUT: !59 = !DILocation(line: 12, column: 1, scope: !57)
// CHECK:STDOUT: !60 = !{!58}
// CHECK:STDOUT: !61 = distinct !DISubprogram(name: "F__carbon_thunk", linkageName: "_CF__carbon_thunk.Final.Main", scope: null, file: !27, line: 14, type: !31, spFlags: DISPFlagDefinition, unit: !28, retainedNodes: !64)
// CHECK:STDOUT: !62 = !DILocalVariable(arg: 1, scope: !61, type: !29)
// CHECK:STDOUT: !63 = !DILocation(line: 14, column: 3, scope: !61)
// CHECK:STDOUT: !64 = !{!62}
// CHECK:STDOUT: !65 = distinct !DISubprogram(name: "F", linkageName: "_CF:thunk:A.Cpp:Base.Main", scope: null, file: !27, line: 19, type: !31, spFlags: DISPFlagDefinition, unit: !28, retainedNodes: !69)
// CHECK:STDOUT: !66 = !DILocalVariable(arg: 1, scope: !65, type: !29)
// CHECK:STDOUT: !68 = !DILocation(line: 19, column: 3, scope: !65)
// CHECK:STDOUT: !69 = !{!66}
// CHECK:STDOUT: !70 = distinct !DISubprogram(name: "__destroy_thunk", linkageName: "_C__destroy_thunk:thunk.Base.Main", scope: null, file: !27, line: 17, type: !31, spFlags: DISPFlagDefinition, unit: !28, retainedNodes: !73)
// CHECK:STDOUT: !71 = !DILocalVariable(arg: 1, scope: !70, type: !29)
// CHECK:STDOUT: !72 = !DILocation(line: 17, column: 1, scope: !70)
// CHECK:STDOUT: !73 = !{!71}
// CHECK:STDOUT: !74 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.294df737cd074d60:core.Destroy.Core", scope: null, file: !27, line: 17, type: !31, spFlags: DISPFlagDefinition, unit: !28, retainedNodes: !77)
// CHECK:STDOUT: !75 = !DILocalVariable(arg: 1, scope: !74, type: !29)
// CHECK:STDOUT: !76 = !DILocation(line: 17, column: 1, scope: !74)
// CHECK:STDOUT: !77 = !{!75}
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.294df737cd074d60:core.Destroy.Core", scope: null, file: !27, line: 17, type: !31, spFlags: DISPFlagDefinition, unit: !28, retainedNodes: !81)
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 1, scope: !78, type: !29)
// CHECK:STDOUT: !80 = !DILocation(line: 17, column: 1, scope: !78)
// CHECK:STDOUT: !81 = !{!79}
// CHECK:STDOUT: !82 = distinct !DISubprogram(name: "F__carbon_thunk", linkageName: "_CF__carbon_thunk.Base.Main", scope: null, file: !27, line: 19, type: !31, spFlags: DISPFlagDefinition, unit: !28, retainedNodes: !85)
// CHECK:STDOUT: !83 = !DILocalVariable(arg: 1, scope: !82, type: !29)
// CHECK:STDOUT: !84 = !DILocation(line: 19, column: 3, scope: !82)
// CHECK:STDOUT: !85 = !{!83}
// CHECK:STDOUT: !86 = !{!"p1 _ZTS1A", !2, i64 0}
// CHECK:STDOUT: !87 = !{!86, !86, i64 0}
// CHECK:STDOUT: !90 = !{}
// CHECK:STDOUT: !91 = !{i64 8}
// CHECK:STDOUT: !93 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !94 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !95 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !96 = !{!"int", !1, i64 0}
// CHECK:STDOUT: !97 = !{!"__libc_errno", !96, i64 0}
// CHECK:STDOUT: !98 = !{!97, !96, i64 0}
// CHECK:STDOUT:
