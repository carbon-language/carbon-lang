// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/std_initializer_list_pointer_pointer.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/std_initializer_list_pointer_pointer.carbon

// --- initializer_list_pointer_pointer.h

namespace std {
  using size_t = decltype(sizeof(0));

  template <typename T> class initializer_list {
   public:
    initializer_list() {}

   private:
    T* begin_ = nullptr;
    T* end_ = nullptr;
  };
}

struct vector_like {
  vector_like(std::initializer_list<int> list) {}
  ~vector_like() {}
};

// --- nontrivial_dtor.h

struct nontrivial_dtor {
  nontrivial_dtor() {}
  ~nontrivial_dtor() {}
};

// --- use_pointer_pointer.carbon

library "[[@TEST_NAME]]";

import Cpp library "initializer_list_pointer_pointer.h";
import Cpp library "nontrivial_dtor.h";

// TODO: When we support more precise modeling of temporary lifetimes, ensure
// that this is called *before* the elements of the underlying array are
// destroyed in the non-vector-like cases.
fn WithinLifetime();

fn InitDirectly() {
  var _: Cpp.std.initializer_list(i32) = (1, 2, 3);
  WithinLifetime();
}

fn InitVectorLike() {
  var _: Cpp.vector_like = (1, 2, 3);
  WithinLifetime();
}

fn InitNontrivialDtor() {
  var _: Cpp.std.initializer_list(Cpp.nontrivial_dtor) = ((), (), ());
  WithinLifetime();
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'use_pointer_pointer.carbon'
// CHECK:STDOUT: source_filename = "use_pointer_pointer.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %"class.std::initializer_list" = type { ptr, ptr }
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN11vector_likeD2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN15nontrivial_dtorD2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN11vector_likeC2ESt16initializer_listIiE = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN15nontrivial_dtorC2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: @array = internal constant [3 x i32] [i32 1, i32 2, i32 3]
// CHECK:STDOUT: @array.loc13_50.15 = internal constant [3 x i32] [i32 1, i32 2, i32 3]
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CWithinLifetime.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CInitDirectly.Main() #0 !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca [16 x i8], align 8, !dbg !15
// CHECK:STDOUT:   %.loc13_50.3.temp = alloca [3 x i32], align 4, !dbg !16
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !15
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc13_50.3.temp), !dbg !16
// CHECK:STDOUT:   %.loc13_50.4.array.index = getelementptr inbounds [3 x i32], ptr %.loc13_50.3.temp, i32 0, i64 0, !dbg !16
// CHECK:STDOUT:   %.loc13_50.7.array.index = getelementptr inbounds [3 x i32], ptr %.loc13_50.3.temp, i32 0, i64 1, !dbg !16
// CHECK:STDOUT:   %.loc13_50.10.array.index = getelementptr inbounds [3 x i32], ptr %.loc13_50.3.temp, i32 0, i64 2, !dbg !16
// CHECK:STDOUT:   %initializer_list.initializer_list.call.init_list.begin = getelementptr inbounds nuw [16 x i8], ptr %_.var, i32 0, i32 0, !dbg !15
// CHECK:STDOUT:   store ptr @array.loc13_50.15, ptr %initializer_list.initializer_list.call.init_list.begin, align 8, !dbg !15
// CHECK:STDOUT:   %initializer_list.initializer_list.call.init_list.end = getelementptr inbounds nuw [16 x i8], ptr %_.var, i32 0, i32 8, !dbg !15
// CHECK:STDOUT:   store ptr getelementptr inbounds ([3 x i32], ptr @array.loc13_50.15, i32 1), ptr %initializer_list.initializer_list.call.init_list.end, align 8, !dbg !15
// CHECK:STDOUT:   call void @_CWithinLifetime.Main(), !dbg !18
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.8b2abdfe25f0b189(ptr @array), !dbg !16
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c3847e2d14f33681(ptr %_.var), !dbg !15
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.b3a0adb9db7bb49e:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc13_50.1.array.index = getelementptr inbounds [3 x i32], ptr %self, i32 0, i64 2, !dbg !22
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.52dd027a4dd17420(ptr %.loc13_50.1.array.index), !dbg !22
// CHECK:STDOUT:   %.loc13_50.4.array.index = getelementptr inbounds [3 x i32], ptr %self, i32 0, i64 1, !dbg !22
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.52dd027a4dd17420(ptr %.loc13_50.4.array.index), !dbg !22
// CHECK:STDOUT:   %.loc13_50.7.array.index = getelementptr inbounds [3 x i32], ptr %self, i32 0, i64 0, !dbg !22
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.52dd027a4dd17420(ptr %.loc13_50.7.array.index), !dbg !22
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:DestroyImpl.e0350233115f3f94.Core:DestroyImpl.Core"(ptr %self) #1 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CInitVectorLike.Main() #0 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca [1 x i8], align 1, !dbg !31
// CHECK:STDOUT:   %.loc18_36.3.temp = alloca [16 x i8], align 8, !dbg !32
// CHECK:STDOUT:   %.loc18_36.5.temp = alloca [3 x i32], align 4, !dbg !32
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !31
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_36.3.temp), !dbg !32
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_36.5.temp), !dbg !32
// CHECK:STDOUT:   %.loc18_36.6.array.index = getelementptr inbounds [3 x i32], ptr %.loc18_36.5.temp, i32 0, i64 0, !dbg !32
// CHECK:STDOUT:   %.loc18_36.9.array.index = getelementptr inbounds [3 x i32], ptr %.loc18_36.5.temp, i32 0, i64 1, !dbg !32
// CHECK:STDOUT:   %.loc18_36.12.array.index = getelementptr inbounds [3 x i32], ptr %.loc18_36.5.temp, i32 0, i64 2, !dbg !32
// CHECK:STDOUT:   %initializer_list.initializer_list.call.init_list.begin = getelementptr inbounds nuw [16 x i8], ptr %.loc18_36.3.temp, i32 0, i32 0, !dbg !32
// CHECK:STDOUT:   store ptr @array.loc13_50.15, ptr %initializer_list.initializer_list.call.init_list.begin, align 8, !dbg !32
// CHECK:STDOUT:   %initializer_list.initializer_list.call.init_list.end = getelementptr inbounds nuw [16 x i8], ptr %.loc18_36.3.temp, i32 0, i32 8, !dbg !32
// CHECK:STDOUT:   store ptr getelementptr inbounds ([3 x i32], ptr @array.loc13_50.15, i32 1), ptr %initializer_list.initializer_list.call.init_list.end, align 8, !dbg !32
// CHECK:STDOUT:   call void @_ZN11vector_likeC1ESt16initializer_listIiE.carbon_thunk._(ptr %.loc18_36.3.temp, ptr %_.var), !dbg !31
// CHECK:STDOUT:   call void @_CWithinLifetime.Main(), !dbg !34
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c3847e2d14f33681(ptr %.loc18_36.3.temp), !dbg !32
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.8b2abdfe25f0b189(ptr @array), !dbg !32
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f8cb70ab0b13eb2c(ptr %_.var), !dbg !31
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN11vector_likeC1ESt16initializer_listIiE.carbon_thunk._(ptr noundef %list, ptr noundef %return) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %list.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %"class.std::initializer_list", align 8
// CHECK:STDOUT:   store ptr %list, ptr %list.addr, align 8, !tbaa !112
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !114
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !114
// CHECK:STDOUT:   %1 = load ptr, ptr %list.addr, align 8, !tbaa !112
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %agg.tmp, ptr align 8 %1, i64 16, i1 false), !tbaa.struct !119
// CHECK:STDOUT:   %2 = getelementptr inbounds nuw { ptr, ptr }, ptr %agg.tmp, i32 0, i32 0
// CHECK:STDOUT:   %3 = load ptr, ptr %2, align 8
// CHECK:STDOUT:   %4 = getelementptr inbounds nuw { ptr, ptr }, ptr %agg.tmp, i32 0, i32 1
// CHECK:STDOUT:   %5 = load ptr, ptr %4, align 8
// CHECK:STDOUT:   call void @_ZN11vector_likeC2ESt16initializer_listIiE(ptr noundef nonnull align 1 dereferenceable(1) %0, ptr %3, ptr %5)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN11vector_likeD2Ev(ptr noundef nonnull align 1 dereferenceable(1) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !114
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:DestroyImpl.2125e58e43ec3b1e.Core:DestroyImpl.Core"(ptr %self) #1 !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CInitNontrivialDtor.Main() #0 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca [16 x i8], align 8, !dbg !41
// CHECK:STDOUT:   %.loc23_69.19.temp = alloca [3 x [1 x i8]], align 1, !dbg !42
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !41
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc23_69.19.temp), !dbg !42
// CHECK:STDOUT:   %.loc23_69.20.array.index = getelementptr inbounds [3 x [1 x i8]], ptr %.loc23_69.19.temp, i32 0, i64 0, !dbg !42
// CHECK:STDOUT:   call void @_ZN15nontrivial_dtorC1Ev.carbon_thunk_tuple.(ptr %.loc23_69.20.array.index), !dbg !42
// CHECK:STDOUT:   %.loc23_69.18.array.index = getelementptr inbounds [3 x [1 x i8]], ptr %.loc23_69.19.temp, i32 0, i64 1, !dbg !42
// CHECK:STDOUT:   call void @_ZN15nontrivial_dtorC1Ev.carbon_thunk_tuple.(ptr %.loc23_69.18.array.index), !dbg !42
// CHECK:STDOUT:   %.loc23_69.17.array.index = getelementptr inbounds [3 x [1 x i8]], ptr %.loc23_69.19.temp, i32 0, i64 2, !dbg !42
// CHECK:STDOUT:   call void @_ZN15nontrivial_dtorC1Ev.carbon_thunk_tuple.(ptr %.loc23_69.17.array.index), !dbg !42
// CHECK:STDOUT:   %initializer_list.initializer_list.call.init_list.begin = getelementptr inbounds nuw [16 x i8], ptr %_.var, i32 0, i32 0, !dbg !41
// CHECK:STDOUT:   store ptr %.loc23_69.19.temp, ptr %initializer_list.initializer_list.call.init_list.begin, align 8, !dbg !41
// CHECK:STDOUT:   %initializer_list.initializer_list.call.init_list.end = getelementptr inbounds nuw [16 x i8], ptr %_.var, i32 0, i32 8, !dbg !41
// CHECK:STDOUT:   %initializer_list.initializer_list.call.array.end = getelementptr inbounds [3 x [1 x i8]], ptr %.loc23_69.19.temp, i32 1, !dbg !41
// CHECK:STDOUT:   store ptr %initializer_list.initializer_list.call.array.end, ptr %initializer_list.initializer_list.call.init_list.end, align 8, !dbg !41
// CHECK:STDOUT:   call void @_CWithinLifetime.Main(), !dbg !44
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.5b795aa496741041(ptr %.loc23_69.19.temp), !dbg !42
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c10e5f7cbdcbc55a(ptr %_.var), !dbg !41
// CHECK:STDOUT:   ret void, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN15nontrivial_dtorC1Ev.carbon_thunk_tuple.(ptr noundef %return) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !121
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !121
// CHECK:STDOUT:   call void @_ZN15nontrivial_dtorC2Ev(ptr noundef nonnull align 1 dereferenceable(1) %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN15nontrivial_dtorD2Ev(ptr noundef nonnull align 1 dereferenceable(1) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !121
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.880d8b86f475ef6d:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !46 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc23_69.1.array.index = getelementptr inbounds [3 x [1 x i8]], ptr %self, i32 0, i64 2, !dbg !48
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.58828228f275ec5f(ptr %.loc23_69.1.array.index), !dbg !48
// CHECK:STDOUT:   %.loc23_69.4.array.index = getelementptr inbounds [3 x [1 x i8]], ptr %self, i32 0, i64 1, !dbg !48
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.58828228f275ec5f(ptr %.loc23_69.4.array.index), !dbg !48
// CHECK:STDOUT:   %.loc23_69.7.array.index = getelementptr inbounds [3 x [1 x i8]], ptr %self, i32 0, i64 0, !dbg !48
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.58828228f275ec5f(ptr %.loc23_69.7.array.index), !dbg !48
// CHECK:STDOUT:   ret void, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:DestroyImpl.f5104a2e7734b530.Core:DestroyImpl.Core"(ptr %self) #1 !dbg !50 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:DestroyImpl.9e1f5645f4dcb35b.Core:DestroyImpl.Core"(ptr %self) #1 !dbg !54 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.8b2abdfe25f0b189(ptr %t) #0 !dbg !59 {
// CHECK:STDOUT:   call void @"_COp.b3a0adb9db7bb49e:core.SubobjectDestroy.Core"(ptr %t), !dbg !63
// CHECK:STDOUT:   ret void, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.c3847e2d14f33681(ptr %t) #0 !dbg !66 {
// CHECK:STDOUT:   call void @"_COp:thunk:DestroyImpl.e0350233115f3f94.Core:DestroyImpl.Core"(ptr %t), !dbg !69
// CHECK:STDOUT:   ret void, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.52dd027a4dd17420(ptr %t) #0 !dbg !73 {
// CHECK:STDOUT:   ret void, !dbg !78
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f8cb70ab0b13eb2c(ptr %t) #0 !dbg !80 {
// CHECK:STDOUT:   call void @"_COp:thunk:DestroyImpl.2125e58e43ec3b1e.Core:DestroyImpl.Core"(ptr %t), !dbg !83
// CHECK:STDOUT:   call void @_ZN11vector_likeD2Ev(ptr %t), !dbg !84
// CHECK:STDOUT:   ret void, !dbg !85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.5b795aa496741041(ptr %t) #0 !dbg !87 {
// CHECK:STDOUT:   call void @"_COp.880d8b86f475ef6d:core.SubobjectDestroy.Core"(ptr %t), !dbg !91
// CHECK:STDOUT:   ret void, !dbg !92
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.c10e5f7cbdcbc55a(ptr %t) #0 !dbg !94 {
// CHECK:STDOUT:   call void @"_COp:thunk:DestroyImpl.9e1f5645f4dcb35b.Core:DestroyImpl.Core"(ptr %t), !dbg !97
// CHECK:STDOUT:   ret void, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.58828228f275ec5f(ptr %t) #0 !dbg !101 {
// CHECK:STDOUT:   call void @"_COp:thunk:DestroyImpl.f5104a2e7734b530.Core:DestroyImpl.Core"(ptr %t), !dbg !104
// CHECK:STDOUT:   call void @_ZN15nontrivial_dtorD2Ev(ptr %t), !dbg !105
// CHECK:STDOUT:   ret void, !dbg !106
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN11vector_likeC2ESt16initializer_listIiE(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr %list.coerce0, ptr %list.coerce1) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %list = alloca %"class.std::initializer_list", align 8
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %0 = getelementptr inbounds nuw { ptr, ptr }, ptr %list, i32 0, i32 0
// CHECK:STDOUT:   store ptr %list.coerce0, ptr %0, align 8
// CHECK:STDOUT:   %1 = getelementptr inbounds nuw { ptr, ptr }, ptr %list, i32 0, i32 1
// CHECK:STDOUT:   store ptr %list.coerce1, ptr %1, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !114
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN15nontrivial_dtorC2Ev(ptr noundef nonnull align 1 dereferenceable(1) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !121
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_ZN15nontrivial_dtorC1Ev.carbon_thunk_tuple., { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.8b2abdfe25f0b189, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.c3847e2d14f33681, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.52dd027a4dd17420, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.58828228f275ec5f, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #2 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #4 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!122, !123, !124, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!126}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "use_pointer_pointer.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !{null}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !10 = !{null, !9}
// CHECK:STDOUT: !11 = !DISubroutineType(types: !10)
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "InitDirectly", linkageName: "_CInitDirectly.Main", scope: null, file: !4, line: 12, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !15 = !DILocation(line: 13, column: 3, scope: !14)
// CHECK:STDOUT: !16 = !DILocation(line: 13, column: 42, scope: !14)
// CHECK:STDOUT: !18 = !DILocation(line: 14, column: 3, scope: !14)
// CHECK:STDOUT: !19 = !DILocation(line: 12, column: 1, scope: !14)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "Op", linkageName: "_COp.b3a0adb9db7bb49e:core.SubobjectDestroy.Core", scope: null, file: !4, line: 13, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !25)
// CHECK:STDOUT: !21 = !DILocalVariable(arg: 1, scope: !20, type: !9)
// CHECK:STDOUT: !22 = !DILocation(line: 13, column: 42, scope: !20)
// CHECK:STDOUT: !23 = !{null, !6}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = !{!21}
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:DestroyImpl.e0350233115f3f94.Core:DestroyImpl.Core", scope: null, file: !4, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !29)
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !26, type: !9)
// CHECK:STDOUT: !28 = !DILocation(line: 0, scope: !26)
// CHECK:STDOUT: !29 = !{!27}
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "InitVectorLike", linkageName: "_CInitVectorLike.Main", scope: null, file: !4, line: 17, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !31 = !DILocation(line: 18, column: 3, scope: !30)
// CHECK:STDOUT: !32 = !DILocation(line: 18, column: 28, scope: !30)
// CHECK:STDOUT: !34 = !DILocation(line: 19, column: 3, scope: !30)
// CHECK:STDOUT: !35 = !DILocation(line: 17, column: 1, scope: !30)
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:DestroyImpl.2125e58e43ec3b1e.Core:DestroyImpl.Core", scope: null, file: !4, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !39)
// CHECK:STDOUT: !37 = !DILocalVariable(arg: 1, scope: !36, type: !9)
// CHECK:STDOUT: !38 = !DILocation(line: 0, scope: !36)
// CHECK:STDOUT: !39 = !{!37}
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "InitNontrivialDtor", linkageName: "_CInitNontrivialDtor.Main", scope: null, file: !4, line: 22, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !41 = !DILocation(line: 23, column: 3, scope: !40)
// CHECK:STDOUT: !42 = !DILocation(line: 23, column: 58, scope: !40)
// CHECK:STDOUT: !44 = !DILocation(line: 24, column: 3, scope: !40)
// CHECK:STDOUT: !45 = !DILocation(line: 22, column: 1, scope: !40)
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "Op", linkageName: "_COp.880d8b86f475ef6d:core.SubobjectDestroy.Core", scope: null, file: !4, line: 23, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !49)
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !46, type: !9)
// CHECK:STDOUT: !48 = !DILocation(line: 23, column: 58, scope: !46)
// CHECK:STDOUT: !49 = !{!47}
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:DestroyImpl.f5104a2e7734b530.Core:DestroyImpl.Core", scope: null, file: !4, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !53)
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !50, type: !9)
// CHECK:STDOUT: !52 = !DILocation(line: 0, scope: !50)
// CHECK:STDOUT: !53 = !{!51}
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:DestroyImpl.9e1f5645f4dcb35b.Core:DestroyImpl.Core", scope: null, file: !4, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !57)
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 1, scope: !54, type: !9)
// CHECK:STDOUT: !56 = !DILocation(line: 0, scope: !54)
// CHECK:STDOUT: !57 = !{!55}
// CHECK:STDOUT: !58 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.8b2abdfe25f0b189", scope: null, file: !58, line: 30, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !65)
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 1, scope: !59, type: !9)
// CHECK:STDOUT: !63 = !DILocation(line: 32, column: 5, scope: !59)
// CHECK:STDOUT: !64 = !DILocation(line: 30, column: 1, scope: !59)
// CHECK:STDOUT: !65 = !{!60}
// CHECK:STDOUT: !66 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.c3847e2d14f33681", scope: null, file: !58, line: 30, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !72)
// CHECK:STDOUT: !67 = !DILocalVariable(arg: 1, scope: !66, type: !9)
// CHECK:STDOUT: !69 = !DILocation(line: 31, column: 5, scope: !66)
// CHECK:STDOUT: !71 = !DILocation(line: 30, column: 1, scope: !66)
// CHECK:STDOUT: !72 = !{!67}
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.52dd027a4dd17420", scope: null, file: !58, line: 30, type: !24, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !79)
// CHECK:STDOUT: !74 = !DILocalVariable(arg: 1, scope: !73, type: !6)
// CHECK:STDOUT: !78 = !DILocation(line: 30, column: 1, scope: !73)
// CHECK:STDOUT: !79 = !{!74}
// CHECK:STDOUT: !80 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f8cb70ab0b13eb2c", scope: null, file: !58, line: 30, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !86)
// CHECK:STDOUT: !81 = !DILocalVariable(arg: 1, scope: !80, type: !9)
// CHECK:STDOUT: !83 = !DILocation(line: 31, column: 5, scope: !80)
// CHECK:STDOUT: !84 = !DILocation(line: 32, column: 5, scope: !80)
// CHECK:STDOUT: !85 = !DILocation(line: 30, column: 1, scope: !80)
// CHECK:STDOUT: !86 = !{!81}
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.5b795aa496741041", scope: null, file: !58, line: 30, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !93)
// CHECK:STDOUT: !88 = !DILocalVariable(arg: 1, scope: !87, type: !9)
// CHECK:STDOUT: !91 = !DILocation(line: 32, column: 5, scope: !87)
// CHECK:STDOUT: !92 = !DILocation(line: 30, column: 1, scope: !87)
// CHECK:STDOUT: !93 = !{!88}
// CHECK:STDOUT: !94 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.c10e5f7cbdcbc55a", scope: null, file: !58, line: 30, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !100)
// CHECK:STDOUT: !95 = !DILocalVariable(arg: 1, scope: !94, type: !9)
// CHECK:STDOUT: !97 = !DILocation(line: 31, column: 5, scope: !94)
// CHECK:STDOUT: !99 = !DILocation(line: 30, column: 1, scope: !94)
// CHECK:STDOUT: !100 = !{!95}
// CHECK:STDOUT: !101 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.58828228f275ec5f", scope: null, file: !58, line: 30, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !107)
// CHECK:STDOUT: !102 = !DILocalVariable(arg: 1, scope: !101, type: !9)
// CHECK:STDOUT: !104 = !DILocation(line: 31, column: 5, scope: !101)
// CHECK:STDOUT: !105 = !DILocation(line: 32, column: 5, scope: !101)
// CHECK:STDOUT: !106 = !DILocation(line: 30, column: 1, scope: !101)
// CHECK:STDOUT: !107 = !{!102}
// CHECK:STDOUT: !108 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !109 = !{!"omnipotent char", !108, i64 0}
// CHECK:STDOUT: !110 = !{!"any pointer", !109, i64 0}
// CHECK:STDOUT: !111 = !{!"p1 _ZTSSt16initializer_listIiE", !110, i64 0}
// CHECK:STDOUT: !112 = !{!111, !111, i64 0}
// CHECK:STDOUT: !113 = !{!"p1 _ZTS11vector_like", !110, i64 0}
// CHECK:STDOUT: !114 = !{!113, !113, i64 0}
// CHECK:STDOUT: !115 = !{!"int", !109, i64 0}
// CHECK:STDOUT: !116 = !{!"p1 int", !110, i64 0}
// CHECK:STDOUT: !118 = !{!116, !116, i64 0}
// CHECK:STDOUT: !119 = !{i64 0, i64 8, !118, i64 8, i64 8, !118}
// CHECK:STDOUT: !120 = !{!"p1 _ZTS15nontrivial_dtor", !110, i64 0}
// CHECK:STDOUT: !121 = !{!120, !120, i64 0}
// CHECK:STDOUT: !122 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !123 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !124 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !125 = !{!"__libc_errno", !115, i64 0}
// CHECK:STDOUT: !126 = !{!125, !115, i64 0}
// CHECK:STDOUT:
