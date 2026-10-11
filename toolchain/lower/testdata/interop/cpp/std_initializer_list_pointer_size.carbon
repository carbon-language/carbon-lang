// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/std_initializer_list_pointer_size.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/std_initializer_list_pointer_size.carbon

// --- initializer_list_pointer_size.h

namespace std {
  using size_t = decltype(sizeof(0));

  template <typename T> class initializer_list {
   public:
    initializer_list() {}

   private:
    T* begin_ = nullptr;
    size_t size_ = 0;
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

// --- use_pointer_size.carbon

library "[[@TEST_NAME]]";

import Cpp library "initializer_list_pointer_size.h";
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
// CHECK:STDOUT: ; ModuleID = 'use_pointer_size.carbon'
// CHECK:STDOUT: source_filename = "use_pointer_size.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %"class.std::initializer_list" = type { ptr, i64 }
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
// CHECK:STDOUT: define void @_CInitDirectly.Main() #0 !dbg !15 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca [16 x i8], align 8, !dbg !16
// CHECK:STDOUT:   %.loc13_50.3.temp = alloca [3 x i32], align 4, !dbg !17
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !16
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc13_50.3.temp), !dbg !17
// CHECK:STDOUT:   %.loc13_50.4.array.index = getelementptr inbounds [3 x i32], ptr %.loc13_50.3.temp, i32 0, i64 0, !dbg !17
// CHECK:STDOUT:   %.loc13_50.7.array.index = getelementptr inbounds [3 x i32], ptr %.loc13_50.3.temp, i32 0, i64 1, !dbg !17
// CHECK:STDOUT:   %.loc13_50.10.array.index = getelementptr inbounds [3 x i32], ptr %.loc13_50.3.temp, i32 0, i64 2, !dbg !17
// CHECK:STDOUT:   %initializer_list.initializer_list.call.init_list.begin = getelementptr inbounds nuw [16 x i8], ptr %_.var, i32 0, i32 0, !dbg !16
// CHECK:STDOUT:   store ptr @array.loc13_50.15, ptr %initializer_list.initializer_list.call.init_list.begin, align 8, !dbg !16
// CHECK:STDOUT:   %initializer_list.initializer_list.call.init_list.size = getelementptr inbounds nuw [16 x i8], ptr %_.var, i32 0, i32 8, !dbg !16
// CHECK:STDOUT:   store i64 3, ptr %initializer_list.initializer_list.call.init_list.size, align 8, !dbg !16
// CHECK:STDOUT:   call void @_CWithinLifetime.Main(), !dbg !19
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.8b2abdfe25f0b189(ptr @array), !dbg !17
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c3847e2d14f33681(ptr %_.var), !dbg !16
// CHECK:STDOUT:   ret void, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.b3a0adb9db7bb49e:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc13_50.1.array.index = getelementptr inbounds [3 x i32], ptr %self, i32 0, i64 2, !dbg !23
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.52dd027a4dd17420(ptr %.loc13_50.1.array.index), !dbg !23
// CHECK:STDOUT:   %.loc13_50.4.array.index = getelementptr inbounds [3 x i32], ptr %self, i32 0, i64 1, !dbg !23
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.52dd027a4dd17420(ptr %.loc13_50.4.array.index), !dbg !23
// CHECK:STDOUT:   %.loc13_50.7.array.index = getelementptr inbounds [3 x i32], ptr %self, i32 0, i64 0, !dbg !23
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.52dd027a4dd17420(ptr %.loc13_50.7.array.index), !dbg !23
// CHECK:STDOUT:   ret void, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:DestroyImpl.e0350233115f3f94.Core:DestroyImpl.Core"(ptr %self) #1 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CInitVectorLike.Main() #0 !dbg !31 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca [1 x i8], align 1, !dbg !32
// CHECK:STDOUT:   %.loc18_36.3.temp = alloca [16 x i8], align 8, !dbg !33
// CHECK:STDOUT:   %.loc18_36.5.temp = alloca [3 x i32], align 4, !dbg !33
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !32
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_36.3.temp), !dbg !33
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_36.5.temp), !dbg !33
// CHECK:STDOUT:   %.loc18_36.6.array.index = getelementptr inbounds [3 x i32], ptr %.loc18_36.5.temp, i32 0, i64 0, !dbg !33
// CHECK:STDOUT:   %.loc18_36.9.array.index = getelementptr inbounds [3 x i32], ptr %.loc18_36.5.temp, i32 0, i64 1, !dbg !33
// CHECK:STDOUT:   %.loc18_36.12.array.index = getelementptr inbounds [3 x i32], ptr %.loc18_36.5.temp, i32 0, i64 2, !dbg !33
// CHECK:STDOUT:   %initializer_list.initializer_list.call.init_list.begin = getelementptr inbounds nuw [16 x i8], ptr %.loc18_36.3.temp, i32 0, i32 0, !dbg !33
// CHECK:STDOUT:   store ptr @array.loc13_50.15, ptr %initializer_list.initializer_list.call.init_list.begin, align 8, !dbg !33
// CHECK:STDOUT:   %initializer_list.initializer_list.call.init_list.size = getelementptr inbounds nuw [16 x i8], ptr %.loc18_36.3.temp, i32 0, i32 8, !dbg !33
// CHECK:STDOUT:   store i64 3, ptr %initializer_list.initializer_list.call.init_list.size, align 8, !dbg !33
// CHECK:STDOUT:   call void @_ZN11vector_likeC1ESt16initializer_listIiE.carbon_thunk._(ptr %.loc18_36.3.temp, ptr %_.var), !dbg !32
// CHECK:STDOUT:   call void @_CWithinLifetime.Main(), !dbg !35
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c3847e2d14f33681(ptr %.loc18_36.3.temp), !dbg !33
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.8b2abdfe25f0b189(ptr @array), !dbg !33
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f8cb70ab0b13eb2c(ptr %_.var), !dbg !32
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN11vector_likeC1ESt16initializer_listIiE.carbon_thunk._(ptr noundef %list, ptr noundef %return) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %list.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %"class.std::initializer_list", align 8
// CHECK:STDOUT:   store ptr %list, ptr %list.addr, align 8, !tbaa !113
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !115
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !115
// CHECK:STDOUT:   %1 = load ptr, ptr %list.addr, align 8, !tbaa !113
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %agg.tmp, ptr align 8 %1, i64 16, i1 false), !tbaa.struct !122
// CHECK:STDOUT:   %2 = getelementptr inbounds nuw { ptr, i64 }, ptr %agg.tmp, i32 0, i32 0
// CHECK:STDOUT:   %3 = load ptr, ptr %2, align 8
// CHECK:STDOUT:   %4 = getelementptr inbounds nuw { ptr, i64 }, ptr %agg.tmp, i32 0, i32 1
// CHECK:STDOUT:   %5 = load i64, ptr %4, align 8
// CHECK:STDOUT:   call void @_ZN11vector_likeC2ESt16initializer_listIiE(ptr noundef nonnull align 1 dereferenceable(1) %0, ptr %3, i64 %5)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN11vector_likeD2Ev(ptr noundef nonnull align 1 dereferenceable(1) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !115
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:DestroyImpl.2125e58e43ec3b1e.Core:DestroyImpl.Core"(ptr %self) #1 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CInitNontrivialDtor.Main() #0 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca [16 x i8], align 8, !dbg !42
// CHECK:STDOUT:   %.loc23_69.19.temp = alloca [3 x [1 x i8]], align 1, !dbg !43
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !42
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc23_69.19.temp), !dbg !43
// CHECK:STDOUT:   %.loc23_69.20.array.index = getelementptr inbounds [3 x [1 x i8]], ptr %.loc23_69.19.temp, i32 0, i64 0, !dbg !43
// CHECK:STDOUT:   call void @_ZN15nontrivial_dtorC1Ev.carbon_thunk_tuple.(ptr %.loc23_69.20.array.index), !dbg !43
// CHECK:STDOUT:   %.loc23_69.18.array.index = getelementptr inbounds [3 x [1 x i8]], ptr %.loc23_69.19.temp, i32 0, i64 1, !dbg !43
// CHECK:STDOUT:   call void @_ZN15nontrivial_dtorC1Ev.carbon_thunk_tuple.(ptr %.loc23_69.18.array.index), !dbg !43
// CHECK:STDOUT:   %.loc23_69.17.array.index = getelementptr inbounds [3 x [1 x i8]], ptr %.loc23_69.19.temp, i32 0, i64 2, !dbg !43
// CHECK:STDOUT:   call void @_ZN15nontrivial_dtorC1Ev.carbon_thunk_tuple.(ptr %.loc23_69.17.array.index), !dbg !43
// CHECK:STDOUT:   %initializer_list.initializer_list.call.init_list.begin = getelementptr inbounds nuw [16 x i8], ptr %_.var, i32 0, i32 0, !dbg !42
// CHECK:STDOUT:   store ptr %.loc23_69.19.temp, ptr %initializer_list.initializer_list.call.init_list.begin, align 8, !dbg !42
// CHECK:STDOUT:   %initializer_list.initializer_list.call.init_list.size = getelementptr inbounds nuw [16 x i8], ptr %_.var, i32 0, i32 8, !dbg !42
// CHECK:STDOUT:   store i64 3, ptr %initializer_list.initializer_list.call.init_list.size, align 8, !dbg !42
// CHECK:STDOUT:   call void @_CWithinLifetime.Main(), !dbg !45
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.5b795aa496741041(ptr %.loc23_69.19.temp), !dbg !43
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c10e5f7cbdcbc55a(ptr %_.var), !dbg !42
// CHECK:STDOUT:   ret void, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN15nontrivial_dtorC1Ev.carbon_thunk_tuple.(ptr noundef %return) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !124
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !124
// CHECK:STDOUT:   call void @_ZN15nontrivial_dtorC2Ev(ptr noundef nonnull align 1 dereferenceable(1) %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN15nontrivial_dtorD2Ev(ptr noundef nonnull align 1 dereferenceable(1) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !124
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.880d8b86f475ef6d:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !47 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc23_69.1.array.index = getelementptr inbounds [3 x [1 x i8]], ptr %self, i32 0, i64 2, !dbg !49
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.58828228f275ec5f(ptr %.loc23_69.1.array.index), !dbg !49
// CHECK:STDOUT:   %.loc23_69.4.array.index = getelementptr inbounds [3 x [1 x i8]], ptr %self, i32 0, i64 1, !dbg !49
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.58828228f275ec5f(ptr %.loc23_69.4.array.index), !dbg !49
// CHECK:STDOUT:   %.loc23_69.7.array.index = getelementptr inbounds [3 x [1 x i8]], ptr %self, i32 0, i64 0, !dbg !49
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.58828228f275ec5f(ptr %.loc23_69.7.array.index), !dbg !49
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:DestroyImpl.f5104a2e7734b530.Core:DestroyImpl.Core"(ptr %self) #1 !dbg !51 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:DestroyImpl.9e1f5645f4dcb35b.Core:DestroyImpl.Core"(ptr %self) #1 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.8b2abdfe25f0b189(ptr %t) #0 !dbg !60 {
// CHECK:STDOUT:   call void @"_COp.b3a0adb9db7bb49e:core.SubobjectDestroy.Core"(ptr %t), !dbg !64
// CHECK:STDOUT:   ret void, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.c3847e2d14f33681(ptr %t) #0 !dbg !67 {
// CHECK:STDOUT:   call void @"_COp:thunk:DestroyImpl.e0350233115f3f94.Core:DestroyImpl.Core"(ptr %t), !dbg !70
// CHECK:STDOUT:   ret void, !dbg !72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.52dd027a4dd17420(ptr %t) #0 !dbg !74 {
// CHECK:STDOUT:   ret void, !dbg !79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f8cb70ab0b13eb2c(ptr %t) #0 !dbg !81 {
// CHECK:STDOUT:   call void @"_COp:thunk:DestroyImpl.2125e58e43ec3b1e.Core:DestroyImpl.Core"(ptr %t), !dbg !84
// CHECK:STDOUT:   call void @_ZN11vector_likeD2Ev(ptr %t), !dbg !85
// CHECK:STDOUT:   ret void, !dbg !86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.5b795aa496741041(ptr %t) #0 !dbg !88 {
// CHECK:STDOUT:   call void @"_COp.880d8b86f475ef6d:core.SubobjectDestroy.Core"(ptr %t), !dbg !92
// CHECK:STDOUT:   ret void, !dbg !93
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.c10e5f7cbdcbc55a(ptr %t) #0 !dbg !95 {
// CHECK:STDOUT:   call void @"_COp:thunk:DestroyImpl.9e1f5645f4dcb35b.Core:DestroyImpl.Core"(ptr %t), !dbg !98
// CHECK:STDOUT:   ret void, !dbg !100
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.58828228f275ec5f(ptr %t) #0 !dbg !102 {
// CHECK:STDOUT:   call void @"_COp:thunk:DestroyImpl.f5104a2e7734b530.Core:DestroyImpl.Core"(ptr %t), !dbg !105
// CHECK:STDOUT:   call void @_ZN15nontrivial_dtorD2Ev(ptr %t), !dbg !106
// CHECK:STDOUT:   ret void, !dbg !107
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN11vector_likeC2ESt16initializer_listIiE(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr %list.coerce0, i64 %list.coerce1) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %list = alloca %"class.std::initializer_list", align 8
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %0 = getelementptr inbounds nuw { ptr, i64 }, ptr %list, i32 0, i32 0
// CHECK:STDOUT:   store ptr %list.coerce0, ptr %0, align 8
// CHECK:STDOUT:   %1 = getelementptr inbounds nuw { ptr, i64 }, ptr %list, i32 0, i32 1
// CHECK:STDOUT:   store i64 %list.coerce1, ptr %1, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !115
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN15nontrivial_dtorC2Ev(ptr noundef nonnull align 1 dereferenceable(1) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !124
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
// CHECK:STDOUT: !llvm.module.flags = !{!125, !126, !127, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!129}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "use_pointer_size.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !{null}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !11 = !{null, !10}
// CHECK:STDOUT: !12 = !DISubroutineType(types: !11)
// CHECK:STDOUT: !15 = distinct !DISubprogram(name: "InitDirectly", linkageName: "_CInitDirectly.Main", scope: null, file: !4, line: 12, type: !9, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !16 = !DILocation(line: 13, column: 3, scope: !15)
// CHECK:STDOUT: !17 = !DILocation(line: 13, column: 42, scope: !15)
// CHECK:STDOUT: !19 = !DILocation(line: 14, column: 3, scope: !15)
// CHECK:STDOUT: !20 = !DILocation(line: 12, column: 1, scope: !15)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "Op", linkageName: "_COp.b3a0adb9db7bb49e:core.SubobjectDestroy.Core", scope: null, file: !4, line: 13, type: !12, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !26)
// CHECK:STDOUT: !22 = !DILocalVariable(arg: 1, scope: !21, type: !10)
// CHECK:STDOUT: !23 = !DILocation(line: 13, column: 42, scope: !21)
// CHECK:STDOUT: !24 = !{null, !6}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{!22}
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:DestroyImpl.e0350233115f3f94.Core:DestroyImpl.Core", scope: null, file: !4, type: !12, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !30)
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 1, scope: !27, type: !10)
// CHECK:STDOUT: !29 = !DILocation(line: 0, scope: !27)
// CHECK:STDOUT: !30 = !{!28}
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "InitVectorLike", linkageName: "_CInitVectorLike.Main", scope: null, file: !4, line: 17, type: !9, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !32 = !DILocation(line: 18, column: 3, scope: !31)
// CHECK:STDOUT: !33 = !DILocation(line: 18, column: 28, scope: !31)
// CHECK:STDOUT: !35 = !DILocation(line: 19, column: 3, scope: !31)
// CHECK:STDOUT: !36 = !DILocation(line: 17, column: 1, scope: !31)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:DestroyImpl.2125e58e43ec3b1e.Core:DestroyImpl.Core", scope: null, file: !4, type: !12, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !40)
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !37, type: !10)
// CHECK:STDOUT: !39 = !DILocation(line: 0, scope: !37)
// CHECK:STDOUT: !40 = !{!38}
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "InitNontrivialDtor", linkageName: "_CInitNontrivialDtor.Main", scope: null, file: !4, line: 22, type: !9, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !42 = !DILocation(line: 23, column: 3, scope: !41)
// CHECK:STDOUT: !43 = !DILocation(line: 23, column: 58, scope: !41)
// CHECK:STDOUT: !45 = !DILocation(line: 24, column: 3, scope: !41)
// CHECK:STDOUT: !46 = !DILocation(line: 22, column: 1, scope: !41)
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "Op", linkageName: "_COp.880d8b86f475ef6d:core.SubobjectDestroy.Core", scope: null, file: !4, line: 23, type: !12, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !50)
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 1, scope: !47, type: !10)
// CHECK:STDOUT: !49 = !DILocation(line: 23, column: 58, scope: !47)
// CHECK:STDOUT: !50 = !{!48}
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:DestroyImpl.f5104a2e7734b530.Core:DestroyImpl.Core", scope: null, file: !4, type: !12, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !54)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 1, scope: !51, type: !10)
// CHECK:STDOUT: !53 = !DILocation(line: 0, scope: !51)
// CHECK:STDOUT: !54 = !{!52}
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:DestroyImpl.9e1f5645f4dcb35b.Core:DestroyImpl.Core", scope: null, file: !4, type: !12, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !58)
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !55, type: !10)
// CHECK:STDOUT: !57 = !DILocation(line: 0, scope: !55)
// CHECK:STDOUT: !58 = !{!56}
// CHECK:STDOUT: !59 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.8b2abdfe25f0b189", scope: null, file: !59, line: 30, type: !12, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !66)
// CHECK:STDOUT: !61 = !DILocalVariable(arg: 1, scope: !60, type: !10)
// CHECK:STDOUT: !64 = !DILocation(line: 32, column: 5, scope: !60)
// CHECK:STDOUT: !65 = !DILocation(line: 30, column: 1, scope: !60)
// CHECK:STDOUT: !66 = !{!61}
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.c3847e2d14f33681", scope: null, file: !59, line: 30, type: !12, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !73)
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 1, scope: !67, type: !10)
// CHECK:STDOUT: !70 = !DILocation(line: 31, column: 5, scope: !67)
// CHECK:STDOUT: !72 = !DILocation(line: 30, column: 1, scope: !67)
// CHECK:STDOUT: !73 = !{!68}
// CHECK:STDOUT: !74 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.52dd027a4dd17420", scope: null, file: !59, line: 30, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !80)
// CHECK:STDOUT: !75 = !DILocalVariable(arg: 1, scope: !74, type: !6)
// CHECK:STDOUT: !79 = !DILocation(line: 30, column: 1, scope: !74)
// CHECK:STDOUT: !80 = !{!75}
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f8cb70ab0b13eb2c", scope: null, file: !59, line: 30, type: !12, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !87)
// CHECK:STDOUT: !82 = !DILocalVariable(arg: 1, scope: !81, type: !10)
// CHECK:STDOUT: !84 = !DILocation(line: 31, column: 5, scope: !81)
// CHECK:STDOUT: !85 = !DILocation(line: 32, column: 5, scope: !81)
// CHECK:STDOUT: !86 = !DILocation(line: 30, column: 1, scope: !81)
// CHECK:STDOUT: !87 = !{!82}
// CHECK:STDOUT: !88 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.5b795aa496741041", scope: null, file: !59, line: 30, type: !12, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !94)
// CHECK:STDOUT: !89 = !DILocalVariable(arg: 1, scope: !88, type: !10)
// CHECK:STDOUT: !92 = !DILocation(line: 32, column: 5, scope: !88)
// CHECK:STDOUT: !93 = !DILocation(line: 30, column: 1, scope: !88)
// CHECK:STDOUT: !94 = !{!89}
// CHECK:STDOUT: !95 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.c10e5f7cbdcbc55a", scope: null, file: !59, line: 30, type: !12, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !101)
// CHECK:STDOUT: !96 = !DILocalVariable(arg: 1, scope: !95, type: !10)
// CHECK:STDOUT: !98 = !DILocation(line: 31, column: 5, scope: !95)
// CHECK:STDOUT: !100 = !DILocation(line: 30, column: 1, scope: !95)
// CHECK:STDOUT: !101 = !{!96}
// CHECK:STDOUT: !102 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.58828228f275ec5f", scope: null, file: !59, line: 30, type: !12, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !108)
// CHECK:STDOUT: !103 = !DILocalVariable(arg: 1, scope: !102, type: !10)
// CHECK:STDOUT: !105 = !DILocation(line: 31, column: 5, scope: !102)
// CHECK:STDOUT: !106 = !DILocation(line: 32, column: 5, scope: !102)
// CHECK:STDOUT: !107 = !DILocation(line: 30, column: 1, scope: !102)
// CHECK:STDOUT: !108 = !{!103}
// CHECK:STDOUT: !109 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !110 = !{!"omnipotent char", !109, i64 0}
// CHECK:STDOUT: !111 = !{!"any pointer", !110, i64 0}
// CHECK:STDOUT: !112 = !{!"p1 _ZTSSt16initializer_listIiE", !111, i64 0}
// CHECK:STDOUT: !113 = !{!112, !112, i64 0}
// CHECK:STDOUT: !114 = !{!"p1 _ZTS11vector_like", !111, i64 0}
// CHECK:STDOUT: !115 = !{!114, !114, i64 0}
// CHECK:STDOUT: !116 = !{!"int", !110, i64 0}
// CHECK:STDOUT: !117 = !{!"p1 int", !111, i64 0}
// CHECK:STDOUT: !118 = !{!"long", !110, i64 0}
// CHECK:STDOUT: !120 = !{!117, !117, i64 0}
// CHECK:STDOUT: !121 = !{!118, !118, i64 0}
// CHECK:STDOUT: !122 = !{i64 0, i64 8, !120, i64 8, i64 8, !121}
// CHECK:STDOUT: !123 = !{!"p1 _ZTS15nontrivial_dtor", !111, i64 0}
// CHECK:STDOUT: !124 = !{!123, !123, i64 0}
// CHECK:STDOUT: !125 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !126 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !127 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !128 = !{!"__libc_errno", !116, i64 0}
// CHECK:STDOUT: !129 = !{!128, !116, i64 0}
// CHECK:STDOUT:
