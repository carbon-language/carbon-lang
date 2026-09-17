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
// CHECK:STDOUT: define void @_CInitDirectly.Main() #0 !dbg !17 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca [16 x i8], align 8, !dbg !18
// CHECK:STDOUT:   %.loc13_50.3.temp = alloca [3 x i32], align 4, !dbg !19
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !18
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc13_50.3.temp), !dbg !19
// CHECK:STDOUT:   %.loc13_50.4.array.index = getelementptr inbounds [3 x i32], ptr %.loc13_50.3.temp, i32 0, i64 0, !dbg !19
// CHECK:STDOUT:   %.loc13_50.7.array.index = getelementptr inbounds [3 x i32], ptr %.loc13_50.3.temp, i32 0, i64 1, !dbg !19
// CHECK:STDOUT:   %.loc13_50.10.array.index = getelementptr inbounds [3 x i32], ptr %.loc13_50.3.temp, i32 0, i64 2, !dbg !19
// CHECK:STDOUT:   %initializer_list.initializer_list.call.init_list.begin = getelementptr inbounds nuw [16 x i8], ptr %_.var, i32 0, i32 0, !dbg !18
// CHECK:STDOUT:   store ptr @array.loc13_50.15, ptr %initializer_list.initializer_list.call.init_list.begin, align 8, !dbg !18
// CHECK:STDOUT:   %initializer_list.initializer_list.call.init_list.size = getelementptr inbounds nuw [16 x i8], ptr %_.var, i32 0, i32 8, !dbg !18
// CHECK:STDOUT:   store i64 3, ptr %initializer_list.initializer_list.call.init_list.size, align 8, !dbg !18
// CHECK:STDOUT:   call void @_CWithinLifetime.Main(), !dbg !21
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.f699181ca81df557(ptr @array), !dbg !19
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.38b139424e9e705f(ptr %_.var), !dbg !18
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.dc7dedd104b837b2:core.Destroy.Core"(ptr %self) #0 !dbg !27 {
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
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.38b139424e9e705f(ptr %.loc18_36.3.temp), !dbg !33
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.f699181ca81df557(ptr @array), !dbg !33
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.ea51de33ab6382d8(ptr %_.var), !dbg !32
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN11vector_likeC1ESt16initializer_listIiE.carbon_thunk._(ptr noundef %list, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %list.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %"class.std::initializer_list", align 8
// CHECK:STDOUT:   store ptr %list, ptr %list.addr, align 8, !tbaa !87
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !89
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !89
// CHECK:STDOUT:   %1 = load ptr, ptr %list.addr, align 8, !tbaa !87
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %agg.tmp, ptr align 8 %1, i64 16, i1 false), !tbaa.struct !96
// CHECK:STDOUT:   %2 = getelementptr inbounds nuw { ptr, i64 }, ptr %agg.tmp, i32 0, i32 0
// CHECK:STDOUT:   %3 = load ptr, ptr %2, align 8
// CHECK:STDOUT:   %4 = getelementptr inbounds nuw { ptr, i64 }, ptr %agg.tmp, i32 0, i32 1
// CHECK:STDOUT:   %5 = load i64, ptr %4, align 8
// CHECK:STDOUT:   call void @_ZN11vector_likeC2ESt16initializer_listIiE(ptr noundef nonnull align 1 dereferenceable(1) %0, ptr %3, i64 %5)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN11vector_likeD2Ev(ptr noundef nonnull align 1 dereferenceable(1) %this) unnamed_addr #2 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !89
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CInitNontrivialDtor.Main() #0 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca [16 x i8], align 8, !dbg !38
// CHECK:STDOUT:   %.loc23_69.19.temp = alloca [3 x [1 x i8]], align 1, !dbg !39
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !38
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc23_69.19.temp), !dbg !39
// CHECK:STDOUT:   %.loc23_69.20.array.index = getelementptr inbounds [3 x [1 x i8]], ptr %.loc23_69.19.temp, i32 0, i64 0, !dbg !39
// CHECK:STDOUT:   call void @_ZN15nontrivial_dtorC1Ev.carbon_thunk_tuple.(ptr %.loc23_69.20.array.index), !dbg !39
// CHECK:STDOUT:   %.loc23_69.18.array.index = getelementptr inbounds [3 x [1 x i8]], ptr %.loc23_69.19.temp, i32 0, i64 1, !dbg !39
// CHECK:STDOUT:   call void @_ZN15nontrivial_dtorC1Ev.carbon_thunk_tuple.(ptr %.loc23_69.18.array.index), !dbg !39
// CHECK:STDOUT:   %.loc23_69.17.array.index = getelementptr inbounds [3 x [1 x i8]], ptr %.loc23_69.19.temp, i32 0, i64 2, !dbg !39
// CHECK:STDOUT:   call void @_ZN15nontrivial_dtorC1Ev.carbon_thunk_tuple.(ptr %.loc23_69.17.array.index), !dbg !39
// CHECK:STDOUT:   %initializer_list.initializer_list.call.init_list.begin = getelementptr inbounds nuw [16 x i8], ptr %_.var, i32 0, i32 0, !dbg !38
// CHECK:STDOUT:   store ptr %.loc23_69.19.temp, ptr %initializer_list.initializer_list.call.init_list.begin, align 8, !dbg !38
// CHECK:STDOUT:   %initializer_list.initializer_list.call.init_list.size = getelementptr inbounds nuw [16 x i8], ptr %_.var, i32 0, i32 8, !dbg !38
// CHECK:STDOUT:   store i64 3, ptr %initializer_list.initializer_list.call.init_list.size, align 8, !dbg !38
// CHECK:STDOUT:   call void @_CWithinLifetime.Main(), !dbg !41
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.4440947f5319481f(ptr %.loc23_69.19.temp), !dbg !39
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.33bf5c55a0b6e3f1(ptr %_.var), !dbg !38
// CHECK:STDOUT:   ret void, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN15nontrivial_dtorC1Ev.carbon_thunk_tuple.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !98
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !98
// CHECK:STDOUT:   call void @_ZN15nontrivial_dtorC2Ev(ptr noundef nonnull align 1 dereferenceable(1) %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN15nontrivial_dtorD2Ev(ptr noundef nonnull align 1 dereferenceable(1) %this) unnamed_addr #2 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !98
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.71d86972d5684aff:core.Destroy.Core"(ptr %self) #0 !dbg !43 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.f699181ca81df557(ptr %self) #0 !dbg !48 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.dc7dedd104b837b2:core.Destroy.Core"(ptr %self), !dbg !52
// CHECK:STDOUT:   ret void, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.38b139424e9e705f(ptr %self) #0 !dbg !55 {
// CHECK:STDOUT:   ret void, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.ea51de33ab6382d8(ptr %self) #0 !dbg !62 {
// CHECK:STDOUT:   call void @_ZN11vector_likeD2Ev(ptr %self), !dbg !66
// CHECK:STDOUT:   ret void, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.4440947f5319481f(ptr %self) #0 !dbg !69 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.71d86972d5684aff:core.Destroy.Core"(ptr %self), !dbg !73
// CHECK:STDOUT:   ret void, !dbg !74
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.33bf5c55a0b6e3f1(ptr %self) #0 !dbg !76 {
// CHECK:STDOUT:   ret void, !dbg !81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN11vector_likeC2ESt16initializer_listIiE(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr %list.coerce0, i64 %list.coerce1) unnamed_addr #2 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %list = alloca %"class.std::initializer_list", align 8
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %0 = getelementptr inbounds nuw { ptr, i64 }, ptr %list, i32 0, i32 0
// CHECK:STDOUT:   store ptr %list.coerce0, ptr %0, align 8
// CHECK:STDOUT:   %1 = getelementptr inbounds nuw { ptr, i64 }, ptr %list, i32 0, i32 1
// CHECK:STDOUT:   store i64 %list.coerce1, ptr %1, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !89
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN15nontrivial_dtorC2Ev(ptr noundef nonnull align 1 dereferenceable(1) %this) unnamed_addr #2 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !98
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_ZN15nontrivial_dtorC1Ev.carbon_thunk_tuple., { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.f699181ca81df557, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.38b139424e9e705f, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!99, !100, !101, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!103}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "use_pointer_size.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !{null}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !10 = !{null, !6}
// CHECK:STDOUT: !11 = !DISubroutineType(types: !10)
// CHECK:STDOUT: !12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !13 = !{null, !12}
// CHECK:STDOUT: !14 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !17 = distinct !DISubprogram(name: "InitDirectly", linkageName: "_CInitDirectly.Main", scope: null, file: !4, line: 12, type: !9, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !18 = !DILocation(line: 13, column: 3, scope: !17)
// CHECK:STDOUT: !19 = !DILocation(line: 13, column: 42, scope: !17)
// CHECK:STDOUT: !21 = !DILocation(line: 14, column: 3, scope: !17)
// CHECK:STDOUT: !22 = !DILocation(line: 12, column: 1, scope: !17)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !4, line: 13, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !26)
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !23, type: !6)
// CHECK:STDOUT: !25 = !DILocation(line: 13, column: 42, scope: !23)
// CHECK:STDOUT: !26 = !{!24}
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.dc7dedd104b837b2:core.Destroy.Core", scope: null, file: !4, line: 13, type: !14, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !30)
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 1, scope: !27, type: !12)
// CHECK:STDOUT: !29 = !DILocation(line: 13, column: 42, scope: !27)
// CHECK:STDOUT: !30 = !{!28}
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "InitVectorLike", linkageName: "_CInitVectorLike.Main", scope: null, file: !4, line: 17, type: !9, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !32 = !DILocation(line: 18, column: 3, scope: !31)
// CHECK:STDOUT: !33 = !DILocation(line: 18, column: 28, scope: !31)
// CHECK:STDOUT: !35 = !DILocation(line: 19, column: 3, scope: !31)
// CHECK:STDOUT: !36 = !DILocation(line: 17, column: 1, scope: !31)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "InitNontrivialDtor", linkageName: "_CInitNontrivialDtor.Main", scope: null, file: !4, line: 22, type: !9, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !38 = !DILocation(line: 23, column: 3, scope: !37)
// CHECK:STDOUT: !39 = !DILocation(line: 23, column: 58, scope: !37)
// CHECK:STDOUT: !41 = !DILocation(line: 24, column: 3, scope: !37)
// CHECK:STDOUT: !42 = !DILocation(line: 22, column: 1, scope: !37)
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.71d86972d5684aff:core.Destroy.Core", scope: null, file: !4, line: 23, type: !14, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !46)
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 1, scope: !43, type: !12)
// CHECK:STDOUT: !45 = !DILocation(line: 23, column: 58, scope: !43)
// CHECK:STDOUT: !46 = !{!44}
// CHECK:STDOUT: !47 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.f699181ca81df557", scope: null, file: !47, line: 26, type: !14, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !54)
// CHECK:STDOUT: !49 = !DILocalVariable(arg: 1, scope: !48, type: !12)
// CHECK:STDOUT: !52 = !DILocation(line: 28, column: 5, scope: !48)
// CHECK:STDOUT: !53 = !DILocation(line: 26, column: 3, scope: !48)
// CHECK:STDOUT: !54 = !{!49}
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.38b139424e9e705f", scope: null, file: !47, line: 26, type: !14, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !61)
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !55, type: !12)
// CHECK:STDOUT: !60 = !DILocation(line: 26, column: 3, scope: !55)
// CHECK:STDOUT: !61 = !{!56}
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.ea51de33ab6382d8", scope: null, file: !47, line: 26, type: !14, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !68)
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !62, type: !12)
// CHECK:STDOUT: !66 = !DILocation(line: 28, column: 5, scope: !62)
// CHECK:STDOUT: !67 = !DILocation(line: 26, column: 3, scope: !62)
// CHECK:STDOUT: !68 = !{!63}
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.4440947f5319481f", scope: null, file: !47, line: 26, type: !14, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !75)
// CHECK:STDOUT: !70 = !DILocalVariable(arg: 1, scope: !69, type: !12)
// CHECK:STDOUT: !73 = !DILocation(line: 28, column: 5, scope: !69)
// CHECK:STDOUT: !74 = !DILocation(line: 26, column: 3, scope: !69)
// CHECK:STDOUT: !75 = !{!70}
// CHECK:STDOUT: !76 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.33bf5c55a0b6e3f1", scope: null, file: !47, line: 26, type: !14, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !82)
// CHECK:STDOUT: !77 = !DILocalVariable(arg: 1, scope: !76, type: !12)
// CHECK:STDOUT: !81 = !DILocation(line: 26, column: 3, scope: !76)
// CHECK:STDOUT: !82 = !{!77}
// CHECK:STDOUT: !83 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !84 = !{!"omnipotent char", !83, i64 0}
// CHECK:STDOUT: !85 = !{!"any pointer", !84, i64 0}
// CHECK:STDOUT: !86 = !{!"p1 _ZTSSt16initializer_listIiE", !85, i64 0}
// CHECK:STDOUT: !87 = !{!86, !86, i64 0}
// CHECK:STDOUT: !88 = !{!"p1 _ZTS11vector_like", !85, i64 0}
// CHECK:STDOUT: !89 = !{!88, !88, i64 0}
// CHECK:STDOUT: !90 = !{!"int", !84, i64 0}
// CHECK:STDOUT: !91 = !{!"p1 int", !85, i64 0}
// CHECK:STDOUT: !92 = !{!"long", !84, i64 0}
// CHECK:STDOUT: !94 = !{!91, !91, i64 0}
// CHECK:STDOUT: !95 = !{!92, !92, i64 0}
// CHECK:STDOUT: !96 = !{i64 0, i64 8, !94, i64 8, i64 8, !95}
// CHECK:STDOUT: !97 = !{!"p1 _ZTS15nontrivial_dtor", !85, i64 0}
// CHECK:STDOUT: !98 = !{!97, !97, i64 0}
// CHECK:STDOUT: !99 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !100 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !101 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !102 = !{!"__libc_errno", !90, i64 0}
// CHECK:STDOUT: !103 = !{!102, !90, i64 0}
// CHECK:STDOUT:
