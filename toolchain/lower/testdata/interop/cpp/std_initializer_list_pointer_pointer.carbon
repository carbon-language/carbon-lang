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
// CHECK:STDOUT: define void @_CInitDirectly.Main() #0 !dbg !16 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca [16 x i8], align 8, !dbg !17
// CHECK:STDOUT:   %.loc13_50.3.temp = alloca [3 x i32], align 4, !dbg !18
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !17
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc13_50.3.temp), !dbg !18
// CHECK:STDOUT:   %.loc13_50.4.array.index = getelementptr inbounds [3 x i32], ptr %.loc13_50.3.temp, i32 0, i64 0, !dbg !18
// CHECK:STDOUT:   %.loc13_50.7.array.index = getelementptr inbounds [3 x i32], ptr %.loc13_50.3.temp, i32 0, i64 1, !dbg !18
// CHECK:STDOUT:   %.loc13_50.10.array.index = getelementptr inbounds [3 x i32], ptr %.loc13_50.3.temp, i32 0, i64 2, !dbg !18
// CHECK:STDOUT:   %initializer_list.initializer_list.call.init_list.begin = getelementptr inbounds nuw [16 x i8], ptr %_.var, i32 0, i32 0, !dbg !17
// CHECK:STDOUT:   store ptr @array.loc13_50.15, ptr %initializer_list.initializer_list.call.init_list.begin, align 8, !dbg !17
// CHECK:STDOUT:   %initializer_list.initializer_list.call.init_list.end = getelementptr inbounds nuw [16 x i8], ptr %_.var, i32 0, i32 8, !dbg !17
// CHECK:STDOUT:   store ptr getelementptr inbounds ([3 x i32], ptr @array.loc13_50.15, i32 1), ptr %initializer_list.initializer_list.call.init_list.end, align 8, !dbg !17
// CHECK:STDOUT:   call void @_CWithinLifetime.Main(), !dbg !20
// CHECK:STDOUT:   call void @"_COp.3bc428c96004ce87:core.Destroy.Core"(ptr @array), !dbg !18
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.70db2d92ffddbef7:core.Destroy.Core"(ptr %self) #0 !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.3bc428c96004ce87:core.Destroy.Core"(ptr %self) #0 !dbg !26 {
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
// CHECK:STDOUT:   call void @"_COp.3bc428c96004ce87:core.Destroy.Core"(ptr @array), !dbg !32
// CHECK:STDOUT:   call void @_ZN11vector_likeD2Ev(ptr %_.var), !dbg !31
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN11vector_likeC1ESt16initializer_listIiE.carbon_thunk._(ptr noundef %list, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %list.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %"class.std::initializer_list", align 8
// CHECK:STDOUT:   store ptr %list, ptr %list.addr, align 8, !tbaa !50
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !52
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !52
// CHECK:STDOUT:   %1 = load ptr, ptr %list.addr, align 8, !tbaa !50
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %agg.tmp, ptr align 8 %1, i64 16, i1 false), !tbaa.struct !57
// CHECK:STDOUT:   %2 = getelementptr inbounds nuw { ptr, ptr }, ptr %agg.tmp, i32 0, i32 0
// CHECK:STDOUT:   %3 = load ptr, ptr %2, align 8
// CHECK:STDOUT:   %4 = getelementptr inbounds nuw { ptr, ptr }, ptr %agg.tmp, i32 0, i32 1
// CHECK:STDOUT:   %5 = load ptr, ptr %4, align 8
// CHECK:STDOUT:   call void @_ZN11vector_likeC2ESt16initializer_listIiE(ptr noundef nonnull align 1 dereferenceable(1) %0, ptr %3, ptr %5)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN11vector_likeD2Ev(ptr noundef nonnull align 1 dereferenceable(1) %this) unnamed_addr #2 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !52
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CInitNontrivialDtor.Main() #0 !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca [16 x i8], align 8, !dbg !37
// CHECK:STDOUT:   %.loc23_69.17.temp = alloca [3 x [1 x i8]], align 1, !dbg !38
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !37
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc23_69.17.temp), !dbg !38
// CHECK:STDOUT:   %.loc23_69.18.array.index = getelementptr inbounds [3 x [1 x i8]], ptr %.loc23_69.17.temp, i32 0, i64 0, !dbg !38
// CHECK:STDOUT:   call void @_ZN15nontrivial_dtorC1Ev.carbon_thunk_tuple.(ptr %.loc23_69.18.array.index), !dbg !38
// CHECK:STDOUT:   %.loc23_69.16.array.index = getelementptr inbounds [3 x [1 x i8]], ptr %.loc23_69.17.temp, i32 0, i64 1, !dbg !38
// CHECK:STDOUT:   call void @_ZN15nontrivial_dtorC1Ev.carbon_thunk_tuple.(ptr %.loc23_69.16.array.index), !dbg !38
// CHECK:STDOUT:   %.loc23_69.15.array.index = getelementptr inbounds [3 x [1 x i8]], ptr %.loc23_69.17.temp, i32 0, i64 2, !dbg !38
// CHECK:STDOUT:   call void @_ZN15nontrivial_dtorC1Ev.carbon_thunk_tuple.(ptr %.loc23_69.15.array.index), !dbg !38
// CHECK:STDOUT:   %initializer_list.initializer_list.call.init_list.begin = getelementptr inbounds nuw [16 x i8], ptr %_.var, i32 0, i32 0, !dbg !37
// CHECK:STDOUT:   store ptr %.loc23_69.17.temp, ptr %initializer_list.initializer_list.call.init_list.begin, align 8, !dbg !37
// CHECK:STDOUT:   %initializer_list.initializer_list.call.init_list.end = getelementptr inbounds nuw [16 x i8], ptr %_.var, i32 0, i32 8, !dbg !37
// CHECK:STDOUT:   %initializer_list.initializer_list.call.array.end = getelementptr inbounds [3 x [1 x i8]], ptr %.loc23_69.17.temp, i32 1, !dbg !37
// CHECK:STDOUT:   store ptr %initializer_list.initializer_list.call.array.end, ptr %initializer_list.initializer_list.call.init_list.end, align 8, !dbg !37
// CHECK:STDOUT:   call void @_CWithinLifetime.Main(), !dbg !40
// CHECK:STDOUT:   call void @"_COp.baa51490b2bdaa96:core.Destroy.Core"(ptr %.loc23_69.17.temp), !dbg !38
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN15nontrivial_dtorC1Ev.carbon_thunk_tuple.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !59
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !59
// CHECK:STDOUT:   call void @_ZN15nontrivial_dtorC2Ev(ptr noundef nonnull align 1 dereferenceable(1) %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN15nontrivial_dtorD2Ev(ptr noundef nonnull align 1 dereferenceable(1) %this) unnamed_addr #2 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !59
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.baa51490b2bdaa96:core.Destroy.Core"(ptr %self) #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN11vector_likeC2ESt16initializer_listIiE(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr %list.coerce0, ptr %list.coerce1) unnamed_addr #2 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %list = alloca %"class.std::initializer_list", align 8
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %0 = getelementptr inbounds nuw { ptr, ptr }, ptr %list, i32 0, i32 0
// CHECK:STDOUT:   store ptr %list.coerce0, ptr %0, align 8
// CHECK:STDOUT:   %1 = getelementptr inbounds nuw { ptr, ptr }, ptr %list, i32 0, i32 1
// CHECK:STDOUT:   store ptr %list.coerce1, ptr %1, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !52
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN15nontrivial_dtorC2Ev(ptr noundef nonnull align 1 dereferenceable(1) %this) unnamed_addr #2 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !59
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_ZN15nontrivial_dtorC1Ev.carbon_thunk_tuple., { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!60, !61, !62, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!64}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "use_pointer_pointer.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !{null}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !9 = !{null, !6}
// CHECK:STDOUT: !10 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !12 = !{null, !11}
// CHECK:STDOUT: !13 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !16 = distinct !DISubprogram(name: "InitDirectly", linkageName: "_CInitDirectly.Main", scope: null, file: !4, line: 12, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !17 = !DILocation(line: 13, column: 3, scope: !16)
// CHECK:STDOUT: !18 = !DILocation(line: 13, column: 42, scope: !16)
// CHECK:STDOUT: !20 = !DILocation(line: 14, column: 3, scope: !16)
// CHECK:STDOUT: !21 = !DILocation(line: 12, column: 1, scope: !16)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "Op", linkageName: "_COp.70db2d92ffddbef7:core.Destroy.Core", scope: null, file: !4, line: 13, type: !10, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !25)
// CHECK:STDOUT: !23 = !DILocalVariable(arg: 1, scope: !22, type: !6)
// CHECK:STDOUT: !24 = !DILocation(line: 13, column: 42, scope: !22)
// CHECK:STDOUT: !25 = !{!23}
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "Op", linkageName: "_COp.3bc428c96004ce87:core.Destroy.Core", scope: null, file: !4, line: 13, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !29)
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !26, type: !11)
// CHECK:STDOUT: !28 = !DILocation(line: 13, column: 42, scope: !26)
// CHECK:STDOUT: !29 = !{!27}
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "InitVectorLike", linkageName: "_CInitVectorLike.Main", scope: null, file: !4, line: 17, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !31 = !DILocation(line: 18, column: 3, scope: !30)
// CHECK:STDOUT: !32 = !DILocation(line: 18, column: 28, scope: !30)
// CHECK:STDOUT: !34 = !DILocation(line: 19, column: 3, scope: !30)
// CHECK:STDOUT: !35 = !DILocation(line: 17, column: 1, scope: !30)
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "InitNontrivialDtor", linkageName: "_CInitNontrivialDtor.Main", scope: null, file: !4, line: 22, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !37 = !DILocation(line: 23, column: 3, scope: !36)
// CHECK:STDOUT: !38 = !DILocation(line: 23, column: 58, scope: !36)
// CHECK:STDOUT: !40 = !DILocation(line: 24, column: 3, scope: !36)
// CHECK:STDOUT: !41 = !DILocation(line: 22, column: 1, scope: !36)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "Op", linkageName: "_COp.baa51490b2bdaa96:core.Destroy.Core", scope: null, file: !4, line: 23, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !45)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !42, type: !11)
// CHECK:STDOUT: !44 = !DILocation(line: 23, column: 58, scope: !42)
// CHECK:STDOUT: !45 = !{!43}
// CHECK:STDOUT: !46 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !47 = !{!"omnipotent char", !46, i64 0}
// CHECK:STDOUT: !48 = !{!"any pointer", !47, i64 0}
// CHECK:STDOUT: !49 = !{!"p1 _ZTSSt16initializer_listIiE", !48, i64 0}
// CHECK:STDOUT: !50 = !{!49, !49, i64 0}
// CHECK:STDOUT: !51 = !{!"p1 _ZTS11vector_like", !48, i64 0}
// CHECK:STDOUT: !52 = !{!51, !51, i64 0}
// CHECK:STDOUT: !53 = !{!"int", !47, i64 0}
// CHECK:STDOUT: !54 = !{!"p1 int", !48, i64 0}
// CHECK:STDOUT: !56 = !{!54, !54, i64 0}
// CHECK:STDOUT: !57 = !{i64 0, i64 8, !56, i64 8, i64 8, !56}
// CHECK:STDOUT: !58 = !{!"p1 _ZTS15nontrivial_dtor", !48, i64 0}
// CHECK:STDOUT: !59 = !{!58, !58, i64 0}
// CHECK:STDOUT: !60 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !61 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !62 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !63 = !{!"__libc_errno", !53, i64 0}
// CHECK:STDOUT: !64 = !{!63, !53, i64 0}
// CHECK:STDOUT:
