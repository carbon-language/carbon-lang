// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/pointer.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/pointer.carbon

// ============================================================================
// Non-null pointers
// ============================================================================

// --- nonnull.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
class NonnullC {};

auto TakeNonnullPtr(NonnullC* _Nonnull) -> void;
auto ReturnNonnullPtr() -> NonnullC* _Nonnull;

auto TakeNonnullPtrWithThunk(NonnullC* _Nonnull, int = 0) -> void;
auto ReturnNonnullPtrWithThunk(int = 0) -> NonnullC* _Nonnull;
''';

fn PassPtr(p: Cpp.NonnullC*) {
  Cpp.TakeNonnullPtr(p);
}

fn ReturnPtr() -> Cpp.NonnullC* {
  return Cpp.ReturnNonnullPtr();
}

fn PassPtrWithThunk(p: Cpp.NonnullC*) {
  Cpp.TakeNonnullPtrWithThunk(p);
}

fn ReturnPtrWithThunk() -> Cpp.NonnullC* {
  return Cpp.ReturnNonnullPtrWithThunk();
}

// ============================================================================
// Nullable pointers
// ============================================================================

// --- nullable.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
class NullableC {};

auto TakeNullablePtr(NullableC*) -> void;
auto ReturnNullablePtr() -> NullableC*;

auto TakeNullablePtrWithThunk(NullableC*, int = 0) -> void;
auto ReturnNullablePtrWithThunk(int = 0) -> NullableC*;
''';

fn PassPtr(_: Core.Optional(Cpp.NullableC*)) {
  // TODO: Add support for passing an optional here.
  // Cpp.TakeNullablePtr(p);
}

fn PassNonnullPtr(p: Cpp.NullableC*) {
  // TODO: The `value_of_initializer` optimization doesn't get used here, so we
  // unnecessarily round-trip this value through memory.
  Cpp.TakeNullablePtr(p);
}

fn ReturnPtr() -> Core.Optional(Cpp.NullableC*) {
  return Cpp.ReturnNullablePtr();
}

fn PassPtrWithThunk(_: Core.Optional(Cpp.NullableC*)) {
  // TODO: Add support for passing an optional here.
  // Cpp.TakeNullablePtrWithThunk(p);
}

fn PassNonnullPtrWithThunk(p: Cpp.NullableC*) {
  Cpp.TakeNullablePtrWithThunk(p);
}

fn ReturnPtrWithThunk() -> Core.Optional(Cpp.NullableC*) {
  return Cpp.ReturnNullablePtrWithThunk();
}

// --- add_const.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
class ConstC {};
ConstC *make();
void take(const ConstC*);
''';

fn Convert() {
  Cpp.take(Cpp.make());
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'nonnull.carbon'
// CHECK:STDOUT: source_filename = "nonnull.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassPtr.Main(ptr %p) #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z14TakeNonnullPtrP8NonnullC(ptr %p), !dbg !18
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z14TakeNonnullPtrP8NonnullC(ptr noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnPtr.Main() #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnNonnullPtr.call = call ptr @_Z16ReturnNonnullPtrv(), !dbg !23
// CHECK:STDOUT:   ret ptr %ReturnNonnullPtr.call, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z16ReturnNonnullPtrv() #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassPtrWithThunk.Main(ptr %p) #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z23TakeNonnullPtrWithThunkP8NonnullCi.carbon_thunk._(ptr %p), !dbg !28
// CHECK:STDOUT:   ret void, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z23TakeNonnullPtrWithThunkP8NonnullCi.carbon_thunk._(ptr noundef %0) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !30
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !30
// CHECK:STDOUT:   call void @_Z23TakeNonnullPtrWithThunkP8NonnullCi(ptr noundef %1, i32 noundef 0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnPtrWithThunk.Main() #0 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnNonnullPtrWithThunk__carbon_thunk.call = call ptr @_Z25ReturnNonnullPtrWithThunki.carbon_thunk.(), !dbg !34
// CHECK:STDOUT:   ret ptr %ReturnNonnullPtrWithThunk__carbon_thunk.call, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal noundef ptr @_Z25ReturnNonnullPtrWithThunki.carbon_thunk.() #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %call = call noundef ptr @_Z25ReturnNonnullPtrWithThunki(i32 noundef 0)
// CHECK:STDOUT:   ret ptr %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z23TakeNonnullPtrWithThunkP8NonnullCi(ptr noundef, i32 noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z25ReturnNonnullPtrWithThunki(i32 noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3, !4, !5, !6}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!7}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "nonnull.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !3 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !5 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !6 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !{!8, !9, i64 0}
// CHECK:STDOUT: !8 = !{!"__libc_errno", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"int", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"omnipotent char", !11, i64 0}
// CHECK:STDOUT: !11 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "PassPtr", linkageName: "_CPassPtr.Main", scope: null, file: !1, line: 14, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !16)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{null, !15}
// CHECK:STDOUT: !15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !16 = !{!17}
// CHECK:STDOUT: !17 = !DILocalVariable(arg: 1, scope: !12, type: !15)
// CHECK:STDOUT: !18 = !DILocation(line: 15, column: 3, scope: !12)
// CHECK:STDOUT: !19 = !DILocation(line: 14, column: 1, scope: !12)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "ReturnPtr", linkageName: "_CReturnPtr.Main", scope: null, file: !1, line: 18, type: !21, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !21 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !22 = !{!15}
// CHECK:STDOUT: !23 = !DILocation(line: 19, column: 10, scope: !20)
// CHECK:STDOUT: !24 = !DILocation(line: 19, column: 3, scope: !20)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "PassPtrWithThunk", linkageName: "_CPassPtrWithThunk.Main", scope: null, file: !1, line: 22, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !26)
// CHECK:STDOUT: !26 = !{!27}
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !25, type: !15)
// CHECK:STDOUT: !28 = !DILocation(line: 23, column: 3, scope: !25)
// CHECK:STDOUT: !29 = !DILocation(line: 22, column: 1, scope: !25)
// CHECK:STDOUT: !30 = !{!31, !31, i64 0}
// CHECK:STDOUT: !31 = !{!"p1 _ZTS8NonnullC", !32, i64 0}
// CHECK:STDOUT: !32 = !{!"any pointer", !10, i64 0}
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "ReturnPtrWithThunk", linkageName: "_CReturnPtrWithThunk.Main", scope: null, file: !1, line: 26, type: !21, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !34 = !DILocation(line: 27, column: 10, scope: !33)
// CHECK:STDOUT: !35 = !DILocation(line: 27, column: 3, scope: !33)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'nullable.carbon'
// CHECK:STDOUT: source_filename = "nullable.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassPtr.Main(ptr %_) #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassNonnullPtr.Main(ptr %p) #0 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc22_23.2.temp = alloca ptr, align 8, !dbg !22
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.75a9143bcbb84fa5"(ptr %p), !dbg !22
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc22_23.2.temp), !dbg !22
// CHECK:STDOUT:   store ptr %U.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc22_23.2.temp, align 8, !dbg !22
// CHECK:STDOUT:   %.loc22_23.4 = load ptr, ptr %.loc22_23.2.temp, align 8, !dbg !22
// CHECK:STDOUT:   call void @_Z15TakeNullablePtrP9NullableC(ptr %.loc22_23.4), !dbg !23
// CHECK:STDOUT:   call void @"_COp.af4d0379a4c35228:core.Destroy.Core"(ptr %.loc22_23.2.temp), !dbg !22
// CHECK:STDOUT:   ret void, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z15TakeNullablePtrP9NullableC(ptr noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.c4feefb53715d971:core.Destroy.Core"(ptr %self) #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.1a4db661cca0482a:core.Destroy.Core"(ptr %self) #0 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.af4d0379a4c35228:core.Destroy.Core"(ptr %self) #0 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnPtr.Main() #0 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnNullablePtr.call = call ptr @_Z17ReturnNullablePtrv(), !dbg !40
// CHECK:STDOUT:   ret ptr %ReturnNullablePtr.call, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z17ReturnNullablePtrv() #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassPtrWithThunk.Main(ptr %_) #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassNonnullPtrWithThunk.Main(ptr %p) #0 !dbg !46 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc35_32.2.temp = alloca ptr, align 8, !dbg !49
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.75a9143bcbb84fa5"(ptr %p), !dbg !49
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc35_32.2.temp), !dbg !49
// CHECK:STDOUT:   store ptr %U.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc35_32.2.temp, align 8, !dbg !49
// CHECK:STDOUT:   %.loc35_32.4 = load ptr, ptr %.loc35_32.2.temp, align 8, !dbg !49
// CHECK:STDOUT:   call void @_Z24TakeNullablePtrWithThunkP9NullableCi.carbon_thunk._(ptr %.loc35_32.4), !dbg !50
// CHECK:STDOUT:   call void @"_COp.af4d0379a4c35228:core.Destroy.Core"(ptr %.loc35_32.2.temp), !dbg !49
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z24TakeNullablePtrWithThunkP9NullableCi.carbon_thunk._(ptr noundef %0) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !52
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !52
// CHECK:STDOUT:   call void @_Z24TakeNullablePtrWithThunkP9NullableCi(ptr noundef %1, i32 noundef 0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnPtrWithThunk.Main() #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnNullablePtrWithThunk__carbon_thunk.call = call ptr @_Z26ReturnNullablePtrWithThunki.carbon_thunk.(), !dbg !56
// CHECK:STDOUT:   ret ptr %ReturnNullablePtrWithThunk__carbon_thunk.call, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal noundef ptr @_Z26ReturnNullablePtrWithThunki.carbon_thunk.() #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %call = call noundef ptr @_Z26ReturnNullablePtrWithThunki(i32 noundef 0)
// CHECK:STDOUT:   ret ptr %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.75a9143bcbb84fa5"(ptr %self) #0 !dbg !58 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.e83c25d1d2fd22c2"(ptr %self), !dbg !64
// CHECK:STDOUT:   ret ptr %1, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.e83c25d1d2fd22c2"(ptr %self) #0 !dbg !66 {
// CHECK:STDOUT:   %1 = call ptr @_CSome.Optional.Core.e83c25d1d2fd22c2(ptr %self), !dbg !69
// CHECK:STDOUT:   ret ptr %1, !dbg !70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CSome.Optional.Core.e83c25d1d2fd22c2(ptr %value) #0 !dbg !71 {
// CHECK:STDOUT:   %1 = call ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.77dd513e5db6ef8d"(ptr %value), !dbg !74
// CHECK:STDOUT:   ret ptr %1, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.77dd513e5db6ef8d"(ptr %self) #0 !dbg !76 {
// CHECK:STDOUT:   %1 = alloca ptr, align 8, !dbg !79
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !79
// CHECK:STDOUT:   store ptr poison, ptr %1, align 8, !dbg !79
// CHECK:STDOUT:   store ptr %self, ptr %1, align 8, !dbg !80
// CHECK:STDOUT:   %2 = load ptr, ptr %1, align 8, !dbg !81
// CHECK:STDOUT:   call void @"_COp.1a4db661cca0482a:core.Destroy.Core"(ptr %1), !dbg !79
// CHECK:STDOUT:   ret ptr %2, !dbg !82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z24TakeNullablePtrWithThunkP9NullableCi(ptr noundef, i32 noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z26ReturnNullablePtrWithThunki(i32 noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.75a9143bcbb84fa5", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 2, 1 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3, !4, !5, !6}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!7}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "nullable.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !3 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !5 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !6 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !{!8, !9, i64 0}
// CHECK:STDOUT: !8 = !{!"__libc_errno", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"int", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"omnipotent char", !11, i64 0}
// CHECK:STDOUT: !11 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "PassPtr", linkageName: "_CPassPtr.Main", scope: null, file: !1, line: 14, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !16)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{null, !15}
// CHECK:STDOUT: !15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !16 = !{!17}
// CHECK:STDOUT: !17 = !DILocalVariable(arg: 1, scope: !12, type: !15)
// CHECK:STDOUT: !18 = !DILocation(line: 14, column: 1, scope: !12)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "PassNonnullPtr", linkageName: "_CPassNonnullPtr.Main", scope: null, file: !1, line: 19, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !20)
// CHECK:STDOUT: !20 = !{!21}
// CHECK:STDOUT: !21 = !DILocalVariable(arg: 1, scope: !19, type: !15)
// CHECK:STDOUT: !22 = !DILocation(line: 22, column: 23, scope: !19)
// CHECK:STDOUT: !23 = !DILocation(line: 22, column: 3, scope: !19)
// CHECK:STDOUT: !24 = !DILocation(line: 19, column: 1, scope: !19)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "Op", linkageName: "_COp.c4feefb53715d971:core.Destroy.Core", scope: null, file: !1, line: 22, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !26)
// CHECK:STDOUT: !26 = !{!27}
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !25, type: !15)
// CHECK:STDOUT: !28 = !DILocation(line: 22, column: 23, scope: !25)
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "Op", linkageName: "_COp.1a4db661cca0482a:core.Destroy.Core", scope: null, file: !1, line: 22, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !30)
// CHECK:STDOUT: !30 = !{!31}
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 1, scope: !29, type: !15)
// CHECK:STDOUT: !32 = !DILocation(line: 22, column: 23, scope: !29)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "Op", linkageName: "_COp.af4d0379a4c35228:core.Destroy.Core", scope: null, file: !1, line: 22, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !34)
// CHECK:STDOUT: !34 = !{!35}
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !33, type: !15)
// CHECK:STDOUT: !36 = !DILocation(line: 22, column: 23, scope: !33)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "ReturnPtr", linkageName: "_CReturnPtr.Main", scope: null, file: !1, line: 25, type: !38, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !38 = !DISubroutineType(types: !39)
// CHECK:STDOUT: !39 = !{!15}
// CHECK:STDOUT: !40 = !DILocation(line: 26, column: 10, scope: !37)
// CHECK:STDOUT: !41 = !DILocation(line: 26, column: 3, scope: !37)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "PassPtrWithThunk", linkageName: "_CPassPtrWithThunk.Main", scope: null, file: !1, line: 29, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !43)
// CHECK:STDOUT: !43 = !{!44}
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 1, scope: !42, type: !15)
// CHECK:STDOUT: !45 = !DILocation(line: 29, column: 1, scope: !42)
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "PassNonnullPtrWithThunk", linkageName: "_CPassNonnullPtrWithThunk.Main", scope: null, file: !1, line: 34, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !47)
// CHECK:STDOUT: !47 = !{!48}
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 1, scope: !46, type: !15)
// CHECK:STDOUT: !49 = !DILocation(line: 35, column: 32, scope: !46)
// CHECK:STDOUT: !50 = !DILocation(line: 35, column: 3, scope: !46)
// CHECK:STDOUT: !51 = !DILocation(line: 34, column: 1, scope: !46)
// CHECK:STDOUT: !52 = !{!53, !53, i64 0}
// CHECK:STDOUT: !53 = !{!"p1 _ZTS9NullableC", !54, i64 0}
// CHECK:STDOUT: !54 = !{!"any pointer", !10, i64 0}
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "ReturnPtrWithThunk", linkageName: "_CReturnPtrWithThunk.Main", scope: null, file: !1, line: 38, type: !38, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !56 = !DILocation(line: 39, column: 10, scope: !55)
// CHECK:STDOUT: !57 = !DILocation(line: 39, column: 3, scope: !55)
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.75a9143bcbb84fa5", scope: null, file: !59, line: 105, type: !60, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !62)
// CHECK:STDOUT: !59 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !60 = !DISubroutineType(types: !61)
// CHECK:STDOUT: !61 = !{!15, !15}
// CHECK:STDOUT: !62 = !{!63}
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !58, type: !15)
// CHECK:STDOUT: !64 = !DILocation(line: 106, column: 12, scope: !58)
// CHECK:STDOUT: !65 = !DILocation(line: 106, column: 5, scope: !58)
// CHECK:STDOUT: !66 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.e83c25d1d2fd22c2", scope: null, file: !59, line: 80, type: !60, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !67)
// CHECK:STDOUT: !67 = !{!68}
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 1, scope: !66, type: !15)
// CHECK:STDOUT: !69 = !DILocation(line: 81, column: 12, scope: !66)
// CHECK:STDOUT: !70 = !DILocation(line: 81, column: 5, scope: !66)
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.e83c25d1d2fd22c2", scope: null, file: !59, line: 33, type: !60, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !72)
// CHECK:STDOUT: !72 = !{!73}
// CHECK:STDOUT: !73 = !DILocalVariable(arg: 1, scope: !71, type: !15)
// CHECK:STDOUT: !74 = !DILocation(line: 34, column: 12, scope: !71)
// CHECK:STDOUT: !75 = !DILocation(line: 34, column: 5, scope: !71)
// CHECK:STDOUT: !76 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.e8f8f92d3d08d149:OptionalStorage.Core.77dd513e5db6ef8d", scope: null, file: !59, line: 166, type: !60, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !77)
// CHECK:STDOUT: !77 = !{!78}
// CHECK:STDOUT: !78 = !DILocalVariable(arg: 1, scope: !76, type: !15)
// CHECK:STDOUT: !79 = !DILocation(line: 167, column: 14, scope: !76)
// CHECK:STDOUT: !80 = !DILocation(line: 169, column: 5, scope: !76)
// CHECK:STDOUT: !81 = !DILocation(line: 167, column: 18, scope: !76)
// CHECK:STDOUT: !82 = !DILocation(line: 170, column: 5, scope: !76)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'add_const.carbon'
// CHECK:STDOUT: source_filename = "add_const.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvert.Main() #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_21.1.temp = alloca ptr, align 8, !dbg !15
// CHECK:STDOUT:   %.loc11_21.5.temp = alloca ptr, align 8, !dbg !15
// CHECK:STDOUT:   %make.call = call ptr @_Z4makev(), !dbg !15
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc11_21.1.temp), !dbg !15
// CHECK:STDOUT:   store ptr %make.call, ptr %.loc11_21.1.temp, align 8, !dbg !15
// CHECK:STDOUT:   %.loc11_21.3 = load ptr, ptr %.loc11_21.1.temp, align 8, !dbg !15
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.f9fd89fc1ccb531f"(ptr %.loc11_21.3), !dbg !15
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc11_21.5.temp), !dbg !15
// CHECK:STDOUT:   store ptr %U.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc11_21.5.temp, align 8, !dbg !15
// CHECK:STDOUT:   %.loc11_21.7 = load ptr, ptr %.loc11_21.5.temp, align 8, !dbg !15
// CHECK:STDOUT:   call void @_Z4takePK6ConstC(ptr %.loc11_21.7), !dbg !16
// CHECK:STDOUT:   call void @"_COp.cde1f9fdf283467b:core.Destroy.Core"(ptr %.loc11_21.5.temp), !dbg !15
// CHECK:STDOUT:   call void @"_COp.a6ac3d473017b2cc:core.Destroy.Core"(ptr %.loc11_21.1.temp), !dbg !15
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z4makev() #1
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z4takePK6ConstC(ptr noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.53a1c4a6a5cbb234:core.Destroy.Core"(ptr %self) #0 !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.fd6b21bba6dfd55d:core.Destroy.Core"(ptr %self) #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.cde1f9fdf283467b:core.Destroy.Core"(ptr %self) #0 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.fc86501cbc1f8185:core.Destroy.Core"(ptr %self) #0 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.b682b9cd8310efa2:core.Destroy.Core"(ptr %self) #0 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.a6ac3d473017b2cc:core.Destroy.Core"(ptr %self) #0 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.f9fd89fc1ccb531f"(ptr %self) #0 !dbg !45 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.Optional.c661605f12aa337a.Core:OptionalAs.1dac4564233fecd1.Core.3efc4c2a62f24739"(ptr %self), !dbg !51
// CHECK:STDOUT:   ret ptr %1, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.Optional.c661605f12aa337a.Core:OptionalAs.1dac4564233fecd1.Core.3efc4c2a62f24739"(ptr %self) #0 !dbg !53 {
// CHECK:STDOUT:   %temp = alloca ptr, align 8, !dbg !56
// CHECK:STDOUT:   %temp1 = alloca ptr, align 8, !dbg !56
// CHECK:STDOUT:   %1 = call i1 @_CHasValue.Optional.Core.0c3518913b4fb338(ptr %self), !dbg !57
// CHECK:STDOUT:   br i1 %1, label %2, label %7, !dbg !58
// CHECK:STDOUT:
// CHECK:STDOUT: 2:                                                ; preds = %0
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !56
// CHECK:STDOUT:   %3 = call ptr @_CGet.Optional.Core.0c3518913b4fb338(ptr %self), !dbg !56
// CHECK:STDOUT:   store ptr %3, ptr %temp, align 8, !dbg !56
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp1), !dbg !56
// CHECK:STDOUT:   %4 = load ptr, ptr %temp, align 8, !dbg !56
// CHECK:STDOUT:   store ptr %4, ptr %temp1, align 8, !dbg !56
// CHECK:STDOUT:   %5 = load ptr, ptr %temp1, align 8, !dbg !56
// CHECK:STDOUT:   %6 = call ptr @_CSome.Optional.Core.305e266058820aa4(ptr %5), !dbg !59
// CHECK:STDOUT:   ret ptr %6, !dbg !60
// CHECK:STDOUT:
// CHECK:STDOUT: 7:                                                ; preds = %0
// CHECK:STDOUT:   %8 = call ptr @_CNone.Optional.Core.305e266058820aa4(), !dbg !61
// CHECK:STDOUT:   ret ptr %8, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.0c3518913b4fb338(ptr %self) #0 !dbg !63 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.e8f8f92d3d08d149:OptionalStorage.Core.21c65d90918a10a4"(ptr %self), !dbg !66
// CHECK:STDOUT:   ret i1 %1, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CGet.Optional.Core.0c3518913b4fb338(ptr %self) #0 !dbg !68 {
// CHECK:STDOUT:   %1 = call ptr @"_CGet.e8f8f92d3d08d149:OptionalStorage.Core.21c65d90918a10a4"(ptr %self), !dbg !71
// CHECK:STDOUT:   ret ptr %1, !dbg !72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CSome.Optional.Core.305e266058820aa4(ptr %value) #0 !dbg !73 {
// CHECK:STDOUT:   %1 = call ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.35c29317da7de352"(ptr %value), !dbg !76
// CHECK:STDOUT:   ret ptr %1, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CNone.Optional.Core.305e266058820aa4() #0 !dbg !78 {
// CHECK:STDOUT:   ret ptr null, !dbg !81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.e8f8f92d3d08d149:OptionalStorage.Core.21c65d90918a10a4"(ptr %value) #0 !dbg !82 {
// CHECK:STDOUT:   %1 = icmp eq ptr %value, null, !dbg !85
// CHECK:STDOUT:   %2 = xor i1 %1, true, !dbg !86
// CHECK:STDOUT:   ret i1 %2, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CGet.e8f8f92d3d08d149:OptionalStorage.Core.21c65d90918a10a4"(ptr %value) #0 !dbg !88 {
// CHECK:STDOUT:   ret ptr %value, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.35c29317da7de352"(ptr %self) #0 !dbg !92 {
// CHECK:STDOUT:   %1 = alloca ptr, align 8, !dbg !95
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !95
// CHECK:STDOUT:   store ptr poison, ptr %1, align 8, !dbg !95
// CHECK:STDOUT:   store ptr %self, ptr %1, align 8, !dbg !96
// CHECK:STDOUT:   %2 = load ptr, ptr %1, align 8, !dbg !97
// CHECK:STDOUT:   call void @"_COp.fd6b21bba6dfd55d:core.Destroy.Core"(ptr %1), !dbg !95
// CHECK:STDOUT:   ret ptr %2, !dbg !98
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 2, 4, 3 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3, !4, !5, !6}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!7}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "add_const.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !3 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !5 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !6 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !{!8, !9, i64 0}
// CHECK:STDOUT: !8 = !{!"__libc_errno", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"int", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"omnipotent char", !11, i64 0}
// CHECK:STDOUT: !11 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Main", scope: null, file: !1, line: 10, type: !13, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{null}
// CHECK:STDOUT: !15 = !DILocation(line: 11, column: 12, scope: !12)
// CHECK:STDOUT: !16 = !DILocation(line: 11, column: 3, scope: !12)
// CHECK:STDOUT: !17 = !DILocation(line: 10, column: 1, scope: !12)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "Op", linkageName: "_COp.53a1c4a6a5cbb234:core.Destroy.Core", scope: null, file: !1, line: 11, type: !19, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !22)
// CHECK:STDOUT: !19 = !DISubroutineType(types: !20)
// CHECK:STDOUT: !20 = !{null, !21}
// CHECK:STDOUT: !21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !22 = !{!23}
// CHECK:STDOUT: !23 = !DILocalVariable(arg: 1, scope: !18, type: !21)
// CHECK:STDOUT: !24 = !DILocation(line: 11, column: 12, scope: !18)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "Op", linkageName: "_COp.fd6b21bba6dfd55d:core.Destroy.Core", scope: null, file: !1, line: 11, type: !19, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !26)
// CHECK:STDOUT: !26 = !{!27}
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !25, type: !21)
// CHECK:STDOUT: !28 = !DILocation(line: 11, column: 12, scope: !25)
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "Op", linkageName: "_COp.cde1f9fdf283467b:core.Destroy.Core", scope: null, file: !1, line: 11, type: !19, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !30)
// CHECK:STDOUT: !30 = !{!31}
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 1, scope: !29, type: !21)
// CHECK:STDOUT: !32 = !DILocation(line: 11, column: 12, scope: !29)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "Op", linkageName: "_COp.fc86501cbc1f8185:core.Destroy.Core", scope: null, file: !1, line: 11, type: !19, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !34)
// CHECK:STDOUT: !34 = !{!35}
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !33, type: !21)
// CHECK:STDOUT: !36 = !DILocation(line: 11, column: 12, scope: !33)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "Op", linkageName: "_COp.b682b9cd8310efa2:core.Destroy.Core", scope: null, file: !1, line: 11, type: !19, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !38)
// CHECK:STDOUT: !38 = !{!39}
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !37, type: !21)
// CHECK:STDOUT: !40 = !DILocation(line: 11, column: 12, scope: !37)
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "Op", linkageName: "_COp.a6ac3d473017b2cc:core.Destroy.Core", scope: null, file: !1, line: 11, type: !19, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !42)
// CHECK:STDOUT: !42 = !{!43}
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !41, type: !21)
// CHECK:STDOUT: !44 = !DILocation(line: 11, column: 12, scope: !41)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.f9fd89fc1ccb531f", scope: null, file: !46, line: 105, type: !47, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !49)
// CHECK:STDOUT: !46 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !47 = !DISubroutineType(types: !48)
// CHECK:STDOUT: !48 = !{!21, !21}
// CHECK:STDOUT: !49 = !{!50}
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 1, scope: !45, type: !21)
// CHECK:STDOUT: !51 = !DILocation(line: 106, column: 12, scope: !45)
// CHECK:STDOUT: !52 = !DILocation(line: 106, column: 5, scope: !45)
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Optional.c661605f12aa337a.Core:OptionalAs.1dac4564233fecd1.Core.3efc4c2a62f24739", scope: null, file: !46, line: 95, type: !47, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !54)
// CHECK:STDOUT: !54 = !{!55}
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 1, scope: !53, type: !21)
// CHECK:STDOUT: !56 = !DILocation(line: 97, column: 31, scope: !53)
// CHECK:STDOUT: !57 = !DILocation(line: 96, column: 9, scope: !53)
// CHECK:STDOUT: !58 = !DILocation(line: 96, column: 8, scope: !53)
// CHECK:STDOUT: !59 = !DILocation(line: 97, column: 14, scope: !53)
// CHECK:STDOUT: !60 = !DILocation(line: 97, column: 7, scope: !53)
// CHECK:STDOUT: !61 = !DILocation(line: 99, column: 12, scope: !53)
// CHECK:STDOUT: !62 = !DILocation(line: 99, column: 5, scope: !53)
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.0c3518913b4fb338", scope: null, file: !46, line: 36, type: !47, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !64)
// CHECK:STDOUT: !64 = !{!65}
// CHECK:STDOUT: !65 = !DILocalVariable(arg: 1, scope: !63, type: !21)
// CHECK:STDOUT: !66 = !DILocation(line: 37, column: 12, scope: !63)
// CHECK:STDOUT: !67 = !DILocation(line: 37, column: 5, scope: !63)
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.0c3518913b4fb338", scope: null, file: !46, line: 39, type: !47, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !69)
// CHECK:STDOUT: !69 = !{!70}
// CHECK:STDOUT: !70 = !DILocalVariable(arg: 1, scope: !68, type: !21)
// CHECK:STDOUT: !71 = !DILocation(line: 40, column: 12, scope: !68)
// CHECK:STDOUT: !72 = !DILocation(line: 40, column: 5, scope: !68)
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.305e266058820aa4", scope: null, file: !46, line: 33, type: !47, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !74)
// CHECK:STDOUT: !74 = !{!75}
// CHECK:STDOUT: !75 = !DILocalVariable(arg: 1, scope: !73, type: !21)
// CHECK:STDOUT: !76 = !DILocation(line: 34, column: 12, scope: !73)
// CHECK:STDOUT: !77 = !DILocation(line: 34, column: 5, scope: !73)
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.305e266058820aa4", scope: null, file: !46, line: 30, type: !79, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !79 = !DISubroutineType(types: !80)
// CHECK:STDOUT: !80 = !{!21}
// CHECK:STDOUT: !81 = !DILocation(line: 31, column: 5, scope: !78)
// CHECK:STDOUT: !82 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.e8f8f92d3d08d149:OptionalStorage.Core.21c65d90918a10a4", scope: null, file: !46, line: 172, type: !47, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !83)
// CHECK:STDOUT: !83 = !{!84}
// CHECK:STDOUT: !84 = !DILocalVariable(arg: 1, scope: !82, type: !21)
// CHECK:STDOUT: !85 = !DILocation(line: 173, column: 16, scope: !82)
// CHECK:STDOUT: !86 = !DILocation(line: 173, column: 12, scope: !82)
// CHECK:STDOUT: !87 = !DILocation(line: 173, column: 5, scope: !82)
// CHECK:STDOUT: !88 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.e8f8f92d3d08d149:OptionalStorage.Core.21c65d90918a10a4", scope: null, file: !46, line: 175, type: !47, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !89)
// CHECK:STDOUT: !89 = !{!90}
// CHECK:STDOUT: !90 = !DILocalVariable(arg: 1, scope: !88, type: !21)
// CHECK:STDOUT: !91 = !DILocation(line: 176, column: 5, scope: !88)
// CHECK:STDOUT: !92 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.e8f8f92d3d08d149:OptionalStorage.Core.35c29317da7de352", scope: null, file: !46, line: 166, type: !47, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !93)
// CHECK:STDOUT: !93 = !{!94}
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 1, scope: !92, type: !21)
// CHECK:STDOUT: !95 = !DILocation(line: 167, column: 14, scope: !92)
// CHECK:STDOUT: !96 = !DILocation(line: 169, column: 5, scope: !92)
// CHECK:STDOUT: !97 = !DILocation(line: 167, column: 18, scope: !92)
// CHECK:STDOUT: !98 = !DILocation(line: 170, column: 5, scope: !92)
// CHECK:STDOUT:
