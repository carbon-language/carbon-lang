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
class C {};

auto TakePtr(C* _Nonnull) -> void;
auto ReturnPtr() -> C* _Nonnull;

auto TakePtrWithThunk(C* _Nonnull, int = 0) -> void;
auto ReturnPtrWithThunk(int = 0) -> C* _Nonnull;
''';

fn PassPtr(p: Cpp.C*) {
  Cpp.TakePtr(p);
}

fn ReturnPtr() -> Cpp.C* {
  return Cpp.ReturnPtr();
}

fn PassPtrWithThunk(p: Cpp.C*) {
  Cpp.TakePtrWithThunk(p);
}

fn ReturnPtrWithThunk() -> Cpp.C* {
  return Cpp.ReturnPtrWithThunk();
}

// ============================================================================
// Nullable pointers
// ============================================================================

// --- nullable.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
class C {};

auto TakePtr(C*) -> void;
auto ReturnPtr() -> C*;

auto TakePtrWithThunk(C*, int = 0) -> void;
auto ReturnPtrWithThunk(int = 0) -> C*;
''';

fn PassPtr(_: Core.Optional(Cpp.C*)) {
  // TODO: Add support for passing an optional here.
  // Cpp.TakePtr(p);
}

fn PassNonnullPtr(p: Cpp.C*) {
  // TODO: The `value_of_initializer` optimization doesn't get used here, so we
  // unnecessarily round-trip this value through memory.
  Cpp.TakePtr(p);
}

fn ReturnPtr() -> Core.Optional(Cpp.C*) {
  return Cpp.ReturnPtr();
}

fn PassPtrWithThunk(_: Core.Optional(Cpp.C*)) {
  // TODO: Add support for passing an optional here.
  // Cpp.TakePtrWithThunk(p);
}

fn PassNonnullPtrWithThunk(p: Cpp.C*) {
  Cpp.TakePtrWithThunk(p);
}

fn ReturnPtrWithThunk() -> Core.Optional(Cpp.C*) {
  return Cpp.ReturnPtrWithThunk();
}

// --- add_const.carbon

import Cpp inline '''
class C {};
C *make();
void take(const C*);
''';

fn Convert() {
  Cpp.take(Cpp.make());
}

// CHECK:STDOUT: ; ModuleID = 'nonnull.carbon'
// CHECK:STDOUT: source_filename = "nonnull.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassPtr.Main(ptr %p) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z7TakePtrP1C(ptr %p), !dbg !17
// CHECK:STDOUT:   ret void, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z7TakePtrP1C(ptr noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnPtr.Main() #0 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnPtr.call = call ptr @_Z9ReturnPtrv(), !dbg !22
// CHECK:STDOUT:   ret ptr %ReturnPtr.call, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z9ReturnPtrv() #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassPtrWithThunk.Main(ptr %p) #0 !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z16TakePtrWithThunkP1Ci.carbon_thunk._(ptr %p), !dbg !27
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z16TakePtrWithThunkP1Ci.carbon_thunk._(ptr noundef %0) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !29
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !29
// CHECK:STDOUT:   call void @_Z16TakePtrWithThunkP1Ci(ptr noundef %1, i32 noundef 0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnPtrWithThunk.Main() #0 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnPtrWithThunk__carbon_thunk.call = call ptr @_Z18ReturnPtrWithThunki.carbon_thunk.(), !dbg !33
// CHECK:STDOUT:   ret ptr %ReturnPtrWithThunk__carbon_thunk.call, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal noundef ptr @_Z18ReturnPtrWithThunki.carbon_thunk.() #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %call = call noundef ptr @_Z18ReturnPtrWithThunki(i32 noundef 0)
// CHECK:STDOUT:   ret ptr %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z16TakePtrWithThunkP1Ci(ptr noundef, i32 noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z18ReturnPtrWithThunki(i32 noundef) #1
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
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "PassPtr", linkageName: "_CPassPtr.Main", scope: null, file: !1, line: 14, type: !12, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !15)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{null, !14}
// CHECK:STDOUT: !14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !15 = !{!16}
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 1, scope: !11, type: !14)
// CHECK:STDOUT: !17 = !DILocation(line: 15, column: 3, scope: !11)
// CHECK:STDOUT: !18 = !DILocation(line: 14, column: 1, scope: !11)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "ReturnPtr", linkageName: "_CReturnPtr.Main", scope: null, file: !1, line: 18, type: !20, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !20 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !21 = !{!14}
// CHECK:STDOUT: !22 = !DILocation(line: 19, column: 10, scope: !19)
// CHECK:STDOUT: !23 = !DILocation(line: 19, column: 3, scope: !19)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "PassPtrWithThunk", linkageName: "_CPassPtrWithThunk.Main", scope: null, file: !1, line: 22, type: !12, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !25)
// CHECK:STDOUT: !25 = !{!26}
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 1, scope: !24, type: !14)
// CHECK:STDOUT: !27 = !DILocation(line: 23, column: 3, scope: !24)
// CHECK:STDOUT: !28 = !DILocation(line: 22, column: 1, scope: !24)
// CHECK:STDOUT: !29 = !{!30, !30, i64 0}
// CHECK:STDOUT: !30 = !{!"p1 _ZTS1C", !31, i64 0}
// CHECK:STDOUT: !31 = !{!"any pointer", !9, i64 0}
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "ReturnPtrWithThunk", linkageName: "_CReturnPtrWithThunk.Main", scope: null, file: !1, line: 26, type: !20, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !33 = !DILocation(line: 27, column: 10, scope: !32)
// CHECK:STDOUT: !34 = !DILocation(line: 27, column: 3, scope: !32)
// CHECK:STDOUT: ; ModuleID = 'nullable.carbon'
// CHECK:STDOUT: source_filename = "nullable.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassPtr.Main(ptr %_) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassNonnullPtr.Main(ptr %p) #0 !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc22_15.2.temp = alloca ptr, align 8, !dbg !21
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.0df7ee04ac3e2bca"(ptr %p), !dbg !21
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc22_15.2.temp), !dbg !21
// CHECK:STDOUT:   store ptr %U.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc22_15.2.temp, align 8, !dbg !21
// CHECK:STDOUT:   %.loc22_15.4 = load ptr, ptr %.loc22_15.2.temp, align 8, !dbg !21
// CHECK:STDOUT:   call void @_Z7TakePtrP1C(ptr %.loc22_15.4), !dbg !22
// CHECK:STDOUT:   call void @"_COp.f38d07f92d84f295:core.Destroy.Core"(ptr %.loc22_15.2.temp), !dbg !21
// CHECK:STDOUT:   ret void, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z7TakePtrP1C(ptr noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.b66d38b36d1f26ac:core.Destroy.Core"(ptr %self) #0 !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.0eaa291c18f0461d:core.Destroy.Core"(ptr %self) #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.f38d07f92d84f295:core.Destroy.Core"(ptr %self) #0 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnPtr.Main() #0 !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnPtr.call = call ptr @_Z9ReturnPtrv(), !dbg !39
// CHECK:STDOUT:   ret ptr %ReturnPtr.call, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z9ReturnPtrv() #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassPtrWithThunk.Main(ptr %_) #0 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassNonnullPtrWithThunk.Main(ptr %p) #0 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc35_24.2.temp = alloca ptr, align 8, !dbg !48
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.0df7ee04ac3e2bca"(ptr %p), !dbg !48
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc35_24.2.temp), !dbg !48
// CHECK:STDOUT:   store ptr %U.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc35_24.2.temp, align 8, !dbg !48
// CHECK:STDOUT:   %.loc35_24.4 = load ptr, ptr %.loc35_24.2.temp, align 8, !dbg !48
// CHECK:STDOUT:   call void @_Z16TakePtrWithThunkP1Ci.carbon_thunk._(ptr %.loc35_24.4), !dbg !49
// CHECK:STDOUT:   call void @"_COp.f38d07f92d84f295:core.Destroy.Core"(ptr %.loc35_24.2.temp), !dbg !48
// CHECK:STDOUT:   ret void, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z16TakePtrWithThunkP1Ci.carbon_thunk._(ptr noundef %0) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !51
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !51
// CHECK:STDOUT:   call void @_Z16TakePtrWithThunkP1Ci(ptr noundef %1, i32 noundef 0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnPtrWithThunk.Main() #0 !dbg !54 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnPtrWithThunk__carbon_thunk.call = call ptr @_Z18ReturnPtrWithThunki.carbon_thunk.(), !dbg !55
// CHECK:STDOUT:   ret ptr %ReturnPtrWithThunk__carbon_thunk.call, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal noundef ptr @_Z18ReturnPtrWithThunki.carbon_thunk.() #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %call = call noundef ptr @_Z18ReturnPtrWithThunki(i32 noundef 0)
// CHECK:STDOUT:   ret ptr %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.0df7ee04ac3e2bca"(ptr %self) #0 !dbg !57 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.ce205433e72b0023"(ptr %self), !dbg !63
// CHECK:STDOUT:   ret ptr %1, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.ce205433e72b0023"(ptr %self) #0 !dbg !65 {
// CHECK:STDOUT:   %1 = call ptr @_CSome.Optional.Core.ce205433e72b0023(ptr %self), !dbg !68
// CHECK:STDOUT:   ret ptr %1, !dbg !69
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CSome.Optional.Core.ce205433e72b0023(ptr %value) #0 !dbg !70 {
// CHECK:STDOUT:   %1 = call ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.fc8b3e0dc71c345e"(ptr %value), !dbg !73
// CHECK:STDOUT:   ret ptr %1, !dbg !74
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.fc8b3e0dc71c345e"(ptr %self) #0 !dbg !75 {
// CHECK:STDOUT:   %1 = alloca ptr, align 8, !dbg !78
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !78
// CHECK:STDOUT:   store ptr poison, ptr %1, align 8, !dbg !78
// CHECK:STDOUT:   store ptr %self, ptr %1, align 8, !dbg !79
// CHECK:STDOUT:   %2 = load ptr, ptr %1, align 8, !dbg !80
// CHECK:STDOUT:   call void @"_COp.0eaa291c18f0461d:core.Destroy.Core"(ptr %1), !dbg !78
// CHECK:STDOUT:   ret ptr %2, !dbg !81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z16TakePtrWithThunkP1Ci(ptr noundef, i32 noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z18ReturnPtrWithThunki(i32 noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.0df7ee04ac3e2bca", { 1, 0 }
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
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "PassPtr", linkageName: "_CPassPtr.Main", scope: null, file: !1, line: 14, type: !12, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !15)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{null, !14}
// CHECK:STDOUT: !14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !15 = !{!16}
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 1, scope: !11, type: !14)
// CHECK:STDOUT: !17 = !DILocation(line: 14, column: 1, scope: !11)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "PassNonnullPtr", linkageName: "_CPassNonnullPtr.Main", scope: null, file: !1, line: 19, type: !12, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !19)
// CHECK:STDOUT: !19 = !{!20}
// CHECK:STDOUT: !20 = !DILocalVariable(arg: 1, scope: !18, type: !14)
// CHECK:STDOUT: !21 = !DILocation(line: 22, column: 15, scope: !18)
// CHECK:STDOUT: !22 = !DILocation(line: 22, column: 3, scope: !18)
// CHECK:STDOUT: !23 = !DILocation(line: 19, column: 1, scope: !18)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "Op", linkageName: "_COp.b66d38b36d1f26ac:core.Destroy.Core", scope: null, file: !1, line: 22, type: !12, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !25)
// CHECK:STDOUT: !25 = !{!26}
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 1, scope: !24, type: !14)
// CHECK:STDOUT: !27 = !DILocation(line: 22, column: 15, scope: !24)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "Op", linkageName: "_COp.0eaa291c18f0461d:core.Destroy.Core", scope: null, file: !1, line: 22, type: !12, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !29)
// CHECK:STDOUT: !29 = !{!30}
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 1, scope: !28, type: !14)
// CHECK:STDOUT: !31 = !DILocation(line: 22, column: 15, scope: !28)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "Op", linkageName: "_COp.f38d07f92d84f295:core.Destroy.Core", scope: null, file: !1, line: 22, type: !12, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !33)
// CHECK:STDOUT: !33 = !{!34}
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 1, scope: !32, type: !14)
// CHECK:STDOUT: !35 = !DILocation(line: 22, column: 15, scope: !32)
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "ReturnPtr", linkageName: "_CReturnPtr.Main", scope: null, file: !1, line: 25, type: !37, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !37 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !38 = !{!14}
// CHECK:STDOUT: !39 = !DILocation(line: 26, column: 10, scope: !36)
// CHECK:STDOUT: !40 = !DILocation(line: 26, column: 3, scope: !36)
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "PassPtrWithThunk", linkageName: "_CPassPtrWithThunk.Main", scope: null, file: !1, line: 29, type: !12, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !42)
// CHECK:STDOUT: !42 = !{!43}
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !41, type: !14)
// CHECK:STDOUT: !44 = !DILocation(line: 29, column: 1, scope: !41)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "PassNonnullPtrWithThunk", linkageName: "_CPassNonnullPtrWithThunk.Main", scope: null, file: !1, line: 34, type: !12, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !46)
// CHECK:STDOUT: !46 = !{!47}
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !45, type: !14)
// CHECK:STDOUT: !48 = !DILocation(line: 35, column: 24, scope: !45)
// CHECK:STDOUT: !49 = !DILocation(line: 35, column: 3, scope: !45)
// CHECK:STDOUT: !50 = !DILocation(line: 34, column: 1, scope: !45)
// CHECK:STDOUT: !51 = !{!52, !52, i64 0}
// CHECK:STDOUT: !52 = !{!"p1 _ZTS1C", !53, i64 0}
// CHECK:STDOUT: !53 = !{!"any pointer", !9, i64 0}
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "ReturnPtrWithThunk", linkageName: "_CReturnPtrWithThunk.Main", scope: null, file: !1, line: 38, type: !37, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !55 = !DILocation(line: 39, column: 10, scope: !54)
// CHECK:STDOUT: !56 = !DILocation(line: 39, column: 3, scope: !54)
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.0df7ee04ac3e2bca", scope: null, file: !58, line: 105, type: !59, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !61)
// CHECK:STDOUT: !58 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !59 = !DISubroutineType(types: !60)
// CHECK:STDOUT: !60 = !{!14, !14}
// CHECK:STDOUT: !61 = !{!62}
// CHECK:STDOUT: !62 = !DILocalVariable(arg: 1, scope: !57, type: !14)
// CHECK:STDOUT: !63 = !DILocation(line: 106, column: 12, scope: !57)
// CHECK:STDOUT: !64 = !DILocation(line: 106, column: 5, scope: !57)
// CHECK:STDOUT: !65 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.ce205433e72b0023", scope: null, file: !58, line: 80, type: !59, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !66)
// CHECK:STDOUT: !66 = !{!67}
// CHECK:STDOUT: !67 = !DILocalVariable(arg: 1, scope: !65, type: !14)
// CHECK:STDOUT: !68 = !DILocation(line: 81, column: 12, scope: !65)
// CHECK:STDOUT: !69 = !DILocation(line: 81, column: 5, scope: !65)
// CHECK:STDOUT: !70 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.ce205433e72b0023", scope: null, file: !58, line: 33, type: !59, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !71)
// CHECK:STDOUT: !71 = !{!72}
// CHECK:STDOUT: !72 = !DILocalVariable(arg: 1, scope: !70, type: !14)
// CHECK:STDOUT: !73 = !DILocation(line: 34, column: 12, scope: !70)
// CHECK:STDOUT: !74 = !DILocation(line: 34, column: 5, scope: !70)
// CHECK:STDOUT: !75 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.e8f8f92d3d08d149:OptionalStorage.Core.fc8b3e0dc71c345e", scope: null, file: !58, line: 166, type: !59, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !76)
// CHECK:STDOUT: !76 = !{!77}
// CHECK:STDOUT: !77 = !DILocalVariable(arg: 1, scope: !75, type: !14)
// CHECK:STDOUT: !78 = !DILocation(line: 167, column: 14, scope: !75)
// CHECK:STDOUT: !79 = !DILocation(line: 169, column: 5, scope: !75)
// CHECK:STDOUT: !80 = !DILocation(line: 167, column: 18, scope: !75)
// CHECK:STDOUT: !81 = !DILocation(line: 170, column: 5, scope: !75)
// CHECK:STDOUT: ; ModuleID = 'add_const.carbon'
// CHECK:STDOUT: source_filename = "add_const.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvert.Main() #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_21.1.temp = alloca ptr, align 8, !dbg !14
// CHECK:STDOUT:   %.loc9_21.5.temp = alloca ptr, align 8, !dbg !14
// CHECK:STDOUT:   %make.call = call ptr @_Z4makev(), !dbg !14
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc9_21.1.temp), !dbg !14
// CHECK:STDOUT:   store ptr %make.call, ptr %.loc9_21.1.temp, align 8, !dbg !14
// CHECK:STDOUT:   %.loc9_21.3 = load ptr, ptr %.loc9_21.1.temp, align 8, !dbg !14
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.26284956f90c5004"(ptr %.loc9_21.3), !dbg !14
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc9_21.5.temp), !dbg !14
// CHECK:STDOUT:   store ptr %U.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc9_21.5.temp, align 8, !dbg !14
// CHECK:STDOUT:   %.loc9_21.7 = load ptr, ptr %.loc9_21.5.temp, align 8, !dbg !14
// CHECK:STDOUT:   call void @_Z4takePK1C(ptr %.loc9_21.7), !dbg !15
// CHECK:STDOUT:   call void @"_COp.4713799e8769f265:core.Destroy.Core"(ptr %.loc9_21.5.temp), !dbg !14
// CHECK:STDOUT:   call void @"_COp.2b115399396a8344:core.Destroy.Core"(ptr %.loc9_21.1.temp), !dbg !14
// CHECK:STDOUT:   ret void, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z4makev() #1
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z4takePK1C(ptr noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.4a199b7f45343bf7:core.Destroy.Core"(ptr %self) #0 !dbg !17 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.95d54adfec6e2c8e:core.Destroy.Core"(ptr %self) #0 !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.4713799e8769f265:core.Destroy.Core"(ptr %self) #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.b66d38b36d1f26ac:core.Destroy.Core"(ptr %self) #0 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.0eaa291c18f0461d:core.Destroy.Core"(ptr %self) #0 !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.2b115399396a8344:core.Destroy.Core"(ptr %self) #0 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.26284956f90c5004"(ptr %self) #0 !dbg !44 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.Optional.c661605f12aa337a.Core:OptionalAs.1dac4564233fecd1.Core.ba0bf13f1b5bc657"(ptr %self), !dbg !50
// CHECK:STDOUT:   ret ptr %1, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.Optional.c661605f12aa337a.Core:OptionalAs.1dac4564233fecd1.Core.ba0bf13f1b5bc657"(ptr %self) #0 !dbg !52 {
// CHECK:STDOUT:   %temp = alloca ptr, align 8, !dbg !55
// CHECK:STDOUT:   %temp1 = alloca ptr, align 8, !dbg !55
// CHECK:STDOUT:   %1 = call i1 @_CHasValue.Optional.Core.0d555305d9583d05(ptr %self), !dbg !56
// CHECK:STDOUT:   br i1 %1, label %2, label %7, !dbg !57
// CHECK:STDOUT:
// CHECK:STDOUT: 2:                                                ; preds = %0
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !55
// CHECK:STDOUT:   %3 = call ptr @_CGet.Optional.Core.0d555305d9583d05(ptr %self), !dbg !55
// CHECK:STDOUT:   store ptr %3, ptr %temp, align 8, !dbg !55
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp1), !dbg !55
// CHECK:STDOUT:   %4 = load ptr, ptr %temp, align 8, !dbg !55
// CHECK:STDOUT:   store ptr %4, ptr %temp1, align 8, !dbg !55
// CHECK:STDOUT:   %5 = load ptr, ptr %temp1, align 8, !dbg !55
// CHECK:STDOUT:   %6 = call ptr @_CSome.Optional.Core.deb37f08ddf50b30(ptr %5), !dbg !58
// CHECK:STDOUT:   ret ptr %6, !dbg !59
// CHECK:STDOUT:
// CHECK:STDOUT: 7:                                                ; preds = %0
// CHECK:STDOUT:   %8 = call ptr @_CNone.Optional.Core.deb37f08ddf50b30(), !dbg !60
// CHECK:STDOUT:   ret ptr %8, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.0d555305d9583d05(ptr %self) #0 !dbg !62 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.e8f8f92d3d08d149:OptionalStorage.Core.fc8b3e0dc71c345e"(ptr %self), !dbg !65
// CHECK:STDOUT:   ret i1 %1, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CGet.Optional.Core.0d555305d9583d05(ptr %self) #0 !dbg !67 {
// CHECK:STDOUT:   %1 = call ptr @"_CGet.e8f8f92d3d08d149:OptionalStorage.Core.fc8b3e0dc71c345e"(ptr %self), !dbg !70
// CHECK:STDOUT:   ret ptr %1, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CSome.Optional.Core.deb37f08ddf50b30(ptr %value) #0 !dbg !72 {
// CHECK:STDOUT:   %1 = call ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.8753681fdb639539"(ptr %value), !dbg !75
// CHECK:STDOUT:   ret ptr %1, !dbg !76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CNone.Optional.Core.deb37f08ddf50b30() #0 !dbg !77 {
// CHECK:STDOUT:   ret ptr null, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.e8f8f92d3d08d149:OptionalStorage.Core.fc8b3e0dc71c345e"(ptr %value) #0 !dbg !81 {
// CHECK:STDOUT:   %1 = icmp eq ptr %value, null, !dbg !84
// CHECK:STDOUT:   %2 = xor i1 %1, true, !dbg !85
// CHECK:STDOUT:   ret i1 %2, !dbg !86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CGet.e8f8f92d3d08d149:OptionalStorage.Core.fc8b3e0dc71c345e"(ptr %value) #0 !dbg !87 {
// CHECK:STDOUT:   ret ptr %value, !dbg !90
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.8753681fdb639539"(ptr %self) #0 !dbg !91 {
// CHECK:STDOUT:   %1 = alloca ptr, align 8, !dbg !94
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !94
// CHECK:STDOUT:   store ptr poison, ptr %1, align 8, !dbg !94
// CHECK:STDOUT:   store ptr %self, ptr %1, align 8, !dbg !95
// CHECK:STDOUT:   %2 = load ptr, ptr %1, align 8, !dbg !96
// CHECK:STDOUT:   call void @"_COp.95d54adfec6e2c8e:core.Destroy.Core"(ptr %1), !dbg !94
// CHECK:STDOUT:   ret ptr %2, !dbg !97
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
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Main", scope: null, file: !1, line: 8, type: !12, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{null}
// CHECK:STDOUT: !14 = !DILocation(line: 9, column: 12, scope: !11)
// CHECK:STDOUT: !15 = !DILocation(line: 9, column: 3, scope: !11)
// CHECK:STDOUT: !16 = !DILocation(line: 8, column: 1, scope: !11)
// CHECK:STDOUT: !17 = distinct !DISubprogram(name: "Op", linkageName: "_COp.4a199b7f45343bf7:core.Destroy.Core", scope: null, file: !1, line: 9, type: !18, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !21)
// CHECK:STDOUT: !18 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !19 = !{null, !20}
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{!22}
// CHECK:STDOUT: !22 = !DILocalVariable(arg: 1, scope: !17, type: !20)
// CHECK:STDOUT: !23 = !DILocation(line: 9, column: 12, scope: !17)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "Op", linkageName: "_COp.95d54adfec6e2c8e:core.Destroy.Core", scope: null, file: !1, line: 9, type: !18, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !25)
// CHECK:STDOUT: !25 = !{!26}
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 1, scope: !24, type: !20)
// CHECK:STDOUT: !27 = !DILocation(line: 9, column: 12, scope: !24)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "Op", linkageName: "_COp.4713799e8769f265:core.Destroy.Core", scope: null, file: !1, line: 9, type: !18, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !29)
// CHECK:STDOUT: !29 = !{!30}
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 1, scope: !28, type: !20)
// CHECK:STDOUT: !31 = !DILocation(line: 9, column: 12, scope: !28)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "Op", linkageName: "_COp.b66d38b36d1f26ac:core.Destroy.Core", scope: null, file: !1, line: 9, type: !18, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !33)
// CHECK:STDOUT: !33 = !{!34}
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 1, scope: !32, type: !20)
// CHECK:STDOUT: !35 = !DILocation(line: 9, column: 12, scope: !32)
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "Op", linkageName: "_COp.0eaa291c18f0461d:core.Destroy.Core", scope: null, file: !1, line: 9, type: !18, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !37)
// CHECK:STDOUT: !37 = !{!38}
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !36, type: !20)
// CHECK:STDOUT: !39 = !DILocation(line: 9, column: 12, scope: !36)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "Op", linkageName: "_COp.2b115399396a8344:core.Destroy.Core", scope: null, file: !1, line: 9, type: !18, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !41)
// CHECK:STDOUT: !41 = !{!42}
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 1, scope: !40, type: !20)
// CHECK:STDOUT: !43 = !DILocation(line: 9, column: 12, scope: !40)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.26284956f90c5004", scope: null, file: !45, line: 105, type: !46, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !48)
// CHECK:STDOUT: !45 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !46 = !DISubroutineType(types: !47)
// CHECK:STDOUT: !47 = !{!20, !20}
// CHECK:STDOUT: !48 = !{!49}
// CHECK:STDOUT: !49 = !DILocalVariable(arg: 1, scope: !44, type: !20)
// CHECK:STDOUT: !50 = !DILocation(line: 106, column: 12, scope: !44)
// CHECK:STDOUT: !51 = !DILocation(line: 106, column: 5, scope: !44)
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Optional.c661605f12aa337a.Core:OptionalAs.1dac4564233fecd1.Core.ba0bf13f1b5bc657", scope: null, file: !45, line: 95, type: !46, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !53)
// CHECK:STDOUT: !53 = !{!54}
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !52, type: !20)
// CHECK:STDOUT: !55 = !DILocation(line: 97, column: 31, scope: !52)
// CHECK:STDOUT: !56 = !DILocation(line: 96, column: 9, scope: !52)
// CHECK:STDOUT: !57 = !DILocation(line: 96, column: 8, scope: !52)
// CHECK:STDOUT: !58 = !DILocation(line: 97, column: 14, scope: !52)
// CHECK:STDOUT: !59 = !DILocation(line: 97, column: 7, scope: !52)
// CHECK:STDOUT: !60 = !DILocation(line: 99, column: 12, scope: !52)
// CHECK:STDOUT: !61 = !DILocation(line: 99, column: 5, scope: !52)
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.0d555305d9583d05", scope: null, file: !45, line: 36, type: !46, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !63)
// CHECK:STDOUT: !63 = !{!64}
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !62, type: !20)
// CHECK:STDOUT: !65 = !DILocation(line: 37, column: 12, scope: !62)
// CHECK:STDOUT: !66 = !DILocation(line: 37, column: 5, scope: !62)
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.0d555305d9583d05", scope: null, file: !45, line: 39, type: !46, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !68)
// CHECK:STDOUT: !68 = !{!69}
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 1, scope: !67, type: !20)
// CHECK:STDOUT: !70 = !DILocation(line: 40, column: 12, scope: !67)
// CHECK:STDOUT: !71 = !DILocation(line: 40, column: 5, scope: !67)
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.deb37f08ddf50b30", scope: null, file: !45, line: 33, type: !46, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !73)
// CHECK:STDOUT: !73 = !{!74}
// CHECK:STDOUT: !74 = !DILocalVariable(arg: 1, scope: !72, type: !20)
// CHECK:STDOUT: !75 = !DILocation(line: 34, column: 12, scope: !72)
// CHECK:STDOUT: !76 = !DILocation(line: 34, column: 5, scope: !72)
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.deb37f08ddf50b30", scope: null, file: !45, line: 30, type: !78, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !78 = !DISubroutineType(types: !79)
// CHECK:STDOUT: !79 = !{!20}
// CHECK:STDOUT: !80 = !DILocation(line: 31, column: 5, scope: !77)
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.e8f8f92d3d08d149:OptionalStorage.Core.fc8b3e0dc71c345e", scope: null, file: !45, line: 172, type: !46, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !82)
// CHECK:STDOUT: !82 = !{!83}
// CHECK:STDOUT: !83 = !DILocalVariable(arg: 1, scope: !81, type: !20)
// CHECK:STDOUT: !84 = !DILocation(line: 173, column: 16, scope: !81)
// CHECK:STDOUT: !85 = !DILocation(line: 173, column: 12, scope: !81)
// CHECK:STDOUT: !86 = !DILocation(line: 173, column: 5, scope: !81)
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.e8f8f92d3d08d149:OptionalStorage.Core.fc8b3e0dc71c345e", scope: null, file: !45, line: 175, type: !46, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !88)
// CHECK:STDOUT: !88 = !{!89}
// CHECK:STDOUT: !89 = !DILocalVariable(arg: 1, scope: !87, type: !20)
// CHECK:STDOUT: !90 = !DILocation(line: 176, column: 5, scope: !87)
// CHECK:STDOUT: !91 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.e8f8f92d3d08d149:OptionalStorage.Core.8753681fdb639539", scope: null, file: !45, line: 166, type: !46, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !92)
// CHECK:STDOUT: !92 = !{!93}
// CHECK:STDOUT: !93 = !DILocalVariable(arg: 1, scope: !91, type: !20)
// CHECK:STDOUT: !94 = !DILocation(line: 167, column: 14, scope: !91)
// CHECK:STDOUT: !95 = !DILocation(line: 169, column: 5, scope: !91)
// CHECK:STDOUT: !96 = !DILocation(line: 167, column: 18, scope: !91)
// CHECK:STDOUT: !97 = !DILocation(line: 170, column: 5, scope: !91)
