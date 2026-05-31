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
// CHECK:STDOUT: !6 = !DIFile(filename: "nonnull.carbon", directory: "")
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "PassPtr", linkageName: "_CPassPtr.Main", scope: null, file: !6, line: 14, type: !12, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !15)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{null, !14}
// CHECK:STDOUT: !14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !15 = !{!16}
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 1, scope: !11, type: !14)
// CHECK:STDOUT: !17 = !DILocation(line: 15, column: 3, scope: !11)
// CHECK:STDOUT: !18 = !DILocation(line: 14, column: 1, scope: !11)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "ReturnPtr", linkageName: "_CReturnPtr.Main", scope: null, file: !6, line: 18, type: !20, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !20 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !21 = !{!14}
// CHECK:STDOUT: !22 = !DILocation(line: 19, column: 10, scope: !19)
// CHECK:STDOUT: !23 = !DILocation(line: 19, column: 3, scope: !19)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "PassPtrWithThunk", linkageName: "_CPassPtrWithThunk.Main", scope: null, file: !6, line: 22, type: !12, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !25)
// CHECK:STDOUT: !25 = !{!26}
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 1, scope: !24, type: !14)
// CHECK:STDOUT: !27 = !DILocation(line: 23, column: 3, scope: !24)
// CHECK:STDOUT: !28 = !DILocation(line: 22, column: 1, scope: !24)
// CHECK:STDOUT: !29 = !{!30, !30, i64 0}
// CHECK:STDOUT: !30 = !{!"p1 _ZTS1C", !31, i64 0}
// CHECK:STDOUT: !31 = !{!"any pointer", !9, i64 0}
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "ReturnPtrWithThunk", linkageName: "_CReturnPtrWithThunk.Main", scope: null, file: !6, line: 26, type: !20, spFlags: DISPFlagDefinition, unit: !5)
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
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.6391a65a70757668:ImplicitAs.116c0b3f4fb552e2.Core.470500545f897853"(ptr %p), !dbg !21
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc22_15.2.temp), !dbg !21
// CHECK:STDOUT:   store ptr %U.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc22_15.2.temp, align 8, !dbg !21
// CHECK:STDOUT:   %.loc22_15.4 = load ptr, ptr %.loc22_15.2.temp, align 8, !dbg !21
// CHECK:STDOUT:   call void @_Z7TakePtrP1C(ptr %.loc22_15.4), !dbg !22
// CHECK:STDOUT:   call void @"_COp.f077811596b04c1e:core.Destroy.Core"(ptr %.loc22_15.2.temp), !dbg !21
// CHECK:STDOUT:   ret void, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z7TakePtrP1C(ptr noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.b8799b2262fcbb9d:core.Destroy.Core"(ptr %self) #0 !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.f077811596b04c1e:core.Destroy.Core"(ptr %self) #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnPtr.Main() #0 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnPtr.call = call ptr @_Z9ReturnPtrv(), !dbg !35
// CHECK:STDOUT:   ret ptr %ReturnPtr.call, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z9ReturnPtrv() #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassPtrWithThunk.Main(ptr %_) #0 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassNonnullPtrWithThunk.Main(ptr %p) #0 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc35_24.2.temp = alloca ptr, align 8, !dbg !44
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.6391a65a70757668:ImplicitAs.116c0b3f4fb552e2.Core.470500545f897853"(ptr %p), !dbg !44
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc35_24.2.temp), !dbg !44
// CHECK:STDOUT:   store ptr %U.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc35_24.2.temp, align 8, !dbg !44
// CHECK:STDOUT:   %.loc35_24.4 = load ptr, ptr %.loc35_24.2.temp, align 8, !dbg !44
// CHECK:STDOUT:   call void @_Z16TakePtrWithThunkP1Ci.carbon_thunk._(ptr %.loc35_24.4), !dbg !45
// CHECK:STDOUT:   call void @"_COp.f077811596b04c1e:core.Destroy.Core"(ptr %.loc35_24.2.temp), !dbg !44
// CHECK:STDOUT:   ret void, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z16TakePtrWithThunkP1Ci.carbon_thunk._(ptr noundef %0) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !47
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !47
// CHECK:STDOUT:   call void @_Z16TakePtrWithThunkP1Ci(ptr noundef %1, i32 noundef 0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnPtrWithThunk.Main() #0 !dbg !50 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnPtrWithThunk__carbon_thunk.call = call ptr @_Z18ReturnPtrWithThunki.carbon_thunk.(), !dbg !51
// CHECK:STDOUT:   ret ptr %ReturnPtrWithThunk__carbon_thunk.call, !dbg !52
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
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.6391a65a70757668:ImplicitAs.116c0b3f4fb552e2.Core.470500545f897853"(ptr %self) #0 !dbg !53 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.ba9a17e320476215:OptionalAs.7eef99cd03a74de4.Core.20b091e87e8b790e"(ptr %self), !dbg !59
// CHECK:STDOUT:   ret ptr %1, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.ba9a17e320476215:OptionalAs.7eef99cd03a74de4.Core.20b091e87e8b790e"(ptr %self) #0 !dbg !61 {
// CHECK:STDOUT:   %1 = call ptr @_CSome.Optional.Core.20b091e87e8b790e(ptr %self), !dbg !64
// CHECK:STDOUT:   ret ptr %1, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CSome.Optional.Core.20b091e87e8b790e(ptr %value) #0 !dbg !66 {
// CHECK:STDOUT:   %1 = call ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.b88531aa40419bf2"(ptr %value), !dbg !69
// CHECK:STDOUT:   ret ptr %1, !dbg !70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.b88531aa40419bf2"(ptr %self) #0 !dbg !71 {
// CHECK:STDOUT:   %1 = alloca ptr, align 8, !dbg !74
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !74
// CHECK:STDOUT:   store ptr poison, ptr %1, align 8, !dbg !74
// CHECK:STDOUT:   store ptr %self, ptr %1, align 8, !dbg !75
// CHECK:STDOUT:   %2 = load ptr, ptr %1, align 8, !dbg !76
// CHECK:STDOUT:   call void @"_COp.b8799b2262fcbb9d:core.Destroy.Core"(ptr %1), !dbg !74
// CHECK:STDOUT:   ret ptr %2, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z16TakePtrWithThunkP1Ci(ptr noundef, i32 noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z18ReturnPtrWithThunki(i32 noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_CConvert.6391a65a70757668:ImplicitAs.116c0b3f4fb552e2.Core.470500545f897853", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 2, 1 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
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
// CHECK:STDOUT: !6 = !DIFile(filename: "nullable.carbon", directory: "")
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "PassPtr", linkageName: "_CPassPtr.Main", scope: null, file: !6, line: 14, type: !12, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !15)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{null, !14}
// CHECK:STDOUT: !14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !15 = !{!16}
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 1, scope: !11, type: !14)
// CHECK:STDOUT: !17 = !DILocation(line: 14, column: 1, scope: !11)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "PassNonnullPtr", linkageName: "_CPassNonnullPtr.Main", scope: null, file: !6, line: 19, type: !12, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !19)
// CHECK:STDOUT: !19 = !{!20}
// CHECK:STDOUT: !20 = !DILocalVariable(arg: 1, scope: !18, type: !14)
// CHECK:STDOUT: !21 = !DILocation(line: 22, column: 15, scope: !18)
// CHECK:STDOUT: !22 = !DILocation(line: 22, column: 3, scope: !18)
// CHECK:STDOUT: !23 = !DILocation(line: 19, column: 1, scope: !18)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "Op", linkageName: "_COp.b8799b2262fcbb9d:core.Destroy.Core", scope: null, file: !6, line: 22, type: !12, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !25)
// CHECK:STDOUT: !25 = !{!26}
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 1, scope: !24, type: !14)
// CHECK:STDOUT: !27 = !DILocation(line: 22, column: 15, scope: !24)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "Op", linkageName: "_COp.f077811596b04c1e:core.Destroy.Core", scope: null, file: !6, line: 22, type: !12, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !29)
// CHECK:STDOUT: !29 = !{!30}
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 1, scope: !28, type: !14)
// CHECK:STDOUT: !31 = !DILocation(line: 22, column: 15, scope: !28)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "ReturnPtr", linkageName: "_CReturnPtr.Main", scope: null, file: !6, line: 25, type: !33, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !33 = !DISubroutineType(types: !34)
// CHECK:STDOUT: !34 = !{!14}
// CHECK:STDOUT: !35 = !DILocation(line: 26, column: 10, scope: !32)
// CHECK:STDOUT: !36 = !DILocation(line: 26, column: 3, scope: !32)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "PassPtrWithThunk", linkageName: "_CPassPtrWithThunk.Main", scope: null, file: !6, line: 29, type: !12, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !38)
// CHECK:STDOUT: !38 = !{!39}
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !37, type: !14)
// CHECK:STDOUT: !40 = !DILocation(line: 29, column: 1, scope: !37)
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "PassNonnullPtrWithThunk", linkageName: "_CPassNonnullPtrWithThunk.Main", scope: null, file: !6, line: 34, type: !12, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !42)
// CHECK:STDOUT: !42 = !{!43}
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !41, type: !14)
// CHECK:STDOUT: !44 = !DILocation(line: 35, column: 24, scope: !41)
// CHECK:STDOUT: !45 = !DILocation(line: 35, column: 3, scope: !41)
// CHECK:STDOUT: !46 = !DILocation(line: 34, column: 1, scope: !41)
// CHECK:STDOUT: !47 = !{!48, !48, i64 0}
// CHECK:STDOUT: !48 = !{!"p1 _ZTS1C", !49, i64 0}
// CHECK:STDOUT: !49 = !{!"any pointer", !9, i64 0}
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "ReturnPtrWithThunk", linkageName: "_CReturnPtrWithThunk.Main", scope: null, file: !6, line: 38, type: !33, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !51 = !DILocation(line: 39, column: 10, scope: !50)
// CHECK:STDOUT: !52 = !DILocation(line: 39, column: 3, scope: !50)
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.6391a65a70757668:ImplicitAs.116c0b3f4fb552e2.Core.470500545f897853", scope: null, file: !54, line: 105, type: !55, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !57)
// CHECK:STDOUT: !54 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !55 = !DISubroutineType(types: !56)
// CHECK:STDOUT: !56 = !{!14, !14}
// CHECK:STDOUT: !57 = !{!58}
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 1, scope: !53, type: !14)
// CHECK:STDOUT: !59 = !DILocation(line: 106, column: 12, scope: !53)
// CHECK:STDOUT: !60 = !DILocation(line: 106, column: 5, scope: !53)
// CHECK:STDOUT: !61 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ba9a17e320476215:OptionalAs.7eef99cd03a74de4.Core.20b091e87e8b790e", scope: null, file: !54, line: 80, type: !55, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !62)
// CHECK:STDOUT: !62 = !{!63}
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !61, type: !14)
// CHECK:STDOUT: !64 = !DILocation(line: 81, column: 12, scope: !61)
// CHECK:STDOUT: !65 = !DILocation(line: 81, column: 5, scope: !61)
// CHECK:STDOUT: !66 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.20b091e87e8b790e", scope: null, file: !54, line: 33, type: !55, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !67)
// CHECK:STDOUT: !67 = !{!68}
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 1, scope: !66, type: !14)
// CHECK:STDOUT: !69 = !DILocation(line: 34, column: 12, scope: !66)
// CHECK:STDOUT: !70 = !DILocation(line: 34, column: 5, scope: !66)
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.e8f8f92d3d08d149:OptionalStorage.Core.b88531aa40419bf2", scope: null, file: !54, line: 166, type: !55, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !72)
// CHECK:STDOUT: !72 = !{!73}
// CHECK:STDOUT: !73 = !DILocalVariable(arg: 1, scope: !71, type: !14)
// CHECK:STDOUT: !74 = !DILocation(line: 167, column: 14, scope: !71)
// CHECK:STDOUT: !75 = !DILocation(line: 169, column: 5, scope: !71)
// CHECK:STDOUT: !76 = !DILocation(line: 167, column: 18, scope: !71)
// CHECK:STDOUT: !77 = !DILocation(line: 170, column: 5, scope: !71)
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
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.6391a65a70757668:ImplicitAs.116c0b3f4fb552e2.Core.e9a39e86438b926b"(ptr %.loc9_21.3), !dbg !14
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc9_21.5.temp), !dbg !14
// CHECK:STDOUT:   store ptr %U.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc9_21.5.temp, align 8, !dbg !14
// CHECK:STDOUT:   %.loc9_21.7 = load ptr, ptr %.loc9_21.5.temp, align 8, !dbg !14
// CHECK:STDOUT:   call void @_Z4takePK1C(ptr %.loc9_21.7), !dbg !15
// CHECK:STDOUT:   call void @"_COp.269ea07577c64bb2:core.Destroy.Core"(ptr %.loc9_21.5.temp), !dbg !14
// CHECK:STDOUT:   call void @"_COp.d665875966a4162d:core.Destroy.Core"(ptr %.loc9_21.1.temp), !dbg !14
// CHECK:STDOUT:   ret void, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z4makev() #1
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z4takePK1C(ptr noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.4038455b5b2635f6:core.Destroy.Core"(ptr %self) #0 !dbg !17 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.269ea07577c64bb2:core.Destroy.Core"(ptr %self) #0 !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.b8799b2262fcbb9d:core.Destroy.Core"(ptr %self) #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.d665875966a4162d:core.Destroy.Core"(ptr %self) #0 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.6391a65a70757668:ImplicitAs.116c0b3f4fb552e2.Core.e9a39e86438b926b"(ptr %self) #0 !dbg !36 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.Optional.5dca1548b0a7c023.Core:OptionalAs.f5e7d4305ac1f327.Core.c757ec210d8fc687"(ptr %self), !dbg !42
// CHECK:STDOUT:   ret ptr %1, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.Optional.5dca1548b0a7c023.Core:OptionalAs.f5e7d4305ac1f327.Core.c757ec210d8fc687"(ptr %self) #0 !dbg !44 {
// CHECK:STDOUT:   %temp = alloca ptr, align 8, !dbg !47
// CHECK:STDOUT:   %temp1 = alloca ptr, align 8, !dbg !47
// CHECK:STDOUT:   %1 = call i1 @_CHasValue.Optional.Core.e8736bdbdea23abc(ptr %self), !dbg !48
// CHECK:STDOUT:   br i1 %1, label %2, label %7, !dbg !49
// CHECK:STDOUT:
// CHECK:STDOUT: 2:                                                ; preds = %0
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !47
// CHECK:STDOUT:   %3 = call ptr @_CGet.Optional.Core.e8736bdbdea23abc(ptr %self), !dbg !47
// CHECK:STDOUT:   store ptr %3, ptr %temp, align 8, !dbg !47
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp1), !dbg !47
// CHECK:STDOUT:   %4 = load ptr, ptr %temp, align 8, !dbg !47
// CHECK:STDOUT:   store ptr %4, ptr %temp1, align 8, !dbg !47
// CHECK:STDOUT:   %5 = load ptr, ptr %temp1, align 8, !dbg !47
// CHECK:STDOUT:   %6 = call ptr @_CSome.Optional.Core.cc4dde828948b100(ptr %5), !dbg !50
// CHECK:STDOUT:   ret ptr %6, !dbg !51
// CHECK:STDOUT:
// CHECK:STDOUT: 7:                                                ; preds = %0
// CHECK:STDOUT:   %8 = call ptr @_CNone.Optional.Core.cc4dde828948b100(), !dbg !52
// CHECK:STDOUT:   ret ptr %8, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.e8736bdbdea23abc(ptr %self) #0 !dbg !54 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.e8f8f92d3d08d149:OptionalStorage.Core.b88531aa40419bf2"(ptr %self), !dbg !57
// CHECK:STDOUT:   ret i1 %1, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CGet.Optional.Core.e8736bdbdea23abc(ptr %self) #0 !dbg !59 {
// CHECK:STDOUT:   %1 = call ptr @"_CGet.e8f8f92d3d08d149:OptionalStorage.Core.b88531aa40419bf2"(ptr %self), !dbg !62
// CHECK:STDOUT:   ret ptr %1, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CSome.Optional.Core.cc4dde828948b100(ptr %value) #0 !dbg !64 {
// CHECK:STDOUT:   %1 = call ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.0e66b20f8d4f0663"(ptr %value), !dbg !67
// CHECK:STDOUT:   ret ptr %1, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CNone.Optional.Core.cc4dde828948b100() #0 !dbg !69 {
// CHECK:STDOUT:   ret ptr null, !dbg !72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.e8f8f92d3d08d149:OptionalStorage.Core.b88531aa40419bf2"(ptr %value) #0 !dbg !73 {
// CHECK:STDOUT:   %1 = icmp eq ptr %value, null, !dbg !76
// CHECK:STDOUT:   %2 = xor i1 %1, true, !dbg !77
// CHECK:STDOUT:   ret i1 %2, !dbg !78
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CGet.e8f8f92d3d08d149:OptionalStorage.Core.b88531aa40419bf2"(ptr %value) #0 !dbg !79 {
// CHECK:STDOUT:   ret ptr %value, !dbg !82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.0e66b20f8d4f0663"(ptr %self) #0 !dbg !83 {
// CHECK:STDOUT:   %1 = alloca ptr, align 8, !dbg !86
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !86
// CHECK:STDOUT:   store ptr poison, ptr %1, align 8, !dbg !86
// CHECK:STDOUT:   store ptr %self, ptr %1, align 8, !dbg !87
// CHECK:STDOUT:   %2 = load ptr, ptr %1, align 8, !dbg !88
// CHECK:STDOUT:   call void @"_COp.4038455b5b2635f6:core.Destroy.Core"(ptr %1), !dbg !86
// CHECK:STDOUT:   ret ptr %2, !dbg !89
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 2, 4, 3 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
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
// CHECK:STDOUT: !6 = !DIFile(filename: "add_const.carbon", directory: "")
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Main", scope: null, file: !6, line: 8, type: !12, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{null}
// CHECK:STDOUT: !14 = !DILocation(line: 9, column: 12, scope: !11)
// CHECK:STDOUT: !15 = !DILocation(line: 9, column: 3, scope: !11)
// CHECK:STDOUT: !16 = !DILocation(line: 8, column: 1, scope: !11)
// CHECK:STDOUT: !17 = distinct !DISubprogram(name: "Op", linkageName: "_COp.4038455b5b2635f6:core.Destroy.Core", scope: null, file: !6, line: 9, type: !18, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !21)
// CHECK:STDOUT: !18 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !19 = !{null, !20}
// CHECK:STDOUT: !20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !21 = !{!22}
// CHECK:STDOUT: !22 = !DILocalVariable(arg: 1, scope: !17, type: !20)
// CHECK:STDOUT: !23 = !DILocation(line: 9, column: 12, scope: !17)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "Op", linkageName: "_COp.269ea07577c64bb2:core.Destroy.Core", scope: null, file: !6, line: 9, type: !18, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !25)
// CHECK:STDOUT: !25 = !{!26}
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 1, scope: !24, type: !20)
// CHECK:STDOUT: !27 = !DILocation(line: 9, column: 12, scope: !24)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "Op", linkageName: "_COp.b8799b2262fcbb9d:core.Destroy.Core", scope: null, file: !6, line: 9, type: !18, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !29)
// CHECK:STDOUT: !29 = !{!30}
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 1, scope: !28, type: !20)
// CHECK:STDOUT: !31 = !DILocation(line: 9, column: 12, scope: !28)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "Op", linkageName: "_COp.d665875966a4162d:core.Destroy.Core", scope: null, file: !6, line: 9, type: !18, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !33)
// CHECK:STDOUT: !33 = !{!34}
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 1, scope: !32, type: !20)
// CHECK:STDOUT: !35 = !DILocation(line: 9, column: 12, scope: !32)
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.6391a65a70757668:ImplicitAs.116c0b3f4fb552e2.Core.e9a39e86438b926b", scope: null, file: !37, line: 105, type: !38, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !40)
// CHECK:STDOUT: !37 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !38 = !DISubroutineType(types: !39)
// CHECK:STDOUT: !39 = !{!20, !20}
// CHECK:STDOUT: !40 = !{!41}
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !36, type: !20)
// CHECK:STDOUT: !42 = !DILocation(line: 106, column: 12, scope: !36)
// CHECK:STDOUT: !43 = !DILocation(line: 106, column: 5, scope: !36)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Optional.5dca1548b0a7c023.Core:OptionalAs.f5e7d4305ac1f327.Core.c757ec210d8fc687", scope: null, file: !37, line: 95, type: !38, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !45)
// CHECK:STDOUT: !45 = !{!46}
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 1, scope: !44, type: !20)
// CHECK:STDOUT: !47 = !DILocation(line: 97, column: 31, scope: !44)
// CHECK:STDOUT: !48 = !DILocation(line: 96, column: 9, scope: !44)
// CHECK:STDOUT: !49 = !DILocation(line: 96, column: 8, scope: !44)
// CHECK:STDOUT: !50 = !DILocation(line: 97, column: 14, scope: !44)
// CHECK:STDOUT: !51 = !DILocation(line: 97, column: 7, scope: !44)
// CHECK:STDOUT: !52 = !DILocation(line: 99, column: 12, scope: !44)
// CHECK:STDOUT: !53 = !DILocation(line: 99, column: 5, scope: !44)
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.e8736bdbdea23abc", scope: null, file: !37, line: 36, type: !38, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !55)
// CHECK:STDOUT: !55 = !{!56}
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !54, type: !20)
// CHECK:STDOUT: !57 = !DILocation(line: 37, column: 12, scope: !54)
// CHECK:STDOUT: !58 = !DILocation(line: 37, column: 5, scope: !54)
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.e8736bdbdea23abc", scope: null, file: !37, line: 39, type: !38, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !60)
// CHECK:STDOUT: !60 = !{!61}
// CHECK:STDOUT: !61 = !DILocalVariable(arg: 1, scope: !59, type: !20)
// CHECK:STDOUT: !62 = !DILocation(line: 40, column: 12, scope: !59)
// CHECK:STDOUT: !63 = !DILocation(line: 40, column: 5, scope: !59)
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.cc4dde828948b100", scope: null, file: !37, line: 33, type: !38, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !65)
// CHECK:STDOUT: !65 = !{!66}
// CHECK:STDOUT: !66 = !DILocalVariable(arg: 1, scope: !64, type: !20)
// CHECK:STDOUT: !67 = !DILocation(line: 34, column: 12, scope: !64)
// CHECK:STDOUT: !68 = !DILocation(line: 34, column: 5, scope: !64)
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.cc4dde828948b100", scope: null, file: !37, line: 30, type: !70, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !70 = !DISubroutineType(types: !71)
// CHECK:STDOUT: !71 = !{!20}
// CHECK:STDOUT: !72 = !DILocation(line: 31, column: 5, scope: !69)
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.e8f8f92d3d08d149:OptionalStorage.Core.b88531aa40419bf2", scope: null, file: !37, line: 172, type: !38, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !74)
// CHECK:STDOUT: !74 = !{!75}
// CHECK:STDOUT: !75 = !DILocalVariable(arg: 1, scope: !73, type: !20)
// CHECK:STDOUT: !76 = !DILocation(line: 173, column: 16, scope: !73)
// CHECK:STDOUT: !77 = !DILocation(line: 173, column: 12, scope: !73)
// CHECK:STDOUT: !78 = !DILocation(line: 173, column: 5, scope: !73)
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.e8f8f92d3d08d149:OptionalStorage.Core.b88531aa40419bf2", scope: null, file: !37, line: 175, type: !38, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !80)
// CHECK:STDOUT: !80 = !{!81}
// CHECK:STDOUT: !81 = !DILocalVariable(arg: 1, scope: !79, type: !20)
// CHECK:STDOUT: !82 = !DILocation(line: 176, column: 5, scope: !79)
// CHECK:STDOUT: !83 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.e8f8f92d3d08d149:OptionalStorage.Core.0e66b20f8d4f0663", scope: null, file: !37, line: 166, type: !38, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !84)
// CHECK:STDOUT: !84 = !{!85}
// CHECK:STDOUT: !85 = !DILocalVariable(arg: 1, scope: !83, type: !20)
// CHECK:STDOUT: !86 = !DILocation(line: 167, column: 14, scope: !83)
// CHECK:STDOUT: !87 = !DILocation(line: 169, column: 5, scope: !83)
// CHECK:STDOUT: !88 = !DILocation(line: 167, column: 18, scope: !83)
// CHECK:STDOUT: !89 = !DILocation(line: 170, column: 5, scope: !83)
