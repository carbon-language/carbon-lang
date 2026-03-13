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

// CHECK:STDOUT: ; ModuleID = 'nonnull.carbon'
// CHECK:STDOUT: source_filename = "nonnull.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define dso_local void @_Z16TakePtrWithThunkP1Ci.carbon_thunk1(ptr noundef %0) #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !11
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !11
// CHECK:STDOUT:   call void @_Z16TakePtrWithThunkP1Ci(ptr noundef %1, i32 noundef 0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z16TakePtrWithThunkP1Ci(ptr noundef, i32 noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define dso_local noundef ptr @_Z18ReturnPtrWithThunki.carbon_thunk0() #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %call = call noundef ptr @_Z18ReturnPtrWithThunki(i32 noundef 0)
// CHECK:STDOUT:   ret ptr %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z18ReturnPtrWithThunki(i32 noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassPtr.Main(ptr %p) #2 !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z7TakePtrP1C(ptr %p), !dbg !20
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z7TakePtrP1C(ptr noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnPtr.Main() #2 !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnPtr.call = call ptr @_Z9ReturnPtrv(), !dbg !25
// CHECK:STDOUT:   ret ptr %ReturnPtr.call, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z9ReturnPtrv() #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassPtrWithThunk.Main(ptr %p) #2 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z16TakePtrWithThunkP1Ci.carbon_thunk1(ptr %p), !dbg !30
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnPtrWithThunk.Main() #2 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnPtrWithThunk__carbon_thunk.call = call ptr @_Z18ReturnPtrWithThunki.carbon_thunk0(), !dbg !33
// CHECK:STDOUT:   ret ptr %ReturnPtrWithThunk__carbon_thunk.call, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nounwind }
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
// CHECK:STDOUT: !11 = !{!12, !12, i64 0}
// CHECK:STDOUT: !12 = !{!"p1 _ZTS1C", !13, i64 0}
// CHECK:STDOUT: !13 = !{!"any pointer", !9, i64 0}
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "PassPtr", linkageName: "_CPassPtr.Main", scope: null, file: !6, line: 14, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !18)
// CHECK:STDOUT: !15 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !16 = !{null, !17}
// CHECK:STDOUT: !17 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 8)
// CHECK:STDOUT: !18 = !{!19}
// CHECK:STDOUT: !19 = !DILocalVariable(arg: 1, scope: !14, type: !17)
// CHECK:STDOUT: !20 = !DILocation(line: 15, column: 3, scope: !14)
// CHECK:STDOUT: !21 = !DILocation(line: 14, column: 1, scope: !14)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "ReturnPtr", linkageName: "_CReturnPtr.Main", scope: null, file: !6, line: 18, type: !23, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !23 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !24 = !{!17}
// CHECK:STDOUT: !25 = !DILocation(line: 19, column: 10, scope: !22)
// CHECK:STDOUT: !26 = !DILocation(line: 19, column: 3, scope: !22)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "PassPtrWithThunk", linkageName: "_CPassPtrWithThunk.Main", scope: null, file: !6, line: 22, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !28)
// CHECK:STDOUT: !28 = !{!29}
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 1, scope: !27, type: !17)
// CHECK:STDOUT: !30 = !DILocation(line: 23, column: 3, scope: !27)
// CHECK:STDOUT: !31 = !DILocation(line: 22, column: 1, scope: !27)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "ReturnPtrWithThunk", linkageName: "_CReturnPtrWithThunk.Main", scope: null, file: !6, line: 26, type: !23, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !33 = !DILocation(line: 27, column: 10, scope: !32)
// CHECK:STDOUT: !34 = !DILocation(line: 27, column: 3, scope: !32)
// CHECK:STDOUT: ; ModuleID = 'nullable.carbon'
// CHECK:STDOUT: source_filename = "nullable.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define dso_local void @_Z16TakePtrWithThunkP1Ci.carbon_thunk1(ptr noundef %0) #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !11
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !11
// CHECK:STDOUT:   call void @_Z16TakePtrWithThunkP1Ci(ptr noundef %1, i32 noundef 0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z16TakePtrWithThunkP1Ci(ptr noundef, i32 noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define dso_local noundef ptr @_Z18ReturnPtrWithThunki.carbon_thunk0() #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %call = call noundef ptr @_Z18ReturnPtrWithThunki(i32 noundef 0)
// CHECK:STDOUT:   ret ptr %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z18ReturnPtrWithThunki(i32 noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassPtr.Main(ptr %_) #2 !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassNonnullPtr.Main(ptr %p) #2 !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc22_15.2.temp = alloca ptr, align 8, !dbg !24
// CHECK:STDOUT:   %U.binding.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.e76fd6d2be138e4a"(ptr %p), !dbg !24
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc22_15.2.temp), !dbg !24
// CHECK:STDOUT:   store ptr %U.binding.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc22_15.2.temp, align 8, !dbg !24
// CHECK:STDOUT:   %.loc22_15.4 = load ptr, ptr %.loc22_15.2.temp, align 8, !dbg !24
// CHECK:STDOUT:   call void @_Z7TakePtrP1C(ptr %.loc22_15.4), !dbg !25
// CHECK:STDOUT:   ret void, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z7TakePtrP1C(ptr noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnPtr.Main() #2 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnPtr.call = call ptr @_Z9ReturnPtrv(), !dbg !30
// CHECK:STDOUT:   ret ptr %ReturnPtr.call, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z9ReturnPtrv() #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassPtrWithThunk.Main(ptr %_) #2 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassNonnullPtrWithThunk.Main(ptr %p) #2 !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc35_24.2.temp = alloca ptr, align 8, !dbg !39
// CHECK:STDOUT:   %U.binding.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.e76fd6d2be138e4a"(ptr %p), !dbg !39
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc35_24.2.temp), !dbg !39
// CHECK:STDOUT:   store ptr %U.binding.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc35_24.2.temp, align 8, !dbg !39
// CHECK:STDOUT:   %.loc35_24.4 = load ptr, ptr %.loc35_24.2.temp, align 8, !dbg !39
// CHECK:STDOUT:   call void @_Z16TakePtrWithThunkP1Ci.carbon_thunk1(ptr %.loc35_24.4), !dbg !40
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnPtrWithThunk.Main() #2 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnPtrWithThunk__carbon_thunk.call = call ptr @_Z18ReturnPtrWithThunki.carbon_thunk0(), !dbg !43
// CHECK:STDOUT:   ret ptr %ReturnPtrWithThunk__carbon_thunk.call, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.e76fd6d2be138e4a"(ptr %self) #2 !dbg !45 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.90961d7b1ce4f089:OptionalAs.0e326e799dad0c64.Core.8f431b36581f9e63"(ptr %self), !dbg !51
// CHECK:STDOUT:   ret ptr %1, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.90961d7b1ce4f089:OptionalAs.0e326e799dad0c64.Core.8f431b36581f9e63"(ptr %self) #2 !dbg !53 {
// CHECK:STDOUT:   %1 = call ptr @_CSome.Optional.Core.8f431b36581f9e63(ptr %self), !dbg !56
// CHECK:STDOUT:   ret ptr %1, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CSome.Optional.Core.8f431b36581f9e63(ptr %value) #2 !dbg !58 {
// CHECK:STDOUT:   %1 = call ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.f53db17714b9f655"(ptr %value), !dbg !61
// CHECK:STDOUT:   ret ptr %1, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.f53db17714b9f655"(ptr %self) #2 !dbg !63 {
// CHECK:STDOUT:   %1 = alloca ptr, align 8, !dbg !66
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !66
// CHECK:STDOUT:   store ptr poison, ptr %1, align 8, !dbg !66
// CHECK:STDOUT:   store ptr %self, ptr %1, align 8, !dbg !67
// CHECK:STDOUT:   %2 = load ptr, ptr %1, align 8, !dbg !68
// CHECK:STDOUT:   ret ptr %2, !dbg !69
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.e76fd6d2be138e4a", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 2, 1 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nounwind }
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
// CHECK:STDOUT: !11 = !{!12, !12, i64 0}
// CHECK:STDOUT: !12 = !{!"p1 _ZTS1C", !13, i64 0}
// CHECK:STDOUT: !13 = !{!"any pointer", !9, i64 0}
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "PassPtr", linkageName: "_CPassPtr.Main", scope: null, file: !6, line: 14, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !18)
// CHECK:STDOUT: !15 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !16 = !{null, !17}
// CHECK:STDOUT: !17 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 8)
// CHECK:STDOUT: !18 = !{!19}
// CHECK:STDOUT: !19 = !DILocalVariable(arg: 1, scope: !14, type: !17)
// CHECK:STDOUT: !20 = !DILocation(line: 14, column: 1, scope: !14)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "PassNonnullPtr", linkageName: "_CPassNonnullPtr.Main", scope: null, file: !6, line: 19, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !22)
// CHECK:STDOUT: !22 = !{!23}
// CHECK:STDOUT: !23 = !DILocalVariable(arg: 1, scope: !21, type: !17)
// CHECK:STDOUT: !24 = !DILocation(line: 22, column: 15, scope: !21)
// CHECK:STDOUT: !25 = !DILocation(line: 22, column: 3, scope: !21)
// CHECK:STDOUT: !26 = !DILocation(line: 19, column: 1, scope: !21)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "ReturnPtr", linkageName: "_CReturnPtr.Main", scope: null, file: !6, line: 25, type: !28, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !28 = !DISubroutineType(types: !29)
// CHECK:STDOUT: !29 = !{!17}
// CHECK:STDOUT: !30 = !DILocation(line: 26, column: 10, scope: !27)
// CHECK:STDOUT: !31 = !DILocation(line: 26, column: 3, scope: !27)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "PassPtrWithThunk", linkageName: "_CPassPtrWithThunk.Main", scope: null, file: !6, line: 29, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !33)
// CHECK:STDOUT: !33 = !{!34}
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 1, scope: !32, type: !17)
// CHECK:STDOUT: !35 = !DILocation(line: 29, column: 1, scope: !32)
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "PassNonnullPtrWithThunk", linkageName: "_CPassNonnullPtrWithThunk.Main", scope: null, file: !6, line: 34, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !37)
// CHECK:STDOUT: !37 = !{!38}
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 1, scope: !36, type: !17)
// CHECK:STDOUT: !39 = !DILocation(line: 35, column: 24, scope: !36)
// CHECK:STDOUT: !40 = !DILocation(line: 35, column: 3, scope: !36)
// CHECK:STDOUT: !41 = !DILocation(line: 34, column: 1, scope: !36)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "ReturnPtrWithThunk", linkageName: "_CReturnPtrWithThunk.Main", scope: null, file: !6, line: 38, type: !28, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !43 = !DILocation(line: 39, column: 10, scope: !42)
// CHECK:STDOUT: !44 = !DILocation(line: 39, column: 3, scope: !42)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.e5cf8fcbb4feaae2:ImplicitAs.0f95c9e18c91e00a.Core.e76fd6d2be138e4a", scope: null, file: !46, line: 96, type: !47, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !49)
// CHECK:STDOUT: !46 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !47 = !DISubroutineType(types: !48)
// CHECK:STDOUT: !48 = !{!17, !17}
// CHECK:STDOUT: !49 = !{!50}
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 1, scope: !45, type: !17)
// CHECK:STDOUT: !51 = !DILocation(line: 97, column: 12, scope: !45)
// CHECK:STDOUT: !52 = !DILocation(line: 97, column: 5, scope: !45)
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.90961d7b1ce4f089:OptionalAs.0e326e799dad0c64.Core.8f431b36581f9e63", scope: null, file: !46, line: 71, type: !47, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !54)
// CHECK:STDOUT: !54 = !{!55}
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 1, scope: !53, type: !17)
// CHECK:STDOUT: !56 = !DILocation(line: 72, column: 12, scope: !53)
// CHECK:STDOUT: !57 = !DILocation(line: 72, column: 5, scope: !53)
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.8f431b36581f9e63", scope: null, file: !46, line: 30, type: !47, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !59)
// CHECK:STDOUT: !59 = !{!60}
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 1, scope: !58, type: !17)
// CHECK:STDOUT: !61 = !DILocation(line: 31, column: 12, scope: !58)
// CHECK:STDOUT: !62 = !DILocation(line: 31, column: 5, scope: !58)
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.e8f8f92d3d08d149:OptionalStorage.Core.f53db17714b9f655", scope: null, file: !46, line: 150, type: !47, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !64)
// CHECK:STDOUT: !64 = !{!65}
// CHECK:STDOUT: !65 = !DILocalVariable(arg: 1, scope: !63, type: !17)
// CHECK:STDOUT: !66 = !DILocation(line: 151, column: 14, scope: !63)
// CHECK:STDOUT: !67 = !DILocation(line: 153, column: 5, scope: !63)
// CHECK:STDOUT: !68 = !DILocation(line: 151, column: 18, scope: !63)
// CHECK:STDOUT: !69 = !DILocation(line: 154, column: 5, scope: !63)
