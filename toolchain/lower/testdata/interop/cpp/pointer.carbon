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
// CHECK:STDOUT: define void @_CPassPtr.Main(ptr %p) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z14TakeNonnullPtrP8NonnullC(ptr %p), !dbg !15
// CHECK:STDOUT:   ret void, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z14TakeNonnullPtrP8NonnullC(ptr noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnPtr.Main() #0 !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnNonnullPtr.call = call ptr @_Z16ReturnNonnullPtrv(), !dbg !21
// CHECK:STDOUT:   ret ptr %ReturnNonnullPtr.call, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z16ReturnNonnullPtrv() #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassPtrWithThunk.Main(ptr %p) #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z23TakeNonnullPtrWithThunkP8NonnullCi.carbon_thunk._(ptr %p), !dbg !27
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z23TakeNonnullPtrWithThunkP8NonnullCi.carbon_thunk._(ptr noundef %0) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !39
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !39
// CHECK:STDOUT:   call void @_Z23TakeNonnullPtrWithThunkP8NonnullCi(ptr noundef %1, i32 noundef 0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnPtrWithThunk.Main() #0 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnNonnullPtrWithThunk__carbon_thunk.call = call ptr @_Z25ReturnNonnullPtrWithThunki.carbon_thunk.(), !dbg !33
// CHECK:STDOUT:   ret ptr %ReturnNonnullPtrWithThunk__carbon_thunk.call, !dbg !34
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!40, !41, !42, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!45}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "nonnull.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !7 = !{null, !6}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !9 = !{!6}
// CHECK:STDOUT: !10 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "PassPtr", linkageName: "_CPassPtr.Main", scope: null, file: !4, line: 14, type: !8, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !17)
// CHECK:STDOUT: !12 = !DILocalVariable(arg: 1, scope: !11, type: !6)
// CHECK:STDOUT: !15 = !DILocation(line: 15, column: 3, scope: !11)
// CHECK:STDOUT: !16 = !DILocation(line: 14, column: 1, scope: !11)
// CHECK:STDOUT: !17 = !{!12}
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "ReturnPtr", linkageName: "_CReturnPtr.Main", scope: null, file: !4, line: 18, type: !10, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !21 = !DILocation(line: 19, column: 10, scope: !18)
// CHECK:STDOUT: !22 = !DILocation(line: 19, column: 3, scope: !18)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "PassPtrWithThunk", linkageName: "_CPassPtrWithThunk.Main", scope: null, file: !4, line: 22, type: !8, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !29)
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !23, type: !6)
// CHECK:STDOUT: !27 = !DILocation(line: 23, column: 3, scope: !23)
// CHECK:STDOUT: !28 = !DILocation(line: 22, column: 1, scope: !23)
// CHECK:STDOUT: !29 = !{!24}
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "ReturnPtrWithThunk", linkageName: "_CReturnPtrWithThunk.Main", scope: null, file: !4, line: 26, type: !10, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !33 = !DILocation(line: 27, column: 10, scope: !30)
// CHECK:STDOUT: !34 = !DILocation(line: 27, column: 3, scope: !30)
// CHECK:STDOUT: !35 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !36 = !{!"omnipotent char", !35, i64 0}
// CHECK:STDOUT: !37 = !{!"any pointer", !36, i64 0}
// CHECK:STDOUT: !38 = !{!"p1 _ZTS8NonnullC", !37, i64 0}
// CHECK:STDOUT: !39 = !{!38, !38, i64 0}
// CHECK:STDOUT: !40 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !41 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !42 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !43 = !{!"int", !36, i64 0}
// CHECK:STDOUT: !44 = !{!"__libc_errno", !43, i64 0}
// CHECK:STDOUT: !45 = !{!44, !43, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'nullable.carbon'
// CHECK:STDOUT: source_filename = "nullable.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassPtr.Main(ptr %_) #0 !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassNonnullPtr.Main(ptr %p) #0 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc22_23.2.temp = alloca ptr, align 8, !dbg !56
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.75a9143bcbb84fa5"(ptr %p), !dbg !56
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc22_23.2.temp), !dbg !56
// CHECK:STDOUT:   store ptr %U.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc22_23.2.temp, align 8, !dbg !56
// CHECK:STDOUT:   %.loc22_23.4 = load ptr, ptr %.loc22_23.2.temp, align 8, !dbg !56
// CHECK:STDOUT:   call void @_Z15TakeNullablePtrP9NullableC(ptr %.loc22_23.4), !dbg !59
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.7716fb2615abd5da(ptr %.loc22_23.2.temp), !dbg !56
// CHECK:STDOUT:   ret void, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z15TakeNullablePtrP9NullableC(ptr noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.6789a803c53632c0:core.Destroy.Core"(ptr %self) #0 !dbg !62 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.55deb2364e2caa71:core.Destroy.Core"(ptr %self) #0 !dbg !66 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.574ea10b2e6709a0:core.Destroy.Core"(ptr %self) #0 !dbg !70 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnPtr.Main() #0 !dbg !74 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnNullablePtr.call = call ptr @_Z17ReturnNullablePtrv(), !dbg !77
// CHECK:STDOUT:   ret ptr %ReturnNullablePtr.call, !dbg !78
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z17ReturnNullablePtrv() #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassPtrWithThunk.Main(ptr %_) #0 !dbg !79 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassNonnullPtrWithThunk.Main(ptr %p) #0 !dbg !84 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc35_32.2.temp = alloca ptr, align 8, !dbg !87
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.75a9143bcbb84fa5"(ptr %p), !dbg !87
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc35_32.2.temp), !dbg !87
// CHECK:STDOUT:   store ptr %U.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc35_32.2.temp, align 8, !dbg !87
// CHECK:STDOUT:   %.loc35_32.4 = load ptr, ptr %.loc35_32.2.temp, align 8, !dbg !87
// CHECK:STDOUT:   call void @_Z24TakeNullablePtrWithThunkP9NullableCi.carbon_thunk._(ptr %.loc35_32.4), !dbg !88
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.7716fb2615abd5da(ptr %.loc35_32.2.temp), !dbg !87
// CHECK:STDOUT:   ret void, !dbg !89
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z24TakeNullablePtrWithThunkP9NullableCi.carbon_thunk._(ptr noundef %0) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !151
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !151
// CHECK:STDOUT:   call void @_Z24TakeNullablePtrWithThunkP9NullableCi(ptr noundef %1, i32 noundef 0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnPtrWithThunk.Main() #0 !dbg !91 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnNullablePtrWithThunk__carbon_thunk.call = call ptr @_Z26ReturnNullablePtrWithThunki.carbon_thunk.(), !dbg !94
// CHECK:STDOUT:   ret ptr %ReturnNullablePtrWithThunk__carbon_thunk.call, !dbg !95
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
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.75a9143bcbb84fa5"(ptr %self) #0 !dbg !97 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.e83c25d1d2fd22c2"(ptr %self), !dbg !102
// CHECK:STDOUT:   ret ptr %1, !dbg !103
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.7716fb2615abd5da(ptr %self) #0 !dbg !106 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.574ea10b2e6709a0:core.Destroy.Core"(ptr %self), !dbg !110
// CHECK:STDOUT:   ret void, !dbg !111
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.e83c25d1d2fd22c2"(ptr %self) #0 !dbg !113 {
// CHECK:STDOUT:   %1 = call ptr @_CSome.Optional.Core.e83c25d1d2fd22c2(ptr %self), !dbg !119
// CHECK:STDOUT:   ret ptr %1, !dbg !120
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CSome.Optional.Core.e83c25d1d2fd22c2(ptr %value) #0 !dbg !122 {
// CHECK:STDOUT:   %1 = call ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.77dd513e5db6ef8d"(ptr %value), !dbg !127
// CHECK:STDOUT:   ret ptr %1, !dbg !128
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.77dd513e5db6ef8d"(ptr %self) #0 !dbg !130 {
// CHECK:STDOUT:   %1 = alloca ptr, align 8, !dbg !135
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !135
// CHECK:STDOUT:   store ptr poison, ptr %1, align 8, !dbg !135
// CHECK:STDOUT:   store ptr %self, ptr %1, align 8, !dbg !139
// CHECK:STDOUT:   %2 = load ptr, ptr %1, align 8, !dbg !137
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.4ffaeb16aa777cfd(ptr %1), !dbg !135
// CHECK:STDOUT:   ret ptr %2, !dbg !141
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.4ffaeb16aa777cfd(ptr %self) #0 !dbg !143 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.55deb2364e2caa71:core.Destroy.Core"(ptr %self), !dbg !147
// CHECK:STDOUT:   ret void, !dbg !148
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z24TakeNullablePtrWithThunkP9NullableCi(ptr noundef, i32 noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z26ReturnNullablePtrWithThunki(i32 noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.75a9143bcbb84fa5", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 2, 1 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.7716fb2615abd5da, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!47}
// CHECK:STDOUT: !llvm.module.flags = !{!40, !41, !42, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!45}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !7 = !{null, !6}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !9 = !{!6}
// CHECK:STDOUT: !10 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !35 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !36 = !{!"omnipotent char", !35, i64 0}
// CHECK:STDOUT: !37 = !{!"any pointer", !36, i64 0}
// CHECK:STDOUT: !40 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !41 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !42 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !43 = !{!"int", !36, i64 0}
// CHECK:STDOUT: !44 = !{!"__libc_errno", !43, i64 0}
// CHECK:STDOUT: !45 = !{!44, !43, i64 0}
// CHECK:STDOUT: !46 = !DIFile(filename: "nullable.carbon", directory: "")
// CHECK:STDOUT: !47 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !46, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "PassPtr", linkageName: "_CPassPtr.Main", scope: null, file: !46, line: 14, type: !8, spFlags: DISPFlagDefinition, unit: !47, retainedNodes: !52)
// CHECK:STDOUT: !49 = !DILocalVariable(arg: 1, scope: !48, type: !6)
// CHECK:STDOUT: !51 = !DILocation(line: 14, column: 1, scope: !48)
// CHECK:STDOUT: !52 = !{!49}
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "PassNonnullPtr", linkageName: "_CPassNonnullPtr.Main", scope: null, file: !46, line: 19, type: !8, spFlags: DISPFlagDefinition, unit: !47, retainedNodes: !61)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !53, type: !6)
// CHECK:STDOUT: !56 = !DILocation(line: 22, column: 23, scope: !53)
// CHECK:STDOUT: !57 = !{!6, !6}
// CHECK:STDOUT: !58 = !DISubroutineType(types: !57)
// CHECK:STDOUT: !59 = !DILocation(line: 22, column: 3, scope: !53)
// CHECK:STDOUT: !60 = !DILocation(line: 19, column: 1, scope: !53)
// CHECK:STDOUT: !61 = !{!54}
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.6789a803c53632c0:core.Destroy.Core", scope: null, file: !46, line: 22, type: !8, spFlags: DISPFlagDefinition, unit: !47, retainedNodes: !65)
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !62, type: !6)
// CHECK:STDOUT: !64 = !DILocation(line: 22, column: 23, scope: !62)
// CHECK:STDOUT: !65 = !{!63}
// CHECK:STDOUT: !66 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.55deb2364e2caa71:core.Destroy.Core", scope: null, file: !46, line: 22, type: !8, spFlags: DISPFlagDefinition, unit: !47, retainedNodes: !69)
// CHECK:STDOUT: !67 = !DILocalVariable(arg: 1, scope: !66, type: !6)
// CHECK:STDOUT: !68 = !DILocation(line: 22, column: 23, scope: !66)
// CHECK:STDOUT: !69 = !{!67}
// CHECK:STDOUT: !70 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.574ea10b2e6709a0:core.Destroy.Core", scope: null, file: !46, line: 22, type: !8, spFlags: DISPFlagDefinition, unit: !47, retainedNodes: !73)
// CHECK:STDOUT: !71 = !DILocalVariable(arg: 1, scope: !70, type: !6)
// CHECK:STDOUT: !72 = !DILocation(line: 22, column: 23, scope: !70)
// CHECK:STDOUT: !73 = !{!71}
// CHECK:STDOUT: !74 = distinct !DISubprogram(name: "ReturnPtr", linkageName: "_CReturnPtr.Main", scope: null, file: !46, line: 25, type: !10, spFlags: DISPFlagDefinition, unit: !47)
// CHECK:STDOUT: !77 = !DILocation(line: 26, column: 10, scope: !74)
// CHECK:STDOUT: !78 = !DILocation(line: 26, column: 3, scope: !74)
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "PassPtrWithThunk", linkageName: "_CPassPtrWithThunk.Main", scope: null, file: !46, line: 29, type: !8, spFlags: DISPFlagDefinition, unit: !47, retainedNodes: !83)
// CHECK:STDOUT: !80 = !DILocalVariable(arg: 1, scope: !79, type: !6)
// CHECK:STDOUT: !82 = !DILocation(line: 29, column: 1, scope: !79)
// CHECK:STDOUT: !83 = !{!80}
// CHECK:STDOUT: !84 = distinct !DISubprogram(name: "PassNonnullPtrWithThunk", linkageName: "_CPassNonnullPtrWithThunk.Main", scope: null, file: !46, line: 34, type: !8, spFlags: DISPFlagDefinition, unit: !47, retainedNodes: !90)
// CHECK:STDOUT: !85 = !DILocalVariable(arg: 1, scope: !84, type: !6)
// CHECK:STDOUT: !87 = !DILocation(line: 35, column: 32, scope: !84)
// CHECK:STDOUT: !88 = !DILocation(line: 35, column: 3, scope: !84)
// CHECK:STDOUT: !89 = !DILocation(line: 34, column: 1, scope: !84)
// CHECK:STDOUT: !90 = !{!85}
// CHECK:STDOUT: !91 = distinct !DISubprogram(name: "ReturnPtrWithThunk", linkageName: "_CReturnPtrWithThunk.Main", scope: null, file: !46, line: 38, type: !10, spFlags: DISPFlagDefinition, unit: !47)
// CHECK:STDOUT: !94 = !DILocation(line: 39, column: 10, scope: !91)
// CHECK:STDOUT: !95 = !DILocation(line: 39, column: 3, scope: !91)
// CHECK:STDOUT: !96 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.75a9143bcbb84fa5", scope: null, file: !96, line: 105, type: !58, spFlags: DISPFlagDefinition, unit: !47, retainedNodes: !104)
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 1, scope: !97, type: !6)
// CHECK:STDOUT: !102 = !DILocation(line: 106, column: 12, scope: !97)
// CHECK:STDOUT: !103 = !DILocation(line: 106, column: 5, scope: !97)
// CHECK:STDOUT: !104 = !{!98}
// CHECK:STDOUT: !105 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !106 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.7716fb2615abd5da", scope: null, file: !105, line: 26, type: !8, spFlags: DISPFlagDefinition, unit: !47, retainedNodes: !112)
// CHECK:STDOUT: !107 = !DILocalVariable(arg: 1, scope: !106, type: !6)
// CHECK:STDOUT: !110 = !DILocation(line: 28, column: 5, scope: !106)
// CHECK:STDOUT: !111 = !DILocation(line: 26, column: 3, scope: !106)
// CHECK:STDOUT: !112 = !{!107}
// CHECK:STDOUT: !113 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.e83c25d1d2fd22c2", scope: null, file: !96, line: 80, type: !58, spFlags: DISPFlagDefinition, unit: !47, retainedNodes: !121)
// CHECK:STDOUT: !114 = !DILocalVariable(arg: 1, scope: !113, type: !6)
// CHECK:STDOUT: !119 = !DILocation(line: 81, column: 12, scope: !113)
// CHECK:STDOUT: !120 = !DILocation(line: 81, column: 5, scope: !113)
// CHECK:STDOUT: !121 = !{!114}
// CHECK:STDOUT: !122 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.e83c25d1d2fd22c2", scope: null, file: !96, line: 33, type: !58, spFlags: DISPFlagDefinition, unit: !47, retainedNodes: !129)
// CHECK:STDOUT: !123 = !DILocalVariable(arg: 1, scope: !122, type: !6)
// CHECK:STDOUT: !127 = !DILocation(line: 34, column: 12, scope: !122)
// CHECK:STDOUT: !128 = !DILocation(line: 34, column: 5, scope: !122)
// CHECK:STDOUT: !129 = !{!123}
// CHECK:STDOUT: !130 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.e8f8f92d3d08d149:OptionalStorage.Core.77dd513e5db6ef8d", scope: null, file: !96, line: 166, type: !58, spFlags: DISPFlagDefinition, unit: !47, retainedNodes: !142)
// CHECK:STDOUT: !131 = !DILocalVariable(arg: 1, scope: !130, type: !6)
// CHECK:STDOUT: !135 = !DILocation(line: 167, column: 14, scope: !130)
// CHECK:STDOUT: !137 = !DILocation(line: 167, column: 18, scope: !130)
// CHECK:STDOUT: !139 = !DILocation(line: 169, column: 5, scope: !130)
// CHECK:STDOUT: !141 = !DILocation(line: 170, column: 5, scope: !130)
// CHECK:STDOUT: !142 = !{!131}
// CHECK:STDOUT: !143 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.4ffaeb16aa777cfd", scope: null, file: !105, line: 26, type: !8, spFlags: DISPFlagDefinition, unit: !47, retainedNodes: !149)
// CHECK:STDOUT: !144 = !DILocalVariable(arg: 1, scope: !143, type: !6)
// CHECK:STDOUT: !147 = !DILocation(line: 28, column: 5, scope: !143)
// CHECK:STDOUT: !148 = !DILocation(line: 26, column: 3, scope: !143)
// CHECK:STDOUT: !149 = !{!144}
// CHECK:STDOUT: !150 = !{!"p1 _ZTS9NullableC", !37, i64 0}
// CHECK:STDOUT: !151 = !{!150, !150, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'add_const.carbon'
// CHECK:STDOUT: source_filename = "add_const.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvert.Main() #0 !dbg !156 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_21.1.temp = alloca ptr, align 8, !dbg !157
// CHECK:STDOUT:   %.loc11_21.5.temp = alloca ptr, align 8, !dbg !157
// CHECK:STDOUT:   %make.call = call ptr @_Z4makev(), !dbg !157
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc11_21.1.temp), !dbg !157
// CHECK:STDOUT:   store ptr %make.call, ptr %.loc11_21.1.temp, align 8, !dbg !157
// CHECK:STDOUT:   %.loc11_21.3 = load ptr, ptr %.loc11_21.1.temp, align 8, !dbg !157
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.7c8d13c8a74a288d"(ptr %.loc11_21.3), !dbg !157
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc11_21.5.temp), !dbg !157
// CHECK:STDOUT:   store ptr %U.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc11_21.5.temp, align 8, !dbg !157
// CHECK:STDOUT:   %.loc11_21.7 = load ptr, ptr %.loc11_21.5.temp, align 8, !dbg !157
// CHECK:STDOUT:   call void @_Z4takePK6ConstC(ptr %.loc11_21.7), !dbg !158
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.65dfedfaf9fb26a5(ptr %.loc11_21.5.temp), !dbg !157
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.b03cd22c39029320(ptr %.loc11_21.1.temp), !dbg !157
// CHECK:STDOUT:   ret void, !dbg !159
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z4makev() #1
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z4takePK6ConstC(ptr noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.bdb5ffe218b7af7d:core.Destroy.Core"(ptr %self) #0 !dbg !160 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !162
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.4ef7d32bafd75f98:core.Destroy.Core"(ptr %self) #0 !dbg !164 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !166
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.19661d0961df889d:core.Destroy.Core"(ptr %self) #0 !dbg !168 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !170
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.e3602e7640dfb844:core.Destroy.Core"(ptr %self) #0 !dbg !172 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !174
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.02018f16aebe62a8:core.Destroy.Core"(ptr %self) #0 !dbg !176 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !178
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.78fd6ee82eacbbc3:core.Destroy.Core"(ptr %self) #0 !dbg !180 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !182
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.7c8d13c8a74a288d"(ptr %self) #0 !dbg !184 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.Optional.c661605f12aa337a.Core:OptionalAs.1dac4564233fecd1.Core.37febc79c382e2d3"(ptr %self), !dbg !189
// CHECK:STDOUT:   ret ptr %1, !dbg !190
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.65dfedfaf9fb26a5(ptr %self) #0 !dbg !192 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.19661d0961df889d:core.Destroy.Core"(ptr %self), !dbg !196
// CHECK:STDOUT:   ret void, !dbg !197
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.b03cd22c39029320(ptr %self) #0 !dbg !199 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.78fd6ee82eacbbc3:core.Destroy.Core"(ptr %self), !dbg !203
// CHECK:STDOUT:   ret void, !dbg !204
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.Optional.c661605f12aa337a.Core:OptionalAs.1dac4564233fecd1.Core.37febc79c382e2d3"(ptr %self) #0 !dbg !206 {
// CHECK:STDOUT:   %temp = alloca ptr, align 8, !dbg !213
// CHECK:STDOUT:   %temp1 = alloca ptr, align 8, !dbg !213
// CHECK:STDOUT:   %1 = call i1 @_CHasValue.Optional.Core.0c3518913b4fb338(ptr %self), !dbg !211
// CHECK:STDOUT:   br i1 %1, label %2, label %7, !dbg !212
// CHECK:STDOUT:
// CHECK:STDOUT: 2:                                                ; preds = %0
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !213
// CHECK:STDOUT:   %3 = call ptr @_CGet.Optional.Core.0c3518913b4fb338(ptr %self), !dbg !213
// CHECK:STDOUT:   store ptr %3, ptr %temp, align 8, !dbg !213
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp1), !dbg !213
// CHECK:STDOUT:   %4 = load ptr, ptr %temp, align 8, !dbg !213
// CHECK:STDOUT:   store ptr %4, ptr %temp1, align 8, !dbg !213
// CHECK:STDOUT:   %5 = load ptr, ptr %temp1, align 8, !dbg !213
// CHECK:STDOUT:   %6 = call ptr @_CSome.Optional.Core.305e266058820aa4(ptr %5), !dbg !214
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.e9903cfbcec022a9(ptr %temp1), !dbg !213
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.50dcfc6e7a37d220(ptr %temp), !dbg !213
// CHECK:STDOUT:   ret ptr %6, !dbg !215
// CHECK:STDOUT:
// CHECK:STDOUT: 7:                                                ; preds = %0
// CHECK:STDOUT:   %8 = call ptr @_CNone.Optional.Core.305e266058820aa4(), !dbg !216
// CHECK:STDOUT:   ret ptr %8, !dbg !217
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.0c3518913b4fb338(ptr %self) #0 !dbg !219 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.e8f8f92d3d08d149:OptionalStorage.Core.21c65d90918a10a4"(ptr %self), !dbg !225
// CHECK:STDOUT:   ret i1 %1, !dbg !226
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CGet.Optional.Core.0c3518913b4fb338(ptr %self) #0 !dbg !228 {
// CHECK:STDOUT:   %1 = call ptr @"_CGet.e8f8f92d3d08d149:OptionalStorage.Core.21c65d90918a10a4"(ptr %self), !dbg !234
// CHECK:STDOUT:   ret ptr %1, !dbg !235
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CSome.Optional.Core.305e266058820aa4(ptr %value) #0 !dbg !237 {
// CHECK:STDOUT:   %1 = call ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.35c29317da7de352"(ptr %value), !dbg !242
// CHECK:STDOUT:   ret ptr %1, !dbg !243
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.e9903cfbcec022a9(ptr %self) #0 !dbg !245 {
// CHECK:STDOUT:   ret void, !dbg !250
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.50dcfc6e7a37d220(ptr %self) #0 !dbg !252 {
// CHECK:STDOUT:   ret void, !dbg !257
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CNone.Optional.Core.305e266058820aa4() #0 !dbg !259 {
// CHECK:STDOUT:   ret ptr null, !dbg !263
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.e8f8f92d3d08d149:OptionalStorage.Core.21c65d90918a10a4"(ptr %value) #0 !dbg !264 {
// CHECK:STDOUT:   %1 = icmp eq ptr %value, null, !dbg !270
// CHECK:STDOUT:   %2 = xor i1 %1, true, !dbg !271
// CHECK:STDOUT:   ret i1 %2, !dbg !272
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CGet.e8f8f92d3d08d149:OptionalStorage.Core.21c65d90918a10a4"(ptr %value) #0 !dbg !274 {
// CHECK:STDOUT:   ret ptr %value, !dbg !280
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.35c29317da7de352"(ptr %self) #0 !dbg !282 {
// CHECK:STDOUT:   %1 = alloca ptr, align 8, !dbg !287
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !287
// CHECK:STDOUT:   store ptr poison, ptr %1, align 8, !dbg !287
// CHECK:STDOUT:   store ptr %self, ptr %1, align 8, !dbg !291
// CHECK:STDOUT:   %2 = load ptr, ptr %1, align 8, !dbg !289
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.bbeac7c36be34c6c(ptr %1), !dbg !287
// CHECK:STDOUT:   ret ptr %2, !dbg !293
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.bbeac7c36be34c6c(ptr %self) #0 !dbg !295 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.4ef7d32bafd75f98:core.Destroy.Core"(ptr %self), !dbg !299
// CHECK:STDOUT:   ret void, !dbg !300
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 2, 4, 3 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!153}
// CHECK:STDOUT: !llvm.module.flags = !{!40, !41, !42, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!45}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !7 = !{null, !6}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !9 = !{!6}
// CHECK:STDOUT: !10 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !35 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !36 = !{!"omnipotent char", !35, i64 0}
// CHECK:STDOUT: !40 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !41 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !42 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !43 = !{!"int", !36, i64 0}
// CHECK:STDOUT: !44 = !{!"__libc_errno", !43, i64 0}
// CHECK:STDOUT: !45 = !{!44, !43, i64 0}
// CHECK:STDOUT: !57 = !{!6, !6}
// CHECK:STDOUT: !58 = !DISubroutineType(types: !57)
// CHECK:STDOUT: !96 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !105 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !152 = !DIFile(filename: "add_const.carbon", directory: "")
// CHECK:STDOUT: !153 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !152, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !154 = !{null}
// CHECK:STDOUT: !155 = !DISubroutineType(types: !154)
// CHECK:STDOUT: !156 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Main", scope: null, file: !152, line: 10, type: !155, spFlags: DISPFlagDefinition, unit: !153)
// CHECK:STDOUT: !157 = !DILocation(line: 11, column: 12, scope: !156)
// CHECK:STDOUT: !158 = !DILocation(line: 11, column: 3, scope: !156)
// CHECK:STDOUT: !159 = !DILocation(line: 10, column: 1, scope: !156)
// CHECK:STDOUT: !160 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.bdb5ffe218b7af7d:core.Destroy.Core", scope: null, file: !152, line: 11, type: !8, spFlags: DISPFlagDefinition, unit: !153, retainedNodes: !163)
// CHECK:STDOUT: !161 = !DILocalVariable(arg: 1, scope: !160, type: !6)
// CHECK:STDOUT: !162 = !DILocation(line: 11, column: 12, scope: !160)
// CHECK:STDOUT: !163 = !{!161}
// CHECK:STDOUT: !164 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.4ef7d32bafd75f98:core.Destroy.Core", scope: null, file: !152, line: 11, type: !8, spFlags: DISPFlagDefinition, unit: !153, retainedNodes: !167)
// CHECK:STDOUT: !165 = !DILocalVariable(arg: 1, scope: !164, type: !6)
// CHECK:STDOUT: !166 = !DILocation(line: 11, column: 12, scope: !164)
// CHECK:STDOUT: !167 = !{!165}
// CHECK:STDOUT: !168 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.19661d0961df889d:core.Destroy.Core", scope: null, file: !152, line: 11, type: !8, spFlags: DISPFlagDefinition, unit: !153, retainedNodes: !171)
// CHECK:STDOUT: !169 = !DILocalVariable(arg: 1, scope: !168, type: !6)
// CHECK:STDOUT: !170 = !DILocation(line: 11, column: 12, scope: !168)
// CHECK:STDOUT: !171 = !{!169}
// CHECK:STDOUT: !172 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.e3602e7640dfb844:core.Destroy.Core", scope: null, file: !152, line: 11, type: !8, spFlags: DISPFlagDefinition, unit: !153, retainedNodes: !175)
// CHECK:STDOUT: !173 = !DILocalVariable(arg: 1, scope: !172, type: !6)
// CHECK:STDOUT: !174 = !DILocation(line: 11, column: 12, scope: !172)
// CHECK:STDOUT: !175 = !{!173}
// CHECK:STDOUT: !176 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.02018f16aebe62a8:core.Destroy.Core", scope: null, file: !152, line: 11, type: !8, spFlags: DISPFlagDefinition, unit: !153, retainedNodes: !179)
// CHECK:STDOUT: !177 = !DILocalVariable(arg: 1, scope: !176, type: !6)
// CHECK:STDOUT: !178 = !DILocation(line: 11, column: 12, scope: !176)
// CHECK:STDOUT: !179 = !{!177}
// CHECK:STDOUT: !180 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.78fd6ee82eacbbc3:core.Destroy.Core", scope: null, file: !152, line: 11, type: !8, spFlags: DISPFlagDefinition, unit: !153, retainedNodes: !183)
// CHECK:STDOUT: !181 = !DILocalVariable(arg: 1, scope: !180, type: !6)
// CHECK:STDOUT: !182 = !DILocation(line: 11, column: 12, scope: !180)
// CHECK:STDOUT: !183 = !{!181}
// CHECK:STDOUT: !184 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.538322f9fda4254f:ImplicitAs.a276ba298c70ba87.Core.7c8d13c8a74a288d", scope: null, file: !96, line: 105, type: !58, spFlags: DISPFlagDefinition, unit: !153, retainedNodes: !191)
// CHECK:STDOUT: !185 = !DILocalVariable(arg: 1, scope: !184, type: !6)
// CHECK:STDOUT: !189 = !DILocation(line: 106, column: 12, scope: !184)
// CHECK:STDOUT: !190 = !DILocation(line: 106, column: 5, scope: !184)
// CHECK:STDOUT: !191 = !{!185}
// CHECK:STDOUT: !192 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.65dfedfaf9fb26a5", scope: null, file: !105, line: 26, type: !8, spFlags: DISPFlagDefinition, unit: !153, retainedNodes: !198)
// CHECK:STDOUT: !193 = !DILocalVariable(arg: 1, scope: !192, type: !6)
// CHECK:STDOUT: !196 = !DILocation(line: 28, column: 5, scope: !192)
// CHECK:STDOUT: !197 = !DILocation(line: 26, column: 3, scope: !192)
// CHECK:STDOUT: !198 = !{!193}
// CHECK:STDOUT: !199 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.b03cd22c39029320", scope: null, file: !105, line: 26, type: !8, spFlags: DISPFlagDefinition, unit: !153, retainedNodes: !205)
// CHECK:STDOUT: !200 = !DILocalVariable(arg: 1, scope: !199, type: !6)
// CHECK:STDOUT: !203 = !DILocation(line: 28, column: 5, scope: !199)
// CHECK:STDOUT: !204 = !DILocation(line: 26, column: 3, scope: !199)
// CHECK:STDOUT: !205 = !{!200}
// CHECK:STDOUT: !206 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Optional.c661605f12aa337a.Core:OptionalAs.1dac4564233fecd1.Core.37febc79c382e2d3", scope: null, file: !96, line: 95, type: !58, spFlags: DISPFlagDefinition, unit: !153, retainedNodes: !218)
// CHECK:STDOUT: !207 = !DILocalVariable(arg: 1, scope: !206, type: !6)
// CHECK:STDOUT: !211 = !DILocation(line: 96, column: 9, scope: !206)
// CHECK:STDOUT: !212 = !DILocation(line: 96, column: 8, scope: !206)
// CHECK:STDOUT: !213 = !DILocation(line: 97, column: 31, scope: !206)
// CHECK:STDOUT: !214 = !DILocation(line: 97, column: 14, scope: !206)
// CHECK:STDOUT: !215 = !DILocation(line: 97, column: 7, scope: !206)
// CHECK:STDOUT: !216 = !DILocation(line: 99, column: 12, scope: !206)
// CHECK:STDOUT: !217 = !DILocation(line: 99, column: 5, scope: !206)
// CHECK:STDOUT: !218 = !{!207}
// CHECK:STDOUT: !219 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.0c3518913b4fb338", scope: null, file: !96, line: 36, type: !58, spFlags: DISPFlagDefinition, unit: !153, retainedNodes: !227)
// CHECK:STDOUT: !220 = !DILocalVariable(arg: 1, scope: !219, type: !6)
// CHECK:STDOUT: !225 = !DILocation(line: 37, column: 12, scope: !219)
// CHECK:STDOUT: !226 = !DILocation(line: 37, column: 5, scope: !219)
// CHECK:STDOUT: !227 = !{!220}
// CHECK:STDOUT: !228 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.0c3518913b4fb338", scope: null, file: !96, line: 39, type: !58, spFlags: DISPFlagDefinition, unit: !153, retainedNodes: !236)
// CHECK:STDOUT: !229 = !DILocalVariable(arg: 1, scope: !228, type: !6)
// CHECK:STDOUT: !234 = !DILocation(line: 40, column: 12, scope: !228)
// CHECK:STDOUT: !235 = !DILocation(line: 40, column: 5, scope: !228)
// CHECK:STDOUT: !236 = !{!229}
// CHECK:STDOUT: !237 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.305e266058820aa4", scope: null, file: !96, line: 33, type: !58, spFlags: DISPFlagDefinition, unit: !153, retainedNodes: !244)
// CHECK:STDOUT: !238 = !DILocalVariable(arg: 1, scope: !237, type: !6)
// CHECK:STDOUT: !242 = !DILocation(line: 34, column: 12, scope: !237)
// CHECK:STDOUT: !243 = !DILocation(line: 34, column: 5, scope: !237)
// CHECK:STDOUT: !244 = !{!238}
// CHECK:STDOUT: !245 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.e9903cfbcec022a9", scope: null, file: !105, line: 26, type: !8, spFlags: DISPFlagDefinition, unit: !153, retainedNodes: !251)
// CHECK:STDOUT: !246 = !DILocalVariable(arg: 1, scope: !245, type: !6)
// CHECK:STDOUT: !250 = !DILocation(line: 26, column: 3, scope: !245)
// CHECK:STDOUT: !251 = !{!246}
// CHECK:STDOUT: !252 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.50dcfc6e7a37d220", scope: null, file: !105, line: 26, type: !8, spFlags: DISPFlagDefinition, unit: !153, retainedNodes: !258)
// CHECK:STDOUT: !253 = !DILocalVariable(arg: 1, scope: !252, type: !6)
// CHECK:STDOUT: !257 = !DILocation(line: 26, column: 3, scope: !252)
// CHECK:STDOUT: !258 = !{!253}
// CHECK:STDOUT: !259 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.305e266058820aa4", scope: null, file: !96, line: 30, type: !10, spFlags: DISPFlagDefinition, unit: !153)
// CHECK:STDOUT: !263 = !DILocation(line: 31, column: 5, scope: !259)
// CHECK:STDOUT: !264 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.e8f8f92d3d08d149:OptionalStorage.Core.21c65d90918a10a4", scope: null, file: !96, line: 172, type: !58, spFlags: DISPFlagDefinition, unit: !153, retainedNodes: !273)
// CHECK:STDOUT: !265 = !DILocalVariable(arg: 1, scope: !264, type: !6)
// CHECK:STDOUT: !270 = !DILocation(line: 173, column: 16, scope: !264)
// CHECK:STDOUT: !271 = !DILocation(line: 173, column: 12, scope: !264)
// CHECK:STDOUT: !272 = !DILocation(line: 173, column: 5, scope: !264)
// CHECK:STDOUT: !273 = !{!265}
// CHECK:STDOUT: !274 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.e8f8f92d3d08d149:OptionalStorage.Core.21c65d90918a10a4", scope: null, file: !96, line: 175, type: !58, spFlags: DISPFlagDefinition, unit: !153, retainedNodes: !281)
// CHECK:STDOUT: !275 = !DILocalVariable(arg: 1, scope: !274, type: !6)
// CHECK:STDOUT: !280 = !DILocation(line: 176, column: 5, scope: !274)
// CHECK:STDOUT: !281 = !{!275}
// CHECK:STDOUT: !282 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.e8f8f92d3d08d149:OptionalStorage.Core.35c29317da7de352", scope: null, file: !96, line: 166, type: !58, spFlags: DISPFlagDefinition, unit: !153, retainedNodes: !294)
// CHECK:STDOUT: !283 = !DILocalVariable(arg: 1, scope: !282, type: !6)
// CHECK:STDOUT: !287 = !DILocation(line: 167, column: 14, scope: !282)
// CHECK:STDOUT: !289 = !DILocation(line: 167, column: 18, scope: !282)
// CHECK:STDOUT: !291 = !DILocation(line: 169, column: 5, scope: !282)
// CHECK:STDOUT: !293 = !DILocation(line: 170, column: 5, scope: !282)
// CHECK:STDOUT: !294 = !{!283}
// CHECK:STDOUT: !295 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.bbeac7c36be34c6c", scope: null, file: !105, line: 26, type: !8, spFlags: DISPFlagDefinition, unit: !153, retainedNodes: !301)
// CHECK:STDOUT: !296 = !DILocalVariable(arg: 1, scope: !295, type: !6)
// CHECK:STDOUT: !299 = !DILocation(line: 28, column: 5, scope: !295)
// CHECK:STDOUT: !300 = !DILocation(line: 26, column: 3, scope: !295)
// CHECK:STDOUT: !301 = !{!296}
// CHECK:STDOUT:
