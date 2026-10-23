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
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.ea416314509b9f57:ImplicitAs.1863e1559591c4e5.Core.4e9bbab1c7c88cc2"(ptr %p), !dbg !56
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc22_23.2.temp), !dbg !56
// CHECK:STDOUT:   store ptr %U.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc22_23.2.temp, align 8, !dbg !56
// CHECK:STDOUT:   %.loc22_23.4 = load ptr, ptr %.loc22_23.2.temp, align 8, !dbg !56
// CHECK:STDOUT:   call void @_Z15TakeNullablePtrP9NullableC(ptr %.loc22_23.4), !dbg !59
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.bbc536feb16fa006(ptr %.loc22_23.2.temp), !dbg !56
// CHECK:STDOUT:   ret void, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z15TakeNullablePtrP9NullableC(ptr noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnPtr.Main() #0 !dbg !62 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnNullablePtr.call = call ptr @_Z17ReturnNullablePtrv(), !dbg !65
// CHECK:STDOUT:   ret ptr %ReturnNullablePtr.call, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z17ReturnNullablePtrv() #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassPtrWithThunk.Main(ptr %_) #0 !dbg !67 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassNonnullPtrWithThunk.Main(ptr %p) #0 !dbg !72 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc35_32.2.temp = alloca ptr, align 8, !dbg !75
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.ea416314509b9f57:ImplicitAs.1863e1559591c4e5.Core.4e9bbab1c7c88cc2"(ptr %p), !dbg !75
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc35_32.2.temp), !dbg !75
// CHECK:STDOUT:   store ptr %U.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc35_32.2.temp, align 8, !dbg !75
// CHECK:STDOUT:   %.loc35_32.4 = load ptr, ptr %.loc35_32.2.temp, align 8, !dbg !75
// CHECK:STDOUT:   call void @_Z24TakeNullablePtrWithThunkP9NullableCi.carbon_thunk._(ptr %.loc35_32.4), !dbg !76
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.bbc536feb16fa006(ptr %.loc35_32.2.temp), !dbg !75
// CHECK:STDOUT:   ret void, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z24TakeNullablePtrWithThunkP9NullableCi.carbon_thunk._(ptr noundef %0) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !139
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !139
// CHECK:STDOUT:   call void @_Z24TakeNullablePtrWithThunkP9NullableCi(ptr noundef %1, i32 noundef 0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnPtrWithThunk.Main() #0 !dbg !79 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnNullablePtrWithThunk__carbon_thunk.call = call ptr @_Z26ReturnNullablePtrWithThunki.carbon_thunk.(), !dbg !82
// CHECK:STDOUT:   ret ptr %ReturnNullablePtrWithThunk__carbon_thunk.call, !dbg !83
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
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.ea416314509b9f57:ImplicitAs.1863e1559591c4e5.Core.4e9bbab1c7c88cc2"(ptr %self) #0 !dbg !85 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.059052cdc9a4441d"(ptr %self), !dbg !90
// CHECK:STDOUT:   ret ptr %1, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.bbc536feb16fa006(ptr %t) #0 !dbg !94 {
// CHECK:STDOUT:   ret void, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.059052cdc9a4441d"(ptr %self) #0 !dbg !101 {
// CHECK:STDOUT:   %1 = call ptr @_CSome.Optional.Core.059052cdc9a4441d(ptr %self), !dbg !107
// CHECK:STDOUT:   ret ptr %1, !dbg !108
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CSome.Optional.Core.059052cdc9a4441d(ptr %value) #0 !dbg !110 {
// CHECK:STDOUT:   %1 = call ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.77dd513e5db6ef8d"(ptr %value), !dbg !115
// CHECK:STDOUT:   ret ptr %1, !dbg !116
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.77dd513e5db6ef8d"(ptr %self) #0 !dbg !118 {
// CHECK:STDOUT:   %1 = alloca ptr, align 8, !dbg !123
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !123
// CHECK:STDOUT:   store ptr poison, ptr %1, align 8, !dbg !123
// CHECK:STDOUT:   store ptr %self, ptr %1, align 8, !dbg !127
// CHECK:STDOUT:   %2 = load ptr, ptr %1, align 8, !dbg !125
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.9abdc217755355a3(ptr %1), !dbg !123
// CHECK:STDOUT:   ret ptr %2, !dbg !129
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.9abdc217755355a3(ptr %t) #0 !dbg !131 {
// CHECK:STDOUT:   ret void, !dbg !136
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z24TakeNullablePtrWithThunkP9NullableCi(ptr noundef, i32 noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z26ReturnNullablePtrWithThunki(i32 noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_CConvert.ea416314509b9f57:ImplicitAs.1863e1559591c4e5.Core.4e9bbab1c7c88cc2", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 2, 1 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.bbc536feb16fa006, { 1, 0 }
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
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "ReturnPtr", linkageName: "_CReturnPtr.Main", scope: null, file: !46, line: 25, type: !10, spFlags: DISPFlagDefinition, unit: !47)
// CHECK:STDOUT: !65 = !DILocation(line: 26, column: 10, scope: !62)
// CHECK:STDOUT: !66 = !DILocation(line: 26, column: 3, scope: !62)
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "PassPtrWithThunk", linkageName: "_CPassPtrWithThunk.Main", scope: null, file: !46, line: 29, type: !8, spFlags: DISPFlagDefinition, unit: !47, retainedNodes: !71)
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 1, scope: !67, type: !6)
// CHECK:STDOUT: !70 = !DILocation(line: 29, column: 1, scope: !67)
// CHECK:STDOUT: !71 = !{!68}
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "PassNonnullPtrWithThunk", linkageName: "_CPassNonnullPtrWithThunk.Main", scope: null, file: !46, line: 34, type: !8, spFlags: DISPFlagDefinition, unit: !47, retainedNodes: !78)
// CHECK:STDOUT: !73 = !DILocalVariable(arg: 1, scope: !72, type: !6)
// CHECK:STDOUT: !75 = !DILocation(line: 35, column: 32, scope: !72)
// CHECK:STDOUT: !76 = !DILocation(line: 35, column: 3, scope: !72)
// CHECK:STDOUT: !77 = !DILocation(line: 34, column: 1, scope: !72)
// CHECK:STDOUT: !78 = !{!73}
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "ReturnPtrWithThunk", linkageName: "_CReturnPtrWithThunk.Main", scope: null, file: !46, line: 38, type: !10, spFlags: DISPFlagDefinition, unit: !47)
// CHECK:STDOUT: !82 = !DILocation(line: 39, column: 10, scope: !79)
// CHECK:STDOUT: !83 = !DILocation(line: 39, column: 3, scope: !79)
// CHECK:STDOUT: !84 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !85 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ea416314509b9f57:ImplicitAs.1863e1559591c4e5.Core.4e9bbab1c7c88cc2", scope: null, file: !84, line: 105, type: !58, spFlags: DISPFlagDefinition, unit: !47, retainedNodes: !92)
// CHECK:STDOUT: !86 = !DILocalVariable(arg: 1, scope: !85, type: !6)
// CHECK:STDOUT: !90 = !DILocation(line: 106, column: 12, scope: !85)
// CHECK:STDOUT: !91 = !DILocation(line: 106, column: 5, scope: !85)
// CHECK:STDOUT: !92 = !{!86}
// CHECK:STDOUT: !93 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !94 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.bbc536feb16fa006", scope: null, file: !93, line: 35, type: !8, spFlags: DISPFlagDefinition, unit: !47, retainedNodes: !100)
// CHECK:STDOUT: !95 = !DILocalVariable(arg: 1, scope: !94, type: !6)
// CHECK:STDOUT: !99 = !DILocation(line: 35, column: 1, scope: !94)
// CHECK:STDOUT: !100 = !{!95}
// CHECK:STDOUT: !101 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.9fdaf39d07a1dc4e:OptionalAs.24b7347a93330e68.Core.059052cdc9a4441d", scope: null, file: !84, line: 80, type: !58, spFlags: DISPFlagDefinition, unit: !47, retainedNodes: !109)
// CHECK:STDOUT: !102 = !DILocalVariable(arg: 1, scope: !101, type: !6)
// CHECK:STDOUT: !107 = !DILocation(line: 81, column: 12, scope: !101)
// CHECK:STDOUT: !108 = !DILocation(line: 81, column: 5, scope: !101)
// CHECK:STDOUT: !109 = !{!102}
// CHECK:STDOUT: !110 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.059052cdc9a4441d", scope: null, file: !84, line: 33, type: !58, spFlags: DISPFlagDefinition, unit: !47, retainedNodes: !117)
// CHECK:STDOUT: !111 = !DILocalVariable(arg: 1, scope: !110, type: !6)
// CHECK:STDOUT: !115 = !DILocation(line: 34, column: 12, scope: !110)
// CHECK:STDOUT: !116 = !DILocation(line: 34, column: 5, scope: !110)
// CHECK:STDOUT: !117 = !{!111}
// CHECK:STDOUT: !118 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.e8f8f92d3d08d149:OptionalStorage.Core.77dd513e5db6ef8d", scope: null, file: !84, line: 166, type: !58, spFlags: DISPFlagDefinition, unit: !47, retainedNodes: !130)
// CHECK:STDOUT: !119 = !DILocalVariable(arg: 1, scope: !118, type: !6)
// CHECK:STDOUT: !123 = !DILocation(line: 167, column: 14, scope: !118)
// CHECK:STDOUT: !125 = !DILocation(line: 167, column: 18, scope: !118)
// CHECK:STDOUT: !127 = !DILocation(line: 169, column: 5, scope: !118)
// CHECK:STDOUT: !129 = !DILocation(line: 170, column: 5, scope: !118)
// CHECK:STDOUT: !130 = !{!119}
// CHECK:STDOUT: !131 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.9abdc217755355a3", scope: null, file: !93, line: 35, type: !8, spFlags: DISPFlagDefinition, unit: !47, retainedNodes: !137)
// CHECK:STDOUT: !132 = !DILocalVariable(arg: 1, scope: !131, type: !6)
// CHECK:STDOUT: !136 = !DILocation(line: 35, column: 1, scope: !131)
// CHECK:STDOUT: !137 = !{!132}
// CHECK:STDOUT: !138 = !{!"p1 _ZTS9NullableC", !37, i64 0}
// CHECK:STDOUT: !139 = !{!138, !138, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'add_const.carbon'
// CHECK:STDOUT: source_filename = "add_const.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvert.Main() #0 !dbg !144 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_21.1.temp = alloca ptr, align 8, !dbg !145
// CHECK:STDOUT:   %.loc11_21.5.temp = alloca ptr, align 8, !dbg !145
// CHECK:STDOUT:   %make.call = call ptr @_Z4makev(), !dbg !145
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc11_21.1.temp), !dbg !145
// CHECK:STDOUT:   store ptr %make.call, ptr %.loc11_21.1.temp, align 8, !dbg !145
// CHECK:STDOUT:   %.loc11_21.3 = load ptr, ptr %.loc11_21.1.temp, align 8, !dbg !145
// CHECK:STDOUT:   %U.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.ea416314509b9f57:ImplicitAs.1863e1559591c4e5.Core.470ca455cde45499"(ptr %.loc11_21.3), !dbg !145
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc11_21.5.temp), !dbg !145
// CHECK:STDOUT:   store ptr %U.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc11_21.5.temp, align 8, !dbg !145
// CHECK:STDOUT:   %.loc11_21.7 = load ptr, ptr %.loc11_21.5.temp, align 8, !dbg !145
// CHECK:STDOUT:   call void @_Z4takePK6ConstC(ptr %.loc11_21.7), !dbg !146
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.55cdda357d448221(ptr %.loc11_21.5.temp), !dbg !145
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.63d1d3e401037503(ptr %.loc11_21.1.temp), !dbg !145
// CHECK:STDOUT:   ret void, !dbg !147
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef ptr @_Z4makev() #1
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z4takePK6ConstC(ptr noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.ea416314509b9f57:ImplicitAs.1863e1559591c4e5.Core.470ca455cde45499"(ptr %self) #0 !dbg !148 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.Optional.d1728a6b8ac984c6.Core:OptionalAs.5cc57aaa5e35eede.Core.397ceadf3cc9326d"(ptr %self), !dbg !153
// CHECK:STDOUT:   ret ptr %1, !dbg !154
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.55cdda357d448221(ptr %t) #0 !dbg !156 {
// CHECK:STDOUT:   ret void, !dbg !161
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.63d1d3e401037503(ptr %t) #0 !dbg !163 {
// CHECK:STDOUT:   ret void, !dbg !168
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.Optional.d1728a6b8ac984c6.Core:OptionalAs.5cc57aaa5e35eede.Core.397ceadf3cc9326d"(ptr %self) #0 !dbg !170 {
// CHECK:STDOUT:   %temp = alloca ptr, align 8, !dbg !177
// CHECK:STDOUT:   %temp1 = alloca ptr, align 8, !dbg !177
// CHECK:STDOUT:   %1 = call i1 @_CHasValue.Optional.Core.128f4c34f2ac2dc8(ptr %self), !dbg !175
// CHECK:STDOUT:   br i1 %1, label %2, label %7, !dbg !176
// CHECK:STDOUT:
// CHECK:STDOUT: 2:                                                ; preds = %0
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp), !dbg !177
// CHECK:STDOUT:   %3 = call ptr @_CGet.Optional.Core.128f4c34f2ac2dc8(ptr %self), !dbg !177
// CHECK:STDOUT:   store ptr %3, ptr %temp, align 8, !dbg !177
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %temp1), !dbg !177
// CHECK:STDOUT:   %4 = load ptr, ptr %temp, align 8, !dbg !177
// CHECK:STDOUT:   store ptr %4, ptr %temp1, align 8, !dbg !177
// CHECK:STDOUT:   %5 = load ptr, ptr %temp1, align 8, !dbg !177
// CHECK:STDOUT:   %6 = call ptr @_CSome.Optional.Core.1c39bd137e62c6ea(ptr %5), !dbg !178
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c9c637f86feae476(ptr %temp1), !dbg !177
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.afe76cd75c1d6f19(ptr %temp), !dbg !177
// CHECK:STDOUT:   ret ptr %6, !dbg !179
// CHECK:STDOUT:
// CHECK:STDOUT: 7:                                                ; preds = %0
// CHECK:STDOUT:   %8 = call ptr @_CNone.Optional.Core.1c39bd137e62c6ea(), !dbg !180
// CHECK:STDOUT:   ret ptr %8, !dbg !181
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CHasValue.Optional.Core.128f4c34f2ac2dc8(ptr %self) #0 !dbg !183 {
// CHECK:STDOUT:   %1 = call i1 @"_CHas.e8f8f92d3d08d149:OptionalStorage.Core.21c65d90918a10a4"(ptr %self), !dbg !189
// CHECK:STDOUT:   ret i1 %1, !dbg !190
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CGet.Optional.Core.128f4c34f2ac2dc8(ptr %self) #0 !dbg !192 {
// CHECK:STDOUT:   %1 = call ptr @"_CGet.e8f8f92d3d08d149:OptionalStorage.Core.21c65d90918a10a4"(ptr %self), !dbg !198
// CHECK:STDOUT:   ret ptr %1, !dbg !199
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CSome.Optional.Core.1c39bd137e62c6ea(ptr %value) #0 !dbg !201 {
// CHECK:STDOUT:   %1 = call ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.35c29317da7de352"(ptr %value), !dbg !206
// CHECK:STDOUT:   ret ptr %1, !dbg !207
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.c9c637f86feae476(ptr %t) #0 !dbg !209 {
// CHECK:STDOUT:   ret void, !dbg !214
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.afe76cd75c1d6f19(ptr %t) #0 !dbg !216 {
// CHECK:STDOUT:   ret void, !dbg !221
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CNone.Optional.Core.1c39bd137e62c6ea() #0 !dbg !223 {
// CHECK:STDOUT:   ret ptr null, !dbg !227
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @"_CHas.e8f8f92d3d08d149:OptionalStorage.Core.21c65d90918a10a4"(ptr %value) #0 !dbg !228 {
// CHECK:STDOUT:   %1 = icmp eq ptr %value, null, !dbg !234
// CHECK:STDOUT:   %2 = xor i1 %1, true, !dbg !235
// CHECK:STDOUT:   ret i1 %2, !dbg !236
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CGet.e8f8f92d3d08d149:OptionalStorage.Core.21c65d90918a10a4"(ptr %value) #0 !dbg !238 {
// CHECK:STDOUT:   ret ptr %value, !dbg !244
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CSome.e8f8f92d3d08d149:OptionalStorage.Core.35c29317da7de352"(ptr %self) #0 !dbg !246 {
// CHECK:STDOUT:   %1 = alloca ptr, align 8, !dbg !251
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !251
// CHECK:STDOUT:   store ptr poison, ptr %1, align 8, !dbg !251
// CHECK:STDOUT:   store ptr %self, ptr %1, align 8, !dbg !255
// CHECK:STDOUT:   %2 = load ptr, ptr %1, align 8, !dbg !253
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.4d5bd971f411a76b(ptr %1), !dbg !251
// CHECK:STDOUT:   ret ptr %2, !dbg !257
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.4d5bd971f411a76b(ptr %t) #0 !dbg !259 {
// CHECK:STDOUT:   ret void, !dbg !264
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 2, 4, 3 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!141}
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
// CHECK:STDOUT: !84 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !93 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !140 = !DIFile(filename: "add_const.carbon", directory: "")
// CHECK:STDOUT: !141 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !140, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !142 = !{null}
// CHECK:STDOUT: !143 = !DISubroutineType(types: !142)
// CHECK:STDOUT: !144 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Main", scope: null, file: !140, line: 10, type: !143, spFlags: DISPFlagDefinition, unit: !141)
// CHECK:STDOUT: !145 = !DILocation(line: 11, column: 12, scope: !144)
// CHECK:STDOUT: !146 = !DILocation(line: 11, column: 3, scope: !144)
// CHECK:STDOUT: !147 = !DILocation(line: 10, column: 1, scope: !144)
// CHECK:STDOUT: !148 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.ea416314509b9f57:ImplicitAs.1863e1559591c4e5.Core.470ca455cde45499", scope: null, file: !84, line: 105, type: !58, spFlags: DISPFlagDefinition, unit: !141, retainedNodes: !155)
// CHECK:STDOUT: !149 = !DILocalVariable(arg: 1, scope: !148, type: !6)
// CHECK:STDOUT: !153 = !DILocation(line: 106, column: 12, scope: !148)
// CHECK:STDOUT: !154 = !DILocation(line: 106, column: 5, scope: !148)
// CHECK:STDOUT: !155 = !{!149}
// CHECK:STDOUT: !156 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.55cdda357d448221", scope: null, file: !93, line: 35, type: !8, spFlags: DISPFlagDefinition, unit: !141, retainedNodes: !162)
// CHECK:STDOUT: !157 = !DILocalVariable(arg: 1, scope: !156, type: !6)
// CHECK:STDOUT: !161 = !DILocation(line: 35, column: 1, scope: !156)
// CHECK:STDOUT: !162 = !{!157}
// CHECK:STDOUT: !163 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.63d1d3e401037503", scope: null, file: !93, line: 35, type: !8, spFlags: DISPFlagDefinition, unit: !141, retainedNodes: !169)
// CHECK:STDOUT: !164 = !DILocalVariable(arg: 1, scope: !163, type: !6)
// CHECK:STDOUT: !168 = !DILocation(line: 35, column: 1, scope: !163)
// CHECK:STDOUT: !169 = !{!164}
// CHECK:STDOUT: !170 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Optional.d1728a6b8ac984c6.Core:OptionalAs.5cc57aaa5e35eede.Core.397ceadf3cc9326d", scope: null, file: !84, line: 95, type: !58, spFlags: DISPFlagDefinition, unit: !141, retainedNodes: !182)
// CHECK:STDOUT: !171 = !DILocalVariable(arg: 1, scope: !170, type: !6)
// CHECK:STDOUT: !175 = !DILocation(line: 96, column: 9, scope: !170)
// CHECK:STDOUT: !176 = !DILocation(line: 96, column: 8, scope: !170)
// CHECK:STDOUT: !177 = !DILocation(line: 97, column: 31, scope: !170)
// CHECK:STDOUT: !178 = !DILocation(line: 97, column: 14, scope: !170)
// CHECK:STDOUT: !179 = !DILocation(line: 97, column: 7, scope: !170)
// CHECK:STDOUT: !180 = !DILocation(line: 99, column: 12, scope: !170)
// CHECK:STDOUT: !181 = !DILocation(line: 99, column: 5, scope: !170)
// CHECK:STDOUT: !182 = !{!171}
// CHECK:STDOUT: !183 = distinct !DISubprogram(name: "HasValue", linkageName: "_CHasValue.Optional.Core.128f4c34f2ac2dc8", scope: null, file: !84, line: 36, type: !58, spFlags: DISPFlagDefinition, unit: !141, retainedNodes: !191)
// CHECK:STDOUT: !184 = !DILocalVariable(arg: 1, scope: !183, type: !6)
// CHECK:STDOUT: !189 = !DILocation(line: 37, column: 12, scope: !183)
// CHECK:STDOUT: !190 = !DILocation(line: 37, column: 5, scope: !183)
// CHECK:STDOUT: !191 = !{!184}
// CHECK:STDOUT: !192 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Optional.Core.128f4c34f2ac2dc8", scope: null, file: !84, line: 39, type: !58, spFlags: DISPFlagDefinition, unit: !141, retainedNodes: !200)
// CHECK:STDOUT: !193 = !DILocalVariable(arg: 1, scope: !192, type: !6)
// CHECK:STDOUT: !198 = !DILocation(line: 40, column: 12, scope: !192)
// CHECK:STDOUT: !199 = !DILocation(line: 40, column: 5, scope: !192)
// CHECK:STDOUT: !200 = !{!193}
// CHECK:STDOUT: !201 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.1c39bd137e62c6ea", scope: null, file: !84, line: 33, type: !58, spFlags: DISPFlagDefinition, unit: !141, retainedNodes: !208)
// CHECK:STDOUT: !202 = !DILocalVariable(arg: 1, scope: !201, type: !6)
// CHECK:STDOUT: !206 = !DILocation(line: 34, column: 12, scope: !201)
// CHECK:STDOUT: !207 = !DILocation(line: 34, column: 5, scope: !201)
// CHECK:STDOUT: !208 = !{!202}
// CHECK:STDOUT: !209 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.c9c637f86feae476", scope: null, file: !93, line: 35, type: !8, spFlags: DISPFlagDefinition, unit: !141, retainedNodes: !215)
// CHECK:STDOUT: !210 = !DILocalVariable(arg: 1, scope: !209, type: !6)
// CHECK:STDOUT: !214 = !DILocation(line: 35, column: 1, scope: !209)
// CHECK:STDOUT: !215 = !{!210}
// CHECK:STDOUT: !216 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.afe76cd75c1d6f19", scope: null, file: !93, line: 35, type: !8, spFlags: DISPFlagDefinition, unit: !141, retainedNodes: !222)
// CHECK:STDOUT: !217 = !DILocalVariable(arg: 1, scope: !216, type: !6)
// CHECK:STDOUT: !221 = !DILocation(line: 35, column: 1, scope: !216)
// CHECK:STDOUT: !222 = !{!217}
// CHECK:STDOUT: !223 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.1c39bd137e62c6ea", scope: null, file: !84, line: 30, type: !10, spFlags: DISPFlagDefinition, unit: !141)
// CHECK:STDOUT: !227 = !DILocation(line: 31, column: 5, scope: !223)
// CHECK:STDOUT: !228 = distinct !DISubprogram(name: "Has", linkageName: "_CHas.e8f8f92d3d08d149:OptionalStorage.Core.21c65d90918a10a4", scope: null, file: !84, line: 172, type: !58, spFlags: DISPFlagDefinition, unit: !141, retainedNodes: !237)
// CHECK:STDOUT: !229 = !DILocalVariable(arg: 1, scope: !228, type: !6)
// CHECK:STDOUT: !234 = !DILocation(line: 173, column: 16, scope: !228)
// CHECK:STDOUT: !235 = !DILocation(line: 173, column: 12, scope: !228)
// CHECK:STDOUT: !236 = !DILocation(line: 173, column: 5, scope: !228)
// CHECK:STDOUT: !237 = !{!229}
// CHECK:STDOUT: !238 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.e8f8f92d3d08d149:OptionalStorage.Core.21c65d90918a10a4", scope: null, file: !84, line: 175, type: !58, spFlags: DISPFlagDefinition, unit: !141, retainedNodes: !245)
// CHECK:STDOUT: !239 = !DILocalVariable(arg: 1, scope: !238, type: !6)
// CHECK:STDOUT: !244 = !DILocation(line: 176, column: 5, scope: !238)
// CHECK:STDOUT: !245 = !{!239}
// CHECK:STDOUT: !246 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.e8f8f92d3d08d149:OptionalStorage.Core.35c29317da7de352", scope: null, file: !84, line: 166, type: !58, spFlags: DISPFlagDefinition, unit: !141, retainedNodes: !258)
// CHECK:STDOUT: !247 = !DILocalVariable(arg: 1, scope: !246, type: !6)
// CHECK:STDOUT: !251 = !DILocation(line: 167, column: 14, scope: !246)
// CHECK:STDOUT: !253 = !DILocation(line: 167, column: 18, scope: !246)
// CHECK:STDOUT: !255 = !DILocation(line: 169, column: 5, scope: !246)
// CHECK:STDOUT: !257 = !DILocation(line: 170, column: 5, scope: !246)
// CHECK:STDOUT: !258 = !{!247}
// CHECK:STDOUT: !259 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.4d5bd971f411a76b", scope: null, file: !93, line: 35, type: !8, spFlags: DISPFlagDefinition, unit: !141, retainedNodes: !265)
// CHECK:STDOUT: !260 = !DILocalVariable(arg: 1, scope: !259, type: !6)
// CHECK:STDOUT: !264 = !DILocation(line: 35, column: 1, scope: !259)
// CHECK:STDOUT: !265 = !{!260}
// CHECK:STDOUT:
