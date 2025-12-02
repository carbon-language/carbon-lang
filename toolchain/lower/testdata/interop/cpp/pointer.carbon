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

fn PassPtr(p: Core.Optional(Cpp.C*)) {
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

fn PassPtrWithThunk(p: Core.Optional(Cpp.C*)) {
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
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassPtr.Main(ptr %p) #0 !dbg !7 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z7TakePtrP1C(ptr %p), !dbg !13
// CHECK:STDOUT:   ret void, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z7TakePtrP1C(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnPtr.Main() #0 !dbg !15 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnPtr.call = call ptr @_Z9ReturnPtrv(), !dbg !18
// CHECK:STDOUT:   ret ptr %ReturnPtr.call, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare ptr @_Z9ReturnPtrv()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassPtrWithThunk.Main(ptr %p) #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z16TakePtrWithThunkP1Ci.carbon_thunk1(ptr %p), !dbg !23
// CHECK:STDOUT:   ret void, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnPtrWithThunk.Main() #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnPtrWithThunk__carbon_thunk.call = call ptr @_Z18ReturnPtrWithThunki.carbon_thunk0(), !dbg !26
// CHECK:STDOUT:   ret ptr %ReturnPtrWithThunk__carbon_thunk.call, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress
// CHECK:STDOUT: define dso_local void @_Z16TakePtrWithThunkP1Ci.carbon_thunk1(ptr %0) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8
// CHECK:STDOUT:   call void @_Z16TakePtrWithThunkP1Ci(ptr %1, i32 0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z16TakePtrWithThunkP1Ci(ptr, i32) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress
// CHECK:STDOUT: define dso_local ptr @_Z18ReturnPtrWithThunki.carbon_thunk0() #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %call = call ptr @_Z18ReturnPtrWithThunki(i32 0)
// CHECK:STDOUT:   ret ptr %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare ptr @_Z18ReturnPtrWithThunki(i32) #2
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="0" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { "no-trapping-math"="true" "stack-protector-buffer-size"="0" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 1, !"wchar_size", i32 4}
// CHECK:STDOUT: !3 = !{i32 8, !"PIC Level", i32 0}
// CHECK:STDOUT: !4 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "nonnull.carbon", directory: "")
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "PassPtr", linkageName: "_CPassPtr.Main", scope: null, file: !6, line: 14, type: !8, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !11)
// CHECK:STDOUT: !8 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !9 = !{null, !10}
// CHECK:STDOUT: !10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 8)
// CHECK:STDOUT: !11 = !{!12}
// CHECK:STDOUT: !12 = !DILocalVariable(arg: 1, scope: !7, type: !10)
// CHECK:STDOUT: !13 = !DILocation(line: 15, column: 3, scope: !7)
// CHECK:STDOUT: !14 = !DILocation(line: 14, column: 1, scope: !7)
// CHECK:STDOUT: !15 = distinct !DISubprogram(name: "ReturnPtr", linkageName: "_CReturnPtr.Main", scope: null, file: !6, line: 18, type: !16, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !16 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !17 = !{!10}
// CHECK:STDOUT: !18 = !DILocation(line: 19, column: 10, scope: !15)
// CHECK:STDOUT: !19 = !DILocation(line: 19, column: 3, scope: !15)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "PassPtrWithThunk", linkageName: "_CPassPtrWithThunk.Main", scope: null, file: !6, line: 22, type: !8, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !21)
// CHECK:STDOUT: !21 = !{!22}
// CHECK:STDOUT: !22 = !DILocalVariable(arg: 1, scope: !20, type: !10)
// CHECK:STDOUT: !23 = !DILocation(line: 23, column: 3, scope: !20)
// CHECK:STDOUT: !24 = !DILocation(line: 22, column: 1, scope: !20)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "ReturnPtrWithThunk", linkageName: "_CReturnPtrWithThunk.Main", scope: null, file: !6, line: 26, type: !16, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !26 = !DILocation(line: 27, column: 10, scope: !25)
// CHECK:STDOUT: !27 = !DILocation(line: 27, column: 3, scope: !25)
// CHECK:STDOUT: ; ModuleID = 'nullable.carbon'
// CHECK:STDOUT: source_filename = "nullable.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassPtr.Main(ptr %p) #0 !dbg !7 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassNonnullPtr.Main(ptr %p) #0 !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc22_15.2.temp = alloca ptr, align 8, !dbg !17
// CHECK:STDOUT:   %U.binding.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.f82b7327366b1770:ImplicitAs.Core.78b0d6e062ff6422"(ptr %p), !dbg !17
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc22_15.2.temp), !dbg !17
// CHECK:STDOUT:   store ptr %U.binding.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc22_15.2.temp, align 8, !dbg !17
// CHECK:STDOUT:   %.loc22_15.4 = load ptr, ptr %.loc22_15.2.temp, align 8, !dbg !17
// CHECK:STDOUT:   call void @_Z7TakePtrP1C(ptr %.loc22_15.4), !dbg !18
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z7TakePtrP1C(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnPtr.Main() #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnPtr.call = call ptr @_Z9ReturnPtrv(), !dbg !23
// CHECK:STDOUT:   ret ptr %ReturnPtr.call, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare ptr @_Z9ReturnPtrv()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassPtrWithThunk.Main(ptr %p) #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassNonnullPtrWithThunk.Main(ptr %p) #0 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc35_24.2.temp = alloca ptr, align 8, !dbg !32
// CHECK:STDOUT:   %U.binding.as_type.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.f82b7327366b1770:ImplicitAs.Core.78b0d6e062ff6422"(ptr %p), !dbg !32
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc35_24.2.temp), !dbg !32
// CHECK:STDOUT:   store ptr %U.binding.as_type.as.ImplicitAs.impl.Convert.call, ptr %.loc35_24.2.temp, align 8, !dbg !32
// CHECK:STDOUT:   %.loc35_24.4 = load ptr, ptr %.loc35_24.2.temp, align 8, !dbg !32
// CHECK:STDOUT:   call void @_Z16TakePtrWithThunkP1Ci.carbon_thunk1(ptr %.loc35_24.4), !dbg !33
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnPtrWithThunk.Main() #0 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnPtrWithThunk__carbon_thunk.call = call ptr @_Z18ReturnPtrWithThunki.carbon_thunk0(), !dbg !36
// CHECK:STDOUT:   ret ptr %ReturnPtrWithThunk__carbon_thunk.call, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.f82b7327366b1770:ImplicitAs.Core.78b0d6e062ff6422"(ptr %self) #0 !dbg !38 {
// CHECK:STDOUT:   %1 = call ptr @"_CConvert.8d16edc1dfe20a7c:OptionalAs.Core.b5eba8f3f1d9af5a"(ptr %self), !dbg !44
// CHECK:STDOUT:   ret ptr %1, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.8d16edc1dfe20a7c:OptionalAs.Core.b5eba8f3f1d9af5a"(ptr %self) #0 !dbg !46 {
// CHECK:STDOUT:   %1 = call ptr @_CSome.Optional.Core.b5eba8f3f1d9af5a(ptr %self), !dbg !49
// CHECK:STDOUT:   ret ptr %1, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CSome.Optional.Core.b5eba8f3f1d9af5a(ptr %value) #0 !dbg !51 {
// CHECK:STDOUT:   %1 = call ptr @"_CSome.4f0b5cc38af595d2:OptionalStorage.Core.f53db17714b9f655"(ptr %value), !dbg !54
// CHECK:STDOUT:   ret ptr %1, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CSome.4f0b5cc38af595d2:OptionalStorage.Core.f53db17714b9f655"(ptr %self) #0 !dbg !56 {
// CHECK:STDOUT:   %1 = alloca ptr, align 8, !dbg !59
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %1), !dbg !59
// CHECK:STDOUT:   store ptr %self, ptr %1, align 8, !dbg !60
// CHECK:STDOUT:   %2 = load ptr, ptr %1, align 8, !dbg !61
// CHECK:STDOUT:   ret ptr %2, !dbg !62
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress
// CHECK:STDOUT: define dso_local void @_Z16TakePtrWithThunkP1Ci.carbon_thunk1(ptr %0) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8
// CHECK:STDOUT:   call void @_Z16TakePtrWithThunkP1Ci(ptr %1, i32 0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z16TakePtrWithThunkP1Ci(ptr, i32) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress
// CHECK:STDOUT: define dso_local ptr @_Z18ReturnPtrWithThunki.carbon_thunk0() #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %call = call ptr @_Z18ReturnPtrWithThunki(i32 0)
// CHECK:STDOUT:   ret ptr %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare ptr @_Z18ReturnPtrWithThunki(i32) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_CConvert.f82b7327366b1770:ImplicitAs.Core.78b0d6e062ff6422", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 2, 1 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { alwaysinline mustprogress "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="0" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { "no-trapping-math"="true" "stack-protector-buffer-size"="0" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 1, !"wchar_size", i32 4}
// CHECK:STDOUT: !3 = !{i32 8, !"PIC Level", i32 0}
// CHECK:STDOUT: !4 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "nullable.carbon", directory: "")
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "PassPtr", linkageName: "_CPassPtr.Main", scope: null, file: !6, line: 14, type: !8, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !11)
// CHECK:STDOUT: !8 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !9 = !{null, !10}
// CHECK:STDOUT: !10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 8)
// CHECK:STDOUT: !11 = !{!12}
// CHECK:STDOUT: !12 = !DILocalVariable(arg: 1, scope: !7, type: !10)
// CHECK:STDOUT: !13 = !DILocation(line: 14, column: 1, scope: !7)
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "PassNonnullPtr", linkageName: "_CPassNonnullPtr.Main", scope: null, file: !6, line: 19, type: !8, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !15)
// CHECK:STDOUT: !15 = !{!16}
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 1, scope: !14, type: !10)
// CHECK:STDOUT: !17 = !DILocation(line: 22, column: 15, scope: !14)
// CHECK:STDOUT: !18 = !DILocation(line: 22, column: 3, scope: !14)
// CHECK:STDOUT: !19 = !DILocation(line: 19, column: 1, scope: !14)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "ReturnPtr", linkageName: "_CReturnPtr.Main", scope: null, file: !6, line: 25, type: !21, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !21 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !22 = !{!10}
// CHECK:STDOUT: !23 = !DILocation(line: 26, column: 10, scope: !20)
// CHECK:STDOUT: !24 = !DILocation(line: 26, column: 3, scope: !20)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "PassPtrWithThunk", linkageName: "_CPassPtrWithThunk.Main", scope: null, file: !6, line: 29, type: !8, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !26)
// CHECK:STDOUT: !26 = !{!27}
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !25, type: !10)
// CHECK:STDOUT: !28 = !DILocation(line: 29, column: 1, scope: !25)
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "PassNonnullPtrWithThunk", linkageName: "_CPassNonnullPtrWithThunk.Main", scope: null, file: !6, line: 34, type: !8, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !30)
// CHECK:STDOUT: !30 = !{!31}
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 1, scope: !29, type: !10)
// CHECK:STDOUT: !32 = !DILocation(line: 35, column: 24, scope: !29)
// CHECK:STDOUT: !33 = !DILocation(line: 35, column: 3, scope: !29)
// CHECK:STDOUT: !34 = !DILocation(line: 34, column: 1, scope: !29)
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "ReturnPtrWithThunk", linkageName: "_CReturnPtrWithThunk.Main", scope: null, file: !6, line: 38, type: !21, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !36 = !DILocation(line: 39, column: 10, scope: !35)
// CHECK:STDOUT: !37 = !DILocation(line: 39, column: 3, scope: !35)
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.f82b7327366b1770:ImplicitAs.Core.78b0d6e062ff6422", scope: null, file: !39, line: 82, type: !40, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !42)
// CHECK:STDOUT: !39 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !40 = !DISubroutineType(types: !41)
// CHECK:STDOUT: !41 = !{!10, !10}
// CHECK:STDOUT: !42 = !{!43}
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !38, type: !10)
// CHECK:STDOUT: !44 = !DILocation(line: 83, column: 12, scope: !38)
// CHECK:STDOUT: !45 = !DILocation(line: 83, column: 5, scope: !38)
// CHECK:STDOUT: !46 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.8d16edc1dfe20a7c:OptionalAs.Core.b5eba8f3f1d9af5a", scope: null, file: !39, line: 68, type: !40, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !47)
// CHECK:STDOUT: !47 = !{!48}
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 1, scope: !46, type: !10)
// CHECK:STDOUT: !49 = !DILocation(line: 69, column: 12, scope: !46)
// CHECK:STDOUT: !50 = !DILocation(line: 69, column: 5, scope: !46)
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.Optional.Core.b5eba8f3f1d9af5a", scope: null, file: !39, line: 29, type: !40, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !52)
// CHECK:STDOUT: !52 = !{!53}
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 1, scope: !51, type: !10)
// CHECK:STDOUT: !54 = !DILocation(line: 30, column: 12, scope: !51)
// CHECK:STDOUT: !55 = !DILocation(line: 30, column: 5, scope: !51)
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "Some", linkageName: "_CSome.4f0b5cc38af595d2:OptionalStorage.Core.f53db17714b9f655", scope: null, file: !39, line: 125, type: !40, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !57)
// CHECK:STDOUT: !57 = !{!58}
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 1, scope: !56, type: !10)
// CHECK:STDOUT: !59 = !DILocation(line: 126, column: 14, scope: !56)
// CHECK:STDOUT: !60 = !DILocation(line: 127, column: 5, scope: !56)
// CHECK:STDOUT: !61 = !DILocation(line: 126, column: 18, scope: !56)
// CHECK:STDOUT: !62 = !DILocation(line: 128, column: 5, scope: !56)
