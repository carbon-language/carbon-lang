// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/reference.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/reference.carbon

// ============================================================================
// Reference params
// ============================================================================

// --- pass_references.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
class C {};

auto TakeCRef(C&) -> void;
auto TakeCRRef(C&&) -> void;
auto TakeConstCRef(const C&) -> void;

auto TakeIntRef(int&) -> void;
auto TakeIntRRef(int&&) -> void;
auto TakeConstIntRef(const int&) -> void;
''';

fn PassRefs() {
  var c: Cpp.C;
  Cpp.TakeCRef(ref c);
  Cpp.TakeCRRef({} as Cpp.C);
  Cpp.TakeConstCRef(c);

  var n: i32;
  Cpp.TakeIntRef(ref n);
  Cpp.TakeIntRRef(42 as i32);
  Cpp.TakeConstIntRef(n);
}

// --- pass_references_via_thunk.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
class C {};
class ForceThunk {};

auto TakeCRef(C&, ForceThunk = {}) -> void;
auto TakeCRRef(const C&, ForceThunk = {}) -> void;
auto TakeConstCRef(const C&, ForceThunk = {}) -> void;

auto TakeIntRef(int&, ForceThunk = {}) -> void;
auto TakeIntRRef(int&&, ForceThunk = {}) -> void;
auto TakeConstIntRef(const int&, ForceThunk = {}) -> void;
''';

fn PassRefs() {
  var c: Cpp.C;
  Cpp.TakeCRef(ref c);
  Cpp.TakeCRRef({} as Cpp.C);
  Cpp.TakeConstCRef(c);

  var n: i32;
  Cpp.TakeIntRef(ref n);
  Cpp.TakeIntRRef(42 as i32);
  Cpp.TakeConstIntRef(n);
}

// ============================================================================
// Reference return values
// ============================================================================

// --- return_references.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
class C {};

auto ReturnCRef() -> C&;
auto ReturnCRRef() -> C&&;
auto ReturnConstCRef() -> const C&;

auto ReturnIntRef() -> int&;
auto ReturnIntRRef() -> int&&;
auto ReturnConstIntRef() -> const int&;
''';

fn GetRefs() {
  var c1: Cpp.C* = Cpp.ReturnCRef();
  var c2: Cpp.C* = Cpp.ReturnCRRef();
  var c3: const Cpp.C* = Cpp.ReturnConstCRef();

  var n1: i32* = Cpp.ReturnIntRef();
  var n2: i32* = Cpp.ReturnIntRRef();
  var n3: const i32* = Cpp.ReturnConstIntRef();
}

// --- return_references_via_thunk.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
class C {};
class ForceThunk {};

auto ReturnCRef(ForceThunk = {}) -> C&;
auto ReturnCRRef(ForceThunk = {}) -> C&&;
auto ReturnConstCRef(ForceThunk = {}) -> const C&;

auto ReturnIntRef(ForceThunk = {}) -> int&;
auto ReturnIntRRef(ForceThunk = {}) -> int&&;
auto ReturnConstIntRef(ForceThunk = {}) -> const int&;
''';

fn GetRefs() {
  var c1: Cpp.C* = Cpp.ReturnCRef();
  var c2: Cpp.C* = Cpp.ReturnCRRef();
  var c3: const Cpp.C* = Cpp.ReturnConstCRef();

  var n1: i32* = Cpp.ReturnIntRef();
  var n2: i32* = Cpp.ReturnIntRRef();
  var n3: const i32* = Cpp.ReturnConstIntRef();
}

// CHECK:STDOUT: ; ModuleID = 'pass_references.carbon'
// CHECK:STDOUT: source_filename = "pass_references.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val.loc19_18.3 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassRefs.Main() #0 !dbg !7 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 8, !dbg !10
// CHECK:STDOUT:   %.loc19_18.2.temp = alloca {}, align 8, !dbg !11
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !12
// CHECK:STDOUT:   %.loc24_22.3.temp = alloca i32, align 4, !dbg !13
// CHECK:STDOUT:   %.loc25_23.2.temp = alloca i32, align 4, !dbg !14
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !10
// CHECK:STDOUT:   call void @_Z8TakeCRefR1C(ptr %c.var), !dbg !15
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc19_18.2.temp), !dbg !11
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %.loc19_18.2.temp, ptr align 1 @C.val.loc19_18.3, i64 0, i1 false), !dbg !11
// CHECK:STDOUT:   call void @_Z9TakeCRRefO1C.carbon_thunk(ptr %.loc19_18.2.temp), !dbg !16
// CHECK:STDOUT:   call void @_Z13TakeConstCRefRK1C.carbon_thunk(ptr %c.var), !dbg !17
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !12
// CHECK:STDOUT:   call void @_Z10TakeIntRefRi(ptr %n.var), !dbg !18
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc24_22.3.temp), !dbg !13
// CHECK:STDOUT:   store i32 42, ptr %.loc24_22.3.temp, align 4, !dbg !13
// CHECK:STDOUT:   call void @_Z11TakeIntRRefOi.carbon_thunk(ptr %.loc24_22.3.temp), !dbg !19
// CHECK:STDOUT:   %.loc25_23.1 = load i32, ptr %n.var, align 4, !dbg !14
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc25_23.2.temp), !dbg !14
// CHECK:STDOUT:   store i32 %.loc25_23.1, ptr %.loc25_23.2.temp, align 4, !dbg !14
// CHECK:STDOUT:   call void @_Z15TakeConstIntRefRKi.carbon_thunk(ptr %.loc25_23.2.temp), !dbg !20
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z8TakeCRefR1C(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z10TakeIntRefRi(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress
// CHECK:STDOUT: define dso_local void @_Z9TakeCRRefO1C.carbon_thunk(ptr %0) #3 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8
// CHECK:STDOUT:   call void @_Z9TakeCRRefO1C(ptr nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z9TakeCRRefO1C(ptr nonnull align 1 dereferenceable(1)) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress
// CHECK:STDOUT: define dso_local void @_Z13TakeConstCRefRK1C.carbon_thunk(ptr %0) #3 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8
// CHECK:STDOUT:   call void @_Z13TakeConstCRefRK1C(ptr nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z13TakeConstCRefRK1C(ptr nonnull align 1 dereferenceable(1)) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress
// CHECK:STDOUT: define dso_local void @_Z11TakeIntRRefOi.carbon_thunk(ptr %0) #3 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8
// CHECK:STDOUT:   call void @_Z11TakeIntRRefOi(ptr nonnull align 4 dereferenceable(4) %1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z11TakeIntRRefOi(ptr nonnull align 4 dereferenceable(4)) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress
// CHECK:STDOUT: define dso_local void @_Z15TakeConstIntRefRKi.carbon_thunk(ptr %0) #3 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8
// CHECK:STDOUT:   call void @_Z15TakeConstIntRefRKi(ptr nonnull align 4 dereferenceable(4) %1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z15TakeConstIntRefRKi(ptr nonnull align 4 dereferenceable(4)) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { alwaysinline mustprogress "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="0" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #4 = { "no-trapping-math"="true" "stack-protector-buffer-size"="0" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 1, !"wchar_size", i32 4}
// CHECK:STDOUT: !3 = !{i32 8, !"PIC Level", i32 0}
// CHECK:STDOUT: !4 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "pass_references.carbon", directory: "")
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "PassRefs", linkageName: "_CPassRefs.Main", scope: null, file: !6, line: 16, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !8 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !9 = !{}
// CHECK:STDOUT: !10 = !DILocation(line: 17, column: 3, scope: !7)
// CHECK:STDOUT: !11 = !DILocation(line: 19, column: 17, scope: !7)
// CHECK:STDOUT: !12 = !DILocation(line: 22, column: 3, scope: !7)
// CHECK:STDOUT: !13 = !DILocation(line: 24, column: 19, scope: !7)
// CHECK:STDOUT: !14 = !DILocation(line: 25, column: 23, scope: !7)
// CHECK:STDOUT: !15 = !DILocation(line: 18, column: 3, scope: !7)
// CHECK:STDOUT: !16 = !DILocation(line: 19, column: 3, scope: !7)
// CHECK:STDOUT: !17 = !DILocation(line: 20, column: 3, scope: !7)
// CHECK:STDOUT: !18 = !DILocation(line: 23, column: 3, scope: !7)
// CHECK:STDOUT: !19 = !DILocation(line: 24, column: 3, scope: !7)
// CHECK:STDOUT: !20 = !DILocation(line: 25, column: 3, scope: !7)
// CHECK:STDOUT: !21 = !DILocation(line: 16, column: 1, scope: !7)
// CHECK:STDOUT: ; ModuleID = 'pass_references_via_thunk.carbon'
// CHECK:STDOUT: source_filename = "pass_references_via_thunk.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %class.ForceThunk = type { i8 }
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val.loc20_18.3 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassRefs.Main() #0 !dbg !7 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 8, !dbg !10
// CHECK:STDOUT:   %.loc20_18.2.temp = alloca {}, align 8, !dbg !11
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !12
// CHECK:STDOUT:   %.loc25_22.3.temp = alloca i32, align 4, !dbg !13
// CHECK:STDOUT:   %.loc26_23.2.temp = alloca i32, align 4, !dbg !14
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !10
// CHECK:STDOUT:   call void @_Z8TakeCRefR1C10ForceThunk.carbon_thunk1(ptr %c.var), !dbg !15
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_18.2.temp), !dbg !11
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %.loc20_18.2.temp, ptr align 1 @C.val.loc20_18.3, i64 0, i1 false), !dbg !11
// CHECK:STDOUT:   call void @_Z9TakeCRRefRK1C10ForceThunk.carbon_thunk1(ptr %.loc20_18.2.temp), !dbg !16
// CHECK:STDOUT:   call void @_Z13TakeConstCRefRK1C10ForceThunk.carbon_thunk1(ptr %c.var), !dbg !17
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !12
// CHECK:STDOUT:   call void @_Z10TakeIntRefRi10ForceThunk.carbon_thunk1(ptr %n.var), !dbg !18
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc25_22.3.temp), !dbg !13
// CHECK:STDOUT:   store i32 42, ptr %.loc25_22.3.temp, align 4, !dbg !13
// CHECK:STDOUT:   call void @_Z11TakeIntRRefOi10ForceThunk.carbon_thunk1(ptr %.loc25_22.3.temp), !dbg !19
// CHECK:STDOUT:   %.loc26_23.1 = load i32, ptr %n.var, align 4, !dbg !14
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc26_23.2.temp), !dbg !14
// CHECK:STDOUT:   store i32 %.loc26_23.1, ptr %.loc26_23.2.temp, align 4, !dbg !14
// CHECK:STDOUT:   call void @_Z15TakeConstIntRefRKi10ForceThunk.carbon_thunk1(ptr %.loc26_23.2.temp), !dbg !20
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress
// CHECK:STDOUT: define dso_local void @_Z8TakeCRefR1C10ForceThunk.carbon_thunk1(ptr nonnull align 1 dereferenceable(1) %0) #3 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %class.ForceThunk, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !nonnull !9
// CHECK:STDOUT:   call void @_Z8TakeCRefR1C10ForceThunk(ptr nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z8TakeCRefR1C10ForceThunk(ptr nonnull align 1 dereferenceable(1)) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress
// CHECK:STDOUT: define dso_local void @_Z9TakeCRRefRK1C10ForceThunk.carbon_thunk1(ptr %0) #3 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %class.ForceThunk, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8
// CHECK:STDOUT:   call void @_Z9TakeCRRefRK1C10ForceThunk(ptr nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z9TakeCRRefRK1C10ForceThunk(ptr nonnull align 1 dereferenceable(1)) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress
// CHECK:STDOUT: define dso_local void @_Z13TakeConstCRefRK1C10ForceThunk.carbon_thunk1(ptr %0) #3 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %class.ForceThunk, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8
// CHECK:STDOUT:   call void @_Z13TakeConstCRefRK1C10ForceThunk(ptr nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z13TakeConstCRefRK1C10ForceThunk(ptr nonnull align 1 dereferenceable(1)) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress
// CHECK:STDOUT: define dso_local void @_Z10TakeIntRefRi10ForceThunk.carbon_thunk1(ptr nonnull align 4 dereferenceable(4) %0) #3 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %class.ForceThunk, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !nonnull !9, !align !22
// CHECK:STDOUT:   call void @_Z10TakeIntRefRi10ForceThunk(ptr nonnull align 4 dereferenceable(4) %1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z10TakeIntRefRi10ForceThunk(ptr nonnull align 4 dereferenceable(4)) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress
// CHECK:STDOUT: define dso_local void @_Z11TakeIntRRefOi10ForceThunk.carbon_thunk1(ptr %0) #3 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %class.ForceThunk, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8
// CHECK:STDOUT:   call void @_Z11TakeIntRRefOi10ForceThunk(ptr nonnull align 4 dereferenceable(4) %1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z11TakeIntRRefOi10ForceThunk(ptr nonnull align 4 dereferenceable(4)) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress
// CHECK:STDOUT: define dso_local void @_Z15TakeConstIntRefRKi10ForceThunk.carbon_thunk1(ptr %0) #3 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %class.ForceThunk, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8
// CHECK:STDOUT:   call void @_Z15TakeConstIntRefRKi10ForceThunk(ptr nonnull align 4 dereferenceable(4) %1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z15TakeConstIntRefRKi10ForceThunk(ptr nonnull align 4 dereferenceable(4)) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { alwaysinline mustprogress "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="0" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #4 = { "no-trapping-math"="true" "stack-protector-buffer-size"="0" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 1, !"wchar_size", i32 4}
// CHECK:STDOUT: !3 = !{i32 8, !"PIC Level", i32 0}
// CHECK:STDOUT: !4 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "pass_references_via_thunk.carbon", directory: "")
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "PassRefs", linkageName: "_CPassRefs.Main", scope: null, file: !6, line: 17, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !8 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !9 = !{}
// CHECK:STDOUT: !10 = !DILocation(line: 18, column: 3, scope: !7)
// CHECK:STDOUT: !11 = !DILocation(line: 20, column: 17, scope: !7)
// CHECK:STDOUT: !12 = !DILocation(line: 23, column: 3, scope: !7)
// CHECK:STDOUT: !13 = !DILocation(line: 25, column: 19, scope: !7)
// CHECK:STDOUT: !14 = !DILocation(line: 26, column: 23, scope: !7)
// CHECK:STDOUT: !15 = !DILocation(line: 19, column: 3, scope: !7)
// CHECK:STDOUT: !16 = !DILocation(line: 20, column: 3, scope: !7)
// CHECK:STDOUT: !17 = !DILocation(line: 21, column: 3, scope: !7)
// CHECK:STDOUT: !18 = !DILocation(line: 24, column: 3, scope: !7)
// CHECK:STDOUT: !19 = !DILocation(line: 25, column: 3, scope: !7)
// CHECK:STDOUT: !20 = !DILocation(line: 26, column: 3, scope: !7)
// CHECK:STDOUT: !21 = !DILocation(line: 17, column: 1, scope: !7)
// CHECK:STDOUT: !22 = !{i64 4}
// CHECK:STDOUT: ; ModuleID = 'return_references.carbon'
// CHECK:STDOUT: source_filename = "return_references.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CGetRefs.Main() #0 !dbg !7 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c1.var = alloca ptr, align 8, !dbg !10
// CHECK:STDOUT:   %c2.var = alloca ptr, align 8, !dbg !11
// CHECK:STDOUT:   %c3.var = alloca ptr, align 8, !dbg !12
// CHECK:STDOUT:   %n1.var = alloca ptr, align 8, !dbg !13
// CHECK:STDOUT:   %n2.var = alloca ptr, align 8, !dbg !14
// CHECK:STDOUT:   %n3.var = alloca ptr, align 8, !dbg !15
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c1.var), !dbg !10
// CHECK:STDOUT:   %ReturnCRef.call = call ptr @_Z10ReturnCRefv(), !dbg !16
// CHECK:STDOUT:   store ptr %ReturnCRef.call, ptr %c1.var, align 8, !dbg !10
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c2.var), !dbg !11
// CHECK:STDOUT:   %ReturnCRRef.call = call ptr @_Z11ReturnCRRefv(), !dbg !17
// CHECK:STDOUT:   store ptr %ReturnCRRef.call, ptr %c2.var, align 8, !dbg !11
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c3.var), !dbg !12
// CHECK:STDOUT:   %ReturnConstCRef.call = call ptr @_Z15ReturnConstCRefv(), !dbg !18
// CHECK:STDOUT:   store ptr %ReturnConstCRef.call, ptr %c3.var, align 8, !dbg !12
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n1.var), !dbg !13
// CHECK:STDOUT:   %ReturnIntRef.call = call ptr @_Z12ReturnIntRefv(), !dbg !19
// CHECK:STDOUT:   store ptr %ReturnIntRef.call, ptr %n1.var, align 8, !dbg !13
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n2.var), !dbg !14
// CHECK:STDOUT:   %ReturnIntRRef.call = call ptr @_Z13ReturnIntRRefv(), !dbg !20
// CHECK:STDOUT:   store ptr %ReturnIntRRef.call, ptr %n2.var, align 8, !dbg !14
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n3.var), !dbg !15
// CHECK:STDOUT:   %ReturnConstIntRef.call = call ptr @_Z17ReturnConstIntRefv(), !dbg !21
// CHECK:STDOUT:   store ptr %ReturnConstIntRef.call, ptr %n3.var, align 8, !dbg !15
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare ptr @_Z10ReturnCRefv()
// CHECK:STDOUT:
// CHECK:STDOUT: declare ptr @_Z11ReturnCRRefv()
// CHECK:STDOUT:
// CHECK:STDOUT: declare ptr @_Z15ReturnConstCRefv()
// CHECK:STDOUT:
// CHECK:STDOUT: declare ptr @_Z12ReturnIntRefv()
// CHECK:STDOUT:
// CHECK:STDOUT: declare ptr @_Z13ReturnIntRRefv()
// CHECK:STDOUT:
// CHECK:STDOUT: declare ptr @_Z17ReturnConstIntRefv()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 1, !"wchar_size", i32 4}
// CHECK:STDOUT: !3 = !{i32 8, !"PIC Level", i32 0}
// CHECK:STDOUT: !4 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "return_references.carbon", directory: "")
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "GetRefs", linkageName: "_CGetRefs.Main", scope: null, file: !6, line: 16, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !8 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !9 = !{}
// CHECK:STDOUT: !10 = !DILocation(line: 17, column: 3, scope: !7)
// CHECK:STDOUT: !11 = !DILocation(line: 18, column: 3, scope: !7)
// CHECK:STDOUT: !12 = !DILocation(line: 19, column: 3, scope: !7)
// CHECK:STDOUT: !13 = !DILocation(line: 21, column: 3, scope: !7)
// CHECK:STDOUT: !14 = !DILocation(line: 22, column: 3, scope: !7)
// CHECK:STDOUT: !15 = !DILocation(line: 23, column: 3, scope: !7)
// CHECK:STDOUT: !16 = !DILocation(line: 17, column: 20, scope: !7)
// CHECK:STDOUT: !17 = !DILocation(line: 18, column: 20, scope: !7)
// CHECK:STDOUT: !18 = !DILocation(line: 19, column: 26, scope: !7)
// CHECK:STDOUT: !19 = !DILocation(line: 21, column: 18, scope: !7)
// CHECK:STDOUT: !20 = !DILocation(line: 22, column: 18, scope: !7)
// CHECK:STDOUT: !21 = !DILocation(line: 23, column: 24, scope: !7)
// CHECK:STDOUT: !22 = !DILocation(line: 16, column: 1, scope: !7)
// CHECK:STDOUT: ; ModuleID = 'return_references_via_thunk.carbon'
// CHECK:STDOUT: source_filename = "return_references_via_thunk.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %class.ForceThunk = type { i8 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CGetRefs.Main() #0 !dbg !7 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c1.var = alloca ptr, align 8, !dbg !10
// CHECK:STDOUT:   %c2.var = alloca ptr, align 8, !dbg !11
// CHECK:STDOUT:   %c3.var = alloca ptr, align 8, !dbg !12
// CHECK:STDOUT:   %n1.var = alloca ptr, align 8, !dbg !13
// CHECK:STDOUT:   %n2.var = alloca ptr, align 8, !dbg !14
// CHECK:STDOUT:   %n3.var = alloca ptr, align 8, !dbg !15
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c1.var), !dbg !10
// CHECK:STDOUT:   %ReturnCRef__carbon_thunk.call = call ptr @_Z10ReturnCRef10ForceThunk.carbon_thunk0(), !dbg !16
// CHECK:STDOUT:   store ptr %ReturnCRef__carbon_thunk.call, ptr %c1.var, align 8, !dbg !10
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c2.var), !dbg !11
// CHECK:STDOUT:   %ReturnCRRef__carbon_thunk.call = call ptr @_Z11ReturnCRRef10ForceThunk.carbon_thunk0(), !dbg !17
// CHECK:STDOUT:   store ptr %ReturnCRRef__carbon_thunk.call, ptr %c2.var, align 8, !dbg !11
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c3.var), !dbg !12
// CHECK:STDOUT:   %ReturnConstCRef__carbon_thunk.call = call ptr @_Z15ReturnConstCRef10ForceThunk.carbon_thunk0(), !dbg !18
// CHECK:STDOUT:   store ptr %ReturnConstCRef__carbon_thunk.call, ptr %c3.var, align 8, !dbg !12
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n1.var), !dbg !13
// CHECK:STDOUT:   %ReturnIntRef__carbon_thunk.call = call ptr @_Z12ReturnIntRef10ForceThunk.carbon_thunk0(), !dbg !19
// CHECK:STDOUT:   store ptr %ReturnIntRef__carbon_thunk.call, ptr %n1.var, align 8, !dbg !13
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n2.var), !dbg !14
// CHECK:STDOUT:   %ReturnIntRRef__carbon_thunk.call = call ptr @_Z13ReturnIntRRef10ForceThunk.carbon_thunk0(), !dbg !20
// CHECK:STDOUT:   store ptr %ReturnIntRRef__carbon_thunk.call, ptr %n2.var, align 8, !dbg !14
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n3.var), !dbg !15
// CHECK:STDOUT:   %ReturnConstIntRef__carbon_thunk.call = call ptr @_Z17ReturnConstIntRef10ForceThunk.carbon_thunk0(), !dbg !21
// CHECK:STDOUT:   store ptr %ReturnConstIntRef__carbon_thunk.call, ptr %n3.var, align 8, !dbg !15
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress
// CHECK:STDOUT: define dso_local nonnull align 1 dereferenceable(1) ptr @_Z10ReturnCRef10ForceThunk.carbon_thunk0() #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %agg.tmp = alloca %class.ForceThunk, align 1
// CHECK:STDOUT:   %call = call nonnull align 1 dereferenceable(1) ptr @_Z10ReturnCRef10ForceThunk()
// CHECK:STDOUT:   ret ptr %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare nonnull align 1 dereferenceable(1) ptr @_Z10ReturnCRef10ForceThunk() #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress
// CHECK:STDOUT: define dso_local nonnull align 1 dereferenceable(1) ptr @_Z11ReturnCRRef10ForceThunk.carbon_thunk0() #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %agg.tmp = alloca %class.ForceThunk, align 1
// CHECK:STDOUT:   %call = call nonnull align 1 dereferenceable(1) ptr @_Z11ReturnCRRef10ForceThunk()
// CHECK:STDOUT:   ret ptr %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare nonnull align 1 dereferenceable(1) ptr @_Z11ReturnCRRef10ForceThunk() #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress
// CHECK:STDOUT: define dso_local nonnull align 1 dereferenceable(1) ptr @_Z15ReturnConstCRef10ForceThunk.carbon_thunk0() #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %agg.tmp = alloca %class.ForceThunk, align 1
// CHECK:STDOUT:   %call = call nonnull align 1 dereferenceable(1) ptr @_Z15ReturnConstCRef10ForceThunk()
// CHECK:STDOUT:   ret ptr %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare nonnull align 1 dereferenceable(1) ptr @_Z15ReturnConstCRef10ForceThunk() #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress
// CHECK:STDOUT: define dso_local nonnull align 4 dereferenceable(4) ptr @_Z12ReturnIntRef10ForceThunk.carbon_thunk0() #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %agg.tmp = alloca %class.ForceThunk, align 1
// CHECK:STDOUT:   %call = call nonnull align 4 dereferenceable(4) ptr @_Z12ReturnIntRef10ForceThunk()
// CHECK:STDOUT:   ret ptr %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare nonnull align 4 dereferenceable(4) ptr @_Z12ReturnIntRef10ForceThunk() #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress
// CHECK:STDOUT: define dso_local nonnull align 4 dereferenceable(4) ptr @_Z13ReturnIntRRef10ForceThunk.carbon_thunk0() #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %agg.tmp = alloca %class.ForceThunk, align 1
// CHECK:STDOUT:   %call = call nonnull align 4 dereferenceable(4) ptr @_Z13ReturnIntRRef10ForceThunk()
// CHECK:STDOUT:   ret ptr %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare nonnull align 4 dereferenceable(4) ptr @_Z13ReturnIntRRef10ForceThunk() #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress
// CHECK:STDOUT: define dso_local nonnull align 4 dereferenceable(4) ptr @_Z17ReturnConstIntRef10ForceThunk.carbon_thunk0() #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %agg.tmp = alloca %class.ForceThunk, align 1
// CHECK:STDOUT:   %call = call nonnull align 4 dereferenceable(4) ptr @_Z17ReturnConstIntRef10ForceThunk()
// CHECK:STDOUT:   ret ptr %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare nonnull align 4 dereferenceable(4) ptr @_Z17ReturnConstIntRef10ForceThunk() #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 5, 4, 3, 2, 1, 0 }
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
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "return_references_via_thunk.carbon", directory: "")
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "GetRefs", linkageName: "_CGetRefs.Main", scope: null, file: !6, line: 17, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !8 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !9 = !{}
// CHECK:STDOUT: !10 = !DILocation(line: 18, column: 3, scope: !7)
// CHECK:STDOUT: !11 = !DILocation(line: 19, column: 3, scope: !7)
// CHECK:STDOUT: !12 = !DILocation(line: 20, column: 3, scope: !7)
// CHECK:STDOUT: !13 = !DILocation(line: 22, column: 3, scope: !7)
// CHECK:STDOUT: !14 = !DILocation(line: 23, column: 3, scope: !7)
// CHECK:STDOUT: !15 = !DILocation(line: 24, column: 3, scope: !7)
// CHECK:STDOUT: !16 = !DILocation(line: 18, column: 20, scope: !7)
// CHECK:STDOUT: !17 = !DILocation(line: 19, column: 20, scope: !7)
// CHECK:STDOUT: !18 = !DILocation(line: 20, column: 26, scope: !7)
// CHECK:STDOUT: !19 = !DILocation(line: 22, column: 18, scope: !7)
// CHECK:STDOUT: !20 = !DILocation(line: 23, column: 18, scope: !7)
// CHECK:STDOUT: !21 = !DILocation(line: 24, column: 24, scope: !7)
// CHECK:STDOUT: !22 = !DILocation(line: 17, column: 1, scope: !7)
