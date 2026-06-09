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
  let ref _: Cpp.C = Cpp.ReturnCRef();
  let ref _: Cpp.C = Cpp.ReturnCRRef();
  let ref _: const Cpp.C = Cpp.ReturnConstCRef();

  let ref _: i32 = Cpp.ReturnIntRef();
  let ref _: i32 = Cpp.ReturnIntRRef();
  let ref _: const i32 = Cpp.ReturnConstIntRef();
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
  let ref _: Cpp.C = Cpp.ReturnCRef();
  let ref _: Cpp.C = Cpp.ReturnCRRef();
  let ref _: const Cpp.C = Cpp.ReturnConstCRef();

  let ref _: i32 = Cpp.ReturnIntRef();
  let ref _: i32 = Cpp.ReturnIntRRef();
  let ref _: const i32 = Cpp.ReturnConstIntRef();
}

// CHECK:STDOUT: ; ModuleID = 'pass_references.carbon'
// CHECK:STDOUT: source_filename = "pass_references.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val.loc19_20 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassRefs.Main() #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !14
// CHECK:STDOUT:   %_.var.1 = alloca {}, align 1, !dbg !15
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !16
// CHECK:STDOUT:   %_.var.2 = alloca i32, align 4, !dbg !17
// CHECK:STDOUT:   %.loc25_23.2.temp = alloca i32, align 4, !dbg !18
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !14
// CHECK:STDOUT:   call void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.67b28f83c7cfd60c"(ptr %c.var), !dbg !14
// CHECK:STDOUT:   call void @_Z8TakeCRefR1C(ptr %c.var), !dbg !19
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.1), !dbg !15
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %_.var.1, ptr align 1 @C.val.loc19_20, i64 0, i1 false), !dbg !15
// CHECK:STDOUT:   call void @_Z9TakeCRRefO1C.carbon_thunk.v(ptr %_.var.1), !dbg !20
// CHECK:STDOUT:   call void @_Z13TakeConstCRefRK1C.carbon_thunk._(ptr %c.var), !dbg !21
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !16
// CHECK:STDOUT:   store i32 poison, ptr %n.var, align 4, !dbg !16
// CHECK:STDOUT:   call void @_Z10TakeIntRefRi(ptr %n.var), !dbg !22
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.2), !dbg !17
// CHECK:STDOUT:   store i32 42, ptr %_.var.2, align 4, !dbg !17
// CHECK:STDOUT:   call void @_Z11TakeIntRRefOi.carbon_thunk.v(ptr %_.var.2), !dbg !23
// CHECK:STDOUT:   %.loc25_23.1 = load i32, ptr %n.var, align 4, !dbg !18
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc25_23.2.temp), !dbg !18
// CHECK:STDOUT:   store i32 %.loc25_23.1, ptr %.loc25_23.2.temp, align 4, !dbg !18
// CHECK:STDOUT:   call void @_Z15TakeConstIntRefRKi.carbon_thunk._(ptr %.loc25_23.2.temp), !dbg !24
// CHECK:STDOUT:   call void @"_COp.5e27612b9dd31a14:core.Destroy.Core"(ptr %.loc25_23.2.temp), !dbg !18
// CHECK:STDOUT:   call void @"_COp.5e27612b9dd31a14:core.Destroy.Core"(ptr %_.var.2), !dbg !17
// CHECK:STDOUT:   call void @"_COp.5e27612b9dd31a14:core.Destroy.Core"(ptr %n.var), !dbg !16
// CHECK:STDOUT:   ret void, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN1CC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !26
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !26
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define void @"_COp:thunk:Default.dc3dd9c349162a91.Core:C.Cpp"(ptr sret({}) %return) #2 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN1CC1Ev.carbon_thunk.(ptr %return), !dbg !33
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z8TakeCRefR1C(ptr noundef nonnull align 1 dereferenceable(1)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z9TakeCRRefO1C.carbon_thunk.v(ptr noundef %0) #4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !26
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !26
// CHECK:STDOUT:   call void @_Z9TakeCRRefO1C(ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z13TakeConstCRefRK1C.carbon_thunk._(ptr noundef %0) #4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !26
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !26
// CHECK:STDOUT:   call void @_Z13TakeConstCRefRK1C(ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z10TakeIntRefRi(ptr noundef nonnull align 4 dereferenceable(4)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z11TakeIntRRefOi.carbon_thunk.v(ptr noundef %0) #4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !34
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !34
// CHECK:STDOUT:   call void @_Z11TakeIntRRefOi(ptr noundef nonnull align 4 dereferenceable(4) %1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z15TakeConstIntRefRKi.carbon_thunk._(ptr noundef %0) #4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !34
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !34
// CHECK:STDOUT:   call void @_Z15TakeConstIntRefRKi(ptr noundef nonnull align 4 dereferenceable(4) %1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.5e27612b9dd31a14:core.Destroy.Core"(ptr %self) #0 !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #5
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.67b28f83c7cfd60c"(ptr sret({}) %return) #0 !dbg !43 {
// CHECK:STDOUT:   call void @"_COp:thunk:Default.dc3dd9c349162a91.Core:C.Cpp"(ptr %return), !dbg !45
// CHECK:STDOUT:   ret void, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #5
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z9TakeCRRefO1C(ptr noundef nonnull align 1 dereferenceable(1)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z13TakeConstCRefRK1C(ptr noundef nonnull align 1 dereferenceable(1)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z11TakeIntRRefOi(ptr noundef nonnull align 4 dereferenceable(4)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z15TakeConstIntRefRKi(ptr noundef nonnull align 4 dereferenceable(4)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_COp.5e27612b9dd31a14:core.Destroy.Core", { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #3 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #4 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #5 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
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
// CHECK:STDOUT: !6 = !DIFile(filename: "pass_references.carbon", directory: "")
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "PassRefs", linkageName: "_CPassRefs.Main", scope: null, file: !6, line: 16, type: !12, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{null}
// CHECK:STDOUT: !14 = !DILocation(line: 17, column: 3, scope: !11)
// CHECK:STDOUT: !15 = !DILocation(line: 8, column: 19, scope: !11)
// CHECK:STDOUT: !16 = !DILocation(line: 22, column: 3, scope: !11)
// CHECK:STDOUT: !17 = !DILocation(line: 12, column: 23, scope: !11)
// CHECK:STDOUT: !18 = !DILocation(line: 25, column: 23, scope: !11)
// CHECK:STDOUT: !19 = !DILocation(line: 18, column: 3, scope: !11)
// CHECK:STDOUT: !20 = !DILocation(line: 19, column: 3, scope: !11)
// CHECK:STDOUT: !21 = !DILocation(line: 20, column: 3, scope: !11)
// CHECK:STDOUT: !22 = !DILocation(line: 23, column: 3, scope: !11)
// CHECK:STDOUT: !23 = !DILocation(line: 24, column: 3, scope: !11)
// CHECK:STDOUT: !24 = !DILocation(line: 25, column: 3, scope: !11)
// CHECK:STDOUT: !25 = !DILocation(line: 16, column: 1, scope: !11)
// CHECK:STDOUT: !26 = !{!27, !27, i64 0}
// CHECK:STDOUT: !27 = !{!"p1 _ZTS1C", !28, i64 0}
// CHECK:STDOUT: !28 = !{!"any pointer", !9, i64 0}
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Default.dc3dd9c349162a91.Core:C.Cpp", scope: null, file: !6, line: 5, type: !30, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !30 = !DISubroutineType(types: !31)
// CHECK:STDOUT: !31 = !{!32}
// CHECK:STDOUT: !32 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !33 = !DILocation(line: 5, column: 7, scope: !29)
// CHECK:STDOUT: !34 = !{!35, !35, i64 0}
// CHECK:STDOUT: !35 = !{!"p1 int", !28, i64 0}
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "Op", linkageName: "_COp.5e27612b9dd31a14:core.Destroy.Core", scope: null, file: !6, line: 25, type: !37, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !40)
// CHECK:STDOUT: !37 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !38 = !{null, !39}
// CHECK:STDOUT: !39 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !40 = !{!41}
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !36, type: !39)
// CHECK:STDOUT: !42 = !DILocation(line: 25, column: 23, scope: !36)
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "Op", linkageName: "_COp.51f3fb62cb986374:DefaultOrUnformed.Core.67b28f83c7cfd60c", scope: null, file: !44, line: 9, type: !30, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !44 = !DIFile(filename: "min_prelude/parts/default.carbon", directory: "")
// CHECK:STDOUT: !45 = !DILocation(line: 9, column: 28, scope: !43)
// CHECK:STDOUT: !46 = !DILocation(line: 9, column: 21, scope: !43)
// CHECK:STDOUT: ; ModuleID = 'pass_references_via_thunk.carbon'
// CHECK:STDOUT: source_filename = "pass_references_via_thunk.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %class.ForceThunk = type { i8 }
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val = internal constant {} zeroinitializer
// CHECK:STDOUT: @C.val.loc20_20.3 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassRefs.Main() #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !14
// CHECK:STDOUT:   %.loc20_18.2.temp = alloca {}, align 1, !dbg !15
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !16
// CHECK:STDOUT:   %_.var = alloca i32, align 4, !dbg !17
// CHECK:STDOUT:   %.loc26_23.2.temp = alloca i32, align 4, !dbg !18
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !14
// CHECK:STDOUT:   call void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.67b28f83c7cfd60c"(ptr %c.var), !dbg !14
// CHECK:STDOUT:   call void @_Z8TakeCRefR1C10ForceThunk.carbon_thunk.r(ptr %c.var), !dbg !19
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_18.2.temp), !dbg !15
// CHECK:STDOUT:   call void @_Z9TakeCRRefRK1C10ForceThunk.carbon_thunk._(ptr @C.val.loc20_20.3), !dbg !20
// CHECK:STDOUT:   call void @_Z13TakeConstCRefRK1C10ForceThunk.carbon_thunk._(ptr %c.var), !dbg !21
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !16
// CHECK:STDOUT:   store i32 poison, ptr %n.var, align 4, !dbg !16
// CHECK:STDOUT:   call void @_Z10TakeIntRefRi10ForceThunk.carbon_thunk.r(ptr %n.var), !dbg !22
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !17
// CHECK:STDOUT:   store i32 42, ptr %_.var, align 4, !dbg !17
// CHECK:STDOUT:   call void @_Z11TakeIntRRefOi10ForceThunk.carbon_thunk.v(ptr %_.var), !dbg !23
// CHECK:STDOUT:   %.loc26_23.1 = load i32, ptr %n.var, align 4, !dbg !18
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc26_23.2.temp), !dbg !18
// CHECK:STDOUT:   store i32 %.loc26_23.1, ptr %.loc26_23.2.temp, align 4, !dbg !18
// CHECK:STDOUT:   call void @_Z15TakeConstIntRefRKi10ForceThunk.carbon_thunk._(ptr %.loc26_23.2.temp), !dbg !24
// CHECK:STDOUT:   call void @"_COp.5e27612b9dd31a14:core.Destroy.Core"(ptr %.loc26_23.2.temp), !dbg !18
// CHECK:STDOUT:   call void @"_COp.5e27612b9dd31a14:core.Destroy.Core"(ptr %_.var), !dbg !17
// CHECK:STDOUT:   call void @"_COp.5e27612b9dd31a14:core.Destroy.Core"(ptr %n.var), !dbg !16
// CHECK:STDOUT:   ret void, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN1CC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !26
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !26
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define void @"_COp:thunk:Default.dc3dd9c349162a91.Core:C.Cpp"(ptr sret({}) %return) #2 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN1CC1Ev.carbon_thunk.(ptr %return), !dbg !33
// CHECK:STDOUT:   ret void, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z8TakeCRefR1C10ForceThunk.carbon_thunk.r(ptr noundef nonnull align 1 dereferenceable(1) %0) #3 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %class.ForceThunk, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !26
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !26, !nonnull !34
// CHECK:STDOUT:   call void @_Z8TakeCRefR1C10ForceThunk(ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z9TakeCRRefRK1C10ForceThunk.carbon_thunk._(ptr noundef %0) #3 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %class.ForceThunk, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !26
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !26
// CHECK:STDOUT:   call void @_Z9TakeCRRefRK1C10ForceThunk(ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z13TakeConstCRefRK1C10ForceThunk.carbon_thunk._(ptr noundef %0) #3 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %class.ForceThunk, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !26
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !26
// CHECK:STDOUT:   call void @_Z13TakeConstCRefRK1C10ForceThunk(ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z10TakeIntRefRi10ForceThunk.carbon_thunk.r(ptr noundef nonnull align 4 dereferenceable(4) %0) #3 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %class.ForceThunk, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !35
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !35, !nonnull !34, !align !37
// CHECK:STDOUT:   call void @_Z10TakeIntRefRi10ForceThunk(ptr noundef nonnull align 4 dereferenceable(4) %1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z11TakeIntRRefOi10ForceThunk.carbon_thunk.v(ptr noundef %0) #3 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %class.ForceThunk, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !35
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !35
// CHECK:STDOUT:   call void @_Z11TakeIntRRefOi10ForceThunk(ptr noundef nonnull align 4 dereferenceable(4) %1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z15TakeConstIntRefRKi10ForceThunk.carbon_thunk._(ptr noundef %0) #3 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %class.ForceThunk, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !35
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !35
// CHECK:STDOUT:   call void @_Z15TakeConstIntRefRKi10ForceThunk(ptr noundef nonnull align 4 dereferenceable(4) %1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.5e27612b9dd31a14:core.Destroy.Core"(ptr %self) #0 !dbg !38 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.67b28f83c7cfd60c"(ptr sret({}) %return) #0 !dbg !45 {
// CHECK:STDOUT:   call void @"_COp:thunk:Default.dc3dd9c349162a91.Core:C.Cpp"(ptr %return), !dbg !47
// CHECK:STDOUT:   ret void, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z8TakeCRefR1C10ForceThunk(ptr noundef nonnull align 1 dereferenceable(1)) #5
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z9TakeCRRefRK1C10ForceThunk(ptr noundef nonnull align 1 dereferenceable(1)) #5
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z13TakeConstCRefRK1C10ForceThunk(ptr noundef nonnull align 1 dereferenceable(1)) #5
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z10TakeIntRefRi10ForceThunk(ptr noundef nonnull align 4 dereferenceable(4)) #5
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z11TakeIntRRefOi10ForceThunk(ptr noundef nonnull align 4 dereferenceable(4)) #5
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z15TakeConstIntRefRKi10ForceThunk(ptr noundef nonnull align 4 dereferenceable(4)) #5
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_COp.5e27612b9dd31a14:core.Destroy.Core", { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #3 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #4 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #5 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
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
// CHECK:STDOUT: !6 = !DIFile(filename: "pass_references_via_thunk.carbon", directory: "")
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "PassRefs", linkageName: "_CPassRefs.Main", scope: null, file: !6, line: 17, type: !12, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{null}
// CHECK:STDOUT: !14 = !DILocation(line: 18, column: 3, scope: !11)
// CHECK:STDOUT: !15 = !DILocation(line: 20, column: 17, scope: !11)
// CHECK:STDOUT: !16 = !DILocation(line: 23, column: 3, scope: !11)
// CHECK:STDOUT: !17 = !DILocation(line: 13, column: 23, scope: !11)
// CHECK:STDOUT: !18 = !DILocation(line: 26, column: 23, scope: !11)
// CHECK:STDOUT: !19 = !DILocation(line: 19, column: 3, scope: !11)
// CHECK:STDOUT: !20 = !DILocation(line: 20, column: 3, scope: !11)
// CHECK:STDOUT: !21 = !DILocation(line: 21, column: 3, scope: !11)
// CHECK:STDOUT: !22 = !DILocation(line: 24, column: 3, scope: !11)
// CHECK:STDOUT: !23 = !DILocation(line: 25, column: 3, scope: !11)
// CHECK:STDOUT: !24 = !DILocation(line: 26, column: 3, scope: !11)
// CHECK:STDOUT: !25 = !DILocation(line: 17, column: 1, scope: !11)
// CHECK:STDOUT: !26 = !{!27, !27, i64 0}
// CHECK:STDOUT: !27 = !{!"p1 _ZTS1C", !28, i64 0}
// CHECK:STDOUT: !28 = !{!"any pointer", !9, i64 0}
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Default.dc3dd9c349162a91.Core:C.Cpp", scope: null, file: !6, line: 5, type: !30, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !30 = !DISubroutineType(types: !31)
// CHECK:STDOUT: !31 = !{!32}
// CHECK:STDOUT: !32 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !33 = !DILocation(line: 5, column: 7, scope: !29)
// CHECK:STDOUT: !34 = !{}
// CHECK:STDOUT: !35 = !{!36, !36, i64 0}
// CHECK:STDOUT: !36 = !{!"p1 int", !28, i64 0}
// CHECK:STDOUT: !37 = !{i64 4}
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "Op", linkageName: "_COp.5e27612b9dd31a14:core.Destroy.Core", scope: null, file: !6, line: 26, type: !39, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !42)
// CHECK:STDOUT: !39 = !DISubroutineType(types: !40)
// CHECK:STDOUT: !40 = !{null, !41}
// CHECK:STDOUT: !41 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !42 = !{!43}
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !38, type: !41)
// CHECK:STDOUT: !44 = !DILocation(line: 26, column: 23, scope: !38)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "Op", linkageName: "_COp.51f3fb62cb986374:DefaultOrUnformed.Core.67b28f83c7cfd60c", scope: null, file: !46, line: 9, type: !30, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !46 = !DIFile(filename: "min_prelude/parts/default.carbon", directory: "")
// CHECK:STDOUT: !47 = !DILocation(line: 9, column: 28, scope: !45)
// CHECK:STDOUT: !48 = !DILocation(line: 9, column: 21, scope: !45)
// CHECK:STDOUT: ; ModuleID = 'return_references.carbon'
// CHECK:STDOUT: source_filename = "return_references.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CGetRefs.Main() #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnCRef.call = call ptr @_Z10ReturnCRefv(), !dbg !14
// CHECK:STDOUT:   %ReturnCRRef.call = call ptr @_Z11ReturnCRRefv(), !dbg !15
// CHECK:STDOUT:   %ReturnConstCRef.call = call ptr @_Z15ReturnConstCRefv(), !dbg !16
// CHECK:STDOUT:   %ReturnIntRef.call = call ptr @_Z12ReturnIntRefv(), !dbg !17
// CHECK:STDOUT:   %ReturnIntRRef.call = call ptr @_Z13ReturnIntRRefv(), !dbg !18
// CHECK:STDOUT:   %ReturnConstIntRef.call = call ptr @_Z17ReturnConstIntRefv(), !dbg !19
// CHECK:STDOUT:   ret void, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef nonnull align 1 dereferenceable(1) ptr @_Z10ReturnCRefv() #1
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef nonnull align 1 dereferenceable(1) ptr @_Z11ReturnCRRefv() #1
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef nonnull align 1 dereferenceable(1) ptr @_Z15ReturnConstCRefv() #1
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef nonnull align 4 dereferenceable(4) ptr @_Z12ReturnIntRefv() #1
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef nonnull align 4 dereferenceable(4) ptr @_Z13ReturnIntRRefv() #1
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef nonnull align 4 dereferenceable(4) ptr @_Z17ReturnConstIntRefv() #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
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
// CHECK:STDOUT: !6 = !DIFile(filename: "return_references.carbon", directory: "")
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "GetRefs", linkageName: "_CGetRefs.Main", scope: null, file: !6, line: 16, type: !12, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{null}
// CHECK:STDOUT: !14 = !DILocation(line: 17, column: 22, scope: !11)
// CHECK:STDOUT: !15 = !DILocation(line: 18, column: 22, scope: !11)
// CHECK:STDOUT: !16 = !DILocation(line: 19, column: 28, scope: !11)
// CHECK:STDOUT: !17 = !DILocation(line: 21, column: 20, scope: !11)
// CHECK:STDOUT: !18 = !DILocation(line: 22, column: 20, scope: !11)
// CHECK:STDOUT: !19 = !DILocation(line: 23, column: 26, scope: !11)
// CHECK:STDOUT: !20 = !DILocation(line: 16, column: 1, scope: !11)
// CHECK:STDOUT: ; ModuleID = 'return_references_via_thunk.carbon'
// CHECK:STDOUT: source_filename = "return_references_via_thunk.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %class.ForceThunk = type { i8 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CGetRefs.Main() #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnCRef__carbon_thunk.call = call ptr @_Z10ReturnCRef10ForceThunk.carbon_thunk.(), !dbg !14
// CHECK:STDOUT:   %ReturnCRRef__carbon_thunk.call = call ptr @_Z11ReturnCRRef10ForceThunk.carbon_thunk.(), !dbg !15
// CHECK:STDOUT:   %ReturnConstCRef__carbon_thunk.call = call ptr @_Z15ReturnConstCRef10ForceThunk.carbon_thunk.(), !dbg !16
// CHECK:STDOUT:   %ReturnIntRef__carbon_thunk.call = call ptr @_Z12ReturnIntRef10ForceThunk.carbon_thunk.(), !dbg !17
// CHECK:STDOUT:   %ReturnIntRRef__carbon_thunk.call = call ptr @_Z13ReturnIntRRef10ForceThunk.carbon_thunk.(), !dbg !18
// CHECK:STDOUT:   %ReturnConstIntRef__carbon_thunk.call = call ptr @_Z17ReturnConstIntRef10ForceThunk.carbon_thunk.(), !dbg !19
// CHECK:STDOUT:   ret void, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal noundef nonnull align 1 dereferenceable(1) ptr @_Z10ReturnCRef10ForceThunk.carbon_thunk.() #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %agg.tmp = alloca %class.ForceThunk, align 1
// CHECK:STDOUT:   %call = call noundef nonnull align 1 dereferenceable(1) ptr @_Z10ReturnCRef10ForceThunk()
// CHECK:STDOUT:   ret ptr %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal noundef nonnull align 1 dereferenceable(1) ptr @_Z11ReturnCRRef10ForceThunk.carbon_thunk.() #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %agg.tmp = alloca %class.ForceThunk, align 1
// CHECK:STDOUT:   %call = call noundef nonnull align 1 dereferenceable(1) ptr @_Z11ReturnCRRef10ForceThunk()
// CHECK:STDOUT:   ret ptr %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal noundef nonnull align 1 dereferenceable(1) ptr @_Z15ReturnConstCRef10ForceThunk.carbon_thunk.() #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %agg.tmp = alloca %class.ForceThunk, align 1
// CHECK:STDOUT:   %call = call noundef nonnull align 1 dereferenceable(1) ptr @_Z15ReturnConstCRef10ForceThunk()
// CHECK:STDOUT:   ret ptr %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal noundef nonnull align 4 dereferenceable(4) ptr @_Z12ReturnIntRef10ForceThunk.carbon_thunk.() #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %agg.tmp = alloca %class.ForceThunk, align 1
// CHECK:STDOUT:   %call = call noundef nonnull align 4 dereferenceable(4) ptr @_Z12ReturnIntRef10ForceThunk()
// CHECK:STDOUT:   ret ptr %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal noundef nonnull align 4 dereferenceable(4) ptr @_Z13ReturnIntRRef10ForceThunk.carbon_thunk.() #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %agg.tmp = alloca %class.ForceThunk, align 1
// CHECK:STDOUT:   %call = call noundef nonnull align 4 dereferenceable(4) ptr @_Z13ReturnIntRRef10ForceThunk()
// CHECK:STDOUT:   ret ptr %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal noundef nonnull align 4 dereferenceable(4) ptr @_Z17ReturnConstIntRef10ForceThunk.carbon_thunk.() #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %agg.tmp = alloca %class.ForceThunk, align 1
// CHECK:STDOUT:   %call = call noundef nonnull align 4 dereferenceable(4) ptr @_Z17ReturnConstIntRef10ForceThunk()
// CHECK:STDOUT:   ret ptr %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef nonnull align 1 dereferenceable(1) ptr @_Z10ReturnCRef10ForceThunk() #2
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef nonnull align 1 dereferenceable(1) ptr @_Z11ReturnCRRef10ForceThunk() #2
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef nonnull align 1 dereferenceable(1) ptr @_Z15ReturnConstCRef10ForceThunk() #2
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef nonnull align 4 dereferenceable(4) ptr @_Z12ReturnIntRef10ForceThunk() #2
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef nonnull align 4 dereferenceable(4) ptr @_Z13ReturnIntRRef10ForceThunk() #2
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef nonnull align 4 dereferenceable(4) ptr @_Z17ReturnConstIntRef10ForceThunk() #2
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
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
// CHECK:STDOUT: !6 = !DIFile(filename: "return_references_via_thunk.carbon", directory: "")
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "GetRefs", linkageName: "_CGetRefs.Main", scope: null, file: !6, line: 17, type: !12, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{null}
// CHECK:STDOUT: !14 = !DILocation(line: 18, column: 22, scope: !11)
// CHECK:STDOUT: !15 = !DILocation(line: 19, column: 22, scope: !11)
// CHECK:STDOUT: !16 = !DILocation(line: 20, column: 28, scope: !11)
// CHECK:STDOUT: !17 = !DILocation(line: 22, column: 20, scope: !11)
// CHECK:STDOUT: !18 = !DILocation(line: 23, column: 20, scope: !11)
// CHECK:STDOUT: !19 = !DILocation(line: 24, column: 26, scope: !11)
// CHECK:STDOUT: !20 = !DILocation(line: 17, column: 1, scope: !11)
