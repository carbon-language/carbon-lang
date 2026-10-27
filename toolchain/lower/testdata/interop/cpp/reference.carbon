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

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'pass_references.carbon'
// CHECK:STDOUT: source_filename = "pass_references.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val.loc19_20 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassRefs.Main() #0 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !31
// CHECK:STDOUT:   %_.var.1 = alloca {}, align 1, !dbg !36
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !40
// CHECK:STDOUT:   %_.var.2 = alloca i32, align 4, !dbg !45
// CHECK:STDOUT:   %.loc25_23.2.temp = alloca i32, align 4, !dbg !47
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !31
// CHECK:STDOUT:   call void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.67b28f83c7cfd60c"(ptr %c.var), !dbg !31
// CHECK:STDOUT:   call void @_Z8TakeCRefR1C(ptr %c.var), !dbg !35
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.1), !dbg !36
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %_.var.1, ptr align 1 @C.val.loc19_20, i64 0, i1 false), !dbg !36
// CHECK:STDOUT:   call void @_Z9TakeCRRefO1C.carbon_thunk.v(ptr %_.var.1), !dbg !37
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.2a99575b423907d2(ptr %_.var.1), !dbg !36
// CHECK:STDOUT:   call void @_Z13TakeConstCRefRK1C.carbon_thunk._(ptr %c.var), !dbg !39
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !40
// CHECK:STDOUT:   store i32 poison, ptr %n.var, align 4, !dbg !40
// CHECK:STDOUT:   call void @_Z10TakeIntRefRi(ptr %n.var), !dbg !44
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.2), !dbg !45
// CHECK:STDOUT:   store i32 42, ptr %_.var.2, align 4, !dbg !45
// CHECK:STDOUT:   call void @_Z11TakeIntRRefOi.carbon_thunk.v(ptr %_.var.2), !dbg !46
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %_.var.2), !dbg !45
// CHECK:STDOUT:   %.loc25_23.1 = load i32, ptr %n.var, align 4, !dbg !47
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc25_23.2.temp), !dbg !47
// CHECK:STDOUT:   store i32 %.loc25_23.1, ptr %.loc25_23.2.temp, align 4, !dbg !47
// CHECK:STDOUT:   call void @_Z15TakeConstIntRefRKi.carbon_thunk._(ptr %.loc25_23.2.temp), !dbg !48
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc25_23.2.temp), !dbg !47
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %n.var), !dbg !40
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.2a99575b423907d2(ptr %c.var), !dbg !31
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN1CC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !76
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !76
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Default.dc3dd9c349162a91.Core:C.Cpp"(ptr sret({}) %return) #2 !dbg !50 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN1CC1Ev.carbon_thunk.(ptr %return), !dbg !52
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z8TakeCRefR1C(ptr noundef nonnull align 1 dereferenceable(1)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z9TakeCRRefO1C.carbon_thunk.v(ptr noundef %0) #4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !76
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !76
// CHECK:STDOUT:   call void @_Z9TakeCRRefO1C(ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z13TakeConstCRefRK1C.carbon_thunk._(ptr noundef %0) #4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !76
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !76
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
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !80
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !80
// CHECK:STDOUT:   call void @_Z11TakeIntRRefOi(ptr noundef nonnull align 4 dereferenceable(4) %1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z15TakeConstIntRefRKi.carbon_thunk._(ptr noundef %0) #4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !80
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !80
// CHECK:STDOUT:   call void @_Z15TakeConstIntRefRKi(ptr noundef nonnull align 4 dereferenceable(4) %1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #5
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.67b28f83c7cfd60c"(ptr sret({}) %return) #0 !dbg !53 {
// CHECK:STDOUT:   call void @"_COp:thunk:Default.dc3dd9c349162a91.Core:C.Cpp"(ptr %return), !dbg !56
// CHECK:STDOUT:   ret void, !dbg !57
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #5
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.2a99575b423907d2(ptr %t) #0 !dbg !58 {
// CHECK:STDOUT:   ret void, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f127f30143696d51(ptr %t) #0 !dbg !65 {
// CHECK:STDOUT:   ret void, !dbg !70
// CHECK:STDOUT: }
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
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.2a99575b423907d2, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.f127f30143696d51, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #3 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #4 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #5 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!81, !82, !83, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!85}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "min_prelude/parts/default.carbon", directory: "")
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !18 = !DIFile(filename: "pass_references.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{null}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{null, !23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{!23}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !28 = !{null, !20}
// CHECK:STDOUT: !29 = !DISubroutineType(types: !28)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "PassRefs", linkageName: "_CPassRefs.Main", scope: null, file: !18, line: 16, type: !22, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !31 = !DILocation(line: 17, column: 3, scope: !30)
// CHECK:STDOUT: !35 = !DILocation(line: 18, column: 3, scope: !30)
// CHECK:STDOUT: !36 = !DILocation(line: 8, column: 19, scope: !30)
// CHECK:STDOUT: !37 = !DILocation(line: 19, column: 3, scope: !30)
// CHECK:STDOUT: !39 = !DILocation(line: 20, column: 3, scope: !30)
// CHECK:STDOUT: !40 = !DILocation(line: 22, column: 3, scope: !30)
// CHECK:STDOUT: !44 = !DILocation(line: 23, column: 3, scope: !30)
// CHECK:STDOUT: !45 = !DILocation(line: 12, column: 23, scope: !30)
// CHECK:STDOUT: !46 = !DILocation(line: 24, column: 3, scope: !30)
// CHECK:STDOUT: !47 = !DILocation(line: 25, column: 23, scope: !30)
// CHECK:STDOUT: !48 = !DILocation(line: 25, column: 3, scope: !30)
// CHECK:STDOUT: !49 = !DILocation(line: 16, column: 1, scope: !30)
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Default.dc3dd9c349162a91.Core:C.Cpp", scope: null, file: !18, line: 5, type: !27, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !52 = !DILocation(line: 5, column: 7, scope: !50)
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "Op", linkageName: "_COp.51f3fb62cb986374:DefaultOrUnformed.Core.67b28f83c7cfd60c", scope: null, file: !4, line: 9, type: !27, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !56 = !DILocation(line: 9, column: 28, scope: !53)
// CHECK:STDOUT: !57 = !DILocation(line: 9, column: 21, scope: !53)
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.2a99575b423907d2", scope: null, file: !10, line: 32, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !64)
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !58, type: !23)
// CHECK:STDOUT: !63 = !DILocation(line: 32, column: 1, scope: !58)
// CHECK:STDOUT: !64 = !{!59}
// CHECK:STDOUT: !65 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f127f30143696d51", scope: null, file: !10, line: 32, type: !29, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !71)
// CHECK:STDOUT: !66 = !DILocalVariable(arg: 1, scope: !65, type: !20)
// CHECK:STDOUT: !70 = !DILocation(line: 32, column: 1, scope: !65)
// CHECK:STDOUT: !71 = !{!66}
// CHECK:STDOUT: !72 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !73 = !{!"omnipotent char", !72, i64 0}
// CHECK:STDOUT: !74 = !{!"any pointer", !73, i64 0}
// CHECK:STDOUT: !75 = !{!"p1 _ZTS1C", !74, i64 0}
// CHECK:STDOUT: !76 = !{!75, !75, i64 0}
// CHECK:STDOUT: !78 = !{!"int", !73, i64 0}
// CHECK:STDOUT: !79 = !{!"p1 int", !74, i64 0}
// CHECK:STDOUT: !80 = !{!79, !79, i64 0}
// CHECK:STDOUT: !81 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !82 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !83 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !84 = !{!"__libc_errno", !78, i64 0}
// CHECK:STDOUT: !85 = !{!84, !78, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
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
// CHECK:STDOUT: define void @_CPassRefs.Main() #0 !dbg !88 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !89
// CHECK:STDOUT:   %.loc20_18.2.temp = alloca {}, align 1, !dbg !94
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !98
// CHECK:STDOUT:   %_.var = alloca i32, align 4, !dbg !103
// CHECK:STDOUT:   %.loc26_23.2.temp = alloca i32, align 4, !dbg !105
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !89
// CHECK:STDOUT:   call void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.67b28f83c7cfd60c"(ptr %c.var), !dbg !89
// CHECK:STDOUT:   call void @_Z8TakeCRefR1C10ForceThunk.carbon_thunk.r(ptr %c.var), !dbg !93
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_18.2.temp), !dbg !94
// CHECK:STDOUT:   call void @_Z9TakeCRRefRK1C10ForceThunk.carbon_thunk._(ptr @C.val.loc20_20.3), !dbg !95
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.2a99575b423907d2(ptr @C.val), !dbg !94
// CHECK:STDOUT:   call void @_Z13TakeConstCRefRK1C10ForceThunk.carbon_thunk._(ptr %c.var), !dbg !97
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !98
// CHECK:STDOUT:   store i32 poison, ptr %n.var, align 4, !dbg !98
// CHECK:STDOUT:   call void @_Z10TakeIntRefRi10ForceThunk.carbon_thunk.r(ptr %n.var), !dbg !102
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !103
// CHECK:STDOUT:   store i32 42, ptr %_.var, align 4, !dbg !103
// CHECK:STDOUT:   call void @_Z11TakeIntRRefOi10ForceThunk.carbon_thunk.v(ptr %_.var), !dbg !104
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %_.var), !dbg !103
// CHECK:STDOUT:   %.loc26_23.1 = load i32, ptr %n.var, align 4, !dbg !105
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc26_23.2.temp), !dbg !105
// CHECK:STDOUT:   store i32 %.loc26_23.1, ptr %.loc26_23.2.temp, align 4, !dbg !105
// CHECK:STDOUT:   call void @_Z15TakeConstIntRefRKi10ForceThunk.carbon_thunk._(ptr %.loc26_23.2.temp), !dbg !106
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc26_23.2.temp), !dbg !105
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %n.var), !dbg !98
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.2a99575b423907d2(ptr %c.var), !dbg !89
// CHECK:STDOUT:   ret void, !dbg !107
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN1CC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !76
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !76
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Default.dc3dd9c349162a91.Core:C.Cpp"(ptr sret({}) %return) #2 !dbg !108 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN1CC1Ev.carbon_thunk.(ptr %return), !dbg !110
// CHECK:STDOUT:   ret void, !dbg !110
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z8TakeCRefR1C10ForceThunk.carbon_thunk.r(ptr noundef nonnull align 1 dereferenceable(1) %0) #3 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %class.ForceThunk, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !76
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !76, !nonnull !130
// CHECK:STDOUT:   call void @_Z8TakeCRefR1C10ForceThunk(ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z9TakeCRRefRK1C10ForceThunk.carbon_thunk._(ptr noundef %0) #3 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %class.ForceThunk, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !76
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !76
// CHECK:STDOUT:   call void @_Z9TakeCRRefRK1C10ForceThunk(ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z13TakeConstCRefRK1C10ForceThunk.carbon_thunk._(ptr noundef %0) #3 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %class.ForceThunk, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !76
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !76
// CHECK:STDOUT:   call void @_Z13TakeConstCRefRK1C10ForceThunk(ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z10TakeIntRefRi10ForceThunk.carbon_thunk.r(ptr noundef nonnull align 4 dereferenceable(4) %0) #3 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %class.ForceThunk, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !80
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !80, !nonnull !130, !align !132
// CHECK:STDOUT:   call void @_Z10TakeIntRefRi10ForceThunk(ptr noundef nonnull align 4 dereferenceable(4) %1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z11TakeIntRRefOi10ForceThunk.carbon_thunk.v(ptr noundef %0) #3 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %class.ForceThunk, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !80
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !80
// CHECK:STDOUT:   call void @_Z11TakeIntRRefOi10ForceThunk(ptr noundef nonnull align 4 dereferenceable(4) %1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z15TakeConstIntRefRKi10ForceThunk.carbon_thunk._(ptr noundef %0) #3 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %class.ForceThunk, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !80
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !80
// CHECK:STDOUT:   call void @_Z15TakeConstIntRefRKi10ForceThunk(ptr noundef nonnull align 4 dereferenceable(4) %1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.67b28f83c7cfd60c"(ptr sret({}) %return) #0 !dbg !111 {
// CHECK:STDOUT:   call void @"_COp:thunk:Default.dc3dd9c349162a91.Core:C.Cpp"(ptr %return), !dbg !114
// CHECK:STDOUT:   ret void, !dbg !115
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.2a99575b423907d2(ptr %t) #0 !dbg !116 {
// CHECK:STDOUT:   ret void, !dbg !121
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f127f30143696d51(ptr %t) #0 !dbg !123 {
// CHECK:STDOUT:   ret void, !dbg !128
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
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.2a99575b423907d2, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.f127f30143696d51, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #3 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #4 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #5 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!87}
// CHECK:STDOUT: !llvm.module.flags = !{!81, !82, !83, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!85}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "min_prelude/parts/default.carbon", directory: "")
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{null}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{null, !23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{!23}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !28 = !{null, !20}
// CHECK:STDOUT: !29 = !DISubroutineType(types: !28)
// CHECK:STDOUT: !72 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !73 = !{!"omnipotent char", !72, i64 0}
// CHECK:STDOUT: !74 = !{!"any pointer", !73, i64 0}
// CHECK:STDOUT: !75 = !{!"p1 _ZTS1C", !74, i64 0}
// CHECK:STDOUT: !76 = !{!75, !75, i64 0}
// CHECK:STDOUT: !78 = !{!"int", !73, i64 0}
// CHECK:STDOUT: !79 = !{!"p1 int", !74, i64 0}
// CHECK:STDOUT: !80 = !{!79, !79, i64 0}
// CHECK:STDOUT: !81 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !82 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !83 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !84 = !{!"__libc_errno", !78, i64 0}
// CHECK:STDOUT: !85 = !{!84, !78, i64 0}
// CHECK:STDOUT: !86 = !DIFile(filename: "pass_references_via_thunk.carbon", directory: "")
// CHECK:STDOUT: !87 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !86, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !88 = distinct !DISubprogram(name: "PassRefs", linkageName: "_CPassRefs.Main", scope: null, file: !86, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !87)
// CHECK:STDOUT: !89 = !DILocation(line: 18, column: 3, scope: !88)
// CHECK:STDOUT: !93 = !DILocation(line: 19, column: 3, scope: !88)
// CHECK:STDOUT: !94 = !DILocation(line: 20, column: 17, scope: !88)
// CHECK:STDOUT: !95 = !DILocation(line: 20, column: 3, scope: !88)
// CHECK:STDOUT: !97 = !DILocation(line: 21, column: 3, scope: !88)
// CHECK:STDOUT: !98 = !DILocation(line: 23, column: 3, scope: !88)
// CHECK:STDOUT: !102 = !DILocation(line: 24, column: 3, scope: !88)
// CHECK:STDOUT: !103 = !DILocation(line: 13, column: 23, scope: !88)
// CHECK:STDOUT: !104 = !DILocation(line: 25, column: 3, scope: !88)
// CHECK:STDOUT: !105 = !DILocation(line: 26, column: 23, scope: !88)
// CHECK:STDOUT: !106 = !DILocation(line: 26, column: 3, scope: !88)
// CHECK:STDOUT: !107 = !DILocation(line: 17, column: 1, scope: !88)
// CHECK:STDOUT: !108 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Default.dc3dd9c349162a91.Core:C.Cpp", scope: null, file: !86, line: 5, type: !27, spFlags: DISPFlagDefinition, unit: !87)
// CHECK:STDOUT: !110 = !DILocation(line: 5, column: 7, scope: !108)
// CHECK:STDOUT: !111 = distinct !DISubprogram(name: "Op", linkageName: "_COp.51f3fb62cb986374:DefaultOrUnformed.Core.67b28f83c7cfd60c", scope: null, file: !4, line: 9, type: !27, spFlags: DISPFlagDefinition, unit: !87)
// CHECK:STDOUT: !114 = !DILocation(line: 9, column: 28, scope: !111)
// CHECK:STDOUT: !115 = !DILocation(line: 9, column: 21, scope: !111)
// CHECK:STDOUT: !116 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.2a99575b423907d2", scope: null, file: !10, line: 32, type: !25, spFlags: DISPFlagDefinition, unit: !87, retainedNodes: !122)
// CHECK:STDOUT: !117 = !DILocalVariable(arg: 1, scope: !116, type: !23)
// CHECK:STDOUT: !121 = !DILocation(line: 32, column: 1, scope: !116)
// CHECK:STDOUT: !122 = !{!117}
// CHECK:STDOUT: !123 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f127f30143696d51", scope: null, file: !10, line: 32, type: !29, spFlags: DISPFlagDefinition, unit: !87, retainedNodes: !129)
// CHECK:STDOUT: !124 = !DILocalVariable(arg: 1, scope: !123, type: !20)
// CHECK:STDOUT: !128 = !DILocation(line: 32, column: 1, scope: !123)
// CHECK:STDOUT: !129 = !{!124}
// CHECK:STDOUT: !130 = !{}
// CHECK:STDOUT: !132 = !{i64 4}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'return_references.carbon'
// CHECK:STDOUT: source_filename = "return_references.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CGetRefs.Main() #0 !dbg !135 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnCRef.call = call ptr @_Z10ReturnCRefv(), !dbg !136
// CHECK:STDOUT:   %ReturnCRRef.call = call ptr @_Z11ReturnCRRefv(), !dbg !139
// CHECK:STDOUT:   %ReturnConstCRef.call = call ptr @_Z15ReturnConstCRefv(), !dbg !142
// CHECK:STDOUT:   %ReturnIntRef.call = call ptr @_Z12ReturnIntRefv(), !dbg !145
// CHECK:STDOUT:   %ReturnIntRRef.call = call ptr @_Z13ReturnIntRRefv(), !dbg !148
// CHECK:STDOUT:   %ReturnConstIntRef.call = call ptr @_Z17ReturnConstIntRefv(), !dbg !151
// CHECK:STDOUT:   ret void, !dbg !154
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!134}
// CHECK:STDOUT: !llvm.module.flags = !{!81, !82, !83, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!85}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !21 = !{null}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !72 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !73 = !{!"omnipotent char", !72, i64 0}
// CHECK:STDOUT: !78 = !{!"int", !73, i64 0}
// CHECK:STDOUT: !81 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !82 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !83 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !84 = !{!"__libc_errno", !78, i64 0}
// CHECK:STDOUT: !85 = !{!84, !78, i64 0}
// CHECK:STDOUT: !133 = !DIFile(filename: "return_references.carbon", directory: "")
// CHECK:STDOUT: !134 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !133, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !135 = distinct !DISubprogram(name: "GetRefs", linkageName: "_CGetRefs.Main", scope: null, file: !133, line: 16, type: !22, spFlags: DISPFlagDefinition, unit: !134)
// CHECK:STDOUT: !136 = !DILocation(line: 17, column: 22, scope: !135)
// CHECK:STDOUT: !139 = !DILocation(line: 18, column: 22, scope: !135)
// CHECK:STDOUT: !142 = !DILocation(line: 19, column: 28, scope: !135)
// CHECK:STDOUT: !145 = !DILocation(line: 21, column: 20, scope: !135)
// CHECK:STDOUT: !148 = !DILocation(line: 22, column: 20, scope: !135)
// CHECK:STDOUT: !151 = !DILocation(line: 23, column: 26, scope: !135)
// CHECK:STDOUT: !154 = !DILocation(line: 16, column: 1, scope: !135)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'return_references_via_thunk.carbon'
// CHECK:STDOUT: source_filename = "return_references_via_thunk.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %class.ForceThunk.23 = type { i8 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CGetRefs.Main() #0 !dbg !157 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnCRef__carbon_thunk.call = call ptr @_Z10ReturnCRef10ForceThunk.carbon_thunk.(), !dbg !158
// CHECK:STDOUT:   %ReturnCRRef__carbon_thunk.call = call ptr @_Z11ReturnCRRef10ForceThunk.carbon_thunk.(), !dbg !161
// CHECK:STDOUT:   %ReturnConstCRef__carbon_thunk.call = call ptr @_Z15ReturnConstCRef10ForceThunk.carbon_thunk.(), !dbg !164
// CHECK:STDOUT:   %ReturnIntRef__carbon_thunk.call = call ptr @_Z12ReturnIntRef10ForceThunk.carbon_thunk.(), !dbg !167
// CHECK:STDOUT:   %ReturnIntRRef__carbon_thunk.call = call ptr @_Z13ReturnIntRRef10ForceThunk.carbon_thunk.(), !dbg !170
// CHECK:STDOUT:   %ReturnConstIntRef__carbon_thunk.call = call ptr @_Z17ReturnConstIntRef10ForceThunk.carbon_thunk.(), !dbg !173
// CHECK:STDOUT:   ret void, !dbg !176
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal noundef nonnull align 1 dereferenceable(1) ptr @_Z10ReturnCRef10ForceThunk.carbon_thunk.() #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %agg.tmp = alloca %class.ForceThunk.23, align 1
// CHECK:STDOUT:   %call = call noundef nonnull align 1 dereferenceable(1) ptr @_Z10ReturnCRef10ForceThunk()
// CHECK:STDOUT:   ret ptr %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal noundef nonnull align 1 dereferenceable(1) ptr @_Z11ReturnCRRef10ForceThunk.carbon_thunk.() #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %agg.tmp = alloca %class.ForceThunk.23, align 1
// CHECK:STDOUT:   %call = call noundef nonnull align 1 dereferenceable(1) ptr @_Z11ReturnCRRef10ForceThunk()
// CHECK:STDOUT:   ret ptr %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal noundef nonnull align 1 dereferenceable(1) ptr @_Z15ReturnConstCRef10ForceThunk.carbon_thunk.() #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %agg.tmp = alloca %class.ForceThunk.23, align 1
// CHECK:STDOUT:   %call = call noundef nonnull align 1 dereferenceable(1) ptr @_Z15ReturnConstCRef10ForceThunk()
// CHECK:STDOUT:   ret ptr %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal noundef nonnull align 4 dereferenceable(4) ptr @_Z12ReturnIntRef10ForceThunk.carbon_thunk.() #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %agg.tmp = alloca %class.ForceThunk.23, align 1
// CHECK:STDOUT:   %call = call noundef nonnull align 4 dereferenceable(4) ptr @_Z12ReturnIntRef10ForceThunk()
// CHECK:STDOUT:   ret ptr %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal noundef nonnull align 4 dereferenceable(4) ptr @_Z13ReturnIntRRef10ForceThunk.carbon_thunk.() #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %agg.tmp = alloca %class.ForceThunk.23, align 1
// CHECK:STDOUT:   %call = call noundef nonnull align 4 dereferenceable(4) ptr @_Z13ReturnIntRRef10ForceThunk()
// CHECK:STDOUT:   ret ptr %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal noundef nonnull align 4 dereferenceable(4) ptr @_Z17ReturnConstIntRef10ForceThunk.carbon_thunk.() #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %agg.tmp = alloca %class.ForceThunk.23, align 1
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!156}
// CHECK:STDOUT: !llvm.module.flags = !{!81, !82, !83, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!85}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !21 = !{null}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !72 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !73 = !{!"omnipotent char", !72, i64 0}
// CHECK:STDOUT: !78 = !{!"int", !73, i64 0}
// CHECK:STDOUT: !81 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !82 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !83 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !84 = !{!"__libc_errno", !78, i64 0}
// CHECK:STDOUT: !85 = !{!84, !78, i64 0}
// CHECK:STDOUT: !155 = !DIFile(filename: "return_references_via_thunk.carbon", directory: "")
// CHECK:STDOUT: !156 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !155, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !157 = distinct !DISubprogram(name: "GetRefs", linkageName: "_CGetRefs.Main", scope: null, file: !155, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !156)
// CHECK:STDOUT: !158 = !DILocation(line: 18, column: 22, scope: !157)
// CHECK:STDOUT: !161 = !DILocation(line: 19, column: 22, scope: !157)
// CHECK:STDOUT: !164 = !DILocation(line: 20, column: 28, scope: !157)
// CHECK:STDOUT: !167 = !DILocation(line: 22, column: 20, scope: !157)
// CHECK:STDOUT: !170 = !DILocation(line: 23, column: 20, scope: !157)
// CHECK:STDOUT: !173 = !DILocation(line: 24, column: 26, scope: !157)
// CHECK:STDOUT: !176 = !DILocation(line: 17, column: 1, scope: !157)
// CHECK:STDOUT:
