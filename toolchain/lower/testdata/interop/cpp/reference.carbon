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
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.e4c092cbec31eca4(ptr %_.var.1), !dbg !36
// CHECK:STDOUT:   call void @_Z13TakeConstCRefRK1C.carbon_thunk._(ptr %c.var), !dbg !39
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !40
// CHECK:STDOUT:   store i32 poison, ptr %n.var, align 4, !dbg !40
// CHECK:STDOUT:   call void @_Z10TakeIntRefRi(ptr %n.var), !dbg !44
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.2), !dbg !45
// CHECK:STDOUT:   store i32 42, ptr %_.var.2, align 4, !dbg !45
// CHECK:STDOUT:   call void @_Z11TakeIntRRefOi.carbon_thunk.v(ptr %_.var.2), !dbg !46
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %_.var.2), !dbg !45
// CHECK:STDOUT:   %.loc25_23.1 = load i32, ptr %n.var, align 4, !dbg !47
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc25_23.2.temp), !dbg !47
// CHECK:STDOUT:   store i32 %.loc25_23.1, ptr %.loc25_23.2.temp, align 4, !dbg !47
// CHECK:STDOUT:   call void @_Z15TakeConstIntRefRKi.carbon_thunk._(ptr %.loc25_23.2.temp), !dbg !48
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %.loc25_23.2.temp), !dbg !47
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %n.var), !dbg !40
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.e4c092cbec31eca4(ptr %c.var), !dbg !31
// CHECK:STDOUT:   ret void, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN1CC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !80
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !80
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
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !80
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !80
// CHECK:STDOUT:   call void @_Z9TakeCRRefO1C(ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z13TakeConstCRefRK1C.carbon_thunk._(ptr noundef %0) #4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !80
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !80
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
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !84
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !84
// CHECK:STDOUT:   call void @_Z11TakeIntRRefOi(ptr noundef nonnull align 4 dereferenceable(4) %1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z15TakeConstIntRefRKi.carbon_thunk._(ptr noundef %0) #4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !84
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !84
// CHECK:STDOUT:   call void @_Z15TakeConstIntRefRKi(ptr noundef nonnull align 4 dereferenceable(4) %1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #5
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.67b28f83c7cfd60c"(ptr sret({}) %return) #0 !dbg !57 {
// CHECK:STDOUT:   call void @"_COp:thunk:Default.dc3dd9c349162a91.Core:C.Cpp"(ptr %return), !dbg !60
// CHECK:STDOUT:   ret void, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #5
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.e4c092cbec31eca4(ptr %self) #0 !dbg !62 {
// CHECK:STDOUT:   ret void, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %self) #0 !dbg !69 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self), !dbg !73
// CHECK:STDOUT:   ret void, !dbg !74
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
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.e4c092cbec31eca4, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #3 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #4 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #5 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!85, !86, !87, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!89}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !12 = !DIFile(filename: "min_prelude/parts/default.carbon", directory: "")
// CHECK:STDOUT: !14 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
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
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !18, line: 12, type: !29, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !56)
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !53, type: !20)
// CHECK:STDOUT: !55 = !DILocation(line: 12, column: 23, scope: !53)
// CHECK:STDOUT: !56 = !{!54}
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "Op", linkageName: "_COp.51f3fb62cb986374:DefaultOrUnformed.Core.67b28f83c7cfd60c", scope: null, file: !12, line: 9, type: !27, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !60 = !DILocation(line: 9, column: 28, scope: !57)
// CHECK:STDOUT: !61 = !DILocation(line: 9, column: 21, scope: !57)
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.e4c092cbec31eca4", scope: null, file: !14, line: 24, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !68)
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !62, type: !23)
// CHECK:STDOUT: !67 = !DILocation(line: 24, column: 3, scope: !62)
// CHECK:STDOUT: !68 = !{!63}
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48", scope: null, file: !14, line: 24, type: !29, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !75)
// CHECK:STDOUT: !70 = !DILocalVariable(arg: 1, scope: !69, type: !20)
// CHECK:STDOUT: !73 = !DILocation(line: 26, column: 5, scope: !69)
// CHECK:STDOUT: !74 = !DILocation(line: 24, column: 3, scope: !69)
// CHECK:STDOUT: !75 = !{!70}
// CHECK:STDOUT: !76 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !77 = !{!"omnipotent char", !76, i64 0}
// CHECK:STDOUT: !78 = !{!"any pointer", !77, i64 0}
// CHECK:STDOUT: !79 = !{!"p1 _ZTS1C", !78, i64 0}
// CHECK:STDOUT: !80 = !{!79, !79, i64 0}
// CHECK:STDOUT: !82 = !{!"int", !77, i64 0}
// CHECK:STDOUT: !83 = !{!"p1 int", !78, i64 0}
// CHECK:STDOUT: !84 = !{!83, !83, i64 0}
// CHECK:STDOUT: !85 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !86 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !87 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !88 = !{!"__libc_errno", !82, i64 0}
// CHECK:STDOUT: !89 = !{!88, !82, i64 0}
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
// CHECK:STDOUT: define void @_CPassRefs.Main() #0 !dbg !92 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %c.var = alloca {}, align 1, !dbg !93
// CHECK:STDOUT:   %.loc20_18.2.temp = alloca {}, align 1, !dbg !98
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !102
// CHECK:STDOUT:   %_.var = alloca i32, align 4, !dbg !107
// CHECK:STDOUT:   %.loc26_23.2.temp = alloca i32, align 4, !dbg !109
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %c.var), !dbg !93
// CHECK:STDOUT:   call void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.67b28f83c7cfd60c"(ptr %c.var), !dbg !93
// CHECK:STDOUT:   call void @_Z8TakeCRefR1C10ForceThunk.carbon_thunk.r(ptr %c.var), !dbg !97
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc20_18.2.temp), !dbg !98
// CHECK:STDOUT:   call void @_Z9TakeCRRefRK1C10ForceThunk.carbon_thunk._(ptr @C.val.loc20_20.3), !dbg !99
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.e4c092cbec31eca4(ptr @C.val), !dbg !98
// CHECK:STDOUT:   call void @_Z13TakeConstCRefRK1C10ForceThunk.carbon_thunk._(ptr %c.var), !dbg !101
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !102
// CHECK:STDOUT:   store i32 poison, ptr %n.var, align 4, !dbg !102
// CHECK:STDOUT:   call void @_Z10TakeIntRefRi10ForceThunk.carbon_thunk.r(ptr %n.var), !dbg !106
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !107
// CHECK:STDOUT:   store i32 42, ptr %_.var, align 4, !dbg !107
// CHECK:STDOUT:   call void @_Z11TakeIntRRefOi10ForceThunk.carbon_thunk.v(ptr %_.var), !dbg !108
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %_.var), !dbg !107
// CHECK:STDOUT:   %.loc26_23.1 = load i32, ptr %n.var, align 4, !dbg !109
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc26_23.2.temp), !dbg !109
// CHECK:STDOUT:   store i32 %.loc26_23.1, ptr %.loc26_23.2.temp, align 4, !dbg !109
// CHECK:STDOUT:   call void @_Z15TakeConstIntRefRKi10ForceThunk.carbon_thunk._(ptr %.loc26_23.2.temp), !dbg !110
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %.loc26_23.2.temp), !dbg !109
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %n.var), !dbg !102
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.e4c092cbec31eca4(ptr %c.var), !dbg !93
// CHECK:STDOUT:   ret void, !dbg !111
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN1CC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !80
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !80
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Default.dc3dd9c349162a91.Core:C.Cpp"(ptr sret({}) %return) #2 !dbg !112 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN1CC1Ev.carbon_thunk.(ptr %return), !dbg !114
// CHECK:STDOUT:   ret void, !dbg !114
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z8TakeCRefR1C10ForceThunk.carbon_thunk.r(ptr noundef nonnull align 1 dereferenceable(1) %0) #3 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %class.ForceThunk, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !80
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !80, !nonnull !138
// CHECK:STDOUT:   call void @_Z8TakeCRefR1C10ForceThunk(ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z9TakeCRRefRK1C10ForceThunk.carbon_thunk._(ptr noundef %0) #3 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %class.ForceThunk, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !80
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !80
// CHECK:STDOUT:   call void @_Z9TakeCRRefRK1C10ForceThunk(ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z13TakeConstCRefRK1C10ForceThunk.carbon_thunk._(ptr noundef %0) #3 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %class.ForceThunk, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !80
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !80
// CHECK:STDOUT:   call void @_Z13TakeConstCRefRK1C10ForceThunk(ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z10TakeIntRefRi10ForceThunk.carbon_thunk.r(ptr noundef nonnull align 4 dereferenceable(4) %0) #3 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %class.ForceThunk, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !84
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !84, !nonnull !138, !align !140
// CHECK:STDOUT:   call void @_Z10TakeIntRefRi10ForceThunk(ptr noundef nonnull align 4 dereferenceable(4) %1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z11TakeIntRRefOi10ForceThunk.carbon_thunk.v(ptr noundef %0) #3 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %class.ForceThunk, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !84
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !84
// CHECK:STDOUT:   call void @_Z11TakeIntRRefOi10ForceThunk(ptr noundef nonnull align 4 dereferenceable(4) %1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !115 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !117
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Z15TakeConstIntRefRKi10ForceThunk.carbon_thunk._(ptr noundef %0) #3 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %class.ForceThunk, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !84
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !84
// CHECK:STDOUT:   call void @_Z15TakeConstIntRefRKi10ForceThunk(ptr noundef nonnull align 4 dereferenceable(4) %1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.67b28f83c7cfd60c"(ptr sret({}) %return) #0 !dbg !119 {
// CHECK:STDOUT:   call void @"_COp:thunk:Default.dc3dd9c349162a91.Core:C.Cpp"(ptr %return), !dbg !122
// CHECK:STDOUT:   ret void, !dbg !123
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.e4c092cbec31eca4(ptr %self) #0 !dbg !124 {
// CHECK:STDOUT:   ret void, !dbg !129
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48(ptr %self) #0 !dbg !131 {
// CHECK:STDOUT:   call void @"_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core"(ptr %self), !dbg !135
// CHECK:STDOUT:   ret void, !dbg !136
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
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.e4c092cbec31eca4, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #3 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #4 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #5 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!91}
// CHECK:STDOUT: !llvm.module.flags = !{!85, !86, !87, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!89}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !12 = !DIFile(filename: "min_prelude/parts/default.carbon", directory: "")
// CHECK:STDOUT: !14 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
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
// CHECK:STDOUT: !76 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !77 = !{!"omnipotent char", !76, i64 0}
// CHECK:STDOUT: !78 = !{!"any pointer", !77, i64 0}
// CHECK:STDOUT: !79 = !{!"p1 _ZTS1C", !78, i64 0}
// CHECK:STDOUT: !80 = !{!79, !79, i64 0}
// CHECK:STDOUT: !82 = !{!"int", !77, i64 0}
// CHECK:STDOUT: !83 = !{!"p1 int", !78, i64 0}
// CHECK:STDOUT: !84 = !{!83, !83, i64 0}
// CHECK:STDOUT: !85 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !86 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !87 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !88 = !{!"__libc_errno", !82, i64 0}
// CHECK:STDOUT: !89 = !{!88, !82, i64 0}
// CHECK:STDOUT: !90 = !DIFile(filename: "pass_references_via_thunk.carbon", directory: "")
// CHECK:STDOUT: !91 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !90, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !92 = distinct !DISubprogram(name: "PassRefs", linkageName: "_CPassRefs.Main", scope: null, file: !90, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !91)
// CHECK:STDOUT: !93 = !DILocation(line: 18, column: 3, scope: !92)
// CHECK:STDOUT: !97 = !DILocation(line: 19, column: 3, scope: !92)
// CHECK:STDOUT: !98 = !DILocation(line: 20, column: 17, scope: !92)
// CHECK:STDOUT: !99 = !DILocation(line: 20, column: 3, scope: !92)
// CHECK:STDOUT: !101 = !DILocation(line: 21, column: 3, scope: !92)
// CHECK:STDOUT: !102 = !DILocation(line: 23, column: 3, scope: !92)
// CHECK:STDOUT: !106 = !DILocation(line: 24, column: 3, scope: !92)
// CHECK:STDOUT: !107 = !DILocation(line: 13, column: 23, scope: !92)
// CHECK:STDOUT: !108 = !DILocation(line: 25, column: 3, scope: !92)
// CHECK:STDOUT: !109 = !DILocation(line: 26, column: 23, scope: !92)
// CHECK:STDOUT: !110 = !DILocation(line: 26, column: 3, scope: !92)
// CHECK:STDOUT: !111 = !DILocation(line: 17, column: 1, scope: !92)
// CHECK:STDOUT: !112 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Default.dc3dd9c349162a91.Core:C.Cpp", scope: null, file: !90, line: 5, type: !27, spFlags: DISPFlagDefinition, unit: !91)
// CHECK:STDOUT: !114 = !DILocation(line: 5, column: 7, scope: !112)
// CHECK:STDOUT: !115 = distinct !DISubprogram(name: "SubobjectDestroy", linkageName: "_CSubobjectDestroy.f58524831c37807e:core.Destroy.Core", scope: null, file: !90, line: 13, type: !29, spFlags: DISPFlagDefinition, unit: !91, retainedNodes: !118)
// CHECK:STDOUT: !116 = !DILocalVariable(arg: 1, scope: !115, type: !20)
// CHECK:STDOUT: !117 = !DILocation(line: 13, column: 23, scope: !115)
// CHECK:STDOUT: !118 = !{!116}
// CHECK:STDOUT: !119 = distinct !DISubprogram(name: "Op", linkageName: "_COp.51f3fb62cb986374:DefaultOrUnformed.Core.67b28f83c7cfd60c", scope: null, file: !12, line: 9, type: !27, spFlags: DISPFlagDefinition, unit: !91)
// CHECK:STDOUT: !122 = !DILocation(line: 9, column: 28, scope: !119)
// CHECK:STDOUT: !123 = !DILocation(line: 9, column: 21, scope: !119)
// CHECK:STDOUT: !124 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.e4c092cbec31eca4", scope: null, file: !14, line: 24, type: !25, spFlags: DISPFlagDefinition, unit: !91, retainedNodes: !130)
// CHECK:STDOUT: !125 = !DILocalVariable(arg: 1, scope: !124, type: !23)
// CHECK:STDOUT: !129 = !DILocation(line: 24, column: 3, scope: !124)
// CHECK:STDOUT: !130 = !{!125}
// CHECK:STDOUT: !131 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.6580e8d2c4d79a48", scope: null, file: !14, line: 24, type: !29, spFlags: DISPFlagDefinition, unit: !91, retainedNodes: !137)
// CHECK:STDOUT: !132 = !DILocalVariable(arg: 1, scope: !131, type: !20)
// CHECK:STDOUT: !135 = !DILocation(line: 26, column: 5, scope: !131)
// CHECK:STDOUT: !136 = !DILocation(line: 24, column: 3, scope: !131)
// CHECK:STDOUT: !137 = !{!132}
// CHECK:STDOUT: !138 = !{}
// CHECK:STDOUT: !140 = !{i64 4}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'return_references.carbon'
// CHECK:STDOUT: source_filename = "return_references.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CGetRefs.Main() #0 !dbg !143 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnCRef.call = call ptr @_Z10ReturnCRefv(), !dbg !144
// CHECK:STDOUT:   %ReturnCRRef.call = call ptr @_Z11ReturnCRRefv(), !dbg !147
// CHECK:STDOUT:   %ReturnConstCRef.call = call ptr @_Z15ReturnConstCRefv(), !dbg !150
// CHECK:STDOUT:   %ReturnIntRef.call = call ptr @_Z12ReturnIntRefv(), !dbg !153
// CHECK:STDOUT:   %ReturnIntRRef.call = call ptr @_Z13ReturnIntRRefv(), !dbg !156
// CHECK:STDOUT:   %ReturnConstIntRef.call = call ptr @_Z17ReturnConstIntRefv(), !dbg !159
// CHECK:STDOUT:   ret void, !dbg !162
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!142}
// CHECK:STDOUT: !llvm.module.flags = !{!85, !86, !87, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!89}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !21 = !{null}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !76 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !77 = !{!"omnipotent char", !76, i64 0}
// CHECK:STDOUT: !82 = !{!"int", !77, i64 0}
// CHECK:STDOUT: !85 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !86 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !87 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !88 = !{!"__libc_errno", !82, i64 0}
// CHECK:STDOUT: !89 = !{!88, !82, i64 0}
// CHECK:STDOUT: !141 = !DIFile(filename: "return_references.carbon", directory: "")
// CHECK:STDOUT: !142 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !141, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !143 = distinct !DISubprogram(name: "GetRefs", linkageName: "_CGetRefs.Main", scope: null, file: !141, line: 16, type: !22, spFlags: DISPFlagDefinition, unit: !142)
// CHECK:STDOUT: !144 = !DILocation(line: 17, column: 22, scope: !143)
// CHECK:STDOUT: !147 = !DILocation(line: 18, column: 22, scope: !143)
// CHECK:STDOUT: !150 = !DILocation(line: 19, column: 28, scope: !143)
// CHECK:STDOUT: !153 = !DILocation(line: 21, column: 20, scope: !143)
// CHECK:STDOUT: !156 = !DILocation(line: 22, column: 20, scope: !143)
// CHECK:STDOUT: !159 = !DILocation(line: 23, column: 26, scope: !143)
// CHECK:STDOUT: !162 = !DILocation(line: 16, column: 1, scope: !143)
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
// CHECK:STDOUT: define void @_CGetRefs.Main() #0 !dbg !165 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %ReturnCRef__carbon_thunk.call = call ptr @_Z10ReturnCRef10ForceThunk.carbon_thunk.(), !dbg !166
// CHECK:STDOUT:   %ReturnCRRef__carbon_thunk.call = call ptr @_Z11ReturnCRRef10ForceThunk.carbon_thunk.(), !dbg !169
// CHECK:STDOUT:   %ReturnConstCRef__carbon_thunk.call = call ptr @_Z15ReturnConstCRef10ForceThunk.carbon_thunk.(), !dbg !172
// CHECK:STDOUT:   %ReturnIntRef__carbon_thunk.call = call ptr @_Z12ReturnIntRef10ForceThunk.carbon_thunk.(), !dbg !175
// CHECK:STDOUT:   %ReturnIntRRef__carbon_thunk.call = call ptr @_Z13ReturnIntRRef10ForceThunk.carbon_thunk.(), !dbg !178
// CHECK:STDOUT:   %ReturnConstIntRef__carbon_thunk.call = call ptr @_Z17ReturnConstIntRef10ForceThunk.carbon_thunk.(), !dbg !181
// CHECK:STDOUT:   ret void, !dbg !184
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!164}
// CHECK:STDOUT: !llvm.module.flags = !{!85, !86, !87, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!89}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !21 = !{null}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !76 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !77 = !{!"omnipotent char", !76, i64 0}
// CHECK:STDOUT: !82 = !{!"int", !77, i64 0}
// CHECK:STDOUT: !85 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !86 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !87 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !88 = !{!"__libc_errno", !82, i64 0}
// CHECK:STDOUT: !89 = !{!88, !82, i64 0}
// CHECK:STDOUT: !163 = !DIFile(filename: "return_references_via_thunk.carbon", directory: "")
// CHECK:STDOUT: !164 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !163, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !165 = distinct !DISubprogram(name: "GetRefs", linkageName: "_CGetRefs.Main", scope: null, file: !163, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !164)
// CHECK:STDOUT: !166 = !DILocation(line: 18, column: 22, scope: !165)
// CHECK:STDOUT: !169 = !DILocation(line: 19, column: 22, scope: !165)
// CHECK:STDOUT: !172 = !DILocation(line: 20, column: 28, scope: !165)
// CHECK:STDOUT: !175 = !DILocation(line: 22, column: 20, scope: !165)
// CHECK:STDOUT: !178 = !DILocation(line: 23, column: 20, scope: !165)
// CHECK:STDOUT: !181 = !DILocation(line: 24, column: 26, scope: !165)
// CHECK:STDOUT: !184 = !DILocation(line: 17, column: 1, scope: !165)
// CHECK:STDOUT:
