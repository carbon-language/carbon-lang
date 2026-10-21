// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
// EXTRA-ARGS: --clang-arg=-fno-exceptions
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/class/import/copy_vs_move.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/class/import/copy_vs_move.carbon

// --- copyable.carbon
library "[[@TEST_NAME]]";

import Cpp;
inline Cpp '''
struct Copyable {
  Copyable(const Copyable&);
  Copyable& operator=(const Copyable&);
  ~Copyable();
};
Copyable make_copyable();
void pass_copyable(Copyable);
''';

var c: Cpp.Copyable = Cpp.make_copyable();

fn PassGlobal() {
  Cpp.pass_copyable(c);
}

fn PassTemporary() {
  Cpp.pass_copyable(Cpp.make_copyable());
}

fn PassValue(c: Cpp.Copyable) {
  Cpp.pass_copyable(c);
}

fn PassRef(ref c: Cpp.Copyable) {
  Cpp.pass_copyable(c);
}

// --- copy_or_move.carbon
library "[[@TEST_NAME]]";

import Cpp;
inline Cpp '''
struct CopyOrMove {
  CopyOrMove(const CopyOrMove&);
  CopyOrMove& operator=(const CopyOrMove&);
  CopyOrMove(CopyOrMove&&);
  CopyOrMove& operator=(CopyOrMove&&);
  ~CopyOrMove();
};
CopyOrMove make_copy_or_move();
void pass_copy_or_move(CopyOrMove);
''';

var c: Cpp.CopyOrMove = Cpp.make_copy_or_move();

fn PassGlobal() {
  Cpp.pass_copy_or_move(c);
}

fn PassTemporary() {
  Cpp.pass_copy_or_move(Cpp.make_copy_or_move());
}

fn PassValue(c: Cpp.CopyOrMove) {
  Cpp.pass_copy_or_move(c);
}

fn PassRef(ref c: Cpp.CopyOrMove) {
  Cpp.pass_copy_or_move(c);
}

// --- move_only.carbon
library "[[@TEST_NAME]]";

import Cpp;
inline Cpp '''
namespace std {
template<typename T> T&& move(T& t) { return static_cast<T&&>(t); }
}

struct MoveOnly {
  MoveOnly(const MoveOnly&) = delete;
  MoveOnly& operator=(const MoveOnly&) = delete;
  MoveOnly(MoveOnly&&);
  MoveOnly& operator=(MoveOnly&&);
  ~MoveOnly();
};
MoveOnly make_move_only();
void pass_move_only(MoveOnly);
''';

var c: Cpp.MoveOnly = Cpp.make_move_only();

fn PassGlobalByMove() {
  // TODO: This should be accepted.
  // Cpp.pass_move_only(Cpp.std.move(ref c));
}

fn PassTemporary() {
  Cpp.pass_move_only(Cpp.make_move_only());
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'copyable.carbon'
// CHECK:STDOUT: source_filename = "copyable.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %struct.Copyable = type { i8 }
// CHECK:STDOUT:
// CHECK:STDOUT: @_Cc.Main = global [1 x i8] zeroinitializer
// CHECK:STDOUT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @_C__global_init.Main, ptr null }]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_Z13make_copyablev.carbon_thunk.(ptr noundef %return) #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !47
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !47
// CHECK:STDOUT:   call void @_Z13make_copyablev(ptr dead_on_unwind writable sret(%struct.Copyable) align 1 %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassGlobal.Main() #1 !dbg !13 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z13pass_copyable8Copyable.carbon_thunk._(ptr @_Cc.Main), !dbg !15
// CHECK:STDOUT:   ret void, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_Z13pass_copyable8Copyable.carbon_thunk._(ptr noundef %0) #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %struct.Copyable, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !47
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !47
// CHECK:STDOUT:   call void @_ZN8CopyableC1ERKS_(ptr noundef nonnull align 1 dereferenceable(1) %agg.tmp, ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   call void @_Z13pass_copyable8Copyable(ptr nofreeobj noundef align 1 dereferenceable(1) %agg.tmp)
// CHECK:STDOUT:   call void @_ZN8CopyableD1Ev(ptr noundef nonnull align 1 dereferenceable(1) %agg.tmp) #1
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassTemporary.Main() #1 !dbg !17 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc21_39.1.temp = alloca [1 x i8], align 1, !dbg !18
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc21_39.1.temp), !dbg !18
// CHECK:STDOUT:   call void @_Z13make_copyablev.carbon_thunk.(ptr %.loc21_39.1.temp), !dbg !18
// CHECK:STDOUT:   call void @_Z13pass_copyable8Copyable.carbon_thunk._(ptr %.loc21_39.1.temp), !dbg !19
// CHECK:STDOUT:   call void @"_CSelfDestruct.fd98d268aadcda21:core.Destroy.Core"(ptr %.loc21_39.1.temp), !dbg !18
// CHECK:STDOUT:   ret void, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: declare void @_ZN8CopyableD1Ev(ptr noundef nonnull align 1 dereferenceable(1)) unnamed_addr #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.fd98d268aadcda21:core.Destroy.Core"(ptr %self) #1 !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN8CopyableD1Ev(ptr %self), !dbg !23
// CHECK:STDOUT:   ret void, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassValue.Main(ptr %c) #1 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z13pass_copyable8Copyable.carbon_thunk._(ptr %c), !dbg !29
// CHECK:STDOUT:   ret void, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassRef.Main(ptr %c) #1 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z13pass_copyable8Copyable.carbon_thunk._(ptr %c), !dbg !36
// CHECK:STDOUT:   ret void, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Main() #1 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z13make_copyablev.carbon_thunk.(ptr @_Cc.Main), !dbg !40
// CHECK:STDOUT:   ret void, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z13make_copyablev(ptr dead_on_unwind writable sret(%struct.Copyable) align 1) #4
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z13pass_copyable8Copyable(ptr nofreeobj noundef align 1 dereferenceable(1)) #4
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN8CopyableC1ERKS_(ptr noundef nonnull align 1 dereferenceable(1), ptr noundef nonnull align 1 dereferenceable(1)) unnamed_addr #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_ZN8CopyableD1Ev, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { nounwind }
// CHECK:STDOUT: attributes #2 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #4 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!7}
// CHECK:STDOUT: !llvm.module.flags = !{!49, !50, !51, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!54}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIFile(filename: "copyable.carbon", directory: "")
// CHECK:STDOUT: !7 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !9 = !{null, !8}
// CHECK:STDOUT: !10 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !11 = !{null}
// CHECK:STDOUT: !12 = !DISubroutineType(types: !11)
// CHECK:STDOUT: !13 = distinct !DISubprogram(name: "PassGlobal", linkageName: "_CPassGlobal.Main", scope: null, file: !6, line: 16, type: !12, spFlags: DISPFlagDefinition, unit: !7)
// CHECK:STDOUT: !15 = !DILocation(line: 17, column: 3, scope: !13)
// CHECK:STDOUT: !16 = !DILocation(line: 16, column: 1, scope: !13)
// CHECK:STDOUT: !17 = distinct !DISubprogram(name: "PassTemporary", linkageName: "_CPassTemporary.Main", scope: null, file: !6, line: 20, type: !12, spFlags: DISPFlagDefinition, unit: !7)
// CHECK:STDOUT: !18 = !DILocation(line: 21, column: 21, scope: !17)
// CHECK:STDOUT: !19 = !DILocation(line: 21, column: 3, scope: !17)
// CHECK:STDOUT: !20 = !DILocation(line: 20, column: 1, scope: !17)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.fd98d268aadcda21:core.Destroy.Core", scope: null, file: !6, line: 21, type: !10, spFlags: DISPFlagDefinition, unit: !7, retainedNodes: !24)
// CHECK:STDOUT: !22 = !DILocalVariable(arg: 1, scope: !21, type: !8)
// CHECK:STDOUT: !23 = !DILocation(line: 21, column: 21, scope: !21)
// CHECK:STDOUT: !24 = !{!22}
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "PassValue", linkageName: "_CPassValue.Main", scope: null, file: !6, line: 24, type: !10, spFlags: DISPFlagDefinition, unit: !7, retainedNodes: !31)
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 1, scope: !25, type: !8)
// CHECK:STDOUT: !29 = !DILocation(line: 25, column: 3, scope: !25)
// CHECK:STDOUT: !30 = !DILocation(line: 24, column: 1, scope: !25)
// CHECK:STDOUT: !31 = !{!26}
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "PassRef", linkageName: "_CPassRef.Main", scope: null, file: !6, line: 28, type: !10, spFlags: DISPFlagDefinition, unit: !7, retainedNodes: !38)
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 1, scope: !32, type: !8)
// CHECK:STDOUT: !36 = !DILocation(line: 29, column: 3, scope: !32)
// CHECK:STDOUT: !37 = !DILocation(line: 28, column: 1, scope: !32)
// CHECK:STDOUT: !38 = !{!33}
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !6, type: !12, spFlags: DISPFlagDefinition, unit: !7)
// CHECK:STDOUT: !40 = !DILocation(line: 14, column: 23, scope: !39)
// CHECK:STDOUT: !42 = !DILocation(line: 0, scope: !39)
// CHECK:STDOUT: !43 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !44 = !{!"omnipotent char", !43, i64 0}
// CHECK:STDOUT: !45 = !{!"any pointer", !44, i64 0}
// CHECK:STDOUT: !46 = !{!"p1 _ZTS8Copyable", !45, i64 0}
// CHECK:STDOUT: !47 = !{!46, !46, i64 0}
// CHECK:STDOUT: !49 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !50 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !51 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !52 = !{!"int", !44, i64 0}
// CHECK:STDOUT: !53 = !{!"__libc_errno", !52, i64 0}
// CHECK:STDOUT: !54 = !{!53, !52, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'copy_or_move.carbon'
// CHECK:STDOUT: source_filename = "copy_or_move.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %struct.CopyOrMove = type { i8 }
// CHECK:STDOUT:
// CHECK:STDOUT: @_Cc.Main = global [1 x i8] zeroinitializer
// CHECK:STDOUT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @_C__global_init.Main, ptr null }]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_Z17make_copy_or_movev.carbon_thunk.(ptr noundef %return) #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !89
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !89
// CHECK:STDOUT:   call void @_Z17make_copy_or_movev(ptr dead_on_unwind writable sret(%struct.CopyOrMove) align 1 %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassGlobal.Main() #1 !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z17pass_copy_or_move10CopyOrMove.carbon_thunk._(ptr @_Cc.Main), !dbg !59
// CHECK:STDOUT:   ret void, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_Z17pass_copy_or_move10CopyOrMove.carbon_thunk._(ptr noundef %0) #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %struct.CopyOrMove, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !89
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !89
// CHECK:STDOUT:   call void @_ZN10CopyOrMoveC1ERKS_(ptr noundef nonnull align 1 dereferenceable(1) %agg.tmp, ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   call void @_Z17pass_copy_or_move10CopyOrMove(ptr nofreeobj noundef align 1 dereferenceable(1) %agg.tmp)
// CHECK:STDOUT:   call void @_ZN10CopyOrMoveD1Ev(ptr noundef nonnull align 1 dereferenceable(1) %agg.tmp) #1
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassTemporary.Main() #1 !dbg !61 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca [1 x i8], align 1, !dbg !62
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !62
// CHECK:STDOUT:   call void @_Z17make_copy_or_movev.carbon_thunk.(ptr %_.var), !dbg !63
// CHECK:STDOUT:   call void @_Z17pass_copy_or_move10CopyOrMove.carbon_thunk.v(ptr %_.var), !dbg !64
// CHECK:STDOUT:   call void @"_CSelfDestruct.47b8ebd18f50db75:core.Destroy.Core"(ptr %_.var), !dbg !62
// CHECK:STDOUT:   ret void, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_Z17pass_copy_or_move10CopyOrMove.carbon_thunk.v(ptr noundef %0) #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %struct.CopyOrMove, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !89
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !89
// CHECK:STDOUT:   call void @_ZN10CopyOrMoveC1EOS_(ptr noundef nonnull align 1 dereferenceable(1) %agg.tmp, ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   call void @_Z17pass_copy_or_move10CopyOrMove(ptr nofreeobj noundef align 1 dereferenceable(1) %agg.tmp)
// CHECK:STDOUT:   call void @_ZN10CopyOrMoveD1Ev(ptr noundef nonnull align 1 dereferenceable(1) %agg.tmp) #1
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: declare void @_ZN10CopyOrMoveD1Ev(ptr noundef nonnull align 1 dereferenceable(1)) unnamed_addr #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.47b8ebd18f50db75:core.Destroy.Core"(ptr %self) #1 !dbg !66 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN10CopyOrMoveD1Ev(ptr %self), !dbg !68
// CHECK:STDOUT:   ret void, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassValue.Main(ptr %c) #1 !dbg !70 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z17pass_copy_or_move10CopyOrMove.carbon_thunk._(ptr %c), !dbg !74
// CHECK:STDOUT:   ret void, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassRef.Main(ptr %c) #1 !dbg !77 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z17pass_copy_or_move10CopyOrMove.carbon_thunk._(ptr %c), !dbg !81
// CHECK:STDOUT:   ret void, !dbg !82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Main() #1 !dbg !84 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z17make_copy_or_movev.carbon_thunk.(ptr @_Cc.Main), !dbg !85
// CHECK:STDOUT:   ret void, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z17make_copy_or_movev(ptr dead_on_unwind writable sret(%struct.CopyOrMove) align 1) #4
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z17pass_copy_or_move10CopyOrMove(ptr nofreeobj noundef align 1 dereferenceable(1)) #4
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN10CopyOrMoveC1ERKS_(ptr noundef nonnull align 1 dereferenceable(1), ptr noundef nonnull align 1 dereferenceable(1)) unnamed_addr #4
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN10CopyOrMoveC1EOS_(ptr noundef nonnull align 1 dereferenceable(1), ptr noundef nonnull align 1 dereferenceable(1)) unnamed_addr #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_ZN10CopyOrMoveD1Ev, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_Z17pass_copy_or_move10CopyOrMove, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { nounwind }
// CHECK:STDOUT: attributes #2 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #4 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!56}
// CHECK:STDOUT: !llvm.module.flags = !{!49, !50, !51, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!54}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !9 = !{null, !8}
// CHECK:STDOUT: !10 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !11 = !{null}
// CHECK:STDOUT: !12 = !DISubroutineType(types: !11)
// CHECK:STDOUT: !43 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !44 = !{!"omnipotent char", !43, i64 0}
// CHECK:STDOUT: !45 = !{!"any pointer", !44, i64 0}
// CHECK:STDOUT: !49 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !50 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !51 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !52 = !{!"int", !44, i64 0}
// CHECK:STDOUT: !53 = !{!"__libc_errno", !52, i64 0}
// CHECK:STDOUT: !54 = !{!53, !52, i64 0}
// CHECK:STDOUT: !55 = !DIFile(filename: "copy_or_move.carbon", directory: "")
// CHECK:STDOUT: !56 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !55, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "PassGlobal", linkageName: "_CPassGlobal.Main", scope: null, file: !55, line: 18, type: !12, spFlags: DISPFlagDefinition, unit: !56)
// CHECK:STDOUT: !59 = !DILocation(line: 19, column: 3, scope: !57)
// CHECK:STDOUT: !60 = !DILocation(line: 18, column: 1, scope: !57)
// CHECK:STDOUT: !61 = distinct !DISubprogram(name: "PassTemporary", linkageName: "_CPassTemporary.Main", scope: null, file: !55, line: 22, type: !12, spFlags: DISPFlagDefinition, unit: !56)
// CHECK:STDOUT: !62 = !DILocation(line: 13, column: 34, scope: !61)
// CHECK:STDOUT: !63 = !DILocation(line: 23, column: 25, scope: !61)
// CHECK:STDOUT: !64 = !DILocation(line: 23, column: 3, scope: !61)
// CHECK:STDOUT: !65 = !DILocation(line: 22, column: 1, scope: !61)
// CHECK:STDOUT: !66 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.47b8ebd18f50db75:core.Destroy.Core", scope: null, file: !55, line: 13, type: !10, spFlags: DISPFlagDefinition, unit: !56, retainedNodes: !69)
// CHECK:STDOUT: !67 = !DILocalVariable(arg: 1, scope: !66, type: !8)
// CHECK:STDOUT: !68 = !DILocation(line: 13, column: 34, scope: !66)
// CHECK:STDOUT: !69 = !{!67}
// CHECK:STDOUT: !70 = distinct !DISubprogram(name: "PassValue", linkageName: "_CPassValue.Main", scope: null, file: !55, line: 26, type: !10, spFlags: DISPFlagDefinition, unit: !56, retainedNodes: !76)
// CHECK:STDOUT: !71 = !DILocalVariable(arg: 1, scope: !70, type: !8)
// CHECK:STDOUT: !74 = !DILocation(line: 27, column: 3, scope: !70)
// CHECK:STDOUT: !75 = !DILocation(line: 26, column: 1, scope: !70)
// CHECK:STDOUT: !76 = !{!71}
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "PassRef", linkageName: "_CPassRef.Main", scope: null, file: !55, line: 30, type: !10, spFlags: DISPFlagDefinition, unit: !56, retainedNodes: !83)
// CHECK:STDOUT: !78 = !DILocalVariable(arg: 1, scope: !77, type: !8)
// CHECK:STDOUT: !81 = !DILocation(line: 31, column: 3, scope: !77)
// CHECK:STDOUT: !82 = !DILocation(line: 30, column: 1, scope: !77)
// CHECK:STDOUT: !83 = !{!78}
// CHECK:STDOUT: !84 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !55, type: !12, spFlags: DISPFlagDefinition, unit: !56)
// CHECK:STDOUT: !85 = !DILocation(line: 16, column: 25, scope: !84)
// CHECK:STDOUT: !87 = !DILocation(line: 0, scope: !84)
// CHECK:STDOUT: !88 = !{!"p1 _ZTS10CopyOrMove", !45, i64 0}
// CHECK:STDOUT: !89 = !{!88, !88, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'move_only.carbon'
// CHECK:STDOUT: source_filename = "move_only.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %struct.MoveOnly = type { i8 }
// CHECK:STDOUT:
// CHECK:STDOUT: @_Cc.Main = global [1 x i8] zeroinitializer
// CHECK:STDOUT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @_C__global_init.Main, ptr null }]
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_Z14make_move_onlyv.carbon_thunk.(ptr noundef %return) #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !109
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !109
// CHECK:STDOUT:   call void @_Z14make_move_onlyv(ptr dead_on_unwind writable sret(%struct.MoveOnly) align 1 %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassGlobalByMove.Main() #1 !dbg !93 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !94
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassTemporary.Main() #1 !dbg !95 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca [1 x i8], align 1, !dbg !96
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !96
// CHECK:STDOUT:   call void @_Z14make_move_onlyv.carbon_thunk.(ptr %_.var), !dbg !97
// CHECK:STDOUT:   call void @_Z14pass_move_only8MoveOnly.carbon_thunk.v(ptr %_.var), !dbg !98
// CHECK:STDOUT:   call void @"_CSelfDestruct.48e50e337b0b966c:core.Destroy.Core"(ptr %_.var), !dbg !96
// CHECK:STDOUT:   ret void, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_Z14pass_move_only8MoveOnly.carbon_thunk.v(ptr noundef %0) #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %struct.MoveOnly, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !109
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !109
// CHECK:STDOUT:   call void @_ZN8MoveOnlyC1EOS_(ptr noundef nonnull align 1 dereferenceable(1) %agg.tmp, ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   call void @_Z14pass_move_only8MoveOnly(ptr nofreeobj noundef align 1 dereferenceable(1) %agg.tmp)
// CHECK:STDOUT:   call void @_ZN8MoveOnlyD1Ev(ptr noundef nonnull align 1 dereferenceable(1) %agg.tmp) #1
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: declare void @_ZN8MoveOnlyD1Ev(ptr noundef nonnull align 1 dereferenceable(1)) unnamed_addr #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.48e50e337b0b966c:core.Destroy.Core"(ptr %self) #1 !dbg !100 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN8MoveOnlyD1Ev(ptr %self), !dbg !102
// CHECK:STDOUT:   ret void, !dbg !102
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Main() #1 !dbg !104 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z14make_move_onlyv.carbon_thunk.(ptr @_Cc.Main), !dbg !105
// CHECK:STDOUT:   ret void, !dbg !107
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z14make_move_onlyv(ptr dead_on_unwind writable sret(%struct.MoveOnly) align 1) #4
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z14pass_move_only8MoveOnly(ptr nofreeobj noundef align 1 dereferenceable(1)) #4
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN8MoveOnlyC1EOS_(ptr noundef nonnull align 1 dereferenceable(1), ptr noundef nonnull align 1 dereferenceable(1)) unnamed_addr #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_ZN8MoveOnlyD1Ev, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { nounwind }
// CHECK:STDOUT: attributes #2 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #4 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!92}
// CHECK:STDOUT: !llvm.module.flags = !{!49, !50, !51, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!54}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !9 = !{null, !8}
// CHECK:STDOUT: !10 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !11 = !{null}
// CHECK:STDOUT: !12 = !DISubroutineType(types: !11)
// CHECK:STDOUT: !43 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !44 = !{!"omnipotent char", !43, i64 0}
// CHECK:STDOUT: !45 = !{!"any pointer", !44, i64 0}
// CHECK:STDOUT: !49 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !50 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !51 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !52 = !{!"int", !44, i64 0}
// CHECK:STDOUT: !53 = !{!"__libc_errno", !52, i64 0}
// CHECK:STDOUT: !54 = !{!53, !52, i64 0}
// CHECK:STDOUT: !91 = !DIFile(filename: "move_only.carbon", directory: "")
// CHECK:STDOUT: !92 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !91, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !93 = distinct !DISubprogram(name: "PassGlobalByMove", linkageName: "_CPassGlobalByMove.Main", scope: null, file: !91, line: 22, type: !12, spFlags: DISPFlagDefinition, unit: !92)
// CHECK:STDOUT: !94 = !DILocation(line: 22, column: 1, scope: !93)
// CHECK:STDOUT: !95 = distinct !DISubprogram(name: "PassTemporary", linkageName: "_CPassTemporary.Main", scope: null, file: !91, line: 27, type: !12, spFlags: DISPFlagDefinition, unit: !92)
// CHECK:STDOUT: !96 = !DILocation(line: 17, column: 29, scope: !95)
// CHECK:STDOUT: !97 = !DILocation(line: 28, column: 22, scope: !95)
// CHECK:STDOUT: !98 = !DILocation(line: 28, column: 3, scope: !95)
// CHECK:STDOUT: !99 = !DILocation(line: 27, column: 1, scope: !95)
// CHECK:STDOUT: !100 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.48e50e337b0b966c:core.Destroy.Core", scope: null, file: !91, line: 17, type: !10, spFlags: DISPFlagDefinition, unit: !92, retainedNodes: !103)
// CHECK:STDOUT: !101 = !DILocalVariable(arg: 1, scope: !100, type: !8)
// CHECK:STDOUT: !102 = !DILocation(line: 17, column: 29, scope: !100)
// CHECK:STDOUT: !103 = !{!101}
// CHECK:STDOUT: !104 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !91, type: !12, spFlags: DISPFlagDefinition, unit: !92)
// CHECK:STDOUT: !105 = !DILocation(line: 20, column: 23, scope: !104)
// CHECK:STDOUT: !107 = !DILocation(line: 0, scope: !104)
// CHECK:STDOUT: !108 = !{!"p1 _ZTS8MoveOnly", !45, i64 0}
// CHECK:STDOUT: !109 = !{!108, !108, i64 0}
// CHECK:STDOUT:
