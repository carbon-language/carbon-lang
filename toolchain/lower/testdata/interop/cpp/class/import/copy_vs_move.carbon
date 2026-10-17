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
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !58
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !58
// CHECK:STDOUT:   call void @_Z13make_copyablev(ptr dead_on_unwind writable sret(%struct.Copyable) align 1 %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassGlobal.Main() #1 !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z13pass_copyable8Copyable.carbon_thunk._(ptr @_Cc.Main), !dbg !23
// CHECK:STDOUT:   ret void, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_Z13pass_copyable8Copyable.carbon_thunk._(ptr noundef %0) #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %struct.Copyable, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !58
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !58
// CHECK:STDOUT:   call void @_ZN8CopyableC1ERKS_(ptr noundef nonnull align 1 dereferenceable(1) %agg.tmp, ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   call void @_Z13pass_copyable8Copyable(ptr nofreeobj noundef align 1 dereferenceable(1) %agg.tmp)
// CHECK:STDOUT:   call void @_ZN8CopyableD1Ev(ptr noundef nonnull align 1 dereferenceable(1) %agg.tmp) #1
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassTemporary.Main() #1 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc21_39.1.temp = alloca [1 x i8], align 1, !dbg !26
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc21_39.1.temp), !dbg !26
// CHECK:STDOUT:   call void @_Z13make_copyablev.carbon_thunk.(ptr %.loc21_39.1.temp), !dbg !26
// CHECK:STDOUT:   call void @_Z13pass_copyable8Copyable.carbon_thunk._(ptr %.loc21_39.1.temp), !dbg !27
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.8317741dd008c11f(ptr %.loc21_39.1.temp), !dbg !26
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: declare void @_ZN8CopyableD1Ev(ptr noundef nonnull align 1 dereferenceable(1)) unnamed_addr #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassValue.Main(ptr %c) #1 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z13pass_copyable8Copyable.carbon_thunk._(ptr %c), !dbg !33
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassRef.Main(ptr %c) #1 !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z13pass_copyable8Copyable.carbon_thunk._(ptr %c), !dbg !40
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.8317741dd008c11f(ptr %t) #1 !dbg !47 {
// CHECK:STDOUT:   call void @_ZN8CopyableD1Ev(ptr %t), !dbg !51
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Main() #1 !dbg !43 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z13make_copyablev.carbon_thunk.(ptr @_Cc.Main), !dbg !44
// CHECK:STDOUT:   ret void, !dbg !46
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!15}
// CHECK:STDOUT: !llvm.module.flags = !{!60, !61, !62, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!65}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !14 = !DIFile(filename: "copyable.carbon", directory: "")
// CHECK:STDOUT: !15 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !14, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !17 = !{null, !16}
// CHECK:STDOUT: !18 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !19 = !{null}
// CHECK:STDOUT: !20 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "PassGlobal", linkageName: "_CPassGlobal.Main", scope: null, file: !14, line: 16, type: !20, spFlags: DISPFlagDefinition, unit: !15)
// CHECK:STDOUT: !23 = !DILocation(line: 17, column: 3, scope: !21)
// CHECK:STDOUT: !24 = !DILocation(line: 16, column: 1, scope: !21)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "PassTemporary", linkageName: "_CPassTemporary.Main", scope: null, file: !14, line: 20, type: !20, spFlags: DISPFlagDefinition, unit: !15)
// CHECK:STDOUT: !26 = !DILocation(line: 21, column: 21, scope: !25)
// CHECK:STDOUT: !27 = !DILocation(line: 21, column: 3, scope: !25)
// CHECK:STDOUT: !28 = !DILocation(line: 20, column: 1, scope: !25)
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "PassValue", linkageName: "_CPassValue.Main", scope: null, file: !14, line: 24, type: !18, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !35)
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 1, scope: !29, type: !16)
// CHECK:STDOUT: !33 = !DILocation(line: 25, column: 3, scope: !29)
// CHECK:STDOUT: !34 = !DILocation(line: 24, column: 1, scope: !29)
// CHECK:STDOUT: !35 = !{!30}
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "PassRef", linkageName: "_CPassRef.Main", scope: null, file: !14, line: 28, type: !18, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !42)
// CHECK:STDOUT: !37 = !DILocalVariable(arg: 1, scope: !36, type: !16)
// CHECK:STDOUT: !40 = !DILocation(line: 29, column: 3, scope: !36)
// CHECK:STDOUT: !41 = !DILocation(line: 28, column: 1, scope: !36)
// CHECK:STDOUT: !42 = !{!37}
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !14, type: !20, spFlags: DISPFlagDefinition, unit: !15)
// CHECK:STDOUT: !44 = !DILocation(line: 14, column: 23, scope: !43)
// CHECK:STDOUT: !46 = !DILocation(line: 0, scope: !43)
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.8317741dd008c11f", scope: null, file: !4, line: 30, type: !18, spFlags: DISPFlagDefinition, unit: !15, retainedNodes: !53)
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 1, scope: !47, type: !16)
// CHECK:STDOUT: !51 = !DILocation(line: 32, column: 5, scope: !47)
// CHECK:STDOUT: !52 = !DILocation(line: 30, column: 1, scope: !47)
// CHECK:STDOUT: !53 = !{!48}
// CHECK:STDOUT: !54 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !55 = !{!"omnipotent char", !54, i64 0}
// CHECK:STDOUT: !56 = !{!"any pointer", !55, i64 0}
// CHECK:STDOUT: !57 = !{!"p1 _ZTS8Copyable", !56, i64 0}
// CHECK:STDOUT: !58 = !{!57, !57, i64 0}
// CHECK:STDOUT: !60 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !61 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !62 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !63 = !{!"int", !55, i64 0}
// CHECK:STDOUT: !64 = !{!"__libc_errno", !63, i64 0}
// CHECK:STDOUT: !65 = !{!64, !63, i64 0}
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
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !103
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !103
// CHECK:STDOUT:   call void @_Z17make_copy_or_movev(ptr dead_on_unwind writable sret(%struct.CopyOrMove) align 1 %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassGlobal.Main() #1 !dbg !68 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z17pass_copy_or_move10CopyOrMove.carbon_thunk._(ptr @_Cc.Main), !dbg !70
// CHECK:STDOUT:   ret void, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_Z17pass_copy_or_move10CopyOrMove.carbon_thunk._(ptr noundef %0) #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %struct.CopyOrMove, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !103
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !103
// CHECK:STDOUT:   call void @_ZN10CopyOrMoveC1ERKS_(ptr noundef nonnull align 1 dereferenceable(1) %agg.tmp, ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   call void @_Z17pass_copy_or_move10CopyOrMove(ptr nofreeobj noundef align 1 dereferenceable(1) %agg.tmp)
// CHECK:STDOUT:   call void @_ZN10CopyOrMoveD1Ev(ptr noundef nonnull align 1 dereferenceable(1) %agg.tmp) #1
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassTemporary.Main() #1 !dbg !72 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca [1 x i8], align 1, !dbg !73
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !73
// CHECK:STDOUT:   call void @_Z17make_copy_or_movev.carbon_thunk.(ptr %_.var), !dbg !74
// CHECK:STDOUT:   call void @_Z17pass_copy_or_move10CopyOrMove.carbon_thunk.v(ptr %_.var), !dbg !75
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.5b53d15ddb55a6e2(ptr %_.var), !dbg !73
// CHECK:STDOUT:   ret void, !dbg !76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_Z17pass_copy_or_move10CopyOrMove.carbon_thunk.v(ptr noundef %0) #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %struct.CopyOrMove, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !103
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !103
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
// CHECK:STDOUT: define void @_CPassValue.Main(ptr %c) #1 !dbg !77 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z17pass_copy_or_move10CopyOrMove.carbon_thunk._(ptr %c), !dbg !81
// CHECK:STDOUT:   ret void, !dbg !82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassRef.Main(ptr %c) #1 !dbg !84 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z17pass_copy_or_move10CopyOrMove.carbon_thunk._(ptr %c), !dbg !88
// CHECK:STDOUT:   ret void, !dbg !89
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.5b53d15ddb55a6e2(ptr %t) #1 !dbg !95 {
// CHECK:STDOUT:   call void @_ZN10CopyOrMoveD1Ev(ptr %t), !dbg !99
// CHECK:STDOUT:   ret void, !dbg !100
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Main() #1 !dbg !91 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z17make_copy_or_movev.carbon_thunk.(ptr @_Cc.Main), !dbg !92
// CHECK:STDOUT:   ret void, !dbg !94
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!67}
// CHECK:STDOUT: !llvm.module.flags = !{!60, !61, !62, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!65}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !17 = !{null, !16}
// CHECK:STDOUT: !18 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !19 = !{null}
// CHECK:STDOUT: !20 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !54 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !55 = !{!"omnipotent char", !54, i64 0}
// CHECK:STDOUT: !56 = !{!"any pointer", !55, i64 0}
// CHECK:STDOUT: !60 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !61 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !62 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !63 = !{!"int", !55, i64 0}
// CHECK:STDOUT: !64 = !{!"__libc_errno", !63, i64 0}
// CHECK:STDOUT: !65 = !{!64, !63, i64 0}
// CHECK:STDOUT: !66 = !DIFile(filename: "copy_or_move.carbon", directory: "")
// CHECK:STDOUT: !67 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !66, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "PassGlobal", linkageName: "_CPassGlobal.Main", scope: null, file: !66, line: 18, type: !20, spFlags: DISPFlagDefinition, unit: !67)
// CHECK:STDOUT: !70 = !DILocation(line: 19, column: 3, scope: !68)
// CHECK:STDOUT: !71 = !DILocation(line: 18, column: 1, scope: !68)
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "PassTemporary", linkageName: "_CPassTemporary.Main", scope: null, file: !66, line: 22, type: !20, spFlags: DISPFlagDefinition, unit: !67)
// CHECK:STDOUT: !73 = !DILocation(line: 13, column: 34, scope: !72)
// CHECK:STDOUT: !74 = !DILocation(line: 23, column: 25, scope: !72)
// CHECK:STDOUT: !75 = !DILocation(line: 23, column: 3, scope: !72)
// CHECK:STDOUT: !76 = !DILocation(line: 22, column: 1, scope: !72)
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "PassValue", linkageName: "_CPassValue.Main", scope: null, file: !66, line: 26, type: !18, spFlags: DISPFlagDefinition, unit: !67, retainedNodes: !83)
// CHECK:STDOUT: !78 = !DILocalVariable(arg: 1, scope: !77, type: !16)
// CHECK:STDOUT: !81 = !DILocation(line: 27, column: 3, scope: !77)
// CHECK:STDOUT: !82 = !DILocation(line: 26, column: 1, scope: !77)
// CHECK:STDOUT: !83 = !{!78}
// CHECK:STDOUT: !84 = distinct !DISubprogram(name: "PassRef", linkageName: "_CPassRef.Main", scope: null, file: !66, line: 30, type: !18, spFlags: DISPFlagDefinition, unit: !67, retainedNodes: !90)
// CHECK:STDOUT: !85 = !DILocalVariable(arg: 1, scope: !84, type: !16)
// CHECK:STDOUT: !88 = !DILocation(line: 31, column: 3, scope: !84)
// CHECK:STDOUT: !89 = !DILocation(line: 30, column: 1, scope: !84)
// CHECK:STDOUT: !90 = !{!85}
// CHECK:STDOUT: !91 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !66, type: !20, spFlags: DISPFlagDefinition, unit: !67)
// CHECK:STDOUT: !92 = !DILocation(line: 16, column: 25, scope: !91)
// CHECK:STDOUT: !94 = !DILocation(line: 0, scope: !91)
// CHECK:STDOUT: !95 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.5b53d15ddb55a6e2", scope: null, file: !4, line: 30, type: !18, spFlags: DISPFlagDefinition, unit: !67, retainedNodes: !101)
// CHECK:STDOUT: !96 = !DILocalVariable(arg: 1, scope: !95, type: !16)
// CHECK:STDOUT: !99 = !DILocation(line: 32, column: 5, scope: !95)
// CHECK:STDOUT: !100 = !DILocation(line: 30, column: 1, scope: !95)
// CHECK:STDOUT: !101 = !{!96}
// CHECK:STDOUT: !102 = !{!"p1 _ZTS10CopyOrMove", !56, i64 0}
// CHECK:STDOUT: !103 = !{!102, !102, i64 0}
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
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !126
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !126
// CHECK:STDOUT:   call void @_Z14make_move_onlyv(ptr dead_on_unwind writable sret(%struct.MoveOnly) align 1 %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassGlobalByMove.Main() #1 !dbg !107 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !108
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassTemporary.Main() #1 !dbg !109 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca [1 x i8], align 1, !dbg !110
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !110
// CHECK:STDOUT:   call void @_Z14make_move_onlyv.carbon_thunk.(ptr %_.var), !dbg !111
// CHECK:STDOUT:   call void @_Z14pass_move_only8MoveOnly.carbon_thunk.v(ptr %_.var), !dbg !112
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.5bf528c71c41a02a(ptr %_.var), !dbg !110
// CHECK:STDOUT:   ret void, !dbg !113
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_Z14pass_move_only8MoveOnly.carbon_thunk.v(ptr noundef %0) #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %struct.MoveOnly, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !126
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !126
// CHECK:STDOUT:   call void @_ZN8MoveOnlyC1EOS_(ptr noundef nonnull align 1 dereferenceable(1) %agg.tmp, ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   call void @_Z14pass_move_only8MoveOnly(ptr nofreeobj noundef align 1 dereferenceable(1) %agg.tmp)
// CHECK:STDOUT:   call void @_ZN8MoveOnlyD1Ev(ptr noundef nonnull align 1 dereferenceable(1) %agg.tmp) #1
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: declare void @_ZN8MoveOnlyD1Ev(ptr noundef nonnull align 1 dereferenceable(1)) unnamed_addr #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.5bf528c71c41a02a(ptr %t) #1 !dbg !118 {
// CHECK:STDOUT:   call void @_ZN8MoveOnlyD1Ev(ptr %t), !dbg !122
// CHECK:STDOUT:   ret void, !dbg !123
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Main() #1 !dbg !114 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z14make_move_onlyv.carbon_thunk.(ptr @_Cc.Main), !dbg !115
// CHECK:STDOUT:   ret void, !dbg !117
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!106}
// CHECK:STDOUT: !llvm.module.flags = !{!60, !61, !62, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!65}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !17 = !{null, !16}
// CHECK:STDOUT: !18 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !19 = !{null}
// CHECK:STDOUT: !20 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !54 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !55 = !{!"omnipotent char", !54, i64 0}
// CHECK:STDOUT: !56 = !{!"any pointer", !55, i64 0}
// CHECK:STDOUT: !60 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !61 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !62 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !63 = !{!"int", !55, i64 0}
// CHECK:STDOUT: !64 = !{!"__libc_errno", !63, i64 0}
// CHECK:STDOUT: !65 = !{!64, !63, i64 0}
// CHECK:STDOUT: !105 = !DIFile(filename: "move_only.carbon", directory: "")
// CHECK:STDOUT: !106 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !105, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !107 = distinct !DISubprogram(name: "PassGlobalByMove", linkageName: "_CPassGlobalByMove.Main", scope: null, file: !105, line: 22, type: !20, spFlags: DISPFlagDefinition, unit: !106)
// CHECK:STDOUT: !108 = !DILocation(line: 22, column: 1, scope: !107)
// CHECK:STDOUT: !109 = distinct !DISubprogram(name: "PassTemporary", linkageName: "_CPassTemporary.Main", scope: null, file: !105, line: 27, type: !20, spFlags: DISPFlagDefinition, unit: !106)
// CHECK:STDOUT: !110 = !DILocation(line: 17, column: 29, scope: !109)
// CHECK:STDOUT: !111 = !DILocation(line: 28, column: 22, scope: !109)
// CHECK:STDOUT: !112 = !DILocation(line: 28, column: 3, scope: !109)
// CHECK:STDOUT: !113 = !DILocation(line: 27, column: 1, scope: !109)
// CHECK:STDOUT: !114 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !105, type: !20, spFlags: DISPFlagDefinition, unit: !106)
// CHECK:STDOUT: !115 = !DILocation(line: 20, column: 23, scope: !114)
// CHECK:STDOUT: !117 = !DILocation(line: 0, scope: !114)
// CHECK:STDOUT: !118 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.5bf528c71c41a02a", scope: null, file: !4, line: 30, type: !18, spFlags: DISPFlagDefinition, unit: !106, retainedNodes: !124)
// CHECK:STDOUT: !119 = !DILocalVariable(arg: 1, scope: !118, type: !16)
// CHECK:STDOUT: !122 = !DILocation(line: 32, column: 5, scope: !118)
// CHECK:STDOUT: !123 = !DILocation(line: 30, column: 1, scope: !118)
// CHECK:STDOUT: !124 = !{!119}
// CHECK:STDOUT: !125 = !{!"p1 _ZTS8MoveOnly", !56, i64 0}
// CHECK:STDOUT: !126 = !{!125, !125, i64 0}
// CHECK:STDOUT:
