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
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !56
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !56
// CHECK:STDOUT:   call void @_Z13make_copyablev(ptr dead_on_unwind writable sret(%struct.Copyable) align 1 %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassGlobal.Main() #1 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z13pass_copyable8Copyable.carbon_thunk._(ptr @_Cc.Main), !dbg !21
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_Z13pass_copyable8Copyable.carbon_thunk._(ptr noundef %0) #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %struct.Copyable, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !56
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !56
// CHECK:STDOUT:   call void @_ZN8CopyableC1ERKS_(ptr noundef nonnull align 1 dereferenceable(1) %agg.tmp, ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   call void @_Z13pass_copyable8Copyable(ptr nofreeobj noundef align 1 dereferenceable(1) %agg.tmp)
// CHECK:STDOUT:   call void @_ZN8CopyableD1Ev(ptr noundef nonnull align 1 dereferenceable(1) %agg.tmp) #1
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassTemporary.Main() #1 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc21_39.1.temp = alloca [1 x i8], align 1, !dbg !24
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc21_39.1.temp), !dbg !24
// CHECK:STDOUT:   call void @_Z13make_copyablev.carbon_thunk.(ptr %.loc21_39.1.temp), !dbg !24
// CHECK:STDOUT:   call void @_Z13pass_copyable8Copyable.carbon_thunk._(ptr %.loc21_39.1.temp), !dbg !25
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.8317741dd008c11f(ptr %.loc21_39.1.temp), !dbg !24
// CHECK:STDOUT:   ret void, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: declare void @_ZN8CopyableD1Ev(ptr noundef nonnull align 1 dereferenceable(1)) unnamed_addr #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassValue.Main(ptr %c) #1 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z13pass_copyable8Copyable.carbon_thunk._(ptr %c), !dbg !31
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassRef.Main(ptr %c) #1 !dbg !34 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z13pass_copyable8Copyable.carbon_thunk._(ptr %c), !dbg !38
// CHECK:STDOUT:   ret void, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.8317741dd008c11f(ptr %t) #1 !dbg !45 {
// CHECK:STDOUT:   call void @_ZN8CopyableD1Ev(ptr %t), !dbg !49
// CHECK:STDOUT:   ret void, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Main() #1 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z13make_copyablev.carbon_thunk.(ptr @_Cc.Main), !dbg !42
// CHECK:STDOUT:   ret void, !dbg !44
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!13}
// CHECK:STDOUT: !llvm.module.flags = !{!58, !59, !60, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!63}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !12 = !DIFile(filename: "copyable.carbon", directory: "")
// CHECK:STDOUT: !13 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !12, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !15 = !{null, !14}
// CHECK:STDOUT: !16 = !DISubroutineType(types: !15)
// CHECK:STDOUT: !17 = !{null}
// CHECK:STDOUT: !18 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "PassGlobal", linkageName: "_CPassGlobal.Main", scope: null, file: !12, line: 16, type: !18, spFlags: DISPFlagDefinition, unit: !13)
// CHECK:STDOUT: !21 = !DILocation(line: 17, column: 3, scope: !19)
// CHECK:STDOUT: !22 = !DILocation(line: 16, column: 1, scope: !19)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "PassTemporary", linkageName: "_CPassTemporary.Main", scope: null, file: !12, line: 20, type: !18, spFlags: DISPFlagDefinition, unit: !13)
// CHECK:STDOUT: !24 = !DILocation(line: 21, column: 21, scope: !23)
// CHECK:STDOUT: !25 = !DILocation(line: 21, column: 3, scope: !23)
// CHECK:STDOUT: !26 = !DILocation(line: 20, column: 1, scope: !23)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "PassValue", linkageName: "_CPassValue.Main", scope: null, file: !12, line: 24, type: !16, spFlags: DISPFlagDefinition, unit: !13, retainedNodes: !33)
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 1, scope: !27, type: !14)
// CHECK:STDOUT: !31 = !DILocation(line: 25, column: 3, scope: !27)
// CHECK:STDOUT: !32 = !DILocation(line: 24, column: 1, scope: !27)
// CHECK:STDOUT: !33 = !{!28}
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "PassRef", linkageName: "_CPassRef.Main", scope: null, file: !12, line: 28, type: !16, spFlags: DISPFlagDefinition, unit: !13, retainedNodes: !40)
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !34, type: !14)
// CHECK:STDOUT: !38 = !DILocation(line: 29, column: 3, scope: !34)
// CHECK:STDOUT: !39 = !DILocation(line: 28, column: 1, scope: !34)
// CHECK:STDOUT: !40 = !{!35}
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !12, type: !18, spFlags: DISPFlagDefinition, unit: !13)
// CHECK:STDOUT: !42 = !DILocation(line: 14, column: 23, scope: !41)
// CHECK:STDOUT: !44 = !DILocation(line: 0, scope: !41)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.8317741dd008c11f", scope: null, file: !4, line: 30, type: !16, spFlags: DISPFlagDefinition, unit: !13, retainedNodes: !51)
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 1, scope: !45, type: !14)
// CHECK:STDOUT: !49 = !DILocation(line: 32, column: 5, scope: !45)
// CHECK:STDOUT: !50 = !DILocation(line: 30, column: 1, scope: !45)
// CHECK:STDOUT: !51 = !{!46}
// CHECK:STDOUT: !52 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !53 = !{!"omnipotent char", !52, i64 0}
// CHECK:STDOUT: !54 = !{!"any pointer", !53, i64 0}
// CHECK:STDOUT: !55 = !{!"p1 _ZTS8Copyable", !54, i64 0}
// CHECK:STDOUT: !56 = !{!55, !55, i64 0}
// CHECK:STDOUT: !58 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !59 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !60 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !61 = !{!"int", !53, i64 0}
// CHECK:STDOUT: !62 = !{!"__libc_errno", !61, i64 0}
// CHECK:STDOUT: !63 = !{!62, !61, i64 0}
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
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !101
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !101
// CHECK:STDOUT:   call void @_Z17make_copy_or_movev(ptr dead_on_unwind writable sret(%struct.CopyOrMove) align 1 %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassGlobal.Main() #1 !dbg !66 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z17pass_copy_or_move10CopyOrMove.carbon_thunk._(ptr @_Cc.Main), !dbg !68
// CHECK:STDOUT:   ret void, !dbg !69
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_Z17pass_copy_or_move10CopyOrMove.carbon_thunk._(ptr noundef %0) #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %struct.CopyOrMove, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !101
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !101
// CHECK:STDOUT:   call void @_ZN10CopyOrMoveC1ERKS_(ptr noundef nonnull align 1 dereferenceable(1) %agg.tmp, ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   call void @_Z17pass_copy_or_move10CopyOrMove(ptr nofreeobj noundef align 1 dereferenceable(1) %agg.tmp)
// CHECK:STDOUT:   call void @_ZN10CopyOrMoveD1Ev(ptr noundef nonnull align 1 dereferenceable(1) %agg.tmp) #1
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassTemporary.Main() #1 !dbg !70 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca [1 x i8], align 1, !dbg !71
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !71
// CHECK:STDOUT:   call void @_Z17make_copy_or_movev.carbon_thunk.(ptr %_.var), !dbg !72
// CHECK:STDOUT:   call void @_Z17pass_copy_or_move10CopyOrMove.carbon_thunk.v(ptr %_.var), !dbg !73
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.5b53d15ddb55a6e2(ptr %_.var), !dbg !71
// CHECK:STDOUT:   ret void, !dbg !74
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_Z17pass_copy_or_move10CopyOrMove.carbon_thunk.v(ptr noundef %0) #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %struct.CopyOrMove, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !101
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !101
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
// CHECK:STDOUT: define void @_CPassValue.Main(ptr %c) #1 !dbg !75 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z17pass_copy_or_move10CopyOrMove.carbon_thunk._(ptr %c), !dbg !79
// CHECK:STDOUT:   ret void, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassRef.Main(ptr %c) #1 !dbg !82 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z17pass_copy_or_move10CopyOrMove.carbon_thunk._(ptr %c), !dbg !86
// CHECK:STDOUT:   ret void, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.5b53d15ddb55a6e2(ptr %t) #1 !dbg !93 {
// CHECK:STDOUT:   call void @_ZN10CopyOrMoveD1Ev(ptr %t), !dbg !97
// CHECK:STDOUT:   ret void, !dbg !98
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Main() #1 !dbg !89 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z17make_copy_or_movev.carbon_thunk.(ptr @_Cc.Main), !dbg !90
// CHECK:STDOUT:   ret void, !dbg !92
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!65}
// CHECK:STDOUT: !llvm.module.flags = !{!58, !59, !60, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!63}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !15 = !{null, !14}
// CHECK:STDOUT: !16 = !DISubroutineType(types: !15)
// CHECK:STDOUT: !17 = !{null}
// CHECK:STDOUT: !18 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !52 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !53 = !{!"omnipotent char", !52, i64 0}
// CHECK:STDOUT: !54 = !{!"any pointer", !53, i64 0}
// CHECK:STDOUT: !58 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !59 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !60 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !61 = !{!"int", !53, i64 0}
// CHECK:STDOUT: !62 = !{!"__libc_errno", !61, i64 0}
// CHECK:STDOUT: !63 = !{!62, !61, i64 0}
// CHECK:STDOUT: !64 = !DIFile(filename: "copy_or_move.carbon", directory: "")
// CHECK:STDOUT: !65 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !64, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !66 = distinct !DISubprogram(name: "PassGlobal", linkageName: "_CPassGlobal.Main", scope: null, file: !64, line: 18, type: !18, spFlags: DISPFlagDefinition, unit: !65)
// CHECK:STDOUT: !68 = !DILocation(line: 19, column: 3, scope: !66)
// CHECK:STDOUT: !69 = !DILocation(line: 18, column: 1, scope: !66)
// CHECK:STDOUT: !70 = distinct !DISubprogram(name: "PassTemporary", linkageName: "_CPassTemporary.Main", scope: null, file: !64, line: 22, type: !18, spFlags: DISPFlagDefinition, unit: !65)
// CHECK:STDOUT: !71 = !DILocation(line: 13, column: 34, scope: !70)
// CHECK:STDOUT: !72 = !DILocation(line: 23, column: 25, scope: !70)
// CHECK:STDOUT: !73 = !DILocation(line: 23, column: 3, scope: !70)
// CHECK:STDOUT: !74 = !DILocation(line: 22, column: 1, scope: !70)
// CHECK:STDOUT: !75 = distinct !DISubprogram(name: "PassValue", linkageName: "_CPassValue.Main", scope: null, file: !64, line: 26, type: !16, spFlags: DISPFlagDefinition, unit: !65, retainedNodes: !81)
// CHECK:STDOUT: !76 = !DILocalVariable(arg: 1, scope: !75, type: !14)
// CHECK:STDOUT: !79 = !DILocation(line: 27, column: 3, scope: !75)
// CHECK:STDOUT: !80 = !DILocation(line: 26, column: 1, scope: !75)
// CHECK:STDOUT: !81 = !{!76}
// CHECK:STDOUT: !82 = distinct !DISubprogram(name: "PassRef", linkageName: "_CPassRef.Main", scope: null, file: !64, line: 30, type: !16, spFlags: DISPFlagDefinition, unit: !65, retainedNodes: !88)
// CHECK:STDOUT: !83 = !DILocalVariable(arg: 1, scope: !82, type: !14)
// CHECK:STDOUT: !86 = !DILocation(line: 31, column: 3, scope: !82)
// CHECK:STDOUT: !87 = !DILocation(line: 30, column: 1, scope: !82)
// CHECK:STDOUT: !88 = !{!83}
// CHECK:STDOUT: !89 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !64, type: !18, spFlags: DISPFlagDefinition, unit: !65)
// CHECK:STDOUT: !90 = !DILocation(line: 16, column: 25, scope: !89)
// CHECK:STDOUT: !92 = !DILocation(line: 0, scope: !89)
// CHECK:STDOUT: !93 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.5b53d15ddb55a6e2", scope: null, file: !4, line: 30, type: !16, spFlags: DISPFlagDefinition, unit: !65, retainedNodes: !99)
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 1, scope: !93, type: !14)
// CHECK:STDOUT: !97 = !DILocation(line: 32, column: 5, scope: !93)
// CHECK:STDOUT: !98 = !DILocation(line: 30, column: 1, scope: !93)
// CHECK:STDOUT: !99 = !{!94}
// CHECK:STDOUT: !100 = !{!"p1 _ZTS10CopyOrMove", !54, i64 0}
// CHECK:STDOUT: !101 = !{!100, !100, i64 0}
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
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !124
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !124
// CHECK:STDOUT:   call void @_Z14make_move_onlyv(ptr dead_on_unwind writable sret(%struct.MoveOnly) align 1 %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassGlobalByMove.Main() #1 !dbg !105 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !106
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassTemporary.Main() #1 !dbg !107 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca [1 x i8], align 1, !dbg !108
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !108
// CHECK:STDOUT:   call void @_Z14make_move_onlyv.carbon_thunk.(ptr %_.var), !dbg !109
// CHECK:STDOUT:   call void @_Z14pass_move_only8MoveOnly.carbon_thunk.v(ptr %_.var), !dbg !110
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.5bf528c71c41a02a(ptr %_.var), !dbg !108
// CHECK:STDOUT:   ret void, !dbg !111
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_Z14pass_move_only8MoveOnly.carbon_thunk.v(ptr noundef %0) #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %struct.MoveOnly, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !124
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !124
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
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.5bf528c71c41a02a(ptr %t) #1 !dbg !116 {
// CHECK:STDOUT:   call void @_ZN8MoveOnlyD1Ev(ptr %t), !dbg !120
// CHECK:STDOUT:   ret void, !dbg !121
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Main() #1 !dbg !112 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z14make_move_onlyv.carbon_thunk.(ptr @_Cc.Main), !dbg !113
// CHECK:STDOUT:   ret void, !dbg !115
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!104}
// CHECK:STDOUT: !llvm.module.flags = !{!58, !59, !60, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!63}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !15 = !{null, !14}
// CHECK:STDOUT: !16 = !DISubroutineType(types: !15)
// CHECK:STDOUT: !17 = !{null}
// CHECK:STDOUT: !18 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !52 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !53 = !{!"omnipotent char", !52, i64 0}
// CHECK:STDOUT: !54 = !{!"any pointer", !53, i64 0}
// CHECK:STDOUT: !58 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !59 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !60 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !61 = !{!"int", !53, i64 0}
// CHECK:STDOUT: !62 = !{!"__libc_errno", !61, i64 0}
// CHECK:STDOUT: !63 = !{!62, !61, i64 0}
// CHECK:STDOUT: !103 = !DIFile(filename: "move_only.carbon", directory: "")
// CHECK:STDOUT: !104 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !103, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !105 = distinct !DISubprogram(name: "PassGlobalByMove", linkageName: "_CPassGlobalByMove.Main", scope: null, file: !103, line: 22, type: !18, spFlags: DISPFlagDefinition, unit: !104)
// CHECK:STDOUT: !106 = !DILocation(line: 22, column: 1, scope: !105)
// CHECK:STDOUT: !107 = distinct !DISubprogram(name: "PassTemporary", linkageName: "_CPassTemporary.Main", scope: null, file: !103, line: 27, type: !18, spFlags: DISPFlagDefinition, unit: !104)
// CHECK:STDOUT: !108 = !DILocation(line: 17, column: 29, scope: !107)
// CHECK:STDOUT: !109 = !DILocation(line: 28, column: 22, scope: !107)
// CHECK:STDOUT: !110 = !DILocation(line: 28, column: 3, scope: !107)
// CHECK:STDOUT: !111 = !DILocation(line: 27, column: 1, scope: !107)
// CHECK:STDOUT: !112 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !103, type: !18, spFlags: DISPFlagDefinition, unit: !104)
// CHECK:STDOUT: !113 = !DILocation(line: 20, column: 23, scope: !112)
// CHECK:STDOUT: !115 = !DILocation(line: 0, scope: !112)
// CHECK:STDOUT: !116 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.5bf528c71c41a02a", scope: null, file: !4, line: 30, type: !16, spFlags: DISPFlagDefinition, unit: !104, retainedNodes: !122)
// CHECK:STDOUT: !117 = !DILocalVariable(arg: 1, scope: !116, type: !14)
// CHECK:STDOUT: !120 = !DILocation(line: 32, column: 5, scope: !116)
// CHECK:STDOUT: !121 = !DILocation(line: 30, column: 1, scope: !116)
// CHECK:STDOUT: !122 = !{!117}
// CHECK:STDOUT: !123 = !{!"p1 _ZTS8MoveOnly", !54, i64 0}
// CHECK:STDOUT: !124 = !{!123, !123, i64 0}
// CHECK:STDOUT:
