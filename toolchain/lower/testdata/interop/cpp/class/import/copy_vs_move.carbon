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
Copyable make();
void pass(Copyable);
''';

var c: Cpp.Copyable = Cpp.make();

fn PassGlobal() {
  Cpp.pass(c);
}

fn PassTemporary() {
  Cpp.pass(Cpp.make());
}

fn PassValue(c: Cpp.Copyable) {
  Cpp.pass(c);
}

fn PassRef(ref c: Cpp.Copyable) {
  Cpp.pass(c);
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
CopyOrMove make();
void pass(CopyOrMove);
''';

var c: Cpp.CopyOrMove = Cpp.make();

fn PassGlobal() {
  Cpp.pass(c);
}

fn PassTemporary() {
  Cpp.pass(Cpp.make());
}

fn PassValue(c: Cpp.CopyOrMove) {
  Cpp.pass(c);
}

fn PassRef(ref c: Cpp.CopyOrMove) {
  Cpp.pass(c);
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
MoveOnly make();
void pass(MoveOnly);
''';

var c: Cpp.MoveOnly = Cpp.make();

fn PassGlobalByMove() {
  // TODO: This should be accepted.
  // Cpp.pass(Cpp.std.move(ref c));
}

fn PassTemporary() {
  Cpp.pass(Cpp.make());
}

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
// CHECK:STDOUT: define internal void @_Z4makev.carbon_thunk.(ptr noundef %return) #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !11
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !11
// CHECK:STDOUT:   call void @_Z4makev(ptr dead_on_unwind writable sret(%struct.Copyable) align 1 %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassGlobal.Main() #1 !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z4pass8Copyable.carbon_thunk._(ptr @_Cc.Main), !dbg !17
// CHECK:STDOUT:   ret void, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_Z4pass8Copyable.carbon_thunk._(ptr noundef %0) #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %struct.Copyable, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !11
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !11
// CHECK:STDOUT:   call void @_ZN8CopyableC1ERKS_(ptr noundef nonnull align 1 dereferenceable(1) %agg.tmp, ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   call void @_Z4pass8Copyable(ptr noundef %agg.tmp)
// CHECK:STDOUT:   call void @_ZN8CopyableD1Ev(ptr noundef nonnull align 1 dereferenceable(1) %agg.tmp) #1
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassTemporary.Main() #1 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc21_21.1.temp = alloca [1 x i8], align 1, !dbg !20
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc21_21.1.temp), !dbg !20
// CHECK:STDOUT:   call void @_Z4makev.carbon_thunk.(ptr %.loc21_21.1.temp), !dbg !20
// CHECK:STDOUT:   call void @_Z4pass8Copyable.carbon_thunk._(ptr %.loc21_21.1.temp), !dbg !21
// CHECK:STDOUT:   call void @_ZN8CopyableD1Ev(ptr %.loc21_21.1.temp), !dbg !20
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: declare void @_ZN8CopyableD1Ev(ptr noundef nonnull align 1 dereferenceable(1)) unnamed_addr #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassValue.Main(ptr %c) #1 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z4pass8Copyable.carbon_thunk._(ptr %c), !dbg !29
// CHECK:STDOUT:   ret void, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassRef.Main(ptr %c) #1 !dbg !31 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z4pass8Copyable.carbon_thunk._(ptr %c), !dbg !34
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Main() #1 !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z4makev.carbon_thunk.(ptr @_Cc.Main), !dbg !37
// CHECK:STDOUT:   ret void, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z4makev(ptr dead_on_unwind writable sret(%struct.Copyable) align 1) #4
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z4pass8Copyable(ptr noundef) #4
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN8CopyableC1ERKS_(ptr noundef nonnull align 1 dereferenceable(1), ptr noundef nonnull align 1 dereferenceable(1)) unnamed_addr #4
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { nounwind }
// CHECK:STDOUT: attributes #2 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #4 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
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
// CHECK:STDOUT: !6 = !DIFile(filename: "copyable.carbon", directory: "")
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = !{!12, !12, i64 0}
// CHECK:STDOUT: !12 = !{!"p1 _ZTS8Copyable", !13, i64 0}
// CHECK:STDOUT: !13 = !{!"any pointer", !9, i64 0}
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "PassGlobal", linkageName: "_CPassGlobal.Main", scope: null, file: !6, line: 16, type: !15, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !15 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !16 = !{null}
// CHECK:STDOUT: !17 = !DILocation(line: 17, column: 3, scope: !14)
// CHECK:STDOUT: !18 = !DILocation(line: 16, column: 1, scope: !14)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "PassTemporary", linkageName: "_CPassTemporary.Main", scope: null, file: !6, line: 20, type: !15, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !20 = !DILocation(line: 21, column: 12, scope: !19)
// CHECK:STDOUT: !21 = !DILocation(line: 21, column: 3, scope: !19)
// CHECK:STDOUT: !22 = !DILocation(line: 20, column: 1, scope: !19)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "PassValue", linkageName: "_CPassValue.Main", scope: null, file: !6, line: 24, type: !24, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !27)
// CHECK:STDOUT: !24 = !DISubroutineType(types: !25)
// CHECK:STDOUT: !25 = !{null, !26}
// CHECK:STDOUT: !26 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !27 = !{!28}
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 1, scope: !23, type: !26)
// CHECK:STDOUT: !29 = !DILocation(line: 25, column: 3, scope: !23)
// CHECK:STDOUT: !30 = !DILocation(line: 24, column: 1, scope: !23)
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "PassRef", linkageName: "_CPassRef.Main", scope: null, file: !6, line: 28, type: !24, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !32)
// CHECK:STDOUT: !32 = !{!33}
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 1, scope: !31, type: !26)
// CHECK:STDOUT: !34 = !DILocation(line: 29, column: 3, scope: !31)
// CHECK:STDOUT: !35 = !DILocation(line: 28, column: 1, scope: !31)
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !6, type: !15, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !37 = !DILocation(line: 14, column: 23, scope: !36)
// CHECK:STDOUT: !38 = !DILocation(line: 0, scope: !36)
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
// CHECK:STDOUT: define internal void @_Z4makev.carbon_thunk.(ptr noundef %return) #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !11
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !11
// CHECK:STDOUT:   call void @_Z4makev(ptr dead_on_unwind writable sret(%struct.CopyOrMove) align 1 %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassGlobal.Main() #1 !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z4pass10CopyOrMove.carbon_thunk._(ptr @_Cc.Main), !dbg !17
// CHECK:STDOUT:   ret void, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_Z4pass10CopyOrMove.carbon_thunk._(ptr noundef %0) #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %struct.CopyOrMove, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !11
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !11
// CHECK:STDOUT:   call void @_ZN10CopyOrMoveC1ERKS_(ptr noundef nonnull align 1 dereferenceable(1) %agg.tmp, ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   call void @_Z4pass10CopyOrMove(ptr noundef %agg.tmp)
// CHECK:STDOUT:   call void @_ZN10CopyOrMoveD1Ev(ptr noundef nonnull align 1 dereferenceable(1) %agg.tmp) #1
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassTemporary.Main() #1 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca [1 x i8], align 1, !dbg !20
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !20
// CHECK:STDOUT:   call void @_Z4makev.carbon_thunk.(ptr %_.var), !dbg !21
// CHECK:STDOUT:   call void @_Z4pass10CopyOrMove.carbon_thunk.v(ptr %_.var), !dbg !22
// CHECK:STDOUT:   call void @_ZN10CopyOrMoveD1Ev(ptr %_.var), !dbg !20
// CHECK:STDOUT:   ret void, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_Z4pass10CopyOrMove.carbon_thunk.v(ptr noundef %0) #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %struct.CopyOrMove, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !11
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !11
// CHECK:STDOUT:   call void @_ZN10CopyOrMoveC1EOS_(ptr noundef nonnull align 1 dereferenceable(1) %agg.tmp, ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   call void @_Z4pass10CopyOrMove(ptr noundef %agg.tmp)
// CHECK:STDOUT:   call void @_ZN10CopyOrMoveD1Ev(ptr noundef nonnull align 1 dereferenceable(1) %agg.tmp) #1
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: declare void @_ZN10CopyOrMoveD1Ev(ptr noundef nonnull align 1 dereferenceable(1)) unnamed_addr #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassValue.Main(ptr %c) #1 !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z4pass10CopyOrMove.carbon_thunk._(ptr %c), !dbg !30
// CHECK:STDOUT:   ret void, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassRef.Main(ptr %c) #1 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z4pass10CopyOrMove.carbon_thunk._(ptr %c), !dbg !35
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define internal void @_C__global_init.Main() #1 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z4makev.carbon_thunk.(ptr @_Cc.Main), !dbg !38
// CHECK:STDOUT:   ret void, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z4makev(ptr dead_on_unwind writable sret(%struct.CopyOrMove) align 1) #4
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z4pass10CopyOrMove(ptr noundef) #4
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN10CopyOrMoveC1ERKS_(ptr noundef nonnull align 1 dereferenceable(1), ptr noundef nonnull align 1 dereferenceable(1)) unnamed_addr #4
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN10CopyOrMoveC1EOS_(ptr noundef nonnull align 1 dereferenceable(1), ptr noundef nonnull align 1 dereferenceable(1)) unnamed_addr #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_ZN10CopyOrMoveD1Ev, { 1, 2, 0 }
// CHECK:STDOUT: uselistorder ptr @_Z4pass10CopyOrMove, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #1 = { nounwind }
// CHECK:STDOUT: attributes #2 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #4 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
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
// CHECK:STDOUT: !6 = !DIFile(filename: "copy_or_move.carbon", directory: "")
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = !{!12, !12, i64 0}
// CHECK:STDOUT: !12 = !{!"p1 _ZTS10CopyOrMove", !13, i64 0}
// CHECK:STDOUT: !13 = !{!"any pointer", !9, i64 0}
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "PassGlobal", linkageName: "_CPassGlobal.Main", scope: null, file: !6, line: 18, type: !15, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !15 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !16 = !{null}
// CHECK:STDOUT: !17 = !DILocation(line: 19, column: 3, scope: !14)
// CHECK:STDOUT: !18 = !DILocation(line: 18, column: 1, scope: !14)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "PassTemporary", linkageName: "_CPassTemporary.Main", scope: null, file: !6, line: 22, type: !15, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !20 = !DILocation(line: 13, column: 21, scope: !19)
// CHECK:STDOUT: !21 = !DILocation(line: 23, column: 12, scope: !19)
// CHECK:STDOUT: !22 = !DILocation(line: 23, column: 3, scope: !19)
// CHECK:STDOUT: !23 = !DILocation(line: 22, column: 1, scope: !19)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "PassValue", linkageName: "_CPassValue.Main", scope: null, file: !6, line: 26, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !28)
// CHECK:STDOUT: !25 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !26 = !{null, !27}
// CHECK:STDOUT: !27 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !28 = !{!29}
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 1, scope: !24, type: !27)
// CHECK:STDOUT: !30 = !DILocation(line: 27, column: 3, scope: !24)
// CHECK:STDOUT: !31 = !DILocation(line: 26, column: 1, scope: !24)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "PassRef", linkageName: "_CPassRef.Main", scope: null, file: !6, line: 30, type: !25, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !33)
// CHECK:STDOUT: !33 = !{!34}
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 1, scope: !32, type: !27)
// CHECK:STDOUT: !35 = !DILocation(line: 31, column: 3, scope: !32)
// CHECK:STDOUT: !36 = !DILocation(line: 30, column: 1, scope: !32)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !6, type: !15, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !38 = !DILocation(line: 16, column: 25, scope: !37)
// CHECK:STDOUT: !39 = !DILocation(line: 0, scope: !37)
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
// CHECK:STDOUT: define internal void @_Z4makev.carbon_thunk.(ptr noundef %return) #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !11
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !11
// CHECK:STDOUT:   call void @_Z4makev(ptr dead_on_unwind writable sret(%struct.MoveOnly) align 1 %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassGlobalByMove.Main() #1 !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassTemporary.Main() #1 !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca [1 x i8], align 1, !dbg !19
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !19
// CHECK:STDOUT:   call void @_Z4makev.carbon_thunk.(ptr %_.var), !dbg !20
// CHECK:STDOUT:   call void @_Z4pass8MoveOnly.carbon_thunk.v(ptr %_.var), !dbg !21
// CHECK:STDOUT:   call void @_ZN8MoveOnlyD1Ev(ptr %_.var), !dbg !19
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_Z4pass8MoveOnly.carbon_thunk.v(ptr noundef %0) #0 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %struct.MoveOnly, align 1
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !11
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !11
// CHECK:STDOUT:   call void @_ZN8MoveOnlyC1EOS_(ptr noundef nonnull align 1 dereferenceable(1) %agg.tmp, ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   call void @_Z4pass8MoveOnly(ptr noundef %agg.tmp)
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
// CHECK:STDOUT: define internal void @_C__global_init.Main() #1 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z4makev.carbon_thunk.(ptr @_Cc.Main), !dbg !24
// CHECK:STDOUT:   ret void, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z4makev(ptr dead_on_unwind writable sret(%struct.MoveOnly) align 1) #4
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z4pass8MoveOnly(ptr noundef) #4
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
// CHECK:STDOUT: !6 = !DIFile(filename: "move_only.carbon", directory: "")
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = !{!12, !12, i64 0}
// CHECK:STDOUT: !12 = !{!"p1 _ZTS8MoveOnly", !13, i64 0}
// CHECK:STDOUT: !13 = !{!"any pointer", !9, i64 0}
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "PassGlobalByMove", linkageName: "_CPassGlobalByMove.Main", scope: null, file: !6, line: 22, type: !15, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !15 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !16 = !{null}
// CHECK:STDOUT: !17 = !DILocation(line: 22, column: 1, scope: !14)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "PassTemporary", linkageName: "_CPassTemporary.Main", scope: null, file: !6, line: 27, type: !15, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !19 = !DILocation(line: 17, column: 19, scope: !18)
// CHECK:STDOUT: !20 = !DILocation(line: 28, column: 12, scope: !18)
// CHECK:STDOUT: !21 = !DILocation(line: 28, column: 3, scope: !18)
// CHECK:STDOUT: !22 = !DILocation(line: 27, column: 1, scope: !18)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "__global_init", linkageName: "_C__global_init.Main", scope: null, file: !6, type: !15, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !24 = !DILocation(line: 20, column: 23, scope: !23)
// CHECK:STDOUT: !25 = !DILocation(line: 0, scope: !23)
