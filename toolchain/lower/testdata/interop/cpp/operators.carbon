// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/operators.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/operators.carbon

// --- eq_with.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
struct A {
  int value;

  friend auto operator==(A x, A y) -> bool {
    return x.value == y.value;
  }

  friend auto operator!=(A x, A y) -> bool {
    return !(x == y);
  }
};
''';

fn CheckEqWith[U:! type, T:! Core.EqWith(U)](x: T, y: U) {
  x == y;
  x != y;
}

fn Driver(x: Cpp.A, y: Cpp.A) {
  CheckEqWith(x, y);
}

// --- ordered_with.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
struct A {
  int value;

  friend auto operator<(A x, A y) -> bool {
    return x.value < y.value;
  }

  friend auto operator>(A x, A y) -> bool {
    return y < x;
  }

  friend auto operator<=(A x, A y) -> bool {
    return !(y < x);
  }

  friend auto operator>=(A x, A y) -> bool {
    return !(x < y);
  }
};
''';

fn CheckOrderedWith[U:! type, T:! Core.OrderedWith(U)](x: T, y: U) {
  x < y;
  x > y;
  x <= y;
  x >= y;
}

fn Driver(x: Cpp.A, y: Cpp.A) {
  CheckOrderedWith(x, y);
}

// CHECK:STDOUT: ; ModuleID = 'eq_with.carbon'
// CHECK:STDOUT: source_filename = "eq_with.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %struct.A = type { i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: $_Zeq1AS_ = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_Zne1AS_ = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CDriver.Main(ptr %x, ptr %y) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CCheckEqWith.Main.bdef5a2ed02e32a1(ptr %x, ptr %y), !dbg !18
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Zeq1AS_.carbon_thunk.__(ptr noundef %x, ptr noundef %y, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.addr = alloca ptr, align 8
// CHECK:STDOUT:   %y.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %struct.A, align 4
// CHECK:STDOUT:   %agg.tmp1 = alloca %struct.A, align 4
// CHECK:STDOUT:   store ptr %x, ptr %x.addr, align 8, !tbaa !20
// CHECK:STDOUT:   store ptr %y, ptr %y.addr, align 8, !tbaa !20
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !23
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !23
// CHECK:STDOUT:   %1 = load ptr, ptr %x.addr, align 8, !tbaa !20
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %1, i64 4, i1 false), !tbaa.struct !25
// CHECK:STDOUT:   %2 = load ptr, ptr %y.addr, align 8, !tbaa !20
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp1, ptr align 4 %2, i64 4, i1 false), !tbaa.struct !25
// CHECK:STDOUT:   %coerce.dive = getelementptr inbounds nuw %struct.A, ptr %agg.tmp, i32 0, i32 0
// CHECK:STDOUT:   %3 = load i32, ptr %coerce.dive, align 4
// CHECK:STDOUT:   %coerce.dive2 = getelementptr inbounds nuw %struct.A, ptr %agg.tmp1, i32 0, i32 0
// CHECK:STDOUT:   %4 = load i32, ptr %coerce.dive2, align 4
// CHECK:STDOUT:   %call = call noundef zeroext i1 @_Zeq1AS_(i32 %3, i32 %4)
// CHECK:STDOUT:   %storedv = zext i1 %call to i8
// CHECK:STDOUT:   store i8 %storedv, ptr %0, align 1, !tbaa !26
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Zne1AS_.carbon_thunk.__(ptr noundef %x, ptr noundef %y, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.addr = alloca ptr, align 8
// CHECK:STDOUT:   %y.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %struct.A, align 4
// CHECK:STDOUT:   %agg.tmp1 = alloca %struct.A, align 4
// CHECK:STDOUT:   store ptr %x, ptr %x.addr, align 8, !tbaa !20
// CHECK:STDOUT:   store ptr %y, ptr %y.addr, align 8, !tbaa !20
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !23
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !23
// CHECK:STDOUT:   %1 = load ptr, ptr %x.addr, align 8, !tbaa !20
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %1, i64 4, i1 false), !tbaa.struct !25
// CHECK:STDOUT:   %2 = load ptr, ptr %y.addr, align 8, !tbaa !20
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp1, ptr align 4 %2, i64 4, i1 false), !tbaa.struct !25
// CHECK:STDOUT:   %coerce.dive = getelementptr inbounds nuw %struct.A, ptr %agg.tmp, i32 0, i32 0
// CHECK:STDOUT:   %3 = load i32, ptr %coerce.dive, align 4
// CHECK:STDOUT:   %coerce.dive2 = getelementptr inbounds nuw %struct.A, ptr %agg.tmp1, i32 0, i32 0
// CHECK:STDOUT:   %4 = load i32, ptr %coerce.dive2, align 4
// CHECK:STDOUT:   %call = call noundef zeroext i1 @_Zne1AS_(i32 %3, i32 %4)
// CHECK:STDOUT:   %storedv = zext i1 %call to i8
// CHECK:STDOUT:   store i8 %storedv, ptr %0, align 1, !tbaa !26
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define i1 @"_CEqual:thunk:EqWith.ffb401eeee8c4872.Core:Cpp"(ptr %self, ptr %other) #2 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.6.temp = alloca i1, align 1, !dbg !34
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.6.temp), !dbg !34
// CHECK:STDOUT:   call void @_Zeq1AS_.carbon_thunk.__(ptr %self, ptr %other, ptr %.6.temp), !dbg !34
// CHECK:STDOUT:   %0 = load i1, ptr %.6.temp, align 1, !dbg !34
// CHECK:STDOUT:   ret i1 %0, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define i1 @"_CNotEqual:thunk:EqWith.ffb401eeee8c4872.Core:Cpp"(ptr %self, ptr %other) #2 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.6.temp = alloca i1, align 1, !dbg !39
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.6.temp), !dbg !39
// CHECK:STDOUT:   call void @_Zne1AS_.carbon_thunk.__(ptr %self, ptr %other, ptr %.6.temp), !dbg !39
// CHECK:STDOUT:   %0 = load i1, ptr %.6.temp, align 1, !dbg !39
// CHECK:STDOUT:   ret i1 %0, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCheckEqWith.Main.bdef5a2ed02e32a1(ptr %x, ptr %y) #0 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.call = call i1 @"_CEqual:thunk:EqWith.ffb401eeee8c4872.Core:Cpp"(ptr %x, ptr %y), !dbg !44
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.call = call i1 @"_CNotEqual:thunk:EqWith.ffb401eeee8c4872.Core:Cpp"(ptr %x, ptr %y), !dbg !45
// CHECK:STDOUT:   ret void, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local noundef zeroext i1 @_Zeq1AS_(i32 %x.coerce, i32 %y.coerce) #4 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x = alloca %struct.A, align 4
// CHECK:STDOUT:   %y = alloca %struct.A, align 4
// CHECK:STDOUT:   %coerce.dive = getelementptr inbounds nuw %struct.A, ptr %x, i32 0, i32 0
// CHECK:STDOUT:   store i32 %x.coerce, ptr %coerce.dive, align 4
// CHECK:STDOUT:   %coerce.dive1 = getelementptr inbounds nuw %struct.A, ptr %y, i32 0, i32 0
// CHECK:STDOUT:   store i32 %y.coerce, ptr %coerce.dive1, align 4
// CHECK:STDOUT:   %value = getelementptr inbounds nuw %struct.A, ptr %x, i32 0, i32 0
// CHECK:STDOUT:   %0 = load i32, ptr %value, align 4, !tbaa !47
// CHECK:STDOUT:   %value2 = getelementptr inbounds nuw %struct.A, ptr %y, i32 0, i32 0
// CHECK:STDOUT:   %1 = load i32, ptr %value2, align 4, !tbaa !47
// CHECK:STDOUT:   %cmp = icmp eq i32 %0, %1
// CHECK:STDOUT:   ret i1 %cmp
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local noundef zeroext i1 @_Zne1AS_(i32 %x.coerce, i32 %y.coerce) #4 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x = alloca %struct.A, align 4
// CHECK:STDOUT:   %y = alloca %struct.A, align 4
// CHECK:STDOUT:   %agg.tmp = alloca %struct.A, align 4
// CHECK:STDOUT:   %agg.tmp2 = alloca %struct.A, align 4
// CHECK:STDOUT:   %coerce.dive = getelementptr inbounds nuw %struct.A, ptr %x, i32 0, i32 0
// CHECK:STDOUT:   store i32 %x.coerce, ptr %coerce.dive, align 4
// CHECK:STDOUT:   %coerce.dive1 = getelementptr inbounds nuw %struct.A, ptr %y, i32 0, i32 0
// CHECK:STDOUT:   store i32 %y.coerce, ptr %coerce.dive1, align 4
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %x, i64 4, i1 false), !tbaa.struct !25
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp2, ptr align 4 %y, i64 4, i1 false), !tbaa.struct !25
// CHECK:STDOUT:   %coerce.dive3 = getelementptr inbounds nuw %struct.A, ptr %agg.tmp, i32 0, i32 0
// CHECK:STDOUT:   %0 = load i32, ptr %coerce.dive3, align 4
// CHECK:STDOUT:   %coerce.dive4 = getelementptr inbounds nuw %struct.A, ptr %agg.tmp2, i32 0, i32 0
// CHECK:STDOUT:   %1 = load i32, ptr %coerce.dive4, align 4
// CHECK:STDOUT:   %call = call noundef zeroext i1 @_Zeq1AS_(i32 %0, i32 %1)
// CHECK:STDOUT:   %lnot = xor i1 %call, true
// CHECK:STDOUT:   ret i1 %lnot
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 0, 1, 5, 4, 3, 2 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #4 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
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
// CHECK:STDOUT: !6 = !DIFile(filename: "eq_with.carbon", directory: "")
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "Driver", linkageName: "_CDriver.Main", scope: null, file: !6, line: 23, type: !12, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !15)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{null, !14, !14}
// CHECK:STDOUT: !14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !15 = !{!16, !17}
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 1, scope: !11, type: !14)
// CHECK:STDOUT: !17 = !DILocalVariable(arg: 2, scope: !11, type: !14)
// CHECK:STDOUT: !18 = !DILocation(line: 24, column: 3, scope: !11)
// CHECK:STDOUT: !19 = !DILocation(line: 23, column: 1, scope: !11)
// CHECK:STDOUT: !20 = !{!21, !21, i64 0}
// CHECK:STDOUT: !21 = !{!"p1 _ZTS1A", !22, i64 0}
// CHECK:STDOUT: !22 = !{!"any pointer", !9, i64 0}
// CHECK:STDOUT: !23 = !{!24, !24, i64 0}
// CHECK:STDOUT: !24 = !{!"p1 bool", !22, i64 0}
// CHECK:STDOUT: !25 = !{i64 0, i64 4, !7}
// CHECK:STDOUT: !26 = !{!27, !27, i64 0}
// CHECK:STDOUT: !27 = !{!"bool", !9, i64 0}
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "Equal", linkageName: "_CEqual:thunk:EqWith.ffb401eeee8c4872.Core:Cpp", scope: null, file: !6, line: 8, type: !29, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !31)
// CHECK:STDOUT: !29 = !DISubroutineType(types: !30)
// CHECK:STDOUT: !30 = !{!14, !14, !14}
// CHECK:STDOUT: !31 = !{!32, !33}
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !28, type: !14)
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 2, scope: !28, type: !14)
// CHECK:STDOUT: !34 = !DILocation(line: 8, column: 15, scope: !28)
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "NotEqual", linkageName: "_CNotEqual:thunk:EqWith.ffb401eeee8c4872.Core:Cpp", scope: null, file: !6, line: 12, type: !29, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !36)
// CHECK:STDOUT: !36 = !{!37, !38}
// CHECK:STDOUT: !37 = !DILocalVariable(arg: 1, scope: !35, type: !14)
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 2, scope: !35, type: !14)
// CHECK:STDOUT: !39 = !DILocation(line: 12, column: 15, scope: !35)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "CheckEqWith", linkageName: "_CCheckEqWith.Main.bdef5a2ed02e32a1", scope: null, file: !6, line: 18, type: !12, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !41)
// CHECK:STDOUT: !41 = !{!42, !43}
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 1, scope: !40, type: !14)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 2, scope: !40, type: !14)
// CHECK:STDOUT: !44 = !DILocation(line: 19, column: 3, scope: !40)
// CHECK:STDOUT: !45 = !DILocation(line: 20, column: 3, scope: !40)
// CHECK:STDOUT: !46 = !DILocation(line: 18, column: 1, scope: !40)
// CHECK:STDOUT: !47 = !{!48, !8, i64 0}
// CHECK:STDOUT: !48 = !{!"_ZTS1A", !8, i64 0}
// CHECK:STDOUT: ; ModuleID = 'ordered_with.carbon'
// CHECK:STDOUT: source_filename = "ordered_with.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %struct.A = type { i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: $_Zlt1AS_ = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_Zle1AS_ = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_Zgt1AS_ = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_Zge1AS_ = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CDriver.Main(ptr %x, ptr %y) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CCheckOrderedWith.Main.c49adfa90bd18905(ptr %x, ptr %y), !dbg !18
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Zlt1AS_.carbon_thunk.__(ptr noundef %x, ptr noundef %y, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.addr = alloca ptr, align 8
// CHECK:STDOUT:   %y.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %struct.A, align 4
// CHECK:STDOUT:   %agg.tmp1 = alloca %struct.A, align 4
// CHECK:STDOUT:   store ptr %x, ptr %x.addr, align 8, !tbaa !20
// CHECK:STDOUT:   store ptr %y, ptr %y.addr, align 8, !tbaa !20
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !23
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !23
// CHECK:STDOUT:   %1 = load ptr, ptr %x.addr, align 8, !tbaa !20
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %1, i64 4, i1 false), !tbaa.struct !25
// CHECK:STDOUT:   %2 = load ptr, ptr %y.addr, align 8, !tbaa !20
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp1, ptr align 4 %2, i64 4, i1 false), !tbaa.struct !25
// CHECK:STDOUT:   %coerce.dive = getelementptr inbounds nuw %struct.A, ptr %agg.tmp, i32 0, i32 0
// CHECK:STDOUT:   %3 = load i32, ptr %coerce.dive, align 4
// CHECK:STDOUT:   %coerce.dive2 = getelementptr inbounds nuw %struct.A, ptr %agg.tmp1, i32 0, i32 0
// CHECK:STDOUT:   %4 = load i32, ptr %coerce.dive2, align 4
// CHECK:STDOUT:   %call = call noundef zeroext i1 @_Zlt1AS_(i32 %3, i32 %4)
// CHECK:STDOUT:   %storedv = zext i1 %call to i8
// CHECK:STDOUT:   store i8 %storedv, ptr %0, align 1, !tbaa !26
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Zle1AS_.carbon_thunk.__(ptr noundef %x, ptr noundef %y, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.addr = alloca ptr, align 8
// CHECK:STDOUT:   %y.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %struct.A, align 4
// CHECK:STDOUT:   %agg.tmp1 = alloca %struct.A, align 4
// CHECK:STDOUT:   store ptr %x, ptr %x.addr, align 8, !tbaa !20
// CHECK:STDOUT:   store ptr %y, ptr %y.addr, align 8, !tbaa !20
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !23
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !23
// CHECK:STDOUT:   %1 = load ptr, ptr %x.addr, align 8, !tbaa !20
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %1, i64 4, i1 false), !tbaa.struct !25
// CHECK:STDOUT:   %2 = load ptr, ptr %y.addr, align 8, !tbaa !20
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp1, ptr align 4 %2, i64 4, i1 false), !tbaa.struct !25
// CHECK:STDOUT:   %coerce.dive = getelementptr inbounds nuw %struct.A, ptr %agg.tmp, i32 0, i32 0
// CHECK:STDOUT:   %3 = load i32, ptr %coerce.dive, align 4
// CHECK:STDOUT:   %coerce.dive2 = getelementptr inbounds nuw %struct.A, ptr %agg.tmp1, i32 0, i32 0
// CHECK:STDOUT:   %4 = load i32, ptr %coerce.dive2, align 4
// CHECK:STDOUT:   %call = call noundef zeroext i1 @_Zle1AS_(i32 %3, i32 %4)
// CHECK:STDOUT:   %storedv = zext i1 %call to i8
// CHECK:STDOUT:   store i8 %storedv, ptr %0, align 1, !tbaa !26
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Zgt1AS_.carbon_thunk.__(ptr noundef %x, ptr noundef %y, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.addr = alloca ptr, align 8
// CHECK:STDOUT:   %y.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %struct.A, align 4
// CHECK:STDOUT:   %agg.tmp1 = alloca %struct.A, align 4
// CHECK:STDOUT:   store ptr %x, ptr %x.addr, align 8, !tbaa !20
// CHECK:STDOUT:   store ptr %y, ptr %y.addr, align 8, !tbaa !20
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !23
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !23
// CHECK:STDOUT:   %1 = load ptr, ptr %x.addr, align 8, !tbaa !20
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %1, i64 4, i1 false), !tbaa.struct !25
// CHECK:STDOUT:   %2 = load ptr, ptr %y.addr, align 8, !tbaa !20
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp1, ptr align 4 %2, i64 4, i1 false), !tbaa.struct !25
// CHECK:STDOUT:   %coerce.dive = getelementptr inbounds nuw %struct.A, ptr %agg.tmp, i32 0, i32 0
// CHECK:STDOUT:   %3 = load i32, ptr %coerce.dive, align 4
// CHECK:STDOUT:   %coerce.dive2 = getelementptr inbounds nuw %struct.A, ptr %agg.tmp1, i32 0, i32 0
// CHECK:STDOUT:   %4 = load i32, ptr %coerce.dive2, align 4
// CHECK:STDOUT:   %call = call noundef zeroext i1 @_Zgt1AS_(i32 %3, i32 %4)
// CHECK:STDOUT:   %storedv = zext i1 %call to i8
// CHECK:STDOUT:   store i8 %storedv, ptr %0, align 1, !tbaa !26
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Zge1AS_.carbon_thunk.__(ptr noundef %x, ptr noundef %y, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.addr = alloca ptr, align 8
// CHECK:STDOUT:   %y.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %struct.A, align 4
// CHECK:STDOUT:   %agg.tmp1 = alloca %struct.A, align 4
// CHECK:STDOUT:   store ptr %x, ptr %x.addr, align 8, !tbaa !20
// CHECK:STDOUT:   store ptr %y, ptr %y.addr, align 8, !tbaa !20
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !23
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !23
// CHECK:STDOUT:   %1 = load ptr, ptr %x.addr, align 8, !tbaa !20
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %1, i64 4, i1 false), !tbaa.struct !25
// CHECK:STDOUT:   %2 = load ptr, ptr %y.addr, align 8, !tbaa !20
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp1, ptr align 4 %2, i64 4, i1 false), !tbaa.struct !25
// CHECK:STDOUT:   %coerce.dive = getelementptr inbounds nuw %struct.A, ptr %agg.tmp, i32 0, i32 0
// CHECK:STDOUT:   %3 = load i32, ptr %coerce.dive, align 4
// CHECK:STDOUT:   %coerce.dive2 = getelementptr inbounds nuw %struct.A, ptr %agg.tmp1, i32 0, i32 0
// CHECK:STDOUT:   %4 = load i32, ptr %coerce.dive2, align 4
// CHECK:STDOUT:   %call = call noundef zeroext i1 @_Zge1AS_(i32 %3, i32 %4)
// CHECK:STDOUT:   %storedv = zext i1 %call to i8
// CHECK:STDOUT:   store i8 %storedv, ptr %0, align 1, !tbaa !26
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define i1 @"_CLess:thunk:OrderedWith.92c0df05bd55f97a.Core:Cpp"(ptr %self, ptr %other) #2 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.6.temp = alloca i1, align 1, !dbg !34
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.6.temp), !dbg !34
// CHECK:STDOUT:   call void @_Zlt1AS_.carbon_thunk.__(ptr %self, ptr %other, ptr %.6.temp), !dbg !34
// CHECK:STDOUT:   %0 = load i1, ptr %.6.temp, align 1, !dbg !34
// CHECK:STDOUT:   ret i1 %0, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define i1 @"_CLessOrEquivalent:thunk:OrderedWith.92c0df05bd55f97a.Core:Cpp"(ptr %self, ptr %other) #2 !dbg !35 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.6.temp = alloca i1, align 1, !dbg !39
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.6.temp), !dbg !39
// CHECK:STDOUT:   call void @_Zle1AS_.carbon_thunk.__(ptr %self, ptr %other, ptr %.6.temp), !dbg !39
// CHECK:STDOUT:   %0 = load i1, ptr %.6.temp, align 1, !dbg !39
// CHECK:STDOUT:   ret i1 %0, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define i1 @"_CGreater:thunk:OrderedWith.92c0df05bd55f97a.Core:Cpp"(ptr %self, ptr %other) #2 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.6.temp = alloca i1, align 1, !dbg !44
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.6.temp), !dbg !44
// CHECK:STDOUT:   call void @_Zgt1AS_.carbon_thunk.__(ptr %self, ptr %other, ptr %.6.temp), !dbg !44
// CHECK:STDOUT:   %0 = load i1, ptr %.6.temp, align 1, !dbg !44
// CHECK:STDOUT:   ret i1 %0, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define i1 @"_CGreaterOrEquivalent:thunk:OrderedWith.92c0df05bd55f97a.Core:Cpp"(ptr %self, ptr %other) #2 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.6.temp = alloca i1, align 1, !dbg !49
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.6.temp), !dbg !49
// CHECK:STDOUT:   call void @_Zge1AS_.carbon_thunk.__(ptr %self, ptr %other, ptr %.6.temp), !dbg !49
// CHECK:STDOUT:   %0 = load i1, ptr %.6.temp, align 1, !dbg !49
// CHECK:STDOUT:   ret i1 %0, !dbg !49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCheckOrderedWith.Main.c49adfa90bd18905(ptr %x, ptr %y) #0 !dbg !50 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.call = call i1 @"_CLess:thunk:OrderedWith.92c0df05bd55f97a.Core:Cpp"(ptr %x, ptr %y), !dbg !54
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.call = call i1 @"_CGreater:thunk:OrderedWith.92c0df05bd55f97a.Core:Cpp"(ptr %x, ptr %y), !dbg !55
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.call = call i1 @"_CLessOrEquivalent:thunk:OrderedWith.92c0df05bd55f97a.Core:Cpp"(ptr %x, ptr %y), !dbg !56
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.call = call i1 @"_CGreaterOrEquivalent:thunk:OrderedWith.92c0df05bd55f97a.Core:Cpp"(ptr %x, ptr %y), !dbg !57
// CHECK:STDOUT:   ret void, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local noundef zeroext i1 @_Zlt1AS_(i32 %x.coerce, i32 %y.coerce) #4 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x = alloca %struct.A, align 4
// CHECK:STDOUT:   %y = alloca %struct.A, align 4
// CHECK:STDOUT:   %coerce.dive = getelementptr inbounds nuw %struct.A, ptr %x, i32 0, i32 0
// CHECK:STDOUT:   store i32 %x.coerce, ptr %coerce.dive, align 4
// CHECK:STDOUT:   %coerce.dive1 = getelementptr inbounds nuw %struct.A, ptr %y, i32 0, i32 0
// CHECK:STDOUT:   store i32 %y.coerce, ptr %coerce.dive1, align 4
// CHECK:STDOUT:   %value = getelementptr inbounds nuw %struct.A, ptr %x, i32 0, i32 0
// CHECK:STDOUT:   %0 = load i32, ptr %value, align 4, !tbaa !59
// CHECK:STDOUT:   %value2 = getelementptr inbounds nuw %struct.A, ptr %y, i32 0, i32 0
// CHECK:STDOUT:   %1 = load i32, ptr %value2, align 4, !tbaa !59
// CHECK:STDOUT:   %cmp = icmp slt i32 %0, %1
// CHECK:STDOUT:   ret i1 %cmp
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local noundef zeroext i1 @_Zle1AS_(i32 %x.coerce, i32 %y.coerce) #4 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x = alloca %struct.A, align 4
// CHECK:STDOUT:   %y = alloca %struct.A, align 4
// CHECK:STDOUT:   %agg.tmp = alloca %struct.A, align 4
// CHECK:STDOUT:   %agg.tmp2 = alloca %struct.A, align 4
// CHECK:STDOUT:   %coerce.dive = getelementptr inbounds nuw %struct.A, ptr %x, i32 0, i32 0
// CHECK:STDOUT:   store i32 %x.coerce, ptr %coerce.dive, align 4
// CHECK:STDOUT:   %coerce.dive1 = getelementptr inbounds nuw %struct.A, ptr %y, i32 0, i32 0
// CHECK:STDOUT:   store i32 %y.coerce, ptr %coerce.dive1, align 4
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %y, i64 4, i1 false), !tbaa.struct !25
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp2, ptr align 4 %x, i64 4, i1 false), !tbaa.struct !25
// CHECK:STDOUT:   %coerce.dive3 = getelementptr inbounds nuw %struct.A, ptr %agg.tmp, i32 0, i32 0
// CHECK:STDOUT:   %0 = load i32, ptr %coerce.dive3, align 4
// CHECK:STDOUT:   %coerce.dive4 = getelementptr inbounds nuw %struct.A, ptr %agg.tmp2, i32 0, i32 0
// CHECK:STDOUT:   %1 = load i32, ptr %coerce.dive4, align 4
// CHECK:STDOUT:   %call = call noundef zeroext i1 @_Zlt1AS_(i32 %0, i32 %1)
// CHECK:STDOUT:   %lnot = xor i1 %call, true
// CHECK:STDOUT:   ret i1 %lnot
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local noundef zeroext i1 @_Zgt1AS_(i32 %x.coerce, i32 %y.coerce) #4 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x = alloca %struct.A, align 4
// CHECK:STDOUT:   %y = alloca %struct.A, align 4
// CHECK:STDOUT:   %agg.tmp = alloca %struct.A, align 4
// CHECK:STDOUT:   %agg.tmp2 = alloca %struct.A, align 4
// CHECK:STDOUT:   %coerce.dive = getelementptr inbounds nuw %struct.A, ptr %x, i32 0, i32 0
// CHECK:STDOUT:   store i32 %x.coerce, ptr %coerce.dive, align 4
// CHECK:STDOUT:   %coerce.dive1 = getelementptr inbounds nuw %struct.A, ptr %y, i32 0, i32 0
// CHECK:STDOUT:   store i32 %y.coerce, ptr %coerce.dive1, align 4
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %y, i64 4, i1 false), !tbaa.struct !25
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp2, ptr align 4 %x, i64 4, i1 false), !tbaa.struct !25
// CHECK:STDOUT:   %coerce.dive3 = getelementptr inbounds nuw %struct.A, ptr %agg.tmp, i32 0, i32 0
// CHECK:STDOUT:   %0 = load i32, ptr %coerce.dive3, align 4
// CHECK:STDOUT:   %coerce.dive4 = getelementptr inbounds nuw %struct.A, ptr %agg.tmp2, i32 0, i32 0
// CHECK:STDOUT:   %1 = load i32, ptr %coerce.dive4, align 4
// CHECK:STDOUT:   %call = call noundef zeroext i1 @_Zlt1AS_(i32 %0, i32 %1)
// CHECK:STDOUT:   ret i1 %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local noundef zeroext i1 @_Zge1AS_(i32 %x.coerce, i32 %y.coerce) #4 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x = alloca %struct.A, align 4
// CHECK:STDOUT:   %y = alloca %struct.A, align 4
// CHECK:STDOUT:   %agg.tmp = alloca %struct.A, align 4
// CHECK:STDOUT:   %agg.tmp2 = alloca %struct.A, align 4
// CHECK:STDOUT:   %coerce.dive = getelementptr inbounds nuw %struct.A, ptr %x, i32 0, i32 0
// CHECK:STDOUT:   store i32 %x.coerce, ptr %coerce.dive, align 4
// CHECK:STDOUT:   %coerce.dive1 = getelementptr inbounds nuw %struct.A, ptr %y, i32 0, i32 0
// CHECK:STDOUT:   store i32 %y.coerce, ptr %coerce.dive1, align 4
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %x, i64 4, i1 false), !tbaa.struct !25
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp2, ptr align 4 %y, i64 4, i1 false), !tbaa.struct !25
// CHECK:STDOUT:   %coerce.dive3 = getelementptr inbounds nuw %struct.A, ptr %agg.tmp, i32 0, i32 0
// CHECK:STDOUT:   %0 = load i32, ptr %coerce.dive3, align 4
// CHECK:STDOUT:   %coerce.dive4 = getelementptr inbounds nuw %struct.A, ptr %agg.tmp2, i32 0, i32 0
// CHECK:STDOUT:   %1 = load i32, ptr %coerce.dive4, align 4
// CHECK:STDOUT:   %call = call noundef zeroext i1 @_Zlt1AS_(i32 %0, i32 %1)
// CHECK:STDOUT:   %lnot = xor i1 %call, true
// CHECK:STDOUT:   ret i1 %lnot
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 0, 1, 4, 5, 8, 9, 13, 12, 11, 10, 7, 6, 3, 2 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #4 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
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
// CHECK:STDOUT: !6 = !DIFile(filename: "ordered_with.carbon", directory: "")
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "Driver", linkageName: "_CDriver.Main", scope: null, file: !6, line: 33, type: !12, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !15)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{null, !14, !14}
// CHECK:STDOUT: !14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !15 = !{!16, !17}
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 1, scope: !11, type: !14)
// CHECK:STDOUT: !17 = !DILocalVariable(arg: 2, scope: !11, type: !14)
// CHECK:STDOUT: !18 = !DILocation(line: 34, column: 3, scope: !11)
// CHECK:STDOUT: !19 = !DILocation(line: 33, column: 1, scope: !11)
// CHECK:STDOUT: !20 = !{!21, !21, i64 0}
// CHECK:STDOUT: !21 = !{!"p1 _ZTS1A", !22, i64 0}
// CHECK:STDOUT: !22 = !{!"any pointer", !9, i64 0}
// CHECK:STDOUT: !23 = !{!24, !24, i64 0}
// CHECK:STDOUT: !24 = !{!"p1 bool", !22, i64 0}
// CHECK:STDOUT: !25 = !{i64 0, i64 4, !7}
// CHECK:STDOUT: !26 = !{!27, !27, i64 0}
// CHECK:STDOUT: !27 = !{!"bool", !9, i64 0}
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "Less", linkageName: "_CLess:thunk:OrderedWith.92c0df05bd55f97a.Core:Cpp", scope: null, file: !6, line: 8, type: !29, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !31)
// CHECK:STDOUT: !29 = !DISubroutineType(types: !30)
// CHECK:STDOUT: !30 = !{!14, !14, !14}
// CHECK:STDOUT: !31 = !{!32, !33}
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !28, type: !14)
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 2, scope: !28, type: !14)
// CHECK:STDOUT: !34 = !DILocation(line: 8, column: 15, scope: !28)
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "LessOrEquivalent", linkageName: "_CLessOrEquivalent:thunk:OrderedWith.92c0df05bd55f97a.Core:Cpp", scope: null, file: !6, line: 16, type: !29, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !36)
// CHECK:STDOUT: !36 = !{!37, !38}
// CHECK:STDOUT: !37 = !DILocalVariable(arg: 1, scope: !35, type: !14)
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 2, scope: !35, type: !14)
// CHECK:STDOUT: !39 = !DILocation(line: 16, column: 15, scope: !35)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "Greater", linkageName: "_CGreater:thunk:OrderedWith.92c0df05bd55f97a.Core:Cpp", scope: null, file: !6, line: 12, type: !29, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !41)
// CHECK:STDOUT: !41 = !{!42, !43}
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 1, scope: !40, type: !14)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 2, scope: !40, type: !14)
// CHECK:STDOUT: !44 = !DILocation(line: 12, column: 15, scope: !40)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "GreaterOrEquivalent", linkageName: "_CGreaterOrEquivalent:thunk:OrderedWith.92c0df05bd55f97a.Core:Cpp", scope: null, file: !6, line: 20, type: !29, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !46)
// CHECK:STDOUT: !46 = !{!47, !48}
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 1, scope: !45, type: !14)
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 2, scope: !45, type: !14)
// CHECK:STDOUT: !49 = !DILocation(line: 20, column: 15, scope: !45)
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "CheckOrderedWith", linkageName: "_CCheckOrderedWith.Main.c49adfa90bd18905", scope: null, file: !6, line: 26, type: !12, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !51)
// CHECK:STDOUT: !51 = !{!52, !53}
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 1, scope: !50, type: !14)
// CHECK:STDOUT: !53 = !DILocalVariable(arg: 2, scope: !50, type: !14)
// CHECK:STDOUT: !54 = !DILocation(line: 27, column: 3, scope: !50)
// CHECK:STDOUT: !55 = !DILocation(line: 28, column: 3, scope: !50)
// CHECK:STDOUT: !56 = !DILocation(line: 29, column: 3, scope: !50)
// CHECK:STDOUT: !57 = !DILocation(line: 30, column: 3, scope: !50)
// CHECK:STDOUT: !58 = !DILocation(line: 26, column: 1, scope: !50)
// CHECK:STDOUT: !59 = !{!60, !8, i64 0}
// CHECK:STDOUT: !60 = !{!"_ZTS1A", !8, i64 0}
