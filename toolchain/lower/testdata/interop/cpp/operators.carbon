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

fn CheckEqWith[U: type, T: Core.EqWith(U)](x: T, y: U) {
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

fn CheckOrderedWith[U: type, T: Core.OrderedWith(U)](x: T, y: U) {
  x < y;
  x > y;
  x <= y;
  x >= y;
}

fn Driver(x: Cpp.A, y: Cpp.A) {
  CheckOrderedWith(x, y);
}

// CHECK:STDOUT: ; ---
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
// CHECK:STDOUT: define void @_CDriver.Main(ptr %x, ptr %y) #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CCheckEqWith.Main.bdef5a2ed02e32a1(ptr %x, ptr %y), !dbg !19
// CHECK:STDOUT:   ret void, !dbg !20
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
// CHECK:STDOUT:   store ptr %x, ptr %x.addr, align 8, !tbaa !21
// CHECK:STDOUT:   store ptr %y, ptr %y.addr, align 8, !tbaa !21
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !24
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !24
// CHECK:STDOUT:   %1 = load ptr, ptr %x.addr, align 8, !tbaa !21
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %1, i64 4, i1 false), !tbaa.struct !26
// CHECK:STDOUT:   %2 = load ptr, ptr %y.addr, align 8, !tbaa !21
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp1, ptr align 4 %2, i64 4, i1 false), !tbaa.struct !26
// CHECK:STDOUT:   %coerce.dive = getelementptr inbounds nuw %struct.A, ptr %agg.tmp, i32 0, i32 0
// CHECK:STDOUT:   %3 = load i32, ptr %coerce.dive, align 4
// CHECK:STDOUT:   %coerce.dive2 = getelementptr inbounds nuw %struct.A, ptr %agg.tmp1, i32 0, i32 0
// CHECK:STDOUT:   %4 = load i32, ptr %coerce.dive2, align 4
// CHECK:STDOUT:   %call = call noundef zeroext i1 @_Zeq1AS_(i32 %3, i32 %4)
// CHECK:STDOUT:   %storedv = zext i1 %call to i8
// CHECK:STDOUT:   store i8 %storedv, ptr %0, align 1, !tbaa !28
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
// CHECK:STDOUT:   store ptr %x, ptr %x.addr, align 8, !tbaa !21
// CHECK:STDOUT:   store ptr %y, ptr %y.addr, align 8, !tbaa !21
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !24
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !24
// CHECK:STDOUT:   %1 = load ptr, ptr %x.addr, align 8, !tbaa !21
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %1, i64 4, i1 false), !tbaa.struct !26
// CHECK:STDOUT:   %2 = load ptr, ptr %y.addr, align 8, !tbaa !21
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp1, ptr align 4 %2, i64 4, i1 false), !tbaa.struct !26
// CHECK:STDOUT:   %coerce.dive = getelementptr inbounds nuw %struct.A, ptr %agg.tmp, i32 0, i32 0
// CHECK:STDOUT:   %3 = load i32, ptr %coerce.dive, align 4
// CHECK:STDOUT:   %coerce.dive2 = getelementptr inbounds nuw %struct.A, ptr %agg.tmp1, i32 0, i32 0
// CHECK:STDOUT:   %4 = load i32, ptr %coerce.dive2, align 4
// CHECK:STDOUT:   %call = call noundef zeroext i1 @_Zne1AS_(i32 %3, i32 %4)
// CHECK:STDOUT:   %storedv = zext i1 %call to i8
// CHECK:STDOUT:   store i8 %storedv, ptr %0, align 1, !tbaa !28
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define i1 @"_CEqual:thunk:EqWith.ffb401eeee8c4872.Core:Cpp"(ptr %self, ptr %other) #2 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.6.temp = alloca i1, align 1, !dbg !36
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.6.temp), !dbg !36
// CHECK:STDOUT:   call void @_Zeq1AS_.carbon_thunk.__(ptr %self, ptr %other, ptr %.6.temp), !dbg !36
// CHECK:STDOUT:   %0 = load i1, ptr %.6.temp, align 1, !dbg !36
// CHECK:STDOUT:   ret i1 %0, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define i1 @"_CNotEqual:thunk:EqWith.ffb401eeee8c4872.Core:Cpp"(ptr %self, ptr %other) #2 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.6.temp = alloca i1, align 1, !dbg !41
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.6.temp), !dbg !41
// CHECK:STDOUT:   call void @_Zne1AS_.carbon_thunk.__(ptr %self, ptr %other, ptr %.6.temp), !dbg !41
// CHECK:STDOUT:   %0 = load i1, ptr %.6.temp, align 1, !dbg !41
// CHECK:STDOUT:   ret i1 %0, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCheckEqWith.Main.bdef5a2ed02e32a1(ptr %x, ptr %y) #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.call = call i1 @"_CEqual:thunk:EqWith.ffb401eeee8c4872.Core:Cpp"(ptr %x, ptr %y), !dbg !46
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.call = call i1 @"_CNotEqual:thunk:EqWith.ffb401eeee8c4872.Core:Cpp"(ptr %x, ptr %y), !dbg !47
// CHECK:STDOUT:   ret void, !dbg !48
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
// CHECK:STDOUT:   %0 = load i32, ptr %value, align 4, !tbaa !49
// CHECK:STDOUT:   %value2 = getelementptr inbounds nuw %struct.A, ptr %y, i32 0, i32 0
// CHECK:STDOUT:   %1 = load i32, ptr %value2, align 4, !tbaa !49
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
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %x, i64 4, i1 false), !tbaa.struct !26
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp2, ptr align 4 %y, i64 4, i1 false), !tbaa.struct !26
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3, !4, !5, !6}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!7}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "eq_with.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !3 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !5 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !6 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !{!8, !9, i64 0}
// CHECK:STDOUT: !8 = !{!"__libc_errno", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"int", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"omnipotent char", !11, i64 0}
// CHECK:STDOUT: !11 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Driver", linkageName: "_CDriver.Main", scope: null, file: !1, line: 23, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !16)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{null, !15, !15}
// CHECK:STDOUT: !15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !16 = !{!17, !18}
// CHECK:STDOUT: !17 = !DILocalVariable(arg: 1, scope: !12, type: !15)
// CHECK:STDOUT: !18 = !DILocalVariable(arg: 2, scope: !12, type: !15)
// CHECK:STDOUT: !19 = !DILocation(line: 24, column: 3, scope: !12)
// CHECK:STDOUT: !20 = !DILocation(line: 23, column: 1, scope: !12)
// CHECK:STDOUT: !21 = !{!22, !22, i64 0}
// CHECK:STDOUT: !22 = !{!"p1 _ZTS1A", !23, i64 0}
// CHECK:STDOUT: !23 = !{!"any pointer", !10, i64 0}
// CHECK:STDOUT: !24 = !{!25, !25, i64 0}
// CHECK:STDOUT: !25 = !{!"p1 bool", !23, i64 0}
// CHECK:STDOUT: !26 = !{i64 0, i64 4, !27}
// CHECK:STDOUT: !27 = !{!9, !9, i64 0}
// CHECK:STDOUT: !28 = !{!29, !29, i64 0}
// CHECK:STDOUT: !29 = !{!"bool", !10, i64 0}
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "Equal", linkageName: "_CEqual:thunk:EqWith.ffb401eeee8c4872.Core:Cpp", scope: null, file: !1, line: 8, type: !31, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !33)
// CHECK:STDOUT: !31 = !DISubroutineType(types: !32)
// CHECK:STDOUT: !32 = !{!15, !15, !15}
// CHECK:STDOUT: !33 = !{!34, !35}
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 1, scope: !30, type: !15)
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 2, scope: !30, type: !15)
// CHECK:STDOUT: !36 = !DILocation(line: 8, column: 15, scope: !30)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "NotEqual", linkageName: "_CNotEqual:thunk:EqWith.ffb401eeee8c4872.Core:Cpp", scope: null, file: !1, line: 12, type: !31, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !38)
// CHECK:STDOUT: !38 = !{!39, !40}
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !37, type: !15)
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 2, scope: !37, type: !15)
// CHECK:STDOUT: !41 = !DILocation(line: 12, column: 15, scope: !37)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "CheckEqWith", linkageName: "_CCheckEqWith.Main.bdef5a2ed02e32a1", scope: null, file: !1, line: 18, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !43)
// CHECK:STDOUT: !43 = !{!44, !45}
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 1, scope: !42, type: !15)
// CHECK:STDOUT: !45 = !DILocalVariable(arg: 2, scope: !42, type: !15)
// CHECK:STDOUT: !46 = !DILocation(line: 19, column: 3, scope: !42)
// CHECK:STDOUT: !47 = !DILocation(line: 20, column: 3, scope: !42)
// CHECK:STDOUT: !48 = !DILocation(line: 18, column: 1, scope: !42)
// CHECK:STDOUT: !49 = !{!50, !9, i64 0}
// CHECK:STDOUT: !50 = !{!"_ZTS1A", !9, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
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
// CHECK:STDOUT: define void @_CDriver.Main(ptr %x, ptr %y) #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CCheckOrderedWith.Main.c49adfa90bd18905(ptr %x, ptr %y), !dbg !19
// CHECK:STDOUT:   ret void, !dbg !20
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
// CHECK:STDOUT:   store ptr %x, ptr %x.addr, align 8, !tbaa !21
// CHECK:STDOUT:   store ptr %y, ptr %y.addr, align 8, !tbaa !21
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !24
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !24
// CHECK:STDOUT:   %1 = load ptr, ptr %x.addr, align 8, !tbaa !21
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %1, i64 4, i1 false), !tbaa.struct !26
// CHECK:STDOUT:   %2 = load ptr, ptr %y.addr, align 8, !tbaa !21
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp1, ptr align 4 %2, i64 4, i1 false), !tbaa.struct !26
// CHECK:STDOUT:   %coerce.dive = getelementptr inbounds nuw %struct.A, ptr %agg.tmp, i32 0, i32 0
// CHECK:STDOUT:   %3 = load i32, ptr %coerce.dive, align 4
// CHECK:STDOUT:   %coerce.dive2 = getelementptr inbounds nuw %struct.A, ptr %agg.tmp1, i32 0, i32 0
// CHECK:STDOUT:   %4 = load i32, ptr %coerce.dive2, align 4
// CHECK:STDOUT:   %call = call noundef zeroext i1 @_Zlt1AS_(i32 %3, i32 %4)
// CHECK:STDOUT:   %storedv = zext i1 %call to i8
// CHECK:STDOUT:   store i8 %storedv, ptr %0, align 1, !tbaa !28
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
// CHECK:STDOUT:   store ptr %x, ptr %x.addr, align 8, !tbaa !21
// CHECK:STDOUT:   store ptr %y, ptr %y.addr, align 8, !tbaa !21
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !24
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !24
// CHECK:STDOUT:   %1 = load ptr, ptr %x.addr, align 8, !tbaa !21
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %1, i64 4, i1 false), !tbaa.struct !26
// CHECK:STDOUT:   %2 = load ptr, ptr %y.addr, align 8, !tbaa !21
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp1, ptr align 4 %2, i64 4, i1 false), !tbaa.struct !26
// CHECK:STDOUT:   %coerce.dive = getelementptr inbounds nuw %struct.A, ptr %agg.tmp, i32 0, i32 0
// CHECK:STDOUT:   %3 = load i32, ptr %coerce.dive, align 4
// CHECK:STDOUT:   %coerce.dive2 = getelementptr inbounds nuw %struct.A, ptr %agg.tmp1, i32 0, i32 0
// CHECK:STDOUT:   %4 = load i32, ptr %coerce.dive2, align 4
// CHECK:STDOUT:   %call = call noundef zeroext i1 @_Zle1AS_(i32 %3, i32 %4)
// CHECK:STDOUT:   %storedv = zext i1 %call to i8
// CHECK:STDOUT:   store i8 %storedv, ptr %0, align 1, !tbaa !28
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
// CHECK:STDOUT:   store ptr %x, ptr %x.addr, align 8, !tbaa !21
// CHECK:STDOUT:   store ptr %y, ptr %y.addr, align 8, !tbaa !21
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !24
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !24
// CHECK:STDOUT:   %1 = load ptr, ptr %x.addr, align 8, !tbaa !21
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %1, i64 4, i1 false), !tbaa.struct !26
// CHECK:STDOUT:   %2 = load ptr, ptr %y.addr, align 8, !tbaa !21
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp1, ptr align 4 %2, i64 4, i1 false), !tbaa.struct !26
// CHECK:STDOUT:   %coerce.dive = getelementptr inbounds nuw %struct.A, ptr %agg.tmp, i32 0, i32 0
// CHECK:STDOUT:   %3 = load i32, ptr %coerce.dive, align 4
// CHECK:STDOUT:   %coerce.dive2 = getelementptr inbounds nuw %struct.A, ptr %agg.tmp1, i32 0, i32 0
// CHECK:STDOUT:   %4 = load i32, ptr %coerce.dive2, align 4
// CHECK:STDOUT:   %call = call noundef zeroext i1 @_Zgt1AS_(i32 %3, i32 %4)
// CHECK:STDOUT:   %storedv = zext i1 %call to i8
// CHECK:STDOUT:   store i8 %storedv, ptr %0, align 1, !tbaa !28
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
// CHECK:STDOUT:   store ptr %x, ptr %x.addr, align 8, !tbaa !21
// CHECK:STDOUT:   store ptr %y, ptr %y.addr, align 8, !tbaa !21
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !24
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !24
// CHECK:STDOUT:   %1 = load ptr, ptr %x.addr, align 8, !tbaa !21
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %1, i64 4, i1 false), !tbaa.struct !26
// CHECK:STDOUT:   %2 = load ptr, ptr %y.addr, align 8, !tbaa !21
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp1, ptr align 4 %2, i64 4, i1 false), !tbaa.struct !26
// CHECK:STDOUT:   %coerce.dive = getelementptr inbounds nuw %struct.A, ptr %agg.tmp, i32 0, i32 0
// CHECK:STDOUT:   %3 = load i32, ptr %coerce.dive, align 4
// CHECK:STDOUT:   %coerce.dive2 = getelementptr inbounds nuw %struct.A, ptr %agg.tmp1, i32 0, i32 0
// CHECK:STDOUT:   %4 = load i32, ptr %coerce.dive2, align 4
// CHECK:STDOUT:   %call = call noundef zeroext i1 @_Zge1AS_(i32 %3, i32 %4)
// CHECK:STDOUT:   %storedv = zext i1 %call to i8
// CHECK:STDOUT:   store i8 %storedv, ptr %0, align 1, !tbaa !28
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define i1 @"_CLess:thunk:OrderedWith.92c0df05bd55f97a.Core:Cpp"(ptr %self, ptr %other) #2 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.6.temp = alloca i1, align 1, !dbg !36
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.6.temp), !dbg !36
// CHECK:STDOUT:   call void @_Zlt1AS_.carbon_thunk.__(ptr %self, ptr %other, ptr %.6.temp), !dbg !36
// CHECK:STDOUT:   %0 = load i1, ptr %.6.temp, align 1, !dbg !36
// CHECK:STDOUT:   ret i1 %0, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define i1 @"_CLessOrEquivalent:thunk:OrderedWith.92c0df05bd55f97a.Core:Cpp"(ptr %self, ptr %other) #2 !dbg !37 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.6.temp = alloca i1, align 1, !dbg !41
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.6.temp), !dbg !41
// CHECK:STDOUT:   call void @_Zle1AS_.carbon_thunk.__(ptr %self, ptr %other, ptr %.6.temp), !dbg !41
// CHECK:STDOUT:   %0 = load i1, ptr %.6.temp, align 1, !dbg !41
// CHECK:STDOUT:   ret i1 %0, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define i1 @"_CGreater:thunk:OrderedWith.92c0df05bd55f97a.Core:Cpp"(ptr %self, ptr %other) #2 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.6.temp = alloca i1, align 1, !dbg !46
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.6.temp), !dbg !46
// CHECK:STDOUT:   call void @_Zgt1AS_.carbon_thunk.__(ptr %self, ptr %other, ptr %.6.temp), !dbg !46
// CHECK:STDOUT:   %0 = load i1, ptr %.6.temp, align 1, !dbg !46
// CHECK:STDOUT:   ret i1 %0, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define i1 @"_CGreaterOrEquivalent:thunk:OrderedWith.92c0df05bd55f97a.Core:Cpp"(ptr %self, ptr %other) #2 !dbg !47 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.6.temp = alloca i1, align 1, !dbg !51
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.6.temp), !dbg !51
// CHECK:STDOUT:   call void @_Zge1AS_.carbon_thunk.__(ptr %self, ptr %other, ptr %.6.temp), !dbg !51
// CHECK:STDOUT:   %0 = load i1, ptr %.6.temp, align 1, !dbg !51
// CHECK:STDOUT:   ret i1 %0, !dbg !51
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCheckOrderedWith.Main.c49adfa90bd18905(ptr %x, ptr %y) #0 !dbg !52 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.call = call i1 @"_CLess:thunk:OrderedWith.92c0df05bd55f97a.Core:Cpp"(ptr %x, ptr %y), !dbg !56
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.call = call i1 @"_CGreater:thunk:OrderedWith.92c0df05bd55f97a.Core:Cpp"(ptr %x, ptr %y), !dbg !57
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.call = call i1 @"_CLessOrEquivalent:thunk:OrderedWith.92c0df05bd55f97a.Core:Cpp"(ptr %x, ptr %y), !dbg !58
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.call = call i1 @"_CGreaterOrEquivalent:thunk:OrderedWith.92c0df05bd55f97a.Core:Cpp"(ptr %x, ptr %y), !dbg !59
// CHECK:STDOUT:   ret void, !dbg !60
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
// CHECK:STDOUT:   %0 = load i32, ptr %value, align 4, !tbaa !61
// CHECK:STDOUT:   %value2 = getelementptr inbounds nuw %struct.A, ptr %y, i32 0, i32 0
// CHECK:STDOUT:   %1 = load i32, ptr %value2, align 4, !tbaa !61
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
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %y, i64 4, i1 false), !tbaa.struct !26
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp2, ptr align 4 %x, i64 4, i1 false), !tbaa.struct !26
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
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %y, i64 4, i1 false), !tbaa.struct !26
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp2, ptr align 4 %x, i64 4, i1 false), !tbaa.struct !26
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
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %x, i64 4, i1 false), !tbaa.struct !26
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp2, ptr align 4 %y, i64 4, i1 false), !tbaa.struct !26
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3, !4, !5, !6}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!7}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "ordered_with.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !3 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !5 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !6 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !{!8, !9, i64 0}
// CHECK:STDOUT: !8 = !{!"__libc_errno", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"int", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"omnipotent char", !11, i64 0}
// CHECK:STDOUT: !11 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Driver", linkageName: "_CDriver.Main", scope: null, file: !1, line: 33, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !16)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{null, !15, !15}
// CHECK:STDOUT: !15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !16 = !{!17, !18}
// CHECK:STDOUT: !17 = !DILocalVariable(arg: 1, scope: !12, type: !15)
// CHECK:STDOUT: !18 = !DILocalVariable(arg: 2, scope: !12, type: !15)
// CHECK:STDOUT: !19 = !DILocation(line: 34, column: 3, scope: !12)
// CHECK:STDOUT: !20 = !DILocation(line: 33, column: 1, scope: !12)
// CHECK:STDOUT: !21 = !{!22, !22, i64 0}
// CHECK:STDOUT: !22 = !{!"p1 _ZTS1A", !23, i64 0}
// CHECK:STDOUT: !23 = !{!"any pointer", !10, i64 0}
// CHECK:STDOUT: !24 = !{!25, !25, i64 0}
// CHECK:STDOUT: !25 = !{!"p1 bool", !23, i64 0}
// CHECK:STDOUT: !26 = !{i64 0, i64 4, !27}
// CHECK:STDOUT: !27 = !{!9, !9, i64 0}
// CHECK:STDOUT: !28 = !{!29, !29, i64 0}
// CHECK:STDOUT: !29 = !{!"bool", !10, i64 0}
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "Less", linkageName: "_CLess:thunk:OrderedWith.92c0df05bd55f97a.Core:Cpp", scope: null, file: !1, line: 8, type: !31, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !33)
// CHECK:STDOUT: !31 = !DISubroutineType(types: !32)
// CHECK:STDOUT: !32 = !{!15, !15, !15}
// CHECK:STDOUT: !33 = !{!34, !35}
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 1, scope: !30, type: !15)
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 2, scope: !30, type: !15)
// CHECK:STDOUT: !36 = !DILocation(line: 8, column: 15, scope: !30)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "LessOrEquivalent", linkageName: "_CLessOrEquivalent:thunk:OrderedWith.92c0df05bd55f97a.Core:Cpp", scope: null, file: !1, line: 16, type: !31, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !38)
// CHECK:STDOUT: !38 = !{!39, !40}
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !37, type: !15)
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 2, scope: !37, type: !15)
// CHECK:STDOUT: !41 = !DILocation(line: 16, column: 15, scope: !37)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "Greater", linkageName: "_CGreater:thunk:OrderedWith.92c0df05bd55f97a.Core:Cpp", scope: null, file: !1, line: 12, type: !31, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !43)
// CHECK:STDOUT: !43 = !{!44, !45}
// CHECK:STDOUT: !44 = !DILocalVariable(arg: 1, scope: !42, type: !15)
// CHECK:STDOUT: !45 = !DILocalVariable(arg: 2, scope: !42, type: !15)
// CHECK:STDOUT: !46 = !DILocation(line: 12, column: 15, scope: !42)
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "GreaterOrEquivalent", linkageName: "_CGreaterOrEquivalent:thunk:OrderedWith.92c0df05bd55f97a.Core:Cpp", scope: null, file: !1, line: 20, type: !31, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !48)
// CHECK:STDOUT: !48 = !{!49, !50}
// CHECK:STDOUT: !49 = !DILocalVariable(arg: 1, scope: !47, type: !15)
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 2, scope: !47, type: !15)
// CHECK:STDOUT: !51 = !DILocation(line: 20, column: 15, scope: !47)
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "CheckOrderedWith", linkageName: "_CCheckOrderedWith.Main.c49adfa90bd18905", scope: null, file: !1, line: 26, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !53)
// CHECK:STDOUT: !53 = !{!54, !55}
// CHECK:STDOUT: !54 = !DILocalVariable(arg: 1, scope: !52, type: !15)
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 2, scope: !52, type: !15)
// CHECK:STDOUT: !56 = !DILocation(line: 27, column: 3, scope: !52)
// CHECK:STDOUT: !57 = !DILocation(line: 28, column: 3, scope: !52)
// CHECK:STDOUT: !58 = !DILocation(line: 29, column: 3, scope: !52)
// CHECK:STDOUT: !59 = !DILocation(line: 30, column: 3, scope: !52)
// CHECK:STDOUT: !60 = !DILocation(line: 26, column: 1, scope: !52)
// CHECK:STDOUT: !61 = !{!62, !9, i64 0}
// CHECK:STDOUT: !62 = !{!"_ZTS1A", !9, i64 0}
// CHECK:STDOUT:
