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
struct EqA {
  int value;

  friend auto operator==(EqA x, EqA y) -> bool {
    return x.value == y.value;
  }

  friend auto operator!=(EqA x, EqA y) -> bool {
    return !(x == y);
  }
};
''';

fn CheckEqWith[U: type, T: Core.EqWith(U)](x: T, y: U) {
  x == y;
  x != y;
}

fn Driver(x: Cpp.EqA, y: Cpp.EqA) {
  CheckEqWith(x, y);
}

// --- ordered_with.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
struct OrderedA {
  int value;

  friend auto operator<(OrderedA x, OrderedA y) -> bool {
    return x.value < y.value;
  }

  friend auto operator>(OrderedA x, OrderedA y) -> bool {
    return y < x;
  }

  friend auto operator<=(OrderedA x, OrderedA y) -> bool {
    return !(y < x);
  }

  friend auto operator>=(OrderedA x, OrderedA y) -> bool {
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

fn Driver(x: Cpp.OrderedA, y: Cpp.OrderedA) {
  CheckOrderedWith(x, y);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'eq_with.carbon'
// CHECK:STDOUT: source_filename = "eq_with.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %struct.EqA = type { i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: $_Zeq3EqAS_ = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_Zne3EqAS_ = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CDriver.Main(ptr %x, ptr %y) #0 !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CCheckEqWith.Main.5d19350ce19a6fbd(ptr %x, ptr %y), !dbg !21
// CHECK:STDOUT:   ret void, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Zeq3EqAS_.carbon_thunk.__(ptr noundef %x, ptr noundef %y, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.addr = alloca ptr, align 8
// CHECK:STDOUT:   %y.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %struct.EqA, align 4
// CHECK:STDOUT:   %agg.tmp1 = alloca %struct.EqA, align 4
// CHECK:STDOUT:   store ptr %x, ptr %x.addr, align 8, !tbaa !51
// CHECK:STDOUT:   store ptr %y, ptr %y.addr, align 8, !tbaa !51
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !54
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !54
// CHECK:STDOUT:   %1 = load ptr, ptr %x.addr, align 8, !tbaa !51
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %1, i64 4, i1 false), !tbaa.struct !58
// CHECK:STDOUT:   %2 = load ptr, ptr %y.addr, align 8, !tbaa !51
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp1, ptr align 4 %2, i64 4, i1 false), !tbaa.struct !58
// CHECK:STDOUT:   %coerce.dive = getelementptr inbounds nuw %struct.EqA, ptr %agg.tmp, i32 0, i32 0
// CHECK:STDOUT:   %3 = load i32, ptr %coerce.dive, align 4
// CHECK:STDOUT:   %coerce.dive2 = getelementptr inbounds nuw %struct.EqA, ptr %agg.tmp1, i32 0, i32 0
// CHECK:STDOUT:   %4 = load i32, ptr %coerce.dive2, align 4
// CHECK:STDOUT:   %call = call noundef zeroext i1 @_Zeq3EqAS_(i32 %3, i32 %4)
// CHECK:STDOUT:   %storedv = zext i1 %call to i8
// CHECK:STDOUT:   store i8 %storedv, ptr %0, align 1, !tbaa !59
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Zne3EqAS_.carbon_thunk.__(ptr noundef %x, ptr noundef %y, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.addr = alloca ptr, align 8
// CHECK:STDOUT:   %y.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %struct.EqA, align 4
// CHECK:STDOUT:   %agg.tmp1 = alloca %struct.EqA, align 4
// CHECK:STDOUT:   store ptr %x, ptr %x.addr, align 8, !tbaa !51
// CHECK:STDOUT:   store ptr %y, ptr %y.addr, align 8, !tbaa !51
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !54
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !54
// CHECK:STDOUT:   %1 = load ptr, ptr %x.addr, align 8, !tbaa !51
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %1, i64 4, i1 false), !tbaa.struct !58
// CHECK:STDOUT:   %2 = load ptr, ptr %y.addr, align 8, !tbaa !51
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp1, ptr align 4 %2, i64 4, i1 false), !tbaa.struct !58
// CHECK:STDOUT:   %coerce.dive = getelementptr inbounds nuw %struct.EqA, ptr %agg.tmp, i32 0, i32 0
// CHECK:STDOUT:   %3 = load i32, ptr %coerce.dive, align 4
// CHECK:STDOUT:   %coerce.dive2 = getelementptr inbounds nuw %struct.EqA, ptr %agg.tmp1, i32 0, i32 0
// CHECK:STDOUT:   %4 = load i32, ptr %coerce.dive2, align 4
// CHECK:STDOUT:   %call = call noundef zeroext i1 @_Zne3EqAS_(i32 %3, i32 %4)
// CHECK:STDOUT:   %storedv = zext i1 %call to i8
// CHECK:STDOUT:   store i8 %storedv, ptr %0, align 1, !tbaa !59
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CEqual:thunk:EqWith.c41e21bae407dec8.Core:Cpp"(ptr %self, ptr %other) #2 !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.6.temp = alloca i1, align 1, !dbg !28
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.6.temp), !dbg !28
// CHECK:STDOUT:   call void @_Zeq3EqAS_.carbon_thunk.__(ptr %self, ptr %other, ptr %.6.temp), !dbg !28
// CHECK:STDOUT:   %0 = load i1, ptr %.6.temp, align 1, !dbg !28
// CHECK:STDOUT:   ret i1 %0, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CNotEqual:thunk:EqWith.c41e21bae407dec8.Core:Cpp"(ptr %self, ptr %other) #2 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.6.temp = alloca i1, align 1, !dbg !34
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.6.temp), !dbg !34
// CHECK:STDOUT:   call void @_Zne3EqAS_.carbon_thunk.__(ptr %self, ptr %other, ptr %.6.temp), !dbg !34
// CHECK:STDOUT:   %0 = load i1, ptr %.6.temp, align 1, !dbg !34
// CHECK:STDOUT:   ret i1 %0, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCheckEqWith.Main.5d19350ce19a6fbd(ptr %x, ptr %y) #0 !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.call = call i1 @"_CEqual:thunk:EqWith.c41e21bae407dec8.Core:Cpp"(ptr %x, ptr %y), !dbg !41
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.call = call i1 @"_CNotEqual:thunk:EqWith.c41e21bae407dec8.Core:Cpp"(ptr %x, ptr %y), !dbg !43
// CHECK:STDOUT:   ret void, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local noundef zeroext i1 @_Zeq3EqAS_(i32 %x.coerce, i32 %y.coerce) #4 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x = alloca %struct.EqA, align 4
// CHECK:STDOUT:   %y = alloca %struct.EqA, align 4
// CHECK:STDOUT:   %coerce.dive = getelementptr inbounds nuw %struct.EqA, ptr %x, i32 0, i32 0
// CHECK:STDOUT:   store i32 %x.coerce, ptr %coerce.dive, align 4
// CHECK:STDOUT:   %coerce.dive1 = getelementptr inbounds nuw %struct.EqA, ptr %y, i32 0, i32 0
// CHECK:STDOUT:   store i32 %y.coerce, ptr %coerce.dive1, align 4
// CHECK:STDOUT:   %value = getelementptr inbounds nuw %struct.EqA, ptr %x, i32 0, i32 0
// CHECK:STDOUT:   %0 = load i32, ptr %value, align 4, !tbaa !60
// CHECK:STDOUT:   %value2 = getelementptr inbounds nuw %struct.EqA, ptr %y, i32 0, i32 0
// CHECK:STDOUT:   %1 = load i32, ptr %value2, align 4, !tbaa !60
// CHECK:STDOUT:   %cmp = icmp eq i32 %0, %1
// CHECK:STDOUT:   ret i1 %cmp
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local noundef zeroext i1 @_Zne3EqAS_(i32 %x.coerce, i32 %y.coerce) #4 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x = alloca %struct.EqA, align 4
// CHECK:STDOUT:   %y = alloca %struct.EqA, align 4
// CHECK:STDOUT:   %agg.tmp = alloca %struct.EqA, align 4
// CHECK:STDOUT:   %agg.tmp2 = alloca %struct.EqA, align 4
// CHECK:STDOUT:   %coerce.dive = getelementptr inbounds nuw %struct.EqA, ptr %x, i32 0, i32 0
// CHECK:STDOUT:   store i32 %x.coerce, ptr %coerce.dive, align 4
// CHECK:STDOUT:   %coerce.dive1 = getelementptr inbounds nuw %struct.EqA, ptr %y, i32 0, i32 0
// CHECK:STDOUT:   store i32 %y.coerce, ptr %coerce.dive1, align 4
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %x, i64 4, i1 false), !tbaa.struct !58
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp2, ptr align 4 %y, i64 4, i1 false), !tbaa.struct !58
// CHECK:STDOUT:   %coerce.dive3 = getelementptr inbounds nuw %struct.EqA, ptr %agg.tmp, i32 0, i32 0
// CHECK:STDOUT:   %0 = load i32, ptr %coerce.dive3, align 4
// CHECK:STDOUT:   %coerce.dive4 = getelementptr inbounds nuw %struct.EqA, ptr %agg.tmp2, i32 0, i32 0
// CHECK:STDOUT:   %1 = load i32, ptr %coerce.dive4, align 4
// CHECK:STDOUT:   %call = call noundef zeroext i1 @_Zeq3EqAS_(i32 %0, i32 %1)
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!61, !62, !63, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!65}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "eq_with.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{null, !7, !7}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !12 = !{!7, !7, !7}
// CHECK:STDOUT: !13 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "Driver", linkageName: "_CDriver.Main", scope: null, file: !4, line: 23, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !23)
// CHECK:STDOUT: !15 = !DILocalVariable(arg: 1, scope: !14, type: !7)
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 2, scope: !14, type: !7)
// CHECK:STDOUT: !21 = !DILocation(line: 24, column: 3, scope: !14)
// CHECK:STDOUT: !22 = !DILocation(line: 23, column: 1, scope: !14)
// CHECK:STDOUT: !23 = !{!15, !16}
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "Equal", linkageName: "_CEqual:thunk:EqWith.c41e21bae407dec8.Core:Cpp", scope: null, file: !4, line: 8, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !29)
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !24, type: !7)
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 2, scope: !24, type: !7)
// CHECK:STDOUT: !28 = !DILocation(line: 8, column: 15, scope: !24)
// CHECK:STDOUT: !29 = !{!25, !26}
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "NotEqual", linkageName: "_CNotEqual:thunk:EqWith.c41e21bae407dec8.Core:Cpp", scope: null, file: !4, line: 12, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !35)
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 1, scope: !30, type: !7)
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 2, scope: !30, type: !7)
// CHECK:STDOUT: !34 = !DILocation(line: 12, column: 15, scope: !30)
// CHECK:STDOUT: !35 = !{!31, !32}
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "CheckEqWith", linkageName: "_CCheckEqWith.Main.5d19350ce19a6fbd", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !46)
// CHECK:STDOUT: !37 = !DILocalVariable(arg: 1, scope: !36, type: !7)
// CHECK:STDOUT: !38 = !DILocalVariable(arg: 2, scope: !36, type: !7)
// CHECK:STDOUT: !41 = !DILocation(line: 19, column: 3, scope: !36)
// CHECK:STDOUT: !43 = !DILocation(line: 20, column: 3, scope: !36)
// CHECK:STDOUT: !45 = !DILocation(line: 18, column: 1, scope: !36)
// CHECK:STDOUT: !46 = !{!37, !38}
// CHECK:STDOUT: !47 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !48 = !{!"omnipotent char", !47, i64 0}
// CHECK:STDOUT: !49 = !{!"any pointer", !48, i64 0}
// CHECK:STDOUT: !50 = !{!"p1 _ZTS3EqA", !49, i64 0}
// CHECK:STDOUT: !51 = !{!50, !50, i64 0}
// CHECK:STDOUT: !52 = !{!"bool", !48, i64 0}
// CHECK:STDOUT: !53 = !{!"p1 bool", !49, i64 0}
// CHECK:STDOUT: !54 = !{!53, !53, i64 0}
// CHECK:STDOUT: !55 = !{!"int", !48, i64 0}
// CHECK:STDOUT: !56 = !{!"_ZTS3EqA", !55, i64 0}
// CHECK:STDOUT: !57 = !{!55, !55, i64 0}
// CHECK:STDOUT: !58 = !{i64 0, i64 4, !57}
// CHECK:STDOUT: !59 = !{!52, !52, i64 0}
// CHECK:STDOUT: !60 = !{!56, !55, i64 0}
// CHECK:STDOUT: !61 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !62 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !63 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !64 = !{!"__libc_errno", !55, i64 0}
// CHECK:STDOUT: !65 = !{!64, !55, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'ordered_with.carbon'
// CHECK:STDOUT: source_filename = "ordered_with.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %struct.OrderedA = type { i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: $_Zlt8OrderedAS_ = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_Zle8OrderedAS_ = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_Zgt8OrderedAS_ = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_Zge8OrderedAS_ = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CDriver.Main(ptr %x, ptr %y) #0 !dbg !68 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CCheckOrderedWith.Main.f3d1df67aae0c1db(ptr %x, ptr %y), !dbg !75
// CHECK:STDOUT:   ret void, !dbg !76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Zlt8OrderedAS_.carbon_thunk.__(ptr noundef %x, ptr noundef %y, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.addr = alloca ptr, align 8
// CHECK:STDOUT:   %y.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %struct.OrderedA, align 4
// CHECK:STDOUT:   %agg.tmp1 = alloca %struct.OrderedA, align 4
// CHECK:STDOUT:   store ptr %x, ptr %x.addr, align 8, !tbaa !118
// CHECK:STDOUT:   store ptr %y, ptr %y.addr, align 8, !tbaa !118
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !54
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !54
// CHECK:STDOUT:   %1 = load ptr, ptr %x.addr, align 8, !tbaa !118
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %1, i64 4, i1 false), !tbaa.struct !58
// CHECK:STDOUT:   %2 = load ptr, ptr %y.addr, align 8, !tbaa !118
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp1, ptr align 4 %2, i64 4, i1 false), !tbaa.struct !58
// CHECK:STDOUT:   %coerce.dive = getelementptr inbounds nuw %struct.OrderedA, ptr %agg.tmp, i32 0, i32 0
// CHECK:STDOUT:   %3 = load i32, ptr %coerce.dive, align 4
// CHECK:STDOUT:   %coerce.dive2 = getelementptr inbounds nuw %struct.OrderedA, ptr %agg.tmp1, i32 0, i32 0
// CHECK:STDOUT:   %4 = load i32, ptr %coerce.dive2, align 4
// CHECK:STDOUT:   %call = call noundef zeroext i1 @_Zlt8OrderedAS_(i32 %3, i32 %4)
// CHECK:STDOUT:   %storedv = zext i1 %call to i8
// CHECK:STDOUT:   store i8 %storedv, ptr %0, align 1, !tbaa !59
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Zle8OrderedAS_.carbon_thunk.__(ptr noundef %x, ptr noundef %y, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.addr = alloca ptr, align 8
// CHECK:STDOUT:   %y.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %struct.OrderedA, align 4
// CHECK:STDOUT:   %agg.tmp1 = alloca %struct.OrderedA, align 4
// CHECK:STDOUT:   store ptr %x, ptr %x.addr, align 8, !tbaa !118
// CHECK:STDOUT:   store ptr %y, ptr %y.addr, align 8, !tbaa !118
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !54
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !54
// CHECK:STDOUT:   %1 = load ptr, ptr %x.addr, align 8, !tbaa !118
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %1, i64 4, i1 false), !tbaa.struct !58
// CHECK:STDOUT:   %2 = load ptr, ptr %y.addr, align 8, !tbaa !118
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp1, ptr align 4 %2, i64 4, i1 false), !tbaa.struct !58
// CHECK:STDOUT:   %coerce.dive = getelementptr inbounds nuw %struct.OrderedA, ptr %agg.tmp, i32 0, i32 0
// CHECK:STDOUT:   %3 = load i32, ptr %coerce.dive, align 4
// CHECK:STDOUT:   %coerce.dive2 = getelementptr inbounds nuw %struct.OrderedA, ptr %agg.tmp1, i32 0, i32 0
// CHECK:STDOUT:   %4 = load i32, ptr %coerce.dive2, align 4
// CHECK:STDOUT:   %call = call noundef zeroext i1 @_Zle8OrderedAS_(i32 %3, i32 %4)
// CHECK:STDOUT:   %storedv = zext i1 %call to i8
// CHECK:STDOUT:   store i8 %storedv, ptr %0, align 1, !tbaa !59
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Zgt8OrderedAS_.carbon_thunk.__(ptr noundef %x, ptr noundef %y, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.addr = alloca ptr, align 8
// CHECK:STDOUT:   %y.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %struct.OrderedA, align 4
// CHECK:STDOUT:   %agg.tmp1 = alloca %struct.OrderedA, align 4
// CHECK:STDOUT:   store ptr %x, ptr %x.addr, align 8, !tbaa !118
// CHECK:STDOUT:   store ptr %y, ptr %y.addr, align 8, !tbaa !118
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !54
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !54
// CHECK:STDOUT:   %1 = load ptr, ptr %x.addr, align 8, !tbaa !118
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %1, i64 4, i1 false), !tbaa.struct !58
// CHECK:STDOUT:   %2 = load ptr, ptr %y.addr, align 8, !tbaa !118
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp1, ptr align 4 %2, i64 4, i1 false), !tbaa.struct !58
// CHECK:STDOUT:   %coerce.dive = getelementptr inbounds nuw %struct.OrderedA, ptr %agg.tmp, i32 0, i32 0
// CHECK:STDOUT:   %3 = load i32, ptr %coerce.dive, align 4
// CHECK:STDOUT:   %coerce.dive2 = getelementptr inbounds nuw %struct.OrderedA, ptr %agg.tmp1, i32 0, i32 0
// CHECK:STDOUT:   %4 = load i32, ptr %coerce.dive2, align 4
// CHECK:STDOUT:   %call = call noundef zeroext i1 @_Zgt8OrderedAS_(i32 %3, i32 %4)
// CHECK:STDOUT:   %storedv = zext i1 %call to i8
// CHECK:STDOUT:   store i8 %storedv, ptr %0, align 1, !tbaa !59
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Zge8OrderedAS_.carbon_thunk.__(ptr noundef %x, ptr noundef %y, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.addr = alloca ptr, align 8
// CHECK:STDOUT:   %y.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %struct.OrderedA, align 4
// CHECK:STDOUT:   %agg.tmp1 = alloca %struct.OrderedA, align 4
// CHECK:STDOUT:   store ptr %x, ptr %x.addr, align 8, !tbaa !118
// CHECK:STDOUT:   store ptr %y, ptr %y.addr, align 8, !tbaa !118
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !54
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !54
// CHECK:STDOUT:   %1 = load ptr, ptr %x.addr, align 8, !tbaa !118
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %1, i64 4, i1 false), !tbaa.struct !58
// CHECK:STDOUT:   %2 = load ptr, ptr %y.addr, align 8, !tbaa !118
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp1, ptr align 4 %2, i64 4, i1 false), !tbaa.struct !58
// CHECK:STDOUT:   %coerce.dive = getelementptr inbounds nuw %struct.OrderedA, ptr %agg.tmp, i32 0, i32 0
// CHECK:STDOUT:   %3 = load i32, ptr %coerce.dive, align 4
// CHECK:STDOUT:   %coerce.dive2 = getelementptr inbounds nuw %struct.OrderedA, ptr %agg.tmp1, i32 0, i32 0
// CHECK:STDOUT:   %4 = load i32, ptr %coerce.dive2, align 4
// CHECK:STDOUT:   %call = call noundef zeroext i1 @_Zge8OrderedAS_(i32 %3, i32 %4)
// CHECK:STDOUT:   %storedv = zext i1 %call to i8
// CHECK:STDOUT:   store i8 %storedv, ptr %0, align 1, !tbaa !59
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CLess:thunk:OrderedWith.f150fc1157a5293b.Core:Cpp"(ptr %self, ptr %other) #2 !dbg !78 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.6.temp = alloca i1, align 1, !dbg !82
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.6.temp), !dbg !82
// CHECK:STDOUT:   call void @_Zlt8OrderedAS_.carbon_thunk.__(ptr %self, ptr %other, ptr %.6.temp), !dbg !82
// CHECK:STDOUT:   %0 = load i1, ptr %.6.temp, align 1, !dbg !82
// CHECK:STDOUT:   ret i1 %0, !dbg !82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CLessOrEquivalent:thunk:OrderedWith.f150fc1157a5293b.Core:Cpp"(ptr %self, ptr %other) #2 !dbg !84 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.6.temp = alloca i1, align 1, !dbg !88
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.6.temp), !dbg !88
// CHECK:STDOUT:   call void @_Zle8OrderedAS_.carbon_thunk.__(ptr %self, ptr %other, ptr %.6.temp), !dbg !88
// CHECK:STDOUT:   %0 = load i1, ptr %.6.temp, align 1, !dbg !88
// CHECK:STDOUT:   ret i1 %0, !dbg !88
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CGreater:thunk:OrderedWith.f150fc1157a5293b.Core:Cpp"(ptr %self, ptr %other) #2 !dbg !90 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.6.temp = alloca i1, align 1, !dbg !94
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.6.temp), !dbg !94
// CHECK:STDOUT:   call void @_Zgt8OrderedAS_.carbon_thunk.__(ptr %self, ptr %other, ptr %.6.temp), !dbg !94
// CHECK:STDOUT:   %0 = load i1, ptr %.6.temp, align 1, !dbg !94
// CHECK:STDOUT:   ret i1 %0, !dbg !94
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CGreaterOrEquivalent:thunk:OrderedWith.f150fc1157a5293b.Core:Cpp"(ptr %self, ptr %other) #2 !dbg !96 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.6.temp = alloca i1, align 1, !dbg !100
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.6.temp), !dbg !100
// CHECK:STDOUT:   call void @_Zge8OrderedAS_.carbon_thunk.__(ptr %self, ptr %other, ptr %.6.temp), !dbg !100
// CHECK:STDOUT:   %0 = load i1, ptr %.6.temp, align 1, !dbg !100
// CHECK:STDOUT:   ret i1 %0, !dbg !100
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCheckOrderedWith.Main.f3d1df67aae0c1db(ptr %x, ptr %y) #0 !dbg !102 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.call = call i1 @"_CLess:thunk:OrderedWith.f150fc1157a5293b.Core:Cpp"(ptr %x, ptr %y), !dbg !107
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.call = call i1 @"_CGreater:thunk:OrderedWith.f150fc1157a5293b.Core:Cpp"(ptr %x, ptr %y), !dbg !109
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.call = call i1 @"_CLessOrEquivalent:thunk:OrderedWith.f150fc1157a5293b.Core:Cpp"(ptr %x, ptr %y), !dbg !111
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.call = call i1 @"_CGreaterOrEquivalent:thunk:OrderedWith.f150fc1157a5293b.Core:Cpp"(ptr %x, ptr %y), !dbg !113
// CHECK:STDOUT:   ret void, !dbg !115
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local noundef zeroext i1 @_Zlt8OrderedAS_(i32 %x.coerce, i32 %y.coerce) #4 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x = alloca %struct.OrderedA, align 4
// CHECK:STDOUT:   %y = alloca %struct.OrderedA, align 4
// CHECK:STDOUT:   %coerce.dive = getelementptr inbounds nuw %struct.OrderedA, ptr %x, i32 0, i32 0
// CHECK:STDOUT:   store i32 %x.coerce, ptr %coerce.dive, align 4
// CHECK:STDOUT:   %coerce.dive1 = getelementptr inbounds nuw %struct.OrderedA, ptr %y, i32 0, i32 0
// CHECK:STDOUT:   store i32 %y.coerce, ptr %coerce.dive1, align 4
// CHECK:STDOUT:   %value = getelementptr inbounds nuw %struct.OrderedA, ptr %x, i32 0, i32 0
// CHECK:STDOUT:   %0 = load i32, ptr %value, align 4, !tbaa !120
// CHECK:STDOUT:   %value2 = getelementptr inbounds nuw %struct.OrderedA, ptr %y, i32 0, i32 0
// CHECK:STDOUT:   %1 = load i32, ptr %value2, align 4, !tbaa !120
// CHECK:STDOUT:   %cmp = icmp slt i32 %0, %1
// CHECK:STDOUT:   ret i1 %cmp
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local noundef zeroext i1 @_Zle8OrderedAS_(i32 %x.coerce, i32 %y.coerce) #4 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x = alloca %struct.OrderedA, align 4
// CHECK:STDOUT:   %y = alloca %struct.OrderedA, align 4
// CHECK:STDOUT:   %agg.tmp = alloca %struct.OrderedA, align 4
// CHECK:STDOUT:   %agg.tmp2 = alloca %struct.OrderedA, align 4
// CHECK:STDOUT:   %coerce.dive = getelementptr inbounds nuw %struct.OrderedA, ptr %x, i32 0, i32 0
// CHECK:STDOUT:   store i32 %x.coerce, ptr %coerce.dive, align 4
// CHECK:STDOUT:   %coerce.dive1 = getelementptr inbounds nuw %struct.OrderedA, ptr %y, i32 0, i32 0
// CHECK:STDOUT:   store i32 %y.coerce, ptr %coerce.dive1, align 4
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %y, i64 4, i1 false), !tbaa.struct !58
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp2, ptr align 4 %x, i64 4, i1 false), !tbaa.struct !58
// CHECK:STDOUT:   %coerce.dive3 = getelementptr inbounds nuw %struct.OrderedA, ptr %agg.tmp, i32 0, i32 0
// CHECK:STDOUT:   %0 = load i32, ptr %coerce.dive3, align 4
// CHECK:STDOUT:   %coerce.dive4 = getelementptr inbounds nuw %struct.OrderedA, ptr %agg.tmp2, i32 0, i32 0
// CHECK:STDOUT:   %1 = load i32, ptr %coerce.dive4, align 4
// CHECK:STDOUT:   %call = call noundef zeroext i1 @_Zlt8OrderedAS_(i32 %0, i32 %1)
// CHECK:STDOUT:   %lnot = xor i1 %call, true
// CHECK:STDOUT:   ret i1 %lnot
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local noundef zeroext i1 @_Zgt8OrderedAS_(i32 %x.coerce, i32 %y.coerce) #4 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x = alloca %struct.OrderedA, align 4
// CHECK:STDOUT:   %y = alloca %struct.OrderedA, align 4
// CHECK:STDOUT:   %agg.tmp = alloca %struct.OrderedA, align 4
// CHECK:STDOUT:   %agg.tmp2 = alloca %struct.OrderedA, align 4
// CHECK:STDOUT:   %coerce.dive = getelementptr inbounds nuw %struct.OrderedA, ptr %x, i32 0, i32 0
// CHECK:STDOUT:   store i32 %x.coerce, ptr %coerce.dive, align 4
// CHECK:STDOUT:   %coerce.dive1 = getelementptr inbounds nuw %struct.OrderedA, ptr %y, i32 0, i32 0
// CHECK:STDOUT:   store i32 %y.coerce, ptr %coerce.dive1, align 4
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %y, i64 4, i1 false), !tbaa.struct !58
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp2, ptr align 4 %x, i64 4, i1 false), !tbaa.struct !58
// CHECK:STDOUT:   %coerce.dive3 = getelementptr inbounds nuw %struct.OrderedA, ptr %agg.tmp, i32 0, i32 0
// CHECK:STDOUT:   %0 = load i32, ptr %coerce.dive3, align 4
// CHECK:STDOUT:   %coerce.dive4 = getelementptr inbounds nuw %struct.OrderedA, ptr %agg.tmp2, i32 0, i32 0
// CHECK:STDOUT:   %1 = load i32, ptr %coerce.dive4, align 4
// CHECK:STDOUT:   %call = call noundef zeroext i1 @_Zlt8OrderedAS_(i32 %0, i32 %1)
// CHECK:STDOUT:   ret i1 %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local noundef zeroext i1 @_Zge8OrderedAS_(i32 %x.coerce, i32 %y.coerce) #4 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x = alloca %struct.OrderedA, align 4
// CHECK:STDOUT:   %y = alloca %struct.OrderedA, align 4
// CHECK:STDOUT:   %agg.tmp = alloca %struct.OrderedA, align 4
// CHECK:STDOUT:   %agg.tmp2 = alloca %struct.OrderedA, align 4
// CHECK:STDOUT:   %coerce.dive = getelementptr inbounds nuw %struct.OrderedA, ptr %x, i32 0, i32 0
// CHECK:STDOUT:   store i32 %x.coerce, ptr %coerce.dive, align 4
// CHECK:STDOUT:   %coerce.dive1 = getelementptr inbounds nuw %struct.OrderedA, ptr %y, i32 0, i32 0
// CHECK:STDOUT:   store i32 %y.coerce, ptr %coerce.dive1, align 4
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %x, i64 4, i1 false), !tbaa.struct !58
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp2, ptr align 4 %y, i64 4, i1 false), !tbaa.struct !58
// CHECK:STDOUT:   %coerce.dive3 = getelementptr inbounds nuw %struct.OrderedA, ptr %agg.tmp, i32 0, i32 0
// CHECK:STDOUT:   %0 = load i32, ptr %coerce.dive3, align 4
// CHECK:STDOUT:   %coerce.dive4 = getelementptr inbounds nuw %struct.OrderedA, ptr %agg.tmp2, i32 0, i32 0
// CHECK:STDOUT:   %1 = load i32, ptr %coerce.dive4, align 4
// CHECK:STDOUT:   %call = call noundef zeroext i1 @_Zlt8OrderedAS_(i32 %0, i32 %1)
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!67}
// CHECK:STDOUT: !llvm.module.flags = !{!61, !62, !63, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!65}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{null, !7, !7}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !12 = !{!7, !7, !7}
// CHECK:STDOUT: !13 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !47 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !48 = !{!"omnipotent char", !47, i64 0}
// CHECK:STDOUT: !49 = !{!"any pointer", !48, i64 0}
// CHECK:STDOUT: !52 = !{!"bool", !48, i64 0}
// CHECK:STDOUT: !53 = !{!"p1 bool", !49, i64 0}
// CHECK:STDOUT: !54 = !{!53, !53, i64 0}
// CHECK:STDOUT: !55 = !{!"int", !48, i64 0}
// CHECK:STDOUT: !57 = !{!55, !55, i64 0}
// CHECK:STDOUT: !58 = !{i64 0, i64 4, !57}
// CHECK:STDOUT: !59 = !{!52, !52, i64 0}
// CHECK:STDOUT: !61 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !62 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !63 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !64 = !{!"__libc_errno", !55, i64 0}
// CHECK:STDOUT: !65 = !{!64, !55, i64 0}
// CHECK:STDOUT: !66 = !DIFile(filename: "ordered_with.carbon", directory: "")
// CHECK:STDOUT: !67 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !66, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "Driver", linkageName: "_CDriver.Main", scope: null, file: !66, line: 33, type: !9, spFlags: DISPFlagDefinition, unit: !67, retainedNodes: !77)
// CHECK:STDOUT: !69 = !DILocalVariable(arg: 1, scope: !68, type: !7)
// CHECK:STDOUT: !70 = !DILocalVariable(arg: 2, scope: !68, type: !7)
// CHECK:STDOUT: !75 = !DILocation(line: 34, column: 3, scope: !68)
// CHECK:STDOUT: !76 = !DILocation(line: 33, column: 1, scope: !68)
// CHECK:STDOUT: !77 = !{!69, !70}
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "Less", linkageName: "_CLess:thunk:OrderedWith.f150fc1157a5293b.Core:Cpp", scope: null, file: !66, line: 8, type: !13, spFlags: DISPFlagDefinition, unit: !67, retainedNodes: !83)
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 1, scope: !78, type: !7)
// CHECK:STDOUT: !80 = !DILocalVariable(arg: 2, scope: !78, type: !7)
// CHECK:STDOUT: !82 = !DILocation(line: 8, column: 15, scope: !78)
// CHECK:STDOUT: !83 = !{!79, !80}
// CHECK:STDOUT: !84 = distinct !DISubprogram(name: "LessOrEquivalent", linkageName: "_CLessOrEquivalent:thunk:OrderedWith.f150fc1157a5293b.Core:Cpp", scope: null, file: !66, line: 16, type: !13, spFlags: DISPFlagDefinition, unit: !67, retainedNodes: !89)
// CHECK:STDOUT: !85 = !DILocalVariable(arg: 1, scope: !84, type: !7)
// CHECK:STDOUT: !86 = !DILocalVariable(arg: 2, scope: !84, type: !7)
// CHECK:STDOUT: !88 = !DILocation(line: 16, column: 15, scope: !84)
// CHECK:STDOUT: !89 = !{!85, !86}
// CHECK:STDOUT: !90 = distinct !DISubprogram(name: "Greater", linkageName: "_CGreater:thunk:OrderedWith.f150fc1157a5293b.Core:Cpp", scope: null, file: !66, line: 12, type: !13, spFlags: DISPFlagDefinition, unit: !67, retainedNodes: !95)
// CHECK:STDOUT: !91 = !DILocalVariable(arg: 1, scope: !90, type: !7)
// CHECK:STDOUT: !92 = !DILocalVariable(arg: 2, scope: !90, type: !7)
// CHECK:STDOUT: !94 = !DILocation(line: 12, column: 15, scope: !90)
// CHECK:STDOUT: !95 = !{!91, !92}
// CHECK:STDOUT: !96 = distinct !DISubprogram(name: "GreaterOrEquivalent", linkageName: "_CGreaterOrEquivalent:thunk:OrderedWith.f150fc1157a5293b.Core:Cpp", scope: null, file: !66, line: 20, type: !13, spFlags: DISPFlagDefinition, unit: !67, retainedNodes: !101)
// CHECK:STDOUT: !97 = !DILocalVariable(arg: 1, scope: !96, type: !7)
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 2, scope: !96, type: !7)
// CHECK:STDOUT: !100 = !DILocation(line: 20, column: 15, scope: !96)
// CHECK:STDOUT: !101 = !{!97, !98}
// CHECK:STDOUT: !102 = distinct !DISubprogram(name: "CheckOrderedWith", linkageName: "_CCheckOrderedWith.Main.f3d1df67aae0c1db", scope: null, file: !66, line: 26, type: !9, spFlags: DISPFlagDefinition, unit: !67, retainedNodes: !116)
// CHECK:STDOUT: !103 = !DILocalVariable(arg: 1, scope: !102, type: !7)
// CHECK:STDOUT: !104 = !DILocalVariable(arg: 2, scope: !102, type: !7)
// CHECK:STDOUT: !107 = !DILocation(line: 27, column: 3, scope: !102)
// CHECK:STDOUT: !109 = !DILocation(line: 28, column: 3, scope: !102)
// CHECK:STDOUT: !111 = !DILocation(line: 29, column: 3, scope: !102)
// CHECK:STDOUT: !113 = !DILocation(line: 30, column: 3, scope: !102)
// CHECK:STDOUT: !115 = !DILocation(line: 26, column: 1, scope: !102)
// CHECK:STDOUT: !116 = !{!103, !104}
// CHECK:STDOUT: !117 = !{!"p1 _ZTS8OrderedA", !49, i64 0}
// CHECK:STDOUT: !118 = !{!117, !117, i64 0}
// CHECK:STDOUT: !119 = !{!"_ZTS8OrderedA", !55, i64 0}
// CHECK:STDOUT: !120 = !{!119, !55, i64 0}
// CHECK:STDOUT:
