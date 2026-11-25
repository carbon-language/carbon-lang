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

fn CheckEq[T: Core.Eq](x: T, y: T) {
  x == y;
  x != y;
}

fn Driver(x: Cpp.EqA, y: Cpp.EqA) {
  CheckEqWith(x, y);
  CheckEq(x, y);
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

fn CheckOrdered[T: Core.Ordered](x: T, y: T) {
  x < y;
  x > y;
  x <= y;
  x >= y;
}

fn Driver(x: Cpp.OrderedA, y: Cpp.OrderedA) {
  CheckOrderedWith(x, y);
  CheckOrdered(x, y);
}

// --- add.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
struct AddA {
  int value;

  friend auto operator+(AddA x, AddA y) -> AddA {
    return {x.value + y.value};
  }

  auto operator+=(AddA other) -> void {
    value += other.value;
  }
};
''';

fn CheckAdd[T: Core.Add & Core.AddAssign](ref x: T, y: T) -> T {
  x += y;
  return x + y;
}

fn Driver(ref x: Cpp.AddA, y: Cpp.AddA) -> Cpp.AddA {
  return CheckAdd(ref x, y);
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
// CHECK:STDOUT:   call void @_CCheckEqWith.Main.7152d97258755740(ptr %x, ptr %y), !dbg !21
// CHECK:STDOUT:   call void @_CCheckEq.Main.c3e8becdad130fe4(ptr %x, ptr %y), !dbg !24
// CHECK:STDOUT:   ret void, !dbg !25
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
// CHECK:STDOUT:   store ptr %x, ptr %x.addr, align 8, !tbaa !65
// CHECK:STDOUT:   store ptr %y, ptr %y.addr, align 8, !tbaa !65
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !68
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !68
// CHECK:STDOUT:   %1 = load ptr, ptr %x.addr, align 8, !tbaa !65
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %1, i64 4, i1 false), !tbaa.struct !72
// CHECK:STDOUT:   %2 = load ptr, ptr %y.addr, align 8, !tbaa !65
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp1, ptr align 4 %2, i64 4, i1 false), !tbaa.struct !72
// CHECK:STDOUT:   %coerce.dive = getelementptr inbounds nuw %struct.EqA, ptr %agg.tmp, i32 0, i32 0
// CHECK:STDOUT:   %3 = load i32, ptr %coerce.dive, align 4
// CHECK:STDOUT:   %coerce.dive2 = getelementptr inbounds nuw %struct.EqA, ptr %agg.tmp1, i32 0, i32 0
// CHECK:STDOUT:   %4 = load i32, ptr %coerce.dive2, align 4
// CHECK:STDOUT:   %call = call noundef zeroext i1 @_Zeq3EqAS_(i32 %3, i32 %4)
// CHECK:STDOUT:   %storedv = zext i1 %call to i8
// CHECK:STDOUT:   store i8 %storedv, ptr %0, align 1, !tbaa !73
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
// CHECK:STDOUT:   store ptr %x, ptr %x.addr, align 8, !tbaa !65
// CHECK:STDOUT:   store ptr %y, ptr %y.addr, align 8, !tbaa !65
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !68
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !68
// CHECK:STDOUT:   %1 = load ptr, ptr %x.addr, align 8, !tbaa !65
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %1, i64 4, i1 false), !tbaa.struct !72
// CHECK:STDOUT:   %2 = load ptr, ptr %y.addr, align 8, !tbaa !65
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp1, ptr align 4 %2, i64 4, i1 false), !tbaa.struct !72
// CHECK:STDOUT:   %coerce.dive = getelementptr inbounds nuw %struct.EqA, ptr %agg.tmp, i32 0, i32 0
// CHECK:STDOUT:   %3 = load i32, ptr %coerce.dive, align 4
// CHECK:STDOUT:   %coerce.dive2 = getelementptr inbounds nuw %struct.EqA, ptr %agg.tmp1, i32 0, i32 0
// CHECK:STDOUT:   %4 = load i32, ptr %coerce.dive2, align 4
// CHECK:STDOUT:   %call = call noundef zeroext i1 @_Zne3EqAS_(i32 %3, i32 %4)
// CHECK:STDOUT:   %storedv = zext i1 %call to i8
// CHECK:STDOUT:   store i8 %storedv, ptr %0, align 1, !tbaa !73
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CEqual:thunk:EqWith.5fc20a8665baac18.Core:Cpp"(ptr %self, ptr %other) #2 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.6.temp = alloca i1, align 1, !dbg !31
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.6.temp), !dbg !31
// CHECK:STDOUT:   call void @_Zeq3EqAS_.carbon_thunk.__(ptr %self, ptr %other, ptr %.6.temp), !dbg !31
// CHECK:STDOUT:   %0 = load i1, ptr %.6.temp, align 1, !dbg !31
// CHECK:STDOUT:   ret i1 %0, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CNotEqual:thunk:EqWith.5fc20a8665baac18.Core:Cpp"(ptr %self, ptr %other) #2 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.6.temp = alloca i1, align 1, !dbg !37
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.6.temp), !dbg !37
// CHECK:STDOUT:   call void @_Zne3EqAS_.carbon_thunk.__(ptr %self, ptr %other, ptr %.6.temp), !dbg !37
// CHECK:STDOUT:   %0 = load i1, ptr %.6.temp, align 1, !dbg !37
// CHECK:STDOUT:   ret i1 %0, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCheckEqWith.Main.7152d97258755740(ptr %x, ptr %y) #0 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.call = call i1 @"_CEqual:thunk:EqWith.5fc20a8665baac18.Core:Cpp"(ptr %x, ptr %y), !dbg !44
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.call = call i1 @"_CNotEqual:thunk:EqWith.5fc20a8665baac18.Core:Cpp"(ptr %x, ptr %y), !dbg !46
// CHECK:STDOUT:   ret void, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCheckEq.Main.c3e8becdad130fe4(ptr %x, ptr %y) #0 !dbg !50 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.call = call i1 @"_CEqual:thunk:EqWith.5fc20a8665baac18.Core:Cpp"(ptr %x, ptr %y), !dbg !55
// CHECK:STDOUT:   %EqWith.WithSelf.NotEqual.call = call i1 @"_CNotEqual:thunk:EqWith.5fc20a8665baac18.Core:Cpp"(ptr %x, ptr %y), !dbg !57
// CHECK:STDOUT:   ret void, !dbg !59
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
// CHECK:STDOUT:   %0 = load i32, ptr %value, align 4, !tbaa !74
// CHECK:STDOUT:   %value2 = getelementptr inbounds nuw %struct.EqA, ptr %y, i32 0, i32 0
// CHECK:STDOUT:   %1 = load i32, ptr %value2, align 4, !tbaa !74
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
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %x, i64 4, i1 false), !tbaa.struct !72
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp2, ptr align 4 %y, i64 4, i1 false), !tbaa.struct !72
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
// CHECK:STDOUT: !llvm.module.flags = !{!75, !76, !77, !78, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!80}
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
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "Driver", linkageName: "_CDriver.Main", scope: null, file: !4, line: 28, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !26)
// CHECK:STDOUT: !15 = !DILocalVariable(arg: 1, scope: !14, type: !7)
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 2, scope: !14, type: !7)
// CHECK:STDOUT: !21 = !DILocation(line: 29, column: 3, scope: !14)
// CHECK:STDOUT: !24 = !DILocation(line: 30, column: 3, scope: !14)
// CHECK:STDOUT: !25 = !DILocation(line: 28, column: 1, scope: !14)
// CHECK:STDOUT: !26 = !{!15, !16}
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "Equal", linkageName: "_CEqual:thunk:EqWith.5fc20a8665baac18.Core:Cpp", scope: null, file: !4, line: 8, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !32)
// CHECK:STDOUT: !28 = !DILocalVariable(arg: 1, scope: !27, type: !7)
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 2, scope: !27, type: !7)
// CHECK:STDOUT: !31 = !DILocation(line: 8, column: 15, scope: !27)
// CHECK:STDOUT: !32 = !{!28, !29}
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "NotEqual", linkageName: "_CNotEqual:thunk:EqWith.5fc20a8665baac18.Core:Cpp", scope: null, file: !4, line: 12, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !38)
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 1, scope: !33, type: !7)
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 2, scope: !33, type: !7)
// CHECK:STDOUT: !37 = !DILocation(line: 12, column: 15, scope: !33)
// CHECK:STDOUT: !38 = !{!34, !35}
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "CheckEqWith", linkageName: "_CCheckEqWith.Main.7152d97258755740", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !49)
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !39, type: !7)
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 2, scope: !39, type: !7)
// CHECK:STDOUT: !44 = !DILocation(line: 19, column: 3, scope: !39)
// CHECK:STDOUT: !46 = !DILocation(line: 20, column: 3, scope: !39)
// CHECK:STDOUT: !48 = !DILocation(line: 18, column: 1, scope: !39)
// CHECK:STDOUT: !49 = !{!40, !41}
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "CheckEq", linkageName: "_CCheckEq.Main.c3e8becdad130fe4", scope: null, file: !4, line: 23, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !60)
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !50, type: !7)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 2, scope: !50, type: !7)
// CHECK:STDOUT: !55 = !DILocation(line: 24, column: 3, scope: !50)
// CHECK:STDOUT: !57 = !DILocation(line: 25, column: 3, scope: !50)
// CHECK:STDOUT: !59 = !DILocation(line: 23, column: 1, scope: !50)
// CHECK:STDOUT: !60 = !{!51, !52}
// CHECK:STDOUT: !61 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !62 = !{!"omnipotent char", !61, i64 0}
// CHECK:STDOUT: !63 = !{!"any pointer", !62, i64 0}
// CHECK:STDOUT: !64 = !{!"p1 _ZTS3EqA", !63, i64 0}
// CHECK:STDOUT: !65 = !{!64, !64, i64 0}
// CHECK:STDOUT: !66 = !{!"bool", !62, i64 0}
// CHECK:STDOUT: !67 = !{!"p1 bool", !63, i64 0}
// CHECK:STDOUT: !68 = !{!67, !67, i64 0}
// CHECK:STDOUT: !69 = !{!"int", !62, i64 0}
// CHECK:STDOUT: !70 = !{!"_ZTS3EqA", !69, i64 0}
// CHECK:STDOUT: !71 = !{!69, !69, i64 0}
// CHECK:STDOUT: !72 = !{i64 0, i64 4, !71}
// CHECK:STDOUT: !73 = !{!66, !66, i64 0}
// CHECK:STDOUT: !74 = !{!70, !69, i64 0}
// CHECK:STDOUT: !75 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !76 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !77 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !78 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !79 = !{!"__libc_errno", !69, i64 0}
// CHECK:STDOUT: !80 = !{!79, !69, i64 0}
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
// CHECK:STDOUT: define void @_CDriver.Main(ptr %x, ptr %y) #0 !dbg !83 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CCheckOrderedWith.Main.b948edeb3216062b(ptr %x, ptr %y), !dbg !90
// CHECK:STDOUT:   call void @_CCheckOrdered.Main.6bee2bb7cb1a3b16(ptr %x, ptr %y), !dbg !93
// CHECK:STDOUT:   ret void, !dbg !94
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
// CHECK:STDOUT:   store ptr %x, ptr %x.addr, align 8, !tbaa !151
// CHECK:STDOUT:   store ptr %y, ptr %y.addr, align 8, !tbaa !151
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !68
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !68
// CHECK:STDOUT:   %1 = load ptr, ptr %x.addr, align 8, !tbaa !151
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %1, i64 4, i1 false), !tbaa.struct !72
// CHECK:STDOUT:   %2 = load ptr, ptr %y.addr, align 8, !tbaa !151
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp1, ptr align 4 %2, i64 4, i1 false), !tbaa.struct !72
// CHECK:STDOUT:   %coerce.dive = getelementptr inbounds nuw %struct.OrderedA, ptr %agg.tmp, i32 0, i32 0
// CHECK:STDOUT:   %3 = load i32, ptr %coerce.dive, align 4
// CHECK:STDOUT:   %coerce.dive2 = getelementptr inbounds nuw %struct.OrderedA, ptr %agg.tmp1, i32 0, i32 0
// CHECK:STDOUT:   %4 = load i32, ptr %coerce.dive2, align 4
// CHECK:STDOUT:   %call = call noundef zeroext i1 @_Zlt8OrderedAS_(i32 %3, i32 %4)
// CHECK:STDOUT:   %storedv = zext i1 %call to i8
// CHECK:STDOUT:   store i8 %storedv, ptr %0, align 1, !tbaa !73
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
// CHECK:STDOUT:   store ptr %x, ptr %x.addr, align 8, !tbaa !151
// CHECK:STDOUT:   store ptr %y, ptr %y.addr, align 8, !tbaa !151
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !68
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !68
// CHECK:STDOUT:   %1 = load ptr, ptr %x.addr, align 8, !tbaa !151
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %1, i64 4, i1 false), !tbaa.struct !72
// CHECK:STDOUT:   %2 = load ptr, ptr %y.addr, align 8, !tbaa !151
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp1, ptr align 4 %2, i64 4, i1 false), !tbaa.struct !72
// CHECK:STDOUT:   %coerce.dive = getelementptr inbounds nuw %struct.OrderedA, ptr %agg.tmp, i32 0, i32 0
// CHECK:STDOUT:   %3 = load i32, ptr %coerce.dive, align 4
// CHECK:STDOUT:   %coerce.dive2 = getelementptr inbounds nuw %struct.OrderedA, ptr %agg.tmp1, i32 0, i32 0
// CHECK:STDOUT:   %4 = load i32, ptr %coerce.dive2, align 4
// CHECK:STDOUT:   %call = call noundef zeroext i1 @_Zle8OrderedAS_(i32 %3, i32 %4)
// CHECK:STDOUT:   %storedv = zext i1 %call to i8
// CHECK:STDOUT:   store i8 %storedv, ptr %0, align 1, !tbaa !73
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
// CHECK:STDOUT:   store ptr %x, ptr %x.addr, align 8, !tbaa !151
// CHECK:STDOUT:   store ptr %y, ptr %y.addr, align 8, !tbaa !151
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !68
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !68
// CHECK:STDOUT:   %1 = load ptr, ptr %x.addr, align 8, !tbaa !151
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %1, i64 4, i1 false), !tbaa.struct !72
// CHECK:STDOUT:   %2 = load ptr, ptr %y.addr, align 8, !tbaa !151
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp1, ptr align 4 %2, i64 4, i1 false), !tbaa.struct !72
// CHECK:STDOUT:   %coerce.dive = getelementptr inbounds nuw %struct.OrderedA, ptr %agg.tmp, i32 0, i32 0
// CHECK:STDOUT:   %3 = load i32, ptr %coerce.dive, align 4
// CHECK:STDOUT:   %coerce.dive2 = getelementptr inbounds nuw %struct.OrderedA, ptr %agg.tmp1, i32 0, i32 0
// CHECK:STDOUT:   %4 = load i32, ptr %coerce.dive2, align 4
// CHECK:STDOUT:   %call = call noundef zeroext i1 @_Zgt8OrderedAS_(i32 %3, i32 %4)
// CHECK:STDOUT:   %storedv = zext i1 %call to i8
// CHECK:STDOUT:   store i8 %storedv, ptr %0, align 1, !tbaa !73
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
// CHECK:STDOUT:   store ptr %x, ptr %x.addr, align 8, !tbaa !151
// CHECK:STDOUT:   store ptr %y, ptr %y.addr, align 8, !tbaa !151
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !68
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !68
// CHECK:STDOUT:   %1 = load ptr, ptr %x.addr, align 8, !tbaa !151
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %1, i64 4, i1 false), !tbaa.struct !72
// CHECK:STDOUT:   %2 = load ptr, ptr %y.addr, align 8, !tbaa !151
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp1, ptr align 4 %2, i64 4, i1 false), !tbaa.struct !72
// CHECK:STDOUT:   %coerce.dive = getelementptr inbounds nuw %struct.OrderedA, ptr %agg.tmp, i32 0, i32 0
// CHECK:STDOUT:   %3 = load i32, ptr %coerce.dive, align 4
// CHECK:STDOUT:   %coerce.dive2 = getelementptr inbounds nuw %struct.OrderedA, ptr %agg.tmp1, i32 0, i32 0
// CHECK:STDOUT:   %4 = load i32, ptr %coerce.dive2, align 4
// CHECK:STDOUT:   %call = call noundef zeroext i1 @_Zge8OrderedAS_(i32 %3, i32 %4)
// CHECK:STDOUT:   %storedv = zext i1 %call to i8
// CHECK:STDOUT:   store i8 %storedv, ptr %0, align 1, !tbaa !73
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CLess:thunk:OrderedWith.f3924169c098de0f.Core:Cpp"(ptr %self, ptr %other) #2 !dbg !96 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.6.temp = alloca i1, align 1, !dbg !100
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.6.temp), !dbg !100
// CHECK:STDOUT:   call void @_Zlt8OrderedAS_.carbon_thunk.__(ptr %self, ptr %other, ptr %.6.temp), !dbg !100
// CHECK:STDOUT:   %0 = load i1, ptr %.6.temp, align 1, !dbg !100
// CHECK:STDOUT:   ret i1 %0, !dbg !100
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CLessOrEquivalent:thunk:OrderedWith.f3924169c098de0f.Core:Cpp"(ptr %self, ptr %other) #2 !dbg !102 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.6.temp = alloca i1, align 1, !dbg !106
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.6.temp), !dbg !106
// CHECK:STDOUT:   call void @_Zle8OrderedAS_.carbon_thunk.__(ptr %self, ptr %other, ptr %.6.temp), !dbg !106
// CHECK:STDOUT:   %0 = load i1, ptr %.6.temp, align 1, !dbg !106
// CHECK:STDOUT:   ret i1 %0, !dbg !106
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CGreater:thunk:OrderedWith.f3924169c098de0f.Core:Cpp"(ptr %self, ptr %other) #2 !dbg !108 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.6.temp = alloca i1, align 1, !dbg !112
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.6.temp), !dbg !112
// CHECK:STDOUT:   call void @_Zgt8OrderedAS_.carbon_thunk.__(ptr %self, ptr %other, ptr %.6.temp), !dbg !112
// CHECK:STDOUT:   %0 = load i1, ptr %.6.temp, align 1, !dbg !112
// CHECK:STDOUT:   ret i1 %0, !dbg !112
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CGreaterOrEquivalent:thunk:OrderedWith.f3924169c098de0f.Core:Cpp"(ptr %self, ptr %other) #2 !dbg !114 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.6.temp = alloca i1, align 1, !dbg !118
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.6.temp), !dbg !118
// CHECK:STDOUT:   call void @_Zge8OrderedAS_.carbon_thunk.__(ptr %self, ptr %other, ptr %.6.temp), !dbg !118
// CHECK:STDOUT:   %0 = load i1, ptr %.6.temp, align 1, !dbg !118
// CHECK:STDOUT:   ret i1 %0, !dbg !118
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCheckOrderedWith.Main.b948edeb3216062b(ptr %x, ptr %y) #0 !dbg !120 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.call = call i1 @"_CLess:thunk:OrderedWith.f3924169c098de0f.Core:Cpp"(ptr %x, ptr %y), !dbg !125
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.call = call i1 @"_CGreater:thunk:OrderedWith.f3924169c098de0f.Core:Cpp"(ptr %x, ptr %y), !dbg !127
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.call = call i1 @"_CLessOrEquivalent:thunk:OrderedWith.f3924169c098de0f.Core:Cpp"(ptr %x, ptr %y), !dbg !129
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.call = call i1 @"_CGreaterOrEquivalent:thunk:OrderedWith.f3924169c098de0f.Core:Cpp"(ptr %x, ptr %y), !dbg !131
// CHECK:STDOUT:   ret void, !dbg !133
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCheckOrdered.Main.6bee2bb7cb1a3b16(ptr %x, ptr %y) #0 !dbg !135 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %OrderedWith.WithSelf.Less.call = call i1 @"_CLess:thunk:OrderedWith.f3924169c098de0f.Core:Cpp"(ptr %x, ptr %y), !dbg !140
// CHECK:STDOUT:   %OrderedWith.WithSelf.Greater.call = call i1 @"_CGreater:thunk:OrderedWith.f3924169c098de0f.Core:Cpp"(ptr %x, ptr %y), !dbg !142
// CHECK:STDOUT:   %OrderedWith.WithSelf.LessOrEquivalent.call = call i1 @"_CLessOrEquivalent:thunk:OrderedWith.f3924169c098de0f.Core:Cpp"(ptr %x, ptr %y), !dbg !144
// CHECK:STDOUT:   %OrderedWith.WithSelf.GreaterOrEquivalent.call = call i1 @"_CGreaterOrEquivalent:thunk:OrderedWith.f3924169c098de0f.Core:Cpp"(ptr %x, ptr %y), !dbg !146
// CHECK:STDOUT:   ret void, !dbg !148
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
// CHECK:STDOUT:   %0 = load i32, ptr %value, align 4, !tbaa !153
// CHECK:STDOUT:   %value2 = getelementptr inbounds nuw %struct.OrderedA, ptr %y, i32 0, i32 0
// CHECK:STDOUT:   %1 = load i32, ptr %value2, align 4, !tbaa !153
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
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %y, i64 4, i1 false), !tbaa.struct !72
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp2, ptr align 4 %x, i64 4, i1 false), !tbaa.struct !72
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
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %y, i64 4, i1 false), !tbaa.struct !72
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp2, ptr align 4 %x, i64 4, i1 false), !tbaa.struct !72
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
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %x, i64 4, i1 false), !tbaa.struct !72
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp2, ptr align 4 %y, i64 4, i1 false), !tbaa.struct !72
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!82}
// CHECK:STDOUT: !llvm.module.flags = !{!75, !76, !77, !78, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!80}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{null, !7, !7}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !12 = !{!7, !7, !7}
// CHECK:STDOUT: !13 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !61 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !62 = !{!"omnipotent char", !61, i64 0}
// CHECK:STDOUT: !63 = !{!"any pointer", !62, i64 0}
// CHECK:STDOUT: !66 = !{!"bool", !62, i64 0}
// CHECK:STDOUT: !67 = !{!"p1 bool", !63, i64 0}
// CHECK:STDOUT: !68 = !{!67, !67, i64 0}
// CHECK:STDOUT: !69 = !{!"int", !62, i64 0}
// CHECK:STDOUT: !71 = !{!69, !69, i64 0}
// CHECK:STDOUT: !72 = !{i64 0, i64 4, !71}
// CHECK:STDOUT: !73 = !{!66, !66, i64 0}
// CHECK:STDOUT: !75 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !76 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !77 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !78 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !79 = !{!"__libc_errno", !69, i64 0}
// CHECK:STDOUT: !80 = !{!79, !69, i64 0}
// CHECK:STDOUT: !81 = !DIFile(filename: "ordered_with.carbon", directory: "")
// CHECK:STDOUT: !82 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !81, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !83 = distinct !DISubprogram(name: "Driver", linkageName: "_CDriver.Main", scope: null, file: !81, line: 40, type: !9, spFlags: DISPFlagDefinition, unit: !82, retainedNodes: !95)
// CHECK:STDOUT: !84 = !DILocalVariable(arg: 1, scope: !83, type: !7)
// CHECK:STDOUT: !85 = !DILocalVariable(arg: 2, scope: !83, type: !7)
// CHECK:STDOUT: !90 = !DILocation(line: 41, column: 3, scope: !83)
// CHECK:STDOUT: !93 = !DILocation(line: 42, column: 3, scope: !83)
// CHECK:STDOUT: !94 = !DILocation(line: 40, column: 1, scope: !83)
// CHECK:STDOUT: !95 = !{!84, !85}
// CHECK:STDOUT: !96 = distinct !DISubprogram(name: "Less", linkageName: "_CLess:thunk:OrderedWith.f3924169c098de0f.Core:Cpp", scope: null, file: !81, line: 8, type: !13, spFlags: DISPFlagDefinition, unit: !82, retainedNodes: !101)
// CHECK:STDOUT: !97 = !DILocalVariable(arg: 1, scope: !96, type: !7)
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 2, scope: !96, type: !7)
// CHECK:STDOUT: !100 = !DILocation(line: 8, column: 15, scope: !96)
// CHECK:STDOUT: !101 = !{!97, !98}
// CHECK:STDOUT: !102 = distinct !DISubprogram(name: "LessOrEquivalent", linkageName: "_CLessOrEquivalent:thunk:OrderedWith.f3924169c098de0f.Core:Cpp", scope: null, file: !81, line: 16, type: !13, spFlags: DISPFlagDefinition, unit: !82, retainedNodes: !107)
// CHECK:STDOUT: !103 = !DILocalVariable(arg: 1, scope: !102, type: !7)
// CHECK:STDOUT: !104 = !DILocalVariable(arg: 2, scope: !102, type: !7)
// CHECK:STDOUT: !106 = !DILocation(line: 16, column: 15, scope: !102)
// CHECK:STDOUT: !107 = !{!103, !104}
// CHECK:STDOUT: !108 = distinct !DISubprogram(name: "Greater", linkageName: "_CGreater:thunk:OrderedWith.f3924169c098de0f.Core:Cpp", scope: null, file: !81, line: 12, type: !13, spFlags: DISPFlagDefinition, unit: !82, retainedNodes: !113)
// CHECK:STDOUT: !109 = !DILocalVariable(arg: 1, scope: !108, type: !7)
// CHECK:STDOUT: !110 = !DILocalVariable(arg: 2, scope: !108, type: !7)
// CHECK:STDOUT: !112 = !DILocation(line: 12, column: 15, scope: !108)
// CHECK:STDOUT: !113 = !{!109, !110}
// CHECK:STDOUT: !114 = distinct !DISubprogram(name: "GreaterOrEquivalent", linkageName: "_CGreaterOrEquivalent:thunk:OrderedWith.f3924169c098de0f.Core:Cpp", scope: null, file: !81, line: 20, type: !13, spFlags: DISPFlagDefinition, unit: !82, retainedNodes: !119)
// CHECK:STDOUT: !115 = !DILocalVariable(arg: 1, scope: !114, type: !7)
// CHECK:STDOUT: !116 = !DILocalVariable(arg: 2, scope: !114, type: !7)
// CHECK:STDOUT: !118 = !DILocation(line: 20, column: 15, scope: !114)
// CHECK:STDOUT: !119 = !{!115, !116}
// CHECK:STDOUT: !120 = distinct !DISubprogram(name: "CheckOrderedWith", linkageName: "_CCheckOrderedWith.Main.b948edeb3216062b", scope: null, file: !81, line: 26, type: !9, spFlags: DISPFlagDefinition, unit: !82, retainedNodes: !134)
// CHECK:STDOUT: !121 = !DILocalVariable(arg: 1, scope: !120, type: !7)
// CHECK:STDOUT: !122 = !DILocalVariable(arg: 2, scope: !120, type: !7)
// CHECK:STDOUT: !125 = !DILocation(line: 27, column: 3, scope: !120)
// CHECK:STDOUT: !127 = !DILocation(line: 28, column: 3, scope: !120)
// CHECK:STDOUT: !129 = !DILocation(line: 29, column: 3, scope: !120)
// CHECK:STDOUT: !131 = !DILocation(line: 30, column: 3, scope: !120)
// CHECK:STDOUT: !133 = !DILocation(line: 26, column: 1, scope: !120)
// CHECK:STDOUT: !134 = !{!121, !122}
// CHECK:STDOUT: !135 = distinct !DISubprogram(name: "CheckOrdered", linkageName: "_CCheckOrdered.Main.6bee2bb7cb1a3b16", scope: null, file: !81, line: 33, type: !9, spFlags: DISPFlagDefinition, unit: !82, retainedNodes: !149)
// CHECK:STDOUT: !136 = !DILocalVariable(arg: 1, scope: !135, type: !7)
// CHECK:STDOUT: !137 = !DILocalVariable(arg: 2, scope: !135, type: !7)
// CHECK:STDOUT: !140 = !DILocation(line: 34, column: 3, scope: !135)
// CHECK:STDOUT: !142 = !DILocation(line: 35, column: 3, scope: !135)
// CHECK:STDOUT: !144 = !DILocation(line: 36, column: 3, scope: !135)
// CHECK:STDOUT: !146 = !DILocation(line: 37, column: 3, scope: !135)
// CHECK:STDOUT: !148 = !DILocation(line: 33, column: 1, scope: !135)
// CHECK:STDOUT: !149 = !{!136, !137}
// CHECK:STDOUT: !150 = !{!"p1 _ZTS8OrderedA", !63, i64 0}
// CHECK:STDOUT: !151 = !{!150, !150, i64 0}
// CHECK:STDOUT: !152 = !{!"_ZTS8OrderedA", !69, i64 0}
// CHECK:STDOUT: !153 = !{!152, !69, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'add.carbon'
// CHECK:STDOUT: source_filename = "add.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %struct.AddA = type { i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: $_Zpl4AddAS_ = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN4AddApLES_ = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CDriver.Main(ptr sret([4 x i8]) %return, ptr %x, ptr %y) #0 !dbg !156 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CCheckAdd.Main.6fc848e265e5367e(ptr %return, ptr %x, ptr %y), !dbg !166
// CHECK:STDOUT:   ret void, !dbg !167
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_Zpl4AddAS_.carbon_thunk.__(ptr noundef %x, ptr noundef %y, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %x.addr = alloca ptr, align 8
// CHECK:STDOUT:   %y.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %struct.AddA, align 4
// CHECK:STDOUT:   %agg.tmp1 = alloca %struct.AddA, align 4
// CHECK:STDOUT:   store ptr %x, ptr %x.addr, align 8, !tbaa !195
// CHECK:STDOUT:   store ptr %y, ptr %y.addr, align 8, !tbaa !195
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !195
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !195
// CHECK:STDOUT:   %1 = load ptr, ptr %x.addr, align 8, !tbaa !195
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %1, i64 4, i1 false), !tbaa.struct !72
// CHECK:STDOUT:   %2 = load ptr, ptr %y.addr, align 8, !tbaa !195
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp1, ptr align 4 %2, i64 4, i1 false), !tbaa.struct !72
// CHECK:STDOUT:   %coerce.dive = getelementptr inbounds nuw %struct.AddA, ptr %agg.tmp, i32 0, i32 0
// CHECK:STDOUT:   %3 = load i32, ptr %coerce.dive, align 4
// CHECK:STDOUT:   %coerce.dive2 = getelementptr inbounds nuw %struct.AddA, ptr %agg.tmp1, i32 0, i32 0
// CHECK:STDOUT:   %4 = load i32, ptr %coerce.dive2, align 4
// CHECK:STDOUT:   %call = call i32 @_Zpl4AddAS_(i32 %3, i32 %4)
// CHECK:STDOUT:   %coerce.dive3 = getelementptr inbounds nuw %struct.AddA, ptr %0, i32 0, i32 0
// CHECK:STDOUT:   store i32 %call, ptr %coerce.dive3, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:AddWith.aff3be28998713d0.Core:Cpp"(ptr sret([4 x i8]) %return, ptr %self, ptr %other) #2 !dbg !169 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Zpl4AddAS_.carbon_thunk.__(ptr %self, ptr %other, ptr %return), !dbg !173
// CHECK:STDOUT:   ret void, !dbg !173
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN4AddApLES_.carbon_thunk.__(ptr noundef nonnull align 4 dereferenceable(4) %this, ptr noundef %other) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %other.addr = alloca ptr, align 8
// CHECK:STDOUT:   %agg.tmp = alloca %struct.AddA, align 4
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !195
// CHECK:STDOUT:   store ptr %other, ptr %other.addr, align 8, !tbaa !195
// CHECK:STDOUT:   %0 = load ptr, ptr %this.addr, align 8, !tbaa !195, !nonnull !198, !align !199
// CHECK:STDOUT:   %1 = load ptr, ptr %other.addr, align 8, !tbaa !195
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %1, i64 4, i1 false), !tbaa.struct !72
// CHECK:STDOUT:   %coerce.dive = getelementptr inbounds nuw %struct.AddA, ptr %agg.tmp, i32 0, i32 0
// CHECK:STDOUT:   %2 = load i32, ptr %coerce.dive, align 4
// CHECK:STDOUT:   call void @_ZN4AddApLES_(ptr noundef nonnull align 4 dereferenceable(4) %0, i32 %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:AddAssignWith.e51804e58e7431fe.Core:AddA.Cpp"(ptr %self, ptr %other) #2 !dbg !175 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN4AddApLES_.carbon_thunk.__(ptr %self, ptr %other), !dbg !179
// CHECK:STDOUT:   ret void, !dbg !179
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCheckAdd.Main.6fc848e265e5367e(ptr sret([4 x i8]) %return, ptr %x, ptr %y) #0 !dbg !181 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp:thunk:AddAssignWith.e51804e58e7431fe.Core:AddA.Cpp"(ptr %x, ptr %y), !dbg !188
// CHECK:STDOUT:   call void @"_COp:thunk:AddWith.aff3be28998713d0.Core:Cpp"(ptr %return, ptr %x, ptr %y), !dbg !190
// CHECK:STDOUT:   ret void, !dbg !192
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local i32 @_Zpl4AddAS_(i32 %x.coerce, i32 %y.coerce) #3 comdat {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %retval = alloca %struct.AddA, align 4
// CHECK:STDOUT:   %x = alloca %struct.AddA, align 4
// CHECK:STDOUT:   %y = alloca %struct.AddA, align 4
// CHECK:STDOUT:   %coerce.dive = getelementptr inbounds nuw %struct.AddA, ptr %x, i32 0, i32 0
// CHECK:STDOUT:   store i32 %x.coerce, ptr %coerce.dive, align 4
// CHECK:STDOUT:   %coerce.dive1 = getelementptr inbounds nuw %struct.AddA, ptr %y, i32 0, i32 0
// CHECK:STDOUT:   store i32 %y.coerce, ptr %coerce.dive1, align 4
// CHECK:STDOUT:   %value = getelementptr inbounds nuw %struct.AddA, ptr %retval, i32 0, i32 0
// CHECK:STDOUT:   %value2 = getelementptr inbounds nuw %struct.AddA, ptr %x, i32 0, i32 0
// CHECK:STDOUT:   %0 = load i32, ptr %value2, align 4, !tbaa !197
// CHECK:STDOUT:   %value3 = getelementptr inbounds nuw %struct.AddA, ptr %y, i32 0, i32 0
// CHECK:STDOUT:   %1 = load i32, ptr %value3, align 4, !tbaa !197
// CHECK:STDOUT:   %add = add nsw i32 %0, %1
// CHECK:STDOUT:   store i32 %add, ptr %value, align 4, !tbaa !197
// CHECK:STDOUT:   %coerce.dive4 = getelementptr inbounds nuw %struct.AddA, ptr %retval, i32 0, i32 0
// CHECK:STDOUT:   %2 = load i32, ptr %coerce.dive4, align 4
// CHECK:STDOUT:   ret i32 %2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN4AddApLES_(ptr noundef nonnull align 4 dereferenceable(4) %this, i32 %other.coerce) #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %other = alloca %struct.AddA, align 4
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %coerce.dive = getelementptr inbounds nuw %struct.AddA, ptr %other, i32 0, i32 0
// CHECK:STDOUT:   store i32 %other.coerce, ptr %coerce.dive, align 4
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !195
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   %value = getelementptr inbounds nuw %struct.AddA, ptr %other, i32 0, i32 0
// CHECK:STDOUT:   %0 = load i32, ptr %value, align 4, !tbaa !197
// CHECK:STDOUT:   %value2 = getelementptr inbounds nuw %struct.AddA, ptr %this1, i32 0, i32 0
// CHECK:STDOUT:   %1 = load i32, ptr %value2, align 4, !tbaa !197
// CHECK:STDOUT:   %add = add nsw i32 %1, %0
// CHECK:STDOUT:   store i32 %add, ptr %value2, align 4, !tbaa !197
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #3 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #4 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!155}
// CHECK:STDOUT: !llvm.module.flags = !{!75, !76, !77, !78, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!80}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{null, !7, !7}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !12 = !{!7, !7, !7}
// CHECK:STDOUT: !13 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !61 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !62 = !{!"omnipotent char", !61, i64 0}
// CHECK:STDOUT: !63 = !{!"any pointer", !62, i64 0}
// CHECK:STDOUT: !69 = !{!"int", !62, i64 0}
// CHECK:STDOUT: !71 = !{!69, !69, i64 0}
// CHECK:STDOUT: !72 = !{i64 0, i64 4, !71}
// CHECK:STDOUT: !75 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !76 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !77 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !78 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !79 = !{!"__libc_errno", !69, i64 0}
// CHECK:STDOUT: !80 = !{!79, !69, i64 0}
// CHECK:STDOUT: !154 = !DIFile(filename: "add.carbon", directory: "")
// CHECK:STDOUT: !155 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !154, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !156 = distinct !DISubprogram(name: "Driver", linkageName: "_CDriver.Main", scope: null, file: !154, line: 23, type: !13, spFlags: DISPFlagDefinition, unit: !155, retainedNodes: !168)
// CHECK:STDOUT: !157 = !DILocalVariable(arg: 1, scope: !156, type: !7)
// CHECK:STDOUT: !158 = !DILocalVariable(arg: 2, scope: !156, type: !7)
// CHECK:STDOUT: !166 = !DILocation(line: 24, column: 10, scope: !156)
// CHECK:STDOUT: !167 = !DILocation(line: 24, column: 3, scope: !156)
// CHECK:STDOUT: !168 = !{!157, !158}
// CHECK:STDOUT: !169 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:AddWith.aff3be28998713d0.Core:Cpp", scope: null, file: !154, line: 8, type: !13, spFlags: DISPFlagDefinition, unit: !155, retainedNodes: !174)
// CHECK:STDOUT: !170 = !DILocalVariable(arg: 1, scope: !169, type: !7)
// CHECK:STDOUT: !171 = !DILocalVariable(arg: 2, scope: !169, type: !7)
// CHECK:STDOUT: !173 = !DILocation(line: 8, column: 15, scope: !169)
// CHECK:STDOUT: !174 = !{!170, !171}
// CHECK:STDOUT: !175 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:AddAssignWith.e51804e58e7431fe.Core:AddA.Cpp", scope: null, file: !154, line: 12, type: !9, spFlags: DISPFlagDefinition, unit: !155, retainedNodes: !180)
// CHECK:STDOUT: !176 = !DILocalVariable(arg: 1, scope: !175, type: !7)
// CHECK:STDOUT: !177 = !DILocalVariable(arg: 2, scope: !175, type: !7)
// CHECK:STDOUT: !179 = !DILocation(line: 12, column: 8, scope: !175)
// CHECK:STDOUT: !180 = !{!176, !177}
// CHECK:STDOUT: !181 = distinct !DISubprogram(name: "CheckAdd", linkageName: "_CCheckAdd.Main.6fc848e265e5367e", scope: null, file: !154, line: 18, type: !13, spFlags: DISPFlagDefinition, unit: !155, retainedNodes: !193)
// CHECK:STDOUT: !182 = !DILocalVariable(arg: 1, scope: !181, type: !7)
// CHECK:STDOUT: !183 = !DILocalVariable(arg: 2, scope: !181, type: !7)
// CHECK:STDOUT: !188 = !DILocation(line: 19, column: 3, scope: !181)
// CHECK:STDOUT: !190 = !DILocation(line: 20, column: 10, scope: !181)
// CHECK:STDOUT: !192 = !DILocation(line: 20, column: 3, scope: !181)
// CHECK:STDOUT: !193 = !{!182, !183}
// CHECK:STDOUT: !194 = !{!"p1 _ZTS4AddA", !63, i64 0}
// CHECK:STDOUT: !195 = !{!194, !194, i64 0}
// CHECK:STDOUT: !196 = !{!"_ZTS4AddA", !69, i64 0}
// CHECK:STDOUT: !197 = !{!196, !69, i64 0}
// CHECK:STDOUT: !198 = !{}
// CHECK:STDOUT: !199 = !{i64 4}
// CHECK:STDOUT:
