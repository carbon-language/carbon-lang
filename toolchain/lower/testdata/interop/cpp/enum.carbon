// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/enum.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/enum.carbon

// --- enum_member.h

struct C {
  enum E : short {
    a,
    b = 7,
    c
  };

  static void F(E);
};

// --- pass_as_arg.carbon

library "[[@TEST_NAME]]";

import Cpp library "enum_member.h";

fn Pass() {
  Cpp.C.F(Cpp.C.a);
  Cpp.C.F(Cpp.C.a);
  Cpp.C.F(Cpp.C.b);
  Cpp.C.F(Cpp.C.c);
  Cpp.C.F(Cpp.C.E.a);
  Cpp.C.F(Cpp.C.E.b);
  Cpp.C.F(Cpp.C.E.c);
}

// --- convert.carbon

library "[[@TEST_NAME]]";

import Cpp library "enum_member.h";

fn TakeI16(n: i16);

fn PassEnum() {
  TakeI16(Cpp.C.b as i16);
}

fn ConvertEnumToI16(e: Cpp.C.E) {
  TakeI16(e as i16);
}

fn ConvertI16ToEnum(n: i16) {
  Cpp.C.F(n as Cpp.C.E);
}

// --- bitmask.h

enum Bits {
  A = 1 << 0,
  B = 1 << 1,
  C = 1 << 2,
};

void Take(Bits b);

// --- use_bitmask.carbon

library "[[@TEST_NAME]]";

import Cpp library "bitmask.h";

// TODO: Once we support interop with C++-defined operators, we should be able
// to use `|` below rather than declaring our own builtin.
fn BitOr(a: Cpp.Bits, b: Cpp.Bits) -> Cpp.Bits = "int.or";

fn Call() {
  Cpp.Take(BitOr(Cpp.A, Cpp.C));
}

// CHECK:STDOUT: ; ModuleID = 'pass_as_arg.carbon'
// CHECK:STDOUT: source_filename = "pass_as_arg.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPass.Main() #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_16.1.temp = alloca i16, align 2, !dbg !15
// CHECK:STDOUT:   %.loc8_16.1.temp = alloca i16, align 2, !dbg !16
// CHECK:STDOUT:   %.loc9_16.1.temp = alloca i16, align 2, !dbg !17
// CHECK:STDOUT:   %.loc10_16.1.temp = alloca i16, align 2, !dbg !18
// CHECK:STDOUT:   %.loc11_18.1.temp = alloca i16, align 2, !dbg !19
// CHECK:STDOUT:   %.loc12_18.1.temp = alloca i16, align 2, !dbg !20
// CHECK:STDOUT:   %.loc13_18.1.temp = alloca i16, align 2, !dbg !21
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_16.1.temp), !dbg !15
// CHECK:STDOUT:   store i16 0, ptr %.loc7_16.1.temp, align 2, !dbg !15
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk(ptr %.loc7_16.1.temp), !dbg !22
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc8_16.1.temp), !dbg !16
// CHECK:STDOUT:   store i16 0, ptr %.loc8_16.1.temp, align 2, !dbg !16
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk(ptr %.loc8_16.1.temp), !dbg !23
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc9_16.1.temp), !dbg !17
// CHECK:STDOUT:   store i16 7, ptr %.loc9_16.1.temp, align 2, !dbg !17
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk(ptr %.loc9_16.1.temp), !dbg !24
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc10_16.1.temp), !dbg !18
// CHECK:STDOUT:   store i16 8, ptr %.loc10_16.1.temp, align 2, !dbg !18
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk(ptr %.loc10_16.1.temp), !dbg !25
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc11_18.1.temp), !dbg !19
// CHECK:STDOUT:   store i16 0, ptr %.loc11_18.1.temp, align 2, !dbg !19
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk(ptr %.loc11_18.1.temp), !dbg !26
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc12_18.1.temp), !dbg !20
// CHECK:STDOUT:   store i16 7, ptr %.loc12_18.1.temp, align 2, !dbg !20
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk(ptr %.loc12_18.1.temp), !dbg !27
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc13_18.1.temp), !dbg !21
// CHECK:STDOUT:   store i16 8, ptr %.loc13_18.1.temp, align 2, !dbg !21
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk(ptr %.loc13_18.1.temp), !dbg !28
// CHECK:STDOUT:   ret void, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define dso_local void @_ZN1C1FENS_1EE.carbon_thunk(ptr noundef %0) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !30
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !30
// CHECK:STDOUT:   %2 = load i16, ptr %1, align 2, !tbaa !32
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE(i16 noundef signext %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN1C1FENS_1EE(i16 noundef signext) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4, !5}
// CHECK:STDOUT: !llvm.dbg.cu = !{!6}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!8}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 1, !"wchar_size", i32 4}
// CHECK:STDOUT: !3 = !{i32 8, !"PIC Level", i32 0}
// CHECK:STDOUT: !4 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !5 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !6 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !7, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !7 = !DIFile(filename: "pass_as_arg.carbon", directory: "")
// CHECK:STDOUT: !8 = !{!9, !9, i64 0}
// CHECK:STDOUT: !9 = !{!"int", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"omnipotent char", !11, i64 0}
// CHECK:STDOUT: !11 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Pass", linkageName: "_CPass.Main", scope: null, file: !7, line: 6, type: !13, spFlags: DISPFlagDefinition, unit: !6)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{null}
// CHECK:STDOUT: !15 = !DILocation(line: 7, column: 11, scope: !12)
// CHECK:STDOUT: !16 = !DILocation(line: 8, column: 11, scope: !12)
// CHECK:STDOUT: !17 = !DILocation(line: 9, column: 11, scope: !12)
// CHECK:STDOUT: !18 = !DILocation(line: 10, column: 11, scope: !12)
// CHECK:STDOUT: !19 = !DILocation(line: 11, column: 11, scope: !12)
// CHECK:STDOUT: !20 = !DILocation(line: 12, column: 11, scope: !12)
// CHECK:STDOUT: !21 = !DILocation(line: 13, column: 11, scope: !12)
// CHECK:STDOUT: !22 = !DILocation(line: 7, column: 3, scope: !12)
// CHECK:STDOUT: !23 = !DILocation(line: 8, column: 3, scope: !12)
// CHECK:STDOUT: !24 = !DILocation(line: 9, column: 3, scope: !12)
// CHECK:STDOUT: !25 = !DILocation(line: 10, column: 3, scope: !12)
// CHECK:STDOUT: !26 = !DILocation(line: 11, column: 3, scope: !12)
// CHECK:STDOUT: !27 = !DILocation(line: 12, column: 3, scope: !12)
// CHECK:STDOUT: !28 = !DILocation(line: 13, column: 3, scope: !12)
// CHECK:STDOUT: !29 = !DILocation(line: 6, column: 1, scope: !12)
// CHECK:STDOUT: !30 = !{!31, !31, i64 0}
// CHECK:STDOUT: !31 = !{!"any pointer", !10, i64 0}
// CHECK:STDOUT: !32 = !{!33, !33, i64 0}
// CHECK:STDOUT: !33 = !{!"_ZTSN1C1EE", !10, i64 0}
// CHECK:STDOUT: ; ModuleID = 'convert.carbon'
// CHECK:STDOUT: source_filename = "convert.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CTakeI16.Main(i16)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassEnum.Main() #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CTakeI16.Main(i16 7), !dbg !15
// CHECK:STDOUT:   ret void, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertEnumToI16.Main(i16 %e) #0 !dbg !17 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CTakeI16.Main(i16 %e), !dbg !23
// CHECK:STDOUT:   ret void, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertI16ToEnum.Main(i16 %n) #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17_13.3.temp = alloca i16, align 2, !dbg !28
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc17_13.3.temp), !dbg !28
// CHECK:STDOUT:   store i16 %n, ptr %.loc17_13.3.temp, align 2, !dbg !28
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk(ptr %.loc17_13.3.temp), !dbg !29
// CHECK:STDOUT:   ret void, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define dso_local void @_ZN1C1FENS_1EE.carbon_thunk(ptr noundef %0) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !31
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !31
// CHECK:STDOUT:   %2 = load i16, ptr %1, align 2, !tbaa !33
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE(i16 noundef signext %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN1C1FENS_1EE(i16 noundef signext) #3
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4, !5}
// CHECK:STDOUT: !llvm.dbg.cu = !{!6}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!8}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 1, !"wchar_size", i32 4}
// CHECK:STDOUT: !3 = !{i32 8, !"PIC Level", i32 0}
// CHECK:STDOUT: !4 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !5 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !6 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !7, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !7 = !DIFile(filename: "convert.carbon", directory: "")
// CHECK:STDOUT: !8 = !{!9, !9, i64 0}
// CHECK:STDOUT: !9 = !{!"int", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"omnipotent char", !11, i64 0}
// CHECK:STDOUT: !11 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "PassEnum", linkageName: "_CPassEnum.Main", scope: null, file: !7, line: 8, type: !13, spFlags: DISPFlagDefinition, unit: !6)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{null}
// CHECK:STDOUT: !15 = !DILocation(line: 9, column: 3, scope: !12)
// CHECK:STDOUT: !16 = !DILocation(line: 8, column: 1, scope: !12)
// CHECK:STDOUT: !17 = distinct !DISubprogram(name: "ConvertEnumToI16", linkageName: "_CConvertEnumToI16.Main", scope: null, file: !7, line: 12, type: !18, spFlags: DISPFlagDefinition, unit: !6, retainedNodes: !21)
// CHECK:STDOUT: !18 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !19 = !{null, !20}
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{!22}
// CHECK:STDOUT: !22 = !DILocalVariable(arg: 1, scope: !17, type: !20)
// CHECK:STDOUT: !23 = !DILocation(line: 13, column: 3, scope: !17)
// CHECK:STDOUT: !24 = !DILocation(line: 12, column: 1, scope: !17)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "ConvertI16ToEnum", linkageName: "_CConvertI16ToEnum.Main", scope: null, file: !7, line: 16, type: !18, spFlags: DISPFlagDefinition, unit: !6, retainedNodes: !26)
// CHECK:STDOUT: !26 = !{!27}
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !25, type: !20)
// CHECK:STDOUT: !28 = !DILocation(line: 17, column: 11, scope: !25)
// CHECK:STDOUT: !29 = !DILocation(line: 17, column: 3, scope: !25)
// CHECK:STDOUT: !30 = !DILocation(line: 16, column: 1, scope: !25)
// CHECK:STDOUT: !31 = !{!32, !32, i64 0}
// CHECK:STDOUT: !32 = !{!"any pointer", !10, i64 0}
// CHECK:STDOUT: !33 = !{!34, !34, i64 0}
// CHECK:STDOUT: !34 = !{!"_ZTSN1C1EE", !10, i64 0}
// CHECK:STDOUT: ; ModuleID = 'use_bitmask.carbon'
// CHECK:STDOUT: source_filename = "use_bitmask.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCall.Main() #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 5), !dbg !15
// CHECK:STDOUT:   ret void, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z4Take4Bits(i32)
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4, !5}
// CHECK:STDOUT: !llvm.dbg.cu = !{!6}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!8}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 1, !"wchar_size", i32 4}
// CHECK:STDOUT: !3 = !{i32 8, !"PIC Level", i32 0}
// CHECK:STDOUT: !4 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !5 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !6 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !7, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !7 = !DIFile(filename: "use_bitmask.carbon", directory: "")
// CHECK:STDOUT: !8 = !{!9, !9, i64 0}
// CHECK:STDOUT: !9 = !{!"int", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"omnipotent char", !11, i64 0}
// CHECK:STDOUT: !11 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Call", linkageName: "_CCall.Main", scope: null, file: !7, line: 10, type: !13, spFlags: DISPFlagDefinition, unit: !6)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{null}
// CHECK:STDOUT: !15 = !DILocation(line: 11, column: 3, scope: !12)
// CHECK:STDOUT: !16 = !DILocation(line: 10, column: 1, scope: !12)
