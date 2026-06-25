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

fn CompileTime() {
  Cpp.Take(Cpp.A & Cpp.C);
  Cpp.Take(Cpp.A | Cpp.C);
  Cpp.Take(Cpp.A ^ Cpp.C);
  Cpp.Take(^Cpp.A);
}

fn Runtime(a: Cpp.Bits, b: Cpp.Bits) {
  Cpp.Take(a & b);
  Cpp.Take(a | b);
  Cpp.Take(a ^ b);
  // TODO: This produces a value with bits set that are not in the range of
  // representable values of the enumeration. Should we produce `^a & 7`
  // instead?
  Cpp.Take(^a);
}

// --- compare.carbon

library "[[@TEST_NAME]]";

import Cpp library "enum_member.h";

fn Compare(a: Cpp.C.E, b: Cpp.C.E) -> bool {
  return a == b;
}

fn CompareGeneric[U:! type, T:! Core.EqWith(U)](x: T, y: U) -> bool {
  return x == y;
}

fn CallCompareGeneric(a: Cpp.C.E, b: Cpp.C.E) -> bool {
  return CompareGeneric(a, b);
}

// --- compare_class.carbon

library "[[@TEST_NAME]]";

import Cpp library "enum_member.h";

inline Cpp '''
enum E2 {};
struct ConvertToEnum {
  operator C::E() const;
  operator E2() const;
};
struct ConvertToEnum2 {
  operator C::E() const;
};
''';

fn Compare(a: Cpp.C.E, b: Cpp.ConvertToEnum) -> bool {
  return a == b;
}

fn Compare2(a: Cpp.E2, b: Cpp.ConvertToEnum) -> bool {
  return a == b;
}

fn Compare3(a: Cpp.C.E, b: Cpp.ConvertToEnum2) -> bool {
  return a == b;
}

fn CompareGeneric[U:! type, T:! Core.EqWith(U)](x: T, y: U) -> bool {
  return x == y;
}

// Vary both arguments to ensure they're both included in the fingerprint for
// the thunk.
fn CallCompareGeneric(a: Cpp.C.E, b: Cpp.ConvertToEnum) -> bool {
  return CompareGeneric(a, b);
}

fn CallCompareGeneric2(a: Cpp.E2, b: Cpp.ConvertToEnum) -> bool {
  return CompareGeneric(a, b);
}

fn CallCompareGeneric3(a: Cpp.C.E, b: Cpp.ConvertToEnum2) -> bool {
  return CompareGeneric(a, b);
}

// CHECK:STDOUT: ; ModuleID = 'pass_as_arg.carbon'
// CHECK:STDOUT: source_filename = "pass_as_arg.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: @int_0 = internal constant i16 0
// CHECK:STDOUT: @int_7 = internal constant i16 7
// CHECK:STDOUT: @int_8 = internal constant i16 8
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPass.Main() #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_16.1.temp = alloca i16, align 2, !dbg !14
// CHECK:STDOUT:   %.loc8_16.1.temp = alloca i16, align 2, !dbg !15
// CHECK:STDOUT:   %.loc9_16.1.temp = alloca i16, align 2, !dbg !16
// CHECK:STDOUT:   %.loc10_16.1.temp = alloca i16, align 2, !dbg !17
// CHECK:STDOUT:   %.loc11_18.1.temp = alloca i16, align 2, !dbg !18
// CHECK:STDOUT:   %.loc12_18.1.temp = alloca i16, align 2, !dbg !19
// CHECK:STDOUT:   %.loc13_18.1.temp = alloca i16, align 2, !dbg !20
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_16.1.temp), !dbg !14
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr @int_0), !dbg !21
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc8_16.1.temp), !dbg !15
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr @int_0), !dbg !22
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc9_16.1.temp), !dbg !16
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr @int_7), !dbg !23
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc10_16.1.temp), !dbg !17
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr @int_8), !dbg !24
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc11_18.1.temp), !dbg !18
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr @int_0), !dbg !25
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc12_18.1.temp), !dbg !19
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr @int_7), !dbg !26
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc13_18.1.temp), !dbg !20
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr @int_8), !dbg !27
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN1C1FENS_1EE.carbon_thunk._(ptr noundef %0) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !29
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !29
// CHECK:STDOUT:   %2 = load i16, ptr %1, align 2, !tbaa !31
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE(i16 noundef signext %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN1C1FENS_1EE(i16 noundef signext) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_ZN1C1FENS_1EE.carbon_thunk._, { 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3, !4, !5, !6}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!7}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "pass_as_arg.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !3 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !5 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !6 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "Pass", linkageName: "_CPass.Main", scope: null, file: !1, line: 6, type: !12, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{null}
// CHECK:STDOUT: !14 = !DILocation(line: 7, column: 11, scope: !11)
// CHECK:STDOUT: !15 = !DILocation(line: 8, column: 11, scope: !11)
// CHECK:STDOUT: !16 = !DILocation(line: 9, column: 11, scope: !11)
// CHECK:STDOUT: !17 = !DILocation(line: 10, column: 11, scope: !11)
// CHECK:STDOUT: !18 = !DILocation(line: 11, column: 11, scope: !11)
// CHECK:STDOUT: !19 = !DILocation(line: 12, column: 11, scope: !11)
// CHECK:STDOUT: !20 = !DILocation(line: 13, column: 11, scope: !11)
// CHECK:STDOUT: !21 = !DILocation(line: 7, column: 3, scope: !11)
// CHECK:STDOUT: !22 = !DILocation(line: 8, column: 3, scope: !11)
// CHECK:STDOUT: !23 = !DILocation(line: 9, column: 3, scope: !11)
// CHECK:STDOUT: !24 = !DILocation(line: 10, column: 3, scope: !11)
// CHECK:STDOUT: !25 = !DILocation(line: 11, column: 3, scope: !11)
// CHECK:STDOUT: !26 = !DILocation(line: 12, column: 3, scope: !11)
// CHECK:STDOUT: !27 = !DILocation(line: 13, column: 3, scope: !11)
// CHECK:STDOUT: !28 = !DILocation(line: 6, column: 1, scope: !11)
// CHECK:STDOUT: !29 = !{!30, !30, i64 0}
// CHECK:STDOUT: !30 = !{!"any pointer", !9, i64 0}
// CHECK:STDOUT: !31 = !{!32, !32, i64 0}
// CHECK:STDOUT: !32 = !{!"_ZTSN1C1EE", !9, i64 0}
// CHECK:STDOUT: ; ModuleID = 'convert.carbon'
// CHECK:STDOUT: source_filename = "convert.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CTakeI16.Main(i16)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassEnum.Main() #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CTakeI16.Main(i16 7), !dbg !14
// CHECK:STDOUT:   ret void, !dbg !15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertEnumToI16.Main(i16 %e) #0 !dbg !16 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CTakeI16.Main(i16 %e), !dbg !22
// CHECK:STDOUT:   ret void, !dbg !23
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertI16ToEnum.Main(i16 %n) #0 !dbg !24 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17_13.3.temp = alloca i16, align 2, !dbg !27
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc17_13.3.temp), !dbg !27
// CHECK:STDOUT:   store i16 %n, ptr %.loc17_13.3.temp, align 2, !dbg !27
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr %.loc17_13.3.temp), !dbg !28
// CHECK:STDOUT:   ret void, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN1C1FENS_1EE.carbon_thunk._(ptr noundef %0) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !30
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !30
// CHECK:STDOUT:   %2 = load i16, ptr %1, align 2, !tbaa !32
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE(i16 noundef signext %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN1C1FENS_1EE(i16 noundef signext) #3
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3, !4, !5, !6}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!7}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "convert.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !3 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !5 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !6 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "PassEnum", linkageName: "_CPassEnum.Main", scope: null, file: !1, line: 8, type: !12, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{null}
// CHECK:STDOUT: !14 = !DILocation(line: 9, column: 3, scope: !11)
// CHECK:STDOUT: !15 = !DILocation(line: 8, column: 1, scope: !11)
// CHECK:STDOUT: !16 = distinct !DISubprogram(name: "ConvertEnumToI16", linkageName: "_CConvertEnumToI16.Main", scope: null, file: !1, line: 12, type: !17, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !20)
// CHECK:STDOUT: !17 = !DISubroutineType(types: !18)
// CHECK:STDOUT: !18 = !{null, !19}
// CHECK:STDOUT: !19 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !20 = !{!21}
// CHECK:STDOUT: !21 = !DILocalVariable(arg: 1, scope: !16, type: !19)
// CHECK:STDOUT: !22 = !DILocation(line: 13, column: 3, scope: !16)
// CHECK:STDOUT: !23 = !DILocation(line: 12, column: 1, scope: !16)
// CHECK:STDOUT: !24 = distinct !DISubprogram(name: "ConvertI16ToEnum", linkageName: "_CConvertI16ToEnum.Main", scope: null, file: !1, line: 16, type: !17, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !25)
// CHECK:STDOUT: !25 = !{!26}
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 1, scope: !24, type: !19)
// CHECK:STDOUT: !27 = !DILocation(line: 17, column: 11, scope: !24)
// CHECK:STDOUT: !28 = !DILocation(line: 17, column: 3, scope: !24)
// CHECK:STDOUT: !29 = !DILocation(line: 16, column: 1, scope: !24)
// CHECK:STDOUT: !30 = !{!31, !31, i64 0}
// CHECK:STDOUT: !31 = !{!"any pointer", !9, i64 0}
// CHECK:STDOUT: !32 = !{!33, !33, i64 0}
// CHECK:STDOUT: !33 = !{!"_ZTSN1C1EE", !9, i64 0}
// CHECK:STDOUT: ; ModuleID = 'use_bitmask.carbon'
// CHECK:STDOUT: source_filename = "use_bitmask.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCompileTime.Main() #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 0), !dbg !14
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 5), !dbg !15
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 5), !dbg !16
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 -2), !dbg !17
// CHECK:STDOUT:   ret void, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z4Take4Bits(i32 noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CRuntime.Main(i32 %a, i32 %b) #0 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Op.call.loc14 = and i32 %a, %b, !dbg !26
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 %Op.call.loc14), !dbg !27
// CHECK:STDOUT:   %Op.call.loc15 = or i32 %a, %b, !dbg !28
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 %Op.call.loc15), !dbg !29
// CHECK:STDOUT:   %Op.call.loc16 = xor i32 %a, %b, !dbg !30
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 %Op.call.loc16), !dbg !31
// CHECK:STDOUT:   %Op.call.loc20 = xor i32 -1, %a, !dbg !32
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 %Op.call.loc20), !dbg !33
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_Z4Take4Bits, { 0, 1, 2, 3, 7, 6, 5, 4 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3, !4, !5, !6}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!7}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "use_bitmask.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !3 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !5 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !6 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "CompileTime", linkageName: "_CCompileTime.Main", scope: null, file: !1, line: 6, type: !12, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{null}
// CHECK:STDOUT: !14 = !DILocation(line: 7, column: 3, scope: !11)
// CHECK:STDOUT: !15 = !DILocation(line: 8, column: 3, scope: !11)
// CHECK:STDOUT: !16 = !DILocation(line: 9, column: 3, scope: !11)
// CHECK:STDOUT: !17 = !DILocation(line: 10, column: 3, scope: !11)
// CHECK:STDOUT: !18 = !DILocation(line: 6, column: 1, scope: !11)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "Runtime", linkageName: "_CRuntime.Main", scope: null, file: !1, line: 13, type: !20, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !23)
// CHECK:STDOUT: !20 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !21 = !{null, !22, !22}
// CHECK:STDOUT: !22 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !23 = !{!24, !25}
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !19, type: !22)
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 2, scope: !19, type: !22)
// CHECK:STDOUT: !26 = !DILocation(line: 14, column: 12, scope: !19)
// CHECK:STDOUT: !27 = !DILocation(line: 14, column: 3, scope: !19)
// CHECK:STDOUT: !28 = !DILocation(line: 15, column: 12, scope: !19)
// CHECK:STDOUT: !29 = !DILocation(line: 15, column: 3, scope: !19)
// CHECK:STDOUT: !30 = !DILocation(line: 16, column: 12, scope: !19)
// CHECK:STDOUT: !31 = !DILocation(line: 16, column: 3, scope: !19)
// CHECK:STDOUT: !32 = !DILocation(line: 20, column: 12, scope: !19)
// CHECK:STDOUT: !33 = !DILocation(line: 20, column: 3, scope: !19)
// CHECK:STDOUT: !34 = !DILocation(line: 13, column: 1, scope: !19)
// CHECK:STDOUT: ; ModuleID = 'compare.carbon'
// CHECK:STDOUT: source_filename = "compare.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCompare.Main(i16 %a, i16 %b) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Equal.call = icmp eq i16 %a, %b, !dbg !19
// CHECK:STDOUT:   ret i1 %Equal.call, !dbg !20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCallCompareGeneric.Main(i16 %a, i16 %b) #0 !dbg !21 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %CompareGeneric.call = call i1 @_CCompareGeneric.Main.80c3fc0239809458(i16 %a, i16 %b), !dbg !25
// CHECK:STDOUT:   ret i1 %CompareGeneric.call, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CCompareGeneric.Main.80c3fc0239809458(i16 %x, i16 %y) #0 !dbg !27 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.call = icmp eq i16 %x, %y, !dbg !31
// CHECK:STDOUT:   ret i1 %EqWith.WithSelf.Equal.call, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3, !4, !5, !6}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!7}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "compare.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !3 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !5 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !6 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "Compare", linkageName: "_CCompare.Main", scope: null, file: !1, line: 6, type: !12, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !16)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{!14, !15, !15}
// CHECK:STDOUT: !14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !15 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !16 = !{!17, !18}
// CHECK:STDOUT: !17 = !DILocalVariable(arg: 1, scope: !11, type: !15)
// CHECK:STDOUT: !18 = !DILocalVariable(arg: 2, scope: !11, type: !15)
// CHECK:STDOUT: !19 = !DILocation(line: 7, column: 10, scope: !11)
// CHECK:STDOUT: !20 = !DILocation(line: 7, column: 3, scope: !11)
// CHECK:STDOUT: !21 = distinct !DISubprogram(name: "CallCompareGeneric", linkageName: "_CCallCompareGeneric.Main", scope: null, file: !1, line: 14, type: !12, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !22)
// CHECK:STDOUT: !22 = !{!23, !24}
// CHECK:STDOUT: !23 = !DILocalVariable(arg: 1, scope: !21, type: !15)
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 2, scope: !21, type: !15)
// CHECK:STDOUT: !25 = !DILocation(line: 15, column: 10, scope: !21)
// CHECK:STDOUT: !26 = !DILocation(line: 15, column: 3, scope: !21)
// CHECK:STDOUT: !27 = distinct !DISubprogram(name: "CompareGeneric", linkageName: "_CCompareGeneric.Main.80c3fc0239809458", scope: null, file: !1, line: 10, type: !12, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !28)
// CHECK:STDOUT: !28 = !{!29, !30}
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 1, scope: !27, type: !15)
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 2, scope: !27, type: !15)
// CHECK:STDOUT: !31 = !DILocation(line: 11, column: 10, scope: !27)
// CHECK:STDOUT: !32 = !DILocation(line: 11, column: 3, scope: !27)
// CHECK:STDOUT: ; ModuleID = 'compare_class.carbon'
// CHECK:STDOUT: source_filename = "compare_class.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCompare.Main(i16 %a, ptr %b) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_15.1.temp = alloca i16, align 2, !dbg !19
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_15.1.temp), !dbg !19
// CHECK:STDOUT:   call void @_ZNK13ConvertToEnumcvN1C1EEEv.carbon_thunk._(ptr %b, ptr %.loc18_15.1.temp), !dbg !19
// CHECK:STDOUT:   %.loc18_15.5 = load i16, ptr %.loc18_15.1.temp, align 2, !dbg !19
// CHECK:STDOUT:   %Equal.call = icmp eq i16 %a, %.loc18_15.5, !dbg !20
// CHECK:STDOUT:   ret i1 %Equal.call, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZNK13ConvertToEnumcvN1C1EEEv.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !22
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !25
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !25
// CHECK:STDOUT:   %1 = load ptr, ptr %this.addr, align 8, !tbaa !22, !nonnull !26
// CHECK:STDOUT:   %call = call noundef signext i16 @_ZNK13ConvertToEnumcvN1C1EEEv(ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   store i16 %call, ptr %0, align 2, !tbaa !27
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCompare2.Main(i32 %a, ptr %b) #0 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %__carbon_thunk.call = call i32 @_ZNK13ConvertToEnumcv2E2Ev.carbon_thunk._(ptr %b), !dbg !36
// CHECK:STDOUT:   %Equal.call = icmp eq i32 %a, %__carbon_thunk.call, !dbg !37
// CHECK:STDOUT:   ret i1 %Equal.call, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal noundef i32 @_ZNK13ConvertToEnumcv2E2Ev.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !22
// CHECK:STDOUT:   %0 = load ptr, ptr %this.addr, align 8, !tbaa !22, !nonnull !26
// CHECK:STDOUT:   %call = call noundef i32 @_ZNK13ConvertToEnumcv2E2Ev(ptr noundef nonnull align 1 dereferenceable(1) %0)
// CHECK:STDOUT:   ret i32 %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCompare3.Main(i16 %a, ptr %b) #0 !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc26_15.1.temp = alloca i16, align 2, !dbg !43
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc26_15.1.temp), !dbg !43
// CHECK:STDOUT:   call void @_ZNK14ConvertToEnum2cvN1C1EEEv.carbon_thunk._(ptr %b, ptr %.loc26_15.1.temp), !dbg !43
// CHECK:STDOUT:   %.loc26_15.5 = load i16, ptr %.loc26_15.1.temp, align 2, !dbg !43
// CHECK:STDOUT:   %Equal.call = icmp eq i16 %a, %.loc26_15.5, !dbg !44
// CHECK:STDOUT:   ret i1 %Equal.call, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZNK14ConvertToEnum2cvN1C1EEEv.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !46
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !25
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !25
// CHECK:STDOUT:   %1 = load ptr, ptr %this.addr, align 8, !tbaa !46, !nonnull !26
// CHECK:STDOUT:   %call = call noundef signext i16 @_ZNK14ConvertToEnum2cvN1C1EEEv(ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   store i16 %call, ptr %0, align 2, !tbaa !27
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCallCompareGeneric.Main(i16 %a, ptr %b) #0 !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %CompareGeneric.call = call i1 @_CCompareGeneric.Main.215a23369ed4d589(i16 %a, ptr %b), !dbg !52
// CHECK:STDOUT:   ret i1 %CompareGeneric.call, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define i1 @"_CEqual:thunk:EqWith.83b9626e83cf3dcc.Core:enclosed"(i16 %self, ptr %other) #2 !dbg !54 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.4.temp = alloca i16, align 2, !dbg !58
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.4.temp), !dbg !58
// CHECK:STDOUT:   call void @_ZNK13ConvertToEnumcvN1C1EEEv.carbon_thunk._(ptr %other, ptr %.4.temp), !dbg !58
// CHECK:STDOUT:   %.8 = load i16, ptr %.4.temp, align 2, !dbg !58
// CHECK:STDOUT:   %Equal.call = icmp eq i16 %self, %.8, !dbg !58
// CHECK:STDOUT:   ret i1 %Equal.call, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define i1 @"_CNotEqual:thunk:EqWith.83b9626e83cf3dcc.Core:enclosed"(i16 %self, ptr %other) #2 !dbg !59 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.4.temp = alloca i16, align 2, !dbg !63
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.4.temp), !dbg !63
// CHECK:STDOUT:   call void @_ZNK13ConvertToEnumcvN1C1EEEv.carbon_thunk._(ptr %other, ptr %.4.temp), !dbg !63
// CHECK:STDOUT:   %.8 = load i16, ptr %.4.temp, align 2, !dbg !63
// CHECK:STDOUT:   %NotEqual.call = icmp ne i16 %self, %.8, !dbg !63
// CHECK:STDOUT:   ret i1 %NotEqual.call, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCallCompareGeneric2.Main(i32 %a, ptr %b) #0 !dbg !64 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %CompareGeneric.call = call i1 @_CCompareGeneric.Main.a5489e454367d917(i32 %a, ptr %b), !dbg !68
// CHECK:STDOUT:   ret i1 %CompareGeneric.call, !dbg !69
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define i1 @"_CEqual:thunk:EqWith.e8fe48de943d9652.Core:enclosed"(i32 %self, ptr %other) #2 !dbg !70 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %__carbon_thunk.call = call i32 @_ZNK13ConvertToEnumcv2E2Ev.carbon_thunk._(ptr %other), !dbg !74
// CHECK:STDOUT:   %Equal.call = icmp eq i32 %self, %__carbon_thunk.call, !dbg !74
// CHECK:STDOUT:   ret i1 %Equal.call, !dbg !74
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define i1 @"_CNotEqual:thunk:EqWith.e8fe48de943d9652.Core:enclosed"(i32 %self, ptr %other) #2 !dbg !75 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %__carbon_thunk.call = call i32 @_ZNK13ConvertToEnumcv2E2Ev.carbon_thunk._(ptr %other), !dbg !79
// CHECK:STDOUT:   %NotEqual.call = icmp ne i32 %self, %__carbon_thunk.call, !dbg !79
// CHECK:STDOUT:   ret i1 %NotEqual.call, !dbg !79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCallCompareGeneric3.Main(i16 %a, ptr %b) #0 !dbg !80 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %CompareGeneric.call = call i1 @_CCompareGeneric.Main.64b737572447efad(i16 %a, ptr %b), !dbg !84
// CHECK:STDOUT:   ret i1 %CompareGeneric.call, !dbg !85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define i1 @"_CEqual:thunk:EqWith.a294139a6615e163.Core:enclosed"(i16 %self, ptr %other) #2 !dbg !86 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.4.temp = alloca i16, align 2, !dbg !90
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.4.temp), !dbg !90
// CHECK:STDOUT:   call void @_ZNK14ConvertToEnum2cvN1C1EEEv.carbon_thunk._(ptr %other, ptr %.4.temp), !dbg !90
// CHECK:STDOUT:   %.8 = load i16, ptr %.4.temp, align 2, !dbg !90
// CHECK:STDOUT:   %Equal.call = icmp eq i16 %self, %.8, !dbg !90
// CHECK:STDOUT:   ret i1 %Equal.call, !dbg !90
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define i1 @"_CNotEqual:thunk:EqWith.a294139a6615e163.Core:enclosed"(i16 %self, ptr %other) #2 !dbg !91 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.4.temp = alloca i16, align 2, !dbg !95
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.4.temp), !dbg !95
// CHECK:STDOUT:   call void @_ZNK14ConvertToEnum2cvN1C1EEEv.carbon_thunk._(ptr %other, ptr %.4.temp), !dbg !95
// CHECK:STDOUT:   %.8 = load i16, ptr %.4.temp, align 2, !dbg !95
// CHECK:STDOUT:   %NotEqual.call = icmp ne i16 %self, %.8, !dbg !95
// CHECK:STDOUT:   ret i1 %NotEqual.call, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CCompareGeneric.Main.215a23369ed4d589(i16 %x, ptr %y) #0 !dbg !96 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.call = call i1 @"_CEqual:thunk:EqWith.83b9626e83cf3dcc.Core:enclosed"(i16 %x, ptr %y), !dbg !100
// CHECK:STDOUT:   ret i1 %EqWith.WithSelf.Equal.call, !dbg !101
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CCompareGeneric.Main.a5489e454367d917(i32 %x, ptr %y) #0 !dbg !102 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.call = call i1 @"_CEqual:thunk:EqWith.e8fe48de943d9652.Core:enclosed"(i32 %x, ptr %y), !dbg !106
// CHECK:STDOUT:   ret i1 %EqWith.WithSelf.Equal.call, !dbg !107
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CCompareGeneric.Main.64b737572447efad(i16 %x, ptr %y) #0 !dbg !108 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.call = call i1 @"_CEqual:thunk:EqWith.a294139a6615e163.Core:enclosed"(i16 %x, ptr %y), !dbg !112
// CHECK:STDOUT:   ret i1 %EqWith.WithSelf.Equal.call, !dbg !113
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef signext i16 @_ZNK13ConvertToEnumcvN1C1EEEv(ptr noundef nonnull align 1 dereferenceable(1)) #4
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef i32 @_ZNK13ConvertToEnumcv2E2Ev(ptr noundef nonnull align 1 dereferenceable(1)) #4
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef signext i16 @_ZNK14ConvertToEnum2cvN1C1EEEv(ptr noundef nonnull align 1 dereferenceable(1)) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #4 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3, !4, !5, !6}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!7}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "compare_class.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !3 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !4 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !5 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !6 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !{!8, !8, i64 0}
// CHECK:STDOUT: !8 = !{!"int", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"omnipotent char", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "Compare", linkageName: "_CCompare.Main", scope: null, file: !1, line: 17, type: !12, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !16)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{!14, !15, !14}
// CHECK:STDOUT: !14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !15 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !16 = !{!17, !18}
// CHECK:STDOUT: !17 = !DILocalVariable(arg: 1, scope: !11, type: !15)
// CHECK:STDOUT: !18 = !DILocalVariable(arg: 2, scope: !11, type: !14)
// CHECK:STDOUT: !19 = !DILocation(line: 18, column: 15, scope: !11)
// CHECK:STDOUT: !20 = !DILocation(line: 18, column: 10, scope: !11)
// CHECK:STDOUT: !21 = !DILocation(line: 18, column: 3, scope: !11)
// CHECK:STDOUT: !22 = !{!23, !23, i64 0}
// CHECK:STDOUT: !23 = !{!"p1 _ZTS13ConvertToEnum", !24, i64 0}
// CHECK:STDOUT: !24 = !{!"any pointer", !9, i64 0}
// CHECK:STDOUT: !25 = !{!24, !24, i64 0}
// CHECK:STDOUT: !26 = !{}
// CHECK:STDOUT: !27 = !{!28, !28, i64 0}
// CHECK:STDOUT: !28 = !{!"_ZTSN1C1EE", !9, i64 0}
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "Compare2", linkageName: "_CCompare2.Main", scope: null, file: !1, line: 21, type: !30, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !33)
// CHECK:STDOUT: !30 = !DISubroutineType(types: !31)
// CHECK:STDOUT: !31 = !{!14, !32, !14}
// CHECK:STDOUT: !32 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !33 = !{!34, !35}
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 1, scope: !29, type: !32)
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 2, scope: !29, type: !14)
// CHECK:STDOUT: !36 = !DILocation(line: 22, column: 15, scope: !29)
// CHECK:STDOUT: !37 = !DILocation(line: 22, column: 10, scope: !29)
// CHECK:STDOUT: !38 = !DILocation(line: 22, column: 3, scope: !29)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "Compare3", linkageName: "_CCompare3.Main", scope: null, file: !1, line: 25, type: !12, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !40)
// CHECK:STDOUT: !40 = !{!41, !42}
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !39, type: !15)
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 2, scope: !39, type: !14)
// CHECK:STDOUT: !43 = !DILocation(line: 26, column: 15, scope: !39)
// CHECK:STDOUT: !44 = !DILocation(line: 26, column: 10, scope: !39)
// CHECK:STDOUT: !45 = !DILocation(line: 26, column: 3, scope: !39)
// CHECK:STDOUT: !46 = !{!47, !47, i64 0}
// CHECK:STDOUT: !47 = !{!"p1 _ZTS14ConvertToEnum2", !24, i64 0}
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "CallCompareGeneric", linkageName: "_CCallCompareGeneric.Main", scope: null, file: !1, line: 35, type: !12, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !49)
// CHECK:STDOUT: !49 = !{!50, !51}
// CHECK:STDOUT: !50 = !DILocalVariable(arg: 1, scope: !48, type: !15)
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 2, scope: !48, type: !14)
// CHECK:STDOUT: !52 = !DILocation(line: 36, column: 10, scope: !48)
// CHECK:STDOUT: !53 = !DILocation(line: 36, column: 3, scope: !48)
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "Equal", linkageName: "_CEqual:thunk:EqWith.83b9626e83cf3dcc.Core:enclosed", scope: null, file: !1, type: !12, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !55)
// CHECK:STDOUT: !55 = !{!56, !57}
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !54, type: !15)
// CHECK:STDOUT: !57 = !DILocalVariable(arg: 2, scope: !54, type: !14)
// CHECK:STDOUT: !58 = !DILocation(line: 0, scope: !54)
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "NotEqual", linkageName: "_CNotEqual:thunk:EqWith.83b9626e83cf3dcc.Core:enclosed", scope: null, file: !1, type: !12, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !60)
// CHECK:STDOUT: !60 = !{!61, !62}
// CHECK:STDOUT: !61 = !DILocalVariable(arg: 1, scope: !59, type: !15)
// CHECK:STDOUT: !62 = !DILocalVariable(arg: 2, scope: !59, type: !14)
// CHECK:STDOUT: !63 = !DILocation(line: 0, scope: !59)
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "CallCompareGeneric2", linkageName: "_CCallCompareGeneric2.Main", scope: null, file: !1, line: 39, type: !30, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !65)
// CHECK:STDOUT: !65 = !{!66, !67}
// CHECK:STDOUT: !66 = !DILocalVariable(arg: 1, scope: !64, type: !32)
// CHECK:STDOUT: !67 = !DILocalVariable(arg: 2, scope: !64, type: !14)
// CHECK:STDOUT: !68 = !DILocation(line: 40, column: 10, scope: !64)
// CHECK:STDOUT: !69 = !DILocation(line: 40, column: 3, scope: !64)
// CHECK:STDOUT: !70 = distinct !DISubprogram(name: "Equal", linkageName: "_CEqual:thunk:EqWith.e8fe48de943d9652.Core:enclosed", scope: null, file: !1, type: !30, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !71)
// CHECK:STDOUT: !71 = !{!72, !73}
// CHECK:STDOUT: !72 = !DILocalVariable(arg: 1, scope: !70, type: !32)
// CHECK:STDOUT: !73 = !DILocalVariable(arg: 2, scope: !70, type: !14)
// CHECK:STDOUT: !74 = !DILocation(line: 0, scope: !70)
// CHECK:STDOUT: !75 = distinct !DISubprogram(name: "NotEqual", linkageName: "_CNotEqual:thunk:EqWith.e8fe48de943d9652.Core:enclosed", scope: null, file: !1, type: !30, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !76)
// CHECK:STDOUT: !76 = !{!77, !78}
// CHECK:STDOUT: !77 = !DILocalVariable(arg: 1, scope: !75, type: !32)
// CHECK:STDOUT: !78 = !DILocalVariable(arg: 2, scope: !75, type: !14)
// CHECK:STDOUT: !79 = !DILocation(line: 0, scope: !75)
// CHECK:STDOUT: !80 = distinct !DISubprogram(name: "CallCompareGeneric3", linkageName: "_CCallCompareGeneric3.Main", scope: null, file: !1, line: 43, type: !12, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !81)
// CHECK:STDOUT: !81 = !{!82, !83}
// CHECK:STDOUT: !82 = !DILocalVariable(arg: 1, scope: !80, type: !15)
// CHECK:STDOUT: !83 = !DILocalVariable(arg: 2, scope: !80, type: !14)
// CHECK:STDOUT: !84 = !DILocation(line: 44, column: 10, scope: !80)
// CHECK:STDOUT: !85 = !DILocation(line: 44, column: 3, scope: !80)
// CHECK:STDOUT: !86 = distinct !DISubprogram(name: "Equal", linkageName: "_CEqual:thunk:EqWith.a294139a6615e163.Core:enclosed", scope: null, file: !1, type: !12, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !87)
// CHECK:STDOUT: !87 = !{!88, !89}
// CHECK:STDOUT: !88 = !DILocalVariable(arg: 1, scope: !86, type: !15)
// CHECK:STDOUT: !89 = !DILocalVariable(arg: 2, scope: !86, type: !14)
// CHECK:STDOUT: !90 = !DILocation(line: 0, scope: !86)
// CHECK:STDOUT: !91 = distinct !DISubprogram(name: "NotEqual", linkageName: "_CNotEqual:thunk:EqWith.a294139a6615e163.Core:enclosed", scope: null, file: !1, type: !12, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !92)
// CHECK:STDOUT: !92 = !{!93, !94}
// CHECK:STDOUT: !93 = !DILocalVariable(arg: 1, scope: !91, type: !15)
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 2, scope: !91, type: !14)
// CHECK:STDOUT: !95 = !DILocation(line: 0, scope: !91)
// CHECK:STDOUT: !96 = distinct !DISubprogram(name: "CompareGeneric", linkageName: "_CCompareGeneric.Main.215a23369ed4d589", scope: null, file: !1, line: 29, type: !12, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !97)
// CHECK:STDOUT: !97 = !{!98, !99}
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 1, scope: !96, type: !15)
// CHECK:STDOUT: !99 = !DILocalVariable(arg: 2, scope: !96, type: !14)
// CHECK:STDOUT: !100 = !DILocation(line: 30, column: 10, scope: !96)
// CHECK:STDOUT: !101 = !DILocation(line: 30, column: 3, scope: !96)
// CHECK:STDOUT: !102 = distinct !DISubprogram(name: "CompareGeneric", linkageName: "_CCompareGeneric.Main.a5489e454367d917", scope: null, file: !1, line: 29, type: !30, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !103)
// CHECK:STDOUT: !103 = !{!104, !105}
// CHECK:STDOUT: !104 = !DILocalVariable(arg: 1, scope: !102, type: !32)
// CHECK:STDOUT: !105 = !DILocalVariable(arg: 2, scope: !102, type: !14)
// CHECK:STDOUT: !106 = !DILocation(line: 30, column: 10, scope: !102)
// CHECK:STDOUT: !107 = !DILocation(line: 30, column: 3, scope: !102)
// CHECK:STDOUT: !108 = distinct !DISubprogram(name: "CompareGeneric", linkageName: "_CCompareGeneric.Main.64b737572447efad", scope: null, file: !1, line: 29, type: !12, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !109)
// CHECK:STDOUT: !109 = !{!110, !111}
// CHECK:STDOUT: !110 = !DILocalVariable(arg: 1, scope: !108, type: !15)
// CHECK:STDOUT: !111 = !DILocalVariable(arg: 2, scope: !108, type: !14)
// CHECK:STDOUT: !112 = !DILocation(line: 30, column: 10, scope: !108)
// CHECK:STDOUT: !113 = !DILocation(line: 30, column: 3, scope: !108)
