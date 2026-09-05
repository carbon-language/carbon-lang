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

fn CompareGeneric[U: type, T: Core.EqWith(U)](x: T, y: U) -> bool {
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

fn CompareGeneric[U: type, T: Core.EqWith(U)](x: T, y: U) -> bool {
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

// CHECK:STDOUT: ; ---
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
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr @int_0), !dbg !22
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc8_16.1.temp), !dbg !16
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr @int_0), !dbg !23
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc9_16.1.temp), !dbg !17
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr @int_7), !dbg !24
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc10_16.1.temp), !dbg !18
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr @int_8), !dbg !25
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc11_18.1.temp), !dbg !19
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr @int_0), !dbg !26
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc12_18.1.temp), !dbg !20
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr @int_7), !dbg !27
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc13_18.1.temp), !dbg !21
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr @int_8), !dbg !28
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
// CHECK:STDOUT: !7 = !{!8, !9, i64 0}
// CHECK:STDOUT: !8 = !{!"__libc_errno", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"int", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"omnipotent char", !11, i64 0}
// CHECK:STDOUT: !11 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Pass", linkageName: "_CPass.Main", scope: null, file: !1, line: 6, type: !13, spFlags: DISPFlagDefinition, unit: !0)
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
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
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
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr %.loc17_13.3.temp), !dbg !29
// CHECK:STDOUT:   ret void, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN1C1FENS_1EE.carbon_thunk._(ptr noundef %0) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !31
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !31
// CHECK:STDOUT:   %2 = load i16, ptr %1, align 2, !tbaa !33
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
// CHECK:STDOUT: !7 = !{!8, !9, i64 0}
// CHECK:STDOUT: !8 = !{!"__libc_errno", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"int", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"omnipotent char", !11, i64 0}
// CHECK:STDOUT: !11 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "PassEnum", linkageName: "_CPassEnum.Main", scope: null, file: !1, line: 8, type: !13, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{null}
// CHECK:STDOUT: !15 = !DILocation(line: 9, column: 3, scope: !12)
// CHECK:STDOUT: !16 = !DILocation(line: 8, column: 1, scope: !12)
// CHECK:STDOUT: !17 = distinct !DISubprogram(name: "ConvertEnumToI16", linkageName: "_CConvertEnumToI16.Main", scope: null, file: !1, line: 12, type: !18, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !21)
// CHECK:STDOUT: !18 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !19 = !{null, !20}
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !{!22}
// CHECK:STDOUT: !22 = !DILocalVariable(arg: 1, scope: !17, type: !20)
// CHECK:STDOUT: !23 = !DILocation(line: 13, column: 3, scope: !17)
// CHECK:STDOUT: !24 = !DILocation(line: 12, column: 1, scope: !17)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "ConvertI16ToEnum", linkageName: "_CConvertI16ToEnum.Main", scope: null, file: !1, line: 16, type: !18, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !26)
// CHECK:STDOUT: !26 = !{!27}
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !25, type: !20)
// CHECK:STDOUT: !28 = !DILocation(line: 17, column: 11, scope: !25)
// CHECK:STDOUT: !29 = !DILocation(line: 17, column: 3, scope: !25)
// CHECK:STDOUT: !30 = !DILocation(line: 16, column: 1, scope: !25)
// CHECK:STDOUT: !31 = !{!32, !32, i64 0}
// CHECK:STDOUT: !32 = !{!"any pointer", !10, i64 0}
// CHECK:STDOUT: !33 = !{!34, !34, i64 0}
// CHECK:STDOUT: !34 = !{!"_ZTSN1C1EE", !10, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'use_bitmask.carbon'
// CHECK:STDOUT: source_filename = "use_bitmask.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCompileTime.Main() #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 0), !dbg !15
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 5), !dbg !16
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 5), !dbg !17
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 -2), !dbg !18
// CHECK:STDOUT:   ret void, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z4Take4Bits(i32 noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CRuntime.Main(i32 %a, i32 %b) #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %BitAndWith.Op.call = and i32 %a, %b, !dbg !27
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 %BitAndWith.Op.call), !dbg !28
// CHECK:STDOUT:   %BitOrWith.Op.call = or i32 %a, %b, !dbg !29
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 %BitOrWith.Op.call), !dbg !30
// CHECK:STDOUT:   %BitXorWith.Op.call = xor i32 %a, %b, !dbg !31
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 %BitXorWith.Op.call), !dbg !32
// CHECK:STDOUT:   %BitComplement.Op.call = xor i32 -1, %a, !dbg !33
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 %BitComplement.Op.call), !dbg !34
// CHECK:STDOUT:   ret void, !dbg !35
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
// CHECK:STDOUT: !7 = !{!8, !9, i64 0}
// CHECK:STDOUT: !8 = !{!"__libc_errno", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"int", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"omnipotent char", !11, i64 0}
// CHECK:STDOUT: !11 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "CompileTime", linkageName: "_CCompileTime.Main", scope: null, file: !1, line: 6, type: !13, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{null}
// CHECK:STDOUT: !15 = !DILocation(line: 7, column: 3, scope: !12)
// CHECK:STDOUT: !16 = !DILocation(line: 8, column: 3, scope: !12)
// CHECK:STDOUT: !17 = !DILocation(line: 9, column: 3, scope: !12)
// CHECK:STDOUT: !18 = !DILocation(line: 10, column: 3, scope: !12)
// CHECK:STDOUT: !19 = !DILocation(line: 6, column: 1, scope: !12)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "Runtime", linkageName: "_CRuntime.Main", scope: null, file: !1, line: 13, type: !21, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !24)
// CHECK:STDOUT: !21 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !22 = !{null, !23, !23}
// CHECK:STDOUT: !23 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !24 = !{!25, !26}
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 1, scope: !20, type: !23)
// CHECK:STDOUT: !26 = !DILocalVariable(arg: 2, scope: !20, type: !23)
// CHECK:STDOUT: !27 = !DILocation(line: 14, column: 12, scope: !20)
// CHECK:STDOUT: !28 = !DILocation(line: 14, column: 3, scope: !20)
// CHECK:STDOUT: !29 = !DILocation(line: 15, column: 12, scope: !20)
// CHECK:STDOUT: !30 = !DILocation(line: 15, column: 3, scope: !20)
// CHECK:STDOUT: !31 = !DILocation(line: 16, column: 12, scope: !20)
// CHECK:STDOUT: !32 = !DILocation(line: 16, column: 3, scope: !20)
// CHECK:STDOUT: !33 = !DILocation(line: 20, column: 12, scope: !20)
// CHECK:STDOUT: !34 = !DILocation(line: 20, column: 3, scope: !20)
// CHECK:STDOUT: !35 = !DILocation(line: 13, column: 1, scope: !20)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'compare.carbon'
// CHECK:STDOUT: source_filename = "compare.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCompare.Main(i16 %a, i16 %b) #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %OrderedWith.Equal.call = icmp eq i16 %a, %b, !dbg !20
// CHECK:STDOUT:   ret i1 %OrderedWith.Equal.call, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCallCompareGeneric.Main(i16 %a, i16 %b) #0 !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %CompareGeneric.call = call i1 @_CCompareGeneric.Main.0476e0c7c3f3c2a2(i16 %a, i16 %b), !dbg !26
// CHECK:STDOUT:   ret i1 %CompareGeneric.call, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CCompareGeneric.Main.0476e0c7c3f3c2a2(i16 %x, i16 %y) #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.call = icmp eq i16 %x, %y, !dbg !32
// CHECK:STDOUT:   ret i1 %EqWith.WithSelf.Equal.call, !dbg !33
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
// CHECK:STDOUT: !7 = !{!8, !9, i64 0}
// CHECK:STDOUT: !8 = !{!"__libc_errno", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"int", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"omnipotent char", !11, i64 0}
// CHECK:STDOUT: !11 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Compare", linkageName: "_CCompare.Main", scope: null, file: !1, line: 6, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !17)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{!15, !16, !16}
// CHECK:STDOUT: !15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !16 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !17 = !{!18, !19}
// CHECK:STDOUT: !18 = !DILocalVariable(arg: 1, scope: !12, type: !16)
// CHECK:STDOUT: !19 = !DILocalVariable(arg: 2, scope: !12, type: !16)
// CHECK:STDOUT: !20 = !DILocation(line: 7, column: 10, scope: !12)
// CHECK:STDOUT: !21 = !DILocation(line: 7, column: 3, scope: !12)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "CallCompareGeneric", linkageName: "_CCallCompareGeneric.Main", scope: null, file: !1, line: 14, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !23)
// CHECK:STDOUT: !23 = !{!24, !25}
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !22, type: !16)
// CHECK:STDOUT: !25 = !DILocalVariable(arg: 2, scope: !22, type: !16)
// CHECK:STDOUT: !26 = !DILocation(line: 15, column: 10, scope: !22)
// CHECK:STDOUT: !27 = !DILocation(line: 15, column: 3, scope: !22)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "CompareGeneric", linkageName: "_CCompareGeneric.Main.0476e0c7c3f3c2a2", scope: null, file: !1, line: 10, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !29)
// CHECK:STDOUT: !29 = !{!30, !31}
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 1, scope: !28, type: !16)
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 2, scope: !28, type: !16)
// CHECK:STDOUT: !32 = !DILocation(line: 11, column: 10, scope: !28)
// CHECK:STDOUT: !33 = !DILocation(line: 11, column: 3, scope: !28)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'compare_class.carbon'
// CHECK:STDOUT: source_filename = "compare_class.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCompare.Main(i16 %a, ptr %b) #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_15.1.temp = alloca i16, align 2, !dbg !20
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_15.1.temp), !dbg !20
// CHECK:STDOUT:   call void @_ZNK13ConvertToEnumcvN1C1EEEv.carbon_thunk._(ptr %b, ptr %.loc18_15.1.temp), !dbg !20
// CHECK:STDOUT:   %.loc18_15.5 = load i16, ptr %.loc18_15.1.temp, align 2, !dbg !20
// CHECK:STDOUT:   %OrderedWith.Equal.call = icmp eq i16 %a, %.loc18_15.5, !dbg !21
// CHECK:STDOUT:   ret i1 %OrderedWith.Equal.call, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZNK13ConvertToEnumcvN1C1EEEv.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !23
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !26
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !26
// CHECK:STDOUT:   %1 = load ptr, ptr %this.addr, align 8, !tbaa !23, !nonnull !27
// CHECK:STDOUT:   %call = call noundef signext i16 @_ZNK13ConvertToEnumcvN1C1EEEv(ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   store i16 %call, ptr %0, align 2, !tbaa !28
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCompare2.Main(i32 %a, ptr %b) #0 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %__carbon_thunk.call = call i32 @_ZNK13ConvertToEnumcv2E2Ev.carbon_thunk._(ptr %b), !dbg !37
// CHECK:STDOUT:   %OrderedWith.Equal.call = icmp eq i32 %a, %__carbon_thunk.call, !dbg !38
// CHECK:STDOUT:   ret i1 %OrderedWith.Equal.call, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal noundef i32 @_ZNK13ConvertToEnumcv2E2Ev.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !23
// CHECK:STDOUT:   %0 = load ptr, ptr %this.addr, align 8, !tbaa !23, !nonnull !27
// CHECK:STDOUT:   %call = call noundef i32 @_ZNK13ConvertToEnumcv2E2Ev(ptr noundef nonnull align 1 dereferenceable(1) %0)
// CHECK:STDOUT:   ret i32 %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCompare3.Main(i16 %a, ptr %b) #0 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc26_15.1.temp = alloca i16, align 2, !dbg !44
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc26_15.1.temp), !dbg !44
// CHECK:STDOUT:   call void @_ZNK14ConvertToEnum2cvN1C1EEEv.carbon_thunk._(ptr %b, ptr %.loc26_15.1.temp), !dbg !44
// CHECK:STDOUT:   %.loc26_15.5 = load i16, ptr %.loc26_15.1.temp, align 2, !dbg !44
// CHECK:STDOUT:   %OrderedWith.Equal.call = icmp eq i16 %a, %.loc26_15.5, !dbg !45
// CHECK:STDOUT:   ret i1 %OrderedWith.Equal.call, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZNK14ConvertToEnum2cvN1C1EEEv.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !47
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !26
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !26
// CHECK:STDOUT:   %1 = load ptr, ptr %this.addr, align 8, !tbaa !47, !nonnull !27
// CHECK:STDOUT:   %call = call noundef signext i16 @_ZNK14ConvertToEnum2cvN1C1EEEv(ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   store i16 %call, ptr %0, align 2, !tbaa !28
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCallCompareGeneric.Main(i16 %a, ptr %b) #0 !dbg !49 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %CompareGeneric.call = call i1 @_CCompareGeneric.Main.f8b275c548b6e015(i16 %a, ptr %b), !dbg !53
// CHECK:STDOUT:   ret i1 %CompareGeneric.call, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CEqual:thunk:EqWith.83b9626e83cf3dcc.Core:OrderedWith.Core"(i16 %self, ptr %other) #2 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.4.temp = alloca i16, align 2, !dbg !59
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.4.temp), !dbg !59
// CHECK:STDOUT:   call void @_ZNK13ConvertToEnumcvN1C1EEEv.carbon_thunk._(ptr %other, ptr %.4.temp), !dbg !59
// CHECK:STDOUT:   %.8 = load i16, ptr %.4.temp, align 2, !dbg !59
// CHECK:STDOUT:   %OrderedWith.Equal.call = icmp eq i16 %self, %.8, !dbg !59
// CHECK:STDOUT:   ret i1 %OrderedWith.Equal.call, !dbg !59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CNotEqual:thunk:EqWith.83b9626e83cf3dcc.Core:OrderedWith.Core"(i16 %self, ptr %other) #2 !dbg !60 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.4.temp = alloca i16, align 2, !dbg !64
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.4.temp), !dbg !64
// CHECK:STDOUT:   call void @_ZNK13ConvertToEnumcvN1C1EEEv.carbon_thunk._(ptr %other, ptr %.4.temp), !dbg !64
// CHECK:STDOUT:   %.8 = load i16, ptr %.4.temp, align 2, !dbg !64
// CHECK:STDOUT:   %OrderedWith.NotEqual.call = icmp ne i16 %self, %.8, !dbg !64
// CHECK:STDOUT:   ret i1 %OrderedWith.NotEqual.call, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCallCompareGeneric2.Main(i32 %a, ptr %b) #0 !dbg !65 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %CompareGeneric.call = call i1 @_CCompareGeneric.Main.f7b86ff4854a74c3(i32 %a, ptr %b), !dbg !69
// CHECK:STDOUT:   ret i1 %CompareGeneric.call, !dbg !70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CEqual:thunk:EqWith.e8fe48de943d9652.Core:OrderedWith.Core"(i32 %self, ptr %other) #2 !dbg !71 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %__carbon_thunk.call = call i32 @_ZNK13ConvertToEnumcv2E2Ev.carbon_thunk._(ptr %other), !dbg !75
// CHECK:STDOUT:   %OrderedWith.Equal.call = icmp eq i32 %self, %__carbon_thunk.call, !dbg !75
// CHECK:STDOUT:   ret i1 %OrderedWith.Equal.call, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CNotEqual:thunk:EqWith.e8fe48de943d9652.Core:OrderedWith.Core"(i32 %self, ptr %other) #2 !dbg !76 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %__carbon_thunk.call = call i32 @_ZNK13ConvertToEnumcv2E2Ev.carbon_thunk._(ptr %other), !dbg !80
// CHECK:STDOUT:   %OrderedWith.NotEqual.call = icmp ne i32 %self, %__carbon_thunk.call, !dbg !80
// CHECK:STDOUT:   ret i1 %OrderedWith.NotEqual.call, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCallCompareGeneric3.Main(i16 %a, ptr %b) #0 !dbg !81 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %CompareGeneric.call = call i1 @_CCompareGeneric.Main.322aacabcfb78b1a(i16 %a, ptr %b), !dbg !85
// CHECK:STDOUT:   ret i1 %CompareGeneric.call, !dbg !86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CEqual:thunk:EqWith.a294139a6615e163.Core:OrderedWith.Core"(i16 %self, ptr %other) #2 !dbg !87 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.4.temp = alloca i16, align 2, !dbg !91
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.4.temp), !dbg !91
// CHECK:STDOUT:   call void @_ZNK14ConvertToEnum2cvN1C1EEEv.carbon_thunk._(ptr %other, ptr %.4.temp), !dbg !91
// CHECK:STDOUT:   %.8 = load i16, ptr %.4.temp, align 2, !dbg !91
// CHECK:STDOUT:   %OrderedWith.Equal.call = icmp eq i16 %self, %.8, !dbg !91
// CHECK:STDOUT:   ret i1 %OrderedWith.Equal.call, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CNotEqual:thunk:EqWith.a294139a6615e163.Core:OrderedWith.Core"(i16 %self, ptr %other) #2 !dbg !92 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.4.temp = alloca i16, align 2, !dbg !96
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.4.temp), !dbg !96
// CHECK:STDOUT:   call void @_ZNK14ConvertToEnum2cvN1C1EEEv.carbon_thunk._(ptr %other, ptr %.4.temp), !dbg !96
// CHECK:STDOUT:   %.8 = load i16, ptr %.4.temp, align 2, !dbg !96
// CHECK:STDOUT:   %OrderedWith.NotEqual.call = icmp ne i16 %self, %.8, !dbg !96
// CHECK:STDOUT:   ret i1 %OrderedWith.NotEqual.call, !dbg !96
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CCompareGeneric.Main.f8b275c548b6e015(i16 %x, ptr %y) #0 !dbg !97 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.call = call i1 @"_CEqual:thunk:EqWith.83b9626e83cf3dcc.Core:OrderedWith.Core"(i16 %x, ptr %y), !dbg !101
// CHECK:STDOUT:   ret i1 %EqWith.WithSelf.Equal.call, !dbg !102
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CCompareGeneric.Main.f7b86ff4854a74c3(i32 %x, ptr %y) #0 !dbg !103 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.call = call i1 @"_CEqual:thunk:EqWith.e8fe48de943d9652.Core:OrderedWith.Core"(i32 %x, ptr %y), !dbg !107
// CHECK:STDOUT:   ret i1 %EqWith.WithSelf.Equal.call, !dbg !108
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CCompareGeneric.Main.322aacabcfb78b1a(i16 %x, ptr %y) #0 !dbg !109 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.call = call i1 @"_CEqual:thunk:EqWith.a294139a6615e163.Core:OrderedWith.Core"(i16 %x, ptr %y), !dbg !113
// CHECK:STDOUT:   ret i1 %EqWith.WithSelf.Equal.call, !dbg !114
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
// CHECK:STDOUT: !7 = !{!8, !9, i64 0}
// CHECK:STDOUT: !8 = !{!"__libc_errno", !9, i64 0}
// CHECK:STDOUT: !9 = !{!"int", !10, i64 0}
// CHECK:STDOUT: !10 = !{!"omnipotent char", !11, i64 0}
// CHECK:STDOUT: !11 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Compare", linkageName: "_CCompare.Main", scope: null, file: !1, line: 17, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !17)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{!15, !16, !15}
// CHECK:STDOUT: !15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !16 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !17 = !{!18, !19}
// CHECK:STDOUT: !18 = !DILocalVariable(arg: 1, scope: !12, type: !16)
// CHECK:STDOUT: !19 = !DILocalVariable(arg: 2, scope: !12, type: !15)
// CHECK:STDOUT: !20 = !DILocation(line: 18, column: 15, scope: !12)
// CHECK:STDOUT: !21 = !DILocation(line: 18, column: 10, scope: !12)
// CHECK:STDOUT: !22 = !DILocation(line: 18, column: 3, scope: !12)
// CHECK:STDOUT: !23 = !{!24, !24, i64 0}
// CHECK:STDOUT: !24 = !{!"p1 _ZTS13ConvertToEnum", !25, i64 0}
// CHECK:STDOUT: !25 = !{!"any pointer", !10, i64 0}
// CHECK:STDOUT: !26 = !{!25, !25, i64 0}
// CHECK:STDOUT: !27 = !{}
// CHECK:STDOUT: !28 = !{!29, !29, i64 0}
// CHECK:STDOUT: !29 = !{!"_ZTSN1C1EE", !10, i64 0}
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "Compare2", linkageName: "_CCompare2.Main", scope: null, file: !1, line: 21, type: !31, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !34)
// CHECK:STDOUT: !31 = !DISubroutineType(types: !32)
// CHECK:STDOUT: !32 = !{!15, !33, !15}
// CHECK:STDOUT: !33 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !34 = !{!35, !36}
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !30, type: !33)
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 2, scope: !30, type: !15)
// CHECK:STDOUT: !37 = !DILocation(line: 22, column: 15, scope: !30)
// CHECK:STDOUT: !38 = !DILocation(line: 22, column: 10, scope: !30)
// CHECK:STDOUT: !39 = !DILocation(line: 22, column: 3, scope: !30)
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "Compare3", linkageName: "_CCompare3.Main", scope: null, file: !1, line: 25, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !41)
// CHECK:STDOUT: !41 = !{!42, !43}
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 1, scope: !40, type: !16)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 2, scope: !40, type: !15)
// CHECK:STDOUT: !44 = !DILocation(line: 26, column: 15, scope: !40)
// CHECK:STDOUT: !45 = !DILocation(line: 26, column: 10, scope: !40)
// CHECK:STDOUT: !46 = !DILocation(line: 26, column: 3, scope: !40)
// CHECK:STDOUT: !47 = !{!48, !48, i64 0}
// CHECK:STDOUT: !48 = !{!"p1 _ZTS14ConvertToEnum2", !25, i64 0}
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "CallCompareGeneric", linkageName: "_CCallCompareGeneric.Main", scope: null, file: !1, line: 35, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !50)
// CHECK:STDOUT: !50 = !{!51, !52}
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !49, type: !16)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 2, scope: !49, type: !15)
// CHECK:STDOUT: !53 = !DILocation(line: 36, column: 10, scope: !49)
// CHECK:STDOUT: !54 = !DILocation(line: 36, column: 3, scope: !49)
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "Equal", linkageName: "_CEqual:thunk:EqWith.83b9626e83cf3dcc.Core:OrderedWith.Core", scope: null, file: !1, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !56)
// CHECK:STDOUT: !56 = !{!57, !58}
// CHECK:STDOUT: !57 = !DILocalVariable(arg: 1, scope: !55, type: !16)
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 2, scope: !55, type: !15)
// CHECK:STDOUT: !59 = !DILocation(line: 0, scope: !55)
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "NotEqual", linkageName: "_CNotEqual:thunk:EqWith.83b9626e83cf3dcc.Core:OrderedWith.Core", scope: null, file: !1, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !61)
// CHECK:STDOUT: !61 = !{!62, !63}
// CHECK:STDOUT: !62 = !DILocalVariable(arg: 1, scope: !60, type: !16)
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 2, scope: !60, type: !15)
// CHECK:STDOUT: !64 = !DILocation(line: 0, scope: !60)
// CHECK:STDOUT: !65 = distinct !DISubprogram(name: "CallCompareGeneric2", linkageName: "_CCallCompareGeneric2.Main", scope: null, file: !1, line: 39, type: !31, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !66)
// CHECK:STDOUT: !66 = !{!67, !68}
// CHECK:STDOUT: !67 = !DILocalVariable(arg: 1, scope: !65, type: !33)
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 2, scope: !65, type: !15)
// CHECK:STDOUT: !69 = !DILocation(line: 40, column: 10, scope: !65)
// CHECK:STDOUT: !70 = !DILocation(line: 40, column: 3, scope: !65)
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "Equal", linkageName: "_CEqual:thunk:EqWith.e8fe48de943d9652.Core:OrderedWith.Core", scope: null, file: !1, type: !31, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !72)
// CHECK:STDOUT: !72 = !{!73, !74}
// CHECK:STDOUT: !73 = !DILocalVariable(arg: 1, scope: !71, type: !33)
// CHECK:STDOUT: !74 = !DILocalVariable(arg: 2, scope: !71, type: !15)
// CHECK:STDOUT: !75 = !DILocation(line: 0, scope: !71)
// CHECK:STDOUT: !76 = distinct !DISubprogram(name: "NotEqual", linkageName: "_CNotEqual:thunk:EqWith.e8fe48de943d9652.Core:OrderedWith.Core", scope: null, file: !1, type: !31, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !77)
// CHECK:STDOUT: !77 = !{!78, !79}
// CHECK:STDOUT: !78 = !DILocalVariable(arg: 1, scope: !76, type: !33)
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 2, scope: !76, type: !15)
// CHECK:STDOUT: !80 = !DILocation(line: 0, scope: !76)
// CHECK:STDOUT: !81 = distinct !DISubprogram(name: "CallCompareGeneric3", linkageName: "_CCallCompareGeneric3.Main", scope: null, file: !1, line: 43, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !82)
// CHECK:STDOUT: !82 = !{!83, !84}
// CHECK:STDOUT: !83 = !DILocalVariable(arg: 1, scope: !81, type: !16)
// CHECK:STDOUT: !84 = !DILocalVariable(arg: 2, scope: !81, type: !15)
// CHECK:STDOUT: !85 = !DILocation(line: 44, column: 10, scope: !81)
// CHECK:STDOUT: !86 = !DILocation(line: 44, column: 3, scope: !81)
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "Equal", linkageName: "_CEqual:thunk:EqWith.a294139a6615e163.Core:OrderedWith.Core", scope: null, file: !1, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !88)
// CHECK:STDOUT: !88 = !{!89, !90}
// CHECK:STDOUT: !89 = !DILocalVariable(arg: 1, scope: !87, type: !16)
// CHECK:STDOUT: !90 = !DILocalVariable(arg: 2, scope: !87, type: !15)
// CHECK:STDOUT: !91 = !DILocation(line: 0, scope: !87)
// CHECK:STDOUT: !92 = distinct !DISubprogram(name: "NotEqual", linkageName: "_CNotEqual:thunk:EqWith.a294139a6615e163.Core:OrderedWith.Core", scope: null, file: !1, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !93)
// CHECK:STDOUT: !93 = !{!94, !95}
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 1, scope: !92, type: !16)
// CHECK:STDOUT: !95 = !DILocalVariable(arg: 2, scope: !92, type: !15)
// CHECK:STDOUT: !96 = !DILocation(line: 0, scope: !92)
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "CompareGeneric", linkageName: "_CCompareGeneric.Main.f8b275c548b6e015", scope: null, file: !1, line: 29, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !98)
// CHECK:STDOUT: !98 = !{!99, !100}
// CHECK:STDOUT: !99 = !DILocalVariable(arg: 1, scope: !97, type: !16)
// CHECK:STDOUT: !100 = !DILocalVariable(arg: 2, scope: !97, type: !15)
// CHECK:STDOUT: !101 = !DILocation(line: 30, column: 10, scope: !97)
// CHECK:STDOUT: !102 = !DILocation(line: 30, column: 3, scope: !97)
// CHECK:STDOUT: !103 = distinct !DISubprogram(name: "CompareGeneric", linkageName: "_CCompareGeneric.Main.f7b86ff4854a74c3", scope: null, file: !1, line: 29, type: !31, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !104)
// CHECK:STDOUT: !104 = !{!105, !106}
// CHECK:STDOUT: !105 = !DILocalVariable(arg: 1, scope: !103, type: !33)
// CHECK:STDOUT: !106 = !DILocalVariable(arg: 2, scope: !103, type: !15)
// CHECK:STDOUT: !107 = !DILocation(line: 30, column: 10, scope: !103)
// CHECK:STDOUT: !108 = !DILocation(line: 30, column: 3, scope: !103)
// CHECK:STDOUT: !109 = distinct !DISubprogram(name: "CompareGeneric", linkageName: "_CCompareGeneric.Main.322aacabcfb78b1a", scope: null, file: !1, line: 29, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !110)
// CHECK:STDOUT: !110 = !{!111, !112}
// CHECK:STDOUT: !111 = !DILocalVariable(arg: 1, scope: !109, type: !16)
// CHECK:STDOUT: !112 = !DILocalVariable(arg: 2, scope: !109, type: !15)
// CHECK:STDOUT: !113 = !DILocation(line: 30, column: 10, scope: !109)
// CHECK:STDOUT: !114 = !DILocation(line: 30, column: 3, scope: !109)
// CHECK:STDOUT:
