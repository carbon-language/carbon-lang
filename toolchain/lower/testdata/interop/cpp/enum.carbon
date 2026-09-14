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
// CHECK:STDOUT:   %.loc7_16.1.temp = alloca i16, align 2, !dbg !13
// CHECK:STDOUT:   %.loc8_16.1.temp = alloca i16, align 2, !dbg !15
// CHECK:STDOUT:   %.loc9_16.1.temp = alloca i16, align 2, !dbg !17
// CHECK:STDOUT:   %.loc10_16.1.temp = alloca i16, align 2, !dbg !19
// CHECK:STDOUT:   %.loc11_18.1.temp = alloca i16, align 2, !dbg !21
// CHECK:STDOUT:   %.loc12_18.1.temp = alloca i16, align 2, !dbg !23
// CHECK:STDOUT:   %.loc13_18.1.temp = alloca i16, align 2, !dbg !25
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_16.1.temp), !dbg !13
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr @int_0), !dbg !14
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc8_16.1.temp), !dbg !15
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr @int_0), !dbg !16
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc9_16.1.temp), !dbg !17
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr @int_7), !dbg !18
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc10_16.1.temp), !dbg !19
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr @int_8), !dbg !20
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc11_18.1.temp), !dbg !21
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr @int_0), !dbg !22
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc12_18.1.temp), !dbg !23
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr @int_7), !dbg !24
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc13_18.1.temp), !dbg !25
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr @int_8), !dbg !26
// CHECK:STDOUT:   ret void, !dbg !27
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
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_ZN1C1FENS_1EE.carbon_thunk._, { 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!34, !35, !36, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!39}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "pass_as_arg.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !7 = !{null}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Pass", linkageName: "_CPass.Main", scope: null, file: !4, line: 6, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !13 = !DILocation(line: 7, column: 11, scope: !12)
// CHECK:STDOUT: !14 = !DILocation(line: 7, column: 3, scope: !12)
// CHECK:STDOUT: !15 = !DILocation(line: 8, column: 11, scope: !12)
// CHECK:STDOUT: !16 = !DILocation(line: 8, column: 3, scope: !12)
// CHECK:STDOUT: !17 = !DILocation(line: 9, column: 11, scope: !12)
// CHECK:STDOUT: !18 = !DILocation(line: 9, column: 3, scope: !12)
// CHECK:STDOUT: !19 = !DILocation(line: 10, column: 11, scope: !12)
// CHECK:STDOUT: !20 = !DILocation(line: 10, column: 3, scope: !12)
// CHECK:STDOUT: !21 = !DILocation(line: 11, column: 11, scope: !12)
// CHECK:STDOUT: !22 = !DILocation(line: 11, column: 3, scope: !12)
// CHECK:STDOUT: !23 = !DILocation(line: 12, column: 11, scope: !12)
// CHECK:STDOUT: !24 = !DILocation(line: 12, column: 3, scope: !12)
// CHECK:STDOUT: !25 = !DILocation(line: 13, column: 11, scope: !12)
// CHECK:STDOUT: !26 = !DILocation(line: 13, column: 3, scope: !12)
// CHECK:STDOUT: !27 = !DILocation(line: 6, column: 1, scope: !12)
// CHECK:STDOUT: !28 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !29 = !{!"omnipotent char", !28, i64 0}
// CHECK:STDOUT: !30 = !{!"any pointer", !29, i64 0}
// CHECK:STDOUT: !31 = !{!30, !30, i64 0}
// CHECK:STDOUT: !32 = !{!"_ZTSN1C1EE", !29, i64 0}
// CHECK:STDOUT: !33 = !{!32, !32, i64 0}
// CHECK:STDOUT: !34 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !35 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !36 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !37 = !{!"int", !29, i64 0}
// CHECK:STDOUT: !38 = !{!"__libc_errno", !37, i64 0}
// CHECK:STDOUT: !39 = !{!38, !37, i64 0}
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
// CHECK:STDOUT: define void @_CPassEnum.Main() #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CTakeI16.Main(i16 7), !dbg !45
// CHECK:STDOUT:   ret void, !dbg !46
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertEnumToI16.Main(i16 %e) #0 !dbg !47 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CTakeI16.Main(i16 %e), !dbg !51
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertI16ToEnum.Main(i16 %n) #0 !dbg !54 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17_13.3.temp = alloca i16, align 2, !dbg !57
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc17_13.3.temp), !dbg !57
// CHECK:STDOUT:   store i16 %n, ptr %.loc17_13.3.temp, align 2, !dbg !57
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr %.loc17_13.3.temp), !dbg !58
// CHECK:STDOUT:   ret void, !dbg !59
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!41}
// CHECK:STDOUT: !llvm.module.flags = !{!34, !35, !36, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!39}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !{null}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !28 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !29 = !{!"omnipotent char", !28, i64 0}
// CHECK:STDOUT: !30 = !{!"any pointer", !29, i64 0}
// CHECK:STDOUT: !31 = !{!30, !30, i64 0}
// CHECK:STDOUT: !32 = !{!"_ZTSN1C1EE", !29, i64 0}
// CHECK:STDOUT: !33 = !{!32, !32, i64 0}
// CHECK:STDOUT: !34 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !35 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !36 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !37 = !{!"int", !29, i64 0}
// CHECK:STDOUT: !38 = !{!"__libc_errno", !37, i64 0}
// CHECK:STDOUT: !39 = !{!38, !37, i64 0}
// CHECK:STDOUT: !40 = !DIFile(filename: "convert.carbon", directory: "")
// CHECK:STDOUT: !41 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !40, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !42 = !{null, !6}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "PassEnum", linkageName: "_CPassEnum.Main", scope: null, file: !40, line: 8, type: !8, spFlags: DISPFlagDefinition, unit: !41)
// CHECK:STDOUT: !45 = !DILocation(line: 9, column: 3, scope: !44)
// CHECK:STDOUT: !46 = !DILocation(line: 8, column: 1, scope: !44)
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "ConvertEnumToI16", linkageName: "_CConvertEnumToI16.Main", scope: null, file: !40, line: 12, type: !43, spFlags: DISPFlagDefinition, unit: !41, retainedNodes: !53)
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 1, scope: !47, type: !6)
// CHECK:STDOUT: !51 = !DILocation(line: 13, column: 3, scope: !47)
// CHECK:STDOUT: !52 = !DILocation(line: 12, column: 1, scope: !47)
// CHECK:STDOUT: !53 = !{!48}
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "ConvertI16ToEnum", linkageName: "_CConvertI16ToEnum.Main", scope: null, file: !40, line: 16, type: !43, spFlags: DISPFlagDefinition, unit: !41, retainedNodes: !60)
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 1, scope: !54, type: !6)
// CHECK:STDOUT: !57 = !DILocation(line: 17, column: 11, scope: !54)
// CHECK:STDOUT: !58 = !DILocation(line: 17, column: 3, scope: !54)
// CHECK:STDOUT: !59 = !DILocation(line: 16, column: 1, scope: !54)
// CHECK:STDOUT: !60 = !{!55}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'use_bitmask.carbon'
// CHECK:STDOUT: source_filename = "use_bitmask.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCompileTime.Main() #0 !dbg !68 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 0), !dbg !69
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 5), !dbg !70
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 5), !dbg !71
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 -2), !dbg !72
// CHECK:STDOUT:   ret void, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z4Take4Bits(i32 noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CRuntime.Main(i32 %a, i32 %b) #0 !dbg !74 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Op.call.loc14 = and i32 %a, %b, !dbg !79
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 %Op.call.loc14), !dbg !81
// CHECK:STDOUT:   %Op.call.loc15 = or i32 %a, %b, !dbg !82
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 %Op.call.loc15), !dbg !84
// CHECK:STDOUT:   %Op.call.loc16 = xor i32 %a, %b, !dbg !85
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 %Op.call.loc16), !dbg !87
// CHECK:STDOUT:   %Op.call.loc20 = xor i32 -1, %a, !dbg !89
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 %Op.call.loc20), !dbg !90
// CHECK:STDOUT:   ret void, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_Z4Take4Bits, { 0, 1, 2, 3, 7, 6, 5, 4 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!62}
// CHECK:STDOUT: !llvm.module.flags = !{!34, !35, !36, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!39}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !{null}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !28 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !29 = !{!"omnipotent char", !28, i64 0}
// CHECK:STDOUT: !34 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !35 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !36 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !37 = !{!"int", !29, i64 0}
// CHECK:STDOUT: !38 = !{!"__libc_errno", !37, i64 0}
// CHECK:STDOUT: !39 = !{!38, !37, i64 0}
// CHECK:STDOUT: !61 = !DIFile(filename: "use_bitmask.carbon", directory: "")
// CHECK:STDOUT: !62 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !61, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !63 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !66 = !{null, !63, !63}
// CHECK:STDOUT: !67 = !DISubroutineType(types: !66)
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "CompileTime", linkageName: "_CCompileTime.Main", scope: null, file: !61, line: 6, type: !8, spFlags: DISPFlagDefinition, unit: !62)
// CHECK:STDOUT: !69 = !DILocation(line: 7, column: 3, scope: !68)
// CHECK:STDOUT: !70 = !DILocation(line: 8, column: 3, scope: !68)
// CHECK:STDOUT: !71 = !DILocation(line: 9, column: 3, scope: !68)
// CHECK:STDOUT: !72 = !DILocation(line: 10, column: 3, scope: !68)
// CHECK:STDOUT: !73 = !DILocation(line: 6, column: 1, scope: !68)
// CHECK:STDOUT: !74 = distinct !DISubprogram(name: "Runtime", linkageName: "_CRuntime.Main", scope: null, file: !61, line: 13, type: !67, spFlags: DISPFlagDefinition, unit: !62, retainedNodes: !92)
// CHECK:STDOUT: !75 = !DILocalVariable(arg: 1, scope: !74, type: !63)
// CHECK:STDOUT: !76 = !DILocalVariable(arg: 2, scope: !74, type: !63)
// CHECK:STDOUT: !79 = !DILocation(line: 14, column: 12, scope: !74)
// CHECK:STDOUT: !81 = !DILocation(line: 14, column: 3, scope: !74)
// CHECK:STDOUT: !82 = !DILocation(line: 15, column: 12, scope: !74)
// CHECK:STDOUT: !84 = !DILocation(line: 15, column: 3, scope: !74)
// CHECK:STDOUT: !85 = !DILocation(line: 16, column: 12, scope: !74)
// CHECK:STDOUT: !87 = !DILocation(line: 16, column: 3, scope: !74)
// CHECK:STDOUT: !89 = !DILocation(line: 20, column: 12, scope: !74)
// CHECK:STDOUT: !90 = !DILocation(line: 20, column: 3, scope: !74)
// CHECK:STDOUT: !91 = !DILocation(line: 13, column: 1, scope: !74)
// CHECK:STDOUT: !92 = !{!75, !76}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'compare.carbon'
// CHECK:STDOUT: source_filename = "compare.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCompare.Main(i16 %a, i16 %b) #0 !dbg !97 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Equal.call = icmp eq i16 %a, %b, !dbg !104
// CHECK:STDOUT:   ret i1 %Equal.call, !dbg !106
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCallCompareGeneric.Main(i16 %a, i16 %b) #0 !dbg !108 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %CompareGeneric.call = call i1 @_CCompareGeneric.Main.80c3fc0239809458(i16 %a, i16 %b), !dbg !117
// CHECK:STDOUT:   ret i1 %CompareGeneric.call, !dbg !118
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CCompareGeneric.Main.80c3fc0239809458(i16 %x, i16 %y) #0 !dbg !120 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.call = icmp eq i16 %x, %y, !dbg !127
// CHECK:STDOUT:   ret i1 %EqWith.WithSelf.Equal.call, !dbg !129
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!94}
// CHECK:STDOUT: !llvm.module.flags = !{!34, !35, !36, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!39}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !28 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !29 = !{!"omnipotent char", !28, i64 0}
// CHECK:STDOUT: !34 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !35 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !36 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !37 = !{!"int", !29, i64 0}
// CHECK:STDOUT: !38 = !{!"__libc_errno", !37, i64 0}
// CHECK:STDOUT: !39 = !{!38, !37, i64 0}
// CHECK:STDOUT: !93 = !DIFile(filename: "compare.carbon", directory: "")
// CHECK:STDOUT: !94 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !93, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !95 = !{!9, !6, !6}
// CHECK:STDOUT: !96 = !DISubroutineType(types: !95)
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "Compare", linkageName: "_CCompare.Main", scope: null, file: !93, line: 6, type: !96, spFlags: DISPFlagDefinition, unit: !94, retainedNodes: !107)
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 1, scope: !97, type: !6)
// CHECK:STDOUT: !99 = !DILocalVariable(arg: 2, scope: !97, type: !6)
// CHECK:STDOUT: !104 = !DILocation(line: 7, column: 10, scope: !97)
// CHECK:STDOUT: !106 = !DILocation(line: 7, column: 3, scope: !97)
// CHECK:STDOUT: !107 = !{!98, !99}
// CHECK:STDOUT: !108 = distinct !DISubprogram(name: "CallCompareGeneric", linkageName: "_CCallCompareGeneric.Main", scope: null, file: !93, line: 14, type: !96, spFlags: DISPFlagDefinition, unit: !94, retainedNodes: !119)
// CHECK:STDOUT: !109 = !DILocalVariable(arg: 1, scope: !108, type: !6)
// CHECK:STDOUT: !110 = !DILocalVariable(arg: 2, scope: !108, type: !6)
// CHECK:STDOUT: !117 = !DILocation(line: 15, column: 10, scope: !108)
// CHECK:STDOUT: !118 = !DILocation(line: 15, column: 3, scope: !108)
// CHECK:STDOUT: !119 = !{!109, !110}
// CHECK:STDOUT: !120 = distinct !DISubprogram(name: "CompareGeneric", linkageName: "_CCompareGeneric.Main.80c3fc0239809458", scope: null, file: !93, line: 10, type: !96, spFlags: DISPFlagDefinition, unit: !94, retainedNodes: !130)
// CHECK:STDOUT: !121 = !DILocalVariable(arg: 1, scope: !120, type: !6)
// CHECK:STDOUT: !122 = !DILocalVariable(arg: 2, scope: !120, type: !6)
// CHECK:STDOUT: !127 = !DILocation(line: 11, column: 10, scope: !120)
// CHECK:STDOUT: !129 = !DILocation(line: 11, column: 3, scope: !120)
// CHECK:STDOUT: !130 = !{!121, !122}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'compare_class.carbon'
// CHECK:STDOUT: source_filename = "compare_class.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCompare.Main(i16 %a, ptr %b) #0 !dbg !141 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_15.1.temp = alloca i16, align 2, !dbg !149
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_15.1.temp), !dbg !149
// CHECK:STDOUT:   call void @_ZNK13ConvertToEnumcvN1C1EEEv.carbon_thunk._(ptr %b, ptr %.loc18_15.1.temp), !dbg !149
// CHECK:STDOUT:   %.loc18_15.5 = load i16, ptr %.loc18_15.1.temp, align 2, !dbg !149
// CHECK:STDOUT:   %Equal.call = icmp eq i16 %a, %.loc18_15.5, !dbg !148
// CHECK:STDOUT:   ret i1 %Equal.call, !dbg !150
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZNK13ConvertToEnumcvN1C1EEEv.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !274
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !31
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !31
// CHECK:STDOUT:   %1 = load ptr, ptr %this.addr, align 8, !tbaa !274, !nonnull !276
// CHECK:STDOUT:   %call = call noundef signext i16 @_ZNK13ConvertToEnumcvN1C1EEEv(ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   store i16 %call, ptr %0, align 2, !tbaa !33
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCompare2.Main(i32 %a, ptr %b) #0 !dbg !152 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %__carbon_thunk.call = call i32 @_ZNK13ConvertToEnumcv2E2Ev.carbon_thunk._(ptr %b), !dbg !160
// CHECK:STDOUT:   %Equal.call = icmp eq i32 %a, %__carbon_thunk.call, !dbg !159
// CHECK:STDOUT:   ret i1 %Equal.call, !dbg !161
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal noundef i32 @_ZNK13ConvertToEnumcv2E2Ev.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !274
// CHECK:STDOUT:   %0 = load ptr, ptr %this.addr, align 8, !tbaa !274, !nonnull !276
// CHECK:STDOUT:   %call = call noundef i32 @_ZNK13ConvertToEnumcv2E2Ev(ptr noundef nonnull align 1 dereferenceable(1) %0)
// CHECK:STDOUT:   ret i32 %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCompare3.Main(i16 %a, ptr %b) #0 !dbg !163 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc26_15.1.temp = alloca i16, align 2, !dbg !171
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc26_15.1.temp), !dbg !171
// CHECK:STDOUT:   call void @_ZNK14ConvertToEnum2cvN1C1EEEv.carbon_thunk._(ptr %b, ptr %.loc26_15.1.temp), !dbg !171
// CHECK:STDOUT:   %.loc26_15.5 = load i16, ptr %.loc26_15.1.temp, align 2, !dbg !171
// CHECK:STDOUT:   %Equal.call = icmp eq i16 %a, %.loc26_15.5, !dbg !170
// CHECK:STDOUT:   ret i1 %Equal.call, !dbg !172
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZNK14ConvertToEnum2cvN1C1EEEv.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !278
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !31
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !31
// CHECK:STDOUT:   %1 = load ptr, ptr %this.addr, align 8, !tbaa !278, !nonnull !276
// CHECK:STDOUT:   %call = call noundef signext i16 @_ZNK14ConvertToEnum2cvN1C1EEEv(ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   store i16 %call, ptr %0, align 2, !tbaa !33
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCallCompareGeneric.Main(i16 %a, ptr %b) #0 !dbg !174 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %CompareGeneric.call = call i1 @_CCompareGeneric.Main.215a23369ed4d589(i16 %a, ptr %b), !dbg !183
// CHECK:STDOUT:   ret i1 %CompareGeneric.call, !dbg !184
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CEqual:thunk:EqWith.83b9626e83cf3dcc.Core:enclosed"(i16 %self, ptr %other) #2 !dbg !186 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.4.temp = alloca i16, align 2, !dbg !189
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.4.temp), !dbg !189
// CHECK:STDOUT:   call void @_ZNK13ConvertToEnumcvN1C1EEEv.carbon_thunk._(ptr %other, ptr %.4.temp), !dbg !189
// CHECK:STDOUT:   %.8 = load i16, ptr %.4.temp, align 2, !dbg !189
// CHECK:STDOUT:   %Equal.call = icmp eq i16 %self, %.8, !dbg !189
// CHECK:STDOUT:   ret i1 %Equal.call, !dbg !189
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CNotEqual:thunk:EqWith.83b9626e83cf3dcc.Core:enclosed"(i16 %self, ptr %other) #2 !dbg !191 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.4.temp = alloca i16, align 2, !dbg !194
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.4.temp), !dbg !194
// CHECK:STDOUT:   call void @_ZNK13ConvertToEnumcvN1C1EEEv.carbon_thunk._(ptr %other, ptr %.4.temp), !dbg !194
// CHECK:STDOUT:   %.8 = load i16, ptr %.4.temp, align 2, !dbg !194
// CHECK:STDOUT:   %NotEqual.call = icmp ne i16 %self, %.8, !dbg !194
// CHECK:STDOUT:   ret i1 %NotEqual.call, !dbg !194
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCallCompareGeneric2.Main(i32 %a, ptr %b) #0 !dbg !196 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %CompareGeneric.call = call i1 @_CCompareGeneric.Main.a5489e454367d917(i32 %a, ptr %b), !dbg !205
// CHECK:STDOUT:   ret i1 %CompareGeneric.call, !dbg !206
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CEqual:thunk:EqWith.e8fe48de943d9652.Core:enclosed"(i32 %self, ptr %other) #2 !dbg !208 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %__carbon_thunk.call = call i32 @_ZNK13ConvertToEnumcv2E2Ev.carbon_thunk._(ptr %other), !dbg !211
// CHECK:STDOUT:   %Equal.call = icmp eq i32 %self, %__carbon_thunk.call, !dbg !211
// CHECK:STDOUT:   ret i1 %Equal.call, !dbg !211
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CNotEqual:thunk:EqWith.e8fe48de943d9652.Core:enclosed"(i32 %self, ptr %other) #2 !dbg !213 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %__carbon_thunk.call = call i32 @_ZNK13ConvertToEnumcv2E2Ev.carbon_thunk._(ptr %other), !dbg !216
// CHECK:STDOUT:   %NotEqual.call = icmp ne i32 %self, %__carbon_thunk.call, !dbg !216
// CHECK:STDOUT:   ret i1 %NotEqual.call, !dbg !216
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCallCompareGeneric3.Main(i16 %a, ptr %b) #0 !dbg !218 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %CompareGeneric.call = call i1 @_CCompareGeneric.Main.64b737572447efad(i16 %a, ptr %b), !dbg !227
// CHECK:STDOUT:   ret i1 %CompareGeneric.call, !dbg !228
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CEqual:thunk:EqWith.a294139a6615e163.Core:enclosed"(i16 %self, ptr %other) #2 !dbg !230 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.4.temp = alloca i16, align 2, !dbg !233
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.4.temp), !dbg !233
// CHECK:STDOUT:   call void @_ZNK14ConvertToEnum2cvN1C1EEEv.carbon_thunk._(ptr %other, ptr %.4.temp), !dbg !233
// CHECK:STDOUT:   %.8 = load i16, ptr %.4.temp, align 2, !dbg !233
// CHECK:STDOUT:   %Equal.call = icmp eq i16 %self, %.8, !dbg !233
// CHECK:STDOUT:   ret i1 %Equal.call, !dbg !233
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CNotEqual:thunk:EqWith.a294139a6615e163.Core:enclosed"(i16 %self, ptr %other) #2 !dbg !235 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.4.temp = alloca i16, align 2, !dbg !238
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.4.temp), !dbg !238
// CHECK:STDOUT:   call void @_ZNK14ConvertToEnum2cvN1C1EEEv.carbon_thunk._(ptr %other, ptr %.4.temp), !dbg !238
// CHECK:STDOUT:   %.8 = load i16, ptr %.4.temp, align 2, !dbg !238
// CHECK:STDOUT:   %NotEqual.call = icmp ne i16 %self, %.8, !dbg !238
// CHECK:STDOUT:   ret i1 %NotEqual.call, !dbg !238
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CCompareGeneric.Main.215a23369ed4d589(i16 %x, ptr %y) #0 !dbg !240 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.call = call i1 @"_CEqual:thunk:EqWith.83b9626e83cf3dcc.Core:enclosed"(i16 %x, ptr %y), !dbg !247
// CHECK:STDOUT:   ret i1 %EqWith.WithSelf.Equal.call, !dbg !249
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CCompareGeneric.Main.a5489e454367d917(i32 %x, ptr %y) #0 !dbg !251 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.call = call i1 @"_CEqual:thunk:EqWith.e8fe48de943d9652.Core:enclosed"(i32 %x, ptr %y), !dbg !258
// CHECK:STDOUT:   ret i1 %EqWith.WithSelf.Equal.call, !dbg !260
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CCompareGeneric.Main.64b737572447efad(i16 %x, ptr %y) #0 !dbg !262 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.call = call i1 @"_CEqual:thunk:EqWith.a294139a6615e163.Core:enclosed"(i16 %x, ptr %y), !dbg !269
// CHECK:STDOUT:   ret i1 %EqWith.WithSelf.Equal.call, !dbg !271
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!132}
// CHECK:STDOUT: !llvm.module.flags = !{!34, !35, !36, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!39}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !28 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !29 = !{!"omnipotent char", !28, i64 0}
// CHECK:STDOUT: !30 = !{!"any pointer", !29, i64 0}
// CHECK:STDOUT: !31 = !{!30, !30, i64 0}
// CHECK:STDOUT: !32 = !{!"_ZTSN1C1EE", !29, i64 0}
// CHECK:STDOUT: !33 = !{!32, !32, i64 0}
// CHECK:STDOUT: !34 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !35 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !36 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !37 = !{!"int", !29, i64 0}
// CHECK:STDOUT: !38 = !{!"__libc_errno", !37, i64 0}
// CHECK:STDOUT: !39 = !{!38, !37, i64 0}
// CHECK:STDOUT: !63 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !131 = !DIFile(filename: "compare_class.carbon", directory: "")
// CHECK:STDOUT: !132 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !131, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !133 = !{!9, !6, !9}
// CHECK:STDOUT: !134 = !DISubroutineType(types: !133)
// CHECK:STDOUT: !137 = !{!9, !63, !9}
// CHECK:STDOUT: !138 = !DISubroutineType(types: !137)
// CHECK:STDOUT: !141 = distinct !DISubprogram(name: "Compare", linkageName: "_CCompare.Main", scope: null, file: !131, line: 17, type: !134, spFlags: DISPFlagDefinition, unit: !132, retainedNodes: !151)
// CHECK:STDOUT: !142 = !DILocalVariable(arg: 1, scope: !141, type: !6)
// CHECK:STDOUT: !143 = !DILocalVariable(arg: 2, scope: !141, type: !9)
// CHECK:STDOUT: !148 = !DILocation(line: 18, column: 10, scope: !141)
// CHECK:STDOUT: !149 = !DILocation(line: 18, column: 15, scope: !141)
// CHECK:STDOUT: !150 = !DILocation(line: 18, column: 3, scope: !141)
// CHECK:STDOUT: !151 = !{!142, !143}
// CHECK:STDOUT: !152 = distinct !DISubprogram(name: "Compare2", linkageName: "_CCompare2.Main", scope: null, file: !131, line: 21, type: !138, spFlags: DISPFlagDefinition, unit: !132, retainedNodes: !162)
// CHECK:STDOUT: !153 = !DILocalVariable(arg: 1, scope: !152, type: !63)
// CHECK:STDOUT: !154 = !DILocalVariable(arg: 2, scope: !152, type: !9)
// CHECK:STDOUT: !159 = !DILocation(line: 22, column: 10, scope: !152)
// CHECK:STDOUT: !160 = !DILocation(line: 22, column: 15, scope: !152)
// CHECK:STDOUT: !161 = !DILocation(line: 22, column: 3, scope: !152)
// CHECK:STDOUT: !162 = !{!153, !154}
// CHECK:STDOUT: !163 = distinct !DISubprogram(name: "Compare3", linkageName: "_CCompare3.Main", scope: null, file: !131, line: 25, type: !134, spFlags: DISPFlagDefinition, unit: !132, retainedNodes: !173)
// CHECK:STDOUT: !164 = !DILocalVariable(arg: 1, scope: !163, type: !6)
// CHECK:STDOUT: !165 = !DILocalVariable(arg: 2, scope: !163, type: !9)
// CHECK:STDOUT: !170 = !DILocation(line: 26, column: 10, scope: !163)
// CHECK:STDOUT: !171 = !DILocation(line: 26, column: 15, scope: !163)
// CHECK:STDOUT: !172 = !DILocation(line: 26, column: 3, scope: !163)
// CHECK:STDOUT: !173 = !{!164, !165}
// CHECK:STDOUT: !174 = distinct !DISubprogram(name: "CallCompareGeneric", linkageName: "_CCallCompareGeneric.Main", scope: null, file: !131, line: 35, type: !134, spFlags: DISPFlagDefinition, unit: !132, retainedNodes: !185)
// CHECK:STDOUT: !175 = !DILocalVariable(arg: 1, scope: !174, type: !6)
// CHECK:STDOUT: !176 = !DILocalVariable(arg: 2, scope: !174, type: !9)
// CHECK:STDOUT: !183 = !DILocation(line: 36, column: 10, scope: !174)
// CHECK:STDOUT: !184 = !DILocation(line: 36, column: 3, scope: !174)
// CHECK:STDOUT: !185 = !{!175, !176}
// CHECK:STDOUT: !186 = distinct !DISubprogram(name: "Equal", linkageName: "_CEqual:thunk:EqWith.83b9626e83cf3dcc.Core:enclosed", scope: null, file: !131, type: !134, spFlags: DISPFlagDefinition, unit: !132, retainedNodes: !190)
// CHECK:STDOUT: !187 = !DILocalVariable(arg: 1, scope: !186, type: !6)
// CHECK:STDOUT: !188 = !DILocalVariable(arg: 2, scope: !186, type: !9)
// CHECK:STDOUT: !189 = !DILocation(line: 0, scope: !186)
// CHECK:STDOUT: !190 = !{!187, !188}
// CHECK:STDOUT: !191 = distinct !DISubprogram(name: "NotEqual", linkageName: "_CNotEqual:thunk:EqWith.83b9626e83cf3dcc.Core:enclosed", scope: null, file: !131, type: !134, spFlags: DISPFlagDefinition, unit: !132, retainedNodes: !195)
// CHECK:STDOUT: !192 = !DILocalVariable(arg: 1, scope: !191, type: !6)
// CHECK:STDOUT: !193 = !DILocalVariable(arg: 2, scope: !191, type: !9)
// CHECK:STDOUT: !194 = !DILocation(line: 0, scope: !191)
// CHECK:STDOUT: !195 = !{!192, !193}
// CHECK:STDOUT: !196 = distinct !DISubprogram(name: "CallCompareGeneric2", linkageName: "_CCallCompareGeneric2.Main", scope: null, file: !131, line: 39, type: !138, spFlags: DISPFlagDefinition, unit: !132, retainedNodes: !207)
// CHECK:STDOUT: !197 = !DILocalVariable(arg: 1, scope: !196, type: !63)
// CHECK:STDOUT: !198 = !DILocalVariable(arg: 2, scope: !196, type: !9)
// CHECK:STDOUT: !205 = !DILocation(line: 40, column: 10, scope: !196)
// CHECK:STDOUT: !206 = !DILocation(line: 40, column: 3, scope: !196)
// CHECK:STDOUT: !207 = !{!197, !198}
// CHECK:STDOUT: !208 = distinct !DISubprogram(name: "Equal", linkageName: "_CEqual:thunk:EqWith.e8fe48de943d9652.Core:enclosed", scope: null, file: !131, type: !138, spFlags: DISPFlagDefinition, unit: !132, retainedNodes: !212)
// CHECK:STDOUT: !209 = !DILocalVariable(arg: 1, scope: !208, type: !63)
// CHECK:STDOUT: !210 = !DILocalVariable(arg: 2, scope: !208, type: !9)
// CHECK:STDOUT: !211 = !DILocation(line: 0, scope: !208)
// CHECK:STDOUT: !212 = !{!209, !210}
// CHECK:STDOUT: !213 = distinct !DISubprogram(name: "NotEqual", linkageName: "_CNotEqual:thunk:EqWith.e8fe48de943d9652.Core:enclosed", scope: null, file: !131, type: !138, spFlags: DISPFlagDefinition, unit: !132, retainedNodes: !217)
// CHECK:STDOUT: !214 = !DILocalVariable(arg: 1, scope: !213, type: !63)
// CHECK:STDOUT: !215 = !DILocalVariable(arg: 2, scope: !213, type: !9)
// CHECK:STDOUT: !216 = !DILocation(line: 0, scope: !213)
// CHECK:STDOUT: !217 = !{!214, !215}
// CHECK:STDOUT: !218 = distinct !DISubprogram(name: "CallCompareGeneric3", linkageName: "_CCallCompareGeneric3.Main", scope: null, file: !131, line: 43, type: !134, spFlags: DISPFlagDefinition, unit: !132, retainedNodes: !229)
// CHECK:STDOUT: !219 = !DILocalVariable(arg: 1, scope: !218, type: !6)
// CHECK:STDOUT: !220 = !DILocalVariable(arg: 2, scope: !218, type: !9)
// CHECK:STDOUT: !227 = !DILocation(line: 44, column: 10, scope: !218)
// CHECK:STDOUT: !228 = !DILocation(line: 44, column: 3, scope: !218)
// CHECK:STDOUT: !229 = !{!219, !220}
// CHECK:STDOUT: !230 = distinct !DISubprogram(name: "Equal", linkageName: "_CEqual:thunk:EqWith.a294139a6615e163.Core:enclosed", scope: null, file: !131, type: !134, spFlags: DISPFlagDefinition, unit: !132, retainedNodes: !234)
// CHECK:STDOUT: !231 = !DILocalVariable(arg: 1, scope: !230, type: !6)
// CHECK:STDOUT: !232 = !DILocalVariable(arg: 2, scope: !230, type: !9)
// CHECK:STDOUT: !233 = !DILocation(line: 0, scope: !230)
// CHECK:STDOUT: !234 = !{!231, !232}
// CHECK:STDOUT: !235 = distinct !DISubprogram(name: "NotEqual", linkageName: "_CNotEqual:thunk:EqWith.a294139a6615e163.Core:enclosed", scope: null, file: !131, type: !134, spFlags: DISPFlagDefinition, unit: !132, retainedNodes: !239)
// CHECK:STDOUT: !236 = !DILocalVariable(arg: 1, scope: !235, type: !6)
// CHECK:STDOUT: !237 = !DILocalVariable(arg: 2, scope: !235, type: !9)
// CHECK:STDOUT: !238 = !DILocation(line: 0, scope: !235)
// CHECK:STDOUT: !239 = !{!236, !237}
// CHECK:STDOUT: !240 = distinct !DISubprogram(name: "CompareGeneric", linkageName: "_CCompareGeneric.Main.215a23369ed4d589", scope: null, file: !131, line: 29, type: !134, spFlags: DISPFlagDefinition, unit: !132, retainedNodes: !250)
// CHECK:STDOUT: !241 = !DILocalVariable(arg: 1, scope: !240, type: !6)
// CHECK:STDOUT: !242 = !DILocalVariable(arg: 2, scope: !240, type: !9)
// CHECK:STDOUT: !247 = !DILocation(line: 30, column: 10, scope: !240)
// CHECK:STDOUT: !249 = !DILocation(line: 30, column: 3, scope: !240)
// CHECK:STDOUT: !250 = !{!241, !242}
// CHECK:STDOUT: !251 = distinct !DISubprogram(name: "CompareGeneric", linkageName: "_CCompareGeneric.Main.a5489e454367d917", scope: null, file: !131, line: 29, type: !138, spFlags: DISPFlagDefinition, unit: !132, retainedNodes: !261)
// CHECK:STDOUT: !252 = !DILocalVariable(arg: 1, scope: !251, type: !63)
// CHECK:STDOUT: !253 = !DILocalVariable(arg: 2, scope: !251, type: !9)
// CHECK:STDOUT: !258 = !DILocation(line: 30, column: 10, scope: !251)
// CHECK:STDOUT: !260 = !DILocation(line: 30, column: 3, scope: !251)
// CHECK:STDOUT: !261 = !{!252, !253}
// CHECK:STDOUT: !262 = distinct !DISubprogram(name: "CompareGeneric", linkageName: "_CCompareGeneric.Main.64b737572447efad", scope: null, file: !131, line: 29, type: !134, spFlags: DISPFlagDefinition, unit: !132, retainedNodes: !272)
// CHECK:STDOUT: !263 = !DILocalVariable(arg: 1, scope: !262, type: !6)
// CHECK:STDOUT: !264 = !DILocalVariable(arg: 2, scope: !262, type: !9)
// CHECK:STDOUT: !269 = !DILocation(line: 30, column: 10, scope: !262)
// CHECK:STDOUT: !271 = !DILocation(line: 30, column: 3, scope: !262)
// CHECK:STDOUT: !272 = !{!263, !264}
// CHECK:STDOUT: !273 = !{!"p1 _ZTS13ConvertToEnum", !30, i64 0}
// CHECK:STDOUT: !274 = !{!273, !273, i64 0}
// CHECK:STDOUT: !276 = !{}
// CHECK:STDOUT: !277 = !{!"p1 _ZTS14ConvertToEnum2", !30, i64 0}
// CHECK:STDOUT: !278 = !{!277, !277, i64 0}
// CHECK:STDOUT:
