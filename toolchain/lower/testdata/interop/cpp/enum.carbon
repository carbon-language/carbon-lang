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
// CHECK:STDOUT:   %.loc8_16.1.temp = alloca i16, align 2, !dbg !17
// CHECK:STDOUT:   %.loc9_16.1.temp = alloca i16, align 2, !dbg !19
// CHECK:STDOUT:   %.loc10_16.1.temp = alloca i16, align 2, !dbg !21
// CHECK:STDOUT:   %.loc11_18.1.temp = alloca i16, align 2, !dbg !23
// CHECK:STDOUT:   %.loc12_18.1.temp = alloca i16, align 2, !dbg !25
// CHECK:STDOUT:   %.loc13_18.1.temp = alloca i16, align 2, !dbg !27
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_16.1.temp), !dbg !13
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr @int_0), !dbg !14
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.63b9759f1df1768a(ptr @int_0), !dbg !13
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc8_16.1.temp), !dbg !17
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr @int_0), !dbg !18
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.63b9759f1df1768a(ptr @int_0), !dbg !17
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc9_16.1.temp), !dbg !19
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr @int_7), !dbg !20
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.63b9759f1df1768a(ptr @int_7), !dbg !19
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc10_16.1.temp), !dbg !21
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr @int_8), !dbg !22
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.63b9759f1df1768a(ptr @int_8), !dbg !21
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc11_18.1.temp), !dbg !23
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr @int_0), !dbg !24
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.63b9759f1df1768a(ptr @int_0), !dbg !23
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc12_18.1.temp), !dbg !25
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr @int_7), !dbg !26
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.63b9759f1df1768a(ptr @int_7), !dbg !25
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc13_18.1.temp), !dbg !27
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr @int_8), !dbg !28
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.63b9759f1df1768a(ptr @int_8), !dbg !27
// CHECK:STDOUT:   ret void, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN1C1FENS_1EE.carbon_thunk._(ptr noundef %0) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !41
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !41
// CHECK:STDOUT:   %2 = load i16, ptr %1, align 2, !tbaa !43
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE(i16 noundef signext %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.63b9759f1df1768a(ptr %self) #0 !dbg !31 {
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN1C1FENS_1EE(i16 noundef signext) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_ZN1C1FENS_1EE.carbon_thunk._, { 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.63b9759f1df1768a, { 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!44, !45, !46, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!49}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "pass_as_arg.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !{null}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Pass", linkageName: "_CPass.Main", scope: null, file: !4, line: 6, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !13 = !DILocation(line: 7, column: 11, scope: !12)
// CHECK:STDOUT: !14 = !DILocation(line: 7, column: 3, scope: !12)
// CHECK:STDOUT: !15 = !{null, !6}
// CHECK:STDOUT: !16 = !DISubroutineType(types: !15)
// CHECK:STDOUT: !17 = !DILocation(line: 8, column: 11, scope: !12)
// CHECK:STDOUT: !18 = !DILocation(line: 8, column: 3, scope: !12)
// CHECK:STDOUT: !19 = !DILocation(line: 9, column: 11, scope: !12)
// CHECK:STDOUT: !20 = !DILocation(line: 9, column: 3, scope: !12)
// CHECK:STDOUT: !21 = !DILocation(line: 10, column: 11, scope: !12)
// CHECK:STDOUT: !22 = !DILocation(line: 10, column: 3, scope: !12)
// CHECK:STDOUT: !23 = !DILocation(line: 11, column: 11, scope: !12)
// CHECK:STDOUT: !24 = !DILocation(line: 11, column: 3, scope: !12)
// CHECK:STDOUT: !25 = !DILocation(line: 12, column: 11, scope: !12)
// CHECK:STDOUT: !26 = !DILocation(line: 12, column: 3, scope: !12)
// CHECK:STDOUT: !27 = !DILocation(line: 13, column: 11, scope: !12)
// CHECK:STDOUT: !28 = !DILocation(line: 13, column: 3, scope: !12)
// CHECK:STDOUT: !29 = !DILocation(line: 6, column: 1, scope: !12)
// CHECK:STDOUT: !30 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.63b9759f1df1768a", scope: null, file: !30, line: 26, type: !16, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !37)
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !31, type: !6)
// CHECK:STDOUT: !36 = !DILocation(line: 26, column: 3, scope: !31)
// CHECK:STDOUT: !37 = !{!32}
// CHECK:STDOUT: !38 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !39 = !{!"omnipotent char", !38, i64 0}
// CHECK:STDOUT: !40 = !{!"any pointer", !39, i64 0}
// CHECK:STDOUT: !41 = !{!40, !40, i64 0}
// CHECK:STDOUT: !42 = !{!"_ZTSN1C1EE", !39, i64 0}
// CHECK:STDOUT: !43 = !{!42, !42, i64 0}
// CHECK:STDOUT: !44 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !45 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !46 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !47 = !{!"int", !39, i64 0}
// CHECK:STDOUT: !48 = !{!"__libc_errno", !47, i64 0}
// CHECK:STDOUT: !49 = !{!48, !47, i64 0}
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
// CHECK:STDOUT: define void @_CPassEnum.Main() #0 !dbg !52 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CTakeI16.Main(i16 7), !dbg !53
// CHECK:STDOUT:   ret void, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertEnumToI16.Main(i16 %e) #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CTakeI16.Main(i16 %e), !dbg !59
// CHECK:STDOUT:   ret void, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertI16ToEnum.Main(i16 %n) #0 !dbg !62 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17_13.3.temp = alloca i16, align 2, !dbg !65
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc17_13.3.temp), !dbg !65
// CHECK:STDOUT:   store i16 %n, ptr %.loc17_13.3.temp, align 2, !dbg !65
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr %.loc17_13.3.temp), !dbg !66
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.63b9759f1df1768a(ptr %.loc17_13.3.temp), !dbg !65
// CHECK:STDOUT:   ret void, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN1C1FENS_1EE.carbon_thunk._(ptr noundef %0) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !41
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !41
// CHECK:STDOUT:   %2 = load i16, ptr %1, align 2, !tbaa !43
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE(i16 noundef signext %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.63b9759f1df1768a(ptr %self) #0 !dbg !69 {
// CHECK:STDOUT:   ret void, !dbg !74
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN1C1FENS_1EE(i16 noundef signext) #3
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!51}
// CHECK:STDOUT: !llvm.module.flags = !{!44, !45, !46, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!49}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !{null}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !15 = !{null, !6}
// CHECK:STDOUT: !16 = !DISubroutineType(types: !15)
// CHECK:STDOUT: !30 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !38 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !39 = !{!"omnipotent char", !38, i64 0}
// CHECK:STDOUT: !40 = !{!"any pointer", !39, i64 0}
// CHECK:STDOUT: !41 = !{!40, !40, i64 0}
// CHECK:STDOUT: !42 = !{!"_ZTSN1C1EE", !39, i64 0}
// CHECK:STDOUT: !43 = !{!42, !42, i64 0}
// CHECK:STDOUT: !44 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !45 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !46 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !47 = !{!"int", !39, i64 0}
// CHECK:STDOUT: !48 = !{!"__libc_errno", !47, i64 0}
// CHECK:STDOUT: !49 = !{!48, !47, i64 0}
// CHECK:STDOUT: !50 = !DIFile(filename: "convert.carbon", directory: "")
// CHECK:STDOUT: !51 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !50, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !52 = distinct !DISubprogram(name: "PassEnum", linkageName: "_CPassEnum.Main", scope: null, file: !50, line: 8, type: !8, spFlags: DISPFlagDefinition, unit: !51)
// CHECK:STDOUT: !53 = !DILocation(line: 9, column: 3, scope: !52)
// CHECK:STDOUT: !54 = !DILocation(line: 8, column: 1, scope: !52)
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "ConvertEnumToI16", linkageName: "_CConvertEnumToI16.Main", scope: null, file: !50, line: 12, type: !16, spFlags: DISPFlagDefinition, unit: !51, retainedNodes: !61)
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !55, type: !6)
// CHECK:STDOUT: !59 = !DILocation(line: 13, column: 3, scope: !55)
// CHECK:STDOUT: !60 = !DILocation(line: 12, column: 1, scope: !55)
// CHECK:STDOUT: !61 = !{!56}
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "ConvertI16ToEnum", linkageName: "_CConvertI16ToEnum.Main", scope: null, file: !50, line: 16, type: !16, spFlags: DISPFlagDefinition, unit: !51, retainedNodes: !68)
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !62, type: !6)
// CHECK:STDOUT: !65 = !DILocation(line: 17, column: 11, scope: !62)
// CHECK:STDOUT: !66 = !DILocation(line: 17, column: 3, scope: !62)
// CHECK:STDOUT: !67 = !DILocation(line: 16, column: 1, scope: !62)
// CHECK:STDOUT: !68 = !{!63}
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.63b9759f1df1768a", scope: null, file: !30, line: 26, type: !16, spFlags: DISPFlagDefinition, unit: !51, retainedNodes: !75)
// CHECK:STDOUT: !70 = !DILocalVariable(arg: 1, scope: !69, type: !6)
// CHECK:STDOUT: !74 = !DILocation(line: 26, column: 3, scope: !69)
// CHECK:STDOUT: !75 = !{!70}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'use_bitmask.carbon'
// CHECK:STDOUT: source_filename = "use_bitmask.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCompileTime.Main() #0 !dbg !83 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 0), !dbg !84
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 5), !dbg !85
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 5), !dbg !86
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 -2), !dbg !87
// CHECK:STDOUT:   ret void, !dbg !88
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z4Take4Bits(i32 noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CRuntime.Main(i32 %a, i32 %b) #0 !dbg !89 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.call = and i32 %a, %b, !dbg !94
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 %BitAndWith.WithSelf.Op.call), !dbg !96
// CHECK:STDOUT:   %BitOrWith.WithSelf.Op.call = or i32 %a, %b, !dbg !97
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 %BitOrWith.WithSelf.Op.call), !dbg !99
// CHECK:STDOUT:   %BitXorWith.WithSelf.Op.call = xor i32 %a, %b, !dbg !100
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 %BitXorWith.WithSelf.Op.call), !dbg !102
// CHECK:STDOUT:   %BitComplement.WithSelf.Op.call = xor i32 -1, %a, !dbg !104
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 %BitComplement.WithSelf.Op.call), !dbg !105
// CHECK:STDOUT:   ret void, !dbg !106
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_Z4Take4Bits, { 0, 1, 2, 3, 7, 6, 5, 4 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!77}
// CHECK:STDOUT: !llvm.module.flags = !{!44, !45, !46, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!49}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !{null}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !38 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !39 = !{!"omnipotent char", !38, i64 0}
// CHECK:STDOUT: !44 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !45 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !46 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !47 = !{!"int", !39, i64 0}
// CHECK:STDOUT: !48 = !{!"__libc_errno", !47, i64 0}
// CHECK:STDOUT: !49 = !{!48, !47, i64 0}
// CHECK:STDOUT: !76 = !DIFile(filename: "use_bitmask.carbon", directory: "")
// CHECK:STDOUT: !77 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !76, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !78 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !81 = !{null, !78, !78}
// CHECK:STDOUT: !82 = !DISubroutineType(types: !81)
// CHECK:STDOUT: !83 = distinct !DISubprogram(name: "CompileTime", linkageName: "_CCompileTime.Main", scope: null, file: !76, line: 6, type: !8, spFlags: DISPFlagDefinition, unit: !77)
// CHECK:STDOUT: !84 = !DILocation(line: 7, column: 3, scope: !83)
// CHECK:STDOUT: !85 = !DILocation(line: 8, column: 3, scope: !83)
// CHECK:STDOUT: !86 = !DILocation(line: 9, column: 3, scope: !83)
// CHECK:STDOUT: !87 = !DILocation(line: 10, column: 3, scope: !83)
// CHECK:STDOUT: !88 = !DILocation(line: 6, column: 1, scope: !83)
// CHECK:STDOUT: !89 = distinct !DISubprogram(name: "Runtime", linkageName: "_CRuntime.Main", scope: null, file: !76, line: 13, type: !82, spFlags: DISPFlagDefinition, unit: !77, retainedNodes: !107)
// CHECK:STDOUT: !90 = !DILocalVariable(arg: 1, scope: !89, type: !78)
// CHECK:STDOUT: !91 = !DILocalVariable(arg: 2, scope: !89, type: !78)
// CHECK:STDOUT: !94 = !DILocation(line: 14, column: 12, scope: !89)
// CHECK:STDOUT: !96 = !DILocation(line: 14, column: 3, scope: !89)
// CHECK:STDOUT: !97 = !DILocation(line: 15, column: 12, scope: !89)
// CHECK:STDOUT: !99 = !DILocation(line: 15, column: 3, scope: !89)
// CHECK:STDOUT: !100 = !DILocation(line: 16, column: 12, scope: !89)
// CHECK:STDOUT: !102 = !DILocation(line: 16, column: 3, scope: !89)
// CHECK:STDOUT: !104 = !DILocation(line: 20, column: 12, scope: !89)
// CHECK:STDOUT: !105 = !DILocation(line: 20, column: 3, scope: !89)
// CHECK:STDOUT: !106 = !DILocation(line: 13, column: 1, scope: !89)
// CHECK:STDOUT: !107 = !{!90, !91}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'compare.carbon'
// CHECK:STDOUT: source_filename = "compare.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCompare.Main(i16 %a, i16 %b) #0 !dbg !112 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %OrderedWith.WithSelf.Equal.call = icmp eq i16 %a, %b, !dbg !119
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.Equal.call, !dbg !121
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCallCompareGeneric.Main(i16 %a, i16 %b) #0 !dbg !123 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %CompareGeneric.call = call i1 @_CCompareGeneric.Main.7602ebbc958eec83(i16 %a, i16 %b), !dbg !132
// CHECK:STDOUT:   ret i1 %CompareGeneric.call, !dbg !133
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CCompareGeneric.Main.7602ebbc958eec83(i16 %x, i16 %y) #0 !dbg !135 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.call = icmp eq i16 %x, %y, !dbg !142
// CHECK:STDOUT:   ret i1 %EqWith.WithSelf.Equal.call, !dbg !144
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!109}
// CHECK:STDOUT: !llvm.module.flags = !{!44, !45, !46, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!49}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !38 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !39 = !{!"omnipotent char", !38, i64 0}
// CHECK:STDOUT: !44 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !45 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !46 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !47 = !{!"int", !39, i64 0}
// CHECK:STDOUT: !48 = !{!"__libc_errno", !47, i64 0}
// CHECK:STDOUT: !49 = !{!48, !47, i64 0}
// CHECK:STDOUT: !108 = !DIFile(filename: "compare.carbon", directory: "")
// CHECK:STDOUT: !109 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !108, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !110 = !{!9, !6, !6}
// CHECK:STDOUT: !111 = !DISubroutineType(types: !110)
// CHECK:STDOUT: !112 = distinct !DISubprogram(name: "Compare", linkageName: "_CCompare.Main", scope: null, file: !108, line: 6, type: !111, spFlags: DISPFlagDefinition, unit: !109, retainedNodes: !122)
// CHECK:STDOUT: !113 = !DILocalVariable(arg: 1, scope: !112, type: !6)
// CHECK:STDOUT: !114 = !DILocalVariable(arg: 2, scope: !112, type: !6)
// CHECK:STDOUT: !119 = !DILocation(line: 7, column: 10, scope: !112)
// CHECK:STDOUT: !121 = !DILocation(line: 7, column: 3, scope: !112)
// CHECK:STDOUT: !122 = !{!113, !114}
// CHECK:STDOUT: !123 = distinct !DISubprogram(name: "CallCompareGeneric", linkageName: "_CCallCompareGeneric.Main", scope: null, file: !108, line: 14, type: !111, spFlags: DISPFlagDefinition, unit: !109, retainedNodes: !134)
// CHECK:STDOUT: !124 = !DILocalVariable(arg: 1, scope: !123, type: !6)
// CHECK:STDOUT: !125 = !DILocalVariable(arg: 2, scope: !123, type: !6)
// CHECK:STDOUT: !132 = !DILocation(line: 15, column: 10, scope: !123)
// CHECK:STDOUT: !133 = !DILocation(line: 15, column: 3, scope: !123)
// CHECK:STDOUT: !134 = !{!124, !125}
// CHECK:STDOUT: !135 = distinct !DISubprogram(name: "CompareGeneric", linkageName: "_CCompareGeneric.Main.7602ebbc958eec83", scope: null, file: !108, line: 10, type: !111, spFlags: DISPFlagDefinition, unit: !109, retainedNodes: !145)
// CHECK:STDOUT: !136 = !DILocalVariable(arg: 1, scope: !135, type: !6)
// CHECK:STDOUT: !137 = !DILocalVariable(arg: 2, scope: !135, type: !6)
// CHECK:STDOUT: !142 = !DILocation(line: 11, column: 10, scope: !135)
// CHECK:STDOUT: !144 = !DILocation(line: 11, column: 3, scope: !135)
// CHECK:STDOUT: !145 = !{!136, !137}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'compare_class.carbon'
// CHECK:STDOUT: source_filename = "compare_class.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCompare.Main(i16 %a, ptr %b) #0 !dbg !156 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_15.1.temp = alloca i16, align 2, !dbg !164
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_15.1.temp), !dbg !164
// CHECK:STDOUT:   call void @_ZNK13ConvertToEnumcvN1C1EEEv.carbon_thunk._(ptr %b, ptr %.loc18_15.1.temp), !dbg !164
// CHECK:STDOUT:   %.loc18_15.5 = load i16, ptr %.loc18_15.1.temp, align 2, !dbg !164
// CHECK:STDOUT:   %OrderedWith.WithSelf.Equal.call = icmp eq i16 %a, %.loc18_15.5, !dbg !163
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.63b9759f1df1768a(ptr %.loc18_15.1.temp), !dbg !164
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.Equal.call, !dbg !165
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZNK13ConvertToEnumcvN1C1EEEv.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !296
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !41
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !41
// CHECK:STDOUT:   %1 = load ptr, ptr %this.addr, align 8, !tbaa !296, !nonnull !298
// CHECK:STDOUT:   %call = call noundef signext i16 @_ZNK13ConvertToEnumcvN1C1EEEv(ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   store i16 %call, ptr %0, align 2, !tbaa !43
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCompare2.Main(i32 %a, ptr %b) #0 !dbg !167 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %__carbon_thunk.call = call i32 @_ZNK13ConvertToEnumcv2E2Ev.carbon_thunk._(ptr %b), !dbg !175
// CHECK:STDOUT:   %OrderedWith.WithSelf.Equal.call = icmp eq i32 %a, %__carbon_thunk.call, !dbg !174
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.Equal.call, !dbg !176
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal noundef i32 @_ZNK13ConvertToEnumcv2E2Ev.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !296
// CHECK:STDOUT:   %0 = load ptr, ptr %this.addr, align 8, !tbaa !296, !nonnull !298
// CHECK:STDOUT:   %call = call noundef i32 @_ZNK13ConvertToEnumcv2E2Ev(ptr noundef nonnull align 1 dereferenceable(1) %0)
// CHECK:STDOUT:   ret i32 %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCompare3.Main(i16 %a, ptr %b) #0 !dbg !178 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc26_15.1.temp = alloca i16, align 2, !dbg !186
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc26_15.1.temp), !dbg !186
// CHECK:STDOUT:   call void @_ZNK14ConvertToEnum2cvN1C1EEEv.carbon_thunk._(ptr %b, ptr %.loc26_15.1.temp), !dbg !186
// CHECK:STDOUT:   %.loc26_15.5 = load i16, ptr %.loc26_15.1.temp, align 2, !dbg !186
// CHECK:STDOUT:   %OrderedWith.WithSelf.Equal.call = icmp eq i16 %a, %.loc26_15.5, !dbg !185
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.63b9759f1df1768a(ptr %.loc26_15.1.temp), !dbg !186
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.Equal.call, !dbg !187
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZNK14ConvertToEnum2cvN1C1EEEv.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !300
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !41
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !41
// CHECK:STDOUT:   %1 = load ptr, ptr %this.addr, align 8, !tbaa !300, !nonnull !298
// CHECK:STDOUT:   %call = call noundef signext i16 @_ZNK14ConvertToEnum2cvN1C1EEEv(ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   store i16 %call, ptr %0, align 2, !tbaa !43
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCallCompareGeneric.Main(i16 %a, ptr %b) #0 !dbg !189 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %CompareGeneric.call = call i1 @_CCompareGeneric.Main.356c7bb457b5b49b(i16 %a, ptr %b), !dbg !198
// CHECK:STDOUT:   ret i1 %CompareGeneric.call, !dbg !199
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CEqual:thunk:EqWith.83b9626e83cf3dcc.Core:OrderedWith.Core"(i16 %self, ptr %other) #2 !dbg !201 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.4.temp = alloca i16, align 2, !dbg !204
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.4.temp), !dbg !204
// CHECK:STDOUT:   call void @_ZNK13ConvertToEnumcvN1C1EEEv.carbon_thunk._(ptr %other, ptr %.4.temp), !dbg !204
// CHECK:STDOUT:   %.8 = load i16, ptr %.4.temp, align 2, !dbg !204
// CHECK:STDOUT:   %OrderedWith.WithSelf.Equal.call = icmp eq i16 %self, %.8, !dbg !204
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.63b9759f1df1768a(ptr %.4.temp), !dbg !204
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.Equal.call, !dbg !204
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CNotEqual:thunk:EqWith.83b9626e83cf3dcc.Core:OrderedWith.Core"(i16 %self, ptr %other) #2 !dbg !206 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.4.temp = alloca i16, align 2, !dbg !209
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.4.temp), !dbg !209
// CHECK:STDOUT:   call void @_ZNK13ConvertToEnumcvN1C1EEEv.carbon_thunk._(ptr %other, ptr %.4.temp), !dbg !209
// CHECK:STDOUT:   %.8 = load i16, ptr %.4.temp, align 2, !dbg !209
// CHECK:STDOUT:   %OrderedWith.WithSelf.NotEqual.call = icmp ne i16 %self, %.8, !dbg !209
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.63b9759f1df1768a(ptr %.4.temp), !dbg !209
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.NotEqual.call, !dbg !209
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCallCompareGeneric2.Main(i32 %a, ptr %b) #0 !dbg !211 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %CompareGeneric.call = call i1 @_CCompareGeneric.Main.4512c7023c79a313(i32 %a, ptr %b), !dbg !220
// CHECK:STDOUT:   ret i1 %CompareGeneric.call, !dbg !221
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CEqual:thunk:EqWith.e8fe48de943d9652.Core:OrderedWith.Core"(i32 %self, ptr %other) #2 !dbg !223 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %__carbon_thunk.call = call i32 @_ZNK13ConvertToEnumcv2E2Ev.carbon_thunk._(ptr %other), !dbg !226
// CHECK:STDOUT:   %OrderedWith.WithSelf.Equal.call = icmp eq i32 %self, %__carbon_thunk.call, !dbg !226
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.Equal.call, !dbg !226
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CNotEqual:thunk:EqWith.e8fe48de943d9652.Core:OrderedWith.Core"(i32 %self, ptr %other) #2 !dbg !228 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %__carbon_thunk.call = call i32 @_ZNK13ConvertToEnumcv2E2Ev.carbon_thunk._(ptr %other), !dbg !231
// CHECK:STDOUT:   %OrderedWith.WithSelf.NotEqual.call = icmp ne i32 %self, %__carbon_thunk.call, !dbg !231
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.NotEqual.call, !dbg !231
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCallCompareGeneric3.Main(i16 %a, ptr %b) #0 !dbg !233 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %CompareGeneric.call = call i1 @_CCompareGeneric.Main.070a42571d1942e2(i16 %a, ptr %b), !dbg !242
// CHECK:STDOUT:   ret i1 %CompareGeneric.call, !dbg !243
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CEqual:thunk:EqWith.a294139a6615e163.Core:OrderedWith.Core"(i16 %self, ptr %other) #2 !dbg !245 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.4.temp = alloca i16, align 2, !dbg !248
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.4.temp), !dbg !248
// CHECK:STDOUT:   call void @_ZNK14ConvertToEnum2cvN1C1EEEv.carbon_thunk._(ptr %other, ptr %.4.temp), !dbg !248
// CHECK:STDOUT:   %.8 = load i16, ptr %.4.temp, align 2, !dbg !248
// CHECK:STDOUT:   %OrderedWith.WithSelf.Equal.call = icmp eq i16 %self, %.8, !dbg !248
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.63b9759f1df1768a(ptr %.4.temp), !dbg !248
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.Equal.call, !dbg !248
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CNotEqual:thunk:EqWith.a294139a6615e163.Core:OrderedWith.Core"(i16 %self, ptr %other) #2 !dbg !250 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.4.temp = alloca i16, align 2, !dbg !253
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.4.temp), !dbg !253
// CHECK:STDOUT:   call void @_ZNK14ConvertToEnum2cvN1C1EEEv.carbon_thunk._(ptr %other, ptr %.4.temp), !dbg !253
// CHECK:STDOUT:   %.8 = load i16, ptr %.4.temp, align 2, !dbg !253
// CHECK:STDOUT:   %OrderedWith.WithSelf.NotEqual.call = icmp ne i16 %self, %.8, !dbg !253
// CHECK:STDOUT:   call void @_CSelfDestruct.Destroy.Core.63b9759f1df1768a(ptr %.4.temp), !dbg !253
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.NotEqual.call, !dbg !253
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Destroy.Core.63b9759f1df1768a(ptr %self) #0 !dbg !255 {
// CHECK:STDOUT:   ret void, !dbg !260
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CCompareGeneric.Main.356c7bb457b5b49b(i16 %x, ptr %y) #0 !dbg !262 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.call = call i1 @"_CEqual:thunk:EqWith.83b9626e83cf3dcc.Core:OrderedWith.Core"(i16 %x, ptr %y), !dbg !269
// CHECK:STDOUT:   ret i1 %EqWith.WithSelf.Equal.call, !dbg !271
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CCompareGeneric.Main.4512c7023c79a313(i32 %x, ptr %y) #0 !dbg !273 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.call = call i1 @"_CEqual:thunk:EqWith.e8fe48de943d9652.Core:OrderedWith.Core"(i32 %x, ptr %y), !dbg !280
// CHECK:STDOUT:   ret i1 %EqWith.WithSelf.Equal.call, !dbg !282
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CCompareGeneric.Main.070a42571d1942e2(i16 %x, ptr %y) #0 !dbg !284 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.call = call i1 @"_CEqual:thunk:EqWith.a294139a6615e163.Core:OrderedWith.Core"(i16 %x, ptr %y), !dbg !291
// CHECK:STDOUT:   ret i1 %EqWith.WithSelf.Equal.call, !dbg !293
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
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Destroy.Core.63b9759f1df1768a, { 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #4 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!147}
// CHECK:STDOUT: !llvm.module.flags = !{!44, !45, !46, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!49}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !15 = !{null, !6}
// CHECK:STDOUT: !16 = !DISubroutineType(types: !15)
// CHECK:STDOUT: !30 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !38 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !39 = !{!"omnipotent char", !38, i64 0}
// CHECK:STDOUT: !40 = !{!"any pointer", !39, i64 0}
// CHECK:STDOUT: !41 = !{!40, !40, i64 0}
// CHECK:STDOUT: !42 = !{!"_ZTSN1C1EE", !39, i64 0}
// CHECK:STDOUT: !43 = !{!42, !42, i64 0}
// CHECK:STDOUT: !44 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !45 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !46 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !47 = !{!"int", !39, i64 0}
// CHECK:STDOUT: !48 = !{!"__libc_errno", !47, i64 0}
// CHECK:STDOUT: !49 = !{!48, !47, i64 0}
// CHECK:STDOUT: !78 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !146 = !DIFile(filename: "compare_class.carbon", directory: "")
// CHECK:STDOUT: !147 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !146, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !148 = !{!9, !6, !9}
// CHECK:STDOUT: !149 = !DISubroutineType(types: !148)
// CHECK:STDOUT: !152 = !{!9, !78, !9}
// CHECK:STDOUT: !153 = !DISubroutineType(types: !152)
// CHECK:STDOUT: !156 = distinct !DISubprogram(name: "Compare", linkageName: "_CCompare.Main", scope: null, file: !146, line: 17, type: !149, spFlags: DISPFlagDefinition, unit: !147, retainedNodes: !166)
// CHECK:STDOUT: !157 = !DILocalVariable(arg: 1, scope: !156, type: !6)
// CHECK:STDOUT: !158 = !DILocalVariable(arg: 2, scope: !156, type: !9)
// CHECK:STDOUT: !163 = !DILocation(line: 18, column: 10, scope: !156)
// CHECK:STDOUT: !164 = !DILocation(line: 18, column: 15, scope: !156)
// CHECK:STDOUT: !165 = !DILocation(line: 18, column: 3, scope: !156)
// CHECK:STDOUT: !166 = !{!157, !158}
// CHECK:STDOUT: !167 = distinct !DISubprogram(name: "Compare2", linkageName: "_CCompare2.Main", scope: null, file: !146, line: 21, type: !153, spFlags: DISPFlagDefinition, unit: !147, retainedNodes: !177)
// CHECK:STDOUT: !168 = !DILocalVariable(arg: 1, scope: !167, type: !78)
// CHECK:STDOUT: !169 = !DILocalVariable(arg: 2, scope: !167, type: !9)
// CHECK:STDOUT: !174 = !DILocation(line: 22, column: 10, scope: !167)
// CHECK:STDOUT: !175 = !DILocation(line: 22, column: 15, scope: !167)
// CHECK:STDOUT: !176 = !DILocation(line: 22, column: 3, scope: !167)
// CHECK:STDOUT: !177 = !{!168, !169}
// CHECK:STDOUT: !178 = distinct !DISubprogram(name: "Compare3", linkageName: "_CCompare3.Main", scope: null, file: !146, line: 25, type: !149, spFlags: DISPFlagDefinition, unit: !147, retainedNodes: !188)
// CHECK:STDOUT: !179 = !DILocalVariable(arg: 1, scope: !178, type: !6)
// CHECK:STDOUT: !180 = !DILocalVariable(arg: 2, scope: !178, type: !9)
// CHECK:STDOUT: !185 = !DILocation(line: 26, column: 10, scope: !178)
// CHECK:STDOUT: !186 = !DILocation(line: 26, column: 15, scope: !178)
// CHECK:STDOUT: !187 = !DILocation(line: 26, column: 3, scope: !178)
// CHECK:STDOUT: !188 = !{!179, !180}
// CHECK:STDOUT: !189 = distinct !DISubprogram(name: "CallCompareGeneric", linkageName: "_CCallCompareGeneric.Main", scope: null, file: !146, line: 35, type: !149, spFlags: DISPFlagDefinition, unit: !147, retainedNodes: !200)
// CHECK:STDOUT: !190 = !DILocalVariable(arg: 1, scope: !189, type: !6)
// CHECK:STDOUT: !191 = !DILocalVariable(arg: 2, scope: !189, type: !9)
// CHECK:STDOUT: !198 = !DILocation(line: 36, column: 10, scope: !189)
// CHECK:STDOUT: !199 = !DILocation(line: 36, column: 3, scope: !189)
// CHECK:STDOUT: !200 = !{!190, !191}
// CHECK:STDOUT: !201 = distinct !DISubprogram(name: "Equal", linkageName: "_CEqual:thunk:EqWith.83b9626e83cf3dcc.Core:OrderedWith.Core", scope: null, file: !146, type: !149, spFlags: DISPFlagDefinition, unit: !147, retainedNodes: !205)
// CHECK:STDOUT: !202 = !DILocalVariable(arg: 1, scope: !201, type: !6)
// CHECK:STDOUT: !203 = !DILocalVariable(arg: 2, scope: !201, type: !9)
// CHECK:STDOUT: !204 = !DILocation(line: 0, scope: !201)
// CHECK:STDOUT: !205 = !{!202, !203}
// CHECK:STDOUT: !206 = distinct !DISubprogram(name: "NotEqual", linkageName: "_CNotEqual:thunk:EqWith.83b9626e83cf3dcc.Core:OrderedWith.Core", scope: null, file: !146, type: !149, spFlags: DISPFlagDefinition, unit: !147, retainedNodes: !210)
// CHECK:STDOUT: !207 = !DILocalVariable(arg: 1, scope: !206, type: !6)
// CHECK:STDOUT: !208 = !DILocalVariable(arg: 2, scope: !206, type: !9)
// CHECK:STDOUT: !209 = !DILocation(line: 0, scope: !206)
// CHECK:STDOUT: !210 = !{!207, !208}
// CHECK:STDOUT: !211 = distinct !DISubprogram(name: "CallCompareGeneric2", linkageName: "_CCallCompareGeneric2.Main", scope: null, file: !146, line: 39, type: !153, spFlags: DISPFlagDefinition, unit: !147, retainedNodes: !222)
// CHECK:STDOUT: !212 = !DILocalVariable(arg: 1, scope: !211, type: !78)
// CHECK:STDOUT: !213 = !DILocalVariable(arg: 2, scope: !211, type: !9)
// CHECK:STDOUT: !220 = !DILocation(line: 40, column: 10, scope: !211)
// CHECK:STDOUT: !221 = !DILocation(line: 40, column: 3, scope: !211)
// CHECK:STDOUT: !222 = !{!212, !213}
// CHECK:STDOUT: !223 = distinct !DISubprogram(name: "Equal", linkageName: "_CEqual:thunk:EqWith.e8fe48de943d9652.Core:OrderedWith.Core", scope: null, file: !146, type: !153, spFlags: DISPFlagDefinition, unit: !147, retainedNodes: !227)
// CHECK:STDOUT: !224 = !DILocalVariable(arg: 1, scope: !223, type: !78)
// CHECK:STDOUT: !225 = !DILocalVariable(arg: 2, scope: !223, type: !9)
// CHECK:STDOUT: !226 = !DILocation(line: 0, scope: !223)
// CHECK:STDOUT: !227 = !{!224, !225}
// CHECK:STDOUT: !228 = distinct !DISubprogram(name: "NotEqual", linkageName: "_CNotEqual:thunk:EqWith.e8fe48de943d9652.Core:OrderedWith.Core", scope: null, file: !146, type: !153, spFlags: DISPFlagDefinition, unit: !147, retainedNodes: !232)
// CHECK:STDOUT: !229 = !DILocalVariable(arg: 1, scope: !228, type: !78)
// CHECK:STDOUT: !230 = !DILocalVariable(arg: 2, scope: !228, type: !9)
// CHECK:STDOUT: !231 = !DILocation(line: 0, scope: !228)
// CHECK:STDOUT: !232 = !{!229, !230}
// CHECK:STDOUT: !233 = distinct !DISubprogram(name: "CallCompareGeneric3", linkageName: "_CCallCompareGeneric3.Main", scope: null, file: !146, line: 43, type: !149, spFlags: DISPFlagDefinition, unit: !147, retainedNodes: !244)
// CHECK:STDOUT: !234 = !DILocalVariable(arg: 1, scope: !233, type: !6)
// CHECK:STDOUT: !235 = !DILocalVariable(arg: 2, scope: !233, type: !9)
// CHECK:STDOUT: !242 = !DILocation(line: 44, column: 10, scope: !233)
// CHECK:STDOUT: !243 = !DILocation(line: 44, column: 3, scope: !233)
// CHECK:STDOUT: !244 = !{!234, !235}
// CHECK:STDOUT: !245 = distinct !DISubprogram(name: "Equal", linkageName: "_CEqual:thunk:EqWith.a294139a6615e163.Core:OrderedWith.Core", scope: null, file: !146, type: !149, spFlags: DISPFlagDefinition, unit: !147, retainedNodes: !249)
// CHECK:STDOUT: !246 = !DILocalVariable(arg: 1, scope: !245, type: !6)
// CHECK:STDOUT: !247 = !DILocalVariable(arg: 2, scope: !245, type: !9)
// CHECK:STDOUT: !248 = !DILocation(line: 0, scope: !245)
// CHECK:STDOUT: !249 = !{!246, !247}
// CHECK:STDOUT: !250 = distinct !DISubprogram(name: "NotEqual", linkageName: "_CNotEqual:thunk:EqWith.a294139a6615e163.Core:OrderedWith.Core", scope: null, file: !146, type: !149, spFlags: DISPFlagDefinition, unit: !147, retainedNodes: !254)
// CHECK:STDOUT: !251 = !DILocalVariable(arg: 1, scope: !250, type: !6)
// CHECK:STDOUT: !252 = !DILocalVariable(arg: 2, scope: !250, type: !9)
// CHECK:STDOUT: !253 = !DILocation(line: 0, scope: !250)
// CHECK:STDOUT: !254 = !{!251, !252}
// CHECK:STDOUT: !255 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Destroy.Core.63b9759f1df1768a", scope: null, file: !30, line: 26, type: !16, spFlags: DISPFlagDefinition, unit: !147, retainedNodes: !261)
// CHECK:STDOUT: !256 = !DILocalVariable(arg: 1, scope: !255, type: !6)
// CHECK:STDOUT: !260 = !DILocation(line: 26, column: 3, scope: !255)
// CHECK:STDOUT: !261 = !{!256}
// CHECK:STDOUT: !262 = distinct !DISubprogram(name: "CompareGeneric", linkageName: "_CCompareGeneric.Main.356c7bb457b5b49b", scope: null, file: !146, line: 29, type: !149, spFlags: DISPFlagDefinition, unit: !147, retainedNodes: !272)
// CHECK:STDOUT: !263 = !DILocalVariable(arg: 1, scope: !262, type: !6)
// CHECK:STDOUT: !264 = !DILocalVariable(arg: 2, scope: !262, type: !9)
// CHECK:STDOUT: !269 = !DILocation(line: 30, column: 10, scope: !262)
// CHECK:STDOUT: !271 = !DILocation(line: 30, column: 3, scope: !262)
// CHECK:STDOUT: !272 = !{!263, !264}
// CHECK:STDOUT: !273 = distinct !DISubprogram(name: "CompareGeneric", linkageName: "_CCompareGeneric.Main.4512c7023c79a313", scope: null, file: !146, line: 29, type: !153, spFlags: DISPFlagDefinition, unit: !147, retainedNodes: !283)
// CHECK:STDOUT: !274 = !DILocalVariable(arg: 1, scope: !273, type: !78)
// CHECK:STDOUT: !275 = !DILocalVariable(arg: 2, scope: !273, type: !9)
// CHECK:STDOUT: !280 = !DILocation(line: 30, column: 10, scope: !273)
// CHECK:STDOUT: !282 = !DILocation(line: 30, column: 3, scope: !273)
// CHECK:STDOUT: !283 = !{!274, !275}
// CHECK:STDOUT: !284 = distinct !DISubprogram(name: "CompareGeneric", linkageName: "_CCompareGeneric.Main.070a42571d1942e2", scope: null, file: !146, line: 29, type: !149, spFlags: DISPFlagDefinition, unit: !147, retainedNodes: !294)
// CHECK:STDOUT: !285 = !DILocalVariable(arg: 1, scope: !284, type: !6)
// CHECK:STDOUT: !286 = !DILocalVariable(arg: 2, scope: !284, type: !9)
// CHECK:STDOUT: !291 = !DILocation(line: 30, column: 10, scope: !284)
// CHECK:STDOUT: !293 = !DILocation(line: 30, column: 3, scope: !284)
// CHECK:STDOUT: !294 = !{!285, !286}
// CHECK:STDOUT: !295 = !{!"p1 _ZTS13ConvertToEnum", !40, i64 0}
// CHECK:STDOUT: !296 = !{!295, !295, i64 0}
// CHECK:STDOUT: !298 = !{}
// CHECK:STDOUT: !299 = !{!"p1 _ZTS14ConvertToEnum2", !40, i64 0}
// CHECK:STDOUT: !300 = !{!299, !299, i64 0}
// CHECK:STDOUT:
