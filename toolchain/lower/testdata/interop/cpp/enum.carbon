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
// CHECK:STDOUT: define void @_CPass.Main() #0 !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_16.1.temp = alloca i16, align 2, !dbg !15
// CHECK:STDOUT:   %.loc8_16.1.temp = alloca i16, align 2, !dbg !17
// CHECK:STDOUT:   %.loc9_16.1.temp = alloca i16, align 2, !dbg !19
// CHECK:STDOUT:   %.loc10_16.1.temp = alloca i16, align 2, !dbg !21
// CHECK:STDOUT:   %.loc11_18.1.temp = alloca i16, align 2, !dbg !23
// CHECK:STDOUT:   %.loc12_18.1.temp = alloca i16, align 2, !dbg !25
// CHECK:STDOUT:   %.loc13_18.1.temp = alloca i16, align 2, !dbg !27
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_16.1.temp), !dbg !15
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr @int_0), !dbg !16
// CHECK:STDOUT:   call void @"_CSelfDestruct.4f81cc432cb50e07:core.Destroy.Core"(ptr @int_0), !dbg !15
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc8_16.1.temp), !dbg !17
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr @int_0), !dbg !18
// CHECK:STDOUT:   call void @"_CSelfDestruct.4f81cc432cb50e07:core.Destroy.Core"(ptr @int_0), !dbg !17
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc9_16.1.temp), !dbg !19
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr @int_7), !dbg !20
// CHECK:STDOUT:   call void @"_CSelfDestruct.4f81cc432cb50e07:core.Destroy.Core"(ptr @int_7), !dbg !19
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc10_16.1.temp), !dbg !21
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr @int_8), !dbg !22
// CHECK:STDOUT:   call void @"_CSelfDestruct.4f81cc432cb50e07:core.Destroy.Core"(ptr @int_8), !dbg !21
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc11_18.1.temp), !dbg !23
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr @int_0), !dbg !24
// CHECK:STDOUT:   call void @"_CSelfDestruct.4f81cc432cb50e07:core.Destroy.Core"(ptr @int_0), !dbg !23
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc12_18.1.temp), !dbg !25
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr @int_7), !dbg !26
// CHECK:STDOUT:   call void @"_CSelfDestruct.4f81cc432cb50e07:core.Destroy.Core"(ptr @int_7), !dbg !25
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc13_18.1.temp), !dbg !27
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr @int_8), !dbg !28
// CHECK:STDOUT:   call void @"_CSelfDestruct.4f81cc432cb50e07:core.Destroy.Core"(ptr @int_8), !dbg !27
// CHECK:STDOUT:   ret void, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN1C1FENS_1EE.carbon_thunk._(ptr noundef %0) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !37
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !37
// CHECK:STDOUT:   %2 = load i16, ptr %1, align 2, !tbaa !39
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE(i16 noundef signext %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.4f81cc432cb50e07:core.Destroy.Core"(ptr %self) #0 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN1C1FENS_1EE(i16 noundef signext) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_ZN1C1FENS_1EE.carbon_thunk._, { 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_CSelfDestruct.4f81cc432cb50e07:core.Destroy.Core", { 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!40, !41, !42, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!45}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "pass_as_arg.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !{null}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !12 = !{null, !6}
// CHECK:STDOUT: !13 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "Pass", linkageName: "_CPass.Main", scope: null, file: !4, line: 6, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !15 = !DILocation(line: 7, column: 11, scope: !14)
// CHECK:STDOUT: !16 = !DILocation(line: 7, column: 3, scope: !14)
// CHECK:STDOUT: !17 = !DILocation(line: 8, column: 11, scope: !14)
// CHECK:STDOUT: !18 = !DILocation(line: 8, column: 3, scope: !14)
// CHECK:STDOUT: !19 = !DILocation(line: 9, column: 11, scope: !14)
// CHECK:STDOUT: !20 = !DILocation(line: 9, column: 3, scope: !14)
// CHECK:STDOUT: !21 = !DILocation(line: 10, column: 11, scope: !14)
// CHECK:STDOUT: !22 = !DILocation(line: 10, column: 3, scope: !14)
// CHECK:STDOUT: !23 = !DILocation(line: 11, column: 11, scope: !14)
// CHECK:STDOUT: !24 = !DILocation(line: 11, column: 3, scope: !14)
// CHECK:STDOUT: !25 = !DILocation(line: 12, column: 11, scope: !14)
// CHECK:STDOUT: !26 = !DILocation(line: 12, column: 3, scope: !14)
// CHECK:STDOUT: !27 = !DILocation(line: 13, column: 11, scope: !14)
// CHECK:STDOUT: !28 = !DILocation(line: 13, column: 3, scope: !14)
// CHECK:STDOUT: !29 = !DILocation(line: 6, column: 1, scope: !14)
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.4f81cc432cb50e07:core.Destroy.Core", scope: null, file: !4, line: 7, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !33)
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 1, scope: !30, type: !6)
// CHECK:STDOUT: !32 = !DILocation(line: 7, column: 11, scope: !30)
// CHECK:STDOUT: !33 = !{!31}
// CHECK:STDOUT: !34 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !35 = !{!"omnipotent char", !34, i64 0}
// CHECK:STDOUT: !36 = !{!"any pointer", !35, i64 0}
// CHECK:STDOUT: !37 = !{!36, !36, i64 0}
// CHECK:STDOUT: !38 = !{!"_ZTSN1C1EE", !35, i64 0}
// CHECK:STDOUT: !39 = !{!38, !38, i64 0}
// CHECK:STDOUT: !40 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !41 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !42 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !43 = !{!"int", !35, i64 0}
// CHECK:STDOUT: !44 = !{!"__libc_errno", !43, i64 0}
// CHECK:STDOUT: !45 = !{!44, !43, i64 0}
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
// CHECK:STDOUT: define void @_CPassEnum.Main() #0 !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CTakeI16.Main(i16 7), !dbg !49
// CHECK:STDOUT:   ret void, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertEnumToI16.Main(i16 %e) #0 !dbg !51 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CTakeI16.Main(i16 %e), !dbg !55
// CHECK:STDOUT:   ret void, !dbg !56
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertI16ToEnum.Main(i16 %n) #0 !dbg !58 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17_13.3.temp = alloca i16, align 2, !dbg !61
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc17_13.3.temp), !dbg !61
// CHECK:STDOUT:   store i16 %n, ptr %.loc17_13.3.temp, align 2, !dbg !61
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr %.loc17_13.3.temp), !dbg !62
// CHECK:STDOUT:   call void @"_CSelfDestruct.4f81cc432cb50e07:core.Destroy.Core"(ptr %.loc17_13.3.temp), !dbg !61
// CHECK:STDOUT:   ret void, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN1C1FENS_1EE.carbon_thunk._(ptr noundef %0) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !37
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !37
// CHECK:STDOUT:   %2 = load i16, ptr %1, align 2, !tbaa !39
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE(i16 noundef signext %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.4f81cc432cb50e07:core.Destroy.Core"(ptr %self) #0 !dbg !65 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !67
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!47}
// CHECK:STDOUT: !llvm.module.flags = !{!40, !41, !42, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!45}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !{null}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !12 = !{null, !6}
// CHECK:STDOUT: !13 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !34 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !35 = !{!"omnipotent char", !34, i64 0}
// CHECK:STDOUT: !36 = !{!"any pointer", !35, i64 0}
// CHECK:STDOUT: !37 = !{!36, !36, i64 0}
// CHECK:STDOUT: !38 = !{!"_ZTSN1C1EE", !35, i64 0}
// CHECK:STDOUT: !39 = !{!38, !38, i64 0}
// CHECK:STDOUT: !40 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !41 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !42 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !43 = !{!"int", !35, i64 0}
// CHECK:STDOUT: !44 = !{!"__libc_errno", !43, i64 0}
// CHECK:STDOUT: !45 = !{!44, !43, i64 0}
// CHECK:STDOUT: !46 = !DIFile(filename: "convert.carbon", directory: "")
// CHECK:STDOUT: !47 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !46, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "PassEnum", linkageName: "_CPassEnum.Main", scope: null, file: !46, line: 8, type: !8, spFlags: DISPFlagDefinition, unit: !47)
// CHECK:STDOUT: !49 = !DILocation(line: 9, column: 3, scope: !48)
// CHECK:STDOUT: !50 = !DILocation(line: 8, column: 1, scope: !48)
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "ConvertEnumToI16", linkageName: "_CConvertEnumToI16.Main", scope: null, file: !46, line: 12, type: !13, spFlags: DISPFlagDefinition, unit: !47, retainedNodes: !57)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 1, scope: !51, type: !6)
// CHECK:STDOUT: !55 = !DILocation(line: 13, column: 3, scope: !51)
// CHECK:STDOUT: !56 = !DILocation(line: 12, column: 1, scope: !51)
// CHECK:STDOUT: !57 = !{!52}
// CHECK:STDOUT: !58 = distinct !DISubprogram(name: "ConvertI16ToEnum", linkageName: "_CConvertI16ToEnum.Main", scope: null, file: !46, line: 16, type: !13, spFlags: DISPFlagDefinition, unit: !47, retainedNodes: !64)
// CHECK:STDOUT: !59 = !DILocalVariable(arg: 1, scope: !58, type: !6)
// CHECK:STDOUT: !61 = !DILocation(line: 17, column: 11, scope: !58)
// CHECK:STDOUT: !62 = !DILocation(line: 17, column: 3, scope: !58)
// CHECK:STDOUT: !63 = !DILocation(line: 16, column: 1, scope: !58)
// CHECK:STDOUT: !64 = !{!59}
// CHECK:STDOUT: !65 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.4f81cc432cb50e07:core.Destroy.Core", scope: null, file: !46, line: 17, type: !13, spFlags: DISPFlagDefinition, unit: !47, retainedNodes: !68)
// CHECK:STDOUT: !66 = !DILocalVariable(arg: 1, scope: !65, type: !6)
// CHECK:STDOUT: !67 = !DILocation(line: 17, column: 11, scope: !65)
// CHECK:STDOUT: !68 = !{!66}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'use_bitmask.carbon'
// CHECK:STDOUT: source_filename = "use_bitmask.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCompileTime.Main() #0 !dbg !76 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 0), !dbg !77
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 5), !dbg !78
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 5), !dbg !79
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 -2), !dbg !80
// CHECK:STDOUT:   ret void, !dbg !81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z4Take4Bits(i32 noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CRuntime.Main(i32 %a, i32 %b) #0 !dbg !82 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.call = and i32 %a, %b, !dbg !87
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 %BitAndWith.WithSelf.Op.call), !dbg !89
// CHECK:STDOUT:   %BitOrWith.WithSelf.Op.call = or i32 %a, %b, !dbg !90
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 %BitOrWith.WithSelf.Op.call), !dbg !92
// CHECK:STDOUT:   %BitXorWith.WithSelf.Op.call = xor i32 %a, %b, !dbg !93
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 %BitXorWith.WithSelf.Op.call), !dbg !95
// CHECK:STDOUT:   %BitComplement.WithSelf.Op.call = xor i32 -1, %a, !dbg !97
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 %BitComplement.WithSelf.Op.call), !dbg !98
// CHECK:STDOUT:   ret void, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_Z4Take4Bits, { 0, 1, 2, 3, 7, 6, 5, 4 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!70}
// CHECK:STDOUT: !llvm.module.flags = !{!40, !41, !42, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!45}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !{null}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !34 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !35 = !{!"omnipotent char", !34, i64 0}
// CHECK:STDOUT: !40 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !41 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !42 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !43 = !{!"int", !35, i64 0}
// CHECK:STDOUT: !44 = !{!"__libc_errno", !43, i64 0}
// CHECK:STDOUT: !45 = !{!44, !43, i64 0}
// CHECK:STDOUT: !69 = !DIFile(filename: "use_bitmask.carbon", directory: "")
// CHECK:STDOUT: !70 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !69, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !71 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !74 = !{null, !71, !71}
// CHECK:STDOUT: !75 = !DISubroutineType(types: !74)
// CHECK:STDOUT: !76 = distinct !DISubprogram(name: "CompileTime", linkageName: "_CCompileTime.Main", scope: null, file: !69, line: 6, type: !8, spFlags: DISPFlagDefinition, unit: !70)
// CHECK:STDOUT: !77 = !DILocation(line: 7, column: 3, scope: !76)
// CHECK:STDOUT: !78 = !DILocation(line: 8, column: 3, scope: !76)
// CHECK:STDOUT: !79 = !DILocation(line: 9, column: 3, scope: !76)
// CHECK:STDOUT: !80 = !DILocation(line: 10, column: 3, scope: !76)
// CHECK:STDOUT: !81 = !DILocation(line: 6, column: 1, scope: !76)
// CHECK:STDOUT: !82 = distinct !DISubprogram(name: "Runtime", linkageName: "_CRuntime.Main", scope: null, file: !69, line: 13, type: !75, spFlags: DISPFlagDefinition, unit: !70, retainedNodes: !100)
// CHECK:STDOUT: !83 = !DILocalVariable(arg: 1, scope: !82, type: !71)
// CHECK:STDOUT: !84 = !DILocalVariable(arg: 2, scope: !82, type: !71)
// CHECK:STDOUT: !87 = !DILocation(line: 14, column: 12, scope: !82)
// CHECK:STDOUT: !89 = !DILocation(line: 14, column: 3, scope: !82)
// CHECK:STDOUT: !90 = !DILocation(line: 15, column: 12, scope: !82)
// CHECK:STDOUT: !92 = !DILocation(line: 15, column: 3, scope: !82)
// CHECK:STDOUT: !93 = !DILocation(line: 16, column: 12, scope: !82)
// CHECK:STDOUT: !95 = !DILocation(line: 16, column: 3, scope: !82)
// CHECK:STDOUT: !97 = !DILocation(line: 20, column: 12, scope: !82)
// CHECK:STDOUT: !98 = !DILocation(line: 20, column: 3, scope: !82)
// CHECK:STDOUT: !99 = !DILocation(line: 13, column: 1, scope: !82)
// CHECK:STDOUT: !100 = !{!83, !84}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'compare.carbon'
// CHECK:STDOUT: source_filename = "compare.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCompare.Main(i16 %a, i16 %b) #0 !dbg !105 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %OrderedWith.WithSelf.Equal.call = icmp eq i16 %a, %b, !dbg !112
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.Equal.call, !dbg !114
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCallCompareGeneric.Main(i16 %a, i16 %b) #0 !dbg !116 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %CompareGeneric.call = call i1 @_CCompareGeneric.Main.7602ebbc958eec83(i16 %a, i16 %b), !dbg !125
// CHECK:STDOUT:   ret i1 %CompareGeneric.call, !dbg !126
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CCompareGeneric.Main.7602ebbc958eec83(i16 %x, i16 %y) #0 !dbg !128 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.call = icmp eq i16 %x, %y, !dbg !135
// CHECK:STDOUT:   ret i1 %EqWith.WithSelf.Equal.call, !dbg !137
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!102}
// CHECK:STDOUT: !llvm.module.flags = !{!40, !41, !42, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!45}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !34 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !35 = !{!"omnipotent char", !34, i64 0}
// CHECK:STDOUT: !40 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !41 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !42 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !43 = !{!"int", !35, i64 0}
// CHECK:STDOUT: !44 = !{!"__libc_errno", !43, i64 0}
// CHECK:STDOUT: !45 = !{!44, !43, i64 0}
// CHECK:STDOUT: !101 = !DIFile(filename: "compare.carbon", directory: "")
// CHECK:STDOUT: !102 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !101, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !103 = !{!9, !6, !6}
// CHECK:STDOUT: !104 = !DISubroutineType(types: !103)
// CHECK:STDOUT: !105 = distinct !DISubprogram(name: "Compare", linkageName: "_CCompare.Main", scope: null, file: !101, line: 6, type: !104, spFlags: DISPFlagDefinition, unit: !102, retainedNodes: !115)
// CHECK:STDOUT: !106 = !DILocalVariable(arg: 1, scope: !105, type: !6)
// CHECK:STDOUT: !107 = !DILocalVariable(arg: 2, scope: !105, type: !6)
// CHECK:STDOUT: !112 = !DILocation(line: 7, column: 10, scope: !105)
// CHECK:STDOUT: !114 = !DILocation(line: 7, column: 3, scope: !105)
// CHECK:STDOUT: !115 = !{!106, !107}
// CHECK:STDOUT: !116 = distinct !DISubprogram(name: "CallCompareGeneric", linkageName: "_CCallCompareGeneric.Main", scope: null, file: !101, line: 14, type: !104, spFlags: DISPFlagDefinition, unit: !102, retainedNodes: !127)
// CHECK:STDOUT: !117 = !DILocalVariable(arg: 1, scope: !116, type: !6)
// CHECK:STDOUT: !118 = !DILocalVariable(arg: 2, scope: !116, type: !6)
// CHECK:STDOUT: !125 = !DILocation(line: 15, column: 10, scope: !116)
// CHECK:STDOUT: !126 = !DILocation(line: 15, column: 3, scope: !116)
// CHECK:STDOUT: !127 = !{!117, !118}
// CHECK:STDOUT: !128 = distinct !DISubprogram(name: "CompareGeneric", linkageName: "_CCompareGeneric.Main.7602ebbc958eec83", scope: null, file: !101, line: 10, type: !104, spFlags: DISPFlagDefinition, unit: !102, retainedNodes: !138)
// CHECK:STDOUT: !129 = !DILocalVariable(arg: 1, scope: !128, type: !6)
// CHECK:STDOUT: !130 = !DILocalVariable(arg: 2, scope: !128, type: !6)
// CHECK:STDOUT: !135 = !DILocation(line: 11, column: 10, scope: !128)
// CHECK:STDOUT: !137 = !DILocation(line: 11, column: 3, scope: !128)
// CHECK:STDOUT: !138 = !{!129, !130}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'compare_class.carbon'
// CHECK:STDOUT: source_filename = "compare_class.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCompare.Main(i16 %a, ptr %b) #0 !dbg !149 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_15.1.temp = alloca i16, align 2, !dbg !157
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_15.1.temp), !dbg !157
// CHECK:STDOUT:   call void @_ZNK13ConvertToEnumcvN1C1EEEv.carbon_thunk._(ptr %b, ptr %.loc18_15.1.temp), !dbg !157
// CHECK:STDOUT:   %.loc18_15.5 = load i16, ptr %.loc18_15.1.temp, align 2, !dbg !157
// CHECK:STDOUT:   %OrderedWith.WithSelf.Equal.call = icmp eq i16 %a, %.loc18_15.5, !dbg !156
// CHECK:STDOUT:   call void @"_CSelfDestruct.4f81cc432cb50e07:core.Destroy.Core"(ptr %.loc18_15.1.temp), !dbg !157
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.Equal.call, !dbg !158
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZNK13ConvertToEnumcvN1C1EEEv.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !286
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !37
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !37
// CHECK:STDOUT:   %1 = load ptr, ptr %this.addr, align 8, !tbaa !286, !nonnull !288
// CHECK:STDOUT:   %call = call noundef signext i16 @_ZNK13ConvertToEnumcvN1C1EEEv(ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   store i16 %call, ptr %0, align 2, !tbaa !39
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.4f81cc432cb50e07:core.Destroy.Core"(ptr %self) #0 !dbg !160 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !162
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCompare2.Main(i32 %a, ptr %b) #0 !dbg !164 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %__carbon_thunk.call = call i32 @_ZNK13ConvertToEnumcv2E2Ev.carbon_thunk._(ptr %b), !dbg !172
// CHECK:STDOUT:   %OrderedWith.WithSelf.Equal.call = icmp eq i32 %a, %__carbon_thunk.call, !dbg !171
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.Equal.call, !dbg !173
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal noundef i32 @_ZNK13ConvertToEnumcv2E2Ev.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !286
// CHECK:STDOUT:   %0 = load ptr, ptr %this.addr, align 8, !tbaa !286, !nonnull !288
// CHECK:STDOUT:   %call = call noundef i32 @_ZNK13ConvertToEnumcv2E2Ev(ptr noundef nonnull align 1 dereferenceable(1) %0)
// CHECK:STDOUT:   ret i32 %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCompare3.Main(i16 %a, ptr %b) #0 !dbg !175 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc26_15.1.temp = alloca i16, align 2, !dbg !183
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc26_15.1.temp), !dbg !183
// CHECK:STDOUT:   call void @_ZNK14ConvertToEnum2cvN1C1EEEv.carbon_thunk._(ptr %b, ptr %.loc26_15.1.temp), !dbg !183
// CHECK:STDOUT:   %.loc26_15.5 = load i16, ptr %.loc26_15.1.temp, align 2, !dbg !183
// CHECK:STDOUT:   %OrderedWith.WithSelf.Equal.call = icmp eq i16 %a, %.loc26_15.5, !dbg !182
// CHECK:STDOUT:   call void @"_CSelfDestruct.4f81cc432cb50e07:core.Destroy.Core"(ptr %.loc26_15.1.temp), !dbg !183
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.Equal.call, !dbg !184
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZNK14ConvertToEnum2cvN1C1EEEv.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !290
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !37
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !37
// CHECK:STDOUT:   %1 = load ptr, ptr %this.addr, align 8, !tbaa !290, !nonnull !288
// CHECK:STDOUT:   %call = call noundef signext i16 @_ZNK14ConvertToEnum2cvN1C1EEEv(ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   store i16 %call, ptr %0, align 2, !tbaa !39
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCallCompareGeneric.Main(i16 %a, ptr %b) #0 !dbg !186 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %CompareGeneric.call = call i1 @_CCompareGeneric.Main.356c7bb457b5b49b(i16 %a, ptr %b), !dbg !195
// CHECK:STDOUT:   ret i1 %CompareGeneric.call, !dbg !196
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CEqual:thunk:EqWith.83b9626e83cf3dcc.Core:OrderedWith.Core"(i16 %self, ptr %other) #2 !dbg !198 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.4.temp = alloca i16, align 2, !dbg !201
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.4.temp), !dbg !201
// CHECK:STDOUT:   call void @_ZNK13ConvertToEnumcvN1C1EEEv.carbon_thunk._(ptr %other, ptr %.4.temp), !dbg !201
// CHECK:STDOUT:   %.8 = load i16, ptr %.4.temp, align 2, !dbg !201
// CHECK:STDOUT:   %OrderedWith.WithSelf.Equal.call = icmp eq i16 %self, %.8, !dbg !201
// CHECK:STDOUT:   call void @"_CSelfDestruct.4f81cc432cb50e07:core.Destroy.Core"(ptr %.4.temp), !dbg !201
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.Equal.call, !dbg !201
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CNotEqual:thunk:EqWith.83b9626e83cf3dcc.Core:OrderedWith.Core"(i16 %self, ptr %other) #2 !dbg !203 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.4.temp = alloca i16, align 2, !dbg !206
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.4.temp), !dbg !206
// CHECK:STDOUT:   call void @_ZNK13ConvertToEnumcvN1C1EEEv.carbon_thunk._(ptr %other, ptr %.4.temp), !dbg !206
// CHECK:STDOUT:   %.8 = load i16, ptr %.4.temp, align 2, !dbg !206
// CHECK:STDOUT:   %OrderedWith.WithSelf.NotEqual.call = icmp ne i16 %self, %.8, !dbg !206
// CHECK:STDOUT:   call void @"_CSelfDestruct.4f81cc432cb50e07:core.Destroy.Core"(ptr %.4.temp), !dbg !206
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.NotEqual.call, !dbg !206
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCallCompareGeneric2.Main(i32 %a, ptr %b) #0 !dbg !208 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %CompareGeneric.call = call i1 @_CCompareGeneric.Main.4512c7023c79a313(i32 %a, ptr %b), !dbg !217
// CHECK:STDOUT:   ret i1 %CompareGeneric.call, !dbg !218
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CEqual:thunk:EqWith.e8fe48de943d9652.Core:OrderedWith.Core"(i32 %self, ptr %other) #2 !dbg !220 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %__carbon_thunk.call = call i32 @_ZNK13ConvertToEnumcv2E2Ev.carbon_thunk._(ptr %other), !dbg !223
// CHECK:STDOUT:   %OrderedWith.WithSelf.Equal.call = icmp eq i32 %self, %__carbon_thunk.call, !dbg !223
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.Equal.call, !dbg !223
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CNotEqual:thunk:EqWith.e8fe48de943d9652.Core:OrderedWith.Core"(i32 %self, ptr %other) #2 !dbg !225 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %__carbon_thunk.call = call i32 @_ZNK13ConvertToEnumcv2E2Ev.carbon_thunk._(ptr %other), !dbg !228
// CHECK:STDOUT:   %OrderedWith.WithSelf.NotEqual.call = icmp ne i32 %self, %__carbon_thunk.call, !dbg !228
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.NotEqual.call, !dbg !228
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCallCompareGeneric3.Main(i16 %a, ptr %b) #0 !dbg !230 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %CompareGeneric.call = call i1 @_CCompareGeneric.Main.070a42571d1942e2(i16 %a, ptr %b), !dbg !239
// CHECK:STDOUT:   ret i1 %CompareGeneric.call, !dbg !240
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CEqual:thunk:EqWith.a294139a6615e163.Core:OrderedWith.Core"(i16 %self, ptr %other) #2 !dbg !242 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.4.temp = alloca i16, align 2, !dbg !245
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.4.temp), !dbg !245
// CHECK:STDOUT:   call void @_ZNK14ConvertToEnum2cvN1C1EEEv.carbon_thunk._(ptr %other, ptr %.4.temp), !dbg !245
// CHECK:STDOUT:   %.8 = load i16, ptr %.4.temp, align 2, !dbg !245
// CHECK:STDOUT:   %OrderedWith.WithSelf.Equal.call = icmp eq i16 %self, %.8, !dbg !245
// CHECK:STDOUT:   call void @"_CSelfDestruct.4f81cc432cb50e07:core.Destroy.Core"(ptr %.4.temp), !dbg !245
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.Equal.call, !dbg !245
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CNotEqual:thunk:EqWith.a294139a6615e163.Core:OrderedWith.Core"(i16 %self, ptr %other) #2 !dbg !247 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.4.temp = alloca i16, align 2, !dbg !250
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.4.temp), !dbg !250
// CHECK:STDOUT:   call void @_ZNK14ConvertToEnum2cvN1C1EEEv.carbon_thunk._(ptr %other, ptr %.4.temp), !dbg !250
// CHECK:STDOUT:   %.8 = load i16, ptr %.4.temp, align 2, !dbg !250
// CHECK:STDOUT:   %OrderedWith.WithSelf.NotEqual.call = icmp ne i16 %self, %.8, !dbg !250
// CHECK:STDOUT:   call void @"_CSelfDestruct.4f81cc432cb50e07:core.Destroy.Core"(ptr %.4.temp), !dbg !250
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.NotEqual.call, !dbg !250
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CCompareGeneric.Main.356c7bb457b5b49b(i16 %x, ptr %y) #0 !dbg !252 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.call = call i1 @"_CEqual:thunk:EqWith.83b9626e83cf3dcc.Core:OrderedWith.Core"(i16 %x, ptr %y), !dbg !259
// CHECK:STDOUT:   ret i1 %EqWith.WithSelf.Equal.call, !dbg !261
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CCompareGeneric.Main.4512c7023c79a313(i32 %x, ptr %y) #0 !dbg !263 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.call = call i1 @"_CEqual:thunk:EqWith.e8fe48de943d9652.Core:OrderedWith.Core"(i32 %x, ptr %y), !dbg !270
// CHECK:STDOUT:   ret i1 %EqWith.WithSelf.Equal.call, !dbg !272
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CCompareGeneric.Main.070a42571d1942e2(i16 %x, ptr %y) #0 !dbg !274 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.call = call i1 @"_CEqual:thunk:EqWith.a294139a6615e163.Core:OrderedWith.Core"(i16 %x, ptr %y), !dbg !281
// CHECK:STDOUT:   ret i1 %EqWith.WithSelf.Equal.call, !dbg !283
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!140}
// CHECK:STDOUT: !llvm.module.flags = !{!40, !41, !42, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!45}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !12 = !{null, !6}
// CHECK:STDOUT: !13 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !34 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !35 = !{!"omnipotent char", !34, i64 0}
// CHECK:STDOUT: !36 = !{!"any pointer", !35, i64 0}
// CHECK:STDOUT: !37 = !{!36, !36, i64 0}
// CHECK:STDOUT: !38 = !{!"_ZTSN1C1EE", !35, i64 0}
// CHECK:STDOUT: !39 = !{!38, !38, i64 0}
// CHECK:STDOUT: !40 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !41 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !42 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !43 = !{!"int", !35, i64 0}
// CHECK:STDOUT: !44 = !{!"__libc_errno", !43, i64 0}
// CHECK:STDOUT: !45 = !{!44, !43, i64 0}
// CHECK:STDOUT: !71 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !139 = !DIFile(filename: "compare_class.carbon", directory: "")
// CHECK:STDOUT: !140 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !139, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !141 = !{!9, !6, !9}
// CHECK:STDOUT: !142 = !DISubroutineType(types: !141)
// CHECK:STDOUT: !145 = !{!9, !71, !9}
// CHECK:STDOUT: !146 = !DISubroutineType(types: !145)
// CHECK:STDOUT: !149 = distinct !DISubprogram(name: "Compare", linkageName: "_CCompare.Main", scope: null, file: !139, line: 17, type: !142, spFlags: DISPFlagDefinition, unit: !140, retainedNodes: !159)
// CHECK:STDOUT: !150 = !DILocalVariable(arg: 1, scope: !149, type: !6)
// CHECK:STDOUT: !151 = !DILocalVariable(arg: 2, scope: !149, type: !9)
// CHECK:STDOUT: !156 = !DILocation(line: 18, column: 10, scope: !149)
// CHECK:STDOUT: !157 = !DILocation(line: 18, column: 15, scope: !149)
// CHECK:STDOUT: !158 = !DILocation(line: 18, column: 3, scope: !149)
// CHECK:STDOUT: !159 = !{!150, !151}
// CHECK:STDOUT: !160 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.4f81cc432cb50e07:core.Destroy.Core", scope: null, file: !139, line: 18, type: !13, spFlags: DISPFlagDefinition, unit: !140, retainedNodes: !163)
// CHECK:STDOUT: !161 = !DILocalVariable(arg: 1, scope: !160, type: !6)
// CHECK:STDOUT: !162 = !DILocation(line: 18, column: 15, scope: !160)
// CHECK:STDOUT: !163 = !{!161}
// CHECK:STDOUT: !164 = distinct !DISubprogram(name: "Compare2", linkageName: "_CCompare2.Main", scope: null, file: !139, line: 21, type: !146, spFlags: DISPFlagDefinition, unit: !140, retainedNodes: !174)
// CHECK:STDOUT: !165 = !DILocalVariable(arg: 1, scope: !164, type: !71)
// CHECK:STDOUT: !166 = !DILocalVariable(arg: 2, scope: !164, type: !9)
// CHECK:STDOUT: !171 = !DILocation(line: 22, column: 10, scope: !164)
// CHECK:STDOUT: !172 = !DILocation(line: 22, column: 15, scope: !164)
// CHECK:STDOUT: !173 = !DILocation(line: 22, column: 3, scope: !164)
// CHECK:STDOUT: !174 = !{!165, !166}
// CHECK:STDOUT: !175 = distinct !DISubprogram(name: "Compare3", linkageName: "_CCompare3.Main", scope: null, file: !139, line: 25, type: !142, spFlags: DISPFlagDefinition, unit: !140, retainedNodes: !185)
// CHECK:STDOUT: !176 = !DILocalVariable(arg: 1, scope: !175, type: !6)
// CHECK:STDOUT: !177 = !DILocalVariable(arg: 2, scope: !175, type: !9)
// CHECK:STDOUT: !182 = !DILocation(line: 26, column: 10, scope: !175)
// CHECK:STDOUT: !183 = !DILocation(line: 26, column: 15, scope: !175)
// CHECK:STDOUT: !184 = !DILocation(line: 26, column: 3, scope: !175)
// CHECK:STDOUT: !185 = !{!176, !177}
// CHECK:STDOUT: !186 = distinct !DISubprogram(name: "CallCompareGeneric", linkageName: "_CCallCompareGeneric.Main", scope: null, file: !139, line: 35, type: !142, spFlags: DISPFlagDefinition, unit: !140, retainedNodes: !197)
// CHECK:STDOUT: !187 = !DILocalVariable(arg: 1, scope: !186, type: !6)
// CHECK:STDOUT: !188 = !DILocalVariable(arg: 2, scope: !186, type: !9)
// CHECK:STDOUT: !195 = !DILocation(line: 36, column: 10, scope: !186)
// CHECK:STDOUT: !196 = !DILocation(line: 36, column: 3, scope: !186)
// CHECK:STDOUT: !197 = !{!187, !188}
// CHECK:STDOUT: !198 = distinct !DISubprogram(name: "Equal", linkageName: "_CEqual:thunk:EqWith.83b9626e83cf3dcc.Core:OrderedWith.Core", scope: null, file: !139, type: !142, spFlags: DISPFlagDefinition, unit: !140, retainedNodes: !202)
// CHECK:STDOUT: !199 = !DILocalVariable(arg: 1, scope: !198, type: !6)
// CHECK:STDOUT: !200 = !DILocalVariable(arg: 2, scope: !198, type: !9)
// CHECK:STDOUT: !201 = !DILocation(line: 0, scope: !198)
// CHECK:STDOUT: !202 = !{!199, !200}
// CHECK:STDOUT: !203 = distinct !DISubprogram(name: "NotEqual", linkageName: "_CNotEqual:thunk:EqWith.83b9626e83cf3dcc.Core:OrderedWith.Core", scope: null, file: !139, type: !142, spFlags: DISPFlagDefinition, unit: !140, retainedNodes: !207)
// CHECK:STDOUT: !204 = !DILocalVariable(arg: 1, scope: !203, type: !6)
// CHECK:STDOUT: !205 = !DILocalVariable(arg: 2, scope: !203, type: !9)
// CHECK:STDOUT: !206 = !DILocation(line: 0, scope: !203)
// CHECK:STDOUT: !207 = !{!204, !205}
// CHECK:STDOUT: !208 = distinct !DISubprogram(name: "CallCompareGeneric2", linkageName: "_CCallCompareGeneric2.Main", scope: null, file: !139, line: 39, type: !146, spFlags: DISPFlagDefinition, unit: !140, retainedNodes: !219)
// CHECK:STDOUT: !209 = !DILocalVariable(arg: 1, scope: !208, type: !71)
// CHECK:STDOUT: !210 = !DILocalVariable(arg: 2, scope: !208, type: !9)
// CHECK:STDOUT: !217 = !DILocation(line: 40, column: 10, scope: !208)
// CHECK:STDOUT: !218 = !DILocation(line: 40, column: 3, scope: !208)
// CHECK:STDOUT: !219 = !{!209, !210}
// CHECK:STDOUT: !220 = distinct !DISubprogram(name: "Equal", linkageName: "_CEqual:thunk:EqWith.e8fe48de943d9652.Core:OrderedWith.Core", scope: null, file: !139, type: !146, spFlags: DISPFlagDefinition, unit: !140, retainedNodes: !224)
// CHECK:STDOUT: !221 = !DILocalVariable(arg: 1, scope: !220, type: !71)
// CHECK:STDOUT: !222 = !DILocalVariable(arg: 2, scope: !220, type: !9)
// CHECK:STDOUT: !223 = !DILocation(line: 0, scope: !220)
// CHECK:STDOUT: !224 = !{!221, !222}
// CHECK:STDOUT: !225 = distinct !DISubprogram(name: "NotEqual", linkageName: "_CNotEqual:thunk:EqWith.e8fe48de943d9652.Core:OrderedWith.Core", scope: null, file: !139, type: !146, spFlags: DISPFlagDefinition, unit: !140, retainedNodes: !229)
// CHECK:STDOUT: !226 = !DILocalVariable(arg: 1, scope: !225, type: !71)
// CHECK:STDOUT: !227 = !DILocalVariable(arg: 2, scope: !225, type: !9)
// CHECK:STDOUT: !228 = !DILocation(line: 0, scope: !225)
// CHECK:STDOUT: !229 = !{!226, !227}
// CHECK:STDOUT: !230 = distinct !DISubprogram(name: "CallCompareGeneric3", linkageName: "_CCallCompareGeneric3.Main", scope: null, file: !139, line: 43, type: !142, spFlags: DISPFlagDefinition, unit: !140, retainedNodes: !241)
// CHECK:STDOUT: !231 = !DILocalVariable(arg: 1, scope: !230, type: !6)
// CHECK:STDOUT: !232 = !DILocalVariable(arg: 2, scope: !230, type: !9)
// CHECK:STDOUT: !239 = !DILocation(line: 44, column: 10, scope: !230)
// CHECK:STDOUT: !240 = !DILocation(line: 44, column: 3, scope: !230)
// CHECK:STDOUT: !241 = !{!231, !232}
// CHECK:STDOUT: !242 = distinct !DISubprogram(name: "Equal", linkageName: "_CEqual:thunk:EqWith.a294139a6615e163.Core:OrderedWith.Core", scope: null, file: !139, type: !142, spFlags: DISPFlagDefinition, unit: !140, retainedNodes: !246)
// CHECK:STDOUT: !243 = !DILocalVariable(arg: 1, scope: !242, type: !6)
// CHECK:STDOUT: !244 = !DILocalVariable(arg: 2, scope: !242, type: !9)
// CHECK:STDOUT: !245 = !DILocation(line: 0, scope: !242)
// CHECK:STDOUT: !246 = !{!243, !244}
// CHECK:STDOUT: !247 = distinct !DISubprogram(name: "NotEqual", linkageName: "_CNotEqual:thunk:EqWith.a294139a6615e163.Core:OrderedWith.Core", scope: null, file: !139, type: !142, spFlags: DISPFlagDefinition, unit: !140, retainedNodes: !251)
// CHECK:STDOUT: !248 = !DILocalVariable(arg: 1, scope: !247, type: !6)
// CHECK:STDOUT: !249 = !DILocalVariable(arg: 2, scope: !247, type: !9)
// CHECK:STDOUT: !250 = !DILocation(line: 0, scope: !247)
// CHECK:STDOUT: !251 = !{!248, !249}
// CHECK:STDOUT: !252 = distinct !DISubprogram(name: "CompareGeneric", linkageName: "_CCompareGeneric.Main.356c7bb457b5b49b", scope: null, file: !139, line: 29, type: !142, spFlags: DISPFlagDefinition, unit: !140, retainedNodes: !262)
// CHECK:STDOUT: !253 = !DILocalVariable(arg: 1, scope: !252, type: !6)
// CHECK:STDOUT: !254 = !DILocalVariable(arg: 2, scope: !252, type: !9)
// CHECK:STDOUT: !259 = !DILocation(line: 30, column: 10, scope: !252)
// CHECK:STDOUT: !261 = !DILocation(line: 30, column: 3, scope: !252)
// CHECK:STDOUT: !262 = !{!253, !254}
// CHECK:STDOUT: !263 = distinct !DISubprogram(name: "CompareGeneric", linkageName: "_CCompareGeneric.Main.4512c7023c79a313", scope: null, file: !139, line: 29, type: !146, spFlags: DISPFlagDefinition, unit: !140, retainedNodes: !273)
// CHECK:STDOUT: !264 = !DILocalVariable(arg: 1, scope: !263, type: !71)
// CHECK:STDOUT: !265 = !DILocalVariable(arg: 2, scope: !263, type: !9)
// CHECK:STDOUT: !270 = !DILocation(line: 30, column: 10, scope: !263)
// CHECK:STDOUT: !272 = !DILocation(line: 30, column: 3, scope: !263)
// CHECK:STDOUT: !273 = !{!264, !265}
// CHECK:STDOUT: !274 = distinct !DISubprogram(name: "CompareGeneric", linkageName: "_CCompareGeneric.Main.070a42571d1942e2", scope: null, file: !139, line: 29, type: !142, spFlags: DISPFlagDefinition, unit: !140, retainedNodes: !284)
// CHECK:STDOUT: !275 = !DILocalVariable(arg: 1, scope: !274, type: !6)
// CHECK:STDOUT: !276 = !DILocalVariable(arg: 2, scope: !274, type: !9)
// CHECK:STDOUT: !281 = !DILocation(line: 30, column: 10, scope: !274)
// CHECK:STDOUT: !283 = !DILocation(line: 30, column: 3, scope: !274)
// CHECK:STDOUT: !284 = !{!275, !276}
// CHECK:STDOUT: !285 = !{!"p1 _ZTS13ConvertToEnum", !36, i64 0}
// CHECK:STDOUT: !286 = !{!285, !285, i64 0}
// CHECK:STDOUT: !288 = !{}
// CHECK:STDOUT: !289 = !{!"p1 _ZTS14ConvertToEnum2", !36, i64 0}
// CHECK:STDOUT: !290 = !{!289, !289, i64 0}
// CHECK:STDOUT:
