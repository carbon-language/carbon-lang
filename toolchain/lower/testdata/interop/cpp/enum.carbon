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
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.dd9c146f84d40ff5(ptr @int_0), !dbg !13
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc8_16.1.temp), !dbg !17
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr @int_0), !dbg !18
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.dd9c146f84d40ff5(ptr @int_0), !dbg !17
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc9_16.1.temp), !dbg !19
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr @int_7), !dbg !20
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.dd9c146f84d40ff5(ptr @int_7), !dbg !19
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc10_16.1.temp), !dbg !21
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr @int_8), !dbg !22
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.dd9c146f84d40ff5(ptr @int_8), !dbg !21
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc11_18.1.temp), !dbg !23
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr @int_0), !dbg !24
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.dd9c146f84d40ff5(ptr @int_0), !dbg !23
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc12_18.1.temp), !dbg !25
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr @int_7), !dbg !26
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.dd9c146f84d40ff5(ptr @int_7), !dbg !25
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc13_18.1.temp), !dbg !27
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr @int_8), !dbg !28
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.dd9c146f84d40ff5(ptr @int_8), !dbg !27
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
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.dd9c146f84d40ff5(ptr %t) #0 !dbg !31 {
// CHECK:STDOUT:   ret void, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN1C1FENS_1EE(i16 noundef signext) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_ZN1C1FENS_1EE.carbon_thunk._, { 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.dd9c146f84d40ff5, { 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!44, !45, !46, !47, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!50}
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
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.dd9c146f84d40ff5", scope: null, file: !30, line: 36, type: !16, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !37)
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !31, type: !6)
// CHECK:STDOUT: !36 = !DILocation(line: 36, column: 1, scope: !31)
// CHECK:STDOUT: !37 = !{!32}
// CHECK:STDOUT: !38 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !39 = !{!"omnipotent char", !38, i64 0}
// CHECK:STDOUT: !40 = !{!"any pointer", !39, i64 0}
// CHECK:STDOUT: !41 = !{!40, !40, i64 0}
// CHECK:STDOUT: !42 = !{!"_ZTSN1C1EE", !39, i64 0}
// CHECK:STDOUT: !43 = !{!42, !42, i64 0}
// CHECK:STDOUT: !44 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !45 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !46 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !47 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !48 = !{!"int", !39, i64 0}
// CHECK:STDOUT: !49 = !{!"__libc_errno", !48, i64 0}
// CHECK:STDOUT: !50 = !{!49, !48, i64 0}
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
// CHECK:STDOUT: define void @_CPassEnum.Main() #0 !dbg !53 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CTakeI16.Main(i16 7), !dbg !54
// CHECK:STDOUT:   ret void, !dbg !55
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertEnumToI16.Main(i16 %e) #0 !dbg !56 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CTakeI16.Main(i16 %e), !dbg !60
// CHECK:STDOUT:   ret void, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertI16ToEnum.Main(i16 %n) #0 !dbg !63 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17_13.3.temp = alloca i16, align 2, !dbg !66
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc17_13.3.temp), !dbg !66
// CHECK:STDOUT:   store i16 %n, ptr %.loc17_13.3.temp, align 2, !dbg !66
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr %.loc17_13.3.temp), !dbg !67
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.dd9c146f84d40ff5(ptr %.loc17_13.3.temp), !dbg !66
// CHECK:STDOUT:   ret void, !dbg !68
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
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.dd9c146f84d40ff5(ptr %t) #0 !dbg !70 {
// CHECK:STDOUT:   ret void, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN1C1FENS_1EE(i16 noundef signext) #3
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!52}
// CHECK:STDOUT: !llvm.module.flags = !{!44, !45, !46, !47, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!50}
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
// CHECK:STDOUT: !44 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !45 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !46 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !47 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !48 = !{!"int", !39, i64 0}
// CHECK:STDOUT: !49 = !{!"__libc_errno", !48, i64 0}
// CHECK:STDOUT: !50 = !{!49, !48, i64 0}
// CHECK:STDOUT: !51 = !DIFile(filename: "convert.carbon", directory: "")
// CHECK:STDOUT: !52 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !51, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !53 = distinct !DISubprogram(name: "PassEnum", linkageName: "_CPassEnum.Main", scope: null, file: !51, line: 8, type: !8, spFlags: DISPFlagDefinition, unit: !52)
// CHECK:STDOUT: !54 = !DILocation(line: 9, column: 3, scope: !53)
// CHECK:STDOUT: !55 = !DILocation(line: 8, column: 1, scope: !53)
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "ConvertEnumToI16", linkageName: "_CConvertEnumToI16.Main", scope: null, file: !51, line: 12, type: !16, spFlags: DISPFlagDefinition, unit: !52, retainedNodes: !62)
// CHECK:STDOUT: !57 = !DILocalVariable(arg: 1, scope: !56, type: !6)
// CHECK:STDOUT: !60 = !DILocation(line: 13, column: 3, scope: !56)
// CHECK:STDOUT: !61 = !DILocation(line: 12, column: 1, scope: !56)
// CHECK:STDOUT: !62 = !{!57}
// CHECK:STDOUT: !63 = distinct !DISubprogram(name: "ConvertI16ToEnum", linkageName: "_CConvertI16ToEnum.Main", scope: null, file: !51, line: 16, type: !16, spFlags: DISPFlagDefinition, unit: !52, retainedNodes: !69)
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !63, type: !6)
// CHECK:STDOUT: !66 = !DILocation(line: 17, column: 11, scope: !63)
// CHECK:STDOUT: !67 = !DILocation(line: 17, column: 3, scope: !63)
// CHECK:STDOUT: !68 = !DILocation(line: 16, column: 1, scope: !63)
// CHECK:STDOUT: !69 = !{!64}
// CHECK:STDOUT: !70 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.dd9c146f84d40ff5", scope: null, file: !30, line: 36, type: !16, spFlags: DISPFlagDefinition, unit: !52, retainedNodes: !76)
// CHECK:STDOUT: !71 = !DILocalVariable(arg: 1, scope: !70, type: !6)
// CHECK:STDOUT: !75 = !DILocation(line: 36, column: 1, scope: !70)
// CHECK:STDOUT: !76 = !{!71}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'use_bitmask.carbon'
// CHECK:STDOUT: source_filename = "use_bitmask.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCompileTime.Main() #0 !dbg !84 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 0), !dbg !85
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 5), !dbg !86
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 5), !dbg !87
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 -2), !dbg !88
// CHECK:STDOUT:   ret void, !dbg !89
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z4Take4Bits(i32 noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CRuntime.Main(i32 %a, i32 %b) #0 !dbg !90 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.call = and i32 %a, %b, !dbg !95
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 %BitAndWith.WithSelf.Op.call), !dbg !97
// CHECK:STDOUT:   %BitOrWith.WithSelf.Op.call = or i32 %a, %b, !dbg !98
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 %BitOrWith.WithSelf.Op.call), !dbg !100
// CHECK:STDOUT:   %BitXorWith.WithSelf.Op.call = xor i32 %a, %b, !dbg !101
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 %BitXorWith.WithSelf.Op.call), !dbg !103
// CHECK:STDOUT:   %BitComplement.WithSelf.Op.call = xor i32 -1, %a, !dbg !105
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 %BitComplement.WithSelf.Op.call), !dbg !106
// CHECK:STDOUT:   ret void, !dbg !107
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_Z4Take4Bits, { 0, 1, 2, 3, 7, 6, 5, 4 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!78}
// CHECK:STDOUT: !llvm.module.flags = !{!44, !45, !46, !47, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!50}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !{null}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !38 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !39 = !{!"omnipotent char", !38, i64 0}
// CHECK:STDOUT: !44 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !45 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !46 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !47 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !48 = !{!"int", !39, i64 0}
// CHECK:STDOUT: !49 = !{!"__libc_errno", !48, i64 0}
// CHECK:STDOUT: !50 = !{!49, !48, i64 0}
// CHECK:STDOUT: !77 = !DIFile(filename: "use_bitmask.carbon", directory: "")
// CHECK:STDOUT: !78 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !77, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !79 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !82 = !{null, !79, !79}
// CHECK:STDOUT: !83 = !DISubroutineType(types: !82)
// CHECK:STDOUT: !84 = distinct !DISubprogram(name: "CompileTime", linkageName: "_CCompileTime.Main", scope: null, file: !77, line: 6, type: !8, spFlags: DISPFlagDefinition, unit: !78)
// CHECK:STDOUT: !85 = !DILocation(line: 7, column: 3, scope: !84)
// CHECK:STDOUT: !86 = !DILocation(line: 8, column: 3, scope: !84)
// CHECK:STDOUT: !87 = !DILocation(line: 9, column: 3, scope: !84)
// CHECK:STDOUT: !88 = !DILocation(line: 10, column: 3, scope: !84)
// CHECK:STDOUT: !89 = !DILocation(line: 6, column: 1, scope: !84)
// CHECK:STDOUT: !90 = distinct !DISubprogram(name: "Runtime", linkageName: "_CRuntime.Main", scope: null, file: !77, line: 13, type: !83, spFlags: DISPFlagDefinition, unit: !78, retainedNodes: !108)
// CHECK:STDOUT: !91 = !DILocalVariable(arg: 1, scope: !90, type: !79)
// CHECK:STDOUT: !92 = !DILocalVariable(arg: 2, scope: !90, type: !79)
// CHECK:STDOUT: !95 = !DILocation(line: 14, column: 12, scope: !90)
// CHECK:STDOUT: !97 = !DILocation(line: 14, column: 3, scope: !90)
// CHECK:STDOUT: !98 = !DILocation(line: 15, column: 12, scope: !90)
// CHECK:STDOUT: !100 = !DILocation(line: 15, column: 3, scope: !90)
// CHECK:STDOUT: !101 = !DILocation(line: 16, column: 12, scope: !90)
// CHECK:STDOUT: !103 = !DILocation(line: 16, column: 3, scope: !90)
// CHECK:STDOUT: !105 = !DILocation(line: 20, column: 12, scope: !90)
// CHECK:STDOUT: !106 = !DILocation(line: 20, column: 3, scope: !90)
// CHECK:STDOUT: !107 = !DILocation(line: 13, column: 1, scope: !90)
// CHECK:STDOUT: !108 = !{!91, !92}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'compare.carbon'
// CHECK:STDOUT: source_filename = "compare.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCompare.Main(i16 %a, i16 %b) #0 !dbg !113 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %OrderedWith.WithSelf.Equal.call = icmp eq i16 %a, %b, !dbg !120
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.Equal.call, !dbg !122
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCallCompareGeneric.Main(i16 %a, i16 %b) #0 !dbg !124 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %CompareGeneric.call = call i1 @_CCompareGeneric.Main.04aebd1c4efdc136(i16 %a, i16 %b), !dbg !133
// CHECK:STDOUT:   ret i1 %CompareGeneric.call, !dbg !134
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CCompareGeneric.Main.04aebd1c4efdc136(i16 %x, i16 %y) #0 !dbg !136 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.call = icmp eq i16 %x, %y, !dbg !143
// CHECK:STDOUT:   ret i1 %EqWith.WithSelf.Equal.call, !dbg !145
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!110}
// CHECK:STDOUT: !llvm.module.flags = !{!44, !45, !46, !47, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!50}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !38 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !39 = !{!"omnipotent char", !38, i64 0}
// CHECK:STDOUT: !44 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !45 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !46 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !47 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !48 = !{!"int", !39, i64 0}
// CHECK:STDOUT: !49 = !{!"__libc_errno", !48, i64 0}
// CHECK:STDOUT: !50 = !{!49, !48, i64 0}
// CHECK:STDOUT: !109 = !DIFile(filename: "compare.carbon", directory: "")
// CHECK:STDOUT: !110 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !109, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !111 = !{!9, !6, !6}
// CHECK:STDOUT: !112 = !DISubroutineType(types: !111)
// CHECK:STDOUT: !113 = distinct !DISubprogram(name: "Compare", linkageName: "_CCompare.Main", scope: null, file: !109, line: 6, type: !112, spFlags: DISPFlagDefinition, unit: !110, retainedNodes: !123)
// CHECK:STDOUT: !114 = !DILocalVariable(arg: 1, scope: !113, type: !6)
// CHECK:STDOUT: !115 = !DILocalVariable(arg: 2, scope: !113, type: !6)
// CHECK:STDOUT: !120 = !DILocation(line: 7, column: 10, scope: !113)
// CHECK:STDOUT: !122 = !DILocation(line: 7, column: 3, scope: !113)
// CHECK:STDOUT: !123 = !{!114, !115}
// CHECK:STDOUT: !124 = distinct !DISubprogram(name: "CallCompareGeneric", linkageName: "_CCallCompareGeneric.Main", scope: null, file: !109, line: 14, type: !112, spFlags: DISPFlagDefinition, unit: !110, retainedNodes: !135)
// CHECK:STDOUT: !125 = !DILocalVariable(arg: 1, scope: !124, type: !6)
// CHECK:STDOUT: !126 = !DILocalVariable(arg: 2, scope: !124, type: !6)
// CHECK:STDOUT: !133 = !DILocation(line: 15, column: 10, scope: !124)
// CHECK:STDOUT: !134 = !DILocation(line: 15, column: 3, scope: !124)
// CHECK:STDOUT: !135 = !{!125, !126}
// CHECK:STDOUT: !136 = distinct !DISubprogram(name: "CompareGeneric", linkageName: "_CCompareGeneric.Main.04aebd1c4efdc136", scope: null, file: !109, line: 10, type: !112, spFlags: DISPFlagDefinition, unit: !110, retainedNodes: !146)
// CHECK:STDOUT: !137 = !DILocalVariable(arg: 1, scope: !136, type: !6)
// CHECK:STDOUT: !138 = !DILocalVariable(arg: 2, scope: !136, type: !6)
// CHECK:STDOUT: !143 = !DILocation(line: 11, column: 10, scope: !136)
// CHECK:STDOUT: !145 = !DILocation(line: 11, column: 3, scope: !136)
// CHECK:STDOUT: !146 = !{!137, !138}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'compare_class.carbon'
// CHECK:STDOUT: source_filename = "compare_class.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCompare.Main(i16 %a, ptr %b) #0 !dbg !157 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_15.1.temp = alloca i16, align 2, !dbg !165
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_15.1.temp), !dbg !165
// CHECK:STDOUT:   call void @_ZNK13ConvertToEnumcvN1C1EEEv.carbon_thunk._(ptr %b, ptr %.loc18_15.1.temp), !dbg !165
// CHECK:STDOUT:   %.loc18_15.5 = load i16, ptr %.loc18_15.1.temp, align 2, !dbg !165
// CHECK:STDOUT:   %OrderedWith.WithSelf.Equal.call = icmp eq i16 %a, %.loc18_15.5, !dbg !164
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.dd9c146f84d40ff5(ptr %.loc18_15.1.temp), !dbg !165
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.Equal.call, !dbg !166
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZNK13ConvertToEnumcvN1C1EEEv.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !297
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !41
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !41
// CHECK:STDOUT:   %1 = load ptr, ptr %this.addr, align 8, !tbaa !297, !nonnull !299
// CHECK:STDOUT:   %call = call noundef i16 @_ZNK13ConvertToEnumcvN1C1EEEv(ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   store i16 %call, ptr %0, align 2, !tbaa !43
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCompare2.Main(i32 %a, ptr %b) #0 !dbg !168 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %__carbon_thunk.call = call i32 @_ZNK13ConvertToEnumcv2E2Ev.carbon_thunk._(ptr %b), !dbg !176
// CHECK:STDOUT:   %OrderedWith.WithSelf.Equal.call = icmp eq i32 %a, %__carbon_thunk.call, !dbg !175
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.Equal.call, !dbg !177
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal noundef i32 @_ZNK13ConvertToEnumcv2E2Ev.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !297
// CHECK:STDOUT:   %0 = load ptr, ptr %this.addr, align 8, !tbaa !297, !nonnull !299
// CHECK:STDOUT:   %call = call noundef i32 @_ZNK13ConvertToEnumcv2E2Ev(ptr noundef nonnull align 1 dereferenceable(1) %0)
// CHECK:STDOUT:   ret i32 %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCompare3.Main(i16 %a, ptr %b) #0 !dbg !179 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc26_15.1.temp = alloca i16, align 2, !dbg !187
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc26_15.1.temp), !dbg !187
// CHECK:STDOUT:   call void @_ZNK14ConvertToEnum2cvN1C1EEEv.carbon_thunk._(ptr %b, ptr %.loc26_15.1.temp), !dbg !187
// CHECK:STDOUT:   %.loc26_15.5 = load i16, ptr %.loc26_15.1.temp, align 2, !dbg !187
// CHECK:STDOUT:   %OrderedWith.WithSelf.Equal.call = icmp eq i16 %a, %.loc26_15.5, !dbg !186
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.dd9c146f84d40ff5(ptr %.loc26_15.1.temp), !dbg !187
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.Equal.call, !dbg !188
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZNK14ConvertToEnum2cvN1C1EEEv.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !301
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !41
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !41
// CHECK:STDOUT:   %1 = load ptr, ptr %this.addr, align 8, !tbaa !301, !nonnull !299
// CHECK:STDOUT:   %call = call noundef i16 @_ZNK14ConvertToEnum2cvN1C1EEEv(ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   store i16 %call, ptr %0, align 2, !tbaa !43
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCallCompareGeneric.Main(i16 %a, ptr %b) #0 !dbg !190 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %CompareGeneric.call = call i1 @_CCompareGeneric.Main.9f771e4a50bafdbe(i16 %a, ptr %b), !dbg !199
// CHECK:STDOUT:   ret i1 %CompareGeneric.call, !dbg !200
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CEqual:thunk:EqWith.289080fd154ab9ff.Core:OrderedWith.Core"(i16 %self, ptr %other) #2 !dbg !202 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.4.temp = alloca i16, align 2, !dbg !205
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.4.temp), !dbg !205
// CHECK:STDOUT:   call void @_ZNK13ConvertToEnumcvN1C1EEEv.carbon_thunk._(ptr %other, ptr %.4.temp), !dbg !205
// CHECK:STDOUT:   %.8 = load i16, ptr %.4.temp, align 2, !dbg !205
// CHECK:STDOUT:   %OrderedWith.WithSelf.Equal.call = icmp eq i16 %self, %.8, !dbg !205
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.dd9c146f84d40ff5(ptr %.4.temp), !dbg !205
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.Equal.call, !dbg !205
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CNotEqual:thunk:EqWith.289080fd154ab9ff.Core:OrderedWith.Core"(i16 %self, ptr %other) #2 !dbg !207 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.4.temp = alloca i16, align 2, !dbg !210
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.4.temp), !dbg !210
// CHECK:STDOUT:   call void @_ZNK13ConvertToEnumcvN1C1EEEv.carbon_thunk._(ptr %other, ptr %.4.temp), !dbg !210
// CHECK:STDOUT:   %.8 = load i16, ptr %.4.temp, align 2, !dbg !210
// CHECK:STDOUT:   %OrderedWith.WithSelf.NotEqual.call = icmp ne i16 %self, %.8, !dbg !210
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.dd9c146f84d40ff5(ptr %.4.temp), !dbg !210
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.NotEqual.call, !dbg !210
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCallCompareGeneric2.Main(i32 %a, ptr %b) #0 !dbg !212 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %CompareGeneric.call = call i1 @_CCompareGeneric.Main.1619d5a761aa4053(i32 %a, ptr %b), !dbg !221
// CHECK:STDOUT:   ret i1 %CompareGeneric.call, !dbg !222
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CEqual:thunk:EqWith.5bf19682558af6f0.Core:OrderedWith.Core"(i32 %self, ptr %other) #2 !dbg !224 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %__carbon_thunk.call = call i32 @_ZNK13ConvertToEnumcv2E2Ev.carbon_thunk._(ptr %other), !dbg !227
// CHECK:STDOUT:   %OrderedWith.WithSelf.Equal.call = icmp eq i32 %self, %__carbon_thunk.call, !dbg !227
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.Equal.call, !dbg !227
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CNotEqual:thunk:EqWith.5bf19682558af6f0.Core:OrderedWith.Core"(i32 %self, ptr %other) #2 !dbg !229 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %__carbon_thunk.call = call i32 @_ZNK13ConvertToEnumcv2E2Ev.carbon_thunk._(ptr %other), !dbg !232
// CHECK:STDOUT:   %OrderedWith.WithSelf.NotEqual.call = icmp ne i32 %self, %__carbon_thunk.call, !dbg !232
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.NotEqual.call, !dbg !232
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCallCompareGeneric3.Main(i16 %a, ptr %b) #0 !dbg !234 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %CompareGeneric.call = call i1 @_CCompareGeneric.Main.f7138dbf4790e4b2(i16 %a, ptr %b), !dbg !243
// CHECK:STDOUT:   ret i1 %CompareGeneric.call, !dbg !244
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CEqual:thunk:EqWith.c237d0a9ef4b3a29.Core:OrderedWith.Core"(i16 %self, ptr %other) #2 !dbg !246 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.4.temp = alloca i16, align 2, !dbg !249
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.4.temp), !dbg !249
// CHECK:STDOUT:   call void @_ZNK14ConvertToEnum2cvN1C1EEEv.carbon_thunk._(ptr %other, ptr %.4.temp), !dbg !249
// CHECK:STDOUT:   %.8 = load i16, ptr %.4.temp, align 2, !dbg !249
// CHECK:STDOUT:   %OrderedWith.WithSelf.Equal.call = icmp eq i16 %self, %.8, !dbg !249
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.dd9c146f84d40ff5(ptr %.4.temp), !dbg !249
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.Equal.call, !dbg !249
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CNotEqual:thunk:EqWith.c237d0a9ef4b3a29.Core:OrderedWith.Core"(i16 %self, ptr %other) #2 !dbg !251 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.4.temp = alloca i16, align 2, !dbg !254
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.4.temp), !dbg !254
// CHECK:STDOUT:   call void @_ZNK14ConvertToEnum2cvN1C1EEEv.carbon_thunk._(ptr %other, ptr %.4.temp), !dbg !254
// CHECK:STDOUT:   %.8 = load i16, ptr %.4.temp, align 2, !dbg !254
// CHECK:STDOUT:   %OrderedWith.WithSelf.NotEqual.call = icmp ne i16 %self, %.8, !dbg !254
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.dd9c146f84d40ff5(ptr %.4.temp), !dbg !254
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.NotEqual.call, !dbg !254
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.dd9c146f84d40ff5(ptr %t) #0 !dbg !256 {
// CHECK:STDOUT:   ret void, !dbg !261
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CCompareGeneric.Main.9f771e4a50bafdbe(i16 %x, ptr %y) #0 !dbg !263 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.call = call i1 @"_CEqual:thunk:EqWith.289080fd154ab9ff.Core:OrderedWith.Core"(i16 %x, ptr %y), !dbg !270
// CHECK:STDOUT:   ret i1 %EqWith.WithSelf.Equal.call, !dbg !272
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CCompareGeneric.Main.1619d5a761aa4053(i32 %x, ptr %y) #0 !dbg !274 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.call = call i1 @"_CEqual:thunk:EqWith.5bf19682558af6f0.Core:OrderedWith.Core"(i32 %x, ptr %y), !dbg !281
// CHECK:STDOUT:   ret i1 %EqWith.WithSelf.Equal.call, !dbg !283
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CCompareGeneric.Main.f7138dbf4790e4b2(i16 %x, ptr %y) #0 !dbg !285 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.call = call i1 @"_CEqual:thunk:EqWith.c237d0a9ef4b3a29.Core:OrderedWith.Core"(i16 %x, ptr %y), !dbg !292
// CHECK:STDOUT:   ret i1 %EqWith.WithSelf.Equal.call, !dbg !294
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef i16 @_ZNK13ConvertToEnumcvN1C1EEEv(ptr noundef nonnull align 1 dereferenceable(1)) #4
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef i32 @_ZNK13ConvertToEnumcv2E2Ev(ptr noundef nonnull align 1 dereferenceable(1)) #4
// CHECK:STDOUT:
// CHECK:STDOUT: declare noundef i16 @_ZNK14ConvertToEnum2cvN1C1EEEv(ptr noundef nonnull align 1 dereferenceable(1)) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.dd9c146f84d40ff5, { 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #4 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!148}
// CHECK:STDOUT: !llvm.module.flags = !{!44, !45, !46, !47, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!50}
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
// CHECK:STDOUT: !44 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !45 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !46 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !47 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !48 = !{!"int", !39, i64 0}
// CHECK:STDOUT: !49 = !{!"__libc_errno", !48, i64 0}
// CHECK:STDOUT: !50 = !{!49, !48, i64 0}
// CHECK:STDOUT: !79 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !147 = !DIFile(filename: "compare_class.carbon", directory: "")
// CHECK:STDOUT: !148 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !147, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !149 = !{!9, !6, !9}
// CHECK:STDOUT: !150 = !DISubroutineType(types: !149)
// CHECK:STDOUT: !153 = !{!9, !79, !9}
// CHECK:STDOUT: !154 = !DISubroutineType(types: !153)
// CHECK:STDOUT: !157 = distinct !DISubprogram(name: "Compare", linkageName: "_CCompare.Main", scope: null, file: !147, line: 17, type: !150, spFlags: DISPFlagDefinition, unit: !148, retainedNodes: !167)
// CHECK:STDOUT: !158 = !DILocalVariable(arg: 1, scope: !157, type: !6)
// CHECK:STDOUT: !159 = !DILocalVariable(arg: 2, scope: !157, type: !9)
// CHECK:STDOUT: !164 = !DILocation(line: 18, column: 10, scope: !157)
// CHECK:STDOUT: !165 = !DILocation(line: 18, column: 15, scope: !157)
// CHECK:STDOUT: !166 = !DILocation(line: 18, column: 3, scope: !157)
// CHECK:STDOUT: !167 = !{!158, !159}
// CHECK:STDOUT: !168 = distinct !DISubprogram(name: "Compare2", linkageName: "_CCompare2.Main", scope: null, file: !147, line: 21, type: !154, spFlags: DISPFlagDefinition, unit: !148, retainedNodes: !178)
// CHECK:STDOUT: !169 = !DILocalVariable(arg: 1, scope: !168, type: !79)
// CHECK:STDOUT: !170 = !DILocalVariable(arg: 2, scope: !168, type: !9)
// CHECK:STDOUT: !175 = !DILocation(line: 22, column: 10, scope: !168)
// CHECK:STDOUT: !176 = !DILocation(line: 22, column: 15, scope: !168)
// CHECK:STDOUT: !177 = !DILocation(line: 22, column: 3, scope: !168)
// CHECK:STDOUT: !178 = !{!169, !170}
// CHECK:STDOUT: !179 = distinct !DISubprogram(name: "Compare3", linkageName: "_CCompare3.Main", scope: null, file: !147, line: 25, type: !150, spFlags: DISPFlagDefinition, unit: !148, retainedNodes: !189)
// CHECK:STDOUT: !180 = !DILocalVariable(arg: 1, scope: !179, type: !6)
// CHECK:STDOUT: !181 = !DILocalVariable(arg: 2, scope: !179, type: !9)
// CHECK:STDOUT: !186 = !DILocation(line: 26, column: 10, scope: !179)
// CHECK:STDOUT: !187 = !DILocation(line: 26, column: 15, scope: !179)
// CHECK:STDOUT: !188 = !DILocation(line: 26, column: 3, scope: !179)
// CHECK:STDOUT: !189 = !{!180, !181}
// CHECK:STDOUT: !190 = distinct !DISubprogram(name: "CallCompareGeneric", linkageName: "_CCallCompareGeneric.Main", scope: null, file: !147, line: 35, type: !150, spFlags: DISPFlagDefinition, unit: !148, retainedNodes: !201)
// CHECK:STDOUT: !191 = !DILocalVariable(arg: 1, scope: !190, type: !6)
// CHECK:STDOUT: !192 = !DILocalVariable(arg: 2, scope: !190, type: !9)
// CHECK:STDOUT: !199 = !DILocation(line: 36, column: 10, scope: !190)
// CHECK:STDOUT: !200 = !DILocation(line: 36, column: 3, scope: !190)
// CHECK:STDOUT: !201 = !{!191, !192}
// CHECK:STDOUT: !202 = distinct !DISubprogram(name: "Equal", linkageName: "_CEqual:thunk:EqWith.289080fd154ab9ff.Core:OrderedWith.Core", scope: null, file: !147, type: !150, spFlags: DISPFlagDefinition, unit: !148, retainedNodes: !206)
// CHECK:STDOUT: !203 = !DILocalVariable(arg: 1, scope: !202, type: !6)
// CHECK:STDOUT: !204 = !DILocalVariable(arg: 2, scope: !202, type: !9)
// CHECK:STDOUT: !205 = !DILocation(line: 0, scope: !202)
// CHECK:STDOUT: !206 = !{!203, !204}
// CHECK:STDOUT: !207 = distinct !DISubprogram(name: "NotEqual", linkageName: "_CNotEqual:thunk:EqWith.289080fd154ab9ff.Core:OrderedWith.Core", scope: null, file: !147, type: !150, spFlags: DISPFlagDefinition, unit: !148, retainedNodes: !211)
// CHECK:STDOUT: !208 = !DILocalVariable(arg: 1, scope: !207, type: !6)
// CHECK:STDOUT: !209 = !DILocalVariable(arg: 2, scope: !207, type: !9)
// CHECK:STDOUT: !210 = !DILocation(line: 0, scope: !207)
// CHECK:STDOUT: !211 = !{!208, !209}
// CHECK:STDOUT: !212 = distinct !DISubprogram(name: "CallCompareGeneric2", linkageName: "_CCallCompareGeneric2.Main", scope: null, file: !147, line: 39, type: !154, spFlags: DISPFlagDefinition, unit: !148, retainedNodes: !223)
// CHECK:STDOUT: !213 = !DILocalVariable(arg: 1, scope: !212, type: !79)
// CHECK:STDOUT: !214 = !DILocalVariable(arg: 2, scope: !212, type: !9)
// CHECK:STDOUT: !221 = !DILocation(line: 40, column: 10, scope: !212)
// CHECK:STDOUT: !222 = !DILocation(line: 40, column: 3, scope: !212)
// CHECK:STDOUT: !223 = !{!213, !214}
// CHECK:STDOUT: !224 = distinct !DISubprogram(name: "Equal", linkageName: "_CEqual:thunk:EqWith.5bf19682558af6f0.Core:OrderedWith.Core", scope: null, file: !147, type: !154, spFlags: DISPFlagDefinition, unit: !148, retainedNodes: !228)
// CHECK:STDOUT: !225 = !DILocalVariable(arg: 1, scope: !224, type: !79)
// CHECK:STDOUT: !226 = !DILocalVariable(arg: 2, scope: !224, type: !9)
// CHECK:STDOUT: !227 = !DILocation(line: 0, scope: !224)
// CHECK:STDOUT: !228 = !{!225, !226}
// CHECK:STDOUT: !229 = distinct !DISubprogram(name: "NotEqual", linkageName: "_CNotEqual:thunk:EqWith.5bf19682558af6f0.Core:OrderedWith.Core", scope: null, file: !147, type: !154, spFlags: DISPFlagDefinition, unit: !148, retainedNodes: !233)
// CHECK:STDOUT: !230 = !DILocalVariable(arg: 1, scope: !229, type: !79)
// CHECK:STDOUT: !231 = !DILocalVariable(arg: 2, scope: !229, type: !9)
// CHECK:STDOUT: !232 = !DILocation(line: 0, scope: !229)
// CHECK:STDOUT: !233 = !{!230, !231}
// CHECK:STDOUT: !234 = distinct !DISubprogram(name: "CallCompareGeneric3", linkageName: "_CCallCompareGeneric3.Main", scope: null, file: !147, line: 43, type: !150, spFlags: DISPFlagDefinition, unit: !148, retainedNodes: !245)
// CHECK:STDOUT: !235 = !DILocalVariable(arg: 1, scope: !234, type: !6)
// CHECK:STDOUT: !236 = !DILocalVariable(arg: 2, scope: !234, type: !9)
// CHECK:STDOUT: !243 = !DILocation(line: 44, column: 10, scope: !234)
// CHECK:STDOUT: !244 = !DILocation(line: 44, column: 3, scope: !234)
// CHECK:STDOUT: !245 = !{!235, !236}
// CHECK:STDOUT: !246 = distinct !DISubprogram(name: "Equal", linkageName: "_CEqual:thunk:EqWith.c237d0a9ef4b3a29.Core:OrderedWith.Core", scope: null, file: !147, type: !150, spFlags: DISPFlagDefinition, unit: !148, retainedNodes: !250)
// CHECK:STDOUT: !247 = !DILocalVariable(arg: 1, scope: !246, type: !6)
// CHECK:STDOUT: !248 = !DILocalVariable(arg: 2, scope: !246, type: !9)
// CHECK:STDOUT: !249 = !DILocation(line: 0, scope: !246)
// CHECK:STDOUT: !250 = !{!247, !248}
// CHECK:STDOUT: !251 = distinct !DISubprogram(name: "NotEqual", linkageName: "_CNotEqual:thunk:EqWith.c237d0a9ef4b3a29.Core:OrderedWith.Core", scope: null, file: !147, type: !150, spFlags: DISPFlagDefinition, unit: !148, retainedNodes: !255)
// CHECK:STDOUT: !252 = !DILocalVariable(arg: 1, scope: !251, type: !6)
// CHECK:STDOUT: !253 = !DILocalVariable(arg: 2, scope: !251, type: !9)
// CHECK:STDOUT: !254 = !DILocation(line: 0, scope: !251)
// CHECK:STDOUT: !255 = !{!252, !253}
// CHECK:STDOUT: !256 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.dd9c146f84d40ff5", scope: null, file: !30, line: 36, type: !16, spFlags: DISPFlagDefinition, unit: !148, retainedNodes: !262)
// CHECK:STDOUT: !257 = !DILocalVariable(arg: 1, scope: !256, type: !6)
// CHECK:STDOUT: !261 = !DILocation(line: 36, column: 1, scope: !256)
// CHECK:STDOUT: !262 = !{!257}
// CHECK:STDOUT: !263 = distinct !DISubprogram(name: "CompareGeneric", linkageName: "_CCompareGeneric.Main.9f771e4a50bafdbe", scope: null, file: !147, line: 29, type: !150, spFlags: DISPFlagDefinition, unit: !148, retainedNodes: !273)
// CHECK:STDOUT: !264 = !DILocalVariable(arg: 1, scope: !263, type: !6)
// CHECK:STDOUT: !265 = !DILocalVariable(arg: 2, scope: !263, type: !9)
// CHECK:STDOUT: !270 = !DILocation(line: 30, column: 10, scope: !263)
// CHECK:STDOUT: !272 = !DILocation(line: 30, column: 3, scope: !263)
// CHECK:STDOUT: !273 = !{!264, !265}
// CHECK:STDOUT: !274 = distinct !DISubprogram(name: "CompareGeneric", linkageName: "_CCompareGeneric.Main.1619d5a761aa4053", scope: null, file: !147, line: 29, type: !154, spFlags: DISPFlagDefinition, unit: !148, retainedNodes: !284)
// CHECK:STDOUT: !275 = !DILocalVariable(arg: 1, scope: !274, type: !79)
// CHECK:STDOUT: !276 = !DILocalVariable(arg: 2, scope: !274, type: !9)
// CHECK:STDOUT: !281 = !DILocation(line: 30, column: 10, scope: !274)
// CHECK:STDOUT: !283 = !DILocation(line: 30, column: 3, scope: !274)
// CHECK:STDOUT: !284 = !{!275, !276}
// CHECK:STDOUT: !285 = distinct !DISubprogram(name: "CompareGeneric", linkageName: "_CCompareGeneric.Main.f7138dbf4790e4b2", scope: null, file: !147, line: 29, type: !150, spFlags: DISPFlagDefinition, unit: !148, retainedNodes: !295)
// CHECK:STDOUT: !286 = !DILocalVariable(arg: 1, scope: !285, type: !6)
// CHECK:STDOUT: !287 = !DILocalVariable(arg: 2, scope: !285, type: !9)
// CHECK:STDOUT: !292 = !DILocation(line: 30, column: 10, scope: !285)
// CHECK:STDOUT: !294 = !DILocation(line: 30, column: 3, scope: !285)
// CHECK:STDOUT: !295 = !{!286, !287}
// CHECK:STDOUT: !296 = !{!"p1 _ZTS13ConvertToEnum", !40, i64 0}
// CHECK:STDOUT: !297 = !{!296, !296, i64 0}
// CHECK:STDOUT: !299 = !{}
// CHECK:STDOUT: !300 = !{!"p1 _ZTS14ConvertToEnum2", !40, i64 0}
// CHECK:STDOUT: !301 = !{!300, !300, i64 0}
// CHECK:STDOUT:
