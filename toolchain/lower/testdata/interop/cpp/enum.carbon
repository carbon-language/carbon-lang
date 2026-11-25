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
// CHECK:STDOUT: declare void @_ZN1C1FENS_1EE(i16 noundef) #3
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
// CHECK:STDOUT: !llvm.module.flags = !{!34, !35, !36, !37, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!40}
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
// CHECK:STDOUT: !34 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !35 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !36 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !37 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !38 = !{!"int", !29, i64 0}
// CHECK:STDOUT: !39 = !{!"__libc_errno", !38, i64 0}
// CHECK:STDOUT: !40 = !{!39, !38, i64 0}
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
// CHECK:STDOUT: define void @_CPassEnum.Main() #0 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CTakeI16.Main(i16 7), !dbg !46
// CHECK:STDOUT:   ret void, !dbg !47
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertEnumToI16.Main(i16 %e) #0 !dbg !48 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CTakeI16.Main(i16 %e), !dbg !52
// CHECK:STDOUT:   ret void, !dbg !53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertI16ToEnum.Main(i16 %n) #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17_13.3.temp = alloca i16, align 2, !dbg !58
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc17_13.3.temp), !dbg !58
// CHECK:STDOUT:   store i16 %n, ptr %.loc17_13.3.temp, align 2, !dbg !58
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr %.loc17_13.3.temp), !dbg !59
// CHECK:STDOUT:   ret void, !dbg !60
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
// CHECK:STDOUT: declare void @_ZN1C1FENS_1EE(i16 noundef) #3
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!42}
// CHECK:STDOUT: !llvm.module.flags = !{!34, !35, !36, !37, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!40}
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
// CHECK:STDOUT: !34 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !35 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !36 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !37 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !38 = !{!"int", !29, i64 0}
// CHECK:STDOUT: !39 = !{!"__libc_errno", !38, i64 0}
// CHECK:STDOUT: !40 = !{!39, !38, i64 0}
// CHECK:STDOUT: !41 = !DIFile(filename: "convert.carbon", directory: "")
// CHECK:STDOUT: !42 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !41, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !43 = !{null, !6}
// CHECK:STDOUT: !44 = !DISubroutineType(types: !43)
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "PassEnum", linkageName: "_CPassEnum.Main", scope: null, file: !41, line: 8, type: !8, spFlags: DISPFlagDefinition, unit: !42)
// CHECK:STDOUT: !46 = !DILocation(line: 9, column: 3, scope: !45)
// CHECK:STDOUT: !47 = !DILocation(line: 8, column: 1, scope: !45)
// CHECK:STDOUT: !48 = distinct !DISubprogram(name: "ConvertEnumToI16", linkageName: "_CConvertEnumToI16.Main", scope: null, file: !41, line: 12, type: !44, spFlags: DISPFlagDefinition, unit: !42, retainedNodes: !54)
// CHECK:STDOUT: !49 = !DILocalVariable(arg: 1, scope: !48, type: !6)
// CHECK:STDOUT: !52 = !DILocation(line: 13, column: 3, scope: !48)
// CHECK:STDOUT: !53 = !DILocation(line: 12, column: 1, scope: !48)
// CHECK:STDOUT: !54 = !{!49}
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "ConvertI16ToEnum", linkageName: "_CConvertI16ToEnum.Main", scope: null, file: !41, line: 16, type: !44, spFlags: DISPFlagDefinition, unit: !42, retainedNodes: !61)
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 1, scope: !55, type: !6)
// CHECK:STDOUT: !58 = !DILocation(line: 17, column: 11, scope: !55)
// CHECK:STDOUT: !59 = !DILocation(line: 17, column: 3, scope: !55)
// CHECK:STDOUT: !60 = !DILocation(line: 16, column: 1, scope: !55)
// CHECK:STDOUT: !61 = !{!56}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'use_bitmask.carbon'
// CHECK:STDOUT: source_filename = "use_bitmask.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCompileTime.Main() #0 !dbg !69 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 0), !dbg !70
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 5), !dbg !71
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 5), !dbg !72
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 -2), !dbg !73
// CHECK:STDOUT:   ret void, !dbg !74
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z4Take4Bits(i32 noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CRuntime.Main(i32 %a, i32 %b) #0 !dbg !75 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.call = and i32 %a, %b, !dbg !80
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 %BitAndWith.WithSelf.Op.call), !dbg !82
// CHECK:STDOUT:   %BitOrWith.WithSelf.Op.call = or i32 %a, %b, !dbg !83
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 %BitOrWith.WithSelf.Op.call), !dbg !85
// CHECK:STDOUT:   %BitXorWith.WithSelf.Op.call = xor i32 %a, %b, !dbg !86
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 %BitXorWith.WithSelf.Op.call), !dbg !88
// CHECK:STDOUT:   %BitComplement.WithSelf.Op.call = xor i32 -1, %a, !dbg !90
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 %BitComplement.WithSelf.Op.call), !dbg !91
// CHECK:STDOUT:   ret void, !dbg !92
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_Z4Take4Bits, { 0, 1, 2, 3, 7, 6, 5, 4 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!63}
// CHECK:STDOUT: !llvm.module.flags = !{!34, !35, !36, !37, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!40}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !{null}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !28 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !29 = !{!"omnipotent char", !28, i64 0}
// CHECK:STDOUT: !34 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !35 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !36 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !37 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !38 = !{!"int", !29, i64 0}
// CHECK:STDOUT: !39 = !{!"__libc_errno", !38, i64 0}
// CHECK:STDOUT: !40 = !{!39, !38, i64 0}
// CHECK:STDOUT: !62 = !DIFile(filename: "use_bitmask.carbon", directory: "")
// CHECK:STDOUT: !63 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !62, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !64 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !67 = !{null, !64, !64}
// CHECK:STDOUT: !68 = !DISubroutineType(types: !67)
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "CompileTime", linkageName: "_CCompileTime.Main", scope: null, file: !62, line: 6, type: !8, spFlags: DISPFlagDefinition, unit: !63)
// CHECK:STDOUT: !70 = !DILocation(line: 7, column: 3, scope: !69)
// CHECK:STDOUT: !71 = !DILocation(line: 8, column: 3, scope: !69)
// CHECK:STDOUT: !72 = !DILocation(line: 9, column: 3, scope: !69)
// CHECK:STDOUT: !73 = !DILocation(line: 10, column: 3, scope: !69)
// CHECK:STDOUT: !74 = !DILocation(line: 6, column: 1, scope: !69)
// CHECK:STDOUT: !75 = distinct !DISubprogram(name: "Runtime", linkageName: "_CRuntime.Main", scope: null, file: !62, line: 13, type: !68, spFlags: DISPFlagDefinition, unit: !63, retainedNodes: !93)
// CHECK:STDOUT: !76 = !DILocalVariable(arg: 1, scope: !75, type: !64)
// CHECK:STDOUT: !77 = !DILocalVariable(arg: 2, scope: !75, type: !64)
// CHECK:STDOUT: !80 = !DILocation(line: 14, column: 12, scope: !75)
// CHECK:STDOUT: !82 = !DILocation(line: 14, column: 3, scope: !75)
// CHECK:STDOUT: !83 = !DILocation(line: 15, column: 12, scope: !75)
// CHECK:STDOUT: !85 = !DILocation(line: 15, column: 3, scope: !75)
// CHECK:STDOUT: !86 = !DILocation(line: 16, column: 12, scope: !75)
// CHECK:STDOUT: !88 = !DILocation(line: 16, column: 3, scope: !75)
// CHECK:STDOUT: !90 = !DILocation(line: 20, column: 12, scope: !75)
// CHECK:STDOUT: !91 = !DILocation(line: 20, column: 3, scope: !75)
// CHECK:STDOUT: !92 = !DILocation(line: 13, column: 1, scope: !75)
// CHECK:STDOUT: !93 = !{!76, !77}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'compare.carbon'
// CHECK:STDOUT: source_filename = "compare.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCompare.Main(i16 %a, i16 %b) #0 !dbg !98 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %OrderedWith.WithSelf.Equal.call = icmp eq i16 %a, %b, !dbg !105
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.Equal.call, !dbg !107
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCallCompareGeneric.Main(i16 %a, i16 %b) #0 !dbg !109 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %CompareGeneric.call = call i1 @_CCompareGeneric.Main.04aebd1c4efdc136(i16 %a, i16 %b), !dbg !118
// CHECK:STDOUT:   ret i1 %CompareGeneric.call, !dbg !119
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CCompareGeneric.Main.04aebd1c4efdc136(i16 %x, i16 %y) #0 !dbg !121 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.call = icmp eq i16 %x, %y, !dbg !128
// CHECK:STDOUT:   ret i1 %EqWith.WithSelf.Equal.call, !dbg !130
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!95}
// CHECK:STDOUT: !llvm.module.flags = !{!34, !35, !36, !37, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!40}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !28 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !29 = !{!"omnipotent char", !28, i64 0}
// CHECK:STDOUT: !34 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !35 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !36 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !37 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !38 = !{!"int", !29, i64 0}
// CHECK:STDOUT: !39 = !{!"__libc_errno", !38, i64 0}
// CHECK:STDOUT: !40 = !{!39, !38, i64 0}
// CHECK:STDOUT: !94 = !DIFile(filename: "compare.carbon", directory: "")
// CHECK:STDOUT: !95 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !94, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !96 = !{!9, !6, !6}
// CHECK:STDOUT: !97 = !DISubroutineType(types: !96)
// CHECK:STDOUT: !98 = distinct !DISubprogram(name: "Compare", linkageName: "_CCompare.Main", scope: null, file: !94, line: 6, type: !97, spFlags: DISPFlagDefinition, unit: !95, retainedNodes: !108)
// CHECK:STDOUT: !99 = !DILocalVariable(arg: 1, scope: !98, type: !6)
// CHECK:STDOUT: !100 = !DILocalVariable(arg: 2, scope: !98, type: !6)
// CHECK:STDOUT: !105 = !DILocation(line: 7, column: 10, scope: !98)
// CHECK:STDOUT: !107 = !DILocation(line: 7, column: 3, scope: !98)
// CHECK:STDOUT: !108 = !{!99, !100}
// CHECK:STDOUT: !109 = distinct !DISubprogram(name: "CallCompareGeneric", linkageName: "_CCallCompareGeneric.Main", scope: null, file: !94, line: 14, type: !97, spFlags: DISPFlagDefinition, unit: !95, retainedNodes: !120)
// CHECK:STDOUT: !110 = !DILocalVariable(arg: 1, scope: !109, type: !6)
// CHECK:STDOUT: !111 = !DILocalVariable(arg: 2, scope: !109, type: !6)
// CHECK:STDOUT: !118 = !DILocation(line: 15, column: 10, scope: !109)
// CHECK:STDOUT: !119 = !DILocation(line: 15, column: 3, scope: !109)
// CHECK:STDOUT: !120 = !{!110, !111}
// CHECK:STDOUT: !121 = distinct !DISubprogram(name: "CompareGeneric", linkageName: "_CCompareGeneric.Main.04aebd1c4efdc136", scope: null, file: !94, line: 10, type: !97, spFlags: DISPFlagDefinition, unit: !95, retainedNodes: !131)
// CHECK:STDOUT: !122 = !DILocalVariable(arg: 1, scope: !121, type: !6)
// CHECK:STDOUT: !123 = !DILocalVariable(arg: 2, scope: !121, type: !6)
// CHECK:STDOUT: !128 = !DILocation(line: 11, column: 10, scope: !121)
// CHECK:STDOUT: !130 = !DILocation(line: 11, column: 3, scope: !121)
// CHECK:STDOUT: !131 = !{!122, !123}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'compare_class.carbon'
// CHECK:STDOUT: source_filename = "compare_class.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCompare.Main(i16 %a, ptr %b) #0 !dbg !142 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_15.1.temp = alloca i16, align 2, !dbg !150
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_15.1.temp), !dbg !150
// CHECK:STDOUT:   call void @_ZNK13ConvertToEnumcvN1C1EEEv.carbon_thunk._(ptr %b, ptr %.loc18_15.1.temp), !dbg !150
// CHECK:STDOUT:   %.loc18_15.5 = load i16, ptr %.loc18_15.1.temp, align 2, !dbg !150
// CHECK:STDOUT:   %OrderedWith.WithSelf.Equal.call = icmp eq i16 %a, %.loc18_15.5, !dbg !149
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.Equal.call, !dbg !151
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZNK13ConvertToEnumcvN1C1EEEv.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !275
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !31
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !31
// CHECK:STDOUT:   %1 = load ptr, ptr %this.addr, align 8, !tbaa !275, !nonnull !277
// CHECK:STDOUT:   %call = call noundef i16 @_ZNK13ConvertToEnumcvN1C1EEEv(ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   store i16 %call, ptr %0, align 2, !tbaa !33
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCompare2.Main(i32 %a, ptr %b) #0 !dbg !153 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %__carbon_thunk.call = call i32 @_ZNK13ConvertToEnumcv2E2Ev.carbon_thunk._(ptr %b), !dbg !161
// CHECK:STDOUT:   %OrderedWith.WithSelf.Equal.call = icmp eq i32 %a, %__carbon_thunk.call, !dbg !160
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.Equal.call, !dbg !162
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal noundef i32 @_ZNK13ConvertToEnumcv2E2Ev.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !275
// CHECK:STDOUT:   %0 = load ptr, ptr %this.addr, align 8, !tbaa !275, !nonnull !277
// CHECK:STDOUT:   %call = call noundef i32 @_ZNK13ConvertToEnumcv2E2Ev(ptr noundef nonnull align 1 dereferenceable(1) %0)
// CHECK:STDOUT:   ret i32 %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCompare3.Main(i16 %a, ptr %b) #0 !dbg !164 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc26_15.1.temp = alloca i16, align 2, !dbg !172
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc26_15.1.temp), !dbg !172
// CHECK:STDOUT:   call void @_ZNK14ConvertToEnum2cvN1C1EEEv.carbon_thunk._(ptr %b, ptr %.loc26_15.1.temp), !dbg !172
// CHECK:STDOUT:   %.loc26_15.5 = load i16, ptr %.loc26_15.1.temp, align 2, !dbg !172
// CHECK:STDOUT:   %OrderedWith.WithSelf.Equal.call = icmp eq i16 %a, %.loc26_15.5, !dbg !171
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.Equal.call, !dbg !173
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZNK14ConvertToEnum2cvN1C1EEEv.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !279
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !31
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !31
// CHECK:STDOUT:   %1 = load ptr, ptr %this.addr, align 8, !tbaa !279, !nonnull !277
// CHECK:STDOUT:   %call = call noundef i16 @_ZNK14ConvertToEnum2cvN1C1EEEv(ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   store i16 %call, ptr %0, align 2, !tbaa !33
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCallCompareGeneric.Main(i16 %a, ptr %b) #0 !dbg !175 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %CompareGeneric.call = call i1 @_CCompareGeneric.Main.9f771e4a50bafdbe(i16 %a, ptr %b), !dbg !184
// CHECK:STDOUT:   ret i1 %CompareGeneric.call, !dbg !185
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CEqual:thunk:EqWith.289080fd154ab9ff.Core:OrderedWith.Core"(i16 %self, ptr %other) #2 !dbg !187 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.4.temp = alloca i16, align 2, !dbg !190
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.4.temp), !dbg !190
// CHECK:STDOUT:   call void @_ZNK13ConvertToEnumcvN1C1EEEv.carbon_thunk._(ptr %other, ptr %.4.temp), !dbg !190
// CHECK:STDOUT:   %.8 = load i16, ptr %.4.temp, align 2, !dbg !190
// CHECK:STDOUT:   %OrderedWith.WithSelf.Equal.call = icmp eq i16 %self, %.8, !dbg !190
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.Equal.call, !dbg !190
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CNotEqual:thunk:EqWith.289080fd154ab9ff.Core:OrderedWith.Core"(i16 %self, ptr %other) #2 !dbg !192 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.4.temp = alloca i16, align 2, !dbg !195
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.4.temp), !dbg !195
// CHECK:STDOUT:   call void @_ZNK13ConvertToEnumcvN1C1EEEv.carbon_thunk._(ptr %other, ptr %.4.temp), !dbg !195
// CHECK:STDOUT:   %.8 = load i16, ptr %.4.temp, align 2, !dbg !195
// CHECK:STDOUT:   %OrderedWith.WithSelf.NotEqual.call = icmp ne i16 %self, %.8, !dbg !195
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.NotEqual.call, !dbg !195
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCallCompareGeneric2.Main(i32 %a, ptr %b) #0 !dbg !197 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %CompareGeneric.call = call i1 @_CCompareGeneric.Main.1619d5a761aa4053(i32 %a, ptr %b), !dbg !206
// CHECK:STDOUT:   ret i1 %CompareGeneric.call, !dbg !207
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CEqual:thunk:EqWith.5bf19682558af6f0.Core:OrderedWith.Core"(i32 %self, ptr %other) #2 !dbg !209 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %__carbon_thunk.call = call i32 @_ZNK13ConvertToEnumcv2E2Ev.carbon_thunk._(ptr %other), !dbg !212
// CHECK:STDOUT:   %OrderedWith.WithSelf.Equal.call = icmp eq i32 %self, %__carbon_thunk.call, !dbg !212
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.Equal.call, !dbg !212
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CNotEqual:thunk:EqWith.5bf19682558af6f0.Core:OrderedWith.Core"(i32 %self, ptr %other) #2 !dbg !214 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %__carbon_thunk.call = call i32 @_ZNK13ConvertToEnumcv2E2Ev.carbon_thunk._(ptr %other), !dbg !217
// CHECK:STDOUT:   %OrderedWith.WithSelf.NotEqual.call = icmp ne i32 %self, %__carbon_thunk.call, !dbg !217
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.NotEqual.call, !dbg !217
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCallCompareGeneric3.Main(i16 %a, ptr %b) #0 !dbg !219 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %CompareGeneric.call = call i1 @_CCompareGeneric.Main.f7138dbf4790e4b2(i16 %a, ptr %b), !dbg !228
// CHECK:STDOUT:   ret i1 %CompareGeneric.call, !dbg !229
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CEqual:thunk:EqWith.c237d0a9ef4b3a29.Core:OrderedWith.Core"(i16 %self, ptr %other) #2 !dbg !231 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.4.temp = alloca i16, align 2, !dbg !234
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.4.temp), !dbg !234
// CHECK:STDOUT:   call void @_ZNK14ConvertToEnum2cvN1C1EEEv.carbon_thunk._(ptr %other, ptr %.4.temp), !dbg !234
// CHECK:STDOUT:   %.8 = load i16, ptr %.4.temp, align 2, !dbg !234
// CHECK:STDOUT:   %OrderedWith.WithSelf.Equal.call = icmp eq i16 %self, %.8, !dbg !234
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.Equal.call, !dbg !234
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CNotEqual:thunk:EqWith.c237d0a9ef4b3a29.Core:OrderedWith.Core"(i16 %self, ptr %other) #2 !dbg !236 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.4.temp = alloca i16, align 2, !dbg !239
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.4.temp), !dbg !239
// CHECK:STDOUT:   call void @_ZNK14ConvertToEnum2cvN1C1EEEv.carbon_thunk._(ptr %other, ptr %.4.temp), !dbg !239
// CHECK:STDOUT:   %.8 = load i16, ptr %.4.temp, align 2, !dbg !239
// CHECK:STDOUT:   %OrderedWith.WithSelf.NotEqual.call = icmp ne i16 %self, %.8, !dbg !239
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.NotEqual.call, !dbg !239
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CCompareGeneric.Main.9f771e4a50bafdbe(i16 %x, ptr %y) #0 !dbg !241 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.call = call i1 @"_CEqual:thunk:EqWith.289080fd154ab9ff.Core:OrderedWith.Core"(i16 %x, ptr %y), !dbg !248
// CHECK:STDOUT:   ret i1 %EqWith.WithSelf.Equal.call, !dbg !250
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CCompareGeneric.Main.1619d5a761aa4053(i32 %x, ptr %y) #0 !dbg !252 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.call = call i1 @"_CEqual:thunk:EqWith.5bf19682558af6f0.Core:OrderedWith.Core"(i32 %x, ptr %y), !dbg !259
// CHECK:STDOUT:   ret i1 %EqWith.WithSelf.Equal.call, !dbg !261
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CCompareGeneric.Main.f7138dbf4790e4b2(i16 %x, ptr %y) #0 !dbg !263 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.call = call i1 @"_CEqual:thunk:EqWith.c237d0a9ef4b3a29.Core:OrderedWith.Core"(i16 %x, ptr %y), !dbg !270
// CHECK:STDOUT:   ret i1 %EqWith.WithSelf.Equal.call, !dbg !272
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
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #4 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!133}
// CHECK:STDOUT: !llvm.module.flags = !{!34, !35, !36, !37, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!40}
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
// CHECK:STDOUT: !34 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !35 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !36 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !37 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !38 = !{!"int", !29, i64 0}
// CHECK:STDOUT: !39 = !{!"__libc_errno", !38, i64 0}
// CHECK:STDOUT: !40 = !{!39, !38, i64 0}
// CHECK:STDOUT: !64 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !132 = !DIFile(filename: "compare_class.carbon", directory: "")
// CHECK:STDOUT: !133 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !132, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !134 = !{!9, !6, !9}
// CHECK:STDOUT: !135 = !DISubroutineType(types: !134)
// CHECK:STDOUT: !138 = !{!9, !64, !9}
// CHECK:STDOUT: !139 = !DISubroutineType(types: !138)
// CHECK:STDOUT: !142 = distinct !DISubprogram(name: "Compare", linkageName: "_CCompare.Main", scope: null, file: !132, line: 17, type: !135, spFlags: DISPFlagDefinition, unit: !133, retainedNodes: !152)
// CHECK:STDOUT: !143 = !DILocalVariable(arg: 1, scope: !142, type: !6)
// CHECK:STDOUT: !144 = !DILocalVariable(arg: 2, scope: !142, type: !9)
// CHECK:STDOUT: !149 = !DILocation(line: 18, column: 10, scope: !142)
// CHECK:STDOUT: !150 = !DILocation(line: 18, column: 15, scope: !142)
// CHECK:STDOUT: !151 = !DILocation(line: 18, column: 3, scope: !142)
// CHECK:STDOUT: !152 = !{!143, !144}
// CHECK:STDOUT: !153 = distinct !DISubprogram(name: "Compare2", linkageName: "_CCompare2.Main", scope: null, file: !132, line: 21, type: !139, spFlags: DISPFlagDefinition, unit: !133, retainedNodes: !163)
// CHECK:STDOUT: !154 = !DILocalVariable(arg: 1, scope: !153, type: !64)
// CHECK:STDOUT: !155 = !DILocalVariable(arg: 2, scope: !153, type: !9)
// CHECK:STDOUT: !160 = !DILocation(line: 22, column: 10, scope: !153)
// CHECK:STDOUT: !161 = !DILocation(line: 22, column: 15, scope: !153)
// CHECK:STDOUT: !162 = !DILocation(line: 22, column: 3, scope: !153)
// CHECK:STDOUT: !163 = !{!154, !155}
// CHECK:STDOUT: !164 = distinct !DISubprogram(name: "Compare3", linkageName: "_CCompare3.Main", scope: null, file: !132, line: 25, type: !135, spFlags: DISPFlagDefinition, unit: !133, retainedNodes: !174)
// CHECK:STDOUT: !165 = !DILocalVariable(arg: 1, scope: !164, type: !6)
// CHECK:STDOUT: !166 = !DILocalVariable(arg: 2, scope: !164, type: !9)
// CHECK:STDOUT: !171 = !DILocation(line: 26, column: 10, scope: !164)
// CHECK:STDOUT: !172 = !DILocation(line: 26, column: 15, scope: !164)
// CHECK:STDOUT: !173 = !DILocation(line: 26, column: 3, scope: !164)
// CHECK:STDOUT: !174 = !{!165, !166}
// CHECK:STDOUT: !175 = distinct !DISubprogram(name: "CallCompareGeneric", linkageName: "_CCallCompareGeneric.Main", scope: null, file: !132, line: 35, type: !135, spFlags: DISPFlagDefinition, unit: !133, retainedNodes: !186)
// CHECK:STDOUT: !176 = !DILocalVariable(arg: 1, scope: !175, type: !6)
// CHECK:STDOUT: !177 = !DILocalVariable(arg: 2, scope: !175, type: !9)
// CHECK:STDOUT: !184 = !DILocation(line: 36, column: 10, scope: !175)
// CHECK:STDOUT: !185 = !DILocation(line: 36, column: 3, scope: !175)
// CHECK:STDOUT: !186 = !{!176, !177}
// CHECK:STDOUT: !187 = distinct !DISubprogram(name: "Equal", linkageName: "_CEqual:thunk:EqWith.289080fd154ab9ff.Core:OrderedWith.Core", scope: null, file: !132, type: !135, spFlags: DISPFlagDefinition, unit: !133, retainedNodes: !191)
// CHECK:STDOUT: !188 = !DILocalVariable(arg: 1, scope: !187, type: !6)
// CHECK:STDOUT: !189 = !DILocalVariable(arg: 2, scope: !187, type: !9)
// CHECK:STDOUT: !190 = !DILocation(line: 0, scope: !187)
// CHECK:STDOUT: !191 = !{!188, !189}
// CHECK:STDOUT: !192 = distinct !DISubprogram(name: "NotEqual", linkageName: "_CNotEqual:thunk:EqWith.289080fd154ab9ff.Core:OrderedWith.Core", scope: null, file: !132, type: !135, spFlags: DISPFlagDefinition, unit: !133, retainedNodes: !196)
// CHECK:STDOUT: !193 = !DILocalVariable(arg: 1, scope: !192, type: !6)
// CHECK:STDOUT: !194 = !DILocalVariable(arg: 2, scope: !192, type: !9)
// CHECK:STDOUT: !195 = !DILocation(line: 0, scope: !192)
// CHECK:STDOUT: !196 = !{!193, !194}
// CHECK:STDOUT: !197 = distinct !DISubprogram(name: "CallCompareGeneric2", linkageName: "_CCallCompareGeneric2.Main", scope: null, file: !132, line: 39, type: !139, spFlags: DISPFlagDefinition, unit: !133, retainedNodes: !208)
// CHECK:STDOUT: !198 = !DILocalVariable(arg: 1, scope: !197, type: !64)
// CHECK:STDOUT: !199 = !DILocalVariable(arg: 2, scope: !197, type: !9)
// CHECK:STDOUT: !206 = !DILocation(line: 40, column: 10, scope: !197)
// CHECK:STDOUT: !207 = !DILocation(line: 40, column: 3, scope: !197)
// CHECK:STDOUT: !208 = !{!198, !199}
// CHECK:STDOUT: !209 = distinct !DISubprogram(name: "Equal", linkageName: "_CEqual:thunk:EqWith.5bf19682558af6f0.Core:OrderedWith.Core", scope: null, file: !132, type: !139, spFlags: DISPFlagDefinition, unit: !133, retainedNodes: !213)
// CHECK:STDOUT: !210 = !DILocalVariable(arg: 1, scope: !209, type: !64)
// CHECK:STDOUT: !211 = !DILocalVariable(arg: 2, scope: !209, type: !9)
// CHECK:STDOUT: !212 = !DILocation(line: 0, scope: !209)
// CHECK:STDOUT: !213 = !{!210, !211}
// CHECK:STDOUT: !214 = distinct !DISubprogram(name: "NotEqual", linkageName: "_CNotEqual:thunk:EqWith.5bf19682558af6f0.Core:OrderedWith.Core", scope: null, file: !132, type: !139, spFlags: DISPFlagDefinition, unit: !133, retainedNodes: !218)
// CHECK:STDOUT: !215 = !DILocalVariable(arg: 1, scope: !214, type: !64)
// CHECK:STDOUT: !216 = !DILocalVariable(arg: 2, scope: !214, type: !9)
// CHECK:STDOUT: !217 = !DILocation(line: 0, scope: !214)
// CHECK:STDOUT: !218 = !{!215, !216}
// CHECK:STDOUT: !219 = distinct !DISubprogram(name: "CallCompareGeneric3", linkageName: "_CCallCompareGeneric3.Main", scope: null, file: !132, line: 43, type: !135, spFlags: DISPFlagDefinition, unit: !133, retainedNodes: !230)
// CHECK:STDOUT: !220 = !DILocalVariable(arg: 1, scope: !219, type: !6)
// CHECK:STDOUT: !221 = !DILocalVariable(arg: 2, scope: !219, type: !9)
// CHECK:STDOUT: !228 = !DILocation(line: 44, column: 10, scope: !219)
// CHECK:STDOUT: !229 = !DILocation(line: 44, column: 3, scope: !219)
// CHECK:STDOUT: !230 = !{!220, !221}
// CHECK:STDOUT: !231 = distinct !DISubprogram(name: "Equal", linkageName: "_CEqual:thunk:EqWith.c237d0a9ef4b3a29.Core:OrderedWith.Core", scope: null, file: !132, type: !135, spFlags: DISPFlagDefinition, unit: !133, retainedNodes: !235)
// CHECK:STDOUT: !232 = !DILocalVariable(arg: 1, scope: !231, type: !6)
// CHECK:STDOUT: !233 = !DILocalVariable(arg: 2, scope: !231, type: !9)
// CHECK:STDOUT: !234 = !DILocation(line: 0, scope: !231)
// CHECK:STDOUT: !235 = !{!232, !233}
// CHECK:STDOUT: !236 = distinct !DISubprogram(name: "NotEqual", linkageName: "_CNotEqual:thunk:EqWith.c237d0a9ef4b3a29.Core:OrderedWith.Core", scope: null, file: !132, type: !135, spFlags: DISPFlagDefinition, unit: !133, retainedNodes: !240)
// CHECK:STDOUT: !237 = !DILocalVariable(arg: 1, scope: !236, type: !6)
// CHECK:STDOUT: !238 = !DILocalVariable(arg: 2, scope: !236, type: !9)
// CHECK:STDOUT: !239 = !DILocation(line: 0, scope: !236)
// CHECK:STDOUT: !240 = !{!237, !238}
// CHECK:STDOUT: !241 = distinct !DISubprogram(name: "CompareGeneric", linkageName: "_CCompareGeneric.Main.9f771e4a50bafdbe", scope: null, file: !132, line: 29, type: !135, spFlags: DISPFlagDefinition, unit: !133, retainedNodes: !251)
// CHECK:STDOUT: !242 = !DILocalVariable(arg: 1, scope: !241, type: !6)
// CHECK:STDOUT: !243 = !DILocalVariable(arg: 2, scope: !241, type: !9)
// CHECK:STDOUT: !248 = !DILocation(line: 30, column: 10, scope: !241)
// CHECK:STDOUT: !250 = !DILocation(line: 30, column: 3, scope: !241)
// CHECK:STDOUT: !251 = !{!242, !243}
// CHECK:STDOUT: !252 = distinct !DISubprogram(name: "CompareGeneric", linkageName: "_CCompareGeneric.Main.1619d5a761aa4053", scope: null, file: !132, line: 29, type: !139, spFlags: DISPFlagDefinition, unit: !133, retainedNodes: !262)
// CHECK:STDOUT: !253 = !DILocalVariable(arg: 1, scope: !252, type: !64)
// CHECK:STDOUT: !254 = !DILocalVariable(arg: 2, scope: !252, type: !9)
// CHECK:STDOUT: !259 = !DILocation(line: 30, column: 10, scope: !252)
// CHECK:STDOUT: !261 = !DILocation(line: 30, column: 3, scope: !252)
// CHECK:STDOUT: !262 = !{!253, !254}
// CHECK:STDOUT: !263 = distinct !DISubprogram(name: "CompareGeneric", linkageName: "_CCompareGeneric.Main.f7138dbf4790e4b2", scope: null, file: !132, line: 29, type: !135, spFlags: DISPFlagDefinition, unit: !133, retainedNodes: !273)
// CHECK:STDOUT: !264 = !DILocalVariable(arg: 1, scope: !263, type: !6)
// CHECK:STDOUT: !265 = !DILocalVariable(arg: 2, scope: !263, type: !9)
// CHECK:STDOUT: !270 = !DILocation(line: 30, column: 10, scope: !263)
// CHECK:STDOUT: !272 = !DILocation(line: 30, column: 3, scope: !263)
// CHECK:STDOUT: !273 = !{!264, !265}
// CHECK:STDOUT: !274 = !{!"p1 _ZTS13ConvertToEnum", !30, i64 0}
// CHECK:STDOUT: !275 = !{!274, !274, i64 0}
// CHECK:STDOUT: !277 = !{}
// CHECK:STDOUT: !278 = !{!"p1 _ZTS14ConvertToEnum2", !30, i64 0}
// CHECK:STDOUT: !279 = !{!278, !278, i64 0}
// CHECK:STDOUT:
