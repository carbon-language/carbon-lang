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
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.da290558fb92070b(ptr @int_0), !dbg !15
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc8_16.1.temp), !dbg !17
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr @int_0), !dbg !18
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.da290558fb92070b(ptr @int_0), !dbg !17
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc9_16.1.temp), !dbg !19
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr @int_7), !dbg !20
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.da290558fb92070b(ptr @int_7), !dbg !19
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc10_16.1.temp), !dbg !21
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr @int_8), !dbg !22
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.da290558fb92070b(ptr @int_8), !dbg !21
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc11_18.1.temp), !dbg !23
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr @int_0), !dbg !24
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.da290558fb92070b(ptr @int_0), !dbg !23
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc12_18.1.temp), !dbg !25
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr @int_7), !dbg !26
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.da290558fb92070b(ptr @int_7), !dbg !25
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc13_18.1.temp), !dbg !27
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr @int_8), !dbg !28
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.da290558fb92070b(ptr @int_8), !dbg !27
// CHECK:STDOUT:   ret void, !dbg !29
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN1C1FENS_1EE.carbon_thunk._(ptr noundef %0) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !45
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !45
// CHECK:STDOUT:   %2 = load i16, ptr %1, align 2, !tbaa !47
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE(i16 noundef signext %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:DestroyImpl.f6bdc88761165680.Core:DestroyImpl.Core"(ptr %self) #2 !dbg !30 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.2 = load i16, ptr %self, align 2, !dbg !32
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.da290558fb92070b(ptr %t) #0 !dbg !35 {
// CHECK:STDOUT:   call void @"_COp:thunk:DestroyImpl.f6bdc88761165680.Core:DestroyImpl.Core"(ptr %t), !dbg !38
// CHECK:STDOUT:   ret void, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN1C1FENS_1EE(i16 noundef signext) #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_ZN1C1FENS_1EE.carbon_thunk._, { 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.da290558fb92070b, { 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #4 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!48, !49, !50, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!53}
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
// CHECK:STDOUT: !30 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:DestroyImpl.f6bdc88761165680.Core:DestroyImpl.Core", scope: null, file: !4, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !33)
// CHECK:STDOUT: !31 = !DILocalVariable(arg: 1, scope: !30, type: !6)
// CHECK:STDOUT: !32 = !DILocation(line: 0, scope: !30)
// CHECK:STDOUT: !33 = !{!31}
// CHECK:STDOUT: !34 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.da290558fb92070b", scope: null, file: !34, line: 30, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !41)
// CHECK:STDOUT: !36 = !DILocalVariable(arg: 1, scope: !35, type: !6)
// CHECK:STDOUT: !38 = !DILocation(line: 31, column: 5, scope: !35)
// CHECK:STDOUT: !40 = !DILocation(line: 30, column: 1, scope: !35)
// CHECK:STDOUT: !41 = !{!36}
// CHECK:STDOUT: !42 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !43 = !{!"omnipotent char", !42, i64 0}
// CHECK:STDOUT: !44 = !{!"any pointer", !43, i64 0}
// CHECK:STDOUT: !45 = !{!44, !44, i64 0}
// CHECK:STDOUT: !46 = !{!"_ZTSN1C1EE", !43, i64 0}
// CHECK:STDOUT: !47 = !{!46, !46, i64 0}
// CHECK:STDOUT: !48 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !49 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !50 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !51 = !{!"int", !43, i64 0}
// CHECK:STDOUT: !52 = !{!"__libc_errno", !51, i64 0}
// CHECK:STDOUT: !53 = !{!52, !51, i64 0}
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
// CHECK:STDOUT: define void @_CPassEnum.Main() #0 !dbg !56 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CTakeI16.Main(i16 7), !dbg !57
// CHECK:STDOUT:   ret void, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertEnumToI16.Main(i16 %e) #0 !dbg !59 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CTakeI16.Main(i16 %e), !dbg !63
// CHECK:STDOUT:   ret void, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertI16ToEnum.Main(i16 %n) #0 !dbg !66 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc17_13.3.temp = alloca i16, align 2, !dbg !69
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc17_13.3.temp), !dbg !69
// CHECK:STDOUT:   store i16 %n, ptr %.loc17_13.3.temp, align 2, !dbg !69
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE.carbon_thunk._(ptr %.loc17_13.3.temp), !dbg !70
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.da290558fb92070b(ptr %.loc17_13.3.temp), !dbg !69
// CHECK:STDOUT:   ret void, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN1C1FENS_1EE.carbon_thunk._(ptr noundef %0) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !45
// CHECK:STDOUT:   %1 = load ptr, ptr %.addr, align 8, !tbaa !45
// CHECK:STDOUT:   %2 = load i16, ptr %1, align 2, !tbaa !47
// CHECK:STDOUT:   call void @_ZN1C1FENS_1EE(i16 noundef signext %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:DestroyImpl.f6bdc88761165680.Core:DestroyImpl.Core"(ptr %self) #2 !dbg !73 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.2 = load i16, ptr %self, align 2, !dbg !75
// CHECK:STDOUT:   ret void, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.da290558fb92070b(ptr %t) #0 !dbg !77 {
// CHECK:STDOUT:   call void @"_COp:thunk:DestroyImpl.f6bdc88761165680.Core:DestroyImpl.Core"(ptr %t), !dbg !80
// CHECK:STDOUT:   ret void, !dbg !82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN1C1FENS_1EE(i16 noundef signext) #4
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #4 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!55}
// CHECK:STDOUT: !llvm.module.flags = !{!48, !49, !50, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!53}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !{null}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !12 = !{null, !6}
// CHECK:STDOUT: !13 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !34 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !42 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !43 = !{!"omnipotent char", !42, i64 0}
// CHECK:STDOUT: !44 = !{!"any pointer", !43, i64 0}
// CHECK:STDOUT: !45 = !{!44, !44, i64 0}
// CHECK:STDOUT: !46 = !{!"_ZTSN1C1EE", !43, i64 0}
// CHECK:STDOUT: !47 = !{!46, !46, i64 0}
// CHECK:STDOUT: !48 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !49 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !50 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !51 = !{!"int", !43, i64 0}
// CHECK:STDOUT: !52 = !{!"__libc_errno", !51, i64 0}
// CHECK:STDOUT: !53 = !{!52, !51, i64 0}
// CHECK:STDOUT: !54 = !DIFile(filename: "convert.carbon", directory: "")
// CHECK:STDOUT: !55 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !54, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "PassEnum", linkageName: "_CPassEnum.Main", scope: null, file: !54, line: 8, type: !8, spFlags: DISPFlagDefinition, unit: !55)
// CHECK:STDOUT: !57 = !DILocation(line: 9, column: 3, scope: !56)
// CHECK:STDOUT: !58 = !DILocation(line: 8, column: 1, scope: !56)
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "ConvertEnumToI16", linkageName: "_CConvertEnumToI16.Main", scope: null, file: !54, line: 12, type: !13, spFlags: DISPFlagDefinition, unit: !55, retainedNodes: !65)
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 1, scope: !59, type: !6)
// CHECK:STDOUT: !63 = !DILocation(line: 13, column: 3, scope: !59)
// CHECK:STDOUT: !64 = !DILocation(line: 12, column: 1, scope: !59)
// CHECK:STDOUT: !65 = !{!60}
// CHECK:STDOUT: !66 = distinct !DISubprogram(name: "ConvertI16ToEnum", linkageName: "_CConvertI16ToEnum.Main", scope: null, file: !54, line: 16, type: !13, spFlags: DISPFlagDefinition, unit: !55, retainedNodes: !72)
// CHECK:STDOUT: !67 = !DILocalVariable(arg: 1, scope: !66, type: !6)
// CHECK:STDOUT: !69 = !DILocation(line: 17, column: 11, scope: !66)
// CHECK:STDOUT: !70 = !DILocation(line: 17, column: 3, scope: !66)
// CHECK:STDOUT: !71 = !DILocation(line: 16, column: 1, scope: !66)
// CHECK:STDOUT: !72 = !{!67}
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:DestroyImpl.f6bdc88761165680.Core:DestroyImpl.Core", scope: null, file: !54, type: !13, spFlags: DISPFlagDefinition, unit: !55, retainedNodes: !76)
// CHECK:STDOUT: !74 = !DILocalVariable(arg: 1, scope: !73, type: !6)
// CHECK:STDOUT: !75 = !DILocation(line: 0, scope: !73)
// CHECK:STDOUT: !76 = !{!74}
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.da290558fb92070b", scope: null, file: !34, line: 30, type: !13, spFlags: DISPFlagDefinition, unit: !55, retainedNodes: !83)
// CHECK:STDOUT: !78 = !DILocalVariable(arg: 1, scope: !77, type: !6)
// CHECK:STDOUT: !80 = !DILocation(line: 31, column: 5, scope: !77)
// CHECK:STDOUT: !82 = !DILocation(line: 30, column: 1, scope: !77)
// CHECK:STDOUT: !83 = !{!78}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'use_bitmask.carbon'
// CHECK:STDOUT: source_filename = "use_bitmask.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCompileTime.Main() #0 !dbg !91 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 0), !dbg !92
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 5), !dbg !93
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 5), !dbg !94
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 -2), !dbg !95
// CHECK:STDOUT:   ret void, !dbg !96
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z4Take4Bits(i32 noundef) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CRuntime.Main(i32 %a, i32 %b) #0 !dbg !97 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.call = and i32 %a, %b, !dbg !102
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 %BitAndWith.WithSelf.Op.call), !dbg !104
// CHECK:STDOUT:   %BitOrWith.WithSelf.Op.call = or i32 %a, %b, !dbg !105
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 %BitOrWith.WithSelf.Op.call), !dbg !107
// CHECK:STDOUT:   %BitXorWith.WithSelf.Op.call = xor i32 %a, %b, !dbg !108
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 %BitXorWith.WithSelf.Op.call), !dbg !110
// CHECK:STDOUT:   %BitComplement.WithSelf.Op.call = xor i32 -1, %a, !dbg !112
// CHECK:STDOUT:   call void @_Z4Take4Bits(i32 %BitComplement.WithSelf.Op.call), !dbg !113
// CHECK:STDOUT:   ret void, !dbg !114
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_Z4Take4Bits, { 0, 1, 2, 3, 7, 6, 5, 4 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!85}
// CHECK:STDOUT: !llvm.module.flags = !{!48, !49, !50, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!53}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !7 = !{null}
// CHECK:STDOUT: !8 = !DISubroutineType(types: !7)
// CHECK:STDOUT: !42 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !43 = !{!"omnipotent char", !42, i64 0}
// CHECK:STDOUT: !48 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !49 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !50 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !51 = !{!"int", !43, i64 0}
// CHECK:STDOUT: !52 = !{!"__libc_errno", !51, i64 0}
// CHECK:STDOUT: !53 = !{!52, !51, i64 0}
// CHECK:STDOUT: !84 = !DIFile(filename: "use_bitmask.carbon", directory: "")
// CHECK:STDOUT: !85 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !84, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !86 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !89 = !{null, !86, !86}
// CHECK:STDOUT: !90 = !DISubroutineType(types: !89)
// CHECK:STDOUT: !91 = distinct !DISubprogram(name: "CompileTime", linkageName: "_CCompileTime.Main", scope: null, file: !84, line: 6, type: !8, spFlags: DISPFlagDefinition, unit: !85)
// CHECK:STDOUT: !92 = !DILocation(line: 7, column: 3, scope: !91)
// CHECK:STDOUT: !93 = !DILocation(line: 8, column: 3, scope: !91)
// CHECK:STDOUT: !94 = !DILocation(line: 9, column: 3, scope: !91)
// CHECK:STDOUT: !95 = !DILocation(line: 10, column: 3, scope: !91)
// CHECK:STDOUT: !96 = !DILocation(line: 6, column: 1, scope: !91)
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "Runtime", linkageName: "_CRuntime.Main", scope: null, file: !84, line: 13, type: !90, spFlags: DISPFlagDefinition, unit: !85, retainedNodes: !115)
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 1, scope: !97, type: !86)
// CHECK:STDOUT: !99 = !DILocalVariable(arg: 2, scope: !97, type: !86)
// CHECK:STDOUT: !102 = !DILocation(line: 14, column: 12, scope: !97)
// CHECK:STDOUT: !104 = !DILocation(line: 14, column: 3, scope: !97)
// CHECK:STDOUT: !105 = !DILocation(line: 15, column: 12, scope: !97)
// CHECK:STDOUT: !107 = !DILocation(line: 15, column: 3, scope: !97)
// CHECK:STDOUT: !108 = !DILocation(line: 16, column: 12, scope: !97)
// CHECK:STDOUT: !110 = !DILocation(line: 16, column: 3, scope: !97)
// CHECK:STDOUT: !112 = !DILocation(line: 20, column: 12, scope: !97)
// CHECK:STDOUT: !113 = !DILocation(line: 20, column: 3, scope: !97)
// CHECK:STDOUT: !114 = !DILocation(line: 13, column: 1, scope: !97)
// CHECK:STDOUT: !115 = !{!98, !99}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'compare.carbon'
// CHECK:STDOUT: source_filename = "compare.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCompare.Main(i16 %a, i16 %b) #0 !dbg !120 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %OrderedWith.WithSelf.Equal.call = icmp eq i16 %a, %b, !dbg !127
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.Equal.call, !dbg !129
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCallCompareGeneric.Main(i16 %a, i16 %b) #0 !dbg !131 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %CompareGeneric.call = call i1 @_CCompareGeneric.Main.7602ebbc958eec83(i16 %a, i16 %b), !dbg !140
// CHECK:STDOUT:   ret i1 %CompareGeneric.call, !dbg !141
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CCompareGeneric.Main.7602ebbc958eec83(i16 %x, i16 %y) #0 !dbg !143 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.call = icmp eq i16 %x, %y, !dbg !150
// CHECK:STDOUT:   ret i1 %EqWith.WithSelf.Equal.call, !dbg !152
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!117}
// CHECK:STDOUT: !llvm.module.flags = !{!48, !49, !50, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!53}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !42 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !43 = !{!"omnipotent char", !42, i64 0}
// CHECK:STDOUT: !48 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !49 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !50 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !51 = !{!"int", !43, i64 0}
// CHECK:STDOUT: !52 = !{!"__libc_errno", !51, i64 0}
// CHECK:STDOUT: !53 = !{!52, !51, i64 0}
// CHECK:STDOUT: !116 = !DIFile(filename: "compare.carbon", directory: "")
// CHECK:STDOUT: !117 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !116, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !118 = !{!9, !6, !6}
// CHECK:STDOUT: !119 = !DISubroutineType(types: !118)
// CHECK:STDOUT: !120 = distinct !DISubprogram(name: "Compare", linkageName: "_CCompare.Main", scope: null, file: !116, line: 6, type: !119, spFlags: DISPFlagDefinition, unit: !117, retainedNodes: !130)
// CHECK:STDOUT: !121 = !DILocalVariable(arg: 1, scope: !120, type: !6)
// CHECK:STDOUT: !122 = !DILocalVariable(arg: 2, scope: !120, type: !6)
// CHECK:STDOUT: !127 = !DILocation(line: 7, column: 10, scope: !120)
// CHECK:STDOUT: !129 = !DILocation(line: 7, column: 3, scope: !120)
// CHECK:STDOUT: !130 = !{!121, !122}
// CHECK:STDOUT: !131 = distinct !DISubprogram(name: "CallCompareGeneric", linkageName: "_CCallCompareGeneric.Main", scope: null, file: !116, line: 14, type: !119, spFlags: DISPFlagDefinition, unit: !117, retainedNodes: !142)
// CHECK:STDOUT: !132 = !DILocalVariable(arg: 1, scope: !131, type: !6)
// CHECK:STDOUT: !133 = !DILocalVariable(arg: 2, scope: !131, type: !6)
// CHECK:STDOUT: !140 = !DILocation(line: 15, column: 10, scope: !131)
// CHECK:STDOUT: !141 = !DILocation(line: 15, column: 3, scope: !131)
// CHECK:STDOUT: !142 = !{!132, !133}
// CHECK:STDOUT: !143 = distinct !DISubprogram(name: "CompareGeneric", linkageName: "_CCompareGeneric.Main.7602ebbc958eec83", scope: null, file: !116, line: 10, type: !119, spFlags: DISPFlagDefinition, unit: !117, retainedNodes: !153)
// CHECK:STDOUT: !144 = !DILocalVariable(arg: 1, scope: !143, type: !6)
// CHECK:STDOUT: !145 = !DILocalVariable(arg: 2, scope: !143, type: !6)
// CHECK:STDOUT: !150 = !DILocation(line: 11, column: 10, scope: !143)
// CHECK:STDOUT: !152 = !DILocation(line: 11, column: 3, scope: !143)
// CHECK:STDOUT: !153 = !{!144, !145}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'compare_class.carbon'
// CHECK:STDOUT: source_filename = "compare_class.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCompare.Main(i16 %a, ptr %b) #0 !dbg !164 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_15.1.temp = alloca i16, align 2, !dbg !172
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_15.1.temp), !dbg !172
// CHECK:STDOUT:   call void @_ZNK13ConvertToEnumcvN1C1EEEv.carbon_thunk._(ptr %b, ptr %.loc18_15.1.temp), !dbg !172
// CHECK:STDOUT:   %.loc18_15.5 = load i16, ptr %.loc18_15.1.temp, align 2, !dbg !172
// CHECK:STDOUT:   %OrderedWith.WithSelf.Equal.call = icmp eq i16 %a, %.loc18_15.5, !dbg !171
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.da290558fb92070b(ptr %.loc18_15.1.temp), !dbg !172
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.Equal.call, !dbg !173
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZNK13ConvertToEnumcvN1C1EEEv.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !308
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !45
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !45
// CHECK:STDOUT:   %1 = load ptr, ptr %this.addr, align 8, !tbaa !308, !nonnull !310
// CHECK:STDOUT:   %call = call noundef signext i16 @_ZNK13ConvertToEnumcvN1C1EEEv(ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   store i16 %call, ptr %0, align 2, !tbaa !47
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:DestroyImpl.f6bdc88761165680.Core:DestroyImpl.Core"(ptr %self) #2 !dbg !175 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.2 = load i16, ptr %self, align 2, !dbg !177
// CHECK:STDOUT:   ret void, !dbg !177
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCompare2.Main(i32 %a, ptr %b) #0 !dbg !179 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %__carbon_thunk.call = call i32 @_ZNK13ConvertToEnumcv2E2Ev.carbon_thunk._(ptr %b), !dbg !187
// CHECK:STDOUT:   %OrderedWith.WithSelf.Equal.call = icmp eq i32 %a, %__carbon_thunk.call, !dbg !186
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.Equal.call, !dbg !188
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal noundef i32 @_ZNK13ConvertToEnumcv2E2Ev.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !308
// CHECK:STDOUT:   %0 = load ptr, ptr %this.addr, align 8, !tbaa !308, !nonnull !310
// CHECK:STDOUT:   %call = call noundef i32 @_ZNK13ConvertToEnumcv2E2Ev(ptr noundef nonnull align 1 dereferenceable(1) %0)
// CHECK:STDOUT:   ret i32 %call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCompare3.Main(i16 %a, ptr %b) #0 !dbg !190 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc26_15.1.temp = alloca i16, align 2, !dbg !198
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc26_15.1.temp), !dbg !198
// CHECK:STDOUT:   call void @_ZNK14ConvertToEnum2cvN1C1EEEv.carbon_thunk._(ptr %b, ptr %.loc26_15.1.temp), !dbg !198
// CHECK:STDOUT:   %.loc26_15.5 = load i16, ptr %.loc26_15.1.temp, align 2, !dbg !198
// CHECK:STDOUT:   %OrderedWith.WithSelf.Equal.call = icmp eq i16 %a, %.loc26_15.5, !dbg !197
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.da290558fb92070b(ptr %.loc26_15.1.temp), !dbg !198
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.Equal.call, !dbg !199
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZNK14ConvertToEnum2cvN1C1EEEv.carbon_thunk._(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !312
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !45
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !45
// CHECK:STDOUT:   %1 = load ptr, ptr %this.addr, align 8, !tbaa !312, !nonnull !310
// CHECK:STDOUT:   %call = call noundef signext i16 @_ZNK14ConvertToEnum2cvN1C1EEEv(ptr noundef nonnull align 1 dereferenceable(1) %1)
// CHECK:STDOUT:   store i16 %call, ptr %0, align 2, !tbaa !47
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCallCompareGeneric.Main(i16 %a, ptr %b) #0 !dbg !201 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %CompareGeneric.call = call i1 @_CCompareGeneric.Main.356c7bb457b5b49b(i16 %a, ptr %b), !dbg !210
// CHECK:STDOUT:   ret i1 %CompareGeneric.call, !dbg !211
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CEqual:thunk:EqWith.83b9626e83cf3dcc.Core:OrderedWith.Core"(i16 %self, ptr %other) #2 !dbg !213 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.4.temp = alloca i16, align 2, !dbg !216
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.4.temp), !dbg !216
// CHECK:STDOUT:   call void @_ZNK13ConvertToEnumcvN1C1EEEv.carbon_thunk._(ptr %other, ptr %.4.temp), !dbg !216
// CHECK:STDOUT:   %.8 = load i16, ptr %.4.temp, align 2, !dbg !216
// CHECK:STDOUT:   %OrderedWith.WithSelf.Equal.call = icmp eq i16 %self, %.8, !dbg !216
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.da290558fb92070b(ptr %.4.temp), !dbg !216
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.Equal.call, !dbg !216
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CNotEqual:thunk:EqWith.83b9626e83cf3dcc.Core:OrderedWith.Core"(i16 %self, ptr %other) #2 !dbg !218 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.4.temp = alloca i16, align 2, !dbg !221
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.4.temp), !dbg !221
// CHECK:STDOUT:   call void @_ZNK13ConvertToEnumcvN1C1EEEv.carbon_thunk._(ptr %other, ptr %.4.temp), !dbg !221
// CHECK:STDOUT:   %.8 = load i16, ptr %.4.temp, align 2, !dbg !221
// CHECK:STDOUT:   %OrderedWith.WithSelf.NotEqual.call = icmp ne i16 %self, %.8, !dbg !221
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.da290558fb92070b(ptr %.4.temp), !dbg !221
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.NotEqual.call, !dbg !221
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCallCompareGeneric2.Main(i32 %a, ptr %b) #0 !dbg !223 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %CompareGeneric.call = call i1 @_CCompareGeneric.Main.4512c7023c79a313(i32 %a, ptr %b), !dbg !232
// CHECK:STDOUT:   ret i1 %CompareGeneric.call, !dbg !233
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CEqual:thunk:EqWith.e8fe48de943d9652.Core:OrderedWith.Core"(i32 %self, ptr %other) #2 !dbg !235 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %__carbon_thunk.call = call i32 @_ZNK13ConvertToEnumcv2E2Ev.carbon_thunk._(ptr %other), !dbg !238
// CHECK:STDOUT:   %OrderedWith.WithSelf.Equal.call = icmp eq i32 %self, %__carbon_thunk.call, !dbg !238
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.Equal.call, !dbg !238
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CNotEqual:thunk:EqWith.e8fe48de943d9652.Core:OrderedWith.Core"(i32 %self, ptr %other) #2 !dbg !240 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %__carbon_thunk.call = call i32 @_ZNK13ConvertToEnumcv2E2Ev.carbon_thunk._(ptr %other), !dbg !243
// CHECK:STDOUT:   %OrderedWith.WithSelf.NotEqual.call = icmp ne i32 %self, %__carbon_thunk.call, !dbg !243
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.NotEqual.call, !dbg !243
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i1 @_CCallCompareGeneric3.Main(i16 %a, ptr %b) #0 !dbg !245 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %CompareGeneric.call = call i1 @_CCompareGeneric.Main.070a42571d1942e2(i16 %a, ptr %b), !dbg !254
// CHECK:STDOUT:   ret i1 %CompareGeneric.call, !dbg !255
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CEqual:thunk:EqWith.a294139a6615e163.Core:OrderedWith.Core"(i16 %self, ptr %other) #2 !dbg !257 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.4.temp = alloca i16, align 2, !dbg !260
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.4.temp), !dbg !260
// CHECK:STDOUT:   call void @_ZNK14ConvertToEnum2cvN1C1EEEv.carbon_thunk._(ptr %other, ptr %.4.temp), !dbg !260
// CHECK:STDOUT:   %.8 = load i16, ptr %.4.temp, align 2, !dbg !260
// CHECK:STDOUT:   %OrderedWith.WithSelf.Equal.call = icmp eq i16 %self, %.8, !dbg !260
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.da290558fb92070b(ptr %.4.temp), !dbg !260
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.Equal.call, !dbg !260
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr i1 @"_CNotEqual:thunk:EqWith.a294139a6615e163.Core:OrderedWith.Core"(i16 %self, ptr %other) #2 !dbg !262 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.4.temp = alloca i16, align 2, !dbg !265
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.4.temp), !dbg !265
// CHECK:STDOUT:   call void @_ZNK14ConvertToEnum2cvN1C1EEEv.carbon_thunk._(ptr %other, ptr %.4.temp), !dbg !265
// CHECK:STDOUT:   %.8 = load i16, ptr %.4.temp, align 2, !dbg !265
// CHECK:STDOUT:   %OrderedWith.WithSelf.NotEqual.call = icmp ne i16 %self, %.8, !dbg !265
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.da290558fb92070b(ptr %.4.temp), !dbg !265
// CHECK:STDOUT:   ret i1 %OrderedWith.WithSelf.NotEqual.call, !dbg !265
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.da290558fb92070b(ptr %t) #0 !dbg !267 {
// CHECK:STDOUT:   call void @"_COp:thunk:DestroyImpl.f6bdc88761165680.Core:DestroyImpl.Core"(ptr %t), !dbg !270
// CHECK:STDOUT:   ret void, !dbg !272
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CCompareGeneric.Main.356c7bb457b5b49b(i16 %x, ptr %y) #0 !dbg !274 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.call = call i1 @"_CEqual:thunk:EqWith.83b9626e83cf3dcc.Core:OrderedWith.Core"(i16 %x, ptr %y), !dbg !281
// CHECK:STDOUT:   ret i1 %EqWith.WithSelf.Equal.call, !dbg !283
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CCompareGeneric.Main.4512c7023c79a313(i32 %x, ptr %y) #0 !dbg !285 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.call = call i1 @"_CEqual:thunk:EqWith.e8fe48de943d9652.Core:OrderedWith.Core"(i32 %x, ptr %y), !dbg !292
// CHECK:STDOUT:   ret i1 %EqWith.WithSelf.Equal.call, !dbg !294
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i1 @_CCompareGeneric.Main.070a42571d1942e2(i16 %x, ptr %y) #0 !dbg !296 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %EqWith.WithSelf.Equal.call = call i1 @"_CEqual:thunk:EqWith.a294139a6615e163.Core:OrderedWith.Core"(i16 %x, ptr %y), !dbg !303
// CHECK:STDOUT:   ret i1 %EqWith.WithSelf.Equal.call, !dbg !305
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
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.da290558fb92070b, { 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #4 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!155}
// CHECK:STDOUT: !llvm.module.flags = !{!48, !49, !50, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!53}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !12 = !{null, !6}
// CHECK:STDOUT: !13 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !34 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !42 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !43 = !{!"omnipotent char", !42, i64 0}
// CHECK:STDOUT: !44 = !{!"any pointer", !43, i64 0}
// CHECK:STDOUT: !45 = !{!44, !44, i64 0}
// CHECK:STDOUT: !46 = !{!"_ZTSN1C1EE", !43, i64 0}
// CHECK:STDOUT: !47 = !{!46, !46, i64 0}
// CHECK:STDOUT: !48 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !49 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !50 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !51 = !{!"int", !43, i64 0}
// CHECK:STDOUT: !52 = !{!"__libc_errno", !51, i64 0}
// CHECK:STDOUT: !53 = !{!52, !51, i64 0}
// CHECK:STDOUT: !86 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !154 = !DIFile(filename: "compare_class.carbon", directory: "")
// CHECK:STDOUT: !155 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !154, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !156 = !{!9, !6, !9}
// CHECK:STDOUT: !157 = !DISubroutineType(types: !156)
// CHECK:STDOUT: !160 = !{!9, !86, !9}
// CHECK:STDOUT: !161 = !DISubroutineType(types: !160)
// CHECK:STDOUT: !164 = distinct !DISubprogram(name: "Compare", linkageName: "_CCompare.Main", scope: null, file: !154, line: 17, type: !157, spFlags: DISPFlagDefinition, unit: !155, retainedNodes: !174)
// CHECK:STDOUT: !165 = !DILocalVariable(arg: 1, scope: !164, type: !6)
// CHECK:STDOUT: !166 = !DILocalVariable(arg: 2, scope: !164, type: !9)
// CHECK:STDOUT: !171 = !DILocation(line: 18, column: 10, scope: !164)
// CHECK:STDOUT: !172 = !DILocation(line: 18, column: 15, scope: !164)
// CHECK:STDOUT: !173 = !DILocation(line: 18, column: 3, scope: !164)
// CHECK:STDOUT: !174 = !{!165, !166}
// CHECK:STDOUT: !175 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:DestroyImpl.f6bdc88761165680.Core:DestroyImpl.Core", scope: null, file: !154, type: !13, spFlags: DISPFlagDefinition, unit: !155, retainedNodes: !178)
// CHECK:STDOUT: !176 = !DILocalVariable(arg: 1, scope: !175, type: !6)
// CHECK:STDOUT: !177 = !DILocation(line: 0, scope: !175)
// CHECK:STDOUT: !178 = !{!176}
// CHECK:STDOUT: !179 = distinct !DISubprogram(name: "Compare2", linkageName: "_CCompare2.Main", scope: null, file: !154, line: 21, type: !161, spFlags: DISPFlagDefinition, unit: !155, retainedNodes: !189)
// CHECK:STDOUT: !180 = !DILocalVariable(arg: 1, scope: !179, type: !86)
// CHECK:STDOUT: !181 = !DILocalVariable(arg: 2, scope: !179, type: !9)
// CHECK:STDOUT: !186 = !DILocation(line: 22, column: 10, scope: !179)
// CHECK:STDOUT: !187 = !DILocation(line: 22, column: 15, scope: !179)
// CHECK:STDOUT: !188 = !DILocation(line: 22, column: 3, scope: !179)
// CHECK:STDOUT: !189 = !{!180, !181}
// CHECK:STDOUT: !190 = distinct !DISubprogram(name: "Compare3", linkageName: "_CCompare3.Main", scope: null, file: !154, line: 25, type: !157, spFlags: DISPFlagDefinition, unit: !155, retainedNodes: !200)
// CHECK:STDOUT: !191 = !DILocalVariable(arg: 1, scope: !190, type: !6)
// CHECK:STDOUT: !192 = !DILocalVariable(arg: 2, scope: !190, type: !9)
// CHECK:STDOUT: !197 = !DILocation(line: 26, column: 10, scope: !190)
// CHECK:STDOUT: !198 = !DILocation(line: 26, column: 15, scope: !190)
// CHECK:STDOUT: !199 = !DILocation(line: 26, column: 3, scope: !190)
// CHECK:STDOUT: !200 = !{!191, !192}
// CHECK:STDOUT: !201 = distinct !DISubprogram(name: "CallCompareGeneric", linkageName: "_CCallCompareGeneric.Main", scope: null, file: !154, line: 35, type: !157, spFlags: DISPFlagDefinition, unit: !155, retainedNodes: !212)
// CHECK:STDOUT: !202 = !DILocalVariable(arg: 1, scope: !201, type: !6)
// CHECK:STDOUT: !203 = !DILocalVariable(arg: 2, scope: !201, type: !9)
// CHECK:STDOUT: !210 = !DILocation(line: 36, column: 10, scope: !201)
// CHECK:STDOUT: !211 = !DILocation(line: 36, column: 3, scope: !201)
// CHECK:STDOUT: !212 = !{!202, !203}
// CHECK:STDOUT: !213 = distinct !DISubprogram(name: "Equal", linkageName: "_CEqual:thunk:EqWith.83b9626e83cf3dcc.Core:OrderedWith.Core", scope: null, file: !154, type: !157, spFlags: DISPFlagDefinition, unit: !155, retainedNodes: !217)
// CHECK:STDOUT: !214 = !DILocalVariable(arg: 1, scope: !213, type: !6)
// CHECK:STDOUT: !215 = !DILocalVariable(arg: 2, scope: !213, type: !9)
// CHECK:STDOUT: !216 = !DILocation(line: 0, scope: !213)
// CHECK:STDOUT: !217 = !{!214, !215}
// CHECK:STDOUT: !218 = distinct !DISubprogram(name: "NotEqual", linkageName: "_CNotEqual:thunk:EqWith.83b9626e83cf3dcc.Core:OrderedWith.Core", scope: null, file: !154, type: !157, spFlags: DISPFlagDefinition, unit: !155, retainedNodes: !222)
// CHECK:STDOUT: !219 = !DILocalVariable(arg: 1, scope: !218, type: !6)
// CHECK:STDOUT: !220 = !DILocalVariable(arg: 2, scope: !218, type: !9)
// CHECK:STDOUT: !221 = !DILocation(line: 0, scope: !218)
// CHECK:STDOUT: !222 = !{!219, !220}
// CHECK:STDOUT: !223 = distinct !DISubprogram(name: "CallCompareGeneric2", linkageName: "_CCallCompareGeneric2.Main", scope: null, file: !154, line: 39, type: !161, spFlags: DISPFlagDefinition, unit: !155, retainedNodes: !234)
// CHECK:STDOUT: !224 = !DILocalVariable(arg: 1, scope: !223, type: !86)
// CHECK:STDOUT: !225 = !DILocalVariable(arg: 2, scope: !223, type: !9)
// CHECK:STDOUT: !232 = !DILocation(line: 40, column: 10, scope: !223)
// CHECK:STDOUT: !233 = !DILocation(line: 40, column: 3, scope: !223)
// CHECK:STDOUT: !234 = !{!224, !225}
// CHECK:STDOUT: !235 = distinct !DISubprogram(name: "Equal", linkageName: "_CEqual:thunk:EqWith.e8fe48de943d9652.Core:OrderedWith.Core", scope: null, file: !154, type: !161, spFlags: DISPFlagDefinition, unit: !155, retainedNodes: !239)
// CHECK:STDOUT: !236 = !DILocalVariable(arg: 1, scope: !235, type: !86)
// CHECK:STDOUT: !237 = !DILocalVariable(arg: 2, scope: !235, type: !9)
// CHECK:STDOUT: !238 = !DILocation(line: 0, scope: !235)
// CHECK:STDOUT: !239 = !{!236, !237}
// CHECK:STDOUT: !240 = distinct !DISubprogram(name: "NotEqual", linkageName: "_CNotEqual:thunk:EqWith.e8fe48de943d9652.Core:OrderedWith.Core", scope: null, file: !154, type: !161, spFlags: DISPFlagDefinition, unit: !155, retainedNodes: !244)
// CHECK:STDOUT: !241 = !DILocalVariable(arg: 1, scope: !240, type: !86)
// CHECK:STDOUT: !242 = !DILocalVariable(arg: 2, scope: !240, type: !9)
// CHECK:STDOUT: !243 = !DILocation(line: 0, scope: !240)
// CHECK:STDOUT: !244 = !{!241, !242}
// CHECK:STDOUT: !245 = distinct !DISubprogram(name: "CallCompareGeneric3", linkageName: "_CCallCompareGeneric3.Main", scope: null, file: !154, line: 43, type: !157, spFlags: DISPFlagDefinition, unit: !155, retainedNodes: !256)
// CHECK:STDOUT: !246 = !DILocalVariable(arg: 1, scope: !245, type: !6)
// CHECK:STDOUT: !247 = !DILocalVariable(arg: 2, scope: !245, type: !9)
// CHECK:STDOUT: !254 = !DILocation(line: 44, column: 10, scope: !245)
// CHECK:STDOUT: !255 = !DILocation(line: 44, column: 3, scope: !245)
// CHECK:STDOUT: !256 = !{!246, !247}
// CHECK:STDOUT: !257 = distinct !DISubprogram(name: "Equal", linkageName: "_CEqual:thunk:EqWith.a294139a6615e163.Core:OrderedWith.Core", scope: null, file: !154, type: !157, spFlags: DISPFlagDefinition, unit: !155, retainedNodes: !261)
// CHECK:STDOUT: !258 = !DILocalVariable(arg: 1, scope: !257, type: !6)
// CHECK:STDOUT: !259 = !DILocalVariable(arg: 2, scope: !257, type: !9)
// CHECK:STDOUT: !260 = !DILocation(line: 0, scope: !257)
// CHECK:STDOUT: !261 = !{!258, !259}
// CHECK:STDOUT: !262 = distinct !DISubprogram(name: "NotEqual", linkageName: "_CNotEqual:thunk:EqWith.a294139a6615e163.Core:OrderedWith.Core", scope: null, file: !154, type: !157, spFlags: DISPFlagDefinition, unit: !155, retainedNodes: !266)
// CHECK:STDOUT: !263 = !DILocalVariable(arg: 1, scope: !262, type: !6)
// CHECK:STDOUT: !264 = !DILocalVariable(arg: 2, scope: !262, type: !9)
// CHECK:STDOUT: !265 = !DILocation(line: 0, scope: !262)
// CHECK:STDOUT: !266 = !{!263, !264}
// CHECK:STDOUT: !267 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.da290558fb92070b", scope: null, file: !34, line: 30, type: !13, spFlags: DISPFlagDefinition, unit: !155, retainedNodes: !273)
// CHECK:STDOUT: !268 = !DILocalVariable(arg: 1, scope: !267, type: !6)
// CHECK:STDOUT: !270 = !DILocation(line: 31, column: 5, scope: !267)
// CHECK:STDOUT: !272 = !DILocation(line: 30, column: 1, scope: !267)
// CHECK:STDOUT: !273 = !{!268}
// CHECK:STDOUT: !274 = distinct !DISubprogram(name: "CompareGeneric", linkageName: "_CCompareGeneric.Main.356c7bb457b5b49b", scope: null, file: !154, line: 29, type: !157, spFlags: DISPFlagDefinition, unit: !155, retainedNodes: !284)
// CHECK:STDOUT: !275 = !DILocalVariable(arg: 1, scope: !274, type: !6)
// CHECK:STDOUT: !276 = !DILocalVariable(arg: 2, scope: !274, type: !9)
// CHECK:STDOUT: !281 = !DILocation(line: 30, column: 10, scope: !274)
// CHECK:STDOUT: !283 = !DILocation(line: 30, column: 3, scope: !274)
// CHECK:STDOUT: !284 = !{!275, !276}
// CHECK:STDOUT: !285 = distinct !DISubprogram(name: "CompareGeneric", linkageName: "_CCompareGeneric.Main.4512c7023c79a313", scope: null, file: !154, line: 29, type: !161, spFlags: DISPFlagDefinition, unit: !155, retainedNodes: !295)
// CHECK:STDOUT: !286 = !DILocalVariable(arg: 1, scope: !285, type: !86)
// CHECK:STDOUT: !287 = !DILocalVariable(arg: 2, scope: !285, type: !9)
// CHECK:STDOUT: !292 = !DILocation(line: 30, column: 10, scope: !285)
// CHECK:STDOUT: !294 = !DILocation(line: 30, column: 3, scope: !285)
// CHECK:STDOUT: !295 = !{!286, !287}
// CHECK:STDOUT: !296 = distinct !DISubprogram(name: "CompareGeneric", linkageName: "_CCompareGeneric.Main.070a42571d1942e2", scope: null, file: !154, line: 29, type: !157, spFlags: DISPFlagDefinition, unit: !155, retainedNodes: !306)
// CHECK:STDOUT: !297 = !DILocalVariable(arg: 1, scope: !296, type: !6)
// CHECK:STDOUT: !298 = !DILocalVariable(arg: 2, scope: !296, type: !9)
// CHECK:STDOUT: !303 = !DILocation(line: 30, column: 10, scope: !296)
// CHECK:STDOUT: !305 = !DILocation(line: 30, column: 3, scope: !296)
// CHECK:STDOUT: !306 = !{!297, !298}
// CHECK:STDOUT: !307 = !{!"p1 _ZTS13ConvertToEnum", !44, i64 0}
// CHECK:STDOUT: !308 = !{!307, !307, i64 0}
// CHECK:STDOUT: !310 = !{}
// CHECK:STDOUT: !311 = !{!"p1 _ZTS14ConvertToEnum2", !44, i64 0}
// CHECK:STDOUT: !312 = !{!311, !311, i64 0}
// CHECK:STDOUT:
