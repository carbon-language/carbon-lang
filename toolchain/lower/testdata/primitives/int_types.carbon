// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/primitives/int_types.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/primitives/int_types.carbon

// --- pass_and_return.carbon

library "[[@TEST_NAME]]";

fn F_i8(a: i8) -> i8 { return a; }
fn F_u16(a: u16) -> u16 { return a; }
fn F_i64(a: i64) -> i64 { return a; }
fn F_u65536(a: u65536) -> u65536 { return a; }

// --- load_and_store.carbon

library "[[@TEST_NAME]]";

fn LoadStore_Bytes(p: Core.Int(24)*, n: Core.Int(24)) -> Core.Int(24) {
  let v: Core.Int(24) = *p;
  *p = n;
  return v;
}

fn LoadStore_NotBytes(p: Core.Int(13)*, n: Core.Int(13)) -> Core.Int(13) {
  let v: Core.Int(13) = *p;
  *p = n;
  return v;
}

// --- convert.carbon

library "[[@TEST_NAME]]";

fn ImplicitWidenSigned(a: i8) -> i32 { return a; }
fn ImplicitWidenUnsigned(a: u8) -> u32 { return a; }
fn ImplicitWidenUnsignedToSigned(a: u8) -> i32 { return a; }
fn ImplicitWidenCustom(a: Core.Int(4)) -> Core.Int(5) { return a; }
fn ImplicitWidenUnsignedCustom(a: Core.UInt(4)) -> Core.UInt(5) { return a; }

// CHECK:STDOUT: ; ModuleID = 'pass_and_return.carbon'
// CHECK:STDOUT: source_filename = "pass_and_return.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i8 @_CF_i8.Main(i8 %a) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i8 %a, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i16 @_CF_u16.Main(i16 %a) #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i16 %a, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CF_i64.Main(i64 %a) #0 !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i64 %a, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i65536 @_CF_u65536.Main(i65536 %a) #0 !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i65536 %a, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "pass_and_return.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "F_i8", linkageName: "_CF_i8.Main", scope: null, file: !3, line: 4, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !8)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !7}
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 8, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !{!9}
// CHECK:STDOUT: !9 = !DILocalVariable(arg: 1, scope: !4, type: !7)
// CHECK:STDOUT: !10 = !DILocation(line: 4, column: 24, scope: !4)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "F_u16", linkageName: "_CF_u16.Main", scope: null, file: !3, line: 5, type: !12, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !15)
// CHECK:STDOUT: !12 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !13 = !{!14, !14}
// CHECK:STDOUT: !14 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !15 = !{!16}
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 1, scope: !11, type: !14)
// CHECK:STDOUT: !17 = !DILocation(line: 5, column: 27, scope: !11)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "F_i64", linkageName: "_CF_i64.Main", scope: null, file: !3, line: 6, type: !19, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !22)
// CHECK:STDOUT: !19 = !DISubroutineType(types: !20)
// CHECK:STDOUT: !20 = !{!21, !21}
// CHECK:STDOUT: !21 = !DIBasicType(name: "int", size: 64, encoding: DW_ATE_signed)
// CHECK:STDOUT: !22 = !{!23}
// CHECK:STDOUT: !23 = !DILocalVariable(arg: 1, scope: !18, type: !21)
// CHECK:STDOUT: !24 = !DILocation(line: 6, column: 27, scope: !18)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "F_u65536", linkageName: "_CF_u65536.Main", scope: null, file: !3, line: 7, type: !26, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !29)
// CHECK:STDOUT: !26 = !DISubroutineType(types: !27)
// CHECK:STDOUT: !27 = !{!28, !28}
// CHECK:STDOUT: !28 = !DIBasicType(name: "int", size: 65536, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !29 = !{!30}
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 1, scope: !25, type: !28)
// CHECK:STDOUT: !31 = !DILocation(line: 7, column: 36, scope: !25)
// CHECK:STDOUT: ; ModuleID = 'load_and_store.carbon'
// CHECK:STDOUT: source_filename = "load_and_store.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i24 @_CLoadStore_Bytes.Main(ptr %p, i24 %n) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc5_25.2 = load i24, ptr %p, align 4, !dbg !12
// CHECK:STDOUT:   store i24 %n, ptr %p, align 4, !dbg !13
// CHECK:STDOUT:   ret i24 %.loc5_25.2, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i13 @_CLoadStore_NotBytes.Main(ptr %p, i13 %n) #0 !dbg !15 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_25.2 = load i16, ptr %p, align 2, !dbg !22
// CHECK:STDOUT:   %.loc11_25.21 = trunc i16 %.loc11_25.2 to i13, !dbg !22
// CHECK:STDOUT:   %0 = zext i13 %n to i16, !dbg !23
// CHECK:STDOUT:   store i16 %0, ptr %p, align 2, !dbg !23
// CHECK:STDOUT:   ret i13 %.loc11_25.21, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "load_and_store.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "LoadStore_Bytes", linkageName: "_CLoadStore_Bytes.Main", scope: null, file: !3, line: 4, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !9)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !8, !7}
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 24, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !9 = !{!10, !11}
// CHECK:STDOUT: !10 = !DILocalVariable(arg: 1, scope: !4, type: !8)
// CHECK:STDOUT: !11 = !DILocalVariable(arg: 2, scope: !4, type: !7)
// CHECK:STDOUT: !12 = !DILocation(line: 5, column: 25, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 6, column: 3, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 7, column: 3, scope: !4)
// CHECK:STDOUT: !15 = distinct !DISubprogram(name: "LoadStore_NotBytes", linkageName: "_CLoadStore_NotBytes.Main", scope: null, file: !3, line: 10, type: !16, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !19)
// CHECK:STDOUT: !16 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !17 = !{!18, !8, !18}
// CHECK:STDOUT: !18 = !DIBasicType(name: "int", size: 13, encoding: DW_ATE_signed)
// CHECK:STDOUT: !19 = !{!20, !21}
// CHECK:STDOUT: !20 = !DILocalVariable(arg: 1, scope: !15, type: !8)
// CHECK:STDOUT: !21 = !DILocalVariable(arg: 2, scope: !15, type: !18)
// CHECK:STDOUT: !22 = !DILocation(line: 11, column: 25, scope: !15)
// CHECK:STDOUT: !23 = !DILocation(line: 12, column: 3, scope: !15)
// CHECK:STDOUT: !24 = !DILocation(line: 13, column: 3, scope: !15)
// CHECK:STDOUT: ; ModuleID = 'convert.carbon'
// CHECK:STDOUT: source_filename = "convert.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CImplicitWidenSigned.Main(i8 %a) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %From.binding.as_type.as.ImplicitAs.impl.Convert.call = call i32 @"_CConvert.272fc3e88346d0d9:ImplicitAs.d993a5132e000405.Core.aa7228341bdcca99"(i8 %a), !dbg !11
// CHECK:STDOUT:   ret i32 %From.binding.as_type.as.ImplicitAs.impl.Convert.call, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CImplicitWidenUnsigned.Main(i8 %a) #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.call = call i32 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.952fa4f2c764f083.Core.0e306488ea9f63ef"(i8 %a), !dbg !19
// CHECK:STDOUT:   ret i32 %UInt.as.ImplicitAs.impl.Convert.call, !dbg !19
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CImplicitWidenUnsignedToSigned.Main(i8 %a) #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.call = call i32 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.952fa4f2c764f083.Core.0237a74cabc89c74"(i8 %a), !dbg !25
// CHECK:STDOUT:   ret i32 %UInt.as.ImplicitAs.impl.Convert.call, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i5 @_CImplicitWidenCustom.Main(i4 %a) #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %From.binding.as_type.as.ImplicitAs.impl.Convert.call = call i5 @"_CConvert.272fc3e88346d0d9:ImplicitAs.d993a5132e000405.Core.61459d64876d7c49"(i4 %a), !dbg !33
// CHECK:STDOUT:   ret i5 %From.binding.as_type.as.ImplicitAs.impl.Convert.call, !dbg !33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i5 @_CImplicitWidenUnsignedCustom.Main(i4 %a) #0 !dbg !34 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.call = call i5 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.952fa4f2c764f083.Core.d1cbb2cbeb4d860b"(i4 %a), !dbg !41
// CHECK:STDOUT:   ret i5 %UInt.as.ImplicitAs.impl.Convert.call, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.272fc3e88346d0d9:ImplicitAs.d993a5132e000405.Core.aa7228341bdcca99"(i8 %self) #0 !dbg !42 {
// CHECK:STDOUT:   %1 = call i8 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.ca1ac10d3fc7c1a8"(i8 %self), !dbg !46
// CHECK:STDOUT:   %2 = sext i8 %1 to i32, !dbg !47
// CHECK:STDOUT:   ret i32 %2, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.952fa4f2c764f083.Core.0e306488ea9f63ef"(i8 %self) #0 !dbg !49 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.UInt.9452f4c51951679b.Core:FromUInt.382a8acefd0525fe.Core.081daf9d9b61e05a"(i8 %self), !dbg !53
// CHECK:STDOUT:   ret i32 %1, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.952fa4f2c764f083.Core.0237a74cabc89c74"(i8 %self) #0 !dbg !55 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.Int.9452f4c51951679b.Core:FromUInt.a15ed48781b6c2b7.Core.069b661f6a62a7bd"(i8 %self), !dbg !58
// CHECK:STDOUT:   ret i32 %1, !dbg !59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i5 @"_CConvert.272fc3e88346d0d9:ImplicitAs.d993a5132e000405.Core.61459d64876d7c49"(i4 %self) #0 !dbg !60 {
// CHECK:STDOUT:   %1 = call i4 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.044c7d430ddd4fd6"(i4 %self), !dbg !63
// CHECK:STDOUT:   %2 = sext i4 %1 to i5, !dbg !64
// CHECK:STDOUT:   ret i5 %2, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i5 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.952fa4f2c764f083.Core.d1cbb2cbeb4d860b"(i4 %self) #0 !dbg !66 {
// CHECK:STDOUT:   %1 = call i5 @"_CConvert.UInt.9452f4c51951679b.Core:FromUInt.382a8acefd0525fe.Core.b0c274f0050b1ce4"(i4 %self), !dbg !69
// CHECK:STDOUT:   ret i5 %1, !dbg !70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.ca1ac10d3fc7c1a8"(i8 %self) #0 !dbg !71 {
// CHECK:STDOUT:   ret i8 %self, !dbg !76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.UInt.9452f4c51951679b.Core:FromUInt.382a8acefd0525fe.Core.081daf9d9b61e05a"(i8 %from) #0 !dbg !77 {
// CHECK:STDOUT:   %1 = call i8 @"_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.ca1ac10d3fc7c1a8"(i8 %from), !dbg !80
// CHECK:STDOUT:   %2 = zext i8 %1 to i32, !dbg !81
// CHECK:STDOUT:   ret i32 %2, !dbg !82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.Int.9452f4c51951679b.Core:FromUInt.a15ed48781b6c2b7.Core.069b661f6a62a7bd"(i8 %from) #0 !dbg !83 {
// CHECK:STDOUT:   %1 = call i8 @"_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.ca1ac10d3fc7c1a8"(i8 %from), !dbg !86
// CHECK:STDOUT:   %2 = zext i8 %1 to i32, !dbg !87
// CHECK:STDOUT:   ret i32 %2, !dbg !88
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i4 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.044c7d430ddd4fd6"(i4 %self) #0 !dbg !89 {
// CHECK:STDOUT:   ret i4 %self, !dbg !94
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i5 @"_CConvert.UInt.9452f4c51951679b.Core:FromUInt.382a8acefd0525fe.Core.b0c274f0050b1ce4"(i4 %from) #0 !dbg !95 {
// CHECK:STDOUT:   %1 = call i4 @"_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.044c7d430ddd4fd6"(i4 %from), !dbg !98
// CHECK:STDOUT:   %2 = zext i4 %1 to i5, !dbg !99
// CHECK:STDOUT:   ret i5 %2, !dbg !100
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.ca1ac10d3fc7c1a8"(i8 %self) #0 !dbg !101 {
// CHECK:STDOUT:   ret i8 %self, !dbg !106
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i4 @"_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.044c7d430ddd4fd6"(i4 %self) #0 !dbg !107 {
// CHECK:STDOUT:   ret i4 %self, !dbg !112
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.ca1ac10d3fc7c1a8", { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "convert.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "ImplicitWidenSigned", linkageName: "_CImplicitWidenSigned.Main", scope: null, file: !3, line: 4, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !9)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !8}
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !DIBasicType(name: "int", size: 8, encoding: DW_ATE_signed)
// CHECK:STDOUT: !9 = !{!10}
// CHECK:STDOUT: !10 = !DILocalVariable(arg: 1, scope: !4, type: !8)
// CHECK:STDOUT: !11 = !DILocation(line: 4, column: 40, scope: !4)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "ImplicitWidenUnsigned", linkageName: "_CImplicitWidenUnsigned.Main", scope: null, file: !3, line: 5, type: !13, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !17)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{!15, !16}
// CHECK:STDOUT: !15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !16 = !DIBasicType(name: "int", size: 8, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !17 = !{!18}
// CHECK:STDOUT: !18 = !DILocalVariable(arg: 1, scope: !12, type: !16)
// CHECK:STDOUT: !19 = !DILocation(line: 5, column: 42, scope: !12)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "ImplicitWidenUnsignedToSigned", linkageName: "_CImplicitWidenUnsignedToSigned.Main", scope: null, file: !3, line: 6, type: !21, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !23)
// CHECK:STDOUT: !21 = !DISubroutineType(types: !22)
// CHECK:STDOUT: !22 = !{!7, !16}
// CHECK:STDOUT: !23 = !{!24}
// CHECK:STDOUT: !24 = !DILocalVariable(arg: 1, scope: !20, type: !16)
// CHECK:STDOUT: !25 = !DILocation(line: 6, column: 50, scope: !20)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "ImplicitWidenCustom", linkageName: "_CImplicitWidenCustom.Main", scope: null, file: !3, line: 7, type: !27, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !31)
// CHECK:STDOUT: !27 = !DISubroutineType(types: !28)
// CHECK:STDOUT: !28 = !{!29, !30}
// CHECK:STDOUT: !29 = !DIBasicType(name: "int", size: 5, encoding: DW_ATE_signed)
// CHECK:STDOUT: !30 = !DIBasicType(name: "int", size: 4, encoding: DW_ATE_signed)
// CHECK:STDOUT: !31 = !{!32}
// CHECK:STDOUT: !32 = !DILocalVariable(arg: 1, scope: !26, type: !30)
// CHECK:STDOUT: !33 = !DILocation(line: 7, column: 57, scope: !26)
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "ImplicitWidenUnsignedCustom", linkageName: "_CImplicitWidenUnsignedCustom.Main", scope: null, file: !3, line: 8, type: !35, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !39)
// CHECK:STDOUT: !35 = !DISubroutineType(types: !36)
// CHECK:STDOUT: !36 = !{!37, !38}
// CHECK:STDOUT: !37 = !DIBasicType(name: "int", size: 5, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !38 = !DIBasicType(name: "int", size: 4, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !39 = !{!40}
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !34, type: !38)
// CHECK:STDOUT: !41 = !DILocation(line: 8, column: 67, scope: !34)
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.272fc3e88346d0d9:ImplicitAs.d993a5132e000405.Core.aa7228341bdcca99", scope: null, file: !43, line: 62, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !44)
// CHECK:STDOUT: !43 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !44 = !{!45}
// CHECK:STDOUT: !45 = !DILocalVariable(arg: 1, scope: !42, type: !8)
// CHECK:STDOUT: !46 = !DILocation(line: 64, column: 17, scope: !42)
// CHECK:STDOUT: !47 = !DILocation(line: 64, column: 12, scope: !42)
// CHECK:STDOUT: !48 = !DILocation(line: 64, column: 5, scope: !42)
// CHECK:STDOUT: !49 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.952fa4f2c764f083.Core.0e306488ea9f63ef", scope: null, file: !50, line: 86, type: !13, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !51)
// CHECK:STDOUT: !50 = !DIFile(filename: "{{.*}}/prelude/types/uint.carbon", directory: "")
// CHECK:STDOUT: !51 = !{!52}
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 1, scope: !49, type: !16)
// CHECK:STDOUT: !53 = !DILocation(line: 87, column: 12, scope: !49)
// CHECK:STDOUT: !54 = !DILocation(line: 87, column: 5, scope: !49)
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.952fa4f2c764f083.Core.0237a74cabc89c74", scope: null, file: !50, line: 86, type: !21, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !56)
// CHECK:STDOUT: !56 = !{!57}
// CHECK:STDOUT: !57 = !DILocalVariable(arg: 1, scope: !55, type: !16)
// CHECK:STDOUT: !58 = !DILocation(line: 87, column: 12, scope: !55)
// CHECK:STDOUT: !59 = !DILocation(line: 87, column: 5, scope: !55)
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.272fc3e88346d0d9:ImplicitAs.d993a5132e000405.Core.61459d64876d7c49", scope: null, file: !43, line: 62, type: !27, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !61)
// CHECK:STDOUT: !61 = !{!62}
// CHECK:STDOUT: !62 = !DILocalVariable(arg: 1, scope: !60, type: !30)
// CHECK:STDOUT: !63 = !DILocation(line: 64, column: 17, scope: !60)
// CHECK:STDOUT: !64 = !DILocation(line: 64, column: 12, scope: !60)
// CHECK:STDOUT: !65 = !DILocation(line: 64, column: 5, scope: !60)
// CHECK:STDOUT: !66 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.952fa4f2c764f083.Core.d1cbb2cbeb4d860b", scope: null, file: !50, line: 86, type: !35, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !67)
// CHECK:STDOUT: !67 = !{!68}
// CHECK:STDOUT: !68 = !DILocalVariable(arg: 1, scope: !66, type: !38)
// CHECK:STDOUT: !69 = !DILocation(line: 87, column: 12, scope: !66)
// CHECK:STDOUT: !70 = !DILocation(line: 87, column: 5, scope: !66)
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.ca1ac10d3fc7c1a8", scope: null, file: !43, line: 57, type: !72, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !74)
// CHECK:STDOUT: !72 = !DISubroutineType(types: !73)
// CHECK:STDOUT: !73 = !{!8, !8}
// CHECK:STDOUT: !74 = !{!75}
// CHECK:STDOUT: !75 = !DILocalVariable(arg: 1, scope: !71, type: !8)
// CHECK:STDOUT: !76 = !DILocation(line: 57, column: 36, scope: !71)
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.UInt.9452f4c51951679b.Core:FromUInt.382a8acefd0525fe.Core.081daf9d9b61e05a", scope: null, file: !50, line: 70, type: !13, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !78)
// CHECK:STDOUT: !78 = !{!79}
// CHECK:STDOUT: !79 = !DILocalVariable(arg: 1, scope: !77, type: !16)
// CHECK:STDOUT: !80 = !DILocation(line: 72, column: 17, scope: !77)
// CHECK:STDOUT: !81 = !DILocation(line: 72, column: 12, scope: !77)
// CHECK:STDOUT: !82 = !DILocation(line: 72, column: 5, scope: !77)
// CHECK:STDOUT: !83 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Int.9452f4c51951679b.Core:FromUInt.a15ed48781b6c2b7.Core.069b661f6a62a7bd", scope: null, file: !50, line: 78, type: !21, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !84)
// CHECK:STDOUT: !84 = !{!85}
// CHECK:STDOUT: !85 = !DILocalVariable(arg: 1, scope: !83, type: !16)
// CHECK:STDOUT: !86 = !DILocation(line: 80, column: 17, scope: !83)
// CHECK:STDOUT: !87 = !DILocation(line: 80, column: 12, scope: !83)
// CHECK:STDOUT: !88 = !DILocation(line: 80, column: 5, scope: !83)
// CHECK:STDOUT: !89 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.044c7d430ddd4fd6", scope: null, file: !43, line: 57, type: !90, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !92)
// CHECK:STDOUT: !90 = !DISubroutineType(types: !91)
// CHECK:STDOUT: !91 = !{!30, !30}
// CHECK:STDOUT: !92 = !{!93}
// CHECK:STDOUT: !93 = !DILocalVariable(arg: 1, scope: !89, type: !30)
// CHECK:STDOUT: !94 = !DILocation(line: 57, column: 36, scope: !89)
// CHECK:STDOUT: !95 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.UInt.9452f4c51951679b.Core:FromUInt.382a8acefd0525fe.Core.b0c274f0050b1ce4", scope: null, file: !50, line: 70, type: !35, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !96)
// CHECK:STDOUT: !96 = !{!97}
// CHECK:STDOUT: !97 = !DILocalVariable(arg: 1, scope: !95, type: !38)
// CHECK:STDOUT: !98 = !DILocation(line: 72, column: 17, scope: !95)
// CHECK:STDOUT: !99 = !DILocation(line: 72, column: 12, scope: !95)
// CHECK:STDOUT: !100 = !DILocation(line: 72, column: 5, scope: !95)
// CHECK:STDOUT: !101 = distinct !DISubprogram(name: "AsUInt", linkageName: "_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.ca1ac10d3fc7c1a8", scope: null, file: !50, line: 58, type: !102, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !104)
// CHECK:STDOUT: !102 = !DISubroutineType(types: !103)
// CHECK:STDOUT: !103 = !{!16, !16}
// CHECK:STDOUT: !104 = !{!105}
// CHECK:STDOUT: !105 = !DILocalVariable(arg: 1, scope: !101, type: !16)
// CHECK:STDOUT: !106 = !DILocation(line: 58, column: 37, scope: !101)
// CHECK:STDOUT: !107 = distinct !DISubprogram(name: "AsUInt", linkageName: "_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.044c7d430ddd4fd6", scope: null, file: !50, line: 58, type: !108, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !110)
// CHECK:STDOUT: !108 = !DISubroutineType(types: !109)
// CHECK:STDOUT: !109 = !{!38, !38}
// CHECK:STDOUT: !110 = !{!111}
// CHECK:STDOUT: !111 = !DILocalVariable(arg: 1, scope: !107, type: !38)
// CHECK:STDOUT: !112 = !DILocation(line: 58, column: 37, scope: !107)
