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

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'pass_and_return.carbon'
// CHECK:STDOUT: source_filename = "pass_and_return.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i8 @_CF_i8.Main(i8 %a) #0 !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i8 %a, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i16 @_CF_u16.Main(i16 %a) #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i16 %a, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i64 @_CF_i64.Main(i64 %a) #0 !dbg !34 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i64 %a, !dbg !40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i65536 @_CF_u65536.Main(i65536 %a) #0 !dbg !42 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i65536 %a, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "pass_and_return.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 8, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !8 = !DIBasicType(name: "int", size: 64, encoding: DW_ATE_signed)
// CHECK:STDOUT: !9 = !DIBasicType(name: "int", size: 65536, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !10 = !{!6, !6}
// CHECK:STDOUT: !11 = !DISubroutineType(types: !10)
// CHECK:STDOUT: !12 = !{!7, !7}
// CHECK:STDOUT: !13 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !14 = !{!8, !8}
// CHECK:STDOUT: !15 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !16 = !{!9, !9}
// CHECK:STDOUT: !17 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "F_i8", linkageName: "_CF_i8.Main", scope: null, file: !4, line: 4, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !25)
// CHECK:STDOUT: !19 = !DILocalVariable(arg: 1, scope: !18, type: !6)
// CHECK:STDOUT: !24 = !DILocation(line: 4, column: 24, scope: !18)
// CHECK:STDOUT: !25 = !{!19}
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "F_u16", linkageName: "_CF_u16.Main", scope: null, file: !4, line: 5, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !33)
// CHECK:STDOUT: !27 = !DILocalVariable(arg: 1, scope: !26, type: !7)
// CHECK:STDOUT: !32 = !DILocation(line: 5, column: 27, scope: !26)
// CHECK:STDOUT: !33 = !{!27}
// CHECK:STDOUT: !34 = distinct !DISubprogram(name: "F_i64", linkageName: "_CF_i64.Main", scope: null, file: !4, line: 6, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !41)
// CHECK:STDOUT: !35 = !DILocalVariable(arg: 1, scope: !34, type: !8)
// CHECK:STDOUT: !40 = !DILocation(line: 6, column: 27, scope: !34)
// CHECK:STDOUT: !41 = !{!35}
// CHECK:STDOUT: !42 = distinct !DISubprogram(name: "F_u65536", linkageName: "_CF_u65536.Main", scope: null, file: !4, line: 7, type: !17, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !49)
// CHECK:STDOUT: !43 = !DILocalVariable(arg: 1, scope: !42, type: !9)
// CHECK:STDOUT: !48 = !DILocation(line: 7, column: 36, scope: !42)
// CHECK:STDOUT: !49 = !{!43}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'load_and_store.carbon'
// CHECK:STDOUT: source_filename = "load_and_store.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i24 @_CLoadStore_Bytes.Main(ptr %p, i24 %n) #0 !dbg !59 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc5_25.2 = load i24, ptr %p, align 4, !dbg !67
// CHECK:STDOUT:   store i24 %n, ptr %p, align 4, !dbg !71
// CHECK:STDOUT:   ret i24 %.loc5_25.2, !dbg !74
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i13 @_CLoadStore_NotBytes.Main(ptr %p, i13 %n) #0 !dbg !76 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_25.2 = load i16, ptr %p, align 2, !dbg !84
// CHECK:STDOUT:   %.loc11_25.21 = trunc i16 %.loc11_25.2 to i13, !dbg !84
// CHECK:STDOUT:   %0 = zext i13 %n to i16, !dbg !88
// CHECK:STDOUT:   store i16 %0, ptr %p, align 2, !dbg !88
// CHECK:STDOUT:   ret i13 %.loc11_25.21, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!51}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !50 = !DIFile(filename: "load_and_store.carbon", directory: "")
// CHECK:STDOUT: !51 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !50, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !52 = !DIBasicType(name: "int", size: 24, encoding: DW_ATE_signed)
// CHECK:STDOUT: !53 = !DIBasicType(name: "int", size: 13, encoding: DW_ATE_signed)
// CHECK:STDOUT: !54 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !55 = !{!52, !54, !52}
// CHECK:STDOUT: !56 = !DISubroutineType(types: !55)
// CHECK:STDOUT: !57 = !{!53, !54, !53}
// CHECK:STDOUT: !58 = !DISubroutineType(types: !57)
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "LoadStore_Bytes", linkageName: "_CLoadStore_Bytes.Main", scope: null, file: !50, line: 4, type: !56, spFlags: DISPFlagDefinition, unit: !51, retainedNodes: !75)
// CHECK:STDOUT: !60 = !DILocalVariable(arg: 1, scope: !59, type: !54)
// CHECK:STDOUT: !61 = !DILocalVariable(arg: 2, scope: !59, type: !52)
// CHECK:STDOUT: !67 = !DILocation(line: 5, column: 25, scope: !59)
// CHECK:STDOUT: !71 = !DILocation(line: 6, column: 3, scope: !59)
// CHECK:STDOUT: !74 = !DILocation(line: 7, column: 3, scope: !59)
// CHECK:STDOUT: !75 = !{!60, !61}
// CHECK:STDOUT: !76 = distinct !DISubprogram(name: "LoadStore_NotBytes", linkageName: "_CLoadStore_NotBytes.Main", scope: null, file: !50, line: 10, type: !58, spFlags: DISPFlagDefinition, unit: !51, retainedNodes: !92)
// CHECK:STDOUT: !77 = !DILocalVariable(arg: 1, scope: !76, type: !54)
// CHECK:STDOUT: !78 = !DILocalVariable(arg: 2, scope: !76, type: !53)
// CHECK:STDOUT: !84 = !DILocation(line: 11, column: 25, scope: !76)
// CHECK:STDOUT: !88 = !DILocation(line: 12, column: 3, scope: !76)
// CHECK:STDOUT: !91 = !DILocation(line: 13, column: 3, scope: !76)
// CHECK:STDOUT: !92 = !{!77, !78}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'convert.carbon'
// CHECK:STDOUT: source_filename = "convert.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CImplicitWidenSigned.Main(i8 %a) #0 !dbg !112 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %From.as_type.as.ImplicitAs.impl.Convert.call = call i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.619f31db71280c12"(i8 %a), !dbg !118
// CHECK:STDOUT:   ret i32 %From.as_type.as.ImplicitAs.impl.Convert.call, !dbg !118
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CImplicitWidenUnsigned.Main(i8 %a) #0 !dbg !120 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.call = call i32 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.9b7e0edf5c8a8a84"(i8 %a), !dbg !126
// CHECK:STDOUT:   ret i32 %UInt.as.ImplicitAs.impl.Convert.call, !dbg !126
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CImplicitWidenUnsignedToSigned.Main(i8 %a) #0 !dbg !128 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.call = call i32 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.de54a264bd4d5c6f"(i8 %a), !dbg !134
// CHECK:STDOUT:   ret i32 %UInt.as.ImplicitAs.impl.Convert.call, !dbg !134
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i5 @_CImplicitWidenCustom.Main(i4 %a) #0 !dbg !136 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %From.as_type.as.ImplicitAs.impl.Convert.call = call i5 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.2ea6d7b983af5ccc"(i4 %a), !dbg !142
// CHECK:STDOUT:   ret i5 %From.as_type.as.ImplicitAs.impl.Convert.call, !dbg !142
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i5 @_CImplicitWidenUnsignedCustom.Main(i4 %a) #0 !dbg !144 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %UInt.as.ImplicitAs.impl.Convert.call = call i5 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.f685d1058ed485bb"(i4 %a), !dbg !150
// CHECK:STDOUT:   ret i5 %UInt.as.ImplicitAs.impl.Convert.call, !dbg !150
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.619f31db71280c12"(i8 %self) #0 !dbg !153 {
// CHECK:STDOUT:   %1 = call i8 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.ca1ac10d3fc7c1a8"(i8 %self), !dbg !158
// CHECK:STDOUT:   %2 = sext i8 %1 to i32, !dbg !159
// CHECK:STDOUT:   ret i32 %2, !dbg !160
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.9b7e0edf5c8a8a84"(i8 %self) #0 !dbg !163 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.UInt.9452f4c51951679b.Core:FromUInt.a1a82301830d1e08.Core.62c2f9adcd5bc12d"(i8 %self), !dbg !169
// CHECK:STDOUT:   ret i32 %1, !dbg !170
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.de54a264bd4d5c6f"(i8 %self) #0 !dbg !172 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.Int.9452f4c51951679b.Core:FromUInt.94db85caea4a44af.Core.d19fcdc7c1bc8ad3"(i8 %self), !dbg !178
// CHECK:STDOUT:   ret i32 %1, !dbg !179
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i5 @"_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.2ea6d7b983af5ccc"(i4 %self) #0 !dbg !181 {
// CHECK:STDOUT:   %1 = call i4 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.044c7d430ddd4fd6"(i4 %self), !dbg !186
// CHECK:STDOUT:   %2 = sext i4 %1 to i5, !dbg !189
// CHECK:STDOUT:   ret i5 %2, !dbg !190
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i5 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.f685d1058ed485bb"(i4 %self) #0 !dbg !192 {
// CHECK:STDOUT:   %1 = call i5 @"_CConvert.UInt.9452f4c51951679b.Core:FromUInt.a1a82301830d1e08.Core.d2c3b6e62f7c89c4"(i4 %self), !dbg !198
// CHECK:STDOUT:   ret i5 %1, !dbg !199
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.ca1ac10d3fc7c1a8"(i8 %self) #0 !dbg !201 {
// CHECK:STDOUT:   ret i8 %self, !dbg !207
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.UInt.9452f4c51951679b.Core:FromUInt.a1a82301830d1e08.Core.62c2f9adcd5bc12d"(i8 %from) #0 !dbg !209 {
// CHECK:STDOUT:   %1 = call i8 @"_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.ca1ac10d3fc7c1a8"(i8 %from), !dbg !214
// CHECK:STDOUT:   %2 = zext i8 %1 to i32, !dbg !217
// CHECK:STDOUT:   ret i32 %2, !dbg !218
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.Int.9452f4c51951679b.Core:FromUInt.94db85caea4a44af.Core.d19fcdc7c1bc8ad3"(i8 %from) #0 !dbg !220 {
// CHECK:STDOUT:   %1 = call i8 @"_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.ca1ac10d3fc7c1a8"(i8 %from), !dbg !225
// CHECK:STDOUT:   %2 = zext i8 %1 to i32, !dbg !226
// CHECK:STDOUT:   ret i32 %2, !dbg !227
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i4 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.044c7d430ddd4fd6"(i4 %self) #0 !dbg !229 {
// CHECK:STDOUT:   ret i4 %self, !dbg !235
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i5 @"_CConvert.UInt.9452f4c51951679b.Core:FromUInt.a1a82301830d1e08.Core.d2c3b6e62f7c89c4"(i4 %from) #0 !dbg !237 {
// CHECK:STDOUT:   %1 = call i4 @"_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.044c7d430ddd4fd6"(i4 %from), !dbg !242
// CHECK:STDOUT:   %2 = zext i4 %1 to i5, !dbg !245
// CHECK:STDOUT:   ret i5 %2, !dbg !246
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.ca1ac10d3fc7c1a8"(i8 %self) #0 !dbg !248 {
// CHECK:STDOUT:   ret i8 %self, !dbg !254
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i4 @"_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.044c7d430ddd4fd6"(i4 %self) #0 !dbg !256 {
// CHECK:STDOUT:   ret i4 %self, !dbg !262
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.ca1ac10d3fc7c1a8", { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!94}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 8, encoding: DW_ATE_signed)
// CHECK:STDOUT: !10 = !{!6, !6}
// CHECK:STDOUT: !11 = !DISubroutineType(types: !10)
// CHECK:STDOUT: !93 = !DIFile(filename: "convert.carbon", directory: "")
// CHECK:STDOUT: !94 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !93, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !95 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !96 = !DIBasicType(name: "int", size: 8, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !97 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !98 = !DIBasicType(name: "int", size: 4, encoding: DW_ATE_signed)
// CHECK:STDOUT: !99 = !DIBasicType(name: "int", size: 5, encoding: DW_ATE_signed)
// CHECK:STDOUT: !100 = !DIBasicType(name: "int", size: 4, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !101 = !DIBasicType(name: "int", size: 5, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !102 = !{!95, !6}
// CHECK:STDOUT: !103 = !DISubroutineType(types: !102)
// CHECK:STDOUT: !104 = !{!97, !96}
// CHECK:STDOUT: !105 = !DISubroutineType(types: !104)
// CHECK:STDOUT: !106 = !{!95, !96}
// CHECK:STDOUT: !107 = !DISubroutineType(types: !106)
// CHECK:STDOUT: !108 = !{!99, !98}
// CHECK:STDOUT: !109 = !DISubroutineType(types: !108)
// CHECK:STDOUT: !110 = !{!101, !100}
// CHECK:STDOUT: !111 = !DISubroutineType(types: !110)
// CHECK:STDOUT: !112 = distinct !DISubprogram(name: "ImplicitWidenSigned", linkageName: "_CImplicitWidenSigned.Main", scope: null, file: !93, line: 4, type: !103, spFlags: DISPFlagDefinition, unit: !94, retainedNodes: !119)
// CHECK:STDOUT: !113 = !DILocalVariable(arg: 1, scope: !112, type: !6)
// CHECK:STDOUT: !118 = !DILocation(line: 4, column: 40, scope: !112)
// CHECK:STDOUT: !119 = !{!113}
// CHECK:STDOUT: !120 = distinct !DISubprogram(name: "ImplicitWidenUnsigned", linkageName: "_CImplicitWidenUnsigned.Main", scope: null, file: !93, line: 5, type: !105, spFlags: DISPFlagDefinition, unit: !94, retainedNodes: !127)
// CHECK:STDOUT: !121 = !DILocalVariable(arg: 1, scope: !120, type: !96)
// CHECK:STDOUT: !126 = !DILocation(line: 5, column: 42, scope: !120)
// CHECK:STDOUT: !127 = !{!121}
// CHECK:STDOUT: !128 = distinct !DISubprogram(name: "ImplicitWidenUnsignedToSigned", linkageName: "_CImplicitWidenUnsignedToSigned.Main", scope: null, file: !93, line: 6, type: !107, spFlags: DISPFlagDefinition, unit: !94, retainedNodes: !135)
// CHECK:STDOUT: !129 = !DILocalVariable(arg: 1, scope: !128, type: !96)
// CHECK:STDOUT: !134 = !DILocation(line: 6, column: 50, scope: !128)
// CHECK:STDOUT: !135 = !{!129}
// CHECK:STDOUT: !136 = distinct !DISubprogram(name: "ImplicitWidenCustom", linkageName: "_CImplicitWidenCustom.Main", scope: null, file: !93, line: 7, type: !109, spFlags: DISPFlagDefinition, unit: !94, retainedNodes: !143)
// CHECK:STDOUT: !137 = !DILocalVariable(arg: 1, scope: !136, type: !98)
// CHECK:STDOUT: !142 = !DILocation(line: 7, column: 57, scope: !136)
// CHECK:STDOUT: !143 = !{!137}
// CHECK:STDOUT: !144 = distinct !DISubprogram(name: "ImplicitWidenUnsignedCustom", linkageName: "_CImplicitWidenUnsignedCustom.Main", scope: null, file: !93, line: 8, type: !111, spFlags: DISPFlagDefinition, unit: !94, retainedNodes: !151)
// CHECK:STDOUT: !145 = !DILocalVariable(arg: 1, scope: !144, type: !100)
// CHECK:STDOUT: !150 = !DILocation(line: 8, column: 67, scope: !144)
// CHECK:STDOUT: !151 = !{!145}
// CHECK:STDOUT: !152 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !153 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.619f31db71280c12", scope: null, file: !152, line: 64, type: !103, spFlags: DISPFlagDefinition, unit: !94, retainedNodes: !161)
// CHECK:STDOUT: !154 = !DILocalVariable(arg: 1, scope: !153, type: !6)
// CHECK:STDOUT: !158 = !DILocation(line: 66, column: 17, scope: !153)
// CHECK:STDOUT: !159 = !DILocation(line: 66, column: 12, scope: !153)
// CHECK:STDOUT: !160 = !DILocation(line: 66, column: 5, scope: !153)
// CHECK:STDOUT: !161 = !{!154}
// CHECK:STDOUT: !162 = !DIFile(filename: "{{.*}}/prelude/types/uint.carbon", directory: "")
// CHECK:STDOUT: !163 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.9b7e0edf5c8a8a84", scope: null, file: !162, line: 88, type: !105, spFlags: DISPFlagDefinition, unit: !94, retainedNodes: !171)
// CHECK:STDOUT: !164 = !DILocalVariable(arg: 1, scope: !163, type: !96)
// CHECK:STDOUT: !169 = !DILocation(line: 89, column: 12, scope: !163)
// CHECK:STDOUT: !170 = !DILocation(line: 89, column: 5, scope: !163)
// CHECK:STDOUT: !171 = !{!164}
// CHECK:STDOUT: !172 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.de54a264bd4d5c6f", scope: null, file: !162, line: 88, type: !107, spFlags: DISPFlagDefinition, unit: !94, retainedNodes: !180)
// CHECK:STDOUT: !173 = !DILocalVariable(arg: 1, scope: !172, type: !96)
// CHECK:STDOUT: !178 = !DILocation(line: 89, column: 12, scope: !172)
// CHECK:STDOUT: !179 = !DILocation(line: 89, column: 5, scope: !172)
// CHECK:STDOUT: !180 = !{!173}
// CHECK:STDOUT: !181 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.45e1f8de18643ddf:ImplicitAs.0bc6489996802dd4.Core.2ea6d7b983af5ccc", scope: null, file: !152, line: 64, type: !109, spFlags: DISPFlagDefinition, unit: !94, retainedNodes: !191)
// CHECK:STDOUT: !182 = !DILocalVariable(arg: 1, scope: !181, type: !98)
// CHECK:STDOUT: !186 = !DILocation(line: 66, column: 17, scope: !181)
// CHECK:STDOUT: !187 = !{!98, !98}
// CHECK:STDOUT: !188 = !DISubroutineType(types: !187)
// CHECK:STDOUT: !189 = !DILocation(line: 66, column: 12, scope: !181)
// CHECK:STDOUT: !190 = !DILocation(line: 66, column: 5, scope: !181)
// CHECK:STDOUT: !191 = !{!182}
// CHECK:STDOUT: !192 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.f685d1058ed485bb", scope: null, file: !162, line: 88, type: !111, spFlags: DISPFlagDefinition, unit: !94, retainedNodes: !200)
// CHECK:STDOUT: !193 = !DILocalVariable(arg: 1, scope: !192, type: !100)
// CHECK:STDOUT: !198 = !DILocation(line: 89, column: 12, scope: !192)
// CHECK:STDOUT: !199 = !DILocation(line: 89, column: 5, scope: !192)
// CHECK:STDOUT: !200 = !{!193}
// CHECK:STDOUT: !201 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.ca1ac10d3fc7c1a8", scope: null, file: !152, line: 59, type: !11, spFlags: DISPFlagDefinition, unit: !94, retainedNodes: !208)
// CHECK:STDOUT: !202 = !DILocalVariable(arg: 1, scope: !201, type: !6)
// CHECK:STDOUT: !207 = !DILocation(line: 59, column: 28, scope: !201)
// CHECK:STDOUT: !208 = !{!202}
// CHECK:STDOUT: !209 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.UInt.9452f4c51951679b.Core:FromUInt.a1a82301830d1e08.Core.62c2f9adcd5bc12d", scope: null, file: !162, line: 72, type: !105, spFlags: DISPFlagDefinition, unit: !94, retainedNodes: !219)
// CHECK:STDOUT: !210 = !DILocalVariable(arg: 1, scope: !209, type: !96)
// CHECK:STDOUT: !214 = !DILocation(line: 74, column: 17, scope: !209)
// CHECK:STDOUT: !215 = !{!96, !96}
// CHECK:STDOUT: !216 = !DISubroutineType(types: !215)
// CHECK:STDOUT: !217 = !DILocation(line: 74, column: 12, scope: !209)
// CHECK:STDOUT: !218 = !DILocation(line: 74, column: 5, scope: !209)
// CHECK:STDOUT: !219 = !{!210}
// CHECK:STDOUT: !220 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Int.9452f4c51951679b.Core:FromUInt.94db85caea4a44af.Core.d19fcdc7c1bc8ad3", scope: null, file: !162, line: 80, type: !107, spFlags: DISPFlagDefinition, unit: !94, retainedNodes: !228)
// CHECK:STDOUT: !221 = !DILocalVariable(arg: 1, scope: !220, type: !96)
// CHECK:STDOUT: !225 = !DILocation(line: 82, column: 17, scope: !220)
// CHECK:STDOUT: !226 = !DILocation(line: 82, column: 12, scope: !220)
// CHECK:STDOUT: !227 = !DILocation(line: 82, column: 5, scope: !220)
// CHECK:STDOUT: !228 = !{!221}
// CHECK:STDOUT: !229 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.044c7d430ddd4fd6", scope: null, file: !152, line: 59, type: !188, spFlags: DISPFlagDefinition, unit: !94, retainedNodes: !236)
// CHECK:STDOUT: !230 = !DILocalVariable(arg: 1, scope: !229, type: !98)
// CHECK:STDOUT: !235 = !DILocation(line: 59, column: 28, scope: !229)
// CHECK:STDOUT: !236 = !{!230}
// CHECK:STDOUT: !237 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.UInt.9452f4c51951679b.Core:FromUInt.a1a82301830d1e08.Core.d2c3b6e62f7c89c4", scope: null, file: !162, line: 72, type: !111, spFlags: DISPFlagDefinition, unit: !94, retainedNodes: !247)
// CHECK:STDOUT: !238 = !DILocalVariable(arg: 1, scope: !237, type: !100)
// CHECK:STDOUT: !242 = !DILocation(line: 74, column: 17, scope: !237)
// CHECK:STDOUT: !243 = !{!100, !100}
// CHECK:STDOUT: !244 = !DISubroutineType(types: !243)
// CHECK:STDOUT: !245 = !DILocation(line: 74, column: 12, scope: !237)
// CHECK:STDOUT: !246 = !DILocation(line: 74, column: 5, scope: !237)
// CHECK:STDOUT: !247 = !{!238}
// CHECK:STDOUT: !248 = distinct !DISubprogram(name: "AsUInt", linkageName: "_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.ca1ac10d3fc7c1a8", scope: null, file: !162, line: 60, type: !216, spFlags: DISPFlagDefinition, unit: !94, retainedNodes: !255)
// CHECK:STDOUT: !249 = !DILocalVariable(arg: 1, scope: !248, type: !96)
// CHECK:STDOUT: !254 = !DILocation(line: 60, column: 29, scope: !248)
// CHECK:STDOUT: !255 = !{!249}
// CHECK:STDOUT: !256 = distinct !DISubprogram(name: "AsUInt", linkageName: "_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.044c7d430ddd4fd6", scope: null, file: !162, line: 60, type: !244, spFlags: DISPFlagDefinition, unit: !94, retainedNodes: !263)
// CHECK:STDOUT: !257 = !DILocalVariable(arg: 1, scope: !256, type: !100)
// CHECK:STDOUT: !262 = !DILocation(line: 60, column: 29, scope: !256)
// CHECK:STDOUT: !263 = !{!257}
// CHECK:STDOUT:
